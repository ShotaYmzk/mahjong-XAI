# fixed_batch_analysis.py - 修正版一括分析システム（有効打牌のみ表示）
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import shutil
from collections import defaultdict, OrderedDict

# プロジェクトモジュールのインポート
try:
    from full_mahjong_parser import parse_full_mahjong_log
    from game_state import GameState, NUM_TILE_TYPES, MAX_EVENT_HISTORY, STATIC_FEATURE_DIM, EVENT_TYPES, NUM_PLAYERS
    from tile_utils import tile_id_to_string, tile_id_to_index, tile_index_to_id
    from predict_enhanced import (
        MahjongTransformerV2WithAttention, 
        load_trained_model,
        predict_discard,
        analyze_attention_weights,
        analyze_with_concept_labels,
        generate_feature_names,
        get_activation_hook,
        activations_storage
    )
    from prompt import (
        format_hand_composition,
        format_all_players_discards,
        analyze_ukeire_for_discard,
        compare_discard_options,
        format_ukeire_analysis,
        format_ukeire_comparison,
        create_comprehensive_prompt
    )
    print("プロジェクトモジュールを正常にインポートしました。")
except ImportError as e:
    print(f"[エラー] プロジェクトモジュールのインポートに失敗しました: {e}")
    exit(1)

# --- 設定 ---
DEFAULT_MODEL_PATH = "../ver_1.1.11/trained_model/mahjong_transformer_v1111_large_compiled_2.pth"
DATA_HDF5_PATH = "/home/ubuntu/Documents/mahjong-XAI/ver_1.1.10/training_data/mahjong_imitation_data_v1110.hdf5"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_valid_top_predictions(probabilities, valid_discard_indices, actual_tile=None):
    """有効な打牌選択肢のみでTop5を作成"""
    if not valid_discard_indices:
        return "有効な打牌選択肢なし"
    
    # 有効な選択肢の確率を取得
    valid_predictions = []
    for tile_index in valid_discard_indices:
        if 0 <= tile_index < len(probabilities):
            tile_name = tile_id_to_string(tile_index_to_id(tile_index))
            prob = probabilities[tile_index]
            valid_predictions.append({
                'tile_index': tile_index,
                'tile': tile_name,
                'probability': prob
            })
    
    # 確率でソート
    valid_predictions.sort(key=lambda x: x['probability'], reverse=True)
    
    # Top5を作成
    lines = []
    for i, pred in enumerate(valid_predictions[:5]):
        rank = i + 1
        tile = pred['tile']
        prob = pred['probability']
        
        # 実際の打牌と一致する場合は印を付ける
        marker = " ★実打牌" if actual_tile and tile == actual_tile.replace('*', '') else ""
        lines.append(f"  {rank}位: {tile} ({prob:.1%}){marker}")
    
    return '\n'.join(lines)

class FixedBatchAnalysisSystem:
    """修正版の一括分析システム（有効打牌のみ表示）"""
    
    def __init__(self, model_path=DEFAULT_MODEL_PATH, output_base_dir="analysis_results"):
        self.model_path = model_path
        self.output_base_dir = Path(output_base_dir)
        self.model = None
        
        # 出力ディレクトリの作成
        self.output_base_dir.mkdir(exist_ok=True)
        
    def load_models(self):
        """必要なモデルをロード"""
        logger.info("モデルをロード中...")
        
        # ダミーの特徴量次元でモデルを初期化（後で実際の次元で再ロード）
        self.model = None
        
        logger.info("モデルのロード完了")
        
    def extract_all_tsumo_moments(self, xml_path, target_round_index, target_player_id=None):
        """1局分の全ツモ局面を抽出"""
        logger.info(f"局面抽出開始: {xml_path}, 局{target_round_index}")
        
        try:
            meta, rounds_data = parse_full_mahjong_log(xml_path)
        except Exception as e:
            logger.error(f"牌譜ファイルの解析エラー: {e}")
            raise
            
        if not (1 <= target_round_index <= len(rounds_data)):
            raise ValueError(f"不正な局インデックス: {target_round_index}")
            
        round_data = rounds_data[target_round_index - 1]
        events = round_data.get("events", [])
        
        tsumo_moments = []
        game_state = GameState()
        game_state.init_round(round_data)
        
        current_tsumo_count = 0
        
        for i, event_xml in enumerate(events):
            tag = event_xml["tag"]
            attrib = event_xml["attrib"]
            
            # ツモイベントの検出
            tsumo_player_id = -1
            tsumo_pai_id = -1
            
            for t_tag, p_id in GameState.TSUMO_TAGS.items():
                if tag.startswith(t_tag) and tag[1:].isdigit():
                    try:
                        tsumo_pai_id = int(tag[1:])
                        tsumo_player_id = p_id
                        break
                    except (ValueError, IndexError):
                        continue
                        
            if tsumo_player_id != -1:
                current_tsumo_count += 1
                
                # プレイヤーフィルタリング
                if target_player_id is None or tsumo_player_id == target_player_id:
                    # 次の打牌を探す
                    actual_discard_info = None
                    search_index = i + 1
                    while search_index < len(events):
                        next_event_xml = events[search_index]
                        next_tag = next_event_xml["tag"]
                        
                        # リーチ宣言はスキップして次を探す
                        if next_tag == "REACH":
                            search_index += 1
                            continue
                        
                        # 打牌イベントをチェック
                        for d_tag, p_id_next in GameState.DISCARD_TAGS.items():
                            if (next_tag.startswith(d_tag) and 
                                next_tag[1:].isdigit() and 
                                p_id_next == tsumo_player_id):
                                try:
                                    discard_pai_id = int(next_tag[1:])
                                    tsumogiri = next_tag[0].islower()
                                    actual_discard_info = {
                                        "player": p_id_next,
                                        "pai": discard_pai_id,
                                        "tsumogiri": tsumogiri,
                                        "xml": next_event_xml
                                    }
                                    break
                                except (ValueError, IndexError):
                                    continue
                        
                        if actual_discard_info is not None:
                            break
                        
                        # 他プレイヤーのイベントが来たら終了
                        other_player_event = False
                        for tag_prefix in ['T', 'U', 'V', 'W', 'D', 'E', 'F', 'G']:
                            if (next_tag.startswith(tag_prefix) and 
                                next_tag[1:].isdigit() and 
                                GameState.TSUMO_TAGS.get(tag_prefix, GameState.DISCARD_TAGS.get(tag_prefix)) != tsumo_player_id):
                                other_player_event = True
                                break
                        
                        if other_player_event:
                            break
                            
                        search_index += 1
                    
                    # 現在の状態をコピーして保存
                    game_state_copy = self._deep_copy_game_state(game_state)
                    
                    tsumo_moments.append({
                        "tsumo_count": current_tsumo_count,
                        "tsumo_info": {
                            "player": tsumo_player_id,
                            "pai": tsumo_pai_id,
                            "xml": event_xml
                        },
                        "actual_discard_info": actual_discard_info,
                        "game_state": game_state_copy,
                        "event_index": i
                    })
                
                # 状態を更新
                game_state.process_tsumo(tsumo_player_id, tsumo_pai_id)
            else:
                # 非ツモイベントの処理
                game_state.process_event(event_xml)
                
        logger.info(f"局面抽出完了: {len(tsumo_moments)}局面")
        return tsumo_moments
    
    def _deep_copy_game_state(self, game_state):
        """ゲーム状態の深いコピーを作成"""
        return game_state  # 簡略化（実際は深いコピーが必要）
        
    def analyze_single_moment(self, tsumo_moment, moment_index, total_moments):
        """単一のツモ局面を分析"""
        logger.info(f"分析中 [{moment_index+1}/{total_moments}]: ツモ{tsumo_moment['tsumo_count']}")
        
        game_state = tsumo_moment["game_state"]
        tsumo_info = tsumo_moment["tsumo_info"]
        actual_discard_info = tsumo_moment["actual_discard_info"]
        player_id = tsumo_info["player"]
        
        # モデルが未ロードの場合はロード
        if self.model is None:
            try:
                event_seq_sample = game_state.get_event_sequence_features()
                static_feat_sample = game_state.get_static_features(player_id)
                
                event_dim = event_seq_sample.shape[1]
                static_dim = len(static_feat_sample)
                
                self.model = load_trained_model(self.model_path, event_dim, static_dim)
                logger.info(f"モデルロード完了: event_dim={event_dim}, static_dim={static_dim}")
            except Exception as e:
                logger.error(f"モデルロードエラー: {e}")
                return None
        
        # 予測を実行
        try:
            predicted_index, predicted_prob, all_probabilities, attention_weights = predict_discard(
                self.model, game_state, player_id, return_attention=True
            )
        except Exception as e:
            logger.error(f"予測エラー: {e}")
            return None
        
        # 有効な打牌選択肢を取得
        valid_discard_indices = game_state.get_valid_discard_options(player_id)
        
        # 有効な選択肢のみでTop5を作成
        actual_tile_str = tile_id_to_string(actual_discard_info["pai"]) if actual_discard_info else ""
        if actual_discard_info and actual_discard_info.get("tsumogiri", False):
            actual_tile_str += "*"
            
        # Top5予測を有効な選択肢のみで作成
        valid_top_predictions = []
        for tile_index in valid_discard_indices:
            if 0 <= tile_index < len(all_probabilities):
                tile_name = tile_id_to_string(tile_index_to_id(tile_index))
                prob = all_probabilities[tile_index]
                valid_top_predictions.append({
                    'rank': len(valid_top_predictions) + 1,
                    'tile': tile_name,
                    'tile_index': tile_index,
                    'probability': prob
                })
        
        # 確率でソート
        valid_top_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        # ランクを再設定
        for i, pred in enumerate(valid_top_predictions):
            pred['rank'] = i + 1
        
        # 分析結果を構築
        analysis_result = {
            "tsumo_count": tsumo_moment["tsumo_count"],
            "player_id": player_id,
            "tsumo_tile": tile_id_to_string(tsumo_info["pai"]),
            "predicted_tile_index": predicted_index,
            "predicted_tile": tile_id_to_string(tile_index_to_id(predicted_index)) if predicted_index != -1 else "不明",
            "predicted_probability": predicted_prob,
            "actual_tile": actual_tile_str,
            "actual_tile_id": actual_discard_info["pai"] if actual_discard_info else -1,
            "valid_discard_indices": valid_discard_indices,
            "valid_top_predictions": valid_top_predictions[:5],  # Top5のみ
            "all_probabilities": all_probabilities.tolist(),
            "hand_composition": [tile_id_to_string(t) for t in game_state.player_hands[player_id]],
            "hand_size": len(game_state.player_hands[player_id]),
            "is_valid_prediction": predicted_index in valid_discard_indices,
            "attention_weights": self._format_attention_analysis(attention_weights, game_state, player_id) if attention_weights else {},
            "game_situation": self._build_game_situation(game_state, player_id),
            "players_state": self._build_players_state(game_state)
        }
        
        return analysis_result
    
    def _format_attention_analysis(self, attention_weights, game_state, player_id):
        """アテンション分析のフォーマット"""
        return analyze_attention_weights(attention_weights, 
                                       game_state.get_event_sequence_features(), 
                                       game_state, player_id)
    
    def _build_game_situation(self, game_state, player_id):
        """ゲーム状況の構築"""
        # 風牌の設定
        winds = ["東", "南", "西", "北"]
        round_wind = winds[game_state.round_num_wind % 4]
        
        # プレイヤーの風
        player_wind_index = (player_id - game_state.dealer) % 4
        player_wind = winds[player_wind_index]
        
        # ドラ表示牌
        dora_names = []
        for dora_id in game_state.dora_indicators:
            dora_names.append(tile_id_to_string(dora_id))
        
        return {
            "round_info": f"{round_wind}{'1局' if game_state.round_num_wind < 4 else '2局' if game_state.round_num_wind < 8 else '3局' if game_state.round_num_wind < 12 else '4局'}",
            "player_wind": player_wind,
            "current_player": player_id,
            "remaining_tiles": game_state.wall_tile_count,
            "dora_indicators": dora_names,
            "kyotaku": game_state.kyotaku,
            "honba": game_state.honba,
            "junme": game_state.junme
        }
    
    def _build_players_state(self, game_state):
        """プレイヤー状態の構築"""
        players_state = {}
        
        for p in range(NUM_PLAYERS):
            # 手牌（自分のみ表示、他は非公開）
            hand = [tile_id_to_string(t) for t in game_state.player_hands[p]]
            
            # 捨て牌
            discards = []
            for tile_id, tsumogiri in game_state.player_discards[p]:
                tile_str = tile_id_to_string(tile_id)
                if tsumogiri:
                    tile_str += "*"
                discards.append(tile_str)
            
            # 副露
            melds = []
            for meld_dict in game_state.player_melds[p]:
                melds.append({
                    "type": meld_dict.get("type", "不明"),
                    "tiles": [tile_id_to_string(t) for t in meld_dict.get("tiles", [])]
                })
            
            players_state[f"player_{p}"] = {
                "hand": hand,
                "discards": discards,
                "melds": melds,
                "reach_status": game_state.player_reach_status[p],
                "reach_junme": game_state.player_reach_junme[p]
            }
        
        return players_state
    
    def generate_prompt_for_moment(self, analysis_result):
        """局面用のプロンプトを生成"""
        # 分析データを統合
        analysis_data = {
            'game_situation': analysis_result['game_situation'],
            'prediction': {
                'predicted_tile': analysis_result['predicted_tile'],
                'predicted_probability': analysis_result['predicted_probability'],
                'top_predictions': analysis_result['valid_top_predictions']  # 有効な選択肢のみ
            },
            'analysis': {
                'attention_weights': analysis_result.get('attention_weights', {}),
                'concept_labels': {}
            },
            'players_state': analysis_result['players_state']
        }
        
        # ツモ牌を追加
        analysis_data['game_situation']['tsumo_tile'] = analysis_result['tsumo_tile']
        analysis_data['game_situation']['actual_discard'] = analysis_result['actual_tile']
        
        return create_comprehensive_prompt(analysis_data)
    
    def save_results(self, analysis_results, xml_path, round_index, player_id):
        """分析結果を保存"""
        xml_name = Path(xml_path).stem
        if player_id is not None:
            output_dir = self.output_base_dir / f"{xml_name}_R{round_index}_P{player_id}"
        else:
            output_dir = self.output_base_dir / f"{xml_name}_R{round_index}_ALL"
        
        output_dir.mkdir(exist_ok=True)
        
        logger.info(f"結果保存中: {output_dir}")
        
        # 各局面の結果を保存
        for i, result in enumerate(analysis_results):
            tsumo_count = result["tsumo_count"]
            
            # 個別フォルダ作成
            moment_dir = output_dir / f"tsumo_{tsumo_count}"
            moment_dir.mkdir(exist_ok=True)
            
            # JSON結果保存
            json_path = moment_dir / "analysis.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # プロンプト生成・保存
            try:
                prompt_text = self.generate_prompt_for_moment(result)
                prompt_path = moment_dir / "prompt.txt"
                with open(prompt_path, 'w', encoding='utf-8') as f:
                    f.write(prompt_text)
            except Exception as e:
                logger.warning(f"プロンプト生成エラー (tsumo_{tsumo_count}): {e}")
        
        # サマリー保存
        self._create_overall_summary(analysis_results, output_dir)
        
        logger.info(f"保存完了: {len(analysis_results)}局面")
    
    def _create_overall_summary(self, analysis_results, output_dir):
        """全体サマリーを作成"""
        summary = {
            "total_moments": len(analysis_results),
            "valid_predictions": sum(1 for r in analysis_results if r["is_valid_prediction"]),
            "average_confidence": np.mean([r["predicted_probability"] for r in analysis_results]),
            "moments": []
        }
        
        for result in analysis_results:
            summary["moments"].append({
                "tsumo_count": result["tsumo_count"],
                "predicted_tile": result["predicted_tile"],
                "actual_tile": result["actual_tile"],
                "confidence": result["predicted_probability"],
                "is_valid": result["is_valid_prediction"],
                "hand_size": result["hand_size"]
            })
        
        # サマリー保存
        summary_path = output_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    
    def run_batch_analysis(self, xml_path, round_index, player_id=None):
        """バッチ分析のメイン実行"""
        logger.info(f"分析開始: {xml_path}, 局{round_index}, プレイヤー{player_id}")
        
        # ツモ局面を抽出
        tsumo_moments = self.extract_all_tsumo_moments(xml_path, round_index, player_id)
        
        if not tsumo_moments:
            logger.warning("分析対象の局面が見つかりませんでした")
            return []
        
        # 各局面を分析
        analysis_results = []
        for i, moment in enumerate(tsumo_moments):
            result = self.analyze_single_moment(moment, i, len(tsumo_moments))
            if result:
                analysis_results.append(result)
        
        # 結果を保存
        if analysis_results:
            self.save_results(analysis_results, xml_path, round_index, player_id)
        
        logger.info(f"分析完了: {len(analysis_results)}局面")
        return analysis_results

def main():
    parser = argparse.ArgumentParser(
        description="修正版の麻雀AI分析システム（有効打牌のみ表示）"
    )
    parser.add_argument("xml_file", help="分析対象の牌譜XMLファイル")
    parser.add_argument("round_index", type=int, help="分析対象の局番号（1から開始）")
    parser.add_argument("--player", type=int, choices=[0,1,2,3], 
                       help="分析対象プレイヤー（指定しない場合は全プレイヤー）")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="モデルファイルパス")
    parser.add_argument("--output", default="fixed_analysis_results", help="出力ディレクトリ")
    
    args = parser.parse_args()
    
    # 分析システムを初期化・実行
    system = FixedBatchAnalysisSystem(args.model, args.output)
    system.run_batch_analysis(args.xml_file, args.round_index, args.player)

if __name__ == "__main__":
    main() 