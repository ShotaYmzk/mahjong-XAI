# batch_analysis.py - 1局分全打牌の一括予測・分析・プロンプト生成システム
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

# SHAPとMatplotlibをインポート
try:
    import shap
    import matplotlib.pyplot as plt
    shap_available = True
except ImportError:
    print("[警告] `shap` または `matplotlib` ライブラリが見つかりません。SHAP説明機能・プロットはスキップされます。")
    shap_available = False

# プロジェクトモジュールのインポート
try:
    from full_mahjong_parser import parse_full_mahjong_log
    from game_state import GameState, NUM_TILE_TYPES, MAX_EVENT_HISTORY, STATIC_FEATURE_DIM, EVENT_TYPES, NUM_PLAYERS
    from tile_utils import tile_id_to_string, tile_id_to_index, tile_index_to_id
    from predict_enhanced import (
        MahjongTransformerV2WithAttention, 
        load_trained_model,
        load_explanation_models,
        predict_discard,
        analyze_attention_weights,
        analyze_with_concept_labels,
        explain_prediction_with_shap,
        generate_feature_names,
        get_activation_hook,
        activations_storage
    )
    from prompt import (
        format_hand_composition,
        format_all_players_discards,
        format_top_predictions,
        analyze_ukeire_for_discard,
        compare_discard_options,
        format_ukeire_analysis,
        format_ukeire_comparison,
        create_comprehensive_prompt
    )
    # shanten.pyの機能をインポート
    from shanten import (
        robust_hand_parser,
        format_tiles_for_display,
        format_shanten,
        get_shanten_after_best_discard,
        analyze_hand_details
    )
    from mahjong.shanten import Shanten
    from mahjong.tile import TilesConverter
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

class BatchAnalysisSystem:
    """1局分の全打牌を一括分析するシステム"""
    
    def __init__(self, model_path=DEFAULT_MODEL_PATH, output_base_dir="analysis_results"):
        self.model_path = model_path
        self.output_base_dir = Path(output_base_dir)
        self.model = None
        self.pca_model = None
        self.kmeans_model = None
        self.concept_labels = None
        
        # 出力ディレクトリの作成
        self.output_base_dir.mkdir(exist_ok=True)
        
    def load_models(self):
        """必要なモデルをロード"""
        logger.info("モデルをロード中...")
        
        # ダミーの特徴量次元でモデルを初期化（後で実際の次元で再ロード）
        self.model = None
        
        # 説明モデルのロード
        self.pca_model, self.kmeans_model, self.concept_labels = load_explanation_models()
        
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
                    # 次の打牌を探す（リーチ宣言を考慮）
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
                        
                        # 打牌が見つかったか、他プレイヤーのイベントが見つかったら終了
                        if actual_discard_info is not None:
                            break
                        
                        # ツモしたプレイヤー以外のイベントが来たら終了
                        # (他プレイヤーのツモや打牌があった場合)
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
                # その他のイベント処理
                try:
                    game_state.process_event(event_xml)
                except Exception as e:
                    logger.warning(f"イベント処理エラー: {e}")
                    continue
                    
        logger.info(f"抽出完了: {len(tsumo_moments)}個のツモ局面")
        return tsumo_moments
        
    def _deep_copy_game_state(self, game_state):
        """GameStateの深いコピー（簡易版）"""
        # 実際の実装では、GameStateの全状態を適切にコピーする必要があります
        # ここでは簡略化
        import copy
        return copy.deepcopy(game_state)
        
    def analyze_single_moment(self, tsumo_moment, moment_index, total_moments):
        """単一のツモ局面を分析"""
        logger.info(f"局面分析中 [{moment_index+1}/{total_moments}]: ツモ{tsumo_moment['tsumo_count']}")
        
        game_state = tsumo_moment["game_state"]
        tsumo_info = tsumo_moment["tsumo_info"]
        actual_discard_info = tsumo_moment["actual_discard_info"]
        player_id = tsumo_info["player"]
        
        # ツモを実行
        game_state.process_tsumo(tsumo_info["player"], tsumo_info["pai"])
        
        # 特徴量生成
        try:
            event_sequence = game_state.get_event_sequence_features()
            static_features = game_state.get_static_features(player_id)
            event_dim = event_sequence.shape[1]
            static_dim = static_features.shape[0]
        except Exception as e:
            logger.error(f"特徴量生成エラー: {e}")
            return None
            
        # モデルが未ロードの場合、正しい次元でロード
        if self.model is None:
            self.model = load_trained_model(self.model_path, event_dim, static_dim)
            
        # 予測実行
        try:
            predicted_index, predicted_prob, all_probabilities, attention_weights = predict_discard(
                self.model, game_state, player_id, return_attention=True
            )
        except Exception as e:
            logger.error(f"予測エラー: {e}")
            return None
            
        # 各種分析
        attention_analysis = analyze_attention_weights(
            attention_weights, event_sequence, game_state, player_id
        )
        
        activation_vector = activations_storage.get('combined_vector')
        concept_analysis = analyze_with_concept_labels(
            self.pca_model, self.kmeans_model, self.concept_labels, activation_vector
        )
        
        # SHAP分析（時間がかかるので簡略化オプション）
        shap_analysis = None
        if shap_available:
            try:
                instance_to_explain = (event_sequence, static_features, None)
                feature_names = generate_feature_names(event_dim, static_dim, event_sequence.shape[0])
                shap_analysis = explain_prediction_with_shap(
                    self.model, DATA_HDF5_PATH, instance_to_explain, 
                    feature_names, predicted_index, n_shap_samples=50, n_bg_summary_samples=25
                )
            except Exception as e:
                logger.warning(f"SHAP分析エラー: {e}")
                
        # 分析結果をまとめる
        analysis_result = {
            "moment_info": {
                "tsumo_count": tsumo_moment["tsumo_count"],
                "player_id": player_id,
                "tsumo_tile": tile_id_to_string(tsumo_info["pai"]),
                "predicted_tile": tile_id_to_string(tile_index_to_id(predicted_index)),
                "predicted_probability": float(predicted_prob),
                "actual_tile": tile_id_to_string(actual_discard_info["pai"]) if actual_discard_info else None
            },
            "game_state": game_state,
            "prediction": {
                "predicted_index": predicted_index,
                "predicted_prob": predicted_prob,
                "all_probabilities": all_probabilities
            },
            "analysis": {
                "attention": attention_analysis,
                "concept": concept_analysis,
                "shap": shap_analysis
            },
            "tsumo_info": tsumo_info,
            "actual_discard_info": actual_discard_info
        }
        
        return analysis_result
        
    def generate_prompt_for_moment(self, analysis_result):
        """個別局面のプロンプトを生成"""
        try:
            # 基本情報の準備
            game_state = analysis_result["game_state"]
            player_id = analysis_result["moment_info"]["player_id"]
            
            # プロンプト用データの構築
            analysis_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "analysis_version": "2.1.0_batch"
                },
                "game_situation": self._build_game_situation(game_state, player_id, analysis_result),
                "players_state": self._build_players_state(game_state),
                "prediction": self._build_prediction_data(analysis_result),
                "analysis": analysis_result["analysis"]
            }
            
            # プロンプト生成
            prompt_text = self._create_comprehensive_prompt(analysis_data)
            
            return {
                "prompt_text": prompt_text,
                "analysis_data": analysis_data
            }
            
        except Exception as e:
            logger.error(f"プロンプト生成エラー: {e}")
            return None
            
    def _build_game_situation(self, game_state, player_id, analysis_result):
        """ゲーム状況情報を構築"""
        from predict_enhanced import get_wind_str
        
        round_str, my_wind_str = get_wind_str(game_state.round_num_wind, player_id, game_state.dealer)
        
        return {
            "round_info": round_str,
            "player_wind": my_wind_str,
            "current_player": player_id,
            "tsumo_tile": analysis_result["moment_info"]["tsumo_tile"],
            "actual_discard": analysis_result["moment_info"]["actual_tile"],
            "dora_indicators": [self._convert_dora_indicator_to_dora(t) for t in game_state.dora_indicators],
            "remaining_tiles": int(game_state.wall_tile_count),
            "kyotaku": int(game_state.kyotaku),
            "honba": int(game_state.honba)
        }
        
    def _build_players_state(self, game_state):
        """プレイヤー状態情報を構築"""
        players_state = {}
        for p in range(NUM_PLAYERS):
            players_state[f"player_{p}"] = {
                "hand": [tile_id_to_string(t) for t in game_state.player_hands[p]],
                "discards": [{"tile": tile_id_to_string(t), "tsumogiri": ts} 
                           for t, ts in game_state.player_discards[p]],
                "melds": [
                    {
                        "type": meld.get('type', '不明'),
                        "tiles": [tile_id_to_string(t) for t in meld.get('tiles', [])],
                        "from_who": meld.get('from_who', -1)
                    } for meld in game_state.player_melds[p]
                ],
                "reach_status": game_state.player_reach_status[p],
                "score": game_state.current_scores[p]
            }
        return players_state
        
    def _build_prediction_data(self, analysis_result):
        """予測データを構築"""
        all_probabilities = analysis_result["prediction"]["all_probabilities"]
        top_indices = np.argsort(all_probabilities)[::-1][:10]
        
        top_predictions = []
        for i, idx in enumerate(top_indices):
            if 0 <= idx < NUM_TILE_TYPES:
                tile_id = tile_index_to_id(idx)
                if tile_id != -1:
                    tile_str = tile_id_to_string(tile_id)
                    if tile_str != "?":
                        top_predictions.append({
                            "rank": i + 1,
                            "tile": tile_str,
                            "probability": float(all_probabilities[idx])
                        })
                        
        return {
            "predicted_tile": analysis_result["moment_info"]["predicted_tile"],
            "predicted_probability": analysis_result["moment_info"]["predicted_probability"],
            "top_predictions": top_predictions
        }
        
    def _create_comprehensive_prompt(self, analysis_data):
        """包括的なプロンプトを生成"""
        # 基本情報の抽出
        game_situation = analysis_data["game_situation"]
        players_state = analysis_data["players_state"]
        prediction = analysis_data["prediction"]
        analysis = analysis_data["analysis"]
        
        # 手牌の整理
        current_player = game_situation["current_player"]
        hand_tiles = players_state[f"player_{current_player}"]["hand"]
        
        # プロンプトの構築
        prompt = f"""あなたは麻雀の専門コーチです。AI分析結果に基づいて、打牌判断の戦術的根拠を分かりやすく説明してください。

【局面状況】
局: {game_situation["round_info"]} ({game_situation["player_wind"]}家)
リーチ者: {sum(1 for p in range(NUM_PLAYERS) if players_state[f"player_{p}"]["reach_status"] == 2)}人
残り牌: {game_situation["remaining_tiles"]}枚
ドラ: {" ".join(game_situation["dora_indicators"])}
供託: {game_situation["kyotaku"]}本 / 本場: {game_situation["honba"]}本場

【自分の手牌】
{" ".join(hand_tiles)} (P{current_player})

【ツモ牌】
{game_situation["tsumo_tile"]}

【各プレイヤーの捨て牌】"""

        # 各プレイヤーの捨て牌を追加
        for p in range(NUM_PLAYERS):
            discards = players_state[f"player_{p}"]["discards"]
            discard_str = " ".join([f"{d['tile']}{'*' if d['tsumogiri'] else ''}" for d in discards])
            if p == current_player:
                prompt += f"\n  P{p}: {discard_str} ← 自分"
            else:
                prompt += f"\n  P{p}: {discard_str}"

        # 各プレイヤーの副露を追加
        prompt += "\n\n【副露】"
        for p in range(NUM_PLAYERS):
            melds = players_state[f"player_{p}"]["melds"]
            meld_str = self._format_melds(melds)
            if p == current_player:
                prompt += f"\n  P{p}: {meld_str} ← 自分"
            else:
                prompt += f"\n  P{p}: {meld_str}"

        # AI判断の追加
        prompt += f"""

【AI判断】
推奨打牌: {prediction["predicted_tile"]} (確信度: {prediction["predicted_probability"]:.1%})
実際打牌: {game_situation["actual_discard"]}

【推奨打牌Top5】"""

        for pred in prediction["top_predictions"][:5]:
            if pred["tile"] == game_situation["actual_discard"]:
                prompt += f"\n  {pred['rank']}位: {pred['tile']} ({pred['probability']:.1%}) ★実打牌"
            else:
                prompt += f"\n  {pred['rank']}位: {pred['tile']} ({pred['probability']:.1%})"

        # 受け入れ分析の追加（詳細版）
        prompt += "\n\n【受け入れ分析比較】"
        ai_tile = prediction["predicted_tile"]
        actual_tile = game_situation["actual_discard"]
        
        # 手牌をshanten.py用の形式に変換
        current_player = game_situation["current_player"]
        hand_tiles = players_state[f"player_{current_player}"]["hand"]
        
        # 手牌の枚数を確認（ツモ後は14枚であるべき）
        if len(hand_tiles) == 14:
            # すでにツモ牌が含まれている場合はそのまま使用
            hand_string = self._convert_tiles_to_shanten_format(hand_tiles)
        elif len(hand_tiles) == 13:
            # 13枚の場合はツモ牌を追加
            tsumo_tile = game_situation["tsumo_tile"]
            all_tiles = hand_tiles + [tsumo_tile]
            hand_string = self._convert_tiles_to_shanten_format(all_tiles)
        else:
            # 異常な枚数の場合はエラーとして扱う
            hand_string = None
        
        if hand_string and ai_tile != actual_tile:
            # 詳細な受け入れ分析を実行
            ukeire_analysis = self._analyze_detailed_ukeire(hand_string, ai_tile, actual_tile)
            
            if isinstance(ukeire_analysis, list) and len(ukeire_analysis) >= 2:
                ai_analysis = ukeire_analysis[0]
                actual_analysis = ukeire_analysis[1]
                
                prompt += f"""
■ AI推奨打牌: {ai_analysis["formatted"]}
■ 実際打牌: {actual_analysis["formatted"]}
■ 比較結果: """
                
                # シャンテン数比較
                if ai_analysis["shanten"] < actual_analysis["shanten"]:
                    prompt += f"AI推奨の方が{actual_analysis['shanten'] - ai_analysis['shanten']}向聴少ない"
                elif ai_analysis["shanten"] > actual_analysis["shanten"]:
                    prompt += f"実際打牌の方が{ai_analysis['shanten'] - actual_analysis['shanten']}向聴少ない"
                else:
                    # 同じシャンテン数の場合は受け入れ枚数で比較
                    if ai_analysis["ukeire_count"] > actual_analysis["ukeire_count"]:
                        prompt += f"同{format_shanten(ai_analysis['shanten'])}だがAI推奨の方が{ai_analysis['ukeire_count'] - actual_analysis['ukeire_count']}枚多く受け入れ"
                    elif ai_analysis["ukeire_count"] < actual_analysis["ukeire_count"]:
                        prompt += f"同{format_shanten(ai_analysis['shanten'])}だが実際打牌の方が{actual_analysis['ukeire_count'] - ai_analysis['ukeire_count']}枚多く受け入れ"
                    else:
                        prompt += f"同{format_shanten(ai_analysis['shanten'])}・同受け入れ枚数"
            else:
                prompt += f"""
■ AI推奨打牌: {ai_tile}
■ 実際打牌: {actual_tile}
■ 比較結果: 受け入れ分析の計算でエラーが発生しました"""
        elif hand_string and ai_tile == actual_tile:
            # AI推奨と実際が一致している場合
            ukeire_analysis = self._analyze_detailed_ukeire(hand_string, ai_tile, ai_tile)
            if isinstance(ukeire_analysis, list) and len(ukeire_analysis) >= 1:
                analysis = ukeire_analysis[0]
                prompt += f"\n■ AI推奨と実際の打牌が一致: {analysis['formatted']}"
            else:
                prompt += f"\n■ AI推奨と実際の打牌が一致: {ai_tile}"
        else:
            prompt += f"""
■ AI推奨打牌: {ai_tile}
■ 実際打牌: {actual_tile}
■ 比較結果: 手牌情報の取得に失敗したため、詳細分析を実行できませんでした"""

        # AI思考プロセスの追加
        prompt += "\n\n【AI思考プロセス】"
        
        # 概念分析
        if analysis.get("concept") and analysis["concept"].get("concept_labels"):
            labels = analysis["concept"]["concept_labels"]
            cluster_id = analysis["concept"]["cluster_id"]
            prompt += f"\n■ 戦略方針: {'/'.join(labels)} (クラスタ: {cluster_id})"
        
        # アテンション分析
        if analysis.get("attention"):
            prompt += "\n■ 注目した相手の動き（層別分析）"
            for layer_key, layer_data in analysis["attention"].items():  # 全ての層を表示
                layer_num = layer_data["layer"]
                prompt += f"\n【Layer {layer_num}】"
                
                for i, event_data in enumerate(layer_data["top_attended_events"][:5]):
                    event_token = event_data["event_token"]
                    weight = event_data["attention_weight"]
                    interpretation = self._interpret_event_token(event_token)
                    prompt += f"\n  {i+1}. {interpretation} (注目度: {weight:.4f})"

        # SHAP分析
        if analysis.get("shap") and analysis["shap"].get("feature_importance"):
            prompt += "\n■ 手牌評価"
            top_features = analysis["shap"]["feature_importance"][:3]
            for name, importance in top_features:
                if "手牌_" in name:
                    tile_name = name.split("手牌_")[-1]
                    if importance > 0:
                        prompt += f"\n・{tile_name}が重要要素(重要度{importance:.3f}) → 保持推奨"
                    else:
                        prompt += f"\n・{tile_name}が重要要素(重要度{importance:.3f}) → 打牌推奨"

        # 解説要求
        prompt += """

【解説要求】
以下の3つの観点から、初心者にも分かりやすく解説してください：

1. **定量的、定性的な判断** (50文字以内)
   てきとうな判断ではなく、定量的、定性的な判断をしてください。

2. **戦術的根拠** (100文字以内)  
   手牌構成、局面状況を踏まえた詳細な戦術理由を説明してください。

3. **代替案検討** (100文字以内)
   他の選択肢と比べてなぜこれがベストか

各項目を明確に分けて、実戦で使える知識として説明してください。"""

        return prompt
        
    def _convert_tiles_to_shanten_format(self, tile_strings):
        """牌の文字列リストをshanten.py用の形式に変換"""
        try:
            man_tiles = []
            pin_tiles = []
            sou_tiles = []
            honor_tiles = []
            
            for tile_str in tile_strings:
                if tile_str.endswith('m'):
                    # 萬子
                    num = tile_str[:-1]
                    if num == '0':  # 赤5萬
                        man_tiles.append('0')
                    else:
                        man_tiles.append(num)
                elif tile_str.endswith('p'):
                    # 筒子
                    num = tile_str[:-1]
                    if num == '0':  # 赤5筒
                        pin_tiles.append('0')
                    else:
                        pin_tiles.append(num)
                elif tile_str.endswith('s'):
                    # 索子
                    num = tile_str[:-1]
                    if num == '0':  # 赤5索
                        sou_tiles.append('0')
                    else:
                        sou_tiles.append(num)
                elif tile_str.endswith('z'):
                    # 字牌
                    num = tile_str[:-1]
                    honor_tiles.append(num)
                else:
                    # その他の字牌の表記（東、南、西、北、白、發、中）
                    tile_mapping = {
                        '東': '1z', '南': '2z', '西': '3z', '北': '4z',
                        '白': '5z', '發': '6z', '中': '7z'
                    }
                    if tile_str in tile_mapping:
                        mapped = tile_mapping[tile_str]
                        honor_tiles.append(mapped[:-1])
            
            # 結果を構築
            result = ""
            if man_tiles:
                result += "".join(sorted(man_tiles)) + "m"
            if pin_tiles:
                result += "".join(sorted(pin_tiles)) + "p"
            if sou_tiles:
                result += "".join(sorted(sou_tiles)) + "s"
            if honor_tiles:
                result += "".join(sorted(honor_tiles)) + "z"
            
            return result
            
        except Exception as e:
            logger.warning(f"牌文字列変換エラー: {e}")
            return None
    
    def _format_melds(self, meld_list_dicts):
        """鳴き情報を人間が読みやすい形式に変換"""
        if not meld_list_dicts:
            return "なし"
        
        meld_strs = []
        for meld_info in meld_list_dicts:
            m_type = meld_info.get('type', '不明')
            m_tiles = meld_info.get('tiles', [])
            from_who_abs = meld_info.get('from_who', -1)
            called_tile = meld_info.get('called_tile', -1)
            
            # 牌を並べる
            tiles_str = " ".join(m_tiles)
            
            # 鳴いた相手の表示
            from_str = f" P{from_who_abs}" if from_who_abs != -1 and m_type not in ["暗槓", "加槓"] else ""
            
            # 鳴いた牌の表示
            trigger_str = f"({called_tile})" if called_tile != -1 and m_type != "暗槓" else ""
            
            meld_strs.append(f"{m_type}{trigger_str}[{tiles_str}]{from_str}")
        
        return " / ".join(meld_strs)
    
    def _convert_hand_to_shanten_format(self, game_state, player_id, tsumo_tile_str):
        """GameStateの手牌データをshanten.py用の文字列形式に変換"""
        try:
            # 手牌を取得
            hand_tiles = game_state.hands[player_id]
            
            # ツモ牌を追加した14枚の手牌を作成
            tiles_with_tsumo = hand_tiles.copy()
            
            # ツモ牌をtile_id_to_indexで変換してカウントに追加
            tsumo_tile_id = None
            for tile_id in range(136):  # 136種の牌IDを検索
                tile_str = tile_id_to_string(tile_id)
                if tile_str == tsumo_tile_str:
                    tsumo_tile_id = tile_id
                    break
            
            if tsumo_tile_id is not None:
                tsumo_index = tile_id_to_index(tsumo_tile_id)
                tiles_with_tsumo[tsumo_index] += 1
            
            # 34種配列から文字列形式に変換
            hand_string = self._tiles_34_to_string(tiles_with_tsumo)
            return hand_string
            
        except Exception as e:
            logger.warning(f"手牌変換エラー: {e}")
            return None
    
    def _tiles_34_to_string(self, tiles_34):
        """34種配列を文字列形式に変換"""
        man_tiles = []
        pin_tiles = []
        sou_tiles = []
        honor_tiles = []
        
        # 萬子 (0-8)
        for i in range(9):
            count = tiles_34[i]
            for _ in range(count):
                if i == 4:  # 赤5萬
                    man_tiles.append('0')
                else:
                    man_tiles.append(str(i + 1))
        
        # 筒子 (9-17)
        for i in range(9):
            count = tiles_34[i + 9]
            for _ in range(count):
                if i == 4:  # 赤5筒
                    pin_tiles.append('0')
                else:
                    pin_tiles.append(str(i + 1))
        
        # 索子 (18-26)
        for i in range(9):
            count = tiles_34[i + 18]
            for _ in range(count):
                if i == 4:  # 赤5索
                    sou_tiles.append('0')
                else:
                    sou_tiles.append(str(i + 1))
        
        # 字牌 (27-33)
        for i in range(7):
            count = tiles_34[i + 27]
            for _ in range(count):
                honor_tiles.append(str(i + 1))
        
        # 文字列を構築
        hand_string = ""
        if man_tiles:
            hand_string += "".join(man_tiles) + "m"
        if pin_tiles:
            hand_string += "".join(pin_tiles) + "p"
        if sou_tiles:
            hand_string += "".join(sou_tiles) + "s"
        if honor_tiles:
            hand_string += "".join(honor_tiles) + "z"
        
        return hand_string
    
    def _analyze_detailed_ukeire(self, hand_string, ai_recommended_tile, actual_tile):
        """詳細な受け入れ分析を実行"""
        try:
            if not hand_string:
                return "手牌情報の取得に失敗しました"
            
            # 手牌を34種配列に変換
            tiles_34_14 = robust_hand_parser(hand_string)
            unique_tiles_in_hand_14 = sorted([i for i, count in enumerate(tiles_34_14) if count > 0])
            
            shanten_calculator = Shanten()
            
            # AI推奨打牌と実際の打牌の受け入れ分析
            analysis_results = []
            
            for discard_tile_str in [ai_recommended_tile, actual_tile]:
                # 打牌する牌のインデックスを特定
                discard_index = None
                for idx in unique_tiles_in_hand_14:
                    tile_str = format_tiles_for_display([idx])
                    if tile_str == discard_tile_str:
                        discard_index = idx
                        break
                
                if discard_index is None:
                    continue
                
                # 打牌後の13枚手牌を作成
                hand_13_tiles = list(tiles_34_14)
                hand_13_tiles[discard_index] -= 1
                
                # シャンテン数を計算
                shanten_13 = shanten_calculator.calculate_shanten(hand_13_tiles)
                
                # 受け入れ牌を計算
                ukeire_tiles = {}
                
                if shanten_13 == 0:  # 聴牌の場合
                    for draw_index in range(34):
                        if tiles_34_14[draw_index] < 4 and draw_index != discard_index:
                            temp_hand_14 = list(hand_13_tiles)
                            temp_hand_14[draw_index] += 1
                            if shanten_calculator.calculate_shanten(temp_hand_14) == -1:
                                remaining_count = 4 - tiles_34_14[draw_index]
                                ukeire_tiles[draw_index] = remaining_count
                else:  # 聴牌していない場合
                    for draw_index in range(34):
                        if tiles_34_14[draw_index] < 4:
                            hand_14_after_draw = list(hand_13_tiles)
                            hand_14_after_draw[draw_index] += 1
                            shanten_after_draw_and_discard = get_shanten_after_best_discard(
                                hand_14_after_draw, shanten_calculator, 'calculate_shanten'
                            )
                            if shanten_after_draw_and_discard < shanten_13:
                                remaining_count = 4 - tiles_34_14[draw_index]
                                ukeire_tiles[draw_index] = remaining_count
                
                total_ukeire_count = sum(ukeire_tiles.values())
                ukeire_str = format_tiles_for_display(sorted(ukeire_tiles.keys()))
                
                analysis_results.append({
                    "discard": discard_tile_str,
                    "shanten": shanten_13,
                    "ukeire_tiles": ukeire_str,
                    "ukeire_count": total_ukeire_count,
                    "formatted": f"打{discard_tile_str} ({format_shanten(shanten_13)}) 摸[{ukeire_str} {total_ukeire_count}枚]"
                })
            
            return analysis_results
            
        except Exception as e:
            logger.warning(f"受け入れ分析エラー: {e}")
            return f"受け入れ分析中にエラーが発生しました: {e}"
        
    def _interpret_event_token(self, event_token):
        """イベントトークンを人間が理解しやすい形に変換"""
        if not event_token or event_token == "PAD":
            return "パディング"
            
        interpretations = {
            "DIS_": "プレイヤー{}が{}を捨てた",
            "TSU_": "プレイヤー{}が{}をツモした", 
            "INI_": "プレイヤー{}の初期配牌",
            "NAK_": "プレイヤー{}が鳴いた",
            "REA_": "プレイヤー{}がリーチした"
        }
        
        for prefix, template in interpretations.items():
            if event_token.startswith(prefix):
                parts = event_token.split('_')
                if len(parts) >= 3:
                    player_info = parts[1] if 'P' in parts[1] else ""
                    tile_info = parts[2] if len(parts) > 2 else ""
                    return template.format(player_info, tile_info)
                break
                
        return event_token
        
    def save_results(self, analysis_results, xml_path, round_index, player_id):
        """分析結果を保存"""
        # 出力ディレクトリの作成
        xml_name = Path(xml_path).stem
        output_dir = self.output_base_dir / f"{xml_name}_R{round_index}_P{player_id}"
        output_dir.mkdir(exist_ok=True)
        
        logger.info(f"結果保存中: {output_dir}")
        
        # 各局面のプロンプトとデータを保存
        for i, result in enumerate(analysis_results):
            if result is None:
                continue
                
            tsumo_count = result["moment_info"]["tsumo_count"]
            
            # プロンプト生成
            prompt_data = self.generate_prompt_for_moment(result)
            if prompt_data is None:
                continue
                
            # ファイル保存
            moment_dir = output_dir / f"tsumo_{tsumo_count:02d}"
            moment_dir.mkdir(exist_ok=True)
            
            # プロンプトテキストの保存
            prompt_file = moment_dir / "prompt.txt"
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(prompt_data["prompt_text"])
                
            # 詳細分析データの保存
            analysis_file = moment_dir / "analysis_data.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(prompt_data["analysis_data"], f, ensure_ascii=False, indent=2, default=str)
                
            # 簡易サマリの保存
            summary_file = moment_dir / "summary.json"
            summary_data = {
                "tsumo_count": tsumo_count,
                "player_id": result["moment_info"]["player_id"],
                "tsumo_tile": result["moment_info"]["tsumo_tile"],
                "predicted_tile": result["moment_info"]["predicted_tile"],
                "predicted_probability": result["moment_info"]["predicted_probability"],
                "actual_tile": result["moment_info"]["actual_tile"],
                "match": result["moment_info"]["predicted_tile"] == result["moment_info"]["actual_tile"]
            }
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
                
        # 全体サマリの作成
        self._create_overall_summary(analysis_results, output_dir)
        
        logger.info(f"保存完了: {len([r for r in analysis_results if r is not None])}個のプロンプト")
        return output_dir
        
    def _create_overall_summary(self, analysis_results, output_dir):
        """全体サマリの作成"""
        valid_results = [r for r in analysis_results if r is not None]
        
        if not valid_results:
            return
            
        # 統計情報の計算
        total_moments = len(valid_results)
        correct_predictions = sum(1 for r in valid_results 
                                 if r["moment_info"]["predicted_tile"] == r["moment_info"]["actual_tile"])
        accuracy = correct_predictions / total_moments if total_moments > 0 else 0
        
        avg_confidence = np.mean([r["moment_info"]["predicted_probability"] for r in valid_results])
        
        summary = {
            "overview": {
                "total_moments": total_moments,
                "correct_predictions": correct_predictions,
                "accuracy": accuracy,
                "average_confidence": float(avg_confidence)
            },
            "moments": [
                {
                    "tsumo_count": r["moment_info"]["tsumo_count"],
                    "predicted_tile": r["moment_info"]["predicted_tile"],
                    "actual_tile": r["moment_info"]["actual_tile"],
                    "confidence": r["moment_info"]["predicted_probability"],
                    "correct": r["moment_info"]["predicted_tile"] == r["moment_info"]["actual_tile"]
                }
                for r in valid_results
            ]
        }
        
        summary_file = output_dir / "overall_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            
    def run_batch_analysis(self, xml_path, round_index, player_id=None):
        """バッチ分析の実行"""
        logger.info("バッチ分析開始")
        
        # モデルロード
        self.load_models()
        
        # ツモ局面の抽出
        tsumo_moments = self.extract_all_tsumo_moments(xml_path, round_index, player_id)
        
        if not tsumo_moments:
            logger.warning("分析対象の局面が見つかりませんでした")
            return None
            
        logger.info(f"分析対象: {len(tsumo_moments)}個の局面")
        
        # 各局面の分析
        analysis_results = []
        for i, moment in enumerate(tsumo_moments):
            try:
                result = self.analyze_single_moment(moment, i, len(tsumo_moments))
                analysis_results.append(result)
            except Exception as e:
                logger.error(f"局面{i+1}の分析でエラー: {e}")
                analysis_results.append(None)
                
        # 結果保存
        output_dir = self.save_results(analysis_results, xml_path, round_index, 
                                     player_id if player_id is not None else "ALL")
        
        logger.info(f"バッチ分析完了: {output_dir}")
        return output_dir

    def _convert_dora_indicator_to_dora(self, indicator_tile_id):
        """ドラ表示牌から実際のドラを計算"""
        try:
            # tile_id_to_string でドラ表示牌の文字列を取得
            indicator_str = tile_id_to_string(indicator_tile_id)
            
            # 数牌の場合
            if indicator_str.endswith('m'):  # 萬子
                num = int(indicator_str[:-1])
                next_num = 1 if num == 9 else num + 1
                return f"{next_num}m"
            elif indicator_str.endswith('p'):  # 筒子
                num = int(indicator_str[:-1])
                next_num = 1 if num == 9 else num + 1
                return f"{next_num}p"
            elif indicator_str.endswith('s'):  # 索子
                num = int(indicator_str[:-1])
                next_num = 1 if num == 9 else num + 1
                return f"{next_num}s"
            elif indicator_str.endswith('z'):  # 字牌
                num = int(indicator_str[:-1])
                # 風牌: 1z(東) → 2z(南) → 3z(西) → 4z(北) → 1z(東)
                if 1 <= num <= 4:
                    next_num = 1 if num == 4 else num + 1
                    return f"{next_num}z"
                # 三元牌: 5z(白) → 6z(發) → 7z(中) → 5z(白)
                elif 5 <= num <= 7:
                    next_num = 5 if num == 7 else num + 1
                    return f"{next_num}z"
            
            # 特殊な表記の場合（東、南、西、北、白、發、中）
            dora_mapping = {
                '東': '南', '南': '西', '西': '北', '北': '東',
                '白': '發', '發': '中', '中': '白'
            }
            
            if indicator_str in dora_mapping:
                return dora_mapping[indicator_str]
            
            # 赤5の場合（0m, 0p, 0s は 5 として扱う）
            if indicator_str == '0m':
                return '6m'
            elif indicator_str == '0p':
                return '6p'
            elif indicator_str == '0s':
                return '6s'
            
            # 変換できない場合はそのまま返す
            return indicator_str
            
        except Exception as e:
            logger.warning(f"ドラ変換エラー: {e}, indicator: {indicator_tile_id}")
            return tile_id_to_string(indicator_tile_id)  # フォールバック


def main():
    parser = argparse.ArgumentParser(description="1局分全打牌の一括予測・分析・プロンプト生成")
    parser.add_argument("xml_file", help="天鳳形式のXML牌譜ファイルへのパス")
    parser.add_argument("round_index", type=int, help="対象局のインデックス (1から開始)")
    parser.add_argument("--player_id", type=int, help="対象プレイヤー (0-3, 未指定時は全プレイヤー)", 
                       choices=[0, 1, 2, 3])
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, 
                       help="学習済みモデルファイルへのパス")
    parser.add_argument("--output_dir", default="analysis_results", 
                       help="出力ディレクトリ")
    
    args = parser.parse_args()
    
    try:
        # バッチ分析システムの初期化
        system = BatchAnalysisSystem(
            model_path=args.model_path,
            output_base_dir=args.output_dir
        )
        
        # 分析実行
        output_dir = system.run_batch_analysis(
            xml_path=args.xml_file,
            round_index=args.round_index,
            player_id=args.player_id
        )
        
        if output_dir:
            print(f"\n=== バッチ分析完了 ===")
            print(f"出力ディレクトリ: {output_dir}")
            print(f"各ツモ局面のプロンプトが tsumo_XX/ フォルダに保存されました")
            print(f"overall_summary.json で全体の統計を確認できます")
            print("\n各フォルダの内容:")
            print("  - prompt.txt: LLM用プロンプト")
            print("  - analysis_data.json: 詳細分析データ")
            print("  - summary.json: 局面サマリ")
        else:
            print("分析に失敗しました")
            
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main() 