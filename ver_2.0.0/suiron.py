#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
麻雀AI推論スクリプト (Mahjong AI Inference Script)
訓練済みTransformerモデルを使用して麻雀の次の牌を予測する

使用方法:
    python suiron.py --model_path checkpoints_v_strong_flat_e500/checkpoint_epoch_214.pth --xml_path test_log.xml
    python suiron.py --model_path trained_model/mahjong_transformer_v_strong_flat_e500.pth --xml_path test_log.xml
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import logging
import sys
import os
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, deque
import xml.etree.ElementTree as ET
import urllib.parse

# プロジェクトモジュールのインポート
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from game_state import NUM_TILE_TYPES, MAX_EVENT_HISTORY, EVENT_TYPES, STATIC_FEATURE_DIM, GameState
    from tile_utils import tile_id_to_index, tile_id_to_string, tile_index_to_id
    from naki_utils import decode_naki
    from full_mahjong_parser import parse_full_mahjong_log
    from debug_and_process import process_event
except ImportError as e:
    print(f"[FATAL ERROR] 必要なモジュールのインポートに失敗: {e}")
    print("game_state.py, tile_utils.py, naki_utils.py, full_mahjong_parser.py, debug_and_process.py が必要です")
    sys.exit(1)

# モデル設定（train.pyと同じ）
D_MODEL = 256
NHEAD = 4
D_HID = 1024
NLAYERS = 4
DROPOUT = 0.1
ACTIVATION = 'gelu'
EVENT_FEATURE_DIM = 6

class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Encoding for Transformer"""
    def __init__(self, d_model, max_len=MAX_EVENT_HISTORY):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_modelは2で割り切れる必要があります。")
        freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('freqs', freqs)
        t = torch.arange(max_len).float()
        freqs_cis = torch.outer(t, self.freqs)
        self.register_buffer("cis", torch.polar(torch.ones_like(freqs_cis), freqs_cis))

    def forward(self, x):
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        shape = [1] * (x.dim() - 2) + [*x_complex.shape[-2:]]
        cis = self.cis[:x.shape[1]].view(*shape)
        x_rotated_complex = x_complex * cis
        x_rotated = torch.view_as_real(x_rotated_complex).flatten(start_dim=2)
        return x_rotated.type_as(x)

class MahjongTransformer(nn.Module):
    """麻雀Transformerモデル（train.pyと同じアーキテクチャ）"""
    def __init__(self, event_feature_dim, static_feature_dim):
        super().__init__()
        self.d_model = D_MODEL
        self.event_encoder = nn.Sequential(
            nn.Linear(event_feature_dim, D_MODEL),
            nn.LayerNorm(D_MODEL),
            nn.Dropout(DROPOUT)
        )
        self.pos_encoder = RotaryPositionalEncoding(D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=D_HID, 
            dropout=DROPOUT, activation=ACTIVATION, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, NLAYERS)
        self.static_encoder = nn.Sequential(
            nn.Linear(static_feature_dim, D_MODEL),
            nn.LayerNorm(D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, D_MODEL),
            nn.LayerNorm(D_MODEL)
        )
        self.attention_pool = nn.Sequential(
            nn.Linear(D_MODEL, 1),
            nn.Softmax(dim=1)
        )
        self.output_head = nn.Sequential(
            nn.Linear(D_MODEL * 2, D_MODEL),
            nn.LayerNorm(D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.LayerNorm(D_MODEL // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT * 0.5),
            nn.Linear(D_MODEL // 2, NUM_TILE_TYPES)
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=nn.init.calculate_gain('relu'))
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, event_seq, static_feat, attention_mask=None):
        event_encoded = self.event_encoder(event_seq)
        x = self.pos_encoder(event_encoded)
        transformer_output = self.transformer_encoder(x, src_key_padding_mask=attention_mask)
        attn_weights = self.attention_pool(transformer_output)
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(-1), 0.0)
        context_vector = torch.sum(attn_weights * transformer_output, dim=1)
        static_encoded = self.static_encoder(static_feat)
        combined = torch.cat([context_vector, static_encoded], dim=1)
        return self.output_head(combined)

class MahjongInference:
    """麻雀AI推論クラス"""
    
    def __init__(self, model_path: str, device: str = None):
        """
        推論クラスの初期化
        
        Args:
            model_path: 訓練済みモデルのパス
            device: 使用するデバイス ('cuda', 'cpu', Noneで自動選択)
        """
        # ログ設定（最初に設定）
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.game_state = GameState()
        
    def _load_model(self, model_path: str) -> MahjongTransformer:
        """訓練済みモデルをロード"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
            
        self.logger.info(f"モデルをロード中: {model_path}")
        
        # モデルインスタンス作成
        model = MahjongTransformer(
            event_feature_dim=EVENT_FEATURE_DIM,
            static_feature_dim=STATIC_FEATURE_DIM
        )
        
        # チェックポイントまたはモデル状態辞書をロード
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            # チェックポイント形式
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                self.logger.info(f"エポック {checkpoint['epoch']} のモデルをロードしました")
        else:
            # 直接モデル状態辞書形式
            model.load_state_dict(checkpoint)
            
        model.to(self.device)
        model.eval()
        
        self.logger.info(f"モデルを {self.device} にロード完了")
        return model
    
    def predict_next_tile(self, event_seq: np.ndarray, static_feat: np.ndarray, 
                         attention_mask: np.ndarray = None) -> Tuple[int, np.ndarray]:
        """
        次の牌を予測
        
        Args:
            event_seq: イベントシーケンス [MAX_EVENT_HISTORY, EVENT_FEATURE_DIM]
            static_feat: 静的特徴量 [STATIC_FEATURE_DIM]
            attention_mask: アテンションマスク [MAX_EVENT_HISTORY]
            
        Returns:
            (predicted_tile_index, prediction_probs): 予測された牌インデックスと確率分布
        """
        with torch.no_grad():
            # NumPy配列をTensorに変換
            event_seq_tensor = torch.from_numpy(event_seq).float().unsqueeze(0).to(self.device)
            static_feat_tensor = torch.from_numpy(static_feat).float().unsqueeze(0).to(self.device)
            
            if attention_mask is not None:
                attention_mask_tensor = torch.from_numpy(attention_mask).bool().unsqueeze(0).to(self.device)
            else:
                attention_mask_tensor = None
                
            # 予測実行
            logits = self.model(event_seq_tensor, static_feat_tensor, attention_mask_tensor)
            probs = torch.softmax(logits, dim=-1)
            
            # 最上位の予測を取得
            predicted_tile_index = torch.argmax(probs, dim=-1).item()
            prediction_probs = probs.squeeze().cpu().numpy()
            
            return predicted_tile_index, prediction_probs
    
    def analyze_game_log(self, xml_path: str, target_round: int = None, 
                        target_event: int = None) -> Dict[str, Any]:
        """
        XMLログファイルを解析して予測を実行
        
        Args:
            xml_path: XMLログファイルのパス
            target_round: 解析対象の局（1から開始、Noneで全局）
            target_event: 解析対象のイベント（Noneで局の最後まで）
            
        Returns:
            解析結果の辞書
        """
        self.logger.info(f"XMLログを解析中: {xml_path}")
        
        # XMLログをパース
        meta, rounds_data = parse_full_mahjong_log(xml_path)
        
        if not rounds_data:
            raise ValueError("解析可能な局データが見つかりません")
            
        self.logger.info(f"解析対象: {len(rounds_data)}局")
        
        results = {
            'meta': meta,
            'predictions': [],
            'summary': {
                'total_rounds': len(rounds_data),
                'analyzed_rounds': 0,
                'total_predictions': 0
            }
        }
        
        # 対象局を決定
        target_rounds = [target_round - 1] if target_round else range(len(rounds_data))
        
        for round_idx in target_rounds:
            if round_idx >= len(rounds_data):
                self.logger.warning(f"局 {target_round} は存在しません（最大: {len(rounds_data)}）")
                continue
                
            round_data = rounds_data[round_idx]
            round_results = self._analyze_round(round_data, target_event)
            results['predictions'].append(round_results)
            results['summary']['analyzed_rounds'] += 1
            results['summary']['total_predictions'] += len(round_results.get('predictions', []))
            
        return results
    
    def _analyze_round(self, round_data: Dict[str, Any], target_event: int = None) -> Dict[str, Any]:
        """単一局の解析"""
        round_index = round_data['round_index']
        events = round_data['events']
        
        self.logger.info(f"局 {round_index} を解析中（イベント数: {len(events)}）")
        
        # ゲーム状態を初期化
        self.game_state.reset_state()
        self.game_state.init_round(round_data)
        
        predictions = []
        analysis_data = {
            'round_index': round_index,
            'init_data': round_data['init'],
            'predictions': predictions,
            'events_processed': 0
        }
        
        # イベント処理
        end_event = target_event if target_event else len(events)
        
        for event_idx, event in enumerate(events[:end_event]):
            if event['tag'] in ['AGARI', 'RYUUKYOKU']:
                # 局終了
                break
            
            # イベントを処理
            try:
                process_event(self.game_state, event['tag'], event['attrib'], event_idx, events, process_only=True)
                analysis_data['events_processed'] += 1
                
                # イベント処理後に予測を実行（ツモイベントの場合）
                # ツモイベントは T123, U93, V44, W134 のような形式
                if event['tag'].startswith(('T', 'U', 'V', 'W')):
                    prediction_result = self._make_prediction_at_event(event_idx, events)
                    if prediction_result:
                        predictions.append(prediction_result)
                        
            except Exception as e:
                self.logger.warning(f"イベント {event_idx} の処理でエラー: {e}")
                
        return analysis_data
    
    def _make_prediction_at_event(self, event_idx: int, events: List[Dict]) -> Optional[Dict[str, Any]]:
        """特定のイベント時点で予測を実行"""
        try:
            # 現在のゲーム状態から特徴量を生成（現在のプレイヤーを取得）
            current_player = self.game_state.current_player
            if current_player == -1:
                current_player = 0  # デフォルトでプレイヤー0
            
            event_seq, static_feat = self.game_state.get_model_input(current_player)
            
            if event_seq is None or static_feat is None:
                return None
                
            # パディングマスクを生成
            padding_mask = (event_seq[:, 0] == EVENT_TYPES["PADDING"])
            
            # 予測実行
            predicted_tile_index, prediction_probs = self.predict_next_tile(
                event_seq, static_feat, padding_mask
            )
            
            # 予測結果を整理
            current_event = events[event_idx]
            predicted_tile_id = tile_index_to_id(predicted_tile_index)
            predicted_tile_name = tile_id_to_string(predicted_tile_id)
            
            # デバッグ情報を追加
            self.logger.debug(f"予測結果 - インデックス: {predicted_tile_index}, ID: {predicted_tile_id}, 名前: {predicted_tile_name}")
            
            # Top-5予測を取得
            top5_indices = np.argsort(prediction_probs)[-5:][::-1]
            top5_predictions = []
            for idx in top5_indices:
                tile_id = tile_index_to_id(idx)
                tile_name = tile_id_to_string(tile_id)
                prob = prediction_probs[idx]
                top5_predictions.append({
                    'tile_index': idx,
                    'tile_id': tile_id,
                    'tile_name': tile_name,
                    'probability': float(prob)
                })
            
            return {
                'event_index': event_idx,
                'event_tag': current_event['tag'],
                'event_attrib': current_event['attrib'],
                'predicted_tile': {
                    'tile_index': predicted_tile_index,
                    'tile_id': predicted_tile_id,
                    'tile_name': predicted_tile_name,
                    'probability': float(prediction_probs[predicted_tile_index])
                },
                'top5_predictions': top5_predictions,
                'game_context': {
                    'current_player': self.game_state.current_player,
                    'junme': self.game_state.junme,
                    'dora_indicators': self.game_state.dora_indicators.copy()
                }
            }
            
        except Exception as e:
            self.logger.warning(f"イベント {event_idx} での予測でエラー: {e}")
            return None
    
    def print_analysis_results(self, results: Dict[str, Any]):
        """解析結果を整形して表示"""
        print("\n" + "="*80)
        print("麻雀AI推論結果")
        print("="*80)
        
        meta = results['meta']
        if 'player_names' in meta:
            print(f"対局者: {', '.join(meta['player_names'])}")
        
        summary = results['summary']
        print(f"解析局数: {summary['analyzed_rounds']}/{summary['total_rounds']}")
        print(f"予測実行回数: {summary['total_predictions']}")
        
        for round_data in results['predictions']:
            round_index = round_data['round_index']
            predictions = round_data['predictions']
            
            print(f"\n--- 第{round_index}局 ---")
            print(f"処理イベント数: {round_data['events_processed']}")
            print(f"予測実行回数: {len(predictions)}")
            
            for i, pred in enumerate(predictions):
                print(f"\n  予測 {i+1}: イベント {pred['event_index']} ({pred['event_tag']})")
                print(f"    予測牌: {pred['predicted_tile']['tile_name']} (確率: {pred['predicted_tile']['probability']:.3f})")
                
                print("    Top-5予測:")
                for j, top_pred in enumerate(pred['top5_predictions'][:3]):  # Top-3のみ表示
                    print(f"      {j+1}. {top_pred['tile_name']} ({top_pred['probability']:.3f})")
                
                context = pred['game_context']
                print(f"    コンテキスト: プレイヤー{context['current_player']}, 巡目{context['junme']:.1f}")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='麻雀AI推論スクリプト')
    parser.add_argument('--model_path', type=str, required=True,
                       help='訓練済みモデルのパス')
    parser.add_argument('--xml_path', type=str, required=True,
                       help='解析するXMLログファイルのパス')
    parser.add_argument('--round', type=int, default=None,
                       help='解析対象の局番号（1から開始、指定なしで全局）')
    parser.add_argument('--event', type=int, default=None,
                       help='解析対象のイベント番号（指定なしで局の最後まで）')
    parser.add_argument('--device', type=str, default=None,
                       help='使用デバイス (cuda/cpu、指定なしで自動選択)')
    parser.add_argument('--output', type=str, default=None,
                       help='結果出力ファイル（JSON形式）')
    
    args = parser.parse_args()
    
    try:
        # 推論インスタンス作成
        inference = MahjongInference(args.model_path, args.device)
        
        # 解析実行
        results = inference.analyze_game_log(args.xml_path, args.round, args.event)
        
        # 結果表示
        inference.print_analysis_results(results)
        
        # 結果保存（指定された場合）
        if args.output:
            import json
            # NumPy配列をリストに変換してJSON化可能にする
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                return obj
            
            # 再帰的にNumPy型を変換
            def clean_for_json(obj):
                if isinstance(obj, dict):
                    return {k: clean_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_for_json(item) for item in obj]
                else:
                    return convert_numpy(obj)
            
            clean_results = clean_for_json(results)
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(clean_results, f, ensure_ascii=False, indent=2)
            print(f"\n結果を {args.output} に保存しました")
            
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

