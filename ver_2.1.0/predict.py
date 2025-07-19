# predict.py (Transformer版 - SHAP説明機能・アテンション可視化付き・日本語化)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import math
import glob
import time
import logging
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
# import types # types モジュールは不要になったため削除
import h5py # h5py を明示的にインポート
import joblib # joblib を明示的にインポート
import json # JSON出力用
from datetime import datetime

# SHAPとMatplotlibをインポート
try:
    import shap
    # 日本語フォントの設定 (環境に応じて調整してください)
    # try:
    #     plt.rcParams['font.family'] = 'IPAexGothic' # 例: IPAexGothic
    # except RuntimeError:
    #     print("[警告] IPAexGothic フォントが見つかりません。デフォルトフォントを使用します。")
    # plt.rcParams['axes.unicode_minus'] = False
    shap_available = True
except ImportError:
    print("[警告] `shap` または `matplotlib` ライブラリが見つかりません。SHAP説明機能・プロットはスキップされます。")
    print("インストールするには: pip install shap matplotlib")
    shap_available = False

# ---プロジェクトモジュールのインポート---
try:
    from full_mahjong_parser import parse_full_mahjong_log
    from game_state import GameState, NUM_TILE_TYPES, MAX_EVENT_HISTORY, STATIC_FEATURE_DIM, EVENT_TYPES, NUM_PLAYERS
    from tile_utils import tile_id_to_string, tile_id_to_index, tile_index_to_id
    from naki_utils import decode_naki
    print("プロジェクトモジュールを正常にインポートしました。")
except ImportError as e:
    print(f"[エラー] プロジェクトモジュールのインポートに失敗しました: {e}")
    print("必要なファイル (game_state.py, tile_utils.pyなど) が同じディレクトリにあるか確認してください。")
    exit(1)

# train2.py から DATA_HDF5_PATH を参照
DATA_HDF5_PATH = "/home/ubuntu/Documents/mahjong-XAI/ver_1.1.10/training_data/mahjong_imitation_data_v1110.hdf5"
if not os.path.exists(DATA_HDF5_PATH):
    print(f"[警告] SHAP背景データ用のHDF5ファイルが見つかりません: {DATA_HDF5_PATH}")

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- クラス定義 (MahjongTransformerV2, RotaryPositionalEncoding, Custom Layers) ---
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_EVENT_HISTORY):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be divisible by 2 for Rotary Positional Encoding.")
        self.d_model = d_model
        self.max_len = max_len
        self.dim_half = d_model // 2
        freqs = 1.0 / (10000 ** (torch.arange(0, self.dim_half).float() / self.dim_half))
        self.register_buffer('freqs', freqs)
        pos_seq = torch.arange(max_len).float()
        self.register_buffer('pos_seq', pos_seq)

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len > self.max_len:
            logging.warning(f"RoPE: Input sequence length {seq_len} > precomputed max_len {self.max_len}. Recomputing positions.")
            positions = torch.arange(seq_len, device=x.device).float().unsqueeze(0)
        else:
            positions = self.pos_seq[:seq_len].unsqueeze(0).to(x.device)
        angles = positions.unsqueeze(-1) * self.freqs.unsqueeze(0).to(x.device)
        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_even_rotated = x_even * cos_angles - x_odd * sin_angles
        x_odd_rotated = x_even * sin_angles + x_odd * cos_angles
        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = x_even_rotated
        x_rotated[..., 1::2] = x_odd_rotated
        return x_rotated

class OriginalMahjongTransformerV2(nn.Module): # train2.py のモデル定義をそのまま使用
    """イベント系列と静的特徴を入力とするTransformerモデル (train2.pyのMahjongTransformerV2と同じ)"""
    def __init__(self, event_feature_dim, static_feature_dim, d_model=256, nhead=4, d_hid=1024, nlayers=4, dropout=0.1, activation='relu', output_dim=NUM_TILE_TYPES):
        super().__init__()
        self.d_model = d_model
        self.event_encoder = nn.Sequential(
            nn.Linear(event_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        self.pos_encoder = RotaryPositionalEncoding(d_model, max_len=MAX_EVENT_HISTORY)
        encoder_layer = nn.TransformerEncoderLayer( # 標準のTransformerEncoderLayer
            d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout,
            activation=activation, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers) # 標準のTransformerEncoder
        self.static_encoder = nn.Sequential(
            nn.Linear(static_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        self.output_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, output_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                gain = nn.init.calculate_gain('relu') if 'relu' in name.lower() or 'gelu' in name.lower() else 1.0
                nn.init.xavier_normal_(p, gain=gain)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, event_seq, static_feat, attention_mask=None):
        event_encoded = self.event_encoder(event_seq)
        pos_encoded = self.pos_encoder(event_encoded)
        transformer_output = self.transformer_encoder(pos_encoded, src_key_padding_mask=attention_mask)
        attn_weights = self.attention_pool(transformer_output)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1)
            attn_weights = attn_weights.masked_fill(mask_expanded, 0.0)
        context_vector = torch.sum(attn_weights * transformer_output, dim=1)
        static_encoded = self.static_encoder(static_feat)
        combined = torch.cat([context_vector, static_encoded], dim=1)
        return self.output_head(combined)

class CustomTransformerEncoderLayerWithAttention(nn.TransformerEncoderLayer):
    """アテンションウェイトを返すカスタムTransformerEncoderLayer（デバッグ機能付き）"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False): # is_causal を追加 (PyTorch 2.0以降)
        # PyTorch 1.9 以降の TransformerEncoderLayer の forward シグネチャに合わせる
        # src_key_padding_mask は self_attn に渡す
        # src_mask は self_attn に渡す
        
        # Self-attention block
        # need_weights=True でアテンションウェイトを取得
        # average_attn_weights=True でヘッドの平均を取る (MultiheadAttentionのデフォルトはFalse)
        # ここでは average_attn_weights=True を指定して (batch, tgt_len, src_len) の形にする
        x = src
        attn_output, self.attn_weights = self.self_attn(x, x, x,
                                                       attn_mask=src_mask,
                                                       key_padding_mask=src_key_padding_mask,
                                                       need_weights=True,
                                                       average_attn_weights=True) # ヘッド平均
        if self.attn_weights is not None:
             logging.debug(f"[CustomLayer] Raw attention weights shape: {self.attn_weights.shape}")
        else:
             logging.debug("[CustomLayer] Attention weights is None!")

        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feedforward block
        x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))
        x = self.norm2(x)
        return x

class MahjongTransformerV2WithAttention(nn.Module):
    """アテンションウェイトを取得できるように修正したモデル"""
    def __init__(self, event_feature_dim, static_feature_dim, d_model=256, nhead=4, d_hid=1024, nlayers=4, dropout=0.1, activation='relu', output_dim=NUM_TILE_TYPES):
        super().__init__()
        self.d_model = d_model
        self.event_encoder = nn.Sequential(
            nn.Linear(event_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        self.pos_encoder = RotaryPositionalEncoding(d_model, max_len=MAX_EVENT_HISTORY)
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayerWithAttention( # カスタムレイヤーを使用
                d_model=d_model, nhead=nhead, dim_feedforward=d_hid,
                dropout=dropout, activation=activation, batch_first=True, norm_first=True
            ) for _ in range(nlayers)
        ])
        self.static_encoder = nn.Sequential(
            nn.Linear(static_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        self.output_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, output_dim)
        )
        self._init_weights()
        # アクティベーション取得用のフック
        self.output_head[0].register_forward_hook(get_activation_hook_for_analysis('combined_vector'))

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                gain = nn.init.calculate_gain('relu') if 'relu' in name.lower() or 'gelu' in name.lower() else 1.0
                nn.init.xavier_normal_(p, gain=gain)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, event_seq, static_feat, attention_mask=None, return_attention=False):
        event_encoded = self.event_encoder(event_seq)
        pos_encoded = self.pos_encoder(event_encoded)
        x = pos_encoded
        attention_weights_all_layers = []
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=attention_mask) # src_mask は通常 Self-Attention では None
            if return_attention and hasattr(layer, 'attn_weights') and layer.attn_weights is not None:
                attention_weights_all_layers.append(layer.attn_weights)
        transformer_output = x
        attn_weights_pool = self.attention_pool(transformer_output) # ここはプーリング用のアテンション
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1)
            attn_weights_pool = attn_weights_pool.masked_fill(mask_expanded, 0.0)
        context_vector = torch.sum(attn_weights_pool * transformer_output, dim=1)
        static_encoded = self.static_encoder(static_feat)
        combined = torch.cat([context_vector, static_encoded], dim=1)
        output = self.output_head(combined)
        if return_attention:
            return output, attention_weights_all_layers
        return output

def convert_to_attention_model(original_model_path, event_dim, static_dim, device):
    """学習済みモデルをロードし、アテンション取得対応版に変換する"""
    logging.info("アテンション対応モデルに変換中...")
    # まず、元のモデル構造で重みをロード
    original_model_instance = OriginalMahjongTransformerV2(
        event_feature_dim=event_dim, static_feature_dim=static_dim
    ).to(device)
    
    checkpoint = torch.load(original_model_path, map_location=device)
    state_dict_to_load = checkpoint.get('model_state_dict', checkpoint)

    if hasattr(original_model_instance, '_orig_mod'): # torch.compile されている場合
        original_model_instance._orig_mod.load_state_dict(state_dict_to_load)
    else:
        original_model_instance.load_state_dict(state_dict_to_load)
    original_model_instance.eval()
    logging.info("元のモデルの重みをロード完了。")

    # 新しいアテンション対応モデルを作成
    attention_model = MahjongTransformerV2WithAttention(
        event_feature_dim=event_dim, static_feature_dim=static_dim,
        d_model=original_model_instance.d_model, # 元のモデルのパラメータを使用
        nhead=original_model_instance.transformer_encoder.layers[0].self_attn.num_heads,
        d_hid=original_model_instance.transformer_encoder.layers[0].linear1.out_features,
        nlayers=len(original_model_instance.transformer_encoder.layers),
        dropout=original_model_instance.transformer_encoder.layers[0].dropout1.p, # dropout値を取得
        activation= 'relu', # train2.py のデフォルト 'relu' or 'gelu'
        output_dim=original_model_instance.output_head[-1].out_features
    ).to(device)

    # 重みを慎重にコピー
    attention_model.event_encoder.load_state_dict(original_model_instance.event_encoder.state_dict())
    attention_model.pos_encoder.load_state_dict(original_model_instance.pos_encoder.state_dict())
    attention_model.static_encoder.load_state_dict(original_model_instance.static_encoder.state_dict())
    attention_model.attention_pool.load_state_dict(original_model_instance.attention_pool.state_dict())
    attention_model.output_head.load_state_dict(original_model_instance.output_head.state_dict())

    # TransformerEncoderLayer の重みを CustomTransformerEncoderLayerWithAttention にコピー
    for i in range(len(attention_model.encoder_layers)):
        original_layer = original_model_instance.transformer_encoder.layers[i]
        custom_layer = attention_model.encoder_layers[i]
        
        custom_layer.self_attn.load_state_dict(original_layer.self_attn.state_dict())
        custom_layer.linear1.load_state_dict(original_layer.linear1.state_dict())
        custom_layer.linear2.load_state_dict(original_layer.linear2.state_dict())
        custom_layer.norm1.load_state_dict(original_layer.norm1.state_dict())
        custom_layer.norm2.load_state_dict(original_layer.norm2.state_dict())
        custom_layer.dropout.p = original_layer.dropout.p # dropout レートもコピー
        custom_layer.dropout1.p = original_layer.dropout1.p
        custom_layer.dropout2.p = original_layer.dropout2.p
        # activation は文字列なので、型が同じならOK

    attention_model.eval()
    logging.info("アテンション対応モデルへの変換完了。")
    return attention_model


# --- 設定 ---
DEFAULT_MODEL_PATH = "../ver_1.1.11/trained_model/mahjong_transformer_v1111_large_compiled_2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# グローバル変数（アクティベーション保存用）
activations_storage = {}

def get_activation_hook_for_analysis(name):
    """中間層のアクティベーションを取得するためのフック"""
    def hook(model, input, output):
        if isinstance(input, tuple) and len(input) > 0:
            activations_storage[name] = input[0].detach().cpu().numpy()
        else:
            activations_storage[name] = input.detach().cpu().numpy()
    return hook

logging.info(f"使用デバイス: {DEVICE}")


# --- ヘルパー関数 (変更なし) ---
def format_hand(hand_ids):
    if not hand_ids: return "なし"
    sorted_ids = sorted(hand_ids, key=lambda t: (tile_id_to_index(t), t))
    return " ".join([tile_id_to_string(t) for t in sorted_ids])

def format_discards(discard_list):
    if not discard_list: return "なし"
    return " ".join([f"{tile_id_to_string(t)}{'*' if tsumogiri else ''}" for t, tsumogiri in discard_list])

def format_melds(meld_list_dicts):
    if not meld_list_dicts: return "なし"
    meld_strs = []
    for meld_info in meld_list_dicts:
        m_type = meld_info.get('type', '不明')
        m_tiles = meld_info.get('tiles', [])
        from_who_abs = meld_info.get('from_who', -1)
        called_tile = meld_info.get('called_tile', -1)
        tiles_str = " ".join([tile_id_to_string(t) for t in sorted(m_tiles, key=lambda x: (tile_id_to_index(x),x))])
        from_str = f" P{from_who_abs}" if from_who_abs != -1 and m_type not in ["暗槓", "加槓"] else ""
        trigger_str = f"({tile_id_to_string(called_tile)})" if called_tile != -1 and m_type != "暗槓" else ""
        meld_strs.append(f"{m_type}{trigger_str}[{tiles_str}]{from_str}")
    return " / ".join(meld_strs)

# --- モデルロード関数 (アテンション対応モデルを返すように変更) ---
def load_trained_model_for_prediction(model_path, event_dim, static_dim, visualize_attention_flag):
    if visualize_attention_flag:
        # アテンション可視化が要求された場合、変換関数を使用
        model = convert_to_attention_model(model_path, event_dim, static_dim, DEVICE)
    else:
        # 通常のモデルロード
        model_params = {
            'event_feature_dim': event_dim, 'static_feature_dim': static_dim,
            'd_model': 256, 'nhead': 4, 'd_hid': 1024, 'nlayers': 4,
            'dropout': 0.1, 'activation': 'relu', 'output_dim': NUM_TILE_TYPES
        }
        logging.info(f"以下のパラメータでモデルを初期化します: {model_params}")
        model = OriginalMahjongTransformerV2(**model_params).to(DEVICE) # 元のモデルクラスを使用
        checkpoint = torch.load(model_path, map_location=DEVICE)
        state_dict_to_load = checkpoint.get('model_state_dict', checkpoint)
        if hasattr(model, '_orig_mod'):
            model._orig_mod.load_state_dict(state_dict_to_load)
        else:
            model.load_state_dict(state_dict_to_load)
        model.eval()
        logging.info(f"モデルを正常に読み込みました: {model_path}")
    return model

# --- 局面復元関数 (変更なし) ---
def reconstruct_game_state_at_tsumo(xml_path, target_round_index, target_tsumo_event_count_in_round):
    logging.info(f"牌譜ファイル {xml_path} を解析中...")
    try:
        meta, rounds_data = parse_full_mahjong_log(xml_path)
    except FileNotFoundError:
        logging.error(f"[エラー] 牌譜ファイルが見つかりません: {xml_path}")
        raise
    except Exception as e:
        logging.error(f"[エラー] 牌譜ファイルの解析中にエラーが発生しました: {e}")
        raise

    if not (1 <= target_round_index <= len(rounds_data)):
        raise ValueError(f"指定された局インデックスが無効です: {target_round_index} (利用可能な範囲: 1-{len(rounds_data)})")

    round_data = rounds_data[target_round_index - 1]
    game_state = GameState()
    logging.info(f"第{target_round_index}局の初期状態を構築中...")
    try:
        game_state.init_round(round_data)
    except Exception as e:
        logging.error(f"[エラー] GameState の初期化中にエラーが発生しました: {e}")
        raise

    current_tsumo_count = 0
    target_tsumo_event_info = None
    actual_discard_event_info = None
    events = round_data.get("events", [])
    logging.info(f"イベントを再生し、{target_tsumo_event_count_in_round} 回目のツモを探します...")

    for i, event_xml in enumerate(events):
        tag = event_xml["tag"]
        attrib = event_xml["attrib"]
        processed_event_this_iteration = False
        try:
            tsumo_player_id = -1; tsumo_pai_id = -1
            for t_tag, p_id in GameState.TSUMO_TAGS.items():
                if tag.startswith(t_tag) and tag[1:].isdigit():
                    try: tsumo_pai_id = int(tag[1:]); tsumo_player_id = p_id; processed_event_this_iteration = True; break
                    except (ValueError, IndexError): continue
            if processed_event_this_iteration:
                current_tsumo_count += 1
                if current_tsumo_count == target_tsumo_event_count_in_round:
                    logging.info(f"ターゲットのツモ ({current_tsumo_count}回目) を発見しました。")
                    target_tsumo_event_info = {"player": tsumo_player_id, "pai": tsumo_pai_id, "xml": event_xml}
                    game_state.process_tsumo(tsumo_player_id, tsumo_pai_id)
                    if i + 1 < len(events):
                        next_event_xml = events[i+1]; next_tag = next_event_xml["tag"]
                        for d_tag, p_id_next in GameState.DISCARD_TAGS.items():
                            if next_tag.startswith(d_tag) and next_tag[1:].isdigit() and p_id_next == tsumo_player_id:
                                try:
                                    discard_pai_id = int(next_tag[1:]); tsumogiri = next_tag[0].islower()
                                    actual_discard_event_info = {"player": p_id_next, "pai": discard_pai_id, "tsumogiri": tsumogiri, "xml": next_event_xml}
                                    break
                                except (ValueError, IndexError): continue
                    logging.info("指定局面の状態復元が完了しました。")
                    return game_state, target_tsumo_event_info, actual_discard_event_info
                else:
                    game_state.process_tsumo(tsumo_player_id, tsumo_pai_id)
                continue
            discard_player_id = -1; discard_pai_id = -1; tsumogiri = False
            for d_tag, p_id in GameState.DISCARD_TAGS.items():
                if tag.startswith(d_tag) and tag[1:].isdigit():
                    try: discard_pai_id = int(tag[1:]); discard_player_id = p_id; tsumogiri = tag[0].islower(); processed_event_this_iteration = True; break
                    except (ValueError, IndexError): continue
            if processed_event_this_iteration:
                game_state.process_discard(discard_player_id, discard_pai_id, tsumogiri)
                continue
            if not processed_event_this_iteration and tag == "N":
                try:
                    naki_player_id = int(attrib.get("who", -1)); meld_code = int(attrib.get("m", "0"))
                    if naki_player_id != -1: game_state.process_naki(naki_player_id, meld_code)
                    processed_event_this_iteration = True
                except (ValueError, KeyError, Exception) as e: logging.warning(f"[警告] 鳴きイベント(N)の処理中にエラー: {e}, Attrib: {attrib}")
                continue
            if not processed_event_this_iteration and tag == "REACH":
                 try:
                     reach_player_id = int(attrib.get("who", -1)); step = int(attrib.get("step", 0))
                     if reach_player_id != -1: game_state.process_reach(reach_player_id, step)
                     processed_event_this_iteration = True
                 except (ValueError, KeyError, Exception) as e: logging.warning(f"[警告] リーチイベント(REACH)の処理中にエラー: {e}, Attrib: {attrib}")
                 continue
            if not processed_event_this_iteration and tag == "DORA":
                 try:
                     hai = int(attrib.get("hai", -1))
                     if hai != -1: game_state.process_dora(hai)
                     processed_event_this_iteration = True
                 except (ValueError, KeyError, Exception) as e: logging.warning(f"[警告] ドラ表示イベント(DORA)の処理中にエラー: {e}, Attrib: {attrib}")
                 continue
            if not processed_event_this_iteration and (tag == "AGARI" or tag == "RYUUKYOKU"):
                 logging.info(f"局終了イベント ({tag}) を検出しました。")
                 try:
                     if tag == "AGARI": game_state.process_agari(attrib)
                     else: game_state.process_ryuukyoku(attrib)
                 except Exception as e: logging.warning(f"[警告] 局終了イベントの処理中にエラー: {e}, Attrib: {attrib}")
                 processed_event_this_iteration = True; break
        except Exception as e:
            logging.error(f"[エラー] イベント {i} (タグ: {tag}, 属性: {attrib}) の処理中に予期せぬエラーが発生しました: {e}", exc_info=True)
            raise e
    raise ValueError(f"指定されたツモ回数 ({target_tsumo_event_count_in_round}) に到達する前に局が終了、またはイベントがありませんでした（局: {target_round_index}）。")

# --- 打牌予測関数 (アテンション対応モデルを使用) ---
def predict_discard(model, game_state: GameState, player_id: int, visualize_attention_flag=False):
    try:
        event_sequence = game_state.get_event_sequence_features()
        static_features = game_state.get_static_features(player_id)
    except Exception as e:
        logging.error(f"[エラー] 特徴量生成中にエラーが発生しました: {e}")
        raise

    seq_tensor = torch.tensor(event_sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    static_tensor = torch.tensor(static_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    padding_code_float = float(EVENT_TYPES["PADDING"])
    mask_tensor = (seq_tensor[:, :, 0] == padding_code_float).to(DEVICE)

    collected_attention_weights = []
    with torch.no_grad():
        try:
            if visualize_attention_flag and isinstance(model, MahjongTransformerV2WithAttention):
                outputs, collected_attention_weights = model(seq_tensor, static_tensor, mask_tensor, return_attention=True)
            else: # 通常の予測、またはモデルがアテンション対応でない場合
                outputs = model(seq_tensor, static_tensor, mask_tensor)
        except Exception as e:
            logging.error(f"[エラー] モデルのforward計算中にエラーが発生しました: {e}", exc_info=True)
            raise
        probabilities = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    valid_discard_indices = game_state.get_valid_discard_options(player_id)
    best_prob = -1.0
    best_index = -1
    if not valid_discard_indices:
        logging.warning("[警告] 有効な打牌選択肢がありません！")
        best_index = np.argmax(probabilities)
        if 0 <= best_index < len(probabilities): best_prob = probabilities[best_index]
        else: logging.error("[エラー] 確率配列から最大値を取得できませんでした。"); return 0, 0.0, probabilities, collected_attention_weights
    else:
        for index in valid_discard_indices:
            if 0 <= index < NUM_TILE_TYPES:
                if probabilities[index] > best_prob: best_prob = probabilities[index]; best_index = index
            else: logging.warning(f"[警告] 無効な牌インデックス {index} が有効選択肢に含まれています。")
        if best_index == -1 and valid_discard_indices:
            logging.warning(f"[警告] 有効牌の中から最良の打牌を決定できませんでした。最初の有効牌 ({valid_discard_indices[0]}) を選択します。")
            best_index = valid_discard_indices[0]
            best_prob = probabilities[best_index] if 0 <= best_index < len(probabilities) else 0.0
    if not (0 <= best_index < NUM_TILE_TYPES):
        logging.error(f"[エラー] 最終的な打牌インデックス ({best_index}) が不正です。")
        return 0, 0.0, probabilities, collected_attention_weights
    return best_index, best_prob, probabilities, collected_attention_weights

# --- 特徴量名生成関数 (変更なし) ---
def generate_feature_names(event_dim_actual, static_dim_actual, seq_len_actual):
    feature_names = []
    logging.info(f"特徴量名を生成中... (シーケンス長: {seq_len_actual}, イベント次元: {event_dim_actual}, 静的次元: {static_dim_actual})")
    event_field_names_base = ["タイプ", "プレイヤー", "牌Idx", "巡目"]
    event_field_names_specific = ["データA", "データB"]
    current_event_names = list(event_field_names_base)
    if event_dim_actual > len(event_field_names_base):
        num_specific_fields = event_dim_actual - len(event_field_names_base)
        current_event_names.extend(event_field_names_specific[:num_specific_fields])
    current_event_names = current_event_names[:event_dim_actual]
    for i in range(seq_len_actual):
        for j, name_suffix in enumerate(current_event_names):
            feature_names.append(f"Event_{i}_{name_suffix}")
    idx_counter_static = 0
    game_context_names = ["局風", "本場", "供託", "親プレイヤーIdx", "壁残枚数", "自身が親か", "巡目(GameState)", "ドラ表示牌数"]
    for name in game_context_names: feature_names.append(f"静的_{name}"); idx_counter_static +=1
    player_specific_names = ["リーチ状態", "リーチ巡目", "自身の捨て牌数", "自身の副露数", "自身の手牌数"]
    for name in player_specific_names: feature_names.append(f"静的_{name}"); idx_counter_static +=1
    tile_kind_names = [tile_id_to_string(tile_index_to_id(i)) for i in range(NUM_TILE_TYPES)]
    for name in tile_kind_names: feature_names.append(f"静的_手牌_{name}"); idx_counter_static +=1
    for name in tile_kind_names: feature_names.append(f"静的_ドラ表示_{name}"); idx_counter_static +=1
    for name in tile_kind_names: feature_names.append(f"静的_自捨牌_{name}"); idx_counter_static +=1
    for name in tile_kind_names: feature_names.append(f"静的_全見え牌_{name}"); idx_counter_static +=1
    player_relative_names = ["自身", "下家", "対面", "上家"]
    for p_name in player_relative_names:
        feature_names.append(f"静的_{p_name}_IsSelf"); idx_counter_static +=1
        feature_names.append(f"静的_{p_name}_ReachAccepted"); idx_counter_static +=1
    expected_total_len = seq_len_actual * event_dim_actual + static_dim_actual
    current_static_len_generated = idx_counter_static
    if current_static_len_generated != STATIC_FEATURE_DIM:
         logging.warning(f"[警告] 生成された静的特徴量名の数 ({current_static_len_generated}) が期待値 ({STATIC_FEATURE_DIM}) と異なります。")
         diff = STATIC_FEATURE_DIM - current_static_len_generated
         if diff > 0:
             for i in range(diff): feature_names.append(f"静的_不明_{i}")
         elif diff < 0:
             feature_names = feature_names[:-(abs(diff))]
    if len(feature_names) != expected_total_len:
        logging.warning(f"[警告] 生成された総特徴量名の数 ({len(feature_names)}) が期待値 ({expected_total_len}) と異なります。")
        diff_total = expected_total_len - len(feature_names)
        if diff_total > 0:
            feature_names.extend([f"総合_不明_{i}" for i in range(diff_total)])
        else:
            feature_names = feature_names[:expected_total_len]
    logging.info(f"特徴量名の生成完了 (合計: {len(feature_names)}個, うち静的: {current_static_len_generated}個)")
    return feature_names

# --- SHAP説明関数 (HDF5インデックスエラー修正済み) ---
def explain_prediction_with_shap(model, background_data_path, instance_to_explain, feature_names, target_class_index, n_shap_samples=100, n_bg_summary_samples=50):
    if not shap_available:
        logging.warning("SHAPライブラリが利用できないため、説明をスキップします。")
        return None
    logging.info("\n--- SHAP 説明生成開始 ---")
    start_time = time.time()
    target_class_name = tile_id_to_string(tile_index_to_id(target_class_index)) if target_class_index != -1 else "N/A"
    logging.info(f"対象クラス: Index={target_class_index}, 牌種={target_class_name}")
    event_seq_instance, static_feat_instance, _ = instance_to_explain
    seq_len = event_seq_instance.shape[0]; event_dim = event_seq_instance.shape[1]
    bg_sequences_list = []; bg_static_features_list = []
    if background_data_path and os.path.exists(background_data_path):
        try:
            with h5py.File(background_data_path, "r", swmr=True) as hf:
                num_total_bg_samples = hf["labels"].shape[0]
                if num_total_bg_samples == 0: logging.warning("[警告] 背景データファイルにサンプルがありません。")
                else:
                    n_samples_to_load = min(n_bg_summary_samples, num_total_bg_samples)
                    sample_indices = np.sort(np.random.choice(num_total_bg_samples, size=n_samples_to_load, replace=False))
                    bg_sequences_list = hf["sequences"][sample_indices]
                    bg_static_features_list = hf["static_features"][sample_indices]
            logging.info(f"{len(bg_sequences_list)} 件の背景データを {background_data_path} からロードしました。")
        except Exception as e: logging.warning(f"[警告] 背景データのロードに失敗: {e}。SHAP説明の質が低下する可能性あり。")
    else: logging.warning("[警告] 背景データファイルが見つからないかパスが指定されていません。")
    # エラー修正: numpy arrayのbooleanチェック
    if len(bg_sequences_list) == 0 or len(bg_static_features_list) == 0:
        logging.warning("[警告] 有効な背景データがないため、ダミーの背景データ（ゼロベクトル）を使用します。")
        bg_sequences_np = np.zeros((1, seq_len, event_dim), dtype=np.float32)
        bg_static_features_np = np.zeros((1, static_feat_instance.shape[0]), dtype=np.float32)
    else:
        bg_sequences_np = np.array(bg_sequences_list, dtype=np.float32)
        bg_static_features_np = np.array(bg_static_features_list, dtype=np.float32)
    def model_predict_proba_flat(flat_input_tensor_np):
        if isinstance(flat_input_tensor_np, np.ndarray): flat_input_tensor = torch.tensor(flat_input_tensor_np, dtype=torch.float32).to(DEVICE)
        else: flat_input_tensor = flat_input_tensor_np.to(DEVICE)
        batch_size = flat_input_tensor.shape[0]
        try:
            event_seq = flat_input_tensor[:, :(seq_len * event_dim)].reshape(batch_size, seq_len, event_dim)
            static_feat = flat_input_tensor[:, (seq_len * event_dim):]
        except Exception as e: logging.error(f"[エラー] SHAPラッパー内でのテンソル再構成に失敗: {e}"); return np.zeros((batch_size,))
        padding_code_float = float(EVENT_TYPES["PADDING"])
        mask = (event_seq[:, :, 0] == padding_code_float)
        with torch.no_grad():
            outputs = model(event_seq, static_feat, mask)
            probabilities = F.softmax(outputs, dim=1)
        return probabilities[:, target_class_index].cpu().numpy()
    bg_flat = np.concatenate([bg_sequences_np.reshape(bg_sequences_np.shape[0], -1), bg_static_features_np], axis=1)
    instance_flat = np.concatenate([event_seq_instance.flatten(), static_feat_instance]).reshape(1, -1)
    background_summary = bg_flat[:min(len(bg_flat), n_bg_summary_samples)]
    if background_summary.shape[0] == 0:
        logging.warning("[警告] 背景データのサマリーが空です。SHAP Explainer はダミー背景で初期化されます。")
        background_summary = np.zeros((1, instance_flat.shape[1]))
    logging.info(f"SHAP背景データとして {background_summary.shape[0]} サンプルを使用します。")
    try:
        logging.info("SHAP KernelExplainer を初期化中...")
        explainer = shap.KernelExplainer(model_predict_proba_flat, background_summary)
    except Exception as e: logging.error(f"[エラー] SHAP Explainer の初期化に失敗: {e}"); return None
    logging.info(f"SHAP値を計算中 (nsamples={n_shap_samples})...")
    try:
        shap_values_for_instance = explainer.shap_values(instance_flat, nsamples=n_shap_samples)
    except Exception as e: logging.error(f"[エラー] SHAP値の計算中にエラーが発生しました: {e}"); return None
    calculation_time = time.time() - start_time
    logging.info(f"SHAP値の計算完了 ({calculation_time:.2f} 秒)")
    shap_values_flat = shap_values_for_instance[0]
    if len(feature_names) != len(shap_values_flat):
         logging.error(f"[エラー] 特徴量名の数 ({len(feature_names)}) と SHAP値の数 ({len(shap_values_flat)}) が一致しません。")
         feature_importance = sorted(enumerate(shap_values_flat), key=lambda x: abs(x[1]), reverse=True)
         logging.info(f"\n影響の大きい特徴量 Top 15 (インデックスとSHAP値):")
         for i, (idx, value) in enumerate(feature_importance[:15]): print(f"  {i+1}. Feature_{idx}: {value:.4f}")
    else:
        feature_importance_dict = dict(zip(feature_names, shap_values_flat))
        feature_importance_sorted = sorted(feature_importance_dict.items(), key=lambda item: abs(item[1]), reverse=True)
        logging.info(f"\n影響の大きい特徴量 Top 15 (SHAP値) - 予測牌種: {target_class_name}:")
        for i, (name, value) in enumerate(feature_importance_sorted[:15]): print(f"  {i+1}. {name}: {value:.4f}")
        
        # 分析結果を構造化して返す
        return {
            'target_class': target_class_name,
            'target_index': target_class_index,
            'feature_importance': feature_importance_sorted,
            'shap_values': {name: float(value) for name, value in feature_importance_dict.items()},
            'calculation_time': calculation_time
        }
    if shap_available:
        try:
            logging.info("SHAP Force Plot を生成・保存中...")
            expected_val_for_plot = explainer.expected_value
            if isinstance(expected_val_for_plot, np.ndarray) and expected_val_for_plot.ndim > 0: expected_val_for_plot = expected_val_for_plot[0]
            shap.force_plot(expected_val_for_plot, shap_values_flat, instance_flat[0], feature_names=feature_names, matplotlib=True, show=False) # instance_flat[0] を使用
            plot_filename = f"shap_force_plot_pred_{target_class_name.replace('*','star')}.png"
            plt.savefig(plot_filename, bbox_inches='tight'); plt.close()
            logging.info(f"SHAP Force Plot を保存しました: {plot_filename}")
        except Exception as plot_e: logging.warning(f"[警告] SHAP プロットの生成または保存に失敗しました: {plot_e}", exc_info=True)
    
    # その他の場合も構造化した形で返す
    return {
        'target_class': target_class_name,
        'target_index': target_class_index,
        'feature_importance': [],
        'shap_values': {},
        'calculation_time': 0.0
    }

# --- アテンション可視化関数 (改良版) ---
def visualize_attention_weights_fixed(attention_weights_all_layers, event_step_labels, save_dir="."):
    if not shap_available: logging.warning("Matplotlib が利用できないため、アテンションの可視化をスキップします。"); return
    if not attention_weights_all_layers: logging.warning("アテンションウェイトが取得できませんでした。可視化をスキップします。"); return
    logging.info(f"アテンションウェイトを取得しました（{len(attention_weights_all_layers)}層）")
    for layer_idx, attn_weights in enumerate(attention_weights_all_layers):
        try:
            logging.debug(f"\n[DEBUG] Layer {layer_idx+1}: Type: {type(attn_weights)}, Shape: {attn_weights.shape if hasattr(attn_weights, 'shape') else 'N/A'}, Dimensions: {attn_weights.dim() if hasattr(attn_weights, 'dim') else 'N/A'}")
            if attn_weights.dim() == 3: attn_map = attn_weights[0].cpu().numpy() # (Batch, Seq, Seq) -> (Seq, Seq)
            elif attn_weights.dim() == 4: attn_map = attn_weights[0].mean(dim=0).cpu().numpy() # (Batch, Heads, Seq, Seq) -> (Seq, Seq)
            elif attn_weights.dim() == 2: attn_map = attn_weights.cpu().numpy() # (Seq, Seq)
            else:
                if attn_weights.dim() == 1:
                    seq_len_sqrt = int(np.sqrt(attn_weights.shape[0]))
                    if seq_len_sqrt * seq_len_sqrt == attn_weights.shape[0]: logging.debug(f"  Reshaping 1D tensor to ({seq_len_sqrt}, {seq_len_sqrt})"); attn_map = attn_weights.reshape(seq_len_sqrt, seq_len_sqrt).cpu().numpy()
                    else: logging.warning(f"  Cannot reshape 1D tensor of size {attn_weights.shape[0]}"); continue
                else: logging.warning(f"  Unexpected dimensions: {attn_weights.dim()}"); continue
            logging.debug(f"  Final attn_map shape: {attn_map.shape}")
            if attn_map.ndim != 2: logging.warning(f"  Error: attn_map is not 2D after processing"); continue
            plt.figure(figsize=(max(12, attn_map.shape[1]*0.5), max(10, attn_map.shape[0]*0.5))) # サイズ調整
            im = plt.imshow(attn_map, cmap='viridis', aspect='auto', interpolation='nearest')
            plt.title(f'Self-Attention Weights (Layer {layer_idx+1})', fontsize=16)
            plt.xlabel('Key Positions (Source Events)', fontsize=14); plt.ylabel('Query Positions (Target Events)', fontsize=14)
            cbar = plt.colorbar(im, label='Attention Weight'); cbar.ax.tick_params(labelsize=10)
            if event_step_labels and len(event_step_labels) > 0:
                num_labels = len(event_step_labels)
                display_step = max(1, num_labels // 20) # 最大20ラベル程度
                display_indices = list(range(0, num_labels, display_step))
                display_labels = [event_step_labels[i] for i in display_indices]
                plt.xticks(display_indices, display_labels, rotation=90, fontsize=8)
                plt.yticks(display_indices, display_labels, fontsize=8)
            plt.tight_layout()
            filename = os.path.join(save_dir, f"attention_layer_{layer_idx+1}_visualization.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logging.info(f"  アテンション可視化を保存しました: {filename}"); plt.close()
        except Exception as e: logging.error(f"\nLayer {layer_idx+1} の可視化中にエラー: {e}", exc_info=True); continue

# --- イベントトークン生成関数 (変更なし) ---
def get_event_tokens_for_attention_visualization(game_state: GameState, seq_len: int):
    tokens = []
    history_list = list(game_state.event_history)
    event_type_names = {v: k for k, v in EVENT_TYPES.items()}
    for event_info in history_list:
        event_type_code = event_info.get("type", -1)
        event_name = event_type_names.get(event_type_code, "UNK_EV")
        player = event_info.get("player", -1)
        tile_idx = event_info.get("tile_index", -1)
        tile_str = tile_id_to_string(tile_index_to_id(tile_idx)) if tile_idx != -1 else ""
        token_str = f"{event_name[:3]}"
        if player != -1: token_str += f"_P{player}"
        if tile_str: token_str += f"_{tile_str}"
        tokens.append(token_str)
    if len(tokens) > seq_len: tokens = tokens[-seq_len:]
    elif len(tokens) < seq_len: tokens.extend([f"PAD"] * (seq_len - len(tokens)))
    return tokens

# --- 局/自風 文字列取得関数 (変更なし) ---
def get_wind_str(round_num_wind, player_id, dealer):
    round_winds = ["東", "南", "西", "北"]; player_winds = ["東", "南", "西", "北"]
    try:
        round_wind_idx = round_num_wind // NUM_PLAYERS; kyoku_num = (round_num_wind % NUM_PLAYERS) + 1
        my_wind_idx = (player_id - dealer + NUM_PLAYERS) % NUM_PLAYERS
        return f"{round_winds[round_wind_idx]}{kyoku_num}局", player_winds[my_wind_idx]
    except (IndexError, TypeError) as e: logging.warning(f"[警告] get_wind_str でエラー発生: {e}"); return "不明局", "不明家"

# --- 概念ラベリング・JSON出力機能 ---

def load_concept_models():
    """ver_2.0.0の概念ラベリングモデルをロード"""
    try:
        pca_model = joblib.load('../ver_2.0.0/pca_model_2.joblib')
        kmeans_model = joblib.load('../ver_2.0.0/kmeans_model_2.joblib')
        concept_labels = joblib.load('../ver_2.0.0/concept_labels_2.joblib')
        logging.info("概念ラベリングモデルを正常にロードしました。")
        return pca_model, kmeans_model, concept_labels
    except FileNotFoundError as e:
        logging.warning(f"概念ラベリングモデルが見つかりません: {e}")
        return None, None, None

def analyze_with_concept_labels(pca_model, kmeans_model, concept_labels, activation_vector):
    """中間表現を概念ラベルで分析"""
    if pca_model is None or kmeans_model is None or concept_labels is None:
        return None
    
    if activation_vector is None:
        return None
    
    try:
        # PCAで次元削減
        activation_pca = pca_model.transform(activation_vector.reshape(1, -1))
        
        # クラスタリング
        cluster_id = kmeans_model.predict(activation_pca)[0]
        
        # 概念ラベルを取得
        labels = concept_labels.get(cluster_id, ['Unknown'])
        
        return {
            'cluster_id': int(cluster_id),
            'concept_labels': labels,
            'pca_components': activation_pca[0].tolist()[:10]  # 最初の10成分のみ
        }
    
    except Exception as e:
        logging.warning(f"概念ラベル分析中にエラー: {e}")
        return None

def analyze_attention_weights(attention_weights, event_sequence, game_state, player_id):
    """アテンションウェイトを分析して重要な特徴量を特定"""
    if not attention_weights:
        return {}
    
    analysis_results = {}
    
    # イベントトークンを生成
    history_list = list(game_state.event_history)
    event_type_names = {v: k for k, v in EVENT_TYPES.items()}
    
    event_tokens = []
    for event_info in history_list:
        event_type_code = event_info.get("type", -1)
        event_name = event_type_names.get(event_type_code, "UNK")
        player = event_info.get("player", -1)
        tile_idx = event_info.get("tile_index", -1)
        tile_str = tile_id_to_string(tile_index_to_id(tile_idx)) if tile_idx != -1 else ""
        
        token_str = f"{event_name[:3]}"
        if player != -1:
            token_str += f"_P{player}"
        if tile_str:
            token_str += f"_{tile_str}"
        event_tokens.append(token_str)
    
    # パディング
    seq_len = event_sequence.shape[0]
    if len(event_tokens) > seq_len:
        event_tokens = event_tokens[-seq_len:]
    elif len(event_tokens) < seq_len:
        event_tokens.extend([f"PAD"] * (seq_len - len(event_tokens)))
    
    # 各層のアテンション分析
    for layer_idx, attn_weights in enumerate(attention_weights):
        try:
            if attn_weights.dim() == 3:
                attn_map = attn_weights[0].cpu().numpy()
            elif attn_weights.dim() == 4:
                attn_map = attn_weights[0].mean(dim=0).cpu().numpy()
            else:
                continue
            
            # 最後のトークンに対するアテンション（現在の決定に最も関連）
            if attn_map.shape[0] > 0:
                last_token_attention = attn_map[-1, :]
                top_indices = np.argsort(last_token_attention)[-10:][::-1]  # 上位10個
                
                layer_analysis = {
                    'layer': layer_idx + 1,
                    'top_attended_events': []
                }
                
                for idx in top_indices:
                    if idx < len(event_tokens):
                        layer_analysis['top_attended_events'].append({
                            'position': int(idx),
                            'event_token': event_tokens[idx],
                            'attention_weight': float(last_token_attention[idx])
                        })
                
                analysis_results[f'layer_{layer_idx + 1}'] = layer_analysis
        
        except Exception as e:
            logging.warning(f"Layer {layer_idx + 1} のアテンション分析中にエラー: {e}")
            continue
    
    return analysis_results

def create_comprehensive_json_output(game_state, player_id, tsumo_info, discard_info,
                                   predicted_index, predicted_prob, all_probabilities,
                                   attention_analysis, concept_analysis, shap_analysis,
                                   xml_path, round_index, tsumo_count):
    """総合的な分析結果をJSON形式で出力"""
    
    # 基本情報
    round_str, my_wind_str = get_wind_str(game_state.round_num_wind, player_id, game_state.dealer)
    predicted_tile_str = tile_id_to_string(tile_index_to_id(predicted_index))
    
    # 各プレイヤーの状態
    players_state = {}
    for p in range(NUM_PLAYERS):
        players_state[f"player_{p}"] = {
            "hand": [tile_id_to_string(t) for t in game_state.player_hands[p]],
            "discards": [{"tile": tile_id_to_string(t), "tsumogiri": ts} for t, ts in game_state.player_discards[p]],
            "melds": [
                {
                    "type": meld.get('type', '不明'),
                    "tiles": [tile_id_to_string(t) for t in meld.get('tiles', [])],
                    "from_who": meld.get('from_who', -1)
                } for meld in game_state.player_melds[p]
            ],
            "reach_status": int(game_state.player_reach_status[p]),
            "score": int(game_state.current_scores[p])
        }
    
    # 予測結果
    top_predictions = []
    top_indices = np.argsort(all_probabilities)[::-1][:10]
    for i, idx in enumerate(top_indices):
        if 0 <= idx < NUM_TILE_TYPES:
            tile_id = tile_index_to_id(idx)
            if tile_id != -1:  # 有効なtile_idのみ処理
                tile_str = tile_id_to_string(tile_id)
                if tile_str != "?":  # 有効な牌名のみ追加
                    top_predictions.append({
                        "rank": i + 1,
                        "tile": tile_str,
                        "probability": float(all_probabilities[idx])
                    })
                else:
                    print(f"[Warning] Invalid tile_str '?' for idx={idx}, tile_id={tile_id}")
            else:
                print(f"[Warning] Invalid tile_id=-1 for idx={idx}")
    
    # 総合結果
    result = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "xml_file": os.path.basename(xml_path),
            "round_index": round_index,
            "tsumo_count": tsumo_count,
            "analysis_version": "2.1.0_enhanced"
        },
        "game_situation": {
            "round_info": round_str,
            "player_wind": my_wind_str,
            "current_player": player_id,
            "tsumo_tile": tile_id_to_string(tsumo_info['pai']) if tsumo_info else None,
            "actual_discard": tile_id_to_string(discard_info['pai']) if discard_info else None,
            "dora_indicators": [tile_id_to_string(t) for t in game_state.dora_indicators],
            "remaining_tiles": int(game_state.wall_tile_count),
            "kyotaku": int(game_state.kyotaku),
            "honba": int(game_state.honba)
        },
        "players_state": players_state,
        "prediction": {
            "predicted_tile": predicted_tile_str,
            "predicted_probability": float(predicted_prob),
            "top_predictions": top_predictions
        },
        "analysis": {
            "attention_weights": attention_analysis,
            "concept_labels": concept_analysis,
            "shap_explanation": shap_analysis
        }
    }
    
    return result

# --- LLM入力用のアテンション値構造化関数 ---

def structure_attention_for_llm(attention_analysis, concept_analysis, shap_analysis, game_state, player_id):
    """アテンション分析結果をLLM用に構造化"""
    
    # 1. 概念レベル集約
    concept_weights = {}
    dominant_concept = "Unknown"
    if concept_analysis:
        # 概念の信頼度を計算（簡易版）
        concept_labels = concept_analysis.get('concept_labels', [])
        if 'Safety' in concept_labels:
            concept_weights['Safety'] = 0.8  # 実際はクラスタからの距離等で計算
        if 'Speed' in concept_labels:
            concept_weights['Speed'] = 0.2
        
        dominant_concept = max(concept_weights.keys()) if concept_weights else "Unknown"
    
    # 2. カテゴリ別特徴量整理
    feature_categories = {
        "hand_tiles": {},
        "opponent_actions": {},
        "game_context": {},
        "strategic_factors": {}
    }
    
    if shap_analysis and 'feature_importance' in shap_analysis:
        for feature_name, importance in shap_analysis['feature_importance'][:15]:
            abs_importance = abs(importance)
            
            if '手牌_' in feature_name:
                feature_categories["hand_tiles"][feature_name] = abs_importance
            elif 'DIS_' in feature_name or 'TSU_' in feature_name:
                feature_categories["opponent_actions"][feature_name] = abs_importance
            elif '巡目' in feature_name or '捨て牌数' in feature_name:
                feature_categories["game_context"][feature_name] = abs_importance
            else:
                feature_categories["strategic_factors"][feature_name] = abs_importance
    
    # 3. アテンション重要イベント
    key_attention_events = []
    if attention_analysis:
        # 最後の層から重要なイベントを抽出
        last_layer_key = f"layer_{len(attention_analysis)}" if attention_analysis else None
        if last_layer_key and last_layer_key in attention_analysis:
            events = attention_analysis[last_layer_key].get('top_attended_events', [])[:5]
            for event in events:
                key_attention_events.append({
                    "event": event.get('event_token', ''),
                    "weight": round(event.get('attention_weight', 0), 3),
                    "interpretation": interpret_event_token(event.get('event_token', ''))
                })
    
    # 4. 局面状況の要約
    situation_summary = {
        "round_info": get_wind_str(game_state.round_num_wind, player_id, game_state.dealer)[0],
        "player_position": get_wind_str(game_state.round_num_wind, player_id, game_state.dealer)[1] + "家",
        "reach_players": sum(1 for status in game_state.player_reach_status if status == 2),
        "dora_count": len(game_state.dora_indicators),
        "remaining_tiles": int(game_state.wall_tile_count)
    }
    
    return {
        "concept_level": {
            "weights": concept_weights,
            "dominant": dominant_concept,
            "cluster_id": concept_analysis.get('cluster_id', -1) if concept_analysis else -1
        },
        "feature_categories": feature_categories,
        "attention_events": key_attention_events,
        "situation": situation_summary,
        "prediction_confidence": shap_analysis.get('target_class', 'Unknown') if shap_analysis else 'Unknown'
    }

def interpret_event_token(event_token):
    """イベントトークンを人間が理解しやすい形に変換"""
    if not event_token:
        return ""
    
    interpretations = {
        "DIS_": "プレイヤーが{}を捨てた",
        "TSU_": "プレイヤーが{}をツモした", 
        "INI_": "プレイヤー{}の初期配牌",
        "PAD": "パディング"
    }
    
    for prefix, template in interpretations.items():
        if event_token.startswith(prefix):
            # 簡易的な解釈
            parts = event_token.split('_')
            if len(parts) >= 3:
                player_info = parts[1] if 'P' in parts[1] else ""
                tile_info = parts[2] if len(parts) > 2 else ""
                return template.format(f"{player_info}_{tile_info}")
            break
    
    return event_token

def create_llm_prompt_with_structured_attention(structured_data, predicted_tile, game_context):
    """構造化されたアテンションデータからLLMプロンプトを生成"""
    
    prompt = f"""あなたは麻雀の専門コーチです。以下のAI分析結果に基づいて、打牌「{predicted_tile}」の根拠を説明してください。

【AI分析結果】
局面: {structured_data['situation']['round_info']} {structured_data['situation']['player_position']}
リーチ者: {structured_data['situation']['reach_players']}人
残り牌: {structured_data['situation']['remaining_tiles']}枚

戦略判断: {structured_data['concept_level']['dominant']} (信頼度: {structured_data['concept_level']['weights']})

重要な要素:
手牌関連: {dict(list(structured_data['feature_categories']['hand_tiles'].items())[:3])}
相手の動き: {dict(list(structured_data['feature_categories']['opponent_actions'].items())[:3])}
局面状況: {dict(list(structured_data['feature_categories']['game_context'].items())[:3])}

注目したイベント:
{[f"{event['event']}(重み{event['weight']})" for event in structured_data['attention_events'][:3]]}

【説明要求】
1. 定量的要約 (50文字以内)
2. 戦術的理由 (150文字以内)  
3. 代替案との比較 (100文字以内)

各項目を明確に分けて回答してください。"""

    return prompt

# --- メイン処理 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学習済み麻雀Transformerモデルを使って打牌を予測し、SHAPで説明・アテンションを可視化します。")
    parser.add_argument("xml_file", help="天鳳形式のXML牌譜ファイルへのパス")
    parser.add_argument("round_index", type=int, help="対象局のインデックス (1から開始)")
    parser.add_argument("tsumo_count", type=int, help="対象局内でのツモ回数 (1から開始)")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help=f"学習済みモデル (.pth) ファイルへのパス (デフォルト: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--shap_samples", type=int, default=100, help="SHAP値計算に使用するサンプル数")
    parser.add_argument("--background_samples", type=int, default=50, help="SHAPの背景データとしてHDF5からサンプリングする数")
    parser.add_argument("--visualize_attention", action='store_true', help="アテンションウェイトを可視化する")
    parser.add_argument("--output_json", help="JSON出力ファイル名")
    parser.add_argument("--enable_concept_labels", action='store_true', help="概念ラベリング機能を有効にする")
    args = parser.parse_args()

    try:
        game_state, tsumo_info, discard_info = reconstruct_game_state_at_tsumo(args.xml_file, args.round_index, args.tsumo_count)
        player_id = game_state.current_player
        if tsumo_info and player_id != tsumo_info["player"]: player_id = tsumo_info["player"]
        elif not tsumo_info: logging.error("[エラー] ツモ情報なし。"); exit()

        event_sequence_instance = game_state.get_event_sequence_features()
        static_features_instance = game_state.get_static_features(player_id)
        event_dim = event_sequence_instance.shape[1]
        static_dim = static_features_instance.shape[0]
        seq_len = event_sequence_instance.shape[0]
        logging.info(f"特徴量次元: イベント次元={event_dim}, 静的次元={static_dim}, シーケンス長={seq_len}")

        model = load_trained_model_for_prediction(args.model_path, event_dim, static_dim, args.visualize_attention or args.enable_concept_labels)

        # 概念ラベリングモデルの読み込み（必要に応じて）
        pca_model, kmeans_model, concept_labels = None, None, None
        if args.enable_concept_labels:
            pca_model, kmeans_model, concept_labels = load_concept_models()

        logging.info("打牌を予測中...")
        predicted_index, predicted_prob, all_probabilities, collected_attention_weights = predict_discard(
            model, game_state, player_id, visualize_attention_flag=args.visualize_attention or args.enable_concept_labels
        )
        predicted_tile_str = tile_id_to_string(tile_index_to_id(predicted_index))
        logging.info("予測完了。")

        actual_discard_str = "N/A"
        if discard_info:
            actual_discard_str = tile_id_to_string(discard_info["pai"])
            if discard_info["tsumogiri"]: actual_discard_str += "*"

        print("\n=== Transformer 予測テスト ===")
        print(f"--- 対象局面 (牌譜: {os.path.basename(args.xml_file)}, 局: {args.round_index}, ツモ巡: {args.tsumo_count}) ---")
        round_str, my_wind_str = get_wind_str(game_state.round_num_wind, player_id, game_state.dealer)
        honba_str = f"{game_state.honba}本場"; kyotaku_str = f"({game_state.kyotaku}供託)" if game_state.kyotaku > 0 else ""
        print(f"局況: {round_str} {honba_str} {kyotaku_str} / プレイヤー: P{player_id} ({my_wind_str}家)")
        tsumo_pai_str = tile_id_to_string(tsumo_info['pai']) if tsumo_info else "不明"
        print(f"ツモ牌: {tsumo_pai_str}"); print(f"現在の巡目 (GameState準拠): {game_state.junme:.1f}")
        print(f"ドラ表示: {' '.join([tile_id_to_string(t) for t in game_state.dora_indicators])}")
        print(f"各家点数: {[f'P{i}:{s}' for i, s in enumerate(game_state.current_scores)]}")
        print("--- 現在の盤面 ---"); print("手牌 (ツモ後):")
        for p in range(NUM_PLAYERS):
            hand_str = format_hand(game_state.player_hands[p])
            reach_indicator = "*" if game_state.player_reach_status[p] == 2 else ("(宣)" if game_state.player_reach_status[p] == 1 else "")
            print(f"  P{p}{reach_indicator}: {hand_str}")
        print("捨て牌:");
        for p in range(NUM_PLAYERS): print(f"  P{p}: {format_discards(game_state.player_discards[p])}")
        print("副露:");
        for p in range(NUM_PLAYERS): print(f"  P{p}: {format_melds(game_state.player_melds[p])}")
        print("-" * 20); print(f"予測された捨て牌 (牌種): {predicted_tile_str}"); print(f"  (確率: {predicted_prob:.4f})")
        print(f"実際の捨て牌: {actual_discard_str}"); print("-" * 20)
        top_n = 5; indices_sorted = np.argsort(all_probabilities)[::-1]; print(f"予測確率 Top {top_n}:")
        for i in range(min(top_n, len(indices_sorted))): # 修正2: min を使用
            idx = indices_sorted[i]
            if 0 <= idx < NUM_TILE_TYPES:
                prob = all_probabilities[idx]; tile_id = tile_index_to_id(idx); tile_str_top = tile_id_to_string(tile_id)
                print(f"  {i+1}. {tile_str_top} ({prob:.4f})")
            else: print(f"  {i+1}. Invalid index {idx}")

        # アテンション分析
        attention_analysis = {}
        if collected_attention_weights:
            attention_analysis = analyze_attention_weights(
                collected_attention_weights, event_sequence_instance, game_state, player_id
            )

        # 概念ラベル分析
        concept_analysis = None
        if args.enable_concept_labels:
            activation_vector = activations_storage.get('combined_vector')
            concept_analysis = analyze_with_concept_labels(
                pca_model, kmeans_model, concept_labels, activation_vector
            )

        # SHAP分析
        shap_analysis = None
        if shap_available:
            try:
                instance_to_explain = (event_sequence_instance, static_features_instance, None)
                feature_names = generate_feature_names(event_dim, static_dim, seq_len)
                shap_analysis = explain_prediction_with_shap(model, DATA_HDF5_PATH, instance_to_explain, feature_names, predicted_index, args.shap_samples, args.background_samples)
            except Exception as shap_e: 
                logging.error(f"\n[エラー] SHAP説明の生成中にエラー: {shap_e}", exc_info=True)

        # JSON出力
        if args.output_json or args.enable_concept_labels:
            json_result = create_comprehensive_json_output(
                game_state, player_id, tsumo_info, discard_info,
                predicted_index, predicted_prob, all_probabilities,
                attention_analysis, concept_analysis, shap_analysis,
                args.xml_file, args.round_index, args.tsumo_count
            )
            
            output_filename = args.output_json or f"prediction_analysis_{args.round_index}_{args.tsumo_count}.json"
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(json_result, f, ensure_ascii=False, indent=2)
            
            print(f"\n詳細な分析結果をJSONファイルに保存しました: {output_filename}")
            
            # LLM用構造化データの生成と表示
            structured_data = structure_attention_for_llm(
                attention_analysis, concept_analysis, shap_analysis, game_state, player_id
            )
            
            # 構造化データをJSONで保存
            llm_data_filename = f"llm_input_{args.round_index}_{args.tsumo_count}.json"
            with open(llm_data_filename, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, ensure_ascii=False, indent=2)
            
            print(f"LLM用構造化データを保存しました: {llm_data_filename}")
            
            # LLMプロンプトの生成と保存
            llm_prompt = create_llm_prompt_with_structured_attention(
                structured_data, predicted_tile_str, game_state
            )
            
            prompt_filename = f"llm_prompt_{args.round_index}_{args.tsumo_count}.txt"
            with open(prompt_filename, 'w', encoding='utf-8') as f:
                f.write(llm_prompt)
            
            print(f"LLMプロンプトを保存しました: {prompt_filename}")
            
            # コンソールで分析結果を表示
            if concept_analysis:
                print(f"\n=== 概念ラベル分析 ===")
                print(f"クラスタID: {concept_analysis['cluster_id']}")
                print(f"概念ラベル: {', '.join(concept_analysis['concept_labels'])}")
                
            if attention_analysis:
                print(f"\n=== アテンション分析 ===")
                for layer_key, layer_data in attention_analysis.items():
                    print(f"{layer_key} (Layer {layer_data['layer']}):")
                    print("  注目度の高いイベント Top 5:")
                    for i, event_data in enumerate(layer_data['top_attended_events'][:5]):
                        print(f"    {i+1}. {event_data['event_token']} (重み: {event_data['attention_weight']:.4f})")
                        
            if shap_analysis and 'feature_importance' in shap_analysis:
                print(f"\n=== SHAP重要度分析 ===")
                print("影響の大きい特徴量 Top 10:")
                for i, (feature_name, importance) in enumerate(shap_analysis['feature_importance'][:10]):
                    print(f"  {i+1}. {feature_name}: {importance:.4f}")
                    
            # LLM用構造化データのプレビュー表示
            print(f"\n=== LLM用構造化データ ===")
            print(f"戦略判断: {structured_data['concept_level']['dominant']}")
            print(f"重要な手牌要素: {list(structured_data['feature_categories']['hand_tiles'].keys())[:3]}")
            print(f"注目イベント: {[event['event'] for event in structured_data['attention_events'][:3]]}")
            print("\n[Generated LLM Prompt Preview]")
            print(llm_prompt[:300] + "..." if len(llm_prompt) > 300 else llm_prompt)
        
        if args.visualize_attention:
            logging.info("\n--- アテンションの可視化 ---")
            if isinstance(model, MahjongTransformerV2WithAttention): # モデルがアテンション対応か確認
                 event_tokens = get_event_tokens_for_attention_visualization(game_state, seq_len)
                 visualize_attention_weights_fixed(collected_attention_weights, event_tokens) # 修正3: 関数呼び出し修正
                 print("\nアテンション可視化画像を生成しました (attention_layer_*.png)")
            else:
                 logging.warning("アテンション可視化が要求されましたが、モデルがアテンション対応版ではありません。スキップします。")
                 
        # 機能説明
        print("\n=== 利用可能な機能 ===")
        print("1. 基本予測: 打牌予測と確率分布")
        print("2. SHAP分析: 予測に影響した特徴量の重要度")
        print("3. 概念ラベル: --enable_concept_labels で中間表現の概念分析")
        print("4. アテンション: --visualize_attention で注意機構の可視化")
        print("5. JSON出力: --output_json でデータをJSON形式で保存")
        print("6. 生成ファイル:")
        print("   - SHAP Force Plot: shap_force_plot_pred_*.png")
        print("   - アテンション画像: attention_layer_*.png")
        print("   - 分析結果JSON: enhanced_analysis.json (または指定ファイル名)")
        print("\n利用例:")
        print("python predict.py test_log.xml 2 10 --enable_concept_labels --visualize_attention --output_json analysis.json")

    except FileNotFoundError as e: logging.error(f"エラー: ファイルが見つかりません - {e}")
    except ValueError as e: logging.error(f"エラー: 値が不正です - {e}")
    except ImportError as e: logging.error(f"エラー: インポートに失敗しました - {e}")
    except AttributeError as e: logging.error(f"エラー: 属性エラー（クラス定義やメソッド呼び出しを確認） - {e}")
    except Exception as e:
        logging.error(f"予期せぬエラーが発生しました: {e}", exc_info=True)