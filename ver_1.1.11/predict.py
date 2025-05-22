# predict_v1110.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer  # 追加
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import math
import glob
import time
import h5py # For HDF5 background data
from tqdm import tqdm
import logging
from datetime import datetime

# Configure logging
LOG_DIR = "script_logs" # ログファイル保存用ディレクトリ
os.makedirs(LOG_DIR, exist_ok=True)
log_file_name = f'prediction_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
log_file_path = os.path.join(LOG_DIR, log_file_name)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"ログは {os.path.abspath(log_file_path)} に出力されます。")

# SHAP and LIME (optional)
try:
    import shap
    shap_available = True
except ImportError:
    logger.warning("`shap` ライブラリが見つかりません。SHAP説明機能は利用できません。`pip install shap` でインストールしてください。")
    shap_available = False

try:
    import lime
    import lime.lime_tabular
    lime_available = True
except ImportError:
    logger.warning("`lime` ライブラリが見つかりません。LIME説明機能は利用できません。`pip install lime` でインストールしてください。")
    lime_available = False

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# --- Project Modules & Constants ---
# Assuming these files are in the same directory or Python path
try:
    from game_state import GameState, NUM_TILE_TYPES, MAX_EVENT_HISTORY, STATIC_FEATURE_DIM, EVENT_TYPES
    # calculate_shanten is used for display only, not model input for this version
    # If game_state.py doesn't have it, provide a dummy or remove the display part
    try:
        from game_state import calculate_shanten
    except ImportError:
        logger.warning("game_state.py に calculate_shanten が見つかりません。向聴数表示はスキップされます。")
        def calculate_shanten(hand_indices, melds_info): return -1, [] # Dummy

    from full_mahjong_parser import parse_full_mahjong_log
    from naki_utils import decode_naki
    from tile_utils import tile_id_to_string, tile_id_to_index, tile_index_to_id
    logger.info("プロジェクトモジュールを正常にインポートしました。")
except ImportError as e:
    logger.error(f"プロジェクトモジュールのインポートに失敗しました: {e}")
    print("スクリプトと同じディレクトリに必要な .py ファイルが存在することを確認してください。")
    exit(1)

# --- Configuration (align with train2.py and preprocess_data.py) ---
# Model architecture hyperparams (must match the trained model if not in checkpoint)
# These are from train2.py's defaults, adjust if your trained model used different values
D_MODEL = 256
NHEAD = 4
D_HID = 1024
NLAYERS = 4
DROPOUT = 0.1
ACTIVATION = 'relu' # Or 'gelu' depending on trained model

DEFAULT_MODEL_PATH = "/home/ubuntu/Documents/Mahjong-XAI/ver_1.1.11/trained_model/mahjong_transformer_v1111_large_compiled.pth"
DATA_HDF5_PATH = "./training_data/mahjong_imitation_data_v1110.hdf5" # For SHAP/LIME background data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
logger.info(f"使用デバイス: {DEVICE}")

# --- Model Class Definitions (Copied from train2.py) ---
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_EVENT_HISTORY): # Use MAX_EVENT_HISTORY
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

class MahjongTransformerV2(nn.Module):
    def __init__(self, event_feature_dim, static_feature_dim, d_model=D_MODEL, nhead=NHEAD, d_hid=D_HID, nlayers=NLAYERS, dropout=DROPOUT, activation=ACTIVATION, output_dim=NUM_TILE_TYPES, max_seq_len=MAX_EVENT_HISTORY):
        super().__init__()
        self.d_model = d_model
        self.event_encoder = nn.Sequential(
            nn.Linear(event_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        self.pos_encoder = RotaryPositionalEncoding(d_model, max_len=max_seq_len)
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout,
            activation=activation, batch_first=True, norm_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.static_encoder = nn.Sequential(
            nn.Linear(static_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
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
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, output_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                gain = nn.init.calculate_gain(ACTIVATION) if ACTIVATION in ['relu', 'leaky_relu'] else 1.0
                nn.init.xavier_normal_(p, gain=gain)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, event_seq, static_feat, attention_mask=None, output_attentions=False):
        event_encoded = self.event_encoder(event_seq)
        pos_encoded = self.pos_encoder(event_encoded)
        
        # Get attention weights from transformer encoder
        all_layer_attentions = []
        if output_attentions:
            # Register hooks to get attention weights from each layer
            def get_attention_hook(name):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        all_layer_attentions.append(output[1])  # output[1] contains attention weights
                    else:
                        all_layer_attentions.append(None)
                return hook

            # Register hooks for each encoder layer
            hooks = []
            for i, layer in enumerate(self.transformer_encoder.layers):
                hook = layer.self_attn.register_forward_hook(get_attention_hook(f'layer_{i}'))
                hooks.append(hook)

        transformer_output = self.transformer_encoder(pos_encoded, src_key_padding_mask=attention_mask)
        
        attn_weights_pool = self.attention_pool(transformer_output)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1)
            attn_weights_pool = attn_weights_pool.masked_fill(mask_expanded, 0.0)
        context_vector = torch.sum(attn_weights_pool * transformer_output, dim=1)
        
        static_encoded = self.static_encoder(static_feat)
        combined = torch.cat([context_vector, static_encoded], dim=1)
        logits = self.output_head(combined)

        if output_attentions:
            # Remove hooks
            for hook in hooks:
                hook.remove()
            # Return logits and attention weights
            return logits, (attn_weights_pool.squeeze(-1), all_layer_attentions)
        return logits

# Store attention weights using hooks
attention_outputs_for_plot = {}

def get_attention_hook(layer_name):
    def hook(module, input, output):
        # output[1] is the attention weights tensor for MultiheadAttention
        # Shape: (batch_size, num_heads, seq_len, seq_len) for self-attention
        # or (batch_size, num_heads, tgt_len, src_len) for encoder-decoder attention
        # For TransformerEncoderLayer, it's self-attention.
        attention_outputs_for_plot[layer_name] = output[1].detach().cpu()
    return hook

# --- Helper Functions (Formatting, etc.) ---
def format_hand(hand_ids):
    if not hand_ids: return "なし"
    sorted_ids = sorted(hand_ids, key=lambda t: (tile_id_to_index(t), t))
    return " ".join([tile_id_to_string(t) for t in sorted_ids])

def format_discards(discard_list):
    if not discard_list: return "なし"
    return " ".join([f"{tile_id_to_string(t)}{'*' if tsumogiri else ''}" for t, tsumogiri in discard_list])

def format_melds(meld_list_dicts): # GameState's meld is a list of dicts
    if not meld_list_dicts: return "なし"
    meld_strs = []
    for meld_info in meld_list_dicts:
        m_type = meld_info.get('type', '不明')
        m_tiles = meld_info.get('tiles', [])
        tiles_str = " ".join([tile_id_to_string(t) for t in sorted(m_tiles, key=lambda x: (tile_id_to_index(x),x))])
        
        from_who_abs = meld_info.get('from_who', -1)
        called_tile = meld_info.get('called_tile', -1)

        from_str = f" P{from_who_abs}" if from_who_abs != -1 and m_type not in ["暗槓", "加槓"] else ""
        trigger_str = f"({tile_id_to_string(called_tile)})" if called_tile != -1 and m_type != "暗槓" else ""
        meld_strs.append(f"{m_type}{trigger_str}[{tiles_str}]{from_str}")
    return " / ".join(meld_strs)


def get_wind_str(round_num_wind, player_id, dealer):
    round_winds = ["東", "南", "西", "北"]
    player_winds = ["東", "南", "西", "北"]
    try:
        round_wind_idx = round_num_wind // 4 # game_state.py uses 0-3 for E1-E4, 4-7 for S1-S4 etc.
        kyoku_num = (round_num_wind % 4) + 1
        my_wind_idx = (player_id - dealer + 4) % 4
        return f"{round_winds[round_wind_idx]}{kyoku_num}局", player_winds[my_wind_idx]
    except Exception: # Catch all for safety
        return "不明局", "不明家"

# --- Model Loading ---
def load_trained_model(model_path, device):
    logger.info(f"モデルファイルをロードしています: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

    try:
        # PyTorch 2.0+ の推奨設定を使用
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        # チェックポイントの構造を確認
        if isinstance(checkpoint, dict):
            # 新しい形式（configを含む）
            if 'config' in checkpoint:
                config = checkpoint['config']
                event_dim = checkpoint['event_dim']
                static_dim = checkpoint['static_dim']
                seq_len = checkpoint['seq_len']
            else:
                # 古い形式（configを含まない）
                logger.info("チェックポイントにconfigキーがありません。デフォルト設定を使用します。")
                config = {
                    'd_model': D_MODEL,
                    'nhead': NHEAD,
                    'd_hid': D_HID,
                    'nlayers': NLAYERS,
                    'dropout': DROPOUT,
                    'activation': ACTIVATION
                }
                # 必要な次元情報を取得
                event_dim = checkpoint.get('event_dim', 6)  # デフォルト値
                static_dim = checkpoint.get('static_dim', STATIC_FEATURE_DIM)
                seq_len = checkpoint.get('seq_len', MAX_EVENT_HISTORY)
        else:
            # チェックポイントが辞書でない場合（古い形式）
            logger.info("チェックポイントが辞書形式ではありません。デフォルト設定を使用します。")
            config = {
                'd_model': D_MODEL,
                'nhead': NHEAD,
                'd_hid': D_HID,
                'nlayers': NLAYERS,
                'dropout': DROPOUT,
                'activation': ACTIVATION
            }
            event_dim = 6  # デフォルト値
            static_dim = STATIC_FEATURE_DIM
            seq_len = MAX_EVENT_HISTORY

        # モデルを初期化
        model = MahjongTransformerV2(
            event_feature_dim=event_dim,
            static_feature_dim=static_dim,
            d_model=config.get('d_model', D_MODEL),
            nhead=config.get('nhead', NHEAD),
            d_hid=config.get('d_hid', D_HID),
            nlayers=config.get('nlayers', NLAYERS),
            dropout=config.get('dropout', DROPOUT),
            activation=config.get('activation', ACTIVATION),
            max_seq_len=seq_len
        )

        # モデルの重みをロード
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model = model.to(device)
        model.eval()
        
        logger.info("モデルのロードが完了しました")
        return model, config

    except Exception as e:
        logger.error(f"モデルのロード中にエラーが発生しました: {str(e)}")
        raise

def load_background_data(data_path, batch_size=1000):
    """Load background data for SHAP/LIME explanations"""
    logger.info(f"背景データをロードしています: {data_path}")
    try:
        with h5py.File(data_path, 'r') as f:
            # Load a subset of data for background
            event_data = torch.tensor(f['event_data'][:batch_size], dtype=torch.float32)
            static_data = torch.tensor(f['static_data'][:batch_size], dtype=torch.float32)
            return event_data, static_data
    except Exception as e:
        logger.error(f"背景データのロード中にエラーが発生しました: {str(e)}")
        raise

# --- Game State Reconstruction ---
def reconstruct_game_state_at_tsumo(xml_path, target_round_index, target_tsumo_event_count_in_round):
    logger.info(f"牌譜ファイル {xml_path} を解析中...")
    meta, rounds_data = parse_full_mahjong_log(xml_path)
    if not (1 <= target_round_index <= len(rounds_data)):
        raise ValueError(f"指定された局インデックスが無効です: {target_round_index} (利用可能な範囲: 1-{len(rounds_data)})")

    round_data = rounds_data[target_round_index - 1]
    game_state = GameState()
    logger.info(f"第{target_round_index}局の初期状態を構築中...")
    game_state.init_round(round_data)

    player_to_watch_for_tsumo = game_state.current_player  # Player whose Nth tsumo we are looking for (e.g., dealer)

    player_tsumo_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    target_tsumo_event_info = None
    actual_discard_event_info = None
    events = round_data.get("events", [])
    logger.info(f"イベントを再生し、プレイヤー {player_to_watch_for_tsumo} の {target_tsumo_event_count_in_round} 回目のツモを探します...")
    logger.info(f"総イベント数: {len(events)}")

    for i, event_xml in enumerate(events):
        tag, attrib = event_xml["tag"], event_xml["attrib"]
        
        is_target_tsumo_moment = False

        # --- Tsumo Event Processing ---
        is_tsumo_event_flag = False
        actual_tsumo_player_idx = -1
        tsumo_pai_id = -1

        for t_char_map, p_idx_map in GameState.TSUMO_TAGS.items():
            if tag.startswith(t_char_map) and tag[1:].isdigit():
                actual_tsumo_player_idx = p_idx_map
                tsumo_pai_id = int(tag[1:])
                is_tsumo_event_flag = True
                break
        
        if is_tsumo_event_flag:
            player_tsumo_counts[actual_tsumo_player_idx] += 1
            # Process the tsumo in GameState to keep it updated
            game_state.process_tsumo(actual_tsumo_player_idx, tsumo_pai_id)

            # Check if this is the target tsumo event we are looking for
            if actual_tsumo_player_idx == player_to_watch_for_tsumo and \
               player_tsumo_counts[actual_tsumo_player_idx] == target_tsumo_event_count_in_round:
                target_tsumo_event_info = {"player": actual_tsumo_player_idx, "pai": tsumo_pai_id, "xml": event_xml}
                is_target_tsumo_moment = True
                logger.info(f"目標のツモイベントを検出: プレイヤー {actual_tsumo_player_idx}, 牌 {tsumo_pai_id} ({tile_id_to_string(tsumo_pai_id)})")
        
        elif tag.startswith(tuple(GameState.DISCARD_TAGS.keys())) and tag[1:].isdigit():
            discard_player_idx = -1
            for d_char_map, p_idx_map in GameState.DISCARD_TAGS.items():
                if tag.startswith(d_char_map):
                    discard_player_idx = p_idx_map
                    break
            if discard_player_idx != -1:
                pai_id = int(tag[1:])
                tsumogiri = tag[0].islower()
                game_state.process_discard(discard_player_idx, pai_id, tsumogiri)

        elif tag == "N":
            naki_player_id = int(attrib.get("who", -1))
            meld_code = int(attrib.get("m", "0"))
            if naki_player_id != -1:
                game_state.process_naki(naki_player_id, meld_code)
        elif tag == "REACH":
            reach_player_id = int(attrib.get("who", -1))
            step = int(attrib.get("step", 0))
            if reach_player_id != -1:
                game_state.process_reach(reach_player_id, step)
        elif tag == "DORA":
            hai = int(attrib.get("hai", -1))
            if hai != -1:
                game_state.process_dora(hai)
        elif tag == "AGARI":
            game_state.process_agari(attrib)
            logger.debug(f"和了イベント処理: {attrib}")
            break 
        elif tag == "RYUUKYOKU":
            game_state.process_ryuukyoku(attrib)
            logger.debug(f"流局イベント処理: {attrib}")
            break

        if is_target_tsumo_moment:
            if i + 1 < len(events):
                next_event_xml = events[i+1]
                next_tag = next_event_xml["tag"]
                
                for d_char_next, p_idx_next in GameState.DISCARD_TAGS.items():
                    if next_tag.startswith(d_char_next) and next_tag[1:].isdigit() and p_idx_next == target_tsumo_event_info["player"]:
                        actual_discard_pai_id = int(next_tag[1:])
                        actual_tsumogiri = next_tag[0].islower()
                        actual_discard_event_info = {
                            "player": p_idx_next,
                            "pai": actual_discard_pai_id,
                            "tsumogiri": actual_tsumogiri,
                            "xml": next_event_xml
                        }
                        logger.info(f"実際の捨て牌を検出: プレイヤー {p_idx_next}, 牌 {actual_discard_pai_id} ({tile_id_to_string(actual_discard_pai_id)}), ツモ切り: {actual_tsumogiri}")
                        break
            logger.info("指定局面の状態復元が完了しました。")
            return game_state, target_tsumo_event_info, actual_discard_event_info

    error_msg = (
        f"プレイヤー {player_to_watch_for_tsumo} の {target_tsumo_event_count_in_round} 回目のツモに到達できませんでした。\\n"
        f"プレイヤーごとのツモ回数:\\n"
        + "\\n".join([f"  プレイヤー {p}: {c}回" for p, c in player_tsumo_counts.items()])
    )
    logger.error(error_msg)
    raise ValueError(error_msg)

# --- Prediction ---
def predict_discard(model, event_seq_tensor, static_feat_tensor, attention_mask_tensor):
    try:
        with torch.no_grad():
            # Input tensors are already prepared and on the correct device
            # Get model prediction with attention weights
            outputs = model(event_seq_tensor, static_feat_tensor, attention_mask=attention_mask_tensor, output_attentions=True)
            
            if isinstance(outputs, tuple):
                logits, attention_weights = outputs
            else:
                logits = outputs
                attention_weights = None

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Get top 5 predictions
            top5_probs, top5_indices = torch.topk(probs[0], 5)
            
            predictions = []
            for prob, idx in zip(top5_probs.cpu().numpy(), top5_indices.cpu().numpy()):
                tile_id = tile_index_to_id(idx) # Ensure tile_index_to_id is defined
                predictions.append({
                    'tile_id': tile_id,
                    'tile_str': tile_id_to_string(tile_id), # Ensure tile_id_to_string is defined
                    'probability': float(prob)
                })
            return predictions, attention_weights

    except Exception as e:
        logger.error(f"予測中にエラーが発生しました: {str(e)}")
        raise

# --- Attention Visualization ---
def plot_attention_map(attention_weights, input_tokens=None, output_tokens=None, layer_name="", head_idx=0, save_path="attention_map.png"):
    if attention_weights is None:
        logger.warning(f"{layer_name} のアテンション重みが None のため、プロットをスキップします。")
        return

    # For self-attention from TransformerEncoderLayer, weights are (Batch, NumHeads, SeqLen, SeqLen)
    # For pooling attention, weights are (Batch, SeqLen)
    
    if attention_weights.ndim == 4: # Encoder self-attention
        # Squeeze batch if 1, select a head
        attn = attention_weights.squeeze(0)[head_idx].numpy()
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(attn, cmap='viridis')
        fig.colorbar(cax)
        
        if input_tokens:
            ax.set_xticks(range(len(input_tokens)))
            ax.set_xticklabels(input_tokens, rotation=90)
            ax.set_yticks(range(len(input_tokens))) # Self-attention, so input_tokens for y-axis too
            ax.set_yticklabels(input_tokens)
        
        ax.set_xlabel("Key (Memory)")
        ax.set_ylabel("Query")
        ax.set_title(f"Attention Map: {layer_name}, Head {head_idx}")

    elif attention_weights.ndim == 2: # Pooling attention (Batch, SeqLen)
        attn = attention_weights.squeeze(0).numpy() # (SeqLen,)
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.bar(range(len(attn)), attn)

        if input_tokens:
            ax.set_xticks(range(len(input_tokens)))
            ax.set_xticklabels(input_tokens, rotation=45, ha="right")
        
        ax.set_ylabel("Attention Weight")
        ax.set_title(f"Attention Weights: {layer_name}")
    else:
        logger.warning(f"未対応のアテンション重み形状: {attention_weights.shape}")
        return

    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"アテンションプロットを保存しました: {save_path}")
    plt.close(fig)

# --- Feature Name Generation (Crucial for SHAP/LIME) ---
def generate_feature_names(event_dim, static_dim, seq_len):
    feature_names = []
    
    # Event Sequence Features
    # From game_state.py: type, player, tile_index, junme, data1, data2
    event_data_base_names = ["EventType", "EventPlayer", "EventTileIdx", "EventJunme"]
    event_data_specific_names = ["EventData1", "EventData2"]
    
    # Ensure event_dim matches expected structure (4 base + 2 specific = 6)
    if event_dim != (len(event_data_base_names) + len(event_data_specific_names)):
        logger.warning(f"generate_feature_names: Event dim ({event_dim}) が想定 (6) と異なります。汎用名を使用します。")
        current_event_names = [f"EventField{j}" for j in range(event_dim)]
    else:
        current_event_names = event_data_base_names + event_data_specific_names

    for i in range(seq_len):
        for j_idx, name_suffix in enumerate(current_event_names):
            feature_names.append(f"Event_{i}_{name_suffix}")

    # Static Features (STATIC_FEATURE_DIM = 157)
    # Order from game_state.py's get_static_features
    # DIM = {"CONTEXT": 8, "PLAYER": 5, "HAND": 34, "DORA_IND": 34, "DISCARDS": 34, "VISIBLE": 34, "POS_REACH": 8}

    # 1. Game Context (8)
    ctx_names = ["RoundWind", "Honba", "Kyotaku", "DealerPlayerIdx", "WallTiles", "IsDealer", "Junme", "NumDoraInd"]
    for name in ctx_names: feature_names.append(f"Ctx_{name}")
    
    # 2. Player Specific (for the target player) (5)
    ps_names = ["ReachStatus", "ReachJunme", "NumOwnDiscards", "NumMelds", "NumHandTiles"]
    for name in ps_names: feature_names.append(f"Player_{name}")

    tile_kind_names = [tile_id_to_string(tile_index_to_id(i)) for i in range(NUM_TILE_TYPES)]

    # 3. Hand Counts (34)
    for i in range(NUM_TILE_TYPES): feature_names.append(f"Hand_{tile_kind_names[i]}")
    
    # 4. Dora Indicators Counts (34)
    for i in range(NUM_TILE_TYPES): feature_names.append(f"DoraInd_{tile_kind_names[i]}")

    # 5. Player Discards Counts (for the target player) (34)
    for i in range(NUM_TILE_TYPES): feature_names.append(f"OwnDiscards_{tile_kind_names[i]}")

    # 6. All Visible Tiles (discards + melds from all players) (34)
    for i in range(NUM_TILE_TYPES): feature_names.append(f"Visible_{tile_kind_names[i]}")

    # 7. Player Positions Relative + Reach Status (8 features)
    # (player_id + p_offset) % NUM_PLAYERS; features[idx] = float(p_abs == player_id); features[idx+1] = float(self.player_reach_status[p_abs] == 2)
    rel_player_labels = ["Self", "Shimocha", "Toimen", "Kamicha"] # Relative to target player
    for i in range(4): # For self, shimocha, toimen, kamicha
        feature_names.append(f"RelPlayer_{rel_player_labels[i]}_IsSelfFlag") # This will be 1 only for Self
        feature_names.append(f"RelPlayer_{rel_player_labels[i]}_ReachAcceptedFlag")
        
    if len(feature_names) != seq_len * event_dim + static_dim:
        logger.warning(f"生成された特徴量名の数 ({len(feature_names)}) が期待値 ({seq_len * event_dim + static_dim}) と異なります。")
        logger.warning(f"  内訳: Event Feats={seq_len*event_dim}, Static Feats (Generated)={len(feature_names)-seq_len*event_dim}, Static Feats (Expected)={static_dim}")
        # Fallback for safety, though this indicates a mismatch in understanding the feature vector structure.
        num_expected_total = seq_len * event_dim + static_dim
        if len(feature_names) < num_expected_total:
            feature_names.extend([f"UnknownFeature_{k}" for k in range(num_expected_total - len(feature_names))])
        else:
            feature_names = feature_names[:num_expected_total]
    
    logger.info(f"特徴量名生成完了 (合計: {len(feature_names)}個)")
    return feature_names

# --- SHAP Explanation ---
def explain_with_shap(model, background_data_tensors, instance_to_explain_tensors, feature_names, target_class_index, n_shap_samples=100, output_dir="shap_plots"):
    if not shap_available:
        logger.warning("SHAP ライブラリが利用できないため、SHAP 説明をスキップします。")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info("\n--- SHAP 説明生成開始 ---")
    start_time = time.time()

    bg_event_seq_tensor, bg_static_feat_tensor = background_data_tensors
    instance_event_seq_tensor, instance_static_feat_tensor, instance_mask_tensor = instance_to_explain_tensors

    # SHAP Explainer用の予測関数ラッパー
    def model_predict_proba_for_shap(flat_input_numpy):
        # SHAPはNumPy配列で入力してくる
        flat_input_tensor = torch.tensor(flat_input_numpy, dtype=torch.float32).to(DEVICE)
        batch_size = flat_input_tensor.shape[0]
        
        # 入力テンソルをイベントシーケンスと静的特徴量に分割
        # instance_event_seq_tensorから形状を取得
        _seq_len = instance_event_seq_tensor.shape[1] 
        _event_dim = instance_event_seq_tensor.shape[2]

        event_seq = flat_input_tensor[:, :(_seq_len * _event_dim)].reshape(batch_size, _seq_len, _event_dim)
        static_feat = flat_input_tensor[:, (_seq_len * _event_dim):]
        
        padding_code = float(EVENT_TYPES["PADDING"])
        mask = (event_seq[:, :, 0] == padding_code)

        with torch.no_grad():
            outputs = model(event_seq, static_feat, mask)
            probabilities = F.softmax(outputs, dim=1)
        
        return probabilities[:, target_class_index].cpu().numpy() # 対象クラスの確率を返す

    # 背景データをフラット化
    bg_flat = np.concatenate([
        bg_event_seq_tensor.cpu().numpy().reshape(bg_event_seq_tensor.shape[0], -1),
        bg_static_feat_tensor.cpu().numpy()
    ], axis=1)

    # 説明対象インスタンスをフラット化
    instance_flat = np.concatenate([
        instance_event_seq_tensor.cpu().numpy().flatten(),
        instance_static_feat_tensor.cpu().numpy()
    ]).reshape(1, -1)

    # 背景データのサマリーを作成 (KernelExplainerの計算量削減のため)
    n_bg_summary = min(50, bg_flat.shape[0]) 
    background_summary = shap.sample(bg_flat, n_bg_summary)
    
    logger.info(f"SHAP KernelExplainer を初期化 (背景サマリー: {n_bg_summary}サンプル)...")
    explainer = shap.KernelExplainer(model_predict_proba_for_shap, background_summary)

    logger.info(f"SHAP値を計算中 (nsamples={n_shap_samples})...")
    shap_values = explainer.shap_values(instance_flat, nsamples=n_shap_samples)
    
    calculation_time = time.time() - start_time
    logger.info(f"SHAP値の計算完了 ({calculation_time:.2f} 秒)")

    shap_values_flat = shap_values[0] # (1, num_features) -> (num_features)
    
    # 特徴量の重要度を出力
    if len(feature_names) != len(shap_values_flat):
        logger.warning(f"特徴量名の数 ({len(feature_names)}) と SHAP値の数 ({len(shap_values_flat)}) が一致しません。")
        feature_importance_sorted = sorted(enumerate(shap_values_flat), key=lambda x: abs(x[1]), reverse=True)
        for i, (idx, value) in enumerate(feature_importance_sorted[:20]):
            logger.info(f"  {i+1}. Feature_{idx}: {value:.4f}")
    else:
        feature_importance_dict = dict(zip(feature_names, shap_values_flat))
        feature_importance_sorted = sorted(feature_importance_dict.items(), key=lambda item: abs(item[1]), reverse=True)
        logger.info(f"\n予測牌 ({tile_id_to_string(tile_index_to_id(target_class_index))}) に対する影響の大きい特徴量 Top 20 (SHAP値):")
        for i, (name, value) in enumerate(feature_importance_sorted[:20]):
            logger.info(f"  {i+1}. {name}: {value:.4f}")

        # SHAP Summary Plot (Bar)
        try:
            shap.summary_plot(shap_values, instance_flat, feature_names=feature_names, plot_type="bar", show=False)
            plt.title(f"SHAP Feature Importance for predicting {tile_id_to_string(tile_index_to_id(target_class_index))}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"shap_summary_bar_pred_{target_class_index}.png"))
            plt.close()
            logger.info(f"SHAP Barプロットを保存しました: {output_dir}/shap_summary_bar_pred_{target_class_index}.png")
            
            # SHAP Force Plot
            force_plot_fig = shap.force_plot(explainer.expected_value, shap_values[0], instance_flat[0], feature_names=feature_names, matplotlib=True, show=False)
            # force_plot_fig.set_size_inches(20, 4) # Adjust size if needed
            plt.title(f"SHAP Force Plot for predicting {tile_id_to_string(tile_index_to_id(target_class_index))}")
            plt.savefig(os.path.join(output_dir, f"shap_force_plot_pred_{target_class_index}.png"), bbox_inches='tight')
            plt.close(force_plot_fig)
            logger.info(f"SHAP Forceプロットを保存しました: {output_dir}/shap_force_plot_pred_{target_class_index}.png")
        except Exception as plot_e:
            logger.warning(f"[警告] SHAP プロットの生成または保存に失敗しました: {plot_e}")
            
    return feature_importance_sorted if 'feature_importance_sorted' in locals() else None


# --- LIME Explanation ---
def explain_with_lime(model, background_data_tensors, instance_to_explain_tensors, feature_names, target_class_index, num_lime_features=10, num_lime_samples=1000, output_dir="lime_plots"):
    if not lime_available:
        logger.warning("LIME ライブラリが利用できないため、LIME 説明をスキップします。")
        return None

    os.makedirs(output_dir, exist_ok=True)
    logger.info("\n--- LIME 説明生成開始 ---")
    start_time = time.time()

    bg_event_seq_tensor, bg_static_feat_tensor = background_data_tensors
    instance_event_seq_tensor, instance_static_feat_tensor, instance_mask_tensor = instance_to_explain_tensors

    # LIME用の予測関数ラッパー
    def model_predict_proba_for_lime(flat_input_numpy):
        flat_input_tensor = torch.tensor(flat_input_numpy, dtype=torch.float32).to(DEVICE)
        batch_size = flat_input_tensor.shape[0]
        
        _seq_len = instance_event_seq_tensor.shape[1]
        _event_dim = instance_event_seq_tensor.shape[2]

        event_seq = flat_input_tensor[:, :(_seq_len * _event_dim)].reshape(batch_size, _seq_len, _event_dim)
        static_feat = flat_input_tensor[:, (_seq_len * _event_dim):]
        
        padding_code = float(EVENT_TYPES["PADDING"])
        mask = (event_seq[:, :, 0] == padding_code)

        with torch.no_grad():
            outputs = model(event_seq, static_feat, mask)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
        return probabilities # LIMEは全クラスの確率を期待

    # 背景データ（トレーニングデータ）をフラット化
    training_data_flat = np.concatenate([
        bg_event_seq_tensor.cpu().numpy().reshape(bg_event_seq_tensor.shape[0], -1),
        bg_static_feat_tensor.cpu().numpy()
    ], axis=1)

    # 説明対象インスタンスをフラット化
    instance_flat = np.concatenate([
        instance_event_seq_tensor.cpu().numpy().flatten(),
        instance_static_feat_tensor.cpu().numpy()
    ])

    # LIME TabularExplainer 初期化
    # mode='classification' / 'regression'
    # discretize_continuous=True Can help with stability
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=training_data_flat,
        feature_names=feature_names,
        class_names=[tile_id_to_string(tile_index_to_id(i)) for i in range(NUM_TILE_TYPES)],
        mode='classification',
        discretize_continuous=True 
    )

    logger.info(f"LIME 説明を計算中 (num_features={num_lime_features}, num_samples={num_lime_samples})...")
    explanation = explainer.explain_instance(
        data_row=instance_flat,
        predict_fn=model_predict_proba_for_lime,
        num_features=num_lime_features,
        top_labels=1, # Only explain the target class (or predicted class)
        num_samples=num_lime_samples # Number of perturbed samples LIME generates
    )
    
    calculation_time = time.time() - start_time
    logger.info(f"LIME 説明の計算完了 ({calculation_time:.2f} 秒)")

    # 結果表示
    predicted_class_name = tile_id_to_string(tile_index_to_id(target_class_index))
    logger.info(f"\n予測牌 ({predicted_class_name}) に対するLIME説明 (Top {num_lime_features}):")
    for feat_name, weight in explanation.as_list(label=target_class_index):
        logger.info(f"  {feat_name}: {weight:.4f}")

    # LIMEプロット保存
    try:
        fig = explanation.as_pyplot_figure(label=target_class_index)
        plt.title(f"LIME Explanation for predicting {predicted_class_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"lime_explanation_pred_{target_class_index}.png"))
        plt.close(fig)
        logger.info(f"LIMEプロットを保存しました: {output_dir}/lime_explanation_pred_{target_class_index}.png")
    except Exception as plot_e:
        logger.warning(f"[警告] LIME プロットの生成または保存に失敗しました: {plot_e}")
        
    return explanation

# --- Visualization Utilities ---
class ActivationHook:
    def __init__(self, name):
        self.name = name
        self.activations = None

    def __call__(self, module, input, output):
        self.activations = output.detach()

class AttentionHook:
    def __init__(self, name):
        self.name = name
        self.attention_weights = None

    def __call__(self, module, input, output):
        # MultiheadAttention の出力は (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) > 1:
            self.attention_weights = output[1].detach()

def register_visualization_hooks(model):
    """モデルの各層にAttentionとActivationのフックを登録"""
    hooks = []
    activation_outputs = {}
    attention_outputs = {}

    # Activation Hooks
    activation_layers = [
        ('event_encoder', model.event_encoder),
        ('static_encoder', model.static_encoder),
        ('output_head', model.output_head)
    ]
    for name, layer in activation_layers:
        hook = ActivationHook(name)
        hooks.append(layer.register_forward_hook(hook))
        activation_outputs[name] = hook

    # Attention Hooks
    # Transformer Encoder Layers
    for i, layer in enumerate(model.transformer_encoder.layers):
        hook = AttentionHook(f'encoder_layer_{i}')
        hooks.append(layer.self_attn.register_forward_hook(hook))
        attention_outputs[f'encoder_layer_{i}'] = hook

    # Attention Pooling
    hook = ActivationHook('attention_pool')
    hooks.append(model.attention_pool.register_forward_hook(hook))
    activation_outputs['attention_pool'] = hook

    return hooks, activation_outputs, attention_outputs

def plot_attention_weights(attention_outputs, seq_len, output_dir):
    """Attention weightsをヒートマップとして可視化"""
    os.makedirs(output_dir, exist_ok=True)
    
    for layer_name, hook in attention_outputs.items():
        if hook.attention_weights is None:
            continue
            
        weights = hook.attention_weights.cpu().numpy()
        if weights.ndim == 4:  # (batch_size, num_heads, seq_len, seq_len)
            weights = weights.squeeze(0)  # Remove batch dimension
            num_heads = weights.shape[0]
            
            for head in range(num_heads):
                plt.figure(figsize=(10, 8))
                plt.imshow(weights[head], cmap='viridis', aspect='auto')
                plt.colorbar()
                plt.title(f'{layer_name} - Head {head}')
                plt.xlabel('Key position')
                plt.ylabel('Query position')
                plt.savefig(os.path.join(output_dir, f'attention_{layer_name}_head_{head}.png'))
                plt.close()

def plot_activations(activation_outputs, output_dir):
    """Activation valuesをヒートマップまたは線グラフとして可視化"""
    os.makedirs(output_dir, exist_ok=True)
    
    for layer_name, hook in activation_outputs.items():
        if hook.activations is None:
            continue
            
        activations = hook.activations.cpu().numpy()
        if activations.ndim > 2:
            activations = activations.squeeze(0)  # Remove batch dimension
            
        plt.figure(figsize=(12, 6))
        if activations.ndim == 2:  # Sequence data
            plt.imshow(activations.T, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title(f'{layer_name} Activations')
            plt.xlabel('Sequence position')
            plt.ylabel('Feature dimension')
        else:  # 1D data
            plt.plot(activations)
            plt.title(f'{layer_name} Activations')
            plt.xlabel('Feature dimension')
            plt.ylabel('Activation value')
        
        plt.savefig(os.path.join(output_dir, f'activation_{layer_name}.png'))
        plt.close()

# --- Main Script ---
def main():
    parser = argparse.ArgumentParser(description="学習済み麻雀Transformerモデルを使って打牌を予測し、解釈します。")
    parser.add_argument("xml_file", help="天鳳形式のXML牌譜ファイルへのパス")
    parser.add_argument("round_index", type=int, help="対象局のインデックス (1から開始)")
    parser.add_argument("tsumo_count", type=int, help="対象局内でのツモ回数 (1から開始)")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help=f"学習済みモデル (.pth) ファイルへのパス (デフォルト: {DEFAULT_MODEL_PATH})")
    
    parser.add_argument("--explain", action="store_true", help="SHAPおよびLIME説明を生成します。")
    parser.add_argument("--shap_samples", type=int, default=100, help="SHAP値計算に使用するKernelExplainerのnsamples数")
    parser.add_argument("--lime_features", type=int, default=10, help="LIMEで表示する特徴量の数")
    parser.add_argument("--lime_samples", type=int, default=1000, help="LIMEが生成する摂動サンプル数")
    parser.add_argument("--background_data_size", type=int, default=100, help="SHAP/LIMEの背景データとしてHDF5から読み込むサンプル数")
    
    parser.add_argument("--visualize", action="store_true", help="AttentionとActivationの可視化を行います。")
    parser.add_argument("--output_dir", default="./prediction_outputs", help="プロットや説明結果の保存ディレクトリ")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    hooks = []  # Initialize hooks list here
    activation_outputs = {}
    attention_outputs = {}

    try:
        # 1. Load Model
        model, config = load_trained_model(args.model_path, DEVICE)

        if args.visualize:
            logger.info("Attention と Activation の可視化フックを登録します...")
            hooks, activation_outputs, attention_outputs = register_visualization_hooks(model)

        # 2. Reconstruct Game State
        game_state, tsumo_info, discard_info = reconstruct_game_state_at_tsumo(
            args.xml_file, args.round_index, args.tsumo_count
        )
        player_id = game_state.current_player # This is the player for whom we are predicting
        if tsumo_info and game_state.current_player != tsumo_info["player"]: 
            # This might happen if the target tsumo event belongs to a player
            # who is not the current player to act (e.g. after naki).
            # For prediction, we usually care about the player whose turn it is.
            # However, if tsumo_info.player is the one who just drew, that's our target.
            player_id = tsumo_info["player"]
            logger.info(f"予測対象プレイヤーをツモイベントのプレイヤー {player_id} に設定しました。")

        if not tsumo_info: 
            raise ValueError("ツモ情報が見つかりません。予測を続行できません。")

        # 3. Get features for the specific instance using get_model_input
        logger.info(f"プレイヤー {player_id} のモデル入力を取得中...")
        # Ensure game_state.py has get_model_input implemented correctly
        event_seq_instance_np, static_feat_instance_np = game_state.get_model_input(player_id)

        # Validate shapes based on config (derived from checkpoint or defaults)
        # event_seq_instance_np should be (seq_len, event_dim)
        # static_feat_instance_np should be (static_dim,)
        expected_seq_len = config.get('seq_len', MAX_EVENT_HISTORY)
        expected_event_dim = config.get('event_dim', 6) # Default from original train.py if not in config
        expected_static_dim = config.get('static_dim', STATIC_FEATURE_DIM)

        if event_seq_instance_np.shape[0] != expected_seq_len or \
           event_seq_instance_np.shape[1] != expected_event_dim:
            raise ValueError(
                f"イベントシーケンスの形状が不正です。期待値: ({expected_seq_len}, {expected_event_dim}), "
                f"実際値: ({event_seq_instance_np.shape[0]}, {event_seq_instance_np.shape[1]})"
            )
        if static_feat_instance_np.shape[0] != expected_static_dim:
            raise ValueError(
                f"静的特徴量の形状が不正です。期待値: ({expected_static_dim},), "
                f"実際値: ({static_feat_instance_np.shape[0]},)"
            )

        instance_event_tensor = torch.tensor(event_seq_instance_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        instance_static_tensor = torch.tensor(static_feat_instance_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # Create attention mask for the instance
        # Assuming EVENT_TYPES["PADDING"] is defined and is the correct padding indicator value in the first feature of event_seq
        padding_code = float(EVENT_TYPES.get("PADDING", 0.0)) # Default to 0.0 if not found
        instance_mask_tensor = (instance_event_tensor[:, :, 0] == padding_code).to(DEVICE)
        
        # 4. Predict Discard using the prepared tensors
        logger.info("打牌を予測中...")
        # Modify predict_discard to accept tensors directly
        predictions, attention_weights = predict_discard(model, instance_event_tensor, instance_static_tensor, instance_mask_tensor)

        # 5. Display results
        logger.info("\n=== 予測結果 ===")
        round_str, my_wind_str = get_wind_str(game_state.round_num_wind, player_id, game_state.dealer)
        logger.info(f"局: {round_str} {game_state.honba}本場 ({game_state.kyotaku}供託) / Player {player_id} ({my_wind_str})")
        logger.info(f"ツモ牌: {tile_id_to_string(tsumo_info['pai'])}")
        logger.info(f"手牌 (ツモ後): {format_hand(game_state.player_hands[player_id])}")
        logger.info(f"予測された捨て牌: {predictions[0]['tile_str']} (確率: {predictions[0]['probability']:.4f})")
        if discard_info:
            actual_discard_str = tile_id_to_string(discard_info['pai']) + ("*" if discard_info["tsumogiri"] else "")
            logger.info(f"実際の捨て牌: {actual_discard_str}")

        # 6. Visualize attention and activations if requested
        if args.visualize:
            logger.info("\nAttention と Activation を可視化中...")
            # Ensure config has seq_len for plot_attention_weights
            seq_len_for_plot = config.get('seq_len', MAX_EVENT_HISTORY)
            
            # Plot attention weights from model output
            if attention_weights is not None:
                pooling_attention, layer_attentions = attention_weights
                # Pooling attentionのみ可視化
                if pooling_attention is not None:
                    plot_attention_map(pooling_attention.cpu().numpy(), output_dir, 'pooling_attention.png')
                # Transformer層のattentionは未対応
                if layer_attentions is not None:
                    logger.warning("Transformer層のattention可視化はPyTorch標準では未対応のためスキップします。")
            
            # Plot attention weights from hooks
            plot_attention_weights(attention_outputs, seq_len_for_plot, args.output_dir)
            plot_activations(activation_outputs, args.output_dir)
            logger.info(f"可視化結果を {args.output_dir} に保存しました。")

        # 7. Generate explanations if requested
        if args.explain:
            logger.info("\nSHAP/LIME 説明を生成中...")
            logger.info("背景データをロード中 (SHAP/LIME用)...")
            if not os.path.exists(DATA_HDF5_PATH):
                logger.error(f"[エラー] 背景データファイルが見つかりません: {DATA_HDF5_PATH}")
            else:
                # Load background data
                bg_event_tensor, bg_static_tensor = load_background_data(DATA_HDF5_PATH, args.background_data_size)
                background_data_tensors = (bg_event_tensor.to(DEVICE), bg_static_tensor.to(DEVICE))
                logger.info(f"{bg_event_tensor.shape[0]} 件の背景データをロードしました。")

                # instance_to_explain_tensors should be (event_seq, static_feat, mask)
                instance_to_explain_tensors = (instance_event_tensor, instance_static_tensor, instance_mask_tensor)
                
                # Generate feature names based on actual dimensions from loaded model/data
                # Using event_dim from config for generate_feature_names
                feature_names = generate_feature_names(
                    event_dim=config.get('event_dim', 6),
                    static_dim=config.get('static_dim', STATIC_FEATURE_DIM),
                    seq_len=config.get('seq_len', MAX_EVENT_HISTORY)
                )

                # Use the tile_id from the top prediction for explanations
                predicted_tile_idx = tile_id_to_index(predictions[0]['tile_id'])

                if shap_available:
                    explain_with_shap(model, background_data_tensors, instance_to_explain_tensors, 
                                      feature_names, predicted_tile_idx, 
                                      n_shap_samples=args.shap_samples, output_dir=args.output_dir)
                if lime_available:
                    explain_with_lime(model, background_data_tensors, instance_to_explain_tensors, 
                                      feature_names, predicted_tile_idx, 
                                      num_lime_features=args.lime_features, 
                                      num_lime_samples=args.lime_samples, output_dir=args.output_dir)
    
    except FileNotFoundError as e: logger.error(f"エラー: ファイルが見つかりません - {e}")
    except ValueError as e: logger.error(f"エラー: 値が不正です - {e}")
    except RuntimeError as e: logger.error(f"エラー: ランタイムエラー (モデル構造やテンソル形状を確認) - {e}")
    except AttributeError as e: # Catch AttributeError specifically for get_model_input
        logger.error(f"エラー: Attributeエラー - {e}")
        if "get_model_input" in str(e):
            logger.error("GameStateオブジェクトに get_model_input メソッドが存在しないようです。game_state.pyを確認してください。")
    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        for hook in hooks:
            hook.remove()
        if hooks:
            logger.info(f"{len(hooks)}個の可視化フックを解除しました。")

if __name__ == "__main__":
    main()