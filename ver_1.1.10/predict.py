# predict_v1110.py
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# SHAP and LIME (optional)
try:
    import shap
    shap_available = True
except ImportError:
    print("[警告] `shap` ライブラリが見つかりません。SHAP説明機能は利用できません。`pip install shap` でインストールしてください。")
    shap_available = False

try:
    import lime
    import lime.lime_tabular
    lime_available = True
except ImportError:
    print("[警告] `lime` ライブラリが見つかりません。LIME説明機能は利用できません。`pip install lime` でインストールしてください。")
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
        print("[警告] game_state.py に calculate_shanten が見つかりません。向聴数表示はスキップされます。")
        def calculate_shanten(hand_indices, melds_info): return -1, [] # Dummy

    from full_mahjong_parser import parse_full_mahjong_log
    from naki_utils import decode_naki
    from tile_utils import tile_id_to_string, tile_id_to_index, tile_index_to_id
    print("プロジェクトモジュールを正常にインポートしました。")
except ImportError as e:
    print(f"[致命的エラー] プロジェクトモジュール (game_state, full_mahjong_parser, etc.) のインポートに失敗しました: {e}")
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

DEFAULT_MODEL_PATH = "./trained_model/mahjong_transformer_v1110_large_compiled.pth"
DATA_HDF5_PATH = "./training_data/mahjong_imitation_data_v1110.hdf5" # For SHAP/LIME background data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"使用デバイス: {DEVICE}")

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
        
        # To get attention weights, we need to hook into the MultiheadAttention modules
        # or modify TransformerEncoder to return them. For simplicity with hooks:
        all_layer_attentions = []
        if output_attentions:
            # This is a simplified way; actual hooks would be needed for internal MHA
            # For now, we'll focus on SHAP/LIME and placeholder for direct attention viz
            # PyTorch's TransformerEncoder doesn't directly return all layer attentions
            # in a straightforward way without custom layers or hooks.
            # We can get the output of the encoder.
            pass

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
            # Placeholder: In a real scenario, `all_layer_attentions` would be populated by hooks
            # For this example, we'll return the pooling attention and None for transformer layer attentions
            return logits, (attn_weights_pool.squeeze(-1), None) 
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
    print(f"モデルファイルをロードしようとしています: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    
    # Checkpoint structure from train2.py
    event_dim = checkpoint.get('event_dim')
    static_dim = checkpoint.get('static_dim')
    seq_len = checkpoint.get('seq_len') # MAX_EVENT_HISTORY

    if event_dim is None or static_dim is None or seq_len is None:
        raise ValueError("Checkpoint is missing 'event_dim', 'static_dim', or 'seq_len'. "
                         "Ensure the model was saved with these parameters by train2.py.")

    # Use architecture hyperparams defined globally or load if in checkpoint (ideal)
    # For now, assumes global D_MODEL, NHEAD etc. are correct for the loaded model
    model = MahjongTransformerV2(
        event_feature_dim=event_dim,
        static_feature_dim=static_dim,
        d_model=D_MODEL, nhead=NHEAD, d_hid=D_HID, nlayers=NLAYERS,
        dropout=DROPOUT, activation=ACTIVATION, output_dim=NUM_TILE_TYPES,
        max_seq_len=seq_len
    ).to(device)

    # Check if it's a compiled model's state_dict or raw model
    if 'model_state_dict' in checkpoint:
        state_dict_to_load = checkpoint['model_state_dict']
    else: # Older format, assuming checkpoint itself is the state_dict
        state_dict_to_load = checkpoint 
    
    # Handle potential "Unexpected key(s) in state_dict: '_orig_mod.xxx'" if loading compiled model state
    # into a non-compiled model, or vice-versa.
    # If _orig_mod keys are present, it means the saved state is from a torch.compile'd model.
    # If the current model is not compiled, strip '_orig_mod.'
    # This logic might need adjustment based on how torch.compile actually saves.
    # Typically, you save model._orig_mod.state_dict() IF it was compiled.
    
    # If the state_dict comes from a compiled model (keys start with _orig_mod)
    # and we are loading into a non-compiled model, we need to strip the prefix.
    # However, train2.py saves model._orig_mod.state_dict() if compiled, so prefix is already stripped.
    
    try:
        model.load_state_dict(state_dict_to_load)
    except RuntimeError as e:
        print(f"モデルのstate_dictロード中にエラー: {e}")
        print("ヒント: モデルアーキテクチャ (D_MODEL, NHEADなど) や保存形式が一致しているか確認してください。")
        # Further debug: print keys
        print("Model state_dict keys (first 5):", list(model.state_dict().keys())[:5])
        print("Checkpoint state_dict keys (first 5):", list(state_dict_to_load.keys())[:5])
        raise
        
    model.eval()
    print(f"モデルのロードに成功しました: {model_path}")
    print(f"  ロードされたモデルの次元: EventFeat={event_dim}, StaticFeat={static_dim}, SeqLen={seq_len}")
    return model, event_dim, static_dim, seq_len

# --- Game State Reconstruction ---
def reconstruct_game_state_at_tsumo(xml_path, target_round_index, target_tsumo_event_count_in_round):
    print(f"牌譜ファイル {xml_path} を解析中...")
    meta, rounds_data = parse_full_mahjong_log(xml_path)
    if not (1 <= target_round_index <= len(rounds_data)):
        raise ValueError(f"指定された局インデックスが無効です: {target_round_index} (利用可能な範囲: 1-{len(rounds_data)})")

    round_data = rounds_data[target_round_index - 1]
    game_state = GameState() # Uses game_state.py's GameState
    print(f"第{target_round_index}局の初期状態を構築中...")
    game_state.init_round(round_data)

    current_tsumo_count = 0
    target_tsumo_event_info = None
    actual_discard_event_info = None
    events = round_data.get("events", [])
    print(f"イベントを再生し、{target_tsumo_event_count_in_round} 回目のツモを探します...")

    for i, event_xml in enumerate(events):
        tag, attrib = event_xml["tag"], event_xml["attrib"]
        
        is_target_tsumo_moment = False

        # Process events using GameState methods
        # Tsumo
        for t_char, p_idx in GameState.TSUMO_TAGS.items():
            if tag.startswith(t_char) and tag[1:].isdigit():
                pai_id = int(tag[1:])
                current_tsumo_count += 1
                if p_idx == game_state.current_player: # Ensure tsumo is for the current player
                    if current_tsumo_count == target_tsumo_event_count_in_round:
                        target_tsumo_event_info = {"player": p_idx, "pai": pai_id, "xml": event_xml}
                        game_state.process_tsumo(p_idx, pai_id)
                        is_target_tsumo_moment = True # Found the target tsumo
                    else:
                        game_state.process_tsumo(p_idx, pai_id)
                else: # Tsumo for another player, just process
                     game_state.process_tsumo(p_idx, pai_id)
                break
        if is_target_tsumo_moment:
             # Find next discard by this player for "actual_discard_event_info"
            if i + 1 < len(events):
                next_event_xml = events[i+1]; next_tag = next_event_xml["tag"]
                for d_char_next, p_idx_next in GameState.DISCARD_TAGS.items():
                    if next_tag.startswith(d_char_next) and next_tag[1:].isdigit() and p_idx_next == target_tsumo_event_info["player"]:
                        actual_discard_pai_id = int(next_tag[1:])
                        actual_tsumogiri = next_tag[0].islower()
                        actual_discard_event_info = {"player": p_idx_next, "pai": actual_discard_pai_id, "tsumogiri": actual_tsumogiri, "xml": next_event_xml}
                        break
            print("指定局面の状態復元が完了しました。")
            return game_state, target_tsumo_event_info, actual_discard_event_info

        # Discard
        for d_char, p_idx in GameState.DISCARD_TAGS.items():
            if tag.startswith(d_char) and tag[1:].isdigit():
                pai_id = int(tag[1:]); tsumogiri = tag[0].islower()
                game_state.process_discard(p_idx, pai_id, tsumogiri)
                break
        
        if tag == "N":
            naki_player_id = int(attrib.get("who", -1)); meld_code = int(attrib.get("m", "0"))
            if naki_player_id != -1: game_state.process_naki(naki_player_id, meld_code)
        elif tag == "REACH":
            reach_player_id = int(attrib.get("who", -1)); step = int(attrib.get("step", 0))
            if reach_player_id != -1: game_state.process_reach(reach_player_id, step)
        elif tag == "DORA":
            hai = int(attrib.get("hai", -1))
            if hai != -1: game_state.process_dora(hai)
        elif tag == "AGARI":
            game_state.process_agari(attrib); break 
        elif tag == "RYUUKYOKU":
            game_state.process_ryuukyoku(attrib); break

    raise ValueError(f"指定されたツモ回数 ({target_tsumo_event_count_in_round}) に到達できませんでした。")

# --- Prediction ---
def predict_discard(model, game_state: GameState, player_id: int, output_attentions=False):
    event_sequence = game_state.get_event_sequence_features()
    static_features = game_state.get_static_features(player_id)

    seq_tensor = torch.tensor(event_sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    static_tensor = torch.tensor(static_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    padding_code = float(EVENT_TYPES["PADDING"])
    mask_tensor = (seq_tensor[:, :, 0] == padding_code).to(DEVICE)

    attentions = None
    with torch.no_grad():
        if output_attentions:
            outputs, attentions = model(seq_tensor, static_tensor, mask_tensor, output_attentions=True)
        else:
            outputs = model(seq_tensor, static_tensor, mask_tensor, output_attentions=False)
            
        probabilities = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    valid_discard_indices = game_state.get_valid_discard_options(player_id)
    best_prob = -1.0; best_index = -1

    if not valid_discard_indices:
        print("[警告] 有効な打牌選択肢がありません！")
        best_index = np.argmax(probabilities)
        best_prob = probabilities[best_index]
    else:
        for index in valid_discard_indices:
            if 0 <= index < NUM_TILE_TYPES:
                if probabilities[index] > best_prob:
                    best_prob = probabilities[index]; best_index = index
        if best_index == -1 and valid_discard_indices: # Fallback
            best_index = valid_discard_indices[0]
            best_prob = probabilities[best_index]

    return best_index, best_prob, probabilities, attentions

# --- Attention Visualization ---
def plot_attention_map(attention_weights, input_tokens=None, output_tokens=None, layer_name="", head_idx=0, save_path="attention_map.png"):
    if attention_weights is None:
        print(f"[警告] {layer_name} のアテンション重みが None のため、プロットをスキップします。")
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
        print(f"[警告] 未対応のアテンション重み形状: {attention_weights.shape}")
        return

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"アテンションプロットを保存しました: {save_path}")
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
        print(f"[警告] generate_feature_names: Event dim ({event_dim}) が想定 (6) と異なります。汎用名を使用します。")
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
        print(f"[エラー] 生成された特徴量名の数 ({len(feature_names)}) が期待値 ({seq_len * event_dim + static_dim}) と異なります。")
        print(f"  内訳: Event Feats={seq_len*event_dim}, Static Feats (Generated)={len(feature_names)-seq_len*event_dim}, Static Feats (Expected)={static_dim}")
        # Fallback for safety, though this indicates a mismatch in understanding the feature vector structure.
        num_expected_total = seq_len * event_dim + static_dim
        if len(feature_names) < num_expected_total:
            feature_names.extend([f"UnknownFeature_{k}" for k in range(num_expected_total - len(feature_names))])
        else:
            feature_names = feature_names[:num_expected_total]
    
    print(f"特徴量名生成完了 (合計: {len(feature_names)}個)")
    return feature_names

# --- SHAP Explanation ---
def explain_with_shap(model, background_data_tensors, instance_to_explain_tensors, feature_names, target_class_index, n_shap_samples=100, output_dir="shap_plots"):
    if not shap_available:
        print("SHAP ライブラリが利用できないため、SHAP 説明をスキップします。")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    print("\n--- SHAP 説明生成開始 ---")
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
    
    print(f"SHAP KernelExplainer を初期化 (背景サマリー: {n_bg_summary}サンプル)...")
    explainer = shap.KernelExplainer(model_predict_proba_for_shap, background_summary)

    print(f"SHAP値を計算中 (nsamples={n_shap_samples})...")
    shap_values = explainer.shap_values(instance_flat, nsamples=n_shap_samples)
    
    calculation_time = time.time() - start_time
    print(f"SHAP値の計算完了 ({calculation_time:.2f} 秒)")

    shap_values_flat = shap_values[0] # (1, num_features) -> (num_features)
    
    # 特徴量の重要度を出力
    if len(feature_names) != len(shap_values_flat):
        print(f"[エラー] 特徴量名の数 ({len(feature_names)}) と SHAP値の数 ({len(shap_values_flat)}) が一致しません。")
        feature_importance_sorted = sorted(enumerate(shap_values_flat), key=lambda x: abs(x[1]), reverse=True)
        for i, (idx, value) in enumerate(feature_importance_sorted[:20]):
            print(f"  {i+1}. Feature_{idx}: {value:.4f}")
    else:
        feature_importance_dict = dict(zip(feature_names, shap_values_flat))
        feature_importance_sorted = sorted(feature_importance_dict.items(), key=lambda item: abs(item[1]), reverse=True)
        print(f"\n予測牌 ({tile_id_to_string(tile_index_to_id(target_class_index))}) に対する影響の大きい特徴量 Top 20 (SHAP値):")
        for i, (name, value) in enumerate(feature_importance_sorted[:20]):
            print(f"  {i+1}. {name}: {value:.4f}")

        # SHAP Summary Plot (Bar)
        try:
            shap.summary_plot(shap_values, instance_flat, feature_names=feature_names, plot_type="bar", show=False)
            plt.title(f"SHAP Feature Importance for predicting {tile_id_to_string(tile_index_to_id(target_class_index))}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"shap_summary_bar_pred_{target_class_index}.png"))
            plt.close()
            print(f"SHAP Barプロットを保存しました: {output_dir}/shap_summary_bar_pred_{target_class_index}.png")
            
            # SHAP Force Plot
            force_plot_fig = shap.force_plot(explainer.expected_value, shap_values[0], instance_flat[0], feature_names=feature_names, matplotlib=True, show=False)
            # force_plot_fig.set_size_inches(20, 4) # Adjust size if needed
            plt.title(f"SHAP Force Plot for predicting {tile_id_to_string(tile_index_to_id(target_class_index))}")
            plt.savefig(os.path.join(output_dir, f"shap_force_plot_pred_{target_class_index}.png"), bbox_inches='tight')
            plt.close(force_plot_fig)
            print(f"SHAP Forceプロットを保存しました: {output_dir}/shap_force_plot_pred_{target_class_index}.png")
        except Exception as plot_e:
            print(f"[警告] SHAP プロットの生成または保存に失敗しました: {plot_e}")
            
    return feature_importance_sorted if 'feature_importance_sorted' in locals() else None


# --- LIME Explanation ---
def explain_with_lime(model, background_data_tensors, instance_to_explain_tensors, feature_names, target_class_index, num_lime_features=10, num_lime_samples=1000, output_dir="lime_plots"):
    if not lime_available:
        print("LIME ライブラリが利用できないため、LIME 説明をスキップします。")
        return None

    os.makedirs(output_dir, exist_ok=True)
    print("\n--- LIME 説明生成開始 ---")
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

    print(f"LIME 説明を計算中 (num_features={num_lime_features}, num_samples={num_lime_samples})...")
    explanation = explainer.explain_instance(
        data_row=instance_flat,
        predict_fn=model_predict_proba_for_lime,
        num_features=num_lime_features,
        top_labels=1, # Only explain the target class (or predicted class)
        num_samples=num_lime_samples # Number of perturbed samples LIME generates
    )
    
    calculation_time = time.time() - start_time
    print(f"LIME 説明の計算完了 ({calculation_time:.2f} 秒)")

    # 結果表示
    predicted_class_name = tile_id_to_string(tile_index_to_id(target_class_index))
    print(f"\n予測牌 ({predicted_class_name}) に対するLIME説明 (Top {num_lime_features}):")
    for feat_name, weight in explanation.as_list(label=target_class_index):
        print(f"  {feat_name}: {weight:.4f}")

    # LIMEプロット保存
    try:
        fig = explanation.as_pyplot_figure(label=target_class_index)
        plt.title(f"LIME Explanation for predicting {predicted_class_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"lime_explanation_pred_{target_class_index}.png"))
        plt.close(fig)
        print(f"LIMEプロットを保存しました: {output_dir}/lime_explanation_pred_{target_class_index}.png")
    except Exception as plot_e:
        print(f"[警告] LIME プロットの生成または保存に失敗しました: {plot_e}")
        
    return explanation

# --- Main Script ---
if __name__ == "__main__":
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
    
    parser.add_argument("--plot_attention", action="store_true", help="アテンションマップをプロットします。")
    parser.add_argument("--output_dir", default="./prediction_outputs", help="プロットや説明結果の保存ディレクトリ")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Attach hooks for attention visualization if requested
    attention_hooks = []
    if args.plot_attention:
        # This part needs to know the model structure to attach hooks correctly.
        # For MahjongTransformerV2, attention is in transformer_encoder.layers[i].self_attn
        # And also in self.attention_pool (though this is simpler, just weights)
        print("[情報] アテンション可視化が要求されました。フックの設定を試みます。")
        # Model needs to be loaded first to attach hooks.
        # So, hook attachment will happen after model load.
        pass


    try:
        # 1. Load Model (also gets feature dimensions from checkpoint)
        model, E_DIM, S_DIM, SEQ_LEN = load_trained_model(args.model_path, DEVICE)

        if args.plot_attention:
            # Register hooks on the loaded model's MHA layers
            # Example for the first TransformerEncoderLayer's self-attention
            # This is highly dependent on the model's internal naming and structure
            try:
                # For nn.TransformerEncoder, layers are in model.transformer_encoder.layers
                # Each layer is nn.TransformerEncoderLayer
                # Inside nn.TransformerEncoderLayer, the self-attention is model.transformer_encoder.layers[idx].self_attn
                for i, layer in enumerate(model.transformer_encoder.layers):
                    hook = layer.self_attn.register_forward_hook(get_attention_hook(f"EncoderLayer_{i}_SelfAttn"))
                    attention_hooks.append(hook)
                print(f"{len(attention_hooks)}個のTransformer Encoder Layerのフックを登録しました。")
            except AttributeError as e:
                print(f"[警告] モデル構造が異なり、アテンションフックの登録に失敗しました: {e}")
                args.plot_attention = False # Disable plotting if hooks can't be set

        # 2. Reconstruct Game State
        game_state, tsumo_info, discard_info = reconstruct_game_state_at_tsumo(
            args.xml_file, args.round_index, args.tsumo_count
        )
        player_id = game_state.current_player
        if tsumo_info and player_id != tsumo_info["player"]: player_id = tsumo_info["player"] # Align player_id
        if not tsumo_info: raise ValueError("ツモ情報が見つかりません。")

        # 3. Get features for the specific instance
        event_seq_instance = game_state.get_event_sequence_features()
        static_feat_instance = game_state.get_static_features(player_id)
        
        # Ensure instance dimensions match model's expected input dimensions
        if event_seq_instance.shape[1] != E_DIM or \
           static_feat_instance.shape[0] != S_DIM or \
           event_seq_instance.shape[0] != SEQ_LEN:
            raise ValueError(f"特徴量次元の不一致: インスタンス(E:{event_seq_instance.shape[1]}, S:{static_feat_instance.shape[0]}, SL:{event_seq_instance.shape[0]}) "
                             f"vs モデル期待値(E:{E_DIM}, S:{S_DIM}, SL:{SEQ_LEN})")

        instance_event_tensor = torch.tensor(event_seq_instance, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        instance_static_tensor = torch.tensor(static_feat_instance, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        padding_code = float(EVENT_TYPES["PADDING"])
        instance_mask_tensor = (instance_event_tensor[:, :, 0] == padding_code).to(DEVICE)
        
        # 4. Predict Discard
        print("打牌を予測中...")
        attention_outputs_for_plot.clear() # Clear previous hook data
        predicted_idx, predicted_prob, all_probs, attentions_tuple = predict_discard(
            model, game_state, player_id, output_attentions=args.plot_attention
        )
        predicted_tile_str = tile_id_to_string(tile_index_to_id(predicted_idx))
        actual_discard_str = "N/A"
        if discard_info:
            actual_discard_str = tile_id_to_string(discard_info["pai"]) + ("*" if discard_info["tsumogiri"] else "")

        # 5. Display Prediction Results
        print("\n=== 予測結果 ===")
        # ... (similar display as before) ...
        round_str, my_wind_str = get_wind_str(game_state.round_num_wind, player_id, game_state.dealer)
        print(f"局: {round_str} {game_state.honba}本場 ({game_state.kyotaku}供託) / Player {player_id} ({my_wind_str})")
        print(f"ツモ牌: {tile_id_to_string(tsumo_info['pai'])}")
        print(f"手牌 (ツモ後): {format_hand(game_state.player_hands[player_id])}")
        # ... (other game state info) ...
        print(f"予測された捨て牌: {predicted_tile_str} (確率: {predicted_prob:.4f})")
        print(f"実際の捨て牌: {actual_discard_str}")
        top_n = 5
        indices_sorted = np.argsort(all_probs)[::-1]
        print(f"予測確率 Top {top_n}:")
        for i in range(top_n):
            idx_ = indices_sorted[i]
            prob_ = all_probs[idx_]
            tile_str_ = tile_id_to_string(tile_index_to_id(idx_)) # Convert index to tile ID then string
            print(f"  {i+1}. {tile_str_} ({idx_}): {prob_:.4f}")

        # 6. Plot Attention (if requested)
        if args.plot_attention and attentions_tuple:
            pooling_attn_weights, _ = attentions_tuple # Second element is None for now
            # Pooling attention
            plot_attention_map(pooling_attn_weights, 
                               input_tokens=[f"E{k}" for k in range(SEQ_LEN)], # Generic event tokens
                               layer_name="Output_AttentionPooling", 
                               save_path=os.path.join(args.output_dir, "attention_pooling.png"))
            
            # Transformer layer attentions (from hooks)
            for layer_name, attn_weights in attention_outputs_for_plot.items():
                 # attn_weights shape: (batch_size, num_heads, seq_len, seq_len)
                 if attn_weights is not None and attn_weights.ndim == 4:
                     num_heads_in_layer = attn_weights.shape[1]
                     for head_idx_to_plot in range(min(num_heads_in_layer, 2)): # Plot first 2 heads
                         plot_attention_map(attn_weights, 
                                            input_tokens=[f"E{k}" for k in range(SEQ_LEN)], 
                                            layer_name=layer_name, 
                                            head_idx=head_idx_to_plot,
                                            save_path=os.path.join(args.output_dir, f"attention_{layer_name}_head{head_idx_to_plot}.png"))
                 else:
                     print(f"[警告] {layer_name} のアテンションデータが不正な形状、または取得できませんでした。")


        # 7. Generate Explanations (if requested)
        if args.explain:
            print("\n背景データをロード中 (SHAP/LIME用)...")
            if not os.path.exists(DATA_HDF5_PATH):
                print(f"[エラー] 背景データファイルが見つかりません: {DATA_HDF5_PATH}")
            else:
                with h5py.File(DATA_HDF5_PATH, "r", swmr=True) as hf:
                    num_total_bg_samples = hf["labels"].shape[0]
                    bg_sample_indices = np.random.choice(num_total_bg_samples, 
                                                         size=min(args.background_data_size, num_total_bg_samples), 
                                                         replace=False)
                    bg_sequences_np = hf["sequences"][sorted(bg_sample_indices)]
                    bg_static_np = hf["static_features"][sorted(bg_sample_indices)]
                
                bg_event_tensor = torch.tensor(bg_sequences_np, dtype=torch.float32).to(DEVICE)
                bg_static_tensor = torch.tensor(bg_static_np, dtype=torch.float32).to(DEVICE)
                background_data_tensors = (bg_event_tensor, bg_static_tensor)
                print(f"{len(bg_sequences_np)} 件の背景データをロードしました。")

                instance_to_explain_tensors = (instance_event_tensor, instance_static_tensor, instance_mask_tensor)
                
                # Generate feature names based on actual dimensions from loaded model/data
                feature_names = generate_feature_names(E_DIM, S_DIM, SEQ_LEN)

                if shap_available:
                    explain_with_shap(model, background_data_tensors, instance_to_explain_tensors, 
                                      feature_names, predicted_idx, 
                                      n_shap_samples=args.shap_samples, output_dir=args.output_dir)
                if lime_available:
                    explain_with_lime(model, background_data_tensors, instance_to_explain_tensors, 
                                      feature_names, predicted_idx, 
                                      num_lime_features=args.lime_features, 
                                      num_lime_samples=args.lime_samples, output_dir=args.output_dir)
    
    except FileNotFoundError as e: print(f"エラー: ファイルが見つかりません - {e}")
    except ValueError as e: print(f"エラー: 値が不正です - {e}")
    except RuntimeError as e: print(f"エラー: ランタイムエラー (モデル構造やテンソル形状を確認) - {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Remove hooks
        for hook in attention_hooks:
            hook.remove()
        if attention_hooks:
             print(f"{len(attention_hooks)}個のアテンションフックを解除しました。")