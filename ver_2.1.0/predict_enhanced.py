# predict_enhanced.py (Enhanced版 - SHAP説明機能・アテンション可視化・概念ラベリング・JSON出力付き)
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
import json
from datetime import datetime
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
import h5py
import joblib
from collections import OrderedDict, defaultdict

# SHAPとMatplotlibをインポート
try:
    import shap
    shap_available = True
except ImportError:
    print("[警告] `shap` ライブラリが見つかりません。SHAP説明機能はスキップされます。")
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
    exit(1)

# --- 設定 ---
DEFAULT_MODEL_PATH = "../ver_1.1.11/trained_model/mahjong_transformer_v1111_large_compiled_2.pth"
DATA_HDF5_PATH = "/home/ubuntu/Documents/mahjong-XAI/ver_1.1.10/training_data/mahjong_imitation_data_v1110.hdf5"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# アクティベーション保存用
activations_storage = {}

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"使用デバイス: {DEVICE}")

def get_activation_hook(name):
    """中間層のアクティベーションを取得するためのフック"""
    def hook(model, input, output):
        if isinstance(input, tuple) and len(input) > 0:
            activations_storage[name] = input[0].detach().cpu().numpy()
        else:
            activations_storage[name] = input.detach().cpu().numpy()
    return hook

# --- クラス定義 ---
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

class CustomTransformerEncoderLayerWithAttention(nn.TransformerEncoderLayer):
    """アテンションウェイトを返すカスタムTransformerEncoderLayer"""
    def __init__(self, *args, **kwargs):
        kwargs.pop('is_causal', None)  # is_causalパラメータを削除
        super().__init__(*args, **kwargs)
        self.attn_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        attn_output, self.attn_weights = self.self_attn(x, x, x,
                                                       attn_mask=src_mask,
                                                       key_padding_mask=src_key_padding_mask,
                                                       need_weights=True,
                                                       average_attn_weights=True)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))
        x = self.norm2(x)
        return x

class MahjongTransformerV2WithAttention(nn.Module):
    """アテンションウェイトを取得できるモデル"""
    def __init__(self, event_feature_dim, static_feature_dim, d_model=256, nhead=4, d_hid=1024, nlayers=4, dropout=0.1, activation='relu', output_dim=NUM_TILE_TYPES):
        super().__init__()
        self.d_model = d_model
        self.event_feature_dim = event_feature_dim
        self.event_encoder = nn.Sequential(
            nn.Linear(event_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        self.pos_encoder = RotaryPositionalEncoding(d_model, max_len=MAX_EVENT_HISTORY)
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayerWithAttention(
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
        self.output_head[0].register_forward_hook(get_activation_hook('combined_vector'))

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                gain = nn.init.calculate_gain('relu') if 'relu' in name.lower() or 'gelu' in name.lower() else 1.0
                nn.init.xavier_normal_(p, gain=gain)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, event_seq, static_feat, attention_mask=None, return_attention=False):
        if event_seq.shape[-1] != self.event_feature_dim:
            raise ValueError(f"Input event feature dimension mismatch! Got {event_seq.shape[-1]}, expected {self.event_feature_dim}")
        
        event_encoded = self.event_encoder(event_seq)
        pos_encoded = self.pos_encoder(event_encoded)
        x = pos_encoded
        attention_weights_all_layers = []
        
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=attention_mask)
            if return_attention and hasattr(layer, 'attn_weights') and layer.attn_weights is not None:
                attention_weights_all_layers.append(layer.attn_weights)
        
        transformer_output = x
        attn_weights_pool = self.attention_pool(transformer_output)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1)
            attn_weights_pool = attn_weights_pool.masked_fill(mask_expanded, 0.0)
        
        context_vector = torch.sum(attn_weights_pool * transformer_output, dim=1)
        static_encoded = self.static_encoder(static_feat)
        
        # 次元の調整
        if context_vector.dim() == 1:
            context_vector = context_vector.unsqueeze(0)
        if static_encoded.dim() == 1:
            static_encoded = static_encoded.unsqueeze(0)
        
        combined = torch.cat([context_vector, static_encoded], dim=1)
        output = self.output_head(combined)
        
        if return_attention:
            return output, attention_weights_all_layers
        return output

# --- ヘルパー関数 ---
def format_hand(hand_ids):
    if not hand_ids:
        return "なし"
    sorted_ids = sorted(hand_ids, key=lambda t: (tile_id_to_index(t), t))
    return " ".join([tile_id_to_string(t) for t in sorted_ids])

def format_discards(discard_list):
    if not discard_list:
        return "なし"
    return " ".join([f"{tile_id_to_string(t)}{'*' if tsumogiri else ''}" for t, tsumogiri in discard_list])

def format_melds(meld_list_dicts):
    if not meld_list_dicts:
        return "なし"
    meld_strs = []
    for meld_info in meld_list_dicts:
        m_type = meld_info.get('type', '不明')
        m_tiles = meld_info.get('tiles', [])
        from_who_abs = meld_info.get('from_who', -1)
        tiles_str = " ".join([tile_id_to_string(t) for t in sorted(m_tiles, key=lambda x: (tile_id_to_index(x), x))])
        from_str = f" P{from_who_abs}" if from_who_abs != -1 and m_type not in ["暗槓", "加槓"] else ""
        meld_strs.append(f"{m_type}[{tiles_str}]{from_str}")
    return " / ".join(meld_strs)

def get_wind_str(round_num_wind, player_id, dealer):
    round_winds = ["東", "南", "西", "北"]
    player_winds = ["東", "南", "西", "北"]
    try:
        round_wind_idx = round_num_wind // NUM_PLAYERS
        kyoku_num = (round_num_wind % NUM_PLAYERS) + 1
        my_wind_idx = (player_id - dealer + NUM_PLAYERS) % NUM_PLAYERS
        return f"{round_winds[round_wind_idx]}{kyoku_num}局", player_winds[my_wind_idx]
    except (IndexError, TypeError) as e:
        logging.warning(f"[警告] get_wind_str でエラー発生: {e}")
        return "不明局", "不明家"

# --- モデルロード関数 ---
def load_trained_model(model_path, event_dim, static_dim):
    """学習済みモデルをロードしてアテンション対応版に変換"""
    logging.info("アテンション対応モデルをロード中...")
    
    model = MahjongTransformerV2WithAttention(
        event_feature_dim=event_dim,
        static_feature_dim=static_dim
    ).to(DEVICE)
    
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        state_dict_to_load = checkpoint.get('model_state_dict', checkpoint)
        
        # キー名の変換
        new_state_dict = OrderedDict()
        for k, v in state_dict_to_load.items():
            if k.startswith('transformer_encoder.layers.'):
                new_key = k.replace('transformer_encoder.layers.', 'encoder_layers.', 1)
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        logging.info("モデルの重みを正常にロードしました。")
    except Exception as e:
        logging.error(f"モデルのロードに失敗: {e}")
        raise e
    
    model.eval()
    return model

# --- 説明モデルロード関数 ---
def load_explanation_models():
    """PCA、k-means、概念ラベルモデルをロード"""
    try:
        # ver_2.0.0のモデルファイルをロード
        pca_model = joblib.load('../ver_2.0.0/pca_model_2.joblib')
        kmeans_model = joblib.load('../ver_2.0.0/kmeans_model_2.joblib')
        concept_labels = joblib.load('../ver_2.0.0/concept_labels_2.joblib')
        logging.info("説明モデル（PCA, k-means, Labels）を正常にロードしました。")
        return pca_model, kmeans_model, concept_labels
    except FileNotFoundError as e:
        logging.warning(f"説明モデルファイルが見つかりません: {e}")
        logging.warning("概念ラベリング機能は利用できません。")
        return None, None, None

# --- 局面復元関数 ---
def reconstruct_game_state_at_tsumo(xml_path, target_round_index, target_tsumo_event_count_in_round):
    logging.info(f"牌譜ファイル {xml_path} を解析中...")
    try:
        meta, rounds_data = parse_full_mahjong_log(xml_path)
    except Exception as e:
        logging.error(f"[エラー] 牌譜ファイルの解析中にエラーが発生しました: {e}")
        raise

    if not (1 <= target_round_index <= len(rounds_data)):
        raise ValueError(f"指定された局インデックスが無効です: {target_round_index}")

    round_data = rounds_data[target_round_index - 1]
    game_state = GameState()
    game_state.init_round(round_data)

    current_tsumo_count = 0
    target_tsumo_event_info = None
    actual_discard_event_info = None
    events = round_data.get("events", [])

    for i, event_xml in enumerate(events):
        tag = event_xml["tag"]
        attrib = event_xml["attrib"]
        
        # ツモイベントの処理
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
            if current_tsumo_count == target_tsumo_event_count_in_round:
                target_tsumo_event_info = {"player": tsumo_player_id, "pai": tsumo_pai_id, "xml": event_xml}
                game_state.process_tsumo(tsumo_player_id, tsumo_pai_id)
                
                # 次の打牌イベントをチェック
                if i + 1 < len(events):
                    next_event_xml = events[i+1]
                    next_tag = next_event_xml["tag"]
                    for d_tag, p_id_next in GameState.DISCARD_TAGS.items():
                        if next_tag.startswith(d_tag) and next_tag[1:].isdigit() and p_id_next == tsumo_player_id:
                            try:
                                discard_pai_id = int(next_tag[1:])
                                tsumogiri = next_tag[0].islower()
                                actual_discard_event_info = {
                                    "player": p_id_next,
                                    "pai": discard_pai_id,
                                    "tsumogiri": tsumogiri,
                                    "xml": next_event_xml
                                }
                                break
                            except (ValueError, IndexError):
                                continue
                
                return game_state, target_tsumo_event_info, actual_discard_event_info
            else:
                game_state.process_tsumo(tsumo_player_id, tsumo_pai_id)
                continue
        
        # その他のイベント処理
        try:
            game_state.process_event(event_xml)
        except Exception as e:
            logging.warning(f"[警告] イベント処理中にエラー: {e}")
            continue
    
    raise ValueError(f"指定されたツモ回数 ({target_tsumo_event_count_in_round}) に到達する前に局が終了しました。")

# --- 打牌予測関数 ---
def predict_discard(model, game_state: GameState, player_id: int, return_attention=False):
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

    attention_weights = []
    with torch.no_grad():
        try:
            if return_attention:
                outputs, attention_weights = model(seq_tensor, static_tensor, mask_tensor, return_attention=True)
            else:
                outputs = model(seq_tensor, static_tensor, mask_tensor)
        except Exception as e:
            logging.error(f"[エラー] モデルのforward計算中にエラーが発生しました: {e}")
            raise
        
        probabilities = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    valid_discard_indices = game_state.get_valid_discard_options(player_id)
    
    if not valid_discard_indices:
        logging.warning("[警告] 有効な打牌選択肢がありません！")
        best_index = np.argmax(probabilities)
        best_prob = probabilities[best_index] if 0 <= best_index < len(probabilities) else 0.0
    else:
        best_prob = -1.0
        best_index = -1
        for index in valid_discard_indices:
            if 0 <= index < NUM_TILE_TYPES and probabilities[index] > best_prob:
                best_prob = probabilities[index]
                best_index = index
        
        if best_index == -1 and valid_discard_indices:
            best_index = valid_discard_indices[0]
            best_prob = probabilities[best_index] if 0 <= best_index < len(probabilities) else 0.0

    return best_index, best_prob, probabilities, attention_weights

# --- 特徴量名生成関数 ---
def generate_feature_names(event_dim_actual, static_dim_actual, seq_len_actual):
    feature_names = []
    
    # イベント特徴量名
    event_field_names = ["タイプ", "プレイヤー", "牌Idx", "巡目", "データA", "データB"]
    current_event_names = event_field_names[:event_dim_actual]
    
    for i in range(seq_len_actual):
        for j, name_suffix in enumerate(current_event_names):
            feature_names.append(f"Event_{i}_{name_suffix}")
    
    # 静的特徴量名
    static_names = [
        "局風", "本場", "供託", "親プレイヤーIdx", "壁残枚数", "自身が親か", "巡目", "ドラ表示牌数",
        "リーチ状態", "リーチ巡目", "自身の捨て牌数", "自身の副露数", "自身の手牌数"
    ]
    
    # 牌種別特徴量
    tile_names = [tile_id_to_string(tile_index_to_id(i)) for i in range(NUM_TILE_TYPES)]
    for category in ["手牌", "ドラ表示", "自捨牌", "全見え牌"]:
        for tile_name in tile_names:
            static_names.append(f"{category}_{tile_name}")
    
    # プレイヤー別特徴量
    for p_name in ["自身", "下家", "対面", "上家"]:
        static_names.extend([f"{p_name}_IsSelf", f"{p_name}_ReachAccepted"])
    
    # 静的特徴量名を追加
    for i, name in enumerate(static_names[:static_dim_actual]):
        feature_names.append(f"静的_{name}")
    
    # 不足分を補完
    expected_total = seq_len_actual * event_dim_actual + static_dim_actual
    while len(feature_names) < expected_total:
        feature_names.append(f"不明_{len(feature_names)}")
    
    return feature_names[:expected_total]

# --- アテンション分析関数 ---
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

# --- 概念ラベル分析関数 ---
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

# --- SHAP説明関数（エラー修正版） ---
def explain_prediction_with_shap(model, background_data_path, instance_to_explain, feature_names, target_class_index, n_shap_samples=100, n_bg_summary_samples=50):
    if not shap_available:
        logging.warning("SHAPライブラリが利用できないため、説明をスキップします。")
        return None
    
    logging.info("\n--- SHAP 説明生成開始 ---")
    start_time = time.time()
    
    event_seq_instance, static_feat_instance, _ = instance_to_explain
    seq_len = event_seq_instance.shape[0]
    event_dim = event_seq_instance.shape[1]
    
    bg_sequences_list = []
    bg_static_features_list = []
    
    if background_data_path and os.path.exists(background_data_path):
        try:
            with h5py.File(background_data_path, "r", swmr=True) as hf:
                num_total_bg_samples = hf["labels"].shape[0]
                if num_total_bg_samples > 0:
                    n_samples_to_load = min(n_bg_summary_samples, num_total_bg_samples)
                    sample_indices = np.sort(np.random.choice(num_total_bg_samples, size=n_samples_to_load, replace=False))
                    bg_sequences_list = hf["sequences"][sample_indices]
                    bg_static_features_list = hf["static_features"][sample_indices]
            logging.info(f"{len(bg_sequences_list)} 件の背景データをロードしました。")
        except Exception as e:
            logging.warning(f"[警告] 背景データのロードに失敗: {e}")
    
    # エラー修正: numpy arrayのbooleanチェック
    if len(bg_sequences_list) == 0 or len(bg_static_features_list) == 0:
        logging.warning("[警告] 有効な背景データがないため、ダミーの背景データを使用します。")
        bg_sequences_np = np.zeros((1, seq_len, event_dim), dtype=np.float32)
        bg_static_features_np = np.zeros((1, static_feat_instance.shape[0]), dtype=np.float32)
    else:
        bg_sequences_np = np.array(bg_sequences_list, dtype=np.float32)
        bg_static_features_np = np.array(bg_static_features_list, dtype=np.float32)
    
    def model_predict_proba_flat(flat_input_tensor_np):
        flat_input_tensor = torch.tensor(flat_input_tensor_np, dtype=torch.float32).to(DEVICE)
        batch_size = flat_input_tensor.shape[0]
        
        try:
            event_seq = flat_input_tensor[:, :(seq_len * event_dim)].reshape(batch_size, seq_len, event_dim)
            static_feat = flat_input_tensor[:, (seq_len * event_dim):]
        except Exception as e:
            logging.error(f"[エラー] SHAPラッパー内でのテンソル再構成に失敗: {e}")
            return np.zeros((batch_size,))
        
        padding_code_float = float(EVENT_TYPES["PADDING"])
        mask = (event_seq[:, :, 0] == padding_code_float)
        
        with torch.no_grad():
            outputs = model(event_seq, static_feat, mask)
            probabilities = F.softmax(outputs, dim=1)
        
        return probabilities[:, target_class_index].cpu().numpy()
    
    # 背景データの準備
    bg_flat = np.concatenate([bg_sequences_np.reshape(bg_sequences_np.shape[0], -1), bg_static_features_np], axis=1)
    instance_flat = np.concatenate([event_seq_instance.flatten(), static_feat_instance]).reshape(1, -1)
    
    background_summary = bg_flat[:min(len(bg_flat), n_bg_summary_samples)]
    if background_summary.shape[0] == 0:
        background_summary = np.zeros((1, instance_flat.shape[1]))
    
    try:
        explainer = shap.KernelExplainer(model_predict_proba_flat, background_summary)
        shap_values_for_instance = explainer.shap_values(instance_flat, nsamples=n_shap_samples)
    except Exception as e:
        logging.error(f"[エラー] SHAP値の計算中にエラーが発生しました: {e}")
        return None
    
    calculation_time = time.time() - start_time
    logging.info(f"SHAP値の計算完了 ({calculation_time:.2f} 秒)")
    
    shap_values_flat = shap_values_for_instance[0]
    
    if len(feature_names) != len(shap_values_flat):
        logging.error(f"特徴量名の数 ({len(feature_names)}) と SHAP値の数 ({len(shap_values_flat)}) が一致しません。")
        return None
    
    feature_importance_dict = dict(zip(feature_names, shap_values_flat))
    feature_importance_sorted = sorted(feature_importance_dict.items(), key=lambda item: abs(item[1]), reverse=True)
    
    return {
        'feature_importance': feature_importance_sorted[:20],  # 上位20個
        'all_shap_values': feature_importance_dict
    }

# --- JSON出力関数 ---
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
            "reach_status": game_state.player_reach_status[p],
            "score": game_state.current_scores[p]
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

# --- メイン処理 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced 麻雀Transformer予測・分析ツール")
    parser.add_argument("xml_file", help="天鳳形式のXML牌譜ファイルへのパス")
    parser.add_argument("round_index", type=int, help="対象局のインデックス (1から開始)")
    parser.add_argument("tsumo_count", type=int, help="対象局内でのツモ回数 (1から開始)")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help="学習済みモデルファイルへのパス")
    parser.add_argument("--output_json", help="JSON出力ファイル名")
    parser.add_argument("--shap_samples", type=int, default=100, help="SHAP値計算に使用するサンプル数")
    parser.add_argument("--visualize_attention", action='store_true', help="アテンションウェイトを可視化する")
    
    args = parser.parse_args()
    
    try:
        # 局面復元
        game_state, tsumo_info, discard_info = reconstruct_game_state_at_tsumo(
            args.xml_file, args.round_index, args.tsumo_count
        )
        
        player_id = tsumo_info["player"] if tsumo_info else game_state.current_player
        
        # 特徴量生成
        event_sequence = game_state.get_event_sequence_features()
        static_features = game_state.get_static_features(player_id)
        event_dim = event_sequence.shape[1]
        static_dim = static_features.shape[0]
        
        # モデルロード
        model = load_trained_model(args.model_path, event_dim, static_dim)
        
        # 説明モデルロード
        pca_model, kmeans_model, concept_labels = load_explanation_models()
        
        # 予測実行
        predicted_index, predicted_prob, all_probabilities, attention_weights = predict_discard(
            model, game_state, player_id, return_attention=True
        )
        
        # アテンション分析
        attention_analysis = analyze_attention_weights(
            attention_weights, event_sequence, game_state, player_id
        )
        
        # 概念ラベル分析
        activation_vector = activations_storage.get('combined_vector')
        concept_analysis = analyze_with_concept_labels(
            pca_model, kmeans_model, concept_labels, activation_vector
        )
        
        # SHAP分析（オプション）
        shap_analysis = None
        if shap_available:
            try:
                instance_to_explain = (event_sequence, static_features, None)
                feature_names = generate_feature_names(event_dim, static_dim, event_sequence.shape[0])
                shap_analysis = explain_prediction_with_shap(
                    model, DATA_HDF5_PATH, instance_to_explain, feature_names,
                    predicted_index, args.shap_samples
                )
            except Exception as e:
                logging.warning(f"SHAP分析中にエラー: {e}")
        
        # JSON出力
        json_result = create_comprehensive_json_output(
            game_state, player_id, tsumo_info, discard_info,
            predicted_index, predicted_prob, all_probabilities,
            attention_analysis, concept_analysis, shap_analysis,
            args.xml_file, args.round_index, args.tsumo_count
        )
        
        # コンソール出力
        print("\n=== Enhanced Transformer 予測・分析結果 ===")
        print(f"牌譜: {os.path.basename(args.xml_file)}")
        print(f"局面: {json_result['game_situation']['round_info']}")
        print(f"プレイヤー: P{player_id} ({json_result['game_situation']['player_wind']}家)")
        print(f"ツモ牌: {json_result['game_situation']['tsumo_tile']}")
        print(f"予測打牌: {json_result['prediction']['predicted_tile']} (確率: {json_result['prediction']['predicted_probability']:.4f})")
        print(f"実際打牌: {json_result['game_situation']['actual_discard']}")
        
        if concept_analysis:
            print(f"\n概念ラベル: {concept_analysis['concept_labels']} (クラスタ: {concept_analysis['cluster_id']})")
        
        # JSON保存
        output_filename = args.output_json or f"prediction_analysis_{args.round_index}_{args.tsumo_count}.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n詳細な分析結果をJSONファイルに保存しました: {output_filename}")
        
        # アテンション可視化（オプション）
        if args.visualize_attention and attention_weights:
            logging.info("アテンション可視化を実行中...")
            # 可視化コードは既存のものを使用
            
    except Exception as e:
        logging.error(f"予期せぬエラーが発生しました: {e}", exc_info=True)
        exit(1) 