# predict.py (Transformer版 - SHAP説明機能付き・アテンション/活性化可視化機能付き)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import math # math をインポート
import glob # 背景データ読み込み用
import time # SHAP計算時間計測用
import logging
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import h5py # HDF5ファイル読み込み用に追加

# SHAPとMatplotlibをインポート (なければインストール: pip install shap matplotlib)
try:
    import shap
    import matplotlib.pyplot as plt
    shap_available = True
except ImportError:
    print("[警告] `shap` または `matplotlib` ライブラリが見つかりません。SHAP説明機能および一部の可視化機能はスキップされます。")
    print("インストールするには: pip install shap matplotlib")
    shap_available = False

from full_mahjong_parser import parse_full_mahjong_log
# 修正されたGameStateと関連クラス・定数をインポート
try:
    from game_state import GameState, NUM_TILE_TYPES, MAX_EVENT_HISTORY, EVENT_TYPES, STATIC_FEATURE_DIM
except ImportError as e:
    print(f"[エラー] game_state.pyからのインポートに失敗しました: {e}")
    print("game_state.py が同じディレクトリにあるか、必要な定義が含まれているか確認してください。")
    from game_state import GameState, NUM_TILE_TYPES # 最低限
    MAX_EVENT_HISTORY = 60
    STATIC_FEATURE_DIM = 157
    EVENT_TYPES = {"PADDING": 8} # 必須
    print("[警告] game_state から MAX_EVENT_HISTORY, EVENT_TYPES, STATIC_FEATURE_DIM をインポートできませんでした。デフォルト値/ダミーを使用します。")


from naki_utils import decode_naki
from tile_utils import tile_id_to_string, tile_id_to_index, tile_index_to_id

# --- クラス定義 (train2.py から MahjongTransformerV2 と RotaryPositionalEncoding をコピー) ---

class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Encoding (RoPE)の実装"""
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

class MahjongTransformerV2(nn.Module):
    """イベント系列と静的特徴を入力とするTransformerモデル"""
    def __init__(self, event_feature_dim, static_feature_dim, d_model=256, nhead=4, d_hid=1024, nlayers=4, dropout=0.1, activation='relu', output_dim=NUM_TILE_TYPES):
        super().__init__()
        self.d_model = d_model
        self.event_encoder = nn.Sequential(
            nn.Linear(event_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        self.pos_encoder = RotaryPositionalEncoding(d_model)
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout,
            activation=activation, batch_first=True, norm_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
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

# --- 設定 ---
NUM_PLAYERS = 4
DEFAULT_MODEL_PATH = "./trained_model/mahjong_transformer_v1111_large_compiled_2.pth"
DEFAULT_BACKGROUND_DATA_PATH = "./training_data/mahjong_imitation_data_v1110.hdf5"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

print(f"使用デバイス: {DEVICE}")

# --- グローバル変数 (フック用) ---
hook_outputs = {}
attention_weights_storage = []
original_mha_forward_methods = {} # 元のforwardメソッドを保存する辞書


# --- ヘルパー関数 ---
def format_hand(hand_ids):
    if not hand_ids: return "なし"
    sorted_ids = sorted(hand_ids, key=lambda t: (tile_id_to_index(t), t))
    return " ".join([tile_id_to_string(t) for t in sorted_ids])

def format_discards(discard_list):
    if not discard_list: return "なし"
    return " ".join([f"{tile_id_to_string(t)}{'*' if tsumogiri else ''}" for t, tsumogiri in discard_list])

def format_melds(meld_list_of_dicts):
    if not meld_list_of_dicts: return "なし"
    meld_strs = []
    for meld_info in meld_list_of_dicts:
        m_type = meld_info.get('type', '不明')
        m_tiles_ids = meld_info.get('tiles', [])
        from_who_abs = meld_info.get('from_who', -1)
        called_tile_id = meld_info.get('called_tile', -1)
        tiles_str = " ".join([tile_id_to_string(t) for t in sorted(m_tiles_ids, key=lambda x: (tile_id_to_index(x), x))])
        from_str = f" P{from_who_abs}" if from_who_abs != -1 and m_type not in ["暗槓", "加槓"] else ""
        trigger_str = f"({tile_id_to_string(called_tile_id)})" if called_tile_id != -1 and m_type != "暗槓" else ""
        meld_strs.append(f"{m_type}{trigger_str}[{tiles_str}]{from_str}")
    return " / ".join(meld_strs)

# --- モデルロード関数 ---
def load_trained_model(model_path, event_dim, static_dim, seq_len):
    try:
        model_params = {
            'event_feature_dim': event_dim,
            'static_feature_dim': static_dim,
            'd_model': 256, 'nhead': 4, 'd_hid': 1024, 'nlayers': 4,
            'dropout': 0.1, 'activation': 'relu', 'output_dim': NUM_TILE_TYPES
        }
        print(f"以下のパラメータでモデルを初期化します: {model_params}")
        model = MahjongTransformerV2(**model_params).to(DEVICE)
        try:
            checkpoint = torch.load(model_path, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            print(f"モデルを正常に読み込みました: {model_path}")
            return model
        except Exception as e:
            print(f"モデルの読み込み中にエラーが発生しました: {e}")
            raise
    except Exception as e:
        print(f"モデルの初期化中にエラーが発生しました: {e}")
        raise

# --- 局面復元関数 ---
def reconstruct_game_state_at_tsumo(xml_path, target_round_index, target_tsumo_event_count_in_round):
    print(f"牌譜ファイル {xml_path} を解析中...")
    try:
        meta, rounds_data = parse_full_mahjong_log(xml_path)
    except FileNotFoundError:
        print(f"[エラー] 牌譜ファイルが見つかりません: {xml_path}")
        raise
    except Exception as e:
        print(f"[エラー] 牌譜ファイルの解析中にエラーが発生しました: {e}")
        raise

    if not (1 <= target_round_index <= len(rounds_data)):
        raise ValueError(f"指定された局インデックスが無効です: {target_round_index} (利用可能な範囲: 1-{len(rounds_data)})")

    round_data = rounds_data[target_round_index - 1]
    game_state = GameState()
    print(f"第{target_round_index}局の初期状態を構築中...")
    try:
        game_state.init_round(round_data)
    except Exception as e:
        print(f"[エラー] GameState の初期化中にエラーが発生しました: {e}")
        raise

    current_tsumo_count = 0
    target_tsumo_event_info = None
    actual_discard_event_info = None
    events = round_data.get("events", [])
    print(f"イベントを再生し、{target_tsumo_event_count_in_round} 回目のツモを探します...")

    for i, event_xml in enumerate(events):
        tag = event_xml["tag"]
        attrib = event_xml["attrib"]
        processed_event_this_iteration = False
        try:
            tsumo_player_id = -1; tsumo_pai_id = -1
            for t_tag, p_id in GameState.TSUMO_TAGS.items():
                if tag.startswith(t_tag) and tag[1:].isdigit():
                    try: tsumo_pai_id = int(tag[1:]); tsumo_player_id = p_id; processed_event_this_iteration = True; break
                    except ValueError: continue
            if processed_event_this_iteration:
                current_tsumo_count += 1
                if current_tsumo_count == target_tsumo_event_count_in_round:
                    # print(f"ターゲットのツモ ({current_tsumo_count}回目) を発見しました。") # ログ削減
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
                                except ValueError: continue
                    print("指定局面の状態復元が完了しました。")
                    return game_state, target_tsumo_event_info, actual_discard_event_info
                else:
                    game_state.process_tsumo(tsumo_player_id, tsumo_pai_id)
                continue

            processed_event_this_iteration = False
            discard_player_id = -1; discard_pai_id = -1; tsumogiri = False
            for d_tag, p_id in GameState.DISCARD_TAGS.items():
                if tag.startswith(d_tag) and tag[1:].isdigit():
                    try: discard_pai_id = int(tag[1:]); discard_player_id = p_id; tsumogiri = tag[0].islower(); processed_event_this_iteration = True; break
                    except ValueError: continue
            if processed_event_this_iteration:
                game_state.process_discard(discard_player_id, discard_pai_id, tsumogiri)
                continue

            if not processed_event_this_iteration and tag == "N":
                try:
                    naki_player_id = int(attrib.get("who", -1)); meld_code = int(attrib.get("m", "0"))
                    if naki_player_id != -1: game_state.process_naki(naki_player_id, meld_code)
                except Exception as e: print(f"[警告] 鳴きイベント(N)の処理中にエラー: {e}, Attrib: {attrib}")
                continue

            if not processed_event_this_iteration and tag == "REACH":
                 try:
                     reach_player_id = int(attrib.get("who", -1)); step = int(attrib.get("step", 0))
                     if reach_player_id != -1: game_state.process_reach(reach_player_id, step)
                 except Exception as e: print(f"[警告] リーチイベント(REACH)の処理中にエラー: {e}, Attrib: {attrib}")
                 continue

            if not processed_event_this_iteration and tag == "DORA":
                 try:
                     hai_attr = attrib.get("hai")
                     if hai_attr is not None and hai_attr.isdigit():
                         hai = int(hai_attr)
                         if hai != -1: game_state.process_dora(hai)
                 except Exception as e: print(f"[警告] ドラ表示イベント(DORA)の処理中にエラー: {e}, Attrib: {attrib}")
                 continue

            if not processed_event_this_iteration and (tag == "AGARI" or tag == "RYUUKYOKU"):
                 # print(f"局終了イベント ({tag}) を検出しました。") # ログ削減
                 try:
                     if tag == "AGARI": game_state.process_agari(attrib)
                     else: game_state.process_ryuukyoku(attrib)
                 except Exception as e: print(f"[警告] 局終了イベントの処理中にエラー: {e}, Attrib: {attrib}")
                 break
        except Exception as e:
            print(f"[エラー] イベント {i} (タグ: {tag}, 属性: {attrib}) の処理中に予期せぬエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            raise e
    raise ValueError(f"指定されたツモ回数 ({target_tsumo_event_count_in_round}) に到達する前に局が終了、またはイベントがありませんでした（局: {target_round_index}）。")

# --- 打牌予測関数 ---
def predict_discard(model, game_state: GameState, player_id: int):
    try:
        event_sequence = game_state.get_event_sequence_features()
        static_features = game_state.get_static_features(player_id)
    except Exception as e:
        print(f"[エラー] 特徴量生成中にエラーが発生しました: {e}")
        raise

    seq_tensor = torch.tensor(event_sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    static_tensor = torch.tensor(static_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    try:
        padding_code = EVENT_TYPES["PADDING"]
    except KeyError:
        print("[警告] EVENT_TYPES['PADDING'] が見つかりません。デフォルト値 8 を使用します。")
        padding_code = 8.0
    except NameError:
        print("[警告] EVENT_TYPES がインポートされていません。デフォルト値 8 を使用します。")
        padding_code = 8.0

    mask_tensor = (seq_tensor[:, :, 0] == padding_code).to(DEVICE)

    with torch.no_grad():
        try:
            outputs = model(seq_tensor, static_tensor, mask_tensor)
        except Exception as e:
            print(f"[エラー] モデルのforward計算中にエラーが発生しました: {e}")
            print(f"  Input shapes: event_seq={seq_tensor.shape}, static_feat={static_tensor.shape}, mask={mask_tensor.shape}")
            raise
        probabilities = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    valid_discard_indices = game_state.get_valid_discard_options(player_id)
    best_prob = -1.0
    best_index = -1

    if not valid_discard_indices:
        print("[警告] 有効な打牌選択肢がありません！リーチ後のツモ切り牌などを確認してください。")
        best_index = np.argmax(probabilities)
        if 0 <= best_index < len(probabilities):
             best_prob = probabilities[best_index]
        else:
             print("[エラー] 確率配列から最大値を取得できませんでした。")
             return 0, 0.0, probabilities
    else:
        for index_val in valid_discard_indices:
            if 0 <= index_val < NUM_TILE_TYPES:
                if probabilities[index_val] > best_prob:
                    best_prob = probabilities[index_val]
                    best_index = index_val
            else: print(f"[警告] 無効な牌インデックス {index_val} が有効選択肢に含まれています。")
        if best_index == -1 and valid_discard_indices: # 有効牌はあるが選択できなかった場合
            # print(f"[警告] 有効牌の中から最良の打牌を決定できませんでした。最初の有効牌 ({valid_discard_indices[0]}) を選択します。")
            best_index = valid_discard_indices[0]
            best_prob = probabilities[best_index] if 0 <= best_index < len(probabilities) else 0.0

    if not (0 <= best_index < NUM_TILE_TYPES):
        print(f"[エラー] 最終的な打牌インデックス ({best_index}) が不正です。")
        return 0, 0.0, probabilities
    return best_index, best_prob, probabilities

# --- 特徴量名生成関数 ---
def generate_feature_names(event_dim, static_dim, seq_len):
    feature_names = []
    event_data_base_names = ["種別", "プレイヤー", "牌Idx", "巡目"]
    event_data_specific_names = ["Evデータ1", "Evデータ2"]
    expected_event_dim = 6
    if event_dim != expected_event_dim:
        current_event_names = event_data_base_names + event_data_specific_names
        if event_dim > len(current_event_names):
            event_names_final = current_event_names + [f"Ev追加データ{i+1}" for i in range(event_dim - len(current_event_names))]
        else:
            event_names_final = current_event_names[:event_dim]
    else:
        event_names_final = event_data_base_names + event_data_specific_names
    for i in range(seq_len):
        for j, name_suffix in enumerate(event_names_final):
            feature_names.append(f"Event_{i}_{name_suffix}")

    game_context_names = ["局風", "本場", "供託", "親ID", "壁残", "自身親?", "巡目", "ドラ表示数"]
    feature_names.extend([f"静_場_{name}" for name in game_context_names])
    player_specific_names = ["自_リーチ状態", "自_リーチ巡", "自_捨牌数", "自_副露数", "自_手牌数"]
    feature_names.extend([f"静_{name}" for name in player_specific_names])
    tile_kind_names_34 = [tile_id_to_string(tile_index_to_id(i)) for i in range(NUM_TILE_TYPES)]
    for tile_name in tile_kind_names_34: feature_names.append(f"静_手牌_{tile_name}")
    for tile_name in tile_kind_names_34: feature_names.append(f"静_ドラ表_{tile_name}")
    for tile_name in tile_kind_names_34: feature_names.append(f"静_自捨牌_{tile_name}")
    for tile_name in tile_kind_names_34: feature_names.append(f"静_全公開_{tile_name}")
    player_relative_names = ["相0(自)", "相1(下)", "相2(対)", "相3(上)"]
    for rel_name in player_relative_names:
        feature_names.append(f"静_位_{rel_name}_自フラグ")
        feature_names.append(f"静_位_{rel_name}_リ成立")
    
    static_feature_start_index = seq_len * event_dim
    num_generated_static_features = len(feature_names) - static_feature_start_index
    if num_generated_static_features != STATIC_FEATURE_DIM:
         if num_generated_static_features < STATIC_FEATURE_DIM:
             diff = STATIC_FEATURE_DIM - num_generated_static_features
             feature_names.extend([f"不明な静的特徴量_{i}" for i in range(diff)])
         else:
             feature_names = feature_names[:static_feature_start_index + STATIC_FEATURE_DIM]
    return feature_names

# --- SHAP説明関数 ---
def explain_prediction_with_shap(model, background_data_tuple, instance_to_explain_tuple, feature_names, target_class_index, n_shap_samples=100):
    global hook_outputs, attention_weights_storage
    hook_outputs = {}
    attention_weights_storage = []

    if not shap_available:
        print("SHAPライブラリが利用できないため、説明をスキップします。")
        return None

    print("\n--- SHAP 説明生成開始 ---")
    start_time = time.time()
    target_class_name = tile_id_to_string(tile_index_to_id(target_class_index)) if 0 <= target_class_index < NUM_TILE_TYPES else "N/A"
    # print(f"対象クラス: Index={target_class_index}, 牌種={target_class_name}") # ログ削減

    event_seq_instance, static_feat_instance, _ = instance_to_explain_tuple
    bg_sequences, bg_static_features = background_data_tuple

    seq_len = event_seq_instance.shape[0]
    event_dim = event_seq_instance.shape[1]

    def model_predict_proba_flat(flat_input_tensor_np):
        if isinstance(flat_input_tensor_np, np.ndarray):
            flat_input_tensor = torch.tensor(flat_input_tensor_np, dtype=torch.float32).to(DEVICE)
        else:
            flat_input_tensor = flat_input_tensor_np.to(DEVICE)

        batch_size = flat_input_tensor.shape[0]
        try:
            event_seq = flat_input_tensor[:, :(seq_len * event_dim)].reshape(batch_size, seq_len, event_dim)
            static_feat = flat_input_tensor[:, (seq_len * event_dim):]
        except Exception:
             return np.zeros((batch_size,))

        try:
            padding_code = EVENT_TYPES["PADDING"]
        except (KeyError, NameError): padding_code = 8.0
        mask = (event_seq[:, :, 0] == padding_code)

        with torch.no_grad():
            outputs = model(event_seq, static_feat, mask)
            probabilities = F.softmax(outputs, dim=1)
        return probabilities[:, target_class_index].cpu().numpy()

    bg_flat = np.concatenate([bg_sequences.reshape(bg_sequences.shape[0], -1), bg_static_features], axis=1)
    instance_flat = np.concatenate([event_seq_instance.flatten(), static_feat_instance]).reshape(1, -1)

    n_bg_summary = min(50, len(bg_flat))
    background_summary = shap.sample(bg_flat, n_bg_summary) if n_bg_summary > 0 and len(bg_flat) > 0 else bg_flat
    
    if background_summary.shape[0] == 0:
        print("[警告] SHAPの背景データが0件です。説明をスキップします。")
        return None

    try:
        explainer = shap.KernelExplainer(model_predict_proba_flat, background_summary)
    except Exception as e:
        print(f"[エラー] SHAP Explainer の初期化に失敗: {e}")
        return None

    # print(f"SHAP値を計算中 (n_shap_samples={n_shap_samples})... これには時間がかかります...") # ログ削減
    try:
        shap_values = explainer.shap_values(instance_flat, nsamples=n_shap_samples)
    except Exception as e:
        print(f"[エラー] SHAP値の計算中にエラーが発生しました: {e}")
        return None

    calculation_time = time.time() - start_time
    print(f"SHAP値の計算完了 ({calculation_time:.2f} 秒)")

    shap_values_flat = shap_values[0]

    if len(feature_names) != len(shap_values_flat):
         feature_importance = sorted(enumerate(shap_values_flat), key=lambda x: abs(x[1]), reverse=True)
         print(f"\n影響の大きい特徴量 Top 15 (インデックスとSHAP値):")
         for i, (idx, value) in enumerate(feature_importance[:15]):
             print(f"  {i+1}. Feature_{idx}: {value:.4f}")
    else:
        feature_importance_dict = dict(zip(feature_names, shap_values_flat))
        feature_importance_sorted = sorted(feature_importance_dict.items(), key=lambda item: abs(item[1]), reverse=True)
        print(f"\n影響の大きい特徴量 Top 15 (SHAP値):")
        for i, (name, value) in enumerate(feature_importance_sorted[:15]):
            print(f"  {i+1}. {name}: {value:.4f}")

    if shap_available:
        try:
            force_plot_fig = plt.figure()
            shap.force_plot(explainer.expected_value, shap_values[0], instance_flat[0], feature_names=feature_names, matplotlib=True, fig=force_plot_fig, show=False)
            safe_target_class_name = target_class_name.replace("/", "_")
            plot_filename = f"shap_force_plot_pred_{safe_target_class_name}.png"
            force_plot_fig.savefig(plot_filename, bbox_inches='tight')
            print(f"SHAP Force Plot を保存しました: {plot_filename}")
            plt.close(force_plot_fig)
        except Exception as plot_e:
            print(f"[警告] SHAP プロットの生成または保存に失敗しました: {plot_e}")
    return feature_importance_dict

# --- 局/自風 文字列取得関数 ---
def get_wind_str(round_num_wind, player_id, dealer):
    round_winds = ["東", "南", "西", "北"]
    player_winds = ["東", "南", "西", "北"]
    try:
        round_wind_idx = round_num_wind // NUM_PLAYERS
        kyoku_num = (round_num_wind % NUM_PLAYERS) + 1
        my_wind_idx = (player_id - dealer + NUM_PLAYERS) % NUM_PLAYERS
        return f"{round_winds[round_wind_idx]}{kyoku_num}局", player_winds[my_wind_idx]
    except (IndexError, TypeError):
        return "不明局", "不明家"

# --- 可視化関連のヘルパー関数 ---
def get_activation_hook(name):
    def hook(model, input, output):
        hook_outputs[name] = output.detach().cpu()
    return hook

def mha_forward_hook_with_weights(layer_name, original_mha_forward_func):
    def new_mha_forward(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, **kwargs):
        global attention_weights_storage
        # print(f"Hooked MHA called for: {layer_name}, Original need_weights: {need_weights}, Original average_attn_weights: {average_attn_weights}") # デバッグ用
        
        # 常にアテンションウェイトを取得し、ヘッド平均されたものを取得するように試みる
        # average_attn_weights は TransformerEncoderLayer のデフォルト呼び出しでは True のはず
        # need_weights も TransformerEncoderLayer のデフォルト呼び出しでは True のはず
        
        attn_output, attn_output_weights = original_mha_forward_func(
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=True,                      # ウェイトを要求
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights, # 呼び出し元の設定に従う
            **kwargs
        )
        
        if attn_output_weights is not None:
            # print(f"  {layer_name}: Appending attention weights, shape: {attn_output_weights.shape}") # デバッグ用
            attention_weights_storage.append(attn_output_weights.detach().cpu())
        else:
            # print(f"  {layer_name}: attn_output_weights is None from original_mha_forward_func call") # デバッグ用
            pass
        
        if not need_weights: # 元の呼び出しが need_weights=False だった場合
            return attn_output
        return attn_output, attn_output_weights
    return new_mha_forward


def visualize_attention_and_activations(model, event_seq_tensor, static_feat_tensor, mask_tensor, feature_names):
    global hook_outputs, attention_weights_storage, original_mha_forward_methods
    hook_outputs = {}
    attention_weights_storage = []
    original_mha_forward_methods = {}
    
    if not shap_available:
        print("Matplotlib が利用できないため、アテンション/活性化の可視化をスキップします。")
        return

    print("\n--- アテンションと活性化の可視化 ---")
    model.eval()
    hooks = []
    
    compiled_model = hasattr(model, '_orig_mod')
    actual_model = model._orig_mod if compiled_model else model

    if hasattr(actual_model, 'transformer_encoder') and hasattr(actual_model.transformer_encoder, 'layers'):
        for i, layer in enumerate(actual_model.transformer_encoder.layers):
            if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, nn.MultiheadAttention):
                layer_name = f"transformer_encoder.layers.{i}.self_attn"
                original_mha_forward_methods[layer_name] = layer.self_attn.forward
                layer.self_attn.forward = mha_forward_hook_with_weights(layer_name, original_mha_forward_methods[layer_name])
            # else: # ログ削減
                # print(f"[情報] TransformerEncoderLayer {i} にself_attn(MultiheadAttention)が見つかりません。")
    # else: # ログ削減
        # print("[警告] モデルにtransformer_encoder.layersが見つかりません。アテンションの可視化をスキップします。")

    try:
        target_activation_layer_name = 'event_encoder_linear1_out'
        if hasattr(actual_model, 'event_encoder') and len(actual_model.event_encoder) > 0 and isinstance(actual_model.event_encoder[0], nn.Linear):
            hooks.append(actual_model.event_encoder[0].register_forward_hook(get_activation_hook(target_activation_layer_name)))
        # else: # ログ削減
            # print(f"[情報] モデルにevent_encoder[0](Linear)が見つかりません。活性化({target_activation_layer_name})の可視化をスキップします。")
    except Exception as e:
        print(f"[警告] event_encoderのフック登録に失敗: {e}")

    with torch.no_grad():
        _ = model(event_seq_tensor, static_feat_tensor, mask_tensor)

    for h in hooks:
        h.remove()
    
    if hasattr(actual_model, 'transformer_encoder') and hasattr(actual_model.transformer_encoder, 'layers'):
        for i, layer in enumerate(actual_model.transformer_encoder.layers):
            layer_name = f"transformer_encoder.layers.{i}.self_attn"
            if layer_name in original_mha_forward_methods:
                layer.self_attn.forward = original_mha_forward_methods[layer_name]
    
    seq_len_actual = (~mask_tensor[0]).sum().item()
    event_step_labels = []
    if seq_len_actual > 0:
        try: padding_code = EVENT_TYPES["PADDING"]
        except (KeyError, NameError): padding_code = 8.0
        for step_idx in range(seq_len_actual):
            event_type_code = event_seq_tensor[0, step_idx, 0].item()
            event_type_str = "PAD"
            for name, code in EVENT_TYPES.items():
                if code == int(event_type_code): event_type_str = name; break
            if event_type_code == padding_code: event_type_str = "PAD"
            tile_idx_code = event_seq_tensor[0, step_idx, 2].item()
            tile_str = ""
            if tile_idx_code > 0:
                try: tile_str = "/" + tile_id_to_string(tile_index_to_id(int(tile_idx_code -1)))
                except: tile_str = f"/T{int(tile_idx_code-1)}"
            event_step_labels.append(f"S{step_idx}:{event_type_str[:3]}{tile_str}")

    if attention_weights_storage:
        for i, attn_weights_layer in enumerate(attention_weights_storage):
            # print(f"Visualizing attention for Layer {i}, shape: {attn_weights_layer.shape}") # デバッグ用
            if attn_weights_layer.dim() == 3: # (Batch, SeqLen, SeqLen)
                attn_map = attn_weights_layer[0].numpy()
            elif attn_weights_layer.dim() == 4: # (Batch, NumHeads, SeqLen, SeqLen)
                attn_map = attn_weights_layer[0].mean(dim=0).numpy() # ヘッド間で平均
                # attn_map = attn_weights_layer[0, 0].numpy() # または最初のヘッド
            else:
                print(f"[警告] Layer {i} のアテンションウェイトの形状がプロットに適していません: {attn_weights_layer.shape}")
                continue

            if seq_len_actual > 0 and attn_map.shape[0] >= seq_len_actual and attn_map.shape[1] >= seq_len_actual:
                plt.figure(figsize=(12, 10))
                plt.imshow(attn_map[:seq_len_actual, :seq_len_actual], cmap='viridis', aspect='auto', vmin=0)
                
                if len(event_step_labels) == seq_len_actual:
                    plt.xticks(np.arange(seq_len_actual), event_step_labels, rotation=90, fontsize=7)
                    plt.yticks(np.arange(seq_len_actual), event_step_labels, fontsize=7)
                
                plt.title(f'Self-Attention Weights (Layer {i})')
                plt.xlabel('Key Positions (Events)')
                plt.ylabel('Query Positions (Events)')
                plt.colorbar(label='Attention Weight')
                plt.tight_layout()
                attn_plot_path = f"attention_layer_{i}_visualization.png"
                plt.savefig(attn_plot_path)
                print(f"アテンションヒートマップ (Layer {i}) を保存しました: {attn_plot_path}")
                plt.close()
            else:
                print(f"[情報] Layer {i} アテンションマップの実際の系列長({seq_len_actual})またはマップ形状({attn_map.shape})が0または不整合のため、プロットをスキップします。")
    else:
        print("[情報] アテンションウェイトが取得できませんでした。transformer_encoderの実装やフックを確認してください。")

    if target_activation_layer_name in hook_outputs:
        activation_map = hook_outputs[target_activation_layer_name][0].cpu().numpy()
        if seq_len_actual > 0 and activation_map.shape[0] >= seq_len_actual :
            plt.figure(figsize=(15, 5))
            plt.imshow(activation_map[:seq_len_actual, :].T, cmap='viridis', aspect='auto')
            plt.title(f'Activations ({target_activation_layer_name})')
            plt.xlabel('Sequence Step (Actual)')
            plt.ylabel('Feature Dimension')
            if len(event_step_labels) == seq_len_actual:
                 plt.xticks(np.arange(seq_len_actual), event_step_labels, rotation=90, fontsize=7)

            plt.colorbar(label='Activation Value')
            plt.tight_layout()
            activation_plot_path = f"activation_{target_activation_layer_name}_visualization.png"
            plt.savefig(activation_plot_path)
            print(f"活性化ヒートマップ ({target_activation_layer_name}) を保存しました: {activation_plot_path}")
            plt.close()
        # else: # ログ削減
            # print(f"[情報] 活性化マップ ({target_activation_layer_name}) の実際の系列長が0、または期待より短いためプロットをスキップ。MapShape: {activation_map.shape}, ActualSeqLen: {seq_len_actual}")

    return attention_weights_storage


# --- メイン処理 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学習済み麻雀Transformerモデルを使って打牌を予測し、SHAPで説明します。")
    parser.add_argument("xml_file", help="天鳳形式のXML牌譜ファイルへのパス")
    parser.add_argument("round_index", type=int, help="対象局のインデックス (1から開始)")
    parser.add_argument("tsumo_count", type=int, help="対象局内でのツモ回数 (1から開始)")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help=f"学習済みモデル (.pth) ファイルへのパス (デフォルト: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--background_data_path", default=DEFAULT_BACKGROUND_DATA_PATH, help=f"SHAP背景データ用HDF5ファイルパス (デフォルト: {DEFAULT_BACKGROUND_DATA_PATH})")
    parser.add_argument("--shap_samples", type=int, default=100, help="SHAP値計算に使用するサンプル数 (KernelExplainer用)")
    parser.add_argument("--background_samples", type=int, default=50, help="SHAPの背景データとしてHDF5から読み込むサンプル数")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()])

    try:
        game_state, tsumo_info, discard_info = reconstruct_game_state_at_tsumo(
            args.xml_file, args.round_index, args.tsumo_count
        )
        player_id = game_state.current_player
        if tsumo_info and player_id != tsumo_info["player"]:
             player_id = tsumo_info["player"]
        elif not tsumo_info:
             print("[エラー] ツモイベント情報が見つかりません。")
             exit()

        try:
            event_sequence_instance = game_state.get_event_sequence_features()
            static_features_instance = game_state.get_static_features(player_id)
            event_dim = event_sequence_instance.shape[1]
            static_dim = static_features_instance.shape[0]
            seq_len = event_sequence_instance.shape[0]
        except Exception as e:
            print(f"[エラー] 特徴量の生成に失敗しました: {e}")
            raise

        model = load_trained_model(args.model_path, event_dim, static_dim, seq_len)
        predicted_index, predicted_prob, all_probabilities = predict_discard(model, game_state, player_id)
        predicted_tile_str = tile_id_to_string(tile_index_to_id(predicted_index))

        actual_discard_str = "N/A (局終了？)"
        if discard_info:
            actual_discard_str = tile_id_to_string(discard_info["pai"])
            if discard_info["tsumogiri"]: actual_discard_str += "*"

        print("\n=== Transformer 予測テスト ===")
        print(f"--- 対象局面 (R:{args.round_index}, TsumoCount:{args.tsumo_count}) ---")
        print(f"牌譜ファイル: {os.path.basename(args.xml_file)}")
        round_str, my_wind_str = get_wind_str(game_state.round_num_wind, player_id, game_state.dealer)
        honba_str = f"{game_state.honba}本場"
        kyotaku_str = f"({game_state.kyotaku}供託)" if game_state.kyotaku > 0 else ""
        print(f"局: {round_str} {honba_str} {kyotaku_str} / プレイヤー: {player_id} ({my_wind_str}家)")
        tsumo_pai_str = tile_id_to_string(tsumo_info['pai']) if tsumo_info else "不明"
        print(f"ツモ牌: {tsumo_pai_str}")
        print(f"巡目: {game_state.junme:.1f}")
        print(f"ドラ表示: {' '.join([tile_id_to_string(t) for t in game_state.dora_indicators])}")
        print(f"点数: {[f'P{i}:{s}' for i, s in enumerate(game_state.current_scores)]}")
        
        print("--- 現在の状態 ---")
        print("手牌 (ツモ後):")
        for p in range(NUM_PLAYERS):
            hand_str = format_hand(game_state.player_hands[p])
            reach_indicator = "*" if game_state.player_reach_status[p] == 2 else ("(宣)" if game_state.player_reach_status[p] == 1 else "")
            print(f"  P{p}{reach_indicator}: {hand_str}")
        print("捨て牌:")
        for p in range(NUM_PLAYERS):
            discard_str = format_discards(game_state.player_discards[p])
            print(f"  P{p}: {discard_str}")
        print("鳴き:")
        for p in range(NUM_PLAYERS):
            meld_str = format_melds(game_state.player_melds[p])
            print(f"  P{p}: {meld_str}")
        print("-" * 20)
        print(f"予測された捨て牌 (牌種): {predicted_tile_str}")
        print(f"予測確率: {predicted_prob:.4f}")
        print(f"実際の捨て牌: {actual_discard_str}")
        print("-" * 20)
        top_n = 5
        indices_sorted = np.argsort(all_probabilities)[::-1]
        print(f"予測確率 Top {top_n}:")
        for i_loop in range(top_n):
            idx = indices_sorted[i_loop]
            prob = all_probabilities[idx]
            tile_str = tile_id_to_string(tile_index_to_id(idx))
            print(f"  {i_loop+1}. {tile_str} ({prob:.4f})")

        if shap_available:
            try:
                background_data_tuple = (np.array([]), np.array([]))
                if not os.path.exists(args.background_data_path):
                    print(f"[警告] SHAP用の背景データファイルが見つかりません: {args.background_data_path}。説明をスキップします。")
                else:
                    try:
                        with h5py.File(args.background_data_path, "r") as hf:
                            if "labels" not in hf or "sequences" not in hf or "static_features" not in hf :
                                print(f"[警告] 背景データファイル {args.background_data_path} に必要なデータセットがありません。説明をスキップします。")
                            else:
                                total_bg_samples = hf["labels"].shape[0]
                                if total_bg_samples == 0:
                                    print("[警告] 背景データファイルにサンプルがありません。説明をスキップします。")
                                else:
                                    n_bg = min(args.background_samples, total_bg_samples)
                                    if n_bg <= 0:
                                        print("[警告] 背景サンプル数が0以下です。説明をスキップします。")
                                    else:
                                        indices_to_load = np.random.choice(total_bg_samples, n_bg, replace=False) if n_bg < total_bg_samples else np.arange(n_bg)
                                        indices_to_load.sort()
                                        bg_sequences_list = [hf["sequences"][k_idx] for k_idx in indices_to_load]
                                        bg_static_features_list = [hf["static_features"][k_idx] for k_idx in indices_to_load]
                                        bg_sequences = np.array(bg_sequences_list)
                                        bg_static_features = np.array(bg_static_features_list)
                                        background_data_tuple = (bg_sequences, bg_static_features)
                    except Exception as bg_load_e:
                        print(f"[警告] 背景データのロード中にエラーが発生しました: {bg_load_e}。説明をスキップします。")
                
                if background_data_tuple[0].shape[0] > 0:
                    instance_to_explain_tuple = (event_sequence_instance, static_features_instance, None)
                    feature_names_for_shap = generate_feature_names(event_dim, static_dim, seq_len)
                    _ = explain_prediction_with_shap(
                        model, background_data_tuple, instance_to_explain_tuple,
                        feature_names_for_shap, predicted_index, n_shap_samples=args.shap_samples
                    )
            except Exception as shap_e:
                print(f"\n[エラー] SHAP説明の生成中にエラーが発生しました: {shap_e}")
        else:
            pass

        if args.xml_file and args.round_index and args.tsumo_count:
            try:
                event_seq_tensor_for_viz = torch.tensor(event_sequence_instance, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                static_feat_tensor_for_viz = torch.tensor(static_features_instance, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                try:
                    padding_code_viz = EVENT_TYPES["PADDING"]
                except (KeyError, NameError): padding_code_viz = 8.0
                mask_tensor_for_viz = (event_seq_tensor_for_viz[:, :, 0] == padding_code_viz).to(DEVICE)
                
                feature_names_for_viz = generate_feature_names(event_dim, static_dim, seq_len)
                attn_weights_list = visualize_attention_and_activations(model, event_seq_tensor_for_viz, static_feat_tensor_for_viz, mask_tensor_for_viz, feature_names_for_viz)
                print("\n[INFO] Attention weights (as numpy arrays) are available as attn_weights_list. Example: attn_weights_list[0].shape =", attn_weights_list[0].shape if attn_weights_list else None)
            except Exception as viz_e:
                print(f"\n[エラー] アテンション/活性化の可視化中にエラーが発生しました: {viz_e}")
                import traceback
                traceback.print_exc()

    except FileNotFoundError as e: print(f"エラー: ファイルが見つかりません - {e}")
    except ValueError as e: print(f"エラー: 値が不正です - {e}")
    except ImportError as e: print(f"エラー: インポートに失敗しました - {e}")
    except AttributeError as e: print(f"エラー: 属性エラー（クラス定義やメソッド呼び出しを確認） - {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()