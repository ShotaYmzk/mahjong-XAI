# predict.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import time
import h5py
import math # RotaryPositionalEncoding で使用
import logging # ログ出力用

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
    # 動作継続のためダミー値を設定するが、基本的にはエラー終了させるべき
    NUM_TILE_TYPES = 34
    MAX_EVENT_HISTORY = 60
    STATIC_FEATURE_DIM = 157
    EVENT_TYPES = {"PADDING": 8} # 必須
    print("[警告] game_state からの定数インポートに失敗。デフォルト値/ダミーを使用します。")


from naki_utils import decode_naki
from tile_utils import tile_id_to_string, tile_id_to_index, tile_index_to_id, tile_index_to_str

# --- 設定 ---
NUM_PLAYERS = 4 # GameState内で使われることがあるので定義
DEFAULT_MODEL_PATH = "./trained_model/mahjong_transformer_v1111_large_compiled_2.pth"
DEFAULT_BACKGROUND_DATA_PATH = "/home/ubuntu/Documents/mahjong-XAI/ver_1.1.10/training_data/mahjong_imitation_data_v1110.hdf5" # preprocess_data.py の出力ファイル名に合わせる
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

print(f"使用デバイス: {DEVICE}")

# --- グローバル変数 (フック用) ---
hook_outputs = {}
attention_weights_storage = []
original_mha_forward_methods = {}


# --- クラス定義 (train2.py から MahjongTransformerV2 と RotaryPositionalEncoding をコピー) ---
class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Encoding (RoPE)の実装"""
    def __init__(self, d_model, max_len=MAX_EVENT_HISTORY): # MAX_EVENT_HISTORY をデフォルト値として使用
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
            # logging.warning(f"RoPE: Input sequence length {seq_len} > precomputed max_len {self.max_len}. Recomputing positions.") # ログが多い場合はコメントアウト
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
        self.pos_encoder = RotaryPositionalEncoding(d_model) # MAX_EVENT_HISTORY はここで使われる
        encoder_layers = nn.TransformerEncoderLayer( # torch.nn.TransformerEncoderLayer を使用
            d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout,
            activation=activation, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers) # torch.nn.TransformerEncoder を使用
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
                # gain は活性化関数に応じて調整 (GELUは通常1.0)
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
            mask_expanded = attention_mask.unsqueeze(-1) # (Batch, SeqLen, 1)
            attn_weights = attn_weights.masked_fill(mask_expanded, 0.0)
        context_vector = torch.sum(attn_weights * transformer_output, dim=1)
        static_encoded = self.static_encoder(static_feat)
        combined = torch.cat([context_vector, static_encoded], dim=1)
        return self.output_head(combined)

# --- ヘルパー関数 ---
def format_hand(hand_ids):
    if not hand_ids: return "なし"
    sorted_ids = sorted(hand_ids, key=lambda t: (tile_id_to_index(t), t))
    return " ".join([tile_id_to_string(t) for t in sorted_ids])

def format_discards(discard_list):
    if not discard_list: return "なし"
    return " ".join([f"{tile_id_to_string(t)}{'*' if tsumogiri else ''}" for t, tsumogiri in discard_list])

def format_melds(meld_list_of_dicts): # game_state.player_melds は dict のリスト
    if not meld_list_of_dicts: return "なし"
    meld_strs = []
    for meld_info in meld_list_of_dicts: # meld_info は dict
        m_type = meld_info.get('type', '不明')
        m_tiles_ids = meld_info.get('tiles', [])
        from_who_abs = meld_info.get('from_who', -1) # game_state.py の process_naki で設定されるキー
        called_tile_id = meld_info.get('called_tile', -1) # game_state.py の process_naki で設定されるキー

        tiles_str = " ".join([tile_id_to_string(t) for t in sorted(m_tiles_ids, key=lambda x: (tile_id_to_index(x), x))])
        from_str = f" P{from_who_abs}" if from_who_abs != -1 and m_type not in ["暗槓", "加槓"] else "" # 暗槓/加槓は自分から
        trigger_str = f"({tile_id_to_string(called_tile_id)})" if called_tile_id != -1 and m_type != "暗槓" else "" # 暗槓はトリガー牌なし
        meld_strs.append(f"{m_type}{trigger_str}[{tiles_str}]{from_str}")
    return " / ".join(meld_strs)

# --- モデルロード関数 ---
def load_trained_model(model_path, event_dim, static_dim, seq_len): # seq_len はRoPEのmax_lenに使われる
    try:
        model_params = {
            'event_feature_dim': event_dim,
            'static_feature_dim': static_dim,
            'd_model': 256, 'nhead': 4, 'd_hid': 1024, 'nlayers': 4,
            'dropout': 0.1, 'activation': 'relu', 'output_dim': NUM_TILE_TYPES
        }
        print(f"以下のパラメータでモデルを初期化します: {model_params}")
        # RotaryPositionalEncoding の max_len は MAX_EVENT_HISTORY を使うので、
        # ここで seq_len を渡す必要はない。MahjongTransformerV2 の __init__ で RotaryPositionalEncoding が初期化される。
        model = MahjongTransformerV2(**model_params).to(DEVICE)
        try:
            checkpoint = torch.load(model_path, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                # torch.compile されている場合、元のモデルの state_dict は _orig_mod にある
                if hasattr(model, '_orig_mod') and isinstance(model._orig_mod, nn.Module):
                     model._orig_mod.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])
            else: # 古い形式のチェックポイント（model_state_dict キーなし）
                if hasattr(model, '_orig_mod') and isinstance(model._orig_mod, nn.Module):
                    model._orig_mod.load_state_dict(checkpoint)
                else:
                    model.load_state_dict(checkpoint)

            model.eval()
            print(f"モデルを正常に読み込みました: {model_path}")
            return model
        except Exception as e:
            print(f"モデルの読み込み中にエラーが発生しました: {e}")
            # 読み込みに失敗した場合、モデルのキーとチェックポイントのキーを表示
            print("Model state_dict keys:", list(model.state_dict().keys())[:5])
            if isinstance(checkpoint, dict):
                print("Checkpoint keys:", list(checkpoint.keys()))
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
    game_state = GameState() # GameState インスタンスを作成
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
        # print(f"  Processing event {i}: <{tag}> {attrib}") # デバッグ用ログ
        try:
            # --- 自摸イベント ---
            tsumo_player_id = -1; tsumo_pai_id = -1
            for t_tag, p_id in GameState.TSUMO_TAGS.items(): # GameState.TSUMO_TAGS を使用
                if tag.startswith(t_tag) and tag[1:].isdigit():
                    try: tsumo_pai_id = int(tag[1:]); tsumo_player_id = p_id; processed_event_this_iteration = True; break
                    except ValueError: continue # 不正な牌IDの場合はスキップ
            if processed_event_this_iteration:
                current_tsumo_count += 1
                # print(f"  Event {i}: P{tsumo_player_id} ツモ {tile_id_to_string(tsumo_pai_id)} (カウント: {current_tsumo_count})") # ログ強化
                if current_tsumo_count == target_tsumo_event_count_in_round:
                    print(f"ターゲットのツモ ({current_tsumo_count}回目) を発見しました。")
                    target_tsumo_event_info = {"player": tsumo_player_id, "pai": tsumo_pai_id, "xml": event_xml}
                    game_state.process_tsumo(tsumo_player_id, tsumo_pai_id) # 状態更新
                    # 次の打牌イベントを探す (なければ None)
                    if i + 1 < len(events):
                        next_event_xml = events[i+1]; next_tag = next_event_xml["tag"]
                        for d_tag, p_id_next in GameState.DISCARD_TAGS.items(): # GameState.DISCARD_TAGS を使用
                            if next_tag.startswith(d_tag) and next_tag[1:].isdigit() and p_id_next == tsumo_player_id:
                                try:
                                    discard_pai_id = int(next_tag[1:]); tsumogiri = next_tag[0].islower()
                                    actual_discard_event_info = {"player": p_id_next, "pai": discard_pai_id, "tsumogiri": tsumogiri, "xml": next_event_xml}
                                    break
                                except ValueError: continue
                    print("指定局面の状態復元が完了しました。")
                    return game_state, target_tsumo_event_info, actual_discard_event_info # 発見したので返す
                else:
                    game_state.process_tsumo(tsumo_player_id, tsumo_pai_id) # 状態更新
                continue # 次のイベントへ

            # --- 打牌イベント ---
            processed_event_this_iteration = False # リセット
            discard_player_id = -1; discard_pai_id = -1; tsumogiri = False
            for d_tag, p_id in GameState.DISCARD_TAGS.items(): # GameState.DISCARD_TAGS を使用
                if tag.startswith(d_tag) and tag[1:].isdigit():
                    try: discard_pai_id = int(tag[1:]); discard_player_id = p_id; tsumogiri = tag[0].islower(); processed_event_this_iteration = True; break
                    except ValueError: continue
            if processed_event_this_iteration:
                # print(f"  Event {i}: P{discard_player_id} 打牌 {tile_id_to_string(discard_pai_id)}{'*' if tsumogiri else ''}") # ログ強化
                game_state.process_discard(discard_player_id, discard_pai_id, tsumogiri)
                continue

            # --- その他のイベント (鳴き、リーチ、ドラ、局終了など) ---
            if not processed_event_this_iteration and tag == "N":
                try:
                    naki_player_id = int(attrib.get("who", -1)); meld_code = int(attrib.get("m", "0"))
                    if naki_player_id != -1:
                        # print(f"  Event {i}: P{naki_player_id} 鳴き (m={meld_code})") # ログ強化
                        game_state.process_naki(naki_player_id, meld_code)
                except Exception as e: print(f"[警告] 鳴きイベント(N)の処理中にエラー: {e}, Attrib: {attrib}")
                continue

            if not processed_event_this_iteration and tag == "REACH":
                 try:
                     reach_player_id = int(attrib.get("who", -1)); step = int(attrib.get("step", 0))
                     if reach_player_id != -1:
                         # print(f"  Event {i}: P{reach_player_id} リーチ (step {step})") # ログ強化
                         game_state.process_reach(reach_player_id, step)
                 except Exception as e: print(f"[警告] リーチイベント(REACH)の処理中にエラー: {e}, Attrib: {attrib}")
                 continue

            if not processed_event_this_iteration and tag == "DORA":
                 try:
                     hai_attr = attrib.get("hai")
                     if hai_attr is not None and hai_attr.isdigit():
                         hai = int(hai_attr)
                         if hai != -1:
                             # print(f"  Event {i}: ドラ表示 {tile_id_to_string(hai)}") # ログ強化
                             game_state.process_dora(hai)
                 except Exception as e: print(f"[警告] ドラ表示イベント(DORA)の処理中にエラー: {e}, Attrib: {attrib}")
                 continue

            if not processed_event_this_iteration and (tag == "AGARI" or tag == "RYUUKYOKU"):
                 print(f"  Event {i}: 局終了イベント ({tag}) を検出しました。")
                 try:
                     if tag == "AGARI": game_state.process_agari(attrib)
                     else: game_state.process_ryuukyoku(attrib)
                 except Exception as e: print(f"[警告] 局終了イベントの処理中にエラー: {e}, Attrib: {attrib}")
                 break # この局のイベント再生は終了
            
            # if not processed_event_this_iteration:
            #     print(f"  Event {i}: 未処理または状態変更なしのタグ <{tag}> {attrib}")


        except Exception as e:
            print(f"[エラー] イベント {i} (タグ: {tag}, 属性: {attrib}) の処理中に予期せぬエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            raise e # エラーを再発生させて処理を中断

    # ループが終了しても指定ツモ回数に達しなかった場合
    raise ValueError(f"指定されたツモ回数 ({target_tsumo_event_count_in_round}) に到達する前に局が終了、またはイベントがありませんでした（局: {target_round_index}）。")


# --- 打牌予測関数 ---
def predict_discard(model, game_state: GameState, player_id: int):
    try:
        event_sequence = game_state.get_event_sequence_features()
        static_features = game_state.get_static_features(player_id)
    except Exception as e:
        print(f"[エラー] 特徴量生成中にエラーが発生しました: {e}")
        raise

    # print(f"Debug: Event sequence shape: {event_sequence.shape}, Static features shape: {static_features.shape}")
    # print(f"Debug: Event sequence (last 3 events, first 4 features): \n{event_sequence[-3:, :4]}")
    # print(f"Debug: Static features (first 10): {static_features[:10]}")

    seq_tensor = torch.tensor(event_sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    static_tensor = torch.tensor(static_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    try:
        # EVENT_TYPES がグローバルに存在するか確認
        padding_code = EVENT_TYPES["PADDING"] if 'EVENT_TYPES' in globals() and isinstance(EVENT_TYPES, dict) and "PADDING" in EVENT_TYPES else 8.0
    except Exception: # NameError や KeyError をキャッチ
        print("[警告] EVENT_TYPES['PADDING'] の取得に失敗。デフォルト値 8.0 を使用します。")
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

    # print(f"Debug: Probabilities sum: {np.sum(probabilities):.4f}")
    # print(f"Debug: Probabilities (first 5): {probabilities[:5]}")

    valid_discard_indices = game_state.get_valid_discard_options(player_id)
    # print(f"Debug: Valid discard options (indices): {valid_discard_indices} for player {player_id}")
    # print(f"Debug: Player {player_id} hand for validation: {format_hand(game_state.player_hands[player_id])}")


    best_prob = -1.0
    best_index = -1

    if not valid_discard_indices:
        print(f"[警告] P{player_id}: 有効な打牌選択肢がありません！リーチ後のツモ切り牌などを確認してください。")
        # フォールバック: 最も確率の高い牌を選択 (手牌になくても)
        best_index = np.argmax(probabilities)
        if 0 <= best_index < len(probabilities):
             best_prob = probabilities[best_index]
        else: # 通常発生しない
             print("[エラー] 確率配列から最大値を取得できませんでした。")
             return 0, 0.0, probabilities # ダミー値を返す
    else:
        for index_val in valid_discard_indices: # index_val は牌種インデックス (0-33)
            if 0 <= index_val < NUM_TILE_TYPES:
                if probabilities[index_val] > best_prob:
                    best_prob = probabilities[index_val]
                    best_index = index_val
            else: print(f"[警告] 無効な牌インデックス {index_val} が有効選択肢に含まれています。")
        
        if best_index == -1: # 有効牌はあるが、何らかの理由で選択できなかった場合 (例: 確率が全て負など、通常ありえない)
            print(f"[警告] 有効牌の中から最良の打牌を決定できませんでした。最初の有効牌 ({valid_discard_indices[0]}) を選択します。")
            best_index = valid_discard_indices[0]
            best_prob = probabilities[best_index] if 0 <= best_index < len(probabilities) else 0.0

    if not (0 <= best_index < NUM_TILE_TYPES): # 最終チェック
        print(f"[エラー] 最終的な打牌インデックス ({best_index}) が不正です。")
        return 0, 0.0, probabilities # ダミー値を返す

    return best_index, best_prob, probabilities

# --- 特徴量名生成関数 ---
def generate_feature_names(event_dim, static_dim, seq_len):
    feature_names = []
    # print(f"特徴量名を生成中... (シーケンス長: {seq_len}, イベント次元: {event_dim}, 静的次元: {static_dim})")

    # 1. イベントシーケンス特徴量名
    # game_state.py の get_event_sequence_features の event_vec の割り当て順に合わせる
    event_data_base_names = ["種別", "プレイヤー", "牌Idx", "巡目"] # 基本4次元
    # イベントタイプごとの追加データ名 (最大2次元を想定)
    event_data_specific_names = ["Evデータ1(ツモ切/鳴種/リ段階)", "Evデータ2(鳴元/リ巡)"]
    
    # event_dim に合わせて event_names_final を構築
    current_event_names = event_data_base_names + event_data_specific_names
    if event_dim > len(current_event_names):
        event_names_final = current_event_names + [f"Ev追加データ{i+1}" for i in range(event_dim - len(current_event_names))]
    else:
        event_names_final = current_event_names[:event_dim]

    for i in range(seq_len):
        for j, name_suffix in enumerate(event_names_final):
            feature_names.append(f"Event_{i}_{name_suffix}")

    # 2. 静的特徴量名 (GameState.get_static_features の実装順に依存！)
    # GameState.get_static_features の DIM 定義と割り当て順を参照
    # DIM_GAME_CONTEXT = 8
    game_context_names = ["局風", "本場", "供託", "親ID", "壁残", "自身親?", "巡目(GS)", "ドラ表示枚数"]
    feature_names.extend([f"静_場_{name}" for name in game_context_names])

    # DIM_PLAYER_SPECIFIC = 5
    player_specific_names = ["自_リーチ状態", "自_リーチ巡", "自_捨牌数", "自_副露数", "自_手牌数"]
    feature_names.extend([f"静_{name}" for name in player_specific_names])

    tile_kind_names_34 = [tile_id_to_string(tile_index_to_id(i)) for i in range(NUM_TILE_TYPES)]
    # DIM_HAND_COUNTS = 34
    for tile_name in tile_kind_names_34: feature_names.append(f"静_手牌_{tile_name}")
    # DIM_DORA_INDICATORS = 34
    for tile_name in tile_kind_names_34: feature_names.append(f"静_ドラ表示牌_{tile_name}") # "ドラ表" から "ドラ表示牌" へ変更
    # DIM_PLAYER_DISCARDS = 34
    for tile_name in tile_kind_names_34: feature_names.append(f"静_自捨牌_{tile_name}")
    # DIM_ALL_VISIBLE = 34
    for tile_name in tile_kind_names_34: feature_names.append(f"静_全公開牌_{tile_name}") # "全公開" から "全公開牌" へ変更

    # DIM_PLAYER_POS_REACH = 8
    player_relative_names = ["相0(自)", "相1(下)", "相2(対)", "相3(上)"] # 4人分
    for rel_name in player_relative_names:
        feature_names.append(f"静_相対_{rel_name}_自身フラグ") # "位" から "相対" へ
        feature_names.append(f"静_相対_{rel_name}_リーチ成立済") # "リ成立" から "リーチ成立済" へ
    
    # --- 次元数チェック ---
    # STATIC_FEATURE_DIM は game_state.py からインポートされた定数を使う
    num_generated_static_features = len(feature_names) - (seq_len * event_dim)
    if num_generated_static_features != STATIC_FEATURE_DIM:
        print(f"[警告] 生成された静的特徴量名の数 ({num_generated_static_features}) が期待値 ({STATIC_FEATURE_DIM}) と異なります。")
        print(f"       GameState.get_static_features の実装と突き合わせてください。")
        # 足りない分/多い分を調整 (デバッグ用)
        if num_generated_static_features < STATIC_FEATURE_DIM:
            diff = STATIC_FEATURE_DIM - num_generated_static_features
            feature_names.extend([f"不明な静的特徴量_{i}" for i in range(diff)])
        else: # 多すぎる場合
            feature_names = feature_names[: (seq_len * event_dim) + STATIC_FEATURE_DIM]

    # print(f"特徴量名の生成完了 (合計: {len(feature_names)}個)")
    return feature_names


# --- SHAP説明関数 ---
def explain_prediction_with_shap(model, background_data_tuple, instance_to_explain_tuple, feature_names, target_class_index, n_shap_samples=100):
    global hook_outputs, attention_weights_storage # グローバル変数の参照を明示
    hook_outputs = {} # クリア
    attention_weights_storage = [] # クリア

    if not shap_available:
        print("SHAPライブラリが利用できないため、説明をスキップします。")
        return None

    print("\n--- SHAP 説明生成開始 ---")
    start_time = time.time()
    # target_class_index は 0-33 の牌種インデックス
    target_class_name = tile_id_to_string(tile_index_to_id(target_class_index)) if 0 <= target_class_index < NUM_TILE_TYPES else "N/A"
    print(f"対象クラス: Index={target_class_index}, 牌種={target_class_name}")

    event_seq_instance, static_feat_instance, _ = instance_to_explain_tuple # マスクはSHAPラッパー内で生成
    bg_sequences, bg_static_features = background_data_tuple

    seq_len = event_seq_instance.shape[0]
    event_dim = event_seq_instance.shape[1]

    def model_predict_proba_flat(flat_input_tensor_np):
        if isinstance(flat_input_tensor_np, np.ndarray):
            flat_input_tensor = torch.tensor(flat_input_tensor_np, dtype=torch.float32).to(DEVICE)
        else: # すでにTensorの場合 (Explainerによる)
            flat_input_tensor = flat_input_tensor_np.to(DEVICE)

        batch_size = flat_input_tensor.shape[0]
        # SequenceとStaticに分割
        try:
            event_seq = flat_input_tensor[:, :(seq_len * event_dim)].reshape(batch_size, seq_len, event_dim)
            static_feat = flat_input_tensor[:, (seq_len * event_dim):]
        except Exception as e:
             print(f"[エラー] SHAPラッパー内でのテンソル再構成に失敗: {e}")
             return np.zeros((batch_size,)) # ダミーの確率を返す

        # パディングマスク生成
        try:
            # EVENT_TYPES がグローバルに存在するか確認
            padding_code = EVENT_TYPES["PADDING"] if 'EVENT_TYPES' in globals() and isinstance(EVENT_TYPES, dict) and "PADDING" in EVENT_TYPES else 8.0
        except Exception:
            padding_code = 8.0
        mask = (event_seq[:, :, 0] == padding_code) # device は event_seq に依存

        # モデル予測
        with torch.no_grad():
            outputs = model(event_seq, static_feat, mask)
            probabilities = F.softmax(outputs, dim=1)

        # 対象クラスの確率をNumpy配列で返す
        return probabilities[:, target_class_index].cpu().numpy()

    # SHAP Explainer の準備
    # KernelExplainerは時間がかかるため、背景データはサンプリングして使う
    bg_flat = np.concatenate([bg_sequences.reshape(bg_sequences.shape[0], -1), bg_static_features], axis=1)
    # instance も flat にしてバッチ次元追加
    instance_flat = np.concatenate([event_seq_instance.flatten(), static_feat_instance]).reshape(1, -1)

    # shap.sampleで背景データをサンプリング (KernelExplainerの計算量削減)
    n_bg_summary = min(50, len(bg_flat)) # 最大50サンプルで要約
    # 背景データが空の場合はサンプリングしない
    background_summary = shap.sample(bg_flat, n_bg_summary) if n_bg_summary > 0 and len(bg_flat) > 0 else bg_flat
    
    if background_summary.shape[0] == 0:
        print("[警告] SHAPの背景データが0件です。説明をスキップします。")
        return None

    try:
        print("SHAP KernelExplainer を初期化中...")
        explainer = shap.KernelExplainer(model_predict_proba_flat, background_summary)
    except Exception as e:
        print(f"[エラー] SHAP Explainer の初期化に失敗: {e}")
        return None

    print(f"SHAP値を計算中 (n_shap_samples={n_shap_samples})... これには時間がかかります...")
    try:
        shap_values = explainer.shap_values(instance_flat, nsamples=n_shap_samples) # nsamples で精度と速度を調整
    except Exception as e:
        print(f"[エラー] SHAP値の計算中にエラーが発生しました: {e}")
        print(f"  Explainer input shape: {instance_flat.shape}")
        print(f"  Background summary shape: {background_summary.shape}")
        return None

    calculation_time = time.time() - start_time
    print(f"SHAP値の計算完了 ({calculation_time:.2f} 秒)")

    shap_values_flat = shap_values[0] # [1, num_features] -> [num_features]

    if len(feature_names) != len(shap_values_flat):
         print(f"[エラー] 特徴量名の数 ({len(feature_names)}) と SHAP値の数 ({len(shap_values_flat)}) が一致しません。generate_feature_names を確認してください。")
         feature_importance = sorted(enumerate(shap_values_flat), key=lambda x: abs(x[1]), reverse=True)
         print(f"\n影響の大きい特徴量 Top 15 (インデックスとSHAP値):")
         for i, (idx, value) in enumerate(feature_importance[:15]):
             print(f"  {i+1}. Feature_{idx}: {value:.4f}")
    else:
        # 特徴量名とSHAP値を紐付け
        feature_importance_dict = dict(zip(feature_names, shap_values_flat))
        # 絶対値でソート
        feature_importance_sorted = sorted(feature_importance_dict.items(), key=lambda item: abs(item[1]), reverse=True)

        print(f"\n影響の大きい特徴量 Top 15 (SHAP値):")
        for i, (name, value) in enumerate(feature_importance_sorted[:15]):
            print(f"  {i+1}. {name}: {value:.4f}")

    if shap_available:
        try:
            print("SHAP Force Plot を生成・保存中...")
            force_plot_fig = plt.figure() # 新しいFigureを作成
            shap.force_plot(explainer.expected_value, shap_values[0], instance_flat[0], feature_names=feature_names, matplotlib=True, fig=force_plot_fig, show=False)
            # ファイル名として安全な文字列に置換
            safe_target_class_name = target_class_name.replace("/", "_").replace("\\", "_")
            plot_filename = f"shap_force_plot_pred_{safe_target_class_name}.png"
            force_plot_fig.savefig(plot_filename, bbox_inches='tight')
            print(f"SHAP Force Plot を保存しました: {plot_filename}")
            plt.close(force_plot_fig) # Figureを閉じる
        except Exception as plot_e:
            print(f"[警告] SHAP プロットの生成または保存に失敗しました: {plot_e}")
    return feature_importance_dict

# --- 局/自風 文字列取得関数 ---
def get_wind_str(round_num_wind, player_id, dealer):
    round_winds = ["東", "南", "西", "北"]
    player_winds = ["東", "南", "西", "北"] # 親から見て反時計回り
    try:
        # 場風: 東=0, 南=1, 西=2, 北=3 (round_num_windから計算)
        round_wind_idx = round_num_wind // NUM_PLAYERS # 0-3:東, 4-7:南 ...
        # 局数: 1-4 (round_num_windから計算)
        kyoku_num = (round_num_wind % NUM_PLAYERS) + 1
        # 自風: 東=0, 南=1, 西=2, 北=3 (player_idとdealerから計算)
        # (player_id - dealer + NUM_PLAYERS) % NUM_PLAYERS
        my_wind_idx = (player_id - dealer + NUM_PLAYERS) % NUM_PLAYERS
        return f"{round_winds[round_wind_idx]}{kyoku_num}局", player_winds[my_wind_idx]
    except (IndexError, TypeError) as e: # エラーハンドリング追加
        print(f"[警告] get_wind_str でエラー発生: {e}. round_num_wind={round_num_wind}, player_id={player_id}, dealer={dealer}")
        return "不明局", "不明家"


# --- 可視化関連のヘルパー関数 ---
def get_activation_hook(name):
    def hook(model, input, output):
        global hook_outputs # グローバル変数を更新するため
        hook_outputs[name] = output.detach().cpu()
    return hook

def mha_forward_hook_with_weights(layer_name, original_mha_forward_func):
    def new_mha_forward(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, **kwargs):
        global attention_weights_storage
        # デバッグログを追加
        print(f"Executing MHA forward for {layer_name}")
        print(f"Input shapes - query: {query.shape}, key: {key.shape}, value: {value.shape}")
        
        # 元の関数を呼び出し
        attn_output, attn_output_weights = original_mha_forward_func(
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=True,  # 必ずTrueに設定
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            **kwargs
        )
        
        # アテンションウェイトの保存
        if attn_output_weights is not None:
            print(f"Storing attention weights for {layer_name}, shape: {attn_output_weights.shape}")
            print(f"Attention weights sample (first 3x3):\n{attn_output_weights[0, 0, :3, :3].detach().cpu().numpy()}")
            attention_weights_storage.append(attn_output_weights.detach().cpu())
        else:
            print(f"Warning: No attention weights returned for {layer_name}")
            print(f"attn_output shape: {attn_output.shape}")
        
        # 元の関数の戻り値と同じ形式で返す
        if not need_weights:
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
        return []

    print("\n--- アテンションと活性化の可視化 ---")
    model.eval()
    hooks = []
    
    # モデルがコンパイルされているか確認
    compiled_model = hasattr(model, '_orig_mod')
    actual_model = model._orig_mod if compiled_model else model
    print(f"Visualizing on model: {'Compiled' if compiled_model else 'Standard'}")

    # デバッグログを追加
    print("Checking transformer encoder structure...")
    if hasattr(actual_model, 'transformer_encoder'):
        print("Found transformer_encoder")
        if hasattr(actual_model.transformer_encoder, 'layers'):
            print(f"Found {len(actual_model.transformer_encoder.layers)} layers")
            for i, layer_module in enumerate(actual_model.transformer_encoder.layers):
                if hasattr(layer_module, 'self_attn'):
                    print(f"Layer {i} has self_attn")
                    if isinstance(layer_module.self_attn, nn.MultiheadAttention):
                        print(f"Layer {i} self_attn is MultiheadAttention")
                        mha_instance = layer_module.self_attn
                        layer_name = f"transformer_encoder.layers.{i}.self_attn"
                        
                        # アテンションフックの登録
                        def create_attention_hook(layer_idx):
                            def attention_hook(module, input, output):
                                if isinstance(output, tuple) and len(output) == 2:
                                    attn_output, attn_weights = output
                                    if attn_weights is not None:
                                        print(f"Layer {layer_idx} attention weights shape: {attn_weights.shape}")
                                        attention_weights_storage.append(attn_weights.detach().cpu())
                                    else:
                                        print(f"Warning: No attention weights in output for layer {layer_idx}")
                                return output
                            return attention_hook
                        
                        # フックを登録
                        hook_handle = mha_instance.register_forward_hook(create_attention_hook(i))
                        hooks.append(hook_handle)
                        print(f"Registered attention hook for layer {i}")
                    else:
                        print(f"Layer {i} self_attn is not MultiheadAttention")
                else:
                    print(f"Layer {i} does not have self_attn")
        else:
            print("transformer_encoder does not have layers")
    else:
        print("Model does not have transformer_encoder")

    try:
        target_activation_layer_name = 'event_encoder_linear1_out'
        if hasattr(actual_model, 'event_encoder') and \
           isinstance(actual_model.event_encoder, nn.Sequential) and \
           len(actual_model.event_encoder) > 0 and \
           isinstance(actual_model.event_encoder[0], nn.Linear):
            activation_hook_handle = actual_model.event_encoder[0].register_forward_hook(
                get_activation_hook(target_activation_layer_name)
            )
            hooks.append(activation_hook_handle)
            print(f"Registered activation hook for: {target_activation_layer_name}")
        else:
            print(f"Could not find event_encoder[0] (Linear) in model")
    except Exception as e:
        print(f"[警告] event_encoder の活性化フック登録に失敗: {e}")

    # モデルの実行
    with torch.no_grad():
        try:
            _ = model(event_seq_tensor, static_feat_tensor, mask_tensor)
            print("Model forward pass completed successfully")
        except Exception as e:
            print(f"Error during model forward pass: {e}")
            raise

    # フックの解除
    for h in hooks:
        h.remove()
    print("Removed all hooks.")
    
    # アテンションウェイトのデバッグログ
    print(f"Number of attention weights collected: {len(attention_weights_storage)}")
    for i, attn_weights in enumerate(attention_weights_storage):
        print(f"Attention weights {i} shape: {attn_weights.shape}")
        if attn_weights.dim() == 4:  # [batch, num_heads, seq_len, seq_len]
            print(f"Layer {i} first head attention weights (first 3x3):")
            print(attn_weights[0, 0, :3, :3].numpy())
            print(f"Layer {i} attention weights mean: {attn_weights.mean().item():.4f}")
            print(f"Layer {i} attention weights max: {attn_weights.max().item():.4f}")

    seq_len_actual = (~mask_tensor[0]).sum().item() if mask_tensor is not None else event_seq_tensor.shape[1]
    event_step_labels = []
    if seq_len_actual > 0 :
        try: padding_code = EVENT_TYPES["PADDING"] if 'EVENT_TYPES' in globals() and isinstance(EVENT_TYPES, dict) and "PADDING" in EVENT_TYPES else 8.0
        except Exception: padding_code = 8.0
        
        for step_idx in range(seq_len_actual):
            event_type_code = event_seq_tensor[0, step_idx, 0].item()
            event_type_str = "UNK"
            if 'EVENT_TYPES' in globals() and isinstance(EVENT_TYPES, dict):
                for name, code_val in EVENT_TYPES.items():
                    if code_val == int(event_type_code): event_type_str = name; break
            if event_type_code == padding_code : event_type_str = "PAD"
            
            tile_idx_code = event_seq_tensor[0, step_idx, 2].item()
            tile_str = ""
            if tile_idx_code > 0:
                try:
                    actual_tile_index = int(tile_idx_code -1) # game_state で +1 されている前提
                    if 0 <= actual_tile_index < NUM_TILE_TYPES:
                         tile_str = "/" + tile_id_to_string(tile_index_to_id(actual_tile_index))
                    else:
                         tile_str = f"/Ti{actual_tile_index}(inv)"
                except Exception:
                     tile_str = f"/T?{int(tile_idx_code)}"
            event_step_labels.append(f"S{step_idx}:{event_type_str[:3]}{tile_str}")
    
    if attention_weights_storage:
        # print(f"取得したアテンションウェイトの数: {len(attention_weights_storage)}")
        for i, attn_weights_layer_tensor in enumerate(attention_weights_storage):
            # print(f"  Layer {i+1} アテンションウェイト形状: {attn_weights_layer_tensor.shape}")
            if attn_weights_layer_tensor.dim() == 3:
                attn_map = attn_weights_layer_tensor[0].numpy()
            elif attn_weights_layer_tensor.dim() == 4:
                attn_map = attn_weights_layer_tensor[0].mean(dim=0).numpy()
            else:
                # print(f"[警告] Layer {i+1} のアテンションウェイトの形状がプロットに適していません: {attn_weights_layer_tensor.shape}")
                continue

            plot_q_len = min(seq_len_actual, attn_map.shape[0])
            plot_k_len = min(seq_len_actual, attn_map.shape[1])

            if plot_q_len > 0 and plot_k_len > 0:
                plt.figure(figsize=(max(8, plot_k_len // 2.5), max(6, plot_q_len // 3)))
                plt.imshow(attn_map[:plot_q_len, :plot_k_len], cmap='viridis', aspect='auto', vmin=0)
                
                if len(event_step_labels) >= plot_q_len and len(event_step_labels) >= plot_k_len:
                    plt.xticks(np.arange(plot_k_len), event_step_labels[:plot_k_len], rotation=90, fontsize=max(5, 10 - plot_k_len // 12))
                    plt.yticks(np.arange(plot_q_len), event_step_labels[:plot_q_len], fontsize=max(5, 10 - plot_q_len // 12))
                
                plt.title(f'Self-Attention Weights (Layer {i+1})')
                plt.xlabel('Key Positions (Events)')
                plt.ylabel('Query Positions (Events)')
                plt.colorbar(label='Attention Weight')
                plt.tight_layout()
                attn_plot_path = f"attention_layer_{i+1}_visualization.png"
                plt.savefig(attn_plot_path)
                print(f"アテンションヒートマップ (Layer {i+1}) を保存しました: {attn_plot_path}")
                plt.close()
            # else:
                # print(f"[情報] Layer {i+1} アテンションマップのプロットに必要な長さが0です。Q_len={plot_q_len}, K_len={plot_k_len}。スキップします。")
    # else:
        # print("[情報] アテンションウェイトが取得できませんでした。")

    if target_activation_layer_name in hook_outputs:
        activation_map_tensor = hook_outputs[target_activation_layer_name]
        if activation_map_tensor.dim() == 3:
            activation_map = activation_map_tensor[0].cpu().numpy()
            # print(f"取得した活性化 ({target_activation_layer_name}) 形状: {activation_map.shape}")
            plot_seq_len = min(seq_len_actual, activation_map.shape[0])
            if plot_seq_len > 0 :
                plt.figure(figsize=(max(12, plot_seq_len // 2), 5))
                plt.imshow(activation_map[:plot_seq_len, :].T, cmap='viridis', aspect='auto')
                plt.title(f'Activations ({target_activation_layer_name})')
                plt.xlabel('Sequence Step (Actual Events)')
                plt.ylabel('Feature Dimension')
                if len(event_step_labels) >= plot_seq_len:
                     plt.xticks(np.arange(plot_seq_len), event_step_labels[:plot_seq_len], rotation=90, fontsize=max(5, 10 - plot_seq_len // 12))
                plt.colorbar(label='Activation Value')
                plt.tight_layout()
                activation_plot_path = f"activation_{target_activation_layer_name}_visualization.png"
                plt.savefig(activation_plot_path)
                print(f"活性化ヒートマップ ({target_activation_layer_name}) を保存しました: {activation_plot_path}")
                plt.close()
            # else:
                # print(f"[情報] 活性化マップ ({target_activation_layer_name}) のプロットに必要な系列長が0です。Seq_len={plot_seq_len}。スキップします。")
        # else:
            # print(f"[警告] 活性化 ({target_activation_layer_name}) の形状が期待と異なります: {activation_map_tensor.shape}")
    # else:
        # print(f"[情報] 活性化 ({target_activation_layer_name}) が取得できませんでした。")
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
    # ログレベル設定の追加
    parser.add_argument("--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="ログレベルを設定します (デフォルト: INFO)")


    args = parser.parse_args()

    # ログ設定
    # logging モジュールがインポートされていることを確認
    if 'logging' in globals():
        numeric_level = getattr(logging, args.loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {args.loglevel}')
        logging.basicConfig(level=numeric_level, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()])
        logging.info(f"ログレベルを {args.loglevel.upper()} に設定しました。")
    else:
        print("[警告] logging モジュールがインポートされていません。ログ出力は限定的になります。")


    try:
        game_state, tsumo_info, discard_info = reconstruct_game_state_at_tsumo(
            args.xml_file, args.round_index, args.tsumo_count
        )
        player_id = game_state.current_player
        if tsumo_info and player_id != tsumo_info["player"]:
             player_id = tsumo_info["player"]
        elif not tsumo_info:
             logging.error("[エラー] ツモイベント情報が見つかりません。")
             exit(1) # エラー終了

        logging.info("予測のための特徴量を生成中...")
        try:
            event_sequence_instance = game_state.get_event_sequence_features()
            static_features_instance = game_state.get_static_features(player_id)
            event_dim = event_sequence_instance.shape[1]
            static_dim = static_features_instance.shape[0]
            seq_len = event_sequence_instance.shape[0] # MAX_EVENT_HISTORY と一致するはず
            logging.info(f"特徴量次元: イベント次元={event_dim}, 静的次元={static_dim}, シーケンス長={seq_len}")
        except Exception as e:
            logging.error(f"[エラー] 特徴量の生成に失敗しました: {e}", exc_info=True)
            raise

        model = load_trained_model(args.model_path, event_dim, static_dim, seq_len)
        logging.info("打牌を予測中...")
        predicted_index, predicted_prob, all_probabilities = predict_discard(model, game_state, player_id)
        # predicted_index は 0-33 の牌種インデックス
        predicted_tile_str = tile_id_to_string(tile_index_to_id(predicted_index)) # 牌種インデックスから代表IDに変換して表示
        logging.info("予測完了。")


        actual_discard_str = "N/A (局終了？)"
        if discard_info:
            actual_discard_str = tile_id_to_string(discard_info["pai"]) # discard_info["pai"] は牌ID
            if discard_info["tsumogiri"]: actual_discard_str += "*"

        print("\n" + "="*15 + " Transformer 予測テスト " + "="*15) # 見出しを分かりやすく
        print(f"--- 対象局面 (牌譜: {os.path.basename(args.xml_file)}, 局: {args.round_index}, ツモ巡: {args.tsumo_count}) ---")
        round_str, my_wind_str = get_wind_str(game_state.round_num_wind, player_id, game_state.dealer)
        honba_str = f"{game_state.honba}本場"
        kyotaku_str = f"({game_state.kyotaku}供託)" if game_state.kyotaku > 0 else ""
        print(f"局況: {round_str} {honba_str} {kyotaku_str} / プレイヤー: P{player_id} ({my_wind_str}家)")
        tsumo_pai_str = tile_id_to_string(tsumo_info['pai']) if tsumo_info else "不明"
        print(f"ツモ牌: {tsumo_pai_str}")
        print(f"現在の巡目 (GameState準拠): {game_state.junme:.1f}")
        print(f"ドラ表示: {' '.join([tile_id_to_string(t) for t in game_state.dora_indicators])}")
        print(f"各家点数: {[f'P{i}:{s}' for i, s in enumerate(game_state.current_scores)]}")
        
        print("--- 現在の盤面 ---")
        print("手牌 (ツモ後):")
        for p in range(NUM_PLAYERS):
            hand_str = format_hand(game_state.player_hands[p])
            reach_indicator = "*" if game_state.player_reach_status[p] == 2 else ("(宣)" if game_state.player_reach_status[p] == 1 else "")
            print(f"  P{p}{reach_indicator}: {hand_str}")
        print("捨て牌:")
        for p in range(NUM_PLAYERS):
            discard_str = format_discards(game_state.player_discards[p])
            print(f"  P{p}: {discard_str}")
        print("副露:")
        for p in range(NUM_PLAYERS):
            meld_str = format_melds(game_state.player_melds[p])
            print(f"  P{p}: {meld_str}")
        print("-" * 20)
        print(f"予測された捨て牌 (牌種): {predicted_tile_str}")
        print(f"予測された牌種インデックス: {predicted_index}")
        print(f"  (確率: {predicted_prob:.4f})")
        print(f"実際の捨て牌: {actual_discard_str}")
        print("-" * 20)
        top_n = 5
        indices_sorted = np.argsort(all_probabilities)[::-1] # 降順ソート
        print(f"予測確率 Top {top_n}:")
        for i_loop in range(min(top_n, len(indices_sorted))):
            idx = indices_sorted[i_loop] # idx は 0-33 の牌種インデックス
            print(f"idx: {idx}")
            tile_str = tile_index_to_str(idx)  # 牌種インデックスから天鳳パイプ形式文字列に変換
            print(f"tile_str: {tile_str}")
            prob = all_probabilities[idx]
            if 0 <= idx < NUM_TILE_TYPES:
                tile_str = tile_id_to_string(tile_index_to_id(idx))  # 牌種インデックスから天鳳パイプ形式文字列に変換
                print(f"  {i_loop+1}. {tile_str.ljust(4)} ({prob:.4f})")
            else:
                print(f"  {i_loop+1}. Index:{idx} (無効) ({prob:.4f})")

        # SHAP説明と可視化
        if shap_available:
            logging.info("\nSHAP説明のための背景データをロード中...")
            try:
                background_data_tuple = (np.array([]), np.array([])) # 初期化
                if not os.path.exists(args.background_data_path):
                    logging.warning(f"[警告] SHAP用の背景データファイルが見つかりません: {args.background_data_path}。説明をスキップします。")
                else:
                    try:
                        with h5py.File(args.background_data_path, "r", swmr=True) as hf:
                            if "labels" not in hf or "sequences" not in hf or "static_features" not in hf :
                                logging.warning(f"[警告] 背景データファイル {args.background_data_path} に必要なデータセットがありません。説明をスキップします。")
                            else:
                                total_bg_samples = hf["labels"].shape[0]
                                if total_bg_samples == 0:
                                    logging.warning("[警告] 背景データファイルにサンプルがありません。説明をスキップします。")
                                else:
                                    n_bg = min(args.background_samples, total_bg_samples)
                                    if n_bg <= 0:
                                        logging.warning("[警告] 背景サンプル数が0以下です。説明をスキップします。")
                                    else:
                                        indices_to_load = np.random.choice(total_bg_samples, n_bg, replace=False)
                                        indices_to_load.sort()
                                        
                                        bg_sequences = hf["sequences"][indices_to_load, ...]
                                        bg_static_features = hf["static_features"][indices_to_load, ...]
                                        background_data_tuple = (bg_sequences, bg_static_features)
                                        logging.info(f"{len(bg_sequences)} 件の背景データをロードしました。")
                    except Exception as bg_load_e:
                        logging.warning(f"[警告] 背景データのロード中にエラーが発生しました: {bg_load_e}。説明をスキップします。")
                
                if background_data_tuple[0].shape[0] > 0:
                    instance_to_explain_tuple = (event_sequence_instance, static_features_instance, None)
                    logging.info("SHAP特徴量名を生成中...")
                    feature_names_for_shap = generate_feature_names(event_dim, static_dim, seq_len)
                    _ = explain_prediction_with_shap(
                        model, background_data_tuple, instance_to_explain_tuple,
                        feature_names_for_shap, predicted_index,
                        n_shap_samples=args.shap_samples
                    )
            except Exception as shap_e:
                logging.error(f"\n[エラー] SHAP説明の生成中にエラーが発生しました: {shap_e}", exc_info=True)
        else:
            print("\nSHAPライブラリまたはMatplotlibが利用できないため、SHAP説明と一部の可視化は生成されません。")

        # アテンションと活性化の可視化
        if shap_available: # matplotlib の可用性で判断
            if args.xml_file and args.round_index and args.tsumo_count:
                try:
                    event_seq_tensor_for_viz = torch.tensor(event_sequence_instance, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    static_feat_tensor_for_viz = torch.tensor(static_features_instance, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    try:
                        padding_code_viz = EVENT_TYPES["PADDING"] if 'EVENT_TYPES' in globals() and isinstance(EVENT_TYPES, dict) and "PADDING" in EVENT_TYPES else 8.0
                    except Exception: padding_code_viz = 8.0
                    mask_tensor_for_viz = (event_seq_tensor_for_viz[:, :, 0] == padding_code_viz).to(DEVICE)
                    
                    logging.info("可視化用特徴量名を生成中...")
                    feature_names_for_viz = generate_feature_names(event_dim, static_dim, seq_len)
                    attn_weights_list = visualize_attention_and_activations(model, event_seq_tensor_for_viz, static_feat_tensor_for_viz, mask_tensor_for_viz, feature_names_for_viz)
                    if attn_weights_list:
                        logging.info(f"\n[INFO] アテンションウェイトが {len(attn_weights_list)} レイヤー分取得/プロットされました。")
                except Exception as viz_e:
                    logging.error(f"\n[エラー] アテンション/活性化の可視化中にエラーが発生しました: {viz_e}", exc_info=True)
        else:
            print("\nMatplotlibが利用できないため、アテンション/活性化の可視化はスキップされます。")


    except FileNotFoundError as e: logging.error(f"エラー: ファイルが見つかりません - {e}")
    except ValueError as e: logging.error(f"エラー: 値が不正です - {e}", exc_info=True)
    except ImportError as e: logging.error(f"エラー: インポートに失敗しました - {e}")
    except AttributeError as e: logging.error(f"エラー: 属性エラー（クラス定義やメソッド呼び出しを確認） - {e}", exc_info=True)
    except Exception as e:
        logging.error(f"予期せぬエラーが発生しました: {e}", exc_info=True)