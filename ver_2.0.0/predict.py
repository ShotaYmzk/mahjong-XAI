import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import logging
import h5py
import matplotlib.pyplot as plt
from collections import OrderedDict # ★★★★★★★★★★★★★★★★★★★★★★★ 追加 ★★★★★★★★★★★★★★★★★★★★★★★

# SHAPのインポート（オプション）
try:
    import shap
    shap_available = True
except ImportError:
    print("[警告] `shap` ライブラリが見つかりません。SHAP説明機能はスキップされます。")
    shap_available = False

# --- プロジェクトモジュールのインポート ---
try:
    # ver_2.0.0 の game_state をインポート
    from game_state import GameState, NUM_TILE_TYPES, MAX_EVENT_HISTORY, STATIC_FEATURE_DIM, EVENT_TYPES, NUM_PLAYERS
    from full_mahjong_parser import parse_full_mahjong_log
    from tile_utils import tile_id_to_string, tile_index_to_id, tile_id_to_index
    from naki_utils import decode_naki
    print("プロジェクトモジュール (ver_2.0.0) を正常にインポートしました。")
except ImportError as e:
    print(f"[エラー] プロジェクトモジュールのインポートに失敗しました: {e}")
    print("必要なファイル (game_state.py, tile_utils.pyなど) が同じディレクトリにあるか確認してください。")
    exit(1)

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- グローバル変数 ---
# train2.py から DATA_HDF5_PATH を参照
DATA_HDF5_PATH = "/home/ubuntu/Documents/mahjong-XAI/ver_1.1.10/training_data/mahjong_imitation_data_v1110.hdf5"
DEFAULT_MODEL_PATH = "/home/ubuntu/Documents/mahjong-XAI/ver_1.1.11/trained_model/mahjong_transformer_v1111_large_compiled.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
logging.info(f"使用デバイス: {DEVICE}")

# フックの結果を保持するためのグローバルストレージ
activations_storage = {}

def get_activation_hook(name):
    """フック関数を生成するクロージャ"""
    def hook(model, input, output):
        # output_headの最初のLinear層への入力(input)が'combined'ベクトル
        activations_storage[name] = input[0].detach().cpu().numpy()
    return hook

# --- モデル定義 (train2.py との互換性を維持) ---
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_EVENT_HISTORY):
        super().__init__()
        if d_model % 2 != 0: raise ValueError("d_model must be divisible by 2 for RoPE.")
        self.dim_half = d_model // 2
        freqs = 1.0 / (10000 ** (torch.arange(0, self.dim_half).float() / self.dim_half))
        self.register_buffer('freqs', freqs)
        pos_seq = torch.arange(max_len).float()
        self.register_buffer('pos_seq', pos_seq)

    def forward(self, x):
        seq_len = x.shape[1]
        positions = self.pos_seq[:seq_len].unsqueeze(0).to(x.device)
        angles = positions.unsqueeze(-1) * self.freqs.unsqueeze(0).to(x.device)
        sin_angles, cos_angles = torch.sin(angles), torch.cos(angles)
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        x_even_rotated = x_even * cos_angles - x_odd * sin_angles
        x_odd_rotated = x_even * sin_angles + x_odd * cos_angles
        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2], x_rotated[..., 1::2] = x_even_rotated, x_odd_rotated
        return x_rotated

class CustomTransformerEncoderLayerWithAttention(nn.TransformerEncoderLayer):
    """アテンションウェイトを返すカスタムTransformerEncoderLayer"""
    def __init__(self, *args, **kwargs):
        # is_causal引数はPyTorch 2.0以降で導入されたため、古いバージョンとの互換性のために削除
        kwargs.pop('is_causal', None)
        super().__init__(*args, **kwargs)
        self.attn_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None): # is_causalを引数から削除
        x = src
        # need_weights=True は self_attn の forward メソッドの引数
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
    """アテンションとアクティベーションを取得できるように修正したモデル"""
    def __init__(self, event_feature_dim, static_feature_dim, d_model=256, nhead=4, d_hid=1024, nlayers=4, dropout=0.1, activation='relu', output_dim=NUM_TILE_TYPES):
        super().__init__()
        self.d_model = d_model
        # game_state.pyから取得したイベント特徴量の次元数
        self.event_feature_dim = event_feature_dim 
        
        self.event_encoder = nn.Sequential(
            nn.Linear(self.event_feature_dim, d_model),
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

        # アクティベーション抽出用のフックを登録
        self.output_head[0].register_forward_hook(get_activation_hook('combined_vector'))

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                gain = nn.init.calculate_gain('relu')
                nn.init.xavier_normal_(p, gain=gain)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, event_seq, static_feat, attention_mask=None, return_attention=False):
        # 入力次元がモデルの期待と異なる場合、エラーを出すか、適切に処理する
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
        combined = torch.cat([context_vector, static_encoded], dim=1)
        output = self.output_head(combined)
        if return_attention:
            return output, attention_weights_all_layers
        return output

# --- ヘルパー関数 ---
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
        tiles_str = " ".join([tile_id_to_string(t) for t in sorted(m_tiles)])
        from_str = f" P{from_who_abs}" if from_who_abs != -1 and m_type not in ["暗槓", "加槓"] else ""
        meld_strs.append(f"{m_type}[{tiles_str}]{from_str}")
    return " / ".join(meld_strs)

def get_wind_str(round_num_wind, player_id, dealer):
    round_winds = ["東", "南", "西", "北"]; player_winds = ["東", "南", "西", "北"]
    round_wind_idx = round_num_wind // NUM_PLAYERS; kyoku_num = (round_num_wind % NUM_PLAYERS) + 1
    my_wind_idx = (player_id - dealer + NUM_PLAYERS) % NUM_PLAYERS
    return f"{round_winds[round_wind_idx]}{kyoku_num}局", player_winds[my_wind_idx]

# --- モデルロード関数 (修正版) ---
def load_trained_model(model_path, event_dim, static_dim):
    """学習済みモデルをロードし、層の名前の不一致を修正する"""
    logging.info(f"モデルをロード中: {model_path}")
    logging.info(f"モデルパラメータ: event_dim={event_dim}, static_dim={static_dim}")
    
    # アテンション対応モデルを作成
    model = MahjongTransformerV2WithAttention(
        event_feature_dim=event_dim,
        static_feature_dim=static_dim,
        d_model=256, nhead=4, d_hid=1024, nlayers=4, dropout=0.1
    ).to(DEVICE)

    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        state_dict_to_load = checkpoint.get('model_state_dict', checkpoint)
        
        # ★★★★★★★★★★★★★★★★★★★★★★★ ここからが修正箇所 ★★★★★★★★★★★★★★★★★★★★★★★
        # 新しい state_dict を作成して、キーの名前を書き換える
        new_state_dict = OrderedDict()
        for k, v in state_dict_to_load.items():
            # 'transformer_encoder.layers.' を 'encoder_layers.' に置換
            if k.startswith('transformer_encoder.layers.'):
                new_key = k.replace('transformer_encoder.layers.', 'encoder_layers.', 1)
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        
        # event_encoder の次元チェック（念のため）
        model_event_dim = model.event_encoder[0].in_features
        if 'event_encoder.0.weight' in new_state_dict:
            ckpt_event_dim = new_state_dict['event_encoder.0.weight'].shape[1]
            if model_event_dim != ckpt_event_dim:
                logging.warning(f"Event feature dimension mismatch! Model expects {model_event_dim}, checkpoint has {ckpt_event_dim}.")
                logging.warning("Attempting to load weights by ignoring the event encoder.")
                keys_to_remove = [k for k in new_state_dict if k.startswith('event_encoder')]
                for k in keys_to_remove:
                    del new_state_dict[k]
                model.load_state_dict(new_state_dict, strict=False)
            else:
                model.load_state_dict(new_state_dict)
        else:
             model.load_state_dict(new_state_dict)

        logging.info("モデルの重みを正常にロードしました（キー名修正済み）。")
        # ★★★★★★★★★★★★★★★★★★★★★★★ ここまでが修正箇所 ★★★★★★★★★★★★★★★★★★★★★★★

    except Exception as e:
        logging.error(f"モデルのロードに失敗しました: {e}", exc_info=True)
        raise e

    model.eval()
    return model

# --- 局面復元関数 ---
def reconstruct_game_state_at_tsumo(xml_path, target_round_index, target_tsumo_event_count_in_round):
    logging.info(f"牌譜ファイル {xml_path} を解析中...")
    try:
        meta, rounds_data = parse_full_mahjong_log(xml_path)
    except Exception as e:
        logging.error(f"牌譜ファイルの解析エラー: {e}")
        raise

    if not (1 <= target_round_index <= len(rounds_data)):
        raise ValueError(f"無効な局インデックス: {target_round_index}")

    round_data = rounds_data[target_round_index - 1]
    game_state = GameState()
    game_state.init_round(round_data)

    current_tsumo_count = 0
    target_tsumo_event_info = None
    actual_discard_event_info = None
    events = round_data.get("events", [])

    for i, event_xml in enumerate(events):
        tag = event_xml["tag"]
        is_tsumo = False
        for t_tag, p_id in GameState.TSUMO_TAGS.items():
            if tag.startswith(t_tag) and tag[1:].isdigit():
                is_tsumo = True
                tsumo_player_id, tsumo_pai_id = p_id, int(tag[1:])
                break
        
        if is_tsumo:
            current_tsumo_count += 1
            if current_tsumo_count == target_tsumo_event_count_in_round:
                logging.info(f"ターゲットのツモ ({current_tsumo_count}回目) を発見。")
                target_tsumo_event_info = {"player": tsumo_player_id, "pai": tsumo_pai_id}
                game_state.process_tsumo(tsumo_player_id, tsumo_pai_id)
                # 次のイベントが打牌か確認
                if i + 1 < len(events):
                    next_event = events[i+1]
                    for d_tag, p_id_next in GameState.DISCARD_TAGS.items():
                        if next_event['tag'].startswith(d_tag) and next_event['tag'][1:].isdigit() and p_id_next == tsumo_player_id:
                            discard_pai_id = int(next_event['tag'][1:])
                            tsumogiri = next_event['tag'][0].islower()
                            actual_discard_event_info = {"player": p_id_next, "pai": discard_pai_id, "tsumogiri": tsumogiri}
                            break
                return game_state, target_tsumo_event_info, actual_discard_event_info
            
        # ターゲットに到達するまで状態を進める
        game_state.process_event(event_xml)

    raise ValueError(f"指定されたツモ回数 ({target_tsumo_event_count_in_round}) に到達できませんでした。")

# --- 打牌予測関数 ---
def predict_discard(model, game_state: GameState, player_id: int):
    event_sequence = game_state.get_event_sequence_features()
    static_features = game_state.get_static_features(player_id)

    seq_tensor = torch.tensor(event_sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    static_tensor = torch.tensor(static_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    padding_code_float = float(EVENT_TYPES["PADDING"])
    mask_tensor = (seq_tensor[:, :, 0] == padding_code_float).to(DEVICE)

    # グローバルなストレージをクリア
    activations_storage.clear()
    
    with torch.no_grad():
        outputs, collected_attention_weights = model(seq_tensor, static_tensor, mask_tensor, return_attention=True)
        probabilities = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    # フックによって格納されたアクティベーションを取得
    activation_vector = activations_storage.get('combined_vector', None)
    if activation_vector is not None:
        activation_vector = activation_vector.squeeze(0) # バッチ次元を削除

    valid_discard_indices = game_state.get_valid_discard_options(player_id)
    best_prob, best_index = -1.0, -1
    if valid_discard_indices:
        for index in valid_discard_indices:
            if 0 <= index < NUM_TILE_TYPES and probabilities[index] > best_prob:
                best_prob = probabilities[index]
                best_index = index
    else: # 有効な打牌がない場合（リーチ後など）
        # リーチ後はツモ切り一択なので、その牌の確率を返す
        drawn_tile_id = game_state.player_hands[player_id][-1]
        best_index = tile_id_to_index(drawn_tile_id)
        best_prob = probabilities[best_index] if best_index != -1 else 0.0

    return best_index, best_prob, probabilities, collected_attention_weights, activation_vector

# --- メイン処理 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学習済み麻雀Transformerモデルを使って打牌を予測し、説明を生成します。")
    parser.add_argument("xml_file", help="天鳳形式のXML牌譜ファイルへのパス")
    parser.add_argument("round_index", type=int, help="対象局のインデックス (1から開始)")
    parser.add_argument("tsumo_count", type=int, help="対象局内でのツモ回数 (1から開始)")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help=f"学習済みモデル (.pth) ファイルへのパス")
    args = parser.parse_args()

    try:
        # 1. 局面復元
        game_state, tsumo_info, discard_info = reconstruct_game_state_at_tsumo(args.xml_file, args.round_index, args.tsumo_count)
        player_id = game_state.current_player

        # 2. 特徴量次元の取得とモデルのロード
        event_seq_dummy = game_state.get_event_sequence_features()
        EVENT_FEATURE_DIM = event_seq_dummy.shape[1]
        
        model = load_trained_model(args.model_path, EVENT_FEATURE_DIM, STATIC_FEATURE_DIM)

        # 3. 打牌予測と説明根拠の取得
        logging.info("打牌を予測中...")
        predicted_index, predicted_prob, all_probabilities, attention_weights, activation_vector = predict_discard(
            model, game_state, player_id
        )
        predicted_tile_str = tile_id_to_string(tile_index_to_id(predicted_index))
        logging.info("予測完了。")

        # 4. 結果表示
        actual_discard_str = "N/A"
        if discard_info:
            actual_discard_str = tile_id_to_string(discard_info["pai"])
            if discard_info["tsumogiri"]: actual_discard_str += "*"

        print("\n=== Transformer 予測テスト (ver_2.0.0) ===")
        print(f"--- 対象局面 (牌譜: {os.path.basename(args.xml_file)}, 局: {args.round_index}, ツモ巡: {args.tsumo_count}) ---")
        round_str, my_wind_str = get_wind_str(game_state.round_num_wind, player_id, game_state.dealer)
        print(f"局況: {round_str} {game_state.honba}本場 ({game_state.kyotaku}供託) / P{player_id} ({my_wind_str}家)")
        print(f"ツモ牌: {tile_id_to_string(tsumo_info['pai']) if tsumo_info else '不明'}")
        print(f"ドラ表示: {' '.join([tile_id_to_string(t) for t in game_state.dora_indicators])}")
        print("--- 現在の盤面 ---")
        for p in range(NUM_PLAYERS):
            hand_str = format_hand(game_state.player_hands[p])
            reach_indicator = "*" if game_state.player_reach_status[p] == 2 else ""
            print(f"  P{p}{reach_indicator}: {hand_str}")
        print("捨て牌:")
        for p in range(NUM_PLAYERS): print(f"  P{p}: {format_discards(game_state.player_discards[p])}")
        print("副露:")
        for p in range(NUM_PLAYERS): print(f"  P{p}: {format_melds(game_state.player_melds[p])}")
        
        print("-" * 20)
        print(f"AIの予測: 打 {predicted_tile_str} (確率: {predicted_prob:.4f})")
        print(f"実際の打牌: {actual_discard_str}")
        print("-" * 20)

        # 5. 予測根拠の表示
        print("--- 予測根拠 ---")
        if activation_vector is not None:
            print(f"  ✅ 局面アクティベーションベクトル (Activation) を取得しました。 (次元: {activation_vector.shape})")
            print(f"     - L2ノルム: {np.linalg.norm(activation_vector):.4f}")
            print(f"     - 平均値: {np.mean(activation_vector):.4f}, 標準偏差: {np.std(activation_vector):.4f}")
        else:
            print("  ❌ 局面アクティベーションベクトルを取得できませんでした。")

        if attention_weights:
            print(f"  ✅ セルフアテンションウェイト (Attention) を取得しました。 ({len(attention_weights)}層分)")
            last_layer_attention = attention_weights[-1].squeeze(0).cpu().numpy()
            avg_attention_per_query = np.mean(last_layer_attention, axis=1)
            most_attended_step = np.argmax(avg_attention_per_query)
            print(f"     - 最終層で最も注目されたイベントステップ: {most_attended_step}")
        else:
            print("  ❌ セルフアテンションウェイトを取得できませんでした。")

    except Exception as e:
        logging.error(f"予期せぬエラーが発生しました: {e}", exc_info=True)