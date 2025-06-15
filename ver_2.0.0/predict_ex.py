import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import logging
import h5py
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
import joblib # モデルのロード用

# --- プロジェクトモジュールのインポート ---
try:
    from game_state import GameState, NUM_TILE_TYPES, MAX_EVENT_HISTORY, STATIC_FEATURE_DIM, EVENT_TYPES, NUM_PLAYERS
    from full_mahjong_parser import parse_full_mahjong_log
    from tile_utils import tile_id_to_string, tile_index_to_id, tile_index_to_str, tile_id_to_index
    from naki_utils import decode_naki
    print("プロジェクトモジュール (ver_2.0.0) を正常にインポートしました。")
except ImportError as e:
    print(f"[エラー] プロジェクトモジュールのインポートに失敗しました: {e}")
    exit(1)

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- グローバル変数 ---
DEFAULT_MODEL_PATH = "/home/ubuntu/Documents/mahjong-XAI/ver_1.1.11/trained_model/mahjong_transformer_v1111_large_compiled.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
logging.info(f"使用デバイス: {DEVICE}")

# 説明モデルのグローバル変数
pca_model = None
kmeans_model = None
concept_labels = None
activations_storage = {}

def get_activation_hook(name):
    def hook(model, input, output):
        activations_storage[name] = input[0].detach().cpu().numpy()
    return hook

# --- モデル定義 ---
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
    def __init__(self, *args, **kwargs):
        kwargs.pop('is_causal', None)
        super().__init__(*args, **kwargs)
        self.attn_weights = None
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        attn_output, self.attn_weights = self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=True, average_attn_weights=True)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))
        x = self.norm2(x)
        return x

class MahjongTransformerV2WithAttention(nn.Module):
    def __init__(self, event_feature_dim, static_feature_dim, d_model=256, nhead=4, d_hid=1024, nlayers=4, dropout=0.1, activation='relu', output_dim=NUM_TILE_TYPES):
        super().__init__()
        self.d_model, self.event_feature_dim = d_model, event_feature_dim
        self.event_encoder = nn.Sequential(nn.Linear(self.event_feature_dim, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout))
        self.pos_encoder = RotaryPositionalEncoding(d_model, max_len=MAX_EVENT_HISTORY)
        self.encoder_layers = nn.ModuleList([CustomTransformerEncoderLayerWithAttention(d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout, activation=activation, batch_first=True, norm_first=True) for _ in range(nlayers)])
        self.static_encoder = nn.Sequential(nn.Linear(static_feature_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        self.attention_pool = nn.Sequential(nn.Linear(d_model, 1), nn.Softmax(dim=1))
        self.output_head = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model // 2), nn.LayerNorm(d_model // 2), nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(d_model // 2, output_dim))
        self._init_weights()
        self.output_head[0].register_forward_hook(get_activation_hook('combined_vector'))
    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1: nn.init.xavier_normal_(p, gain=nn.init.calculate_gain('relu'))
            elif 'bias' in name: nn.init.zeros_(p)
    
    def forward(self, event_seq, static_feat, attention_mask=None, return_attention=False):
        if event_seq.shape[-1] != self.event_feature_dim: raise ValueError(f"Input event feature dimension mismatch! Got {event_seq.shape[-1]}, expected {self.event_feature_dim}")
        event_encoded = self.event_encoder(event_seq)
        pos_encoded = self.pos_encoder(event_encoded)
        x = pos_encoded
        attention_weights_all_layers = []
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=attention_mask)
            if return_attention and hasattr(layer, 'attn_weights') and layer.attn_weights is not None: attention_weights_all_layers.append(layer.attn_weights)
        transformer_output = x
        attn_weights_pool = self.attention_pool(transformer_output)
        if attention_mask is not None: attn_weights_pool = attn_weights_pool.masked_fill(attention_mask.unsqueeze(-1), 0.0)
        context_vector = torch.sum(attn_weights_pool * transformer_output, dim=1)
        static_encoded = self.static_encoder(static_feat)
        if context_vector.dim() == 1: context_vector = context_vector.unsqueeze(0)
        if static_encoded.dim() == 1: static_encoded = static_encoded.unsqueeze(0)
        combined = torch.cat([context_vector, static_encoded], dim=1)
        output = self.output_head(combined)
        return (output, attention_weights_all_layers) if return_attention else output

# --- ヘルパー関数 ---
def format_hand(hand_ids): return " ".join([tile_id_to_string(t) for t in sorted(hand_ids, key=lambda t: (tile_id_to_index(t), t))]) if hand_ids else "なし"
def format_discards(discard_list): return " ".join([f"{tile_id_to_string(t)}{'*' if ts else ''}" for t, ts in discard_list]) if discard_list else "なし"
def format_melds(meld_list_dicts):
    if not meld_list_dicts: return "なし"
    meld_strs = []
    for m_info in meld_list_dicts:
        m_type, m_tiles, from_who = m_info.get('type', '?'), m_info.get('tiles', []), m_info.get('from_who', -1)
        tiles_str = " ".join([tile_id_to_string(t) for t in sorted(m_tiles)])
        from_str = f" P{from_who}" if from_who != -1 and m_type not in ["暗槓", "加槓"] else ""
        meld_strs.append(f"{m_type}[{tiles_str}]{from_str}")
    return " / ".join(meld_strs)
def get_wind_str(round_num, player_id, dealer):
    round_winds, player_winds = ["東", "南", "西", "北"], ["東", "南", "西", "北"]
    return f"{round_winds[round_num//4]}{round_num%4+1}局", player_winds[(player_id-dealer+4)%4]

# --- モデルロード関数 ---
def load_trained_model(model_path, event_dim, static_dim):
    model = MahjongTransformerV2WithAttention(event_feature_dim=event_dim, static_feature_dim=static_dim).to(DEVICE)
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        state_dict_to_load = checkpoint.get('model_state_dict', checkpoint)
        new_state_dict = OrderedDict()
        for k, v in state_dict_to_load.items():
            new_key = k.replace('transformer_encoder.layers.', 'encoder_layers.', 1) if k.startswith('transformer_encoder.layers.') else k
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict, strict=True)
        logging.info("モデルの重みを正常にロードしました（キー名修正済み）。")
    except Exception as e:
        logging.error(f"モデルのロードに失敗: {e}", exc_info=True)
        raise e
    model.eval()
    return model

# --- 局面復元関数 ---
def reconstruct_game_state_at_tsumo(xml_path, target_round_index, target_tsumo_count):
    meta, rounds_data = parse_full_mahjong_log(xml_path)
    if not (1 <= target_round_index <= len(rounds_data)): raise ValueError("無効な局インデックス")
    round_data = rounds_data[target_round_index - 1]
    game_state = GameState()
    game_state.init_round(round_data)
    current_tsumo_count, tsumo_info, discard_info = 0, None, None
    events = round_data.get("events", [])
    for i, event_xml in enumerate(events):
        tag = event_xml["tag"]
        is_tsumo = any(tag.startswith(t) and tag[1:].isdigit() for t, p in GameState.TSUMO_TAGS.items())
        if is_tsumo:
            current_tsumo_count += 1
            if current_tsumo_count == target_tsumo_count:
                tsumo_player_id, tsumo_pai_id = [(p, int(tag[1:])) for t, p in GameState.TSUMO_TAGS.items() if tag.startswith(t)][0]
                tsumo_info = {"player": tsumo_player_id, "pai": tsumo_pai_id}
                game_state.process_tsumo(tsumo_player_id, tsumo_pai_id)
                if i + 1 < len(events):
                    next_event = events[i+1]
                    for d_tag, p_id_next in GameState.DISCARD_TAGS.items():
                        if next_event['tag'].startswith(d_tag) and next_event['tag'][1:].isdigit() and p_id_next == tsumo_player_id:
                            discard_info = {"player": p_id_next, "pai": int(next_event['tag'][1:]), "tsumogiri": next_event['tag'][0].islower()}
                            break
                return game_state, tsumo_info, discard_info
        game_state.process_event(event_xml)
    raise ValueError("指定されたツモ回数に到達できませんでした。")

# --- 説明生成関数 ---
def get_explanation_models():
    try:
        pca = joblib.load('pca_model_2.joblib')
        kmeans = joblib.load('kmeans_model_2.joblib')
        labels = joblib.load('concept_labels_2.joblib')
        logging.info("説明モデル（PCA, k-means, Labels）を正常にロードしました。")
        return pca, kmeans, labels
    except FileNotFoundError as e:
        logging.error(f"説明モデルファイルが見つかりません: {e}")
        logging.error("analyze_clusters2.py を先に実行して、モデルファイルを生成してください。")
        return None, None, None

def analyze_and_explain(probabilities, attention_weights, activation_vector):
    if activation_vector is None:
        return {"error": "アクティベーションベクトルが取得できませんでした。"}
    activation_pca = pca_model.transform(activation_vector.reshape(1, -1))
    cluster_id = kmeans_model.predict(activation_pca)[0]
    concept_names = concept_labels.get(cluster_id, ["Unknown"])
    last_layer_attention = attention_weights[-1].squeeze(0).cpu().numpy()
    attention_score = np.sum(last_layer_attention, axis=0)
    attention_score_normalized = attention_score / np.sum(attention_score)
    most_attended_step = np.argmax(attention_score_normalized)
    top_n = 3
    sorted_indices = np.argsort(probabilities)[::-1]
    top_discards = [f"打{tile_index_to_str(idx)}({probabilities[idx]:.1%})" for idx in sorted_indices[:top_n]]
    quantitative_summary = " / ".join(top_discards)
    predicted_tile_str = tile_index_to_str(sorted_indices[0])
    concept_str = " & ".join(concept_names)
    qualitative_reason = f"AIは現在「{concept_str}」を重視しています。"
    if "Safety" in concept_names:
        qualitative_reason += f" 安全性を最優先し、放銃リスクの低い {predicted_tile_str} を選択しました。"
    elif "Speed" in concept_names:
        qualitative_reason += f" 最速聴牌を目指し、手が進む {predicted_tile_str} を選択しました。"
    elif "Value" in concept_names:
        qualitative_reason += f" 打点上昇を狙い、価値の高い {predicted_tile_str} を残す選択をしました。"
    else:
        qualitative_reason += f" バランスを考慮し、最も確率の高い {predicted_tile_str} を選択しました。"
    second_best_idx = sorted_indices[1]
    second_best_tile = tile_index_to_str(second_best_idx)
    prob_diff = probabilities[sorted_indices[0]] - probabilities[second_best_idx]
    comparative_summary = f"次点の「打{second_best_tile}」と比較して、{prob_diff:.1%}ほど優位と判断しました。"
    return {
        "quantitative": quantitative_summary,
        "qualitative": qualitative_reason,
        "comparative": comparative_summary,
        "concept": f"{concept_str} (Cluster {cluster_id})",
        "attention_focus": f"Step {most_attended_step} (Attention: {attention_score_normalized[most_attended_step]:.1%})"
    }

# --- メイン処理 (最終版) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学習済み麻雀Transformerモデルで打牌を予測し、その根拠を説明します。")
    parser.add_argument("xml_file", help="天鳳形式のXML牌譜ファイルへのパス")
    parser.add_argument("round_index", type=int, help="対象局のインデックス (1から開始)")
    parser.add_argument("tsumo_count", type=int, help="対象局内でのツモ回数 (1から開始)")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help=f"学習済みモデル (.pth) ファイルへのパス")
    args = parser.parse_args()

    try:
        # 1. モデルロード
        event_dim_dummy = GameState().get_event_sequence_features().shape[1]
        model = load_trained_model(args.model_path, event_dim_dummy, STATIC_FEATURE_DIM)
        pca_model, kmeans_model, concept_labels = get_explanation_models()
        if not all([pca_model, kmeans_model, concept_labels]): exit(1)

        # 2. 局面復元
        game_state, tsumo_info, discard_info = reconstruct_game_state_at_tsumo(args.xml_file, args.round_index, args.tsumo_count)
        player_id = game_state.current_player

        # 3. 打牌予測
        activations_storage.clear()
        with torch.no_grad():
            event_seq = torch.tensor(game_state.get_event_sequence_features(), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            static_feat = torch.tensor(game_state.get_static_features(player_id), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            mask = (event_seq[:, :, 0] == float(EVENT_TYPES["PADDING"])).to(DEVICE)
            outputs, attention_weights = model(event_seq, static_feat, mask, return_attention=True)
            probabilities = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
        
        activation_vector = activations_storage.get('combined_vector')
        
        # 4. 結果表示
        predicted_index = np.argmax(probabilities)
        predicted_tile_str = tile_index_to_str(predicted_index)
        actual_discard_str = "N/A"
        if discard_info:
            actual_discard_str = tile_index_to_str(tile_id_to_index(discard_info["pai"]))
            if discard_info["tsumogiri"]: actual_discard_str += "*"

        print("\n" + "="*20 + " 局面情報 " + "="*20)
        round_str, my_wind_str = get_wind_str(game_state.round_num_wind, player_id, game_state.dealer)
        print(f"局況: {round_str} {game_state.honba}本場 ({game_state.kyotaku}供託) / P{player_id} ({my_wind_str}家)")
        print(f"ツモ牌: {tile_id_to_string(tsumo_info['pai']) if tsumo_info else '不明'}")
        print(f"ドラ表示: {' '.join([tile_id_to_string(t) for t in game_state.dora_indicators])}")
        print(f"手牌 (ツモ後): {format_hand(game_state.player_hands[player_id])}")
        
        print("\n" + "="*20 + " AIの判断 " + "="*20)
        print(f"AIの予測: 打 {predicted_tile_str} (実際の打牌: {actual_discard_str})")

        # 5. 打牌根拠の説明
        explanation = analyze_and_explain(probabilities, attention_weights, activation_vector)
        if "error" in explanation:
            print(f"説明生成エラー: {explanation['error']}")
        else:
            print("\n--- 思考のサマリー ---")
            print(f"  AIの思考タイプ: {explanation['concept']}")
            print(f"  最も注目した過去のイベント: {explanation['attention_focus']}")
            print("\n--- 打牌根拠の説明 ---")
            print(f"1. 定量的要約: {explanation['quantitative']}")
            print(f"2. 戦術的理由: {explanation['qualitative']}")
            print(f"3. 比較分析: {explanation['comparative']}")
            
    except Exception as e:
        logging.error(f"予期せぬエラーが発生しました: {e}", exc_info=True)