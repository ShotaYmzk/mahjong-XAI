# predict.py (Transformer版 - クラス定義を内部に含む)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import math # math をインポート
from full_mahjong_parser import parse_full_mahjong_log
# 修正されたGameStateと関連クラス・定数をインポート
# game_state.py に MAX_EVENT_HISTORY が定義されているか確認
try:
    from game_state import GameState, NUM_TILE_TYPES, MAX_EVENT_HISTORY
except ImportError:
    print("[Warning] Could not import MAX_EVENT_HISTORY from game_state. Using default value.")
    # データ生成時のシーケンス長が不明な場合、ロード時に取得するしかない
    # または、ここでデフォルト値を設定 (例: 60)
    from game_state import GameState, NUM_TILE_TYPES
    MAX_EVENT_HISTORY = 60 # データ生成時と合わせる必要あり

from naki_utils import decode_naki
from tile_utils import tile_id_to_string, tile_id_to_index

# --- クラス定義をここにコピー ---
# (train.py から MahjongTransformerModel と PositionalEncoding をコピー)
# (注意: 必ず学習・保存時に使用したバージョンのクラス定義をコピーしてください)

NUM_TILE_TYPES = 34 # train.pyから持ってくる

class PositionalEncoding(nn.Module):
    """Transformer用の位置エンコーディング (batch_first=True 対応)"""
    # ★★★ 学習・保存時と同じ PositionalEncoding クラス定義 ★★★
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500): # max_len はモデルロード時に実際の値で初期化されるべき
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        # d_modelが0や負にならないことを保証
        if d_model <= 0: raise ValueError("d_model must be positive")
        # div_termの計算を安全に
        div_term_base = torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        div_term = torch.exp(div_term_base)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        # d_modelが奇数か偶数かで処理を分ける
        if d_model % 2 == 0:
             pe[:, 1::2] = torch.cos(position * div_term)
        else:
             # d_modelが奇数の場合、最後の列は計算しないか、前の値をコピーするなど
             # ここでは div_term の最後の要素を使わずに cos を計算
             if div_term.shape[0] > 0: # div_termが空でないことを確認
                pe[:, 1::2] = torch.cos(position * div_term[:-1])
             # 最後の列は 0 のまま or sin/cos の繰り返しパターンを考慮して設定

        self.register_buffer('pe', pe.unsqueeze(0)) # バッチ次元を追加 [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] (batch_first=Trueのため)
        """
        # xのシーケンス長に合わせてpeをスライスし、加算する
        if x.size(1) > self.pe.size(1):
             raise ValueError(f"Input sequence length ({x.size(1)}) exceeds PositionalEncoding max_len ({self.pe.size(1)})")
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MahjongTransformerModel(nn.Module):
    # ★★★ 学習・保存時と同じ MahjongTransformerModel クラス定義 ★★★
    def __init__(self,
                 event_feature_dim: int,
                 static_feature_dim: int,
                 # ↓↓↓ 学習・保存時と同じデフォルト値 or 固定値を記述 ↓↓↓
                 d_model: int = 128,
                 nhead: int = 4,
                 d_hid: int = 256,  # ← エラーから推測される学習時の値
                 nlayers: int = 2,
                 dropout: float = 0.1,
                 output_dim: int = NUM_TILE_TYPES,
                 max_seq_len: int = 60): # ← 実際のシーケンス長を使う
        super().__init__()
        # 引数チェック
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        if d_hid <= 0: raise ValueError("d_hid must be positive")
        if d_model <= 0: raise ValueError("d_model must be positive")

        self.d_model = d_model

        # 1. Embedding + Positional Encoding
        self.event_encoder = nn.Linear(event_feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)

        # 2. Transformer Encoder
        try:
            encoder_layer_norm_first = False # PyTorch 1.9 以降のデフォルトは False
            # PyTorchのバージョンによっては norm_first のデフォルトが異なる可能性がある
            # 必要ならバージョンを確認して調整
            # from pkg_resources import parse_version
            # if parse_version(torch.__version__) >= parse_version("1.9"):
            #     encoder_layer_norm_first = False # または True に設定
            encoder_layers = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_hid,
                dropout=dropout,
                batch_first=True,
                norm_first=encoder_layer_norm_first # PyTorchのバージョン互換性のため明示的に指定を検討
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        except TypeError as e:
             # 古いPyTorchバージョンなどで norm_first がない場合
             print(f"[Warning] Instantiating TransformerEncoderLayer without norm_first (possibly older PyTorch): {e}")
             encoder_layers = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_hid,
                dropout=dropout,
                # batch_first は比較的新しい引数なので、古い場合はここでエラーになる可能性も
             )
             # batch_firstがない古いバージョンでは、入力のpermuteが必要になる
             print("[Warning] batch_first=True might not be supported or default. Input permutation might be needed if errors occur later.")
             self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)


        # 3. 静的特徴量用エンコーダー (MLP)
        static_encoder_input_dim = static_feature_dim
        static_encoder_hid_dim = max(1, d_model // 2) # 隠れ層次元が0にならないように
        self.static_encoder = nn.Sequential(
            nn.Linear(static_encoder_input_dim, static_encoder_hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(static_encoder_hid_dim, static_encoder_hid_dim) # 出力次元も hid_dim に
        )

        # 4. 結合層と最終出力層 (MLP)
        decoder_input_dim = d_model + static_encoder_hid_dim # 結合後の次元
        decoder_hid_dim = max(1, d_model // 2)
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, decoder_hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hid_dim, output_dim)
        )

        self._init_weights() # メソッド名を変更 (_init_weights)

    # メソッド名を _init_weights に変更してカプセル化を示唆
    def _init_weights(self) -> None:
        initrange = 0.1
        # event_encoder の初期化
        if hasattr(self.event_encoder, 'weight') and self.event_encoder.weight is not None:
            self.event_encoder.weight.data.uniform_(-initrange, initrange)
        if hasattr(self.event_encoder, 'bias') and self.event_encoder.bias is not None:
            self.event_encoder.bias.data.zero_()
        # static_encoder と decoder 内の Linear 層も初期化 (推奨)
        for layer in self.static_encoder:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                if layer.bias is not None:
                    layer.bias.data.zero_()
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, event_seq: torch.Tensor, static_feat: torch.Tensor, src_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # forward メソッド全体も学習時と同じものを記述
        # (前のコードと同じはずなので省略せず記述)
        # 1. イベントシーケンス処理
        embedded_seq = self.event_encoder(event_seq) * math.sqrt(self.d_model)
        pos_encoded_seq = self.pos_encoder(embedded_seq) # PositionalEncoding適用

        # Transformer Encoder
        # batch_first=True なのでマスクの形状は [batch_size, seq_len]
        transformer_output = self.transformer_encoder(pos_encoded_seq, src_key_padding_mask=src_padding_mask)

        # 平均プーリング (マスク考慮)
        if src_padding_mask is not None:
            active_elements_mask = ~src_padding_mask # Trueが有効
            seq_len_valid = active_elements_mask.sum(dim=1, keepdim=True).float()
            seq_len_valid = torch.max(seq_len_valid, torch.tensor(1.0, device=event_seq.device)) # 0除算防止
            masked_output = transformer_output.masked_fill(src_padding_mask.unsqueeze(-1), 0.0)
            transformer_pooled = masked_output.sum(dim=1) / seq_len_valid
        else:
            transformer_pooled = transformer_output.mean(dim=1) # マスクがない場合

        # 2. 静的特徴量処理
        encoded_static = self.static_encoder(static_feat)

        # 3. 結合して最終予測
        combined_features = torch.cat((transformer_pooled, encoded_static), dim=1)
        output = self.decoder(combined_features)

        return output

# --- ここまでクラス定義コピー ---


# --- 設定 ---
NUM_PLAYERS = 4
# DEFAULT_MODEL_PATH = "./trained_model/mahjong_transformer_model.pth" # パス確認
DEFAULT_MODEL_PATH = "/Users/yamazakiakirafutoshi/VScode/mahjong_XAI/ver_1.1.6/trained_model/mahjong_transformer_model.pth" # フルパス指定の例
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")) # MPSも考慮

print(f"Using device: {DEVICE}")


# --- ヘルパー関数 (変更なし) ---
def format_hand(hand_ids):
    if not hand_ids: return "なし"
    return " ".join([tile_id_to_string(t) for t in sorted(hand_ids, key=lambda t: (tile_id_to_index(t), t))])

def format_discards(discard_list):
    if not discard_list: return "なし"
    return " ".join([f"{tile_id_to_string(t)}{'*' if tsumogiri else ''}" for t, tsumogiri in discard_list])

def format_melds(meld_list):
    if not meld_list: return "なし"
    meld_strs = []
    for m_type, m_tiles, trigger_id, from_who_abs in meld_list:
        tiles_str = " ".join([tile_id_to_string(t) for t in m_tiles])
        from_str = f" P{from_who_abs}" if from_who_abs != -1 and m_type not in ["暗槓", "加槓"] else ""
        trigger_str = f"({tile_id_to_string(trigger_id)})" if trigger_id != -1 else ""
        meld_strs.append(f"{m_type}{trigger_str}[{tiles_str}]{from_str}")
    return " / ".join(meld_strs)


# --- モデルロード関数 ---
def load_trained_model(model_path, event_dim, static_dim, seq_len):
    """訓練済みTransformerモデルをロードする"""
    print(f"Attempting to load model from: {model_path}")
    if not os.path.exists(model_path):
        # ファイルが見つからない場合、カレントディレクトリも確認
        alt_path = os.path.join(os.getcwd(), os.path.basename(model_path))
        if os.path.exists(alt_path):
             print(f"Model not found at {model_path}, using {alt_path}")
             model_path = alt_path
        else:
             raise FileNotFoundError(f"Model file not found at specified path: {model_path} or CWD: {alt_path}")

    # モデルインスタンス化 (ファイル内に定義されたクラスを使用)
    # ★★★ 学習時と同じパラメータを指定 ★★★
    model_params = {
        'event_feature_dim': event_dim,
        'static_feature_dim': static_dim,
        'max_seq_len': seq_len,
        'd_model': 128,
        'nhead': 4,
        'd_hid': 256, # ← 学習時と同じ値 (エラーから推測)
        'nlayers': 2,
        'dropout': 0.1
    }
    print(f"Instantiating MahjongTransformerModel with params: {model_params}")
    # ★★★ ここでファイル内に定義されたクラスを直接呼び出す ★★★
    model = MahjongTransformerModel(**model_params).to(DEVICE)

    try:
        # state_dictをロード
        print(f"Loading state_dict...")
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval() # 評価モード
        print(f"Transformer model loaded successfully from: {model_path}")
        return model
    except RuntimeError as e:
        print(f"[Error] Failed to load model state_dict (RuntimeError): {e}")
        print(f"  >>> This usually means the model architecture defined here <<<")
        print(f"  >>> does not match the architecture saved in '{model_path}'. <<<")
        print(f"  >>> Double-check ALL hyperparameters (d_model, nhead, d_hid, nlayers, etc.) <<<")
        print(f"  >>> in the MahjongTransformerModel class definition above. <<<")
        # 保存された state_dict のキーと形状をいくつか表示してみる (デバッグ用)
        print("\n--- Checkpoint State Dict Keys (Sample) ---")
        try:
            count = 0
            for key, value in state_dict.items():
                print(f"  {key}: {value.shape}")
                count += 1
                if count >= 10: break
        except Exception as inspect_e:
             print(f"Could not inspect state_dict: {inspect_e}")
        print("------------------------------------------")
        raise e
    except Exception as e:
        print(f"[Error] Failed to load model state_dict (Other Exception): {e}")
        raise e

# --- 局面復元関数 ---
def reconstruct_game_state_at_tsumo(xml_path, target_round_index, target_tsumo_event_count_in_round):
    """指定局面までのGameStateを復元"""
    # (この関数は前のコードから変更なし)
    meta, rounds_data = parse_full_mahjong_log(xml_path)
    if not (1 <= target_round_index <= len(rounds_data)):
        raise ValueError(f"Invalid target_round_index: {target_round_index}. Must be between 1 and {len(rounds_data)}.")
    round_data = rounds_data[target_round_index - 1]
    game_state = GameState()
    try:
        # GameStateの初期化でエラーが発生しないか確認
        game_state.init_round(round_data)
    except Exception as e:
        print(f"[Error] Failed during game_state.init_round: {e}")
        raise

    current_tsumo_count = 0
    target_tsumo_event_info = None
    actual_discard_event_info = None
    events = round_data.get("events", [])

    for i, event_xml in enumerate(events):
        tag = event_xml["tag"]
        attrib = event_xml["attrib"]
        processed = False
        try:
            # --- 自摸イベント ---
            tsumo_player_id = -1; tsumo_pai_id = -1
            for t_tag, p_id in GameState.TSUMO_TAGS.items():
                if tag.startswith(t_tag):
                    try: tsumo_pai_id = int(tag[1:]); tsumo_player_id = p_id; processed = True; break
                    except ValueError: continue
            if processed:
                current_tsumo_count += 1
                if current_tsumo_count == target_tsumo_event_count_in_round:
                    target_tsumo_event_info = {"player": tsumo_player_id, "pai": tsumo_pai_id, "xml": event_xml}
                    game_state.process_tsumo(tsumo_player_id, tsumo_pai_id) # 状態更新
                    if i + 1 < len(events): # 次の打牌イベントを探す
                        next_event_xml = events[i+1]; next_tag = next_event_xml["tag"]
                        for d_tag, p_id_next in GameState.DISCARD_TAGS.items():
                            if next_tag.startswith(d_tag) and p_id_next == tsumo_player_id:
                                try:
                                    discard_pai_id = int(next_tag[1:]); tsumogiri = next_tag[0].islower()
                                    actual_discard_event_info = {"player": p_id_next, "pai": discard_pai_id, "tsumogiri": tsumogiri, "xml": next_event_xml}
                                    break
                                except ValueError: continue
                    return game_state, target_tsumo_event_info, actual_discard_event_info # 発見したので返す
                else:
                    game_state.process_tsumo(tsumo_player_id, tsumo_pai_id) # 状態更新
                continue

            # --- 打牌イベント ---
            discard_player_id = -1; discard_pai_id = -1; tsumogiri = False
            for d_tag, p_id in GameState.DISCARD_TAGS.items():
                if tag.startswith(d_tag):
                    try: discard_pai_id = int(tag[1:]); discard_player_id = p_id; tsumogiri = tag[0].islower(); processed = True; break
                    except ValueError: continue
            if processed:
                game_state.process_discard(discard_player_id, discard_pai_id, tsumogiri)
                continue

            # --- 鳴きイベント ---
            if not processed and tag == "N":
                naki_player_id = int(attrib.get("who", -1)); meld_code = int(attrib.get("m", "0"))
                if naki_player_id != -1: game_state.process_naki(naki_player_id, meld_code)
                processed = True; continue

            # --- リーチイベント ---
            if not processed and tag == "REACH":
                 reach_player_id = int(attrib.get("who", -1)); step = int(attrib.get("step", 0))
                 if reach_player_id != -1: game_state.process_reach(reach_player_id, step)
                 processed = True; continue

            # --- DORA イベント ---
            if not processed and tag == "DORA":
                 hai = int(attrib.get("hai", -1))
                 if hai != -1: game_state.process_dora(hai)
                 processed = True; continue

            # --- 局終了イベント ---
            if not processed and (tag == "AGARI" or tag == "RYUUKYOKU"):
                 if tag == "AGARI": game_state.process_agari(attrib)
                 else: game_state.process_ryuukyoku(attrib)
                 processed = True; break

        except Exception as e:
            print(f"[Error] Processing event failed: Tag={tag}, Attrib={attrib}, Error={e}")
            # エラーが発生しても、指定ツモ回数に達していない場合は処理を続けるか検討
            # ここではエラー発生で関数を抜けるように raise する
            raise e

    # ループが終了しても見つからなかった場合
    raise ValueError(f"Target tsumo event count {target_tsumo_event_count_in_round} not reached in round {target_round_index}.")


# --- 打牌予測関数 ---
def predict_discard(model, game_state: GameState, player_id: int):
    """モデルで打牌予測を行い、手牌の制約を考慮して最終的な捨て牌を決定する。"""
    # (この関数は前のコードから変更なし)
    try:
        event_sequence = game_state.get_event_sequence_features()
        static_features = game_state.get_static_features(player_id)
    except Exception as e:
        print(f"[Error] Failed to get features from game_state: {e}")
        raise

    seq_tensor = torch.tensor(event_sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    static_tensor = torch.tensor(static_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    padding_code = 8 # GameState.EVENT_TYPES["PADDING"]
    mask_tensor = (seq_tensor[:, :, 0] == padding_code).to(DEVICE)

    with torch.no_grad():
        outputs = model(seq_tensor, static_tensor, mask_tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    valid_discard_indices = game_state.get_valid_discard_options(player_id)
    best_prob = -1.0
    best_index = -1

    if not valid_discard_indices:
        print("[Error] No valid discard options found!")
        best_index = np.argmax(probabilities)
        best_prob = probabilities[best_index]
    else:
        for index in valid_discard_indices:
            if 0 <= index < NUM_TILE_TYPES:
                if probabilities[index] > best_prob:
                    best_prob = probabilities[index]
                    best_index = index
            else: print(f"[Warning] Invalid index {index} in valid options.")
        if best_index == -1:
            best_index = valid_discard_indices[0]
            best_prob = probabilities[best_index]
            print(f"[Warning] Could not determine best discard, choosing first valid: {best_index}")

    if best_index == -1:
        print("[Error] Failed to select any discard tile index!")
        # 安全のため、有効牌があれば最初のものを、なければ確率最大のものを返す
        best_index = valid_discard_indices[0] if valid_discard_indices else np.argmax(probabilities)
        best_prob = probabilities[best_index]

    return best_index, best_prob, probabilities


# --- 局/自風 文字列取得関数 ---
def get_wind_str(round_num_wind, player_id, dealer):
    """局風と自風の文字列を返す"""
    # (この関数は前のコードから変更なし)
    round_winds = ["東", "南", "西", "北"]
    player_winds = ["東", "南", "西", "北"]
    try:
        round_wind_str = round_winds[round_num_wind // NUM_PLAYERS]
        kyoku_num = (round_num_wind % NUM_PLAYERS) + 1
        my_wind_str = player_winds[(player_id - dealer + NUM_PLAYERS) % NUM_PLAYERS]
        return f"{round_wind_str}{kyoku_num}局", my_wind_str
    except IndexError:
        return "不明局", "不明家"


# --- メイン処理 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict discard tile using a trained Mahjong Transformer model.")
    parser.add_argument("xml_file", help="Path to the Tenhou XML log file.")
    parser.add_argument("round_index", type=int, help="Round index within the game (1-based).")
    parser.add_argument("tsumo_count", type=int, help="Tsumo count index within the round (1-based).")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help=f"Path to the trained model file (default: {DEFAULT_MODEL_PATH})")

    args = parser.parse_args()

    try:
        # 1. 指定局面までのGameStateを復元
        print(f"Reconstructing game state for {args.xml_file}, Round {args.round_index}, Tsumo {args.tsumo_count}...")
        game_state, tsumo_info, discard_info = reconstruct_game_state_at_tsumo(
            args.xml_file, args.round_index, args.tsumo_count
        )
        print("Game state reconstruction complete.")

        # 予測を行うプレイヤーID
        player_id = game_state.current_player
        if tsumo_info and player_id != tsumo_info["player"]:
             print(f"[Warning] Mismatch game_state.current_player ({player_id}) and tsumo event player ({tsumo_info['player']}). Using tsumo event player.")
             player_id = tsumo_info["player"]
        elif not tsumo_info:
             print("[Error] Tsumo event information is missing.")
             exit() # 致命的なエラー

        # 2. モデル入力次元数を取得
        print("Generating features for prediction...")
        try:
            temp_seq = game_state.get_event_sequence_features()
            temp_static = game_state.get_static_features(player_id)
            event_dim = temp_seq.shape[1]
            static_dim = temp_static.shape[0]
            seq_len = temp_seq.shape[0]
            print(f"Feature dimensions: Event Dim={event_dim}, Static Dim={static_dim}, Seq Len={seq_len}")
        except Exception as e:
            print(f"[Error] Failed to generate features: {e}")
            raise

        # 3. モデルをロード
        model = load_trained_model(args.model_path, event_dim, static_dim, seq_len)

        # 4. 打牌予測を実行
        print("Predicting discard...")
        predicted_index, predicted_prob, all_probabilities = predict_discard(
            model, game_state, player_id
        )
        predicted_tile_str = tile_id_to_string(predicted_index * 4) # 代表牌
        print("Prediction complete.")

        # 5. 実際の捨て牌を取得
        actual_discard_str = "N/A (End of game?)"
        if discard_info:
            actual_discard_str = tile_id_to_string(discard_info["pai"])
            if discard_info["tsumogiri"]: actual_discard_str += "*"

        # 6. 結果を表示
        print("\n=== Transformer 予測テスト ===")
        print(f"--- 対象局面 (R:{args.round_index}, TsumoCount:{args.tsumo_count}) ---")
        # ... (以降の表示部分は変更なし) ...
        print(f"XMLファイル: {os.path.basename(args.xml_file)}")
        round_str, my_wind_str = get_wind_str(game_state.round_num_wind, player_id, game_state.dealer)
        honba_str = f"{game_state.honba}本場"
        kyotaku_str = f"({game_state.kyotaku}供託)" if game_state.kyotaku > 0 else ""
        print(f"局: {round_str} {honba_str} {kyotaku_str} / プレイヤー: {player_id} ({my_wind_str}家)")
        print(f"ツモ牌: {tile_id_to_string(tsumo_info['pai'])}")
        print(f"巡目: {game_state.junme:.2f}")
        print(f"ドラ表示: {' '.join([tile_id_to_string(t) for t in game_state.dora_indicators])}")
        print(f"点数: {[f'P{i}:{s}' for i, s in enumerate(game_state.current_scores)]}")
        print("--- 状態 ---")
        print("手牌 (ツモ後):")
        for p in range(NUM_PLAYERS):
            hand_str = format_hand(game_state.player_hands[p])
            reach_str = "*" if game_state.player_reach_status[p] == 2 else ("(宣)" if game_state.player_reach_status[p] == 1 else "")
            print(f"  P{p}{reach_str}: {hand_str}")
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
        print(f"Top {top_n} predictions:")
        for i in range(top_n):
            idx = indices_sorted[i]
            prob = all_probabilities[idx]
            tile_str = tile_id_to_string(idx * 4)
            print(f"  {i+1}. {tile_str} ({prob:.4f})")

    except FileNotFoundError as e: print(f"エラー: ファイルが見つかりません - {e}")
    except ValueError as e: print(f"エラー: 値が不正です - {e}")
    except ImportError as e: print(f"エラー: インポートに失敗しました - {e}")
    except AttributeError as e: print(f"エラー: 属性エラー（クラス定義やメソッド呼び出しを確認） - {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()