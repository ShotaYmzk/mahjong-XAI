# predict.py (Transformer版)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
from full_mahjong_parser import parse_full_mahjong_log
# 修正されたGameStateと関連クラス・定数をインポート
from game_state import GameState, NUM_TILE_TYPES, MAX_EVENT_HISTORY
from naki_utils import decode_naki
from tile_utils import tile_id_to_string, tile_id_to_index
# train.pyからモデル定義をインポート (必要ならsys.path設定)
try:
    from train import MahjongTransformerModel # Transformerモデルをインポート
except ImportError:
    print("[Error] Could not import MahjongTransformerModel from train.py.")
    # 必要ならモデル定義をここにコピー
    exit()

# --- 設定 ---
NUM_PLAYERS = 4
DEFAULT_MODEL_PATH = "./trained_model/mahjong_transformer_model.pth"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
# DEVICE = torch.device("cpu") # CPU強制

# --- ヘルパー関数 (format_hand, format_discards, format_melds は変更なし、ただし format_melds は game_state の出力に依存) ---
def format_hand(hand_ids):
    # ... (変更なし) ...
    if not hand_ids: return "なし"
    return " ".join([tile_id_to_string(t) for t in sorted(hand_ids, key=lambda t: (tile_id_to_index(t), t))])

def format_discards(discard_list):
    # ... (変更なし) ...
    if not discard_list: return "なし"
    return " ".join([f"{tile_id_to_string(t)}{'*' if tsumogiri else ''}" for t, tsumogiri in discard_list])

def format_melds(meld_list):
    # ... (変更なし, game_state.player_melds の形式に依存) ...
    if not meld_list: return "なし"
    meld_strs = []
    for m_type, m_tiles, trigger_id, from_who_abs in meld_list:
        tiles_str = " ".join([tile_id_to_string(t) for t in m_tiles])
        from_str = f" P{from_who_abs}" if from_who_abs != -1 and m_type not in ["暗槓", "加槓"] else ""
        trigger_str = f"({tile_id_to_string(trigger_id)})" if trigger_id != -1 else ""
        meld_strs.append(f"{m_type}{trigger_str}[{tiles_str}]{from_str}")
    return " / ".join(meld_strs)


def load_trained_model(model_path, event_dim, static_dim, seq_len):
    """訓練済みTransformerモデルをロードする"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # モデルインスタンス化 (train.pyと同じパラメータ設定が必要)
    # TODO: モデルのハイパーパラメータを保存・ロードする仕組みが望ましい
    model = MahjongTransformerModel(
        event_feature_dim=event_dim,
        static_feature_dim=static_dim,
        max_seq_len=seq_len,
        # d_model, nhead, nlayersなどもtrain.pyと合わせる
    ).to(DEVICE)

    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval() # 評価モード
        print(f"Transformer model loaded from: {model_path}")
        return model
    except Exception as e:
        print(f"[Error] Failed to load model state_dict: {e}")
        print(f"  Check model hyperparameters (d_model, nhead, etc.) match the saved model.")
        raise e


def reconstruct_game_state_at_tsumo(xml_path, target_round_index, target_tsumo_event_count_in_round):
    """
    指定されたXMLとイベントインデックスまでのGameStateを復元し、
    その自摸イベント発生"直後"のGameStateと、関連イベント情報を返す。
    """
    meta, rounds_data = parse_full_mahjong_log(xml_path)

    if not (1 <= target_round_index <= len(rounds_data)):
        raise ValueError(f"Invalid target_round_index: {target_round_index}. Must be between 1 and {len(rounds_data)}.")

    round_data = rounds_data[target_round_index - 1]
    game_state = GameState() # 修正されたGameStateを使用
    game_state.init_round(round_data)

    current_tsumo_count = 0
    target_tsumo_event_info = None
    actual_discard_event_info = None

    events = round_data.get("events", [])
    for i, event_xml in enumerate(events):
        tag = event_xml["tag"]
        attrib = event_xml["attrib"]
        processed = False

        # --- 自摸イベント ---
        tsumo_player_id = -1
        tsumo_pai_id = -1
        for t_tag, p_id in GameState.TSUMO_TAGS.items():
            if tag.startswith(t_tag):
                try:
                    tsumo_pai_id = int(tag[1:])
                    tsumo_player_id = p_id
                    processed = True
                    break
                except ValueError: continue

        if processed: # 自摸イベントの場合
            current_tsumo_count += 1
            # 指定された自摸イベントか？
            if current_tsumo_count == target_tsumo_event_count_in_round:
                target_tsumo_event_info = {"player": tsumo_player_id, "pai": tsumo_pai_id, "xml": event_xml}
                # ★★★ この自摸イベントを処理した直後の状態を返す ★★★
                game_state.process_tsumo(tsumo_player_id, tsumo_pai_id)

                # 次のイベントが対応する打牌か確認
                if i + 1 < len(events):
                    next_event_xml = events[i+1]
                    next_tag = next_event_xml["tag"]
                    for d_tag, p_id_next in GameState.DISCARD_TAGS.items():
                        if next_tag.startswith(d_tag) and p_id_next == tsumo_player_id:
                            try:
                                discard_pai_id = int(next_tag[1:])
                                tsumogiri = next_tag[0].islower()
                                actual_discard_event_info = {"player": p_id_next, "pai": discard_pai_id, "tsumogiri": tsumogiri, "xml": next_event_xml}
                                break
                            except ValueError: continue
                # GameStateと関連情報を返す
                return game_state, target_tsumo_event_info, actual_discard_event_info
            else:
                # ターゲット以前の自摸イベントは通常通り処理
                game_state.process_tsumo(tsumo_player_id, tsumo_pai_id)
            continue # 次のイベントへ

        # --- 打牌イベント ---
        discard_player_id = -1
        discard_pai_id = -1
        tsumogiri = False
        for d_tag, p_id in GameState.DISCARD_TAGS.items():
            if tag.startswith(d_tag):
                try:
                    discard_pai_id = int(tag[1:])
                    discard_player_id = p_id
                    tsumogiri = tag[0].islower()
                    processed = True
                    break
                except ValueError: continue
        if processed:
            game_state.process_discard(discard_player_id, discard_pai_id, tsumogiri)
            continue

        # --- 鳴きイベント ---
        if not processed and tag == "N":
            # ... (鳴き処理: game_state.process_naki) ...
            try:
                naki_player_id = int(attrib.get("who", -1)); meld_code = int(attrib.get("m", "0"))
                if naki_player_id != -1: game_state.process_naki(naki_player_id, meld_code)
                processed = True
            except Exception as e: print(f"[Warn] N process error: {e}")
            continue

        # --- リーチイベント ---
        if not processed and tag == "REACH":
             # ... (リーチ処理: game_state.process_reach) ...
             try:
                 reach_player_id = int(attrib.get("who", -1)); step = int(attrib.get("step", 0))
                 if reach_player_id != -1: game_state.process_reach(reach_player_id, step)
                 processed = True
             except Exception as e: print(f"[Warn] Reach process error: {e}")
             continue

        # --- DORA イベント ---
        if not processed and tag == "DORA":
             # ... (ドラ処理: game_state.process_dora) ...
             try:
                 hai = int(attrib.get("hai", -1))
                 if hai != -1: game_state.process_dora(hai)
                 processed = True
             except Exception as e: print(f"[Warn] Dora process error: {e}")
             continue

        # --- 局終了イベント ---
        if not processed and (tag == "AGARI" or tag == "RYUUKYOKU"):
             # ... (局終了処理) ...
             if tag == "AGARI": game_state.process_agari(attrib)
             else: game_state.process_ryuukyoku(attrib)
             processed = True
             break # 局終了

    # ループが終了しても見つからなかった場合
    raise ValueError(f"Target tsumo event count {target_tsumo_event_count_in_round} not reached in round {target_round_index}.")


def predict_discard(model, game_state: GameState, player_id: int):
    """モデルで打牌予測を行い、手牌の制約を考慮して最終的な捨て牌を決定する。"""
    # 1. モデル入力用の特徴量を取得
    event_sequence = game_state.get_event_sequence_features()
    static_features = game_state.get_static_features(player_id)

    # NumPy配列をTensorに変換し、バッチ次元を追加
    seq_tensor = torch.tensor(event_sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    static_tensor = torch.tensor(static_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # パディングマスク生成
    padding_code = GameState.EVENT_TYPES["PADDING"]
    mask_tensor = (seq_tensor[:, :, 0] == padding_code).to(DEVICE) # [batch_size, seq_len]

    # 2. モデルで予測実行
    with torch.no_grad():
        outputs = model(seq_tensor, static_tensor, mask_tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    # 3. 有効な打牌選択肢を取得
    valid_discard_indices = game_state.get_valid_discard_options(player_id)

    # 4. 有効な選択肢の中から最も確率の高いものを選択
    best_prob = -1.0
    best_index = -1

    if not valid_discard_indices:
        # 通常、リーチ後ツモ切りなどで最低1つはあるはず
        print("[Error] No valid discard options found!")
        # フォールバック: 全ての牌種から最も確率の高いものを選択（ゲームルール違反の可能性）
        best_index = np.argmax(probabilities)
        best_prob = probabilities[best_index]
    else:
        for index in valid_discard_indices:
            if 0 <= index < NUM_TILE_TYPES: # インデックス範囲チェック
                if probabilities[index] > best_prob:
                    best_prob = probabilities[index]
                    best_index = index
            else:
                print(f"[Warning] Invalid index {index} found in valid options.")

        # もしvalidな牌の中で一つも選ばれなかった場合 (通常ありえない)
        if best_index == -1 and valid_discard_indices:
            best_index = valid_discard_indices[0] # 最初の有効牌を選択
            best_prob = probabilities[best_index]
            print(f"[Warning] Could not determine best discard, choosing first valid: {best_index}")
        elif best_index == -1: # validな牌もない場合 (Errorで捕捉済みのはず)
             best_index = np.argmax(probabilities) # 最終手段
             best_prob = probabilities[best_index]


    # best_index が -1 のままならエラー
    if best_index == -1:
        print("[Error] Failed to select any discard tile index!")
        best_index = 0 # ダミー値
        best_prob = 0.0

    return best_index, best_prob, probabilities


def get_wind_str(round_num_wind, player_id, dealer):
    """局風と自風の文字列を返す"""
    round_winds = ["東", "南", "西", "北"] # 場風
    player_winds = ["東", "南", "西", "北"] # 自風
    try:
        round_wind_str = round_winds[round_num_wind // NUM_PLAYERS] # 東/南/西/北
        kyoku_num = (round_num_wind % NUM_PLAYERS) + 1 # 1-4局
        my_wind_str = player_winds[(player_id - dealer + NUM_PLAYERS) % NUM_PLAYERS]
        return f"{round_wind_str}{kyoku_num}局", my_wind_str
    except IndexError:
        return "不明局", "不明家"


# --- メイン処理 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict discard tile using a trained Mahjong Transformer model.")
    parser.add_argument("xml_file", help="Path to the Tenhou XML log file.")
    # round_index を 1-8 (東1-南4など) -> round_num_wind (0-7) に内部変換するか、
    # または単に局のインデックス(1から開始)として扱うか決める。ここでは局インデックス(1-based)とする。
    parser.add_argument("round_index", type=int, help="Round index within the game (1-based).")
    # tsumo_index を「その局で何回目のツモか」に変更
    parser.add_argument("tsumo_count", type=int, help="Tsumo count index within the round (1-based).")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help=f"Path to the trained model file (default: {DEFAULT_MODEL_PATH})")

    args = parser.parse_args()

    try:
        # 1. 指定局面までのGameStateを復元
        # reconstruct_game_state_at_tsumo は指定ツモ直後のGameStateを返す
        game_state, tsumo_info, discard_info = reconstruct_game_state_at_tsumo(
            args.xml_file, args.round_index, args.tsumo_count
        )

        # 予測を行うプレイヤーID
        player_id = game_state.current_player # process_tsumo後なので、ツモったプレイヤーになっているはず
        if player_id != tsumo_info["player"]:
             print(f"[Warning] Mismatch between game_state.current_player ({game_state.current_player}) and tsumo event player ({tsumo_info['player']})")
             player_id = tsumo_info["player"] # イベント情報に基づいたプレイヤーを使う

        # 2. モデルの入力次元数を取得 (学習済みモデルの構造から取得するのが理想)
        # ここではGameStateから取得するが、学習時と一致している必要がある
        # ダミーの特徴量を生成して次元数を取得（非効率だがpredict単体実行のため）
        temp_seq = game_state.get_event_sequence_features()
        temp_static = game_state.get_static_features(player_id)
        event_dim = temp_seq.shape[1]
        static_dim = temp_static.shape[0]
        seq_len = temp_seq.shape[0] # = MAX_EVENT_HISTORY

        # 3. モデルをロード
        model = load_trained_model(args.model_path, event_dim, static_dim, seq_len)

        # 4. 打牌予測を実行
        predicted_index, predicted_prob, all_probabilities = predict_discard(
            model, game_state, player_id
        )
        # 予測された牌種インデックスから代表的な牌文字列へ
        # (0-33 のインデックスに対応する牌を返す。赤ドラは区別しない)
        predicted_tile_str = tile_id_to_string(predicted_index * 4)


        # 5. 実際の捨て牌を取得
        actual_discard_str = "N/A"
        if discard_info:
            actual_discard_str = tile_id_to_string(discard_info["pai"])
            if discard_info["tsumogiri"]: actual_discard_str += "*"

        # 6. 結果を表示
        print("\n=== Transformer 予測テスト ===")
        print(f"--- 対象局面 (R:{args.round_index}, TsumoCount:{args.tsumo_count}) ---")
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
        # 上位 N 件の予測を表示 (デバッグ用)
        top_n = 5
        indices_sorted = np.argsort(all_probabilities)[::-1] # 降順ソート
        print(f"Top {top_n} predictions:")
        for i in range(top_n):
            idx = indices_sorted[i]
            prob = all_probabilities[idx]
            tile_str = tile_id_to_string(idx * 4)
            print(f"  {i+1}. {tile_str} ({prob:.4f})")

    except FileNotFoundError as e: print(f"エラー: {e}")
    except ValueError as e: print(f"エラー: {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()