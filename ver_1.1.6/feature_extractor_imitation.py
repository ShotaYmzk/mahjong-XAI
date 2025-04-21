# feature_extractor_imitation.py (Transformer対応版)
import numpy as np
import re
from full_mahjong_parser import parse_full_mahjong_log, tile_id_to_string # xml_parser.pyからインポート
# 修正されたGameStateと、追加した定数をインポート
from game_state import GameState, NUM_TILE_TYPES, NUM_PLAYERS, MAX_EVENT_HISTORY
from tile_utils import tile_id_to_index

def extract_features_labels_for_imitation(xml_path: str):
    """
    天鳳XMLログから模倣学習用の (イベントシーケンス, 静的状態, 正解打牌) ペアを抽出する。
    """
    meta, rounds_data = parse_full_mahjong_log(xml_path)
    all_sequences = []
    all_static_features = []
    all_labels = []

    game_state = GameState()
    # 直前の状態は GameState オブジェクト自体が保持する

    for round_data in rounds_data:
        try:
            game_state.init_round(round_data)

            for event in round_data.get("events", []):
                tag = event["tag"]
                attrib = event["attrib"]
                processed = False # イベントが処理されたか

                # --- 自摸イベント ---
                tsumo_player_id = -1
                tsumo_pai_id = -1
                for t_tag, p_id in GameState.TSUMO_TAGS.items():
                    if tag.startswith(t_tag):
                        try:
                            tsumo_pai_id = int(tag[1:])
                            tsumo_player_id = p_id
                            break
                        except ValueError: continue

                if tsumo_player_id != -1:
                    processed = True
                    # ★★★ 打牌選択の瞬間 ★★★
                    # この時点でのシーケンスと静的状態を取得
                    try:
                        # 1. イベントシーケンスを取得 (パディング済み)
                        current_sequence = game_state.get_event_sequence_features()
                        # 2. 静的状態ベクトルを取得 (ツモ牌を含む手牌で計算)
                        game_state.process_tsumo(tsumo_player_id, tsumo_pai_id) # 先に状態を更新
                        current_static = game_state.get_static_features(tsumo_player_id)

                        # 次の打牌イベントを待つために一時保存
                        last_decision_point = {
                            "sequence": current_sequence,
                            "static": current_static,
                            "player": tsumo_player_id
                        }
                    except Exception as e:
                         print(f"[Error] Failed to extract features at tsumo: {e}. Skipping.")
                         import traceback
                         traceback.print_exc()
                         last_decision_point = None # エラー発生時はクリア
                    continue # 次の打牌イベントでラベルと紐付け

                # --- 打牌イベント ---
                if not processed:
                    discard_player_id = -1
                    discard_pai_id = -1
                    tsumogiri = False
                    for d_tag, p_id in GameState.DISCARD_TAGS.items():
                        if tag.startswith(d_tag):
                            try:
                                discard_pai_id = int(tag[1:])
                                discard_player_id = p_id
                                tsumogiri = tag[0].islower()
                                break
                            except ValueError: continue

                    if discard_player_id != -1:
                        processed = True
                        # 直前の自摸時の特徴量と紐付けて学習データを生成
                        if last_decision_point and last_decision_point["player"] == discard_player_id:
                            # 正解ラベル (捨てた牌のインデックス 0-33)
                            label = tile_id_to_index(discard_pai_id)
                            if label != -1: # 不正な牌でなければ追加
                                all_sequences.append(last_decision_point["sequence"])
                                all_static_features.append(last_decision_point["static"])
                                all_labels.append(label)
                            else:
                                print(f"[Warning] Invalid discard label for tile {discard_pai_id}. Skipping.")

                            last_decision_point = None # 使用済みのためクリア
                        else:
                            # 鳴き後やエラー後などで、対応する状態がない場合がある
                            # print(f"[Debug] Discard event for player {discard_player_id} without preceding tsumo state.")
                            pass

                        # 状態を更新 (自摸処理は上で終わっているので、打牌処理のみ)
                        # ★注意: process_tsumoが先に呼ばれているので、ここでは捨てるだけ
                        try:
                             if discard_pai_id in game_state.player_hands[discard_player_id]:
                                 game_state.process_discard(discard_player_id, discard_pai_id, tsumogiri)
                             else:
                                 # process_tsumoで手牌に追加されているはずの牌がない -> 異常
                                 print(f"[Error] Discard tile {tile_id_to_string(discard_pai_id)} not found in hand after tsumo for P{discard_player_id}. Hand: {[tile_id_to_string(t) for t in game_state.player_hands[discard_player_id]]}")
                                 # 強制的に状態を進める (エラーが多い場合は要見直し)
                                 game_state.player_discards[discard_player_id].append((discard_pai_id, tsumogiri))
                                 game_state._add_event("DISCARD", player=discard_player_id, tile=discard_pai_id, data={"tsumogiri": tsumogiri, "error": "not_in_hand_after_tsumo"})
                                 game_state.last_discard_event_player = discard_player_id; # ...
                                 if game_state.player_reach_status[discard_player_id] == 1: # リーチ成立処理も行う
                                     game_state.player_reach_status[discard_player_id] = 2; #...
                                     game_state._add_event("REACH", player=discard_player_id, data={"step": 2})
                                 game_state.current_player = (discard_player_id + 1) % NUM_PLAYERS


                        except Exception as e:
                             print(f"[Error] during process_discard after feature extraction: {e}")
                        continue

                # --- 鳴きイベント ---
                if not processed and tag == "N":
                    processed = True
                    last_decision_point = None # 鳴きが発生したら直前の自摸は無効
                    try:
                        naki_player_id = int(attrib.get("who", -1))
                        meld_code = int(attrib.get("m", "0"))
                        if naki_player_id != -1:
                            game_state.process_naki(naki_player_id, meld_code)
                    except ValueError: print(f"[Warning] Invalid N tag: {attrib}")
                    except Exception as e: print(f"[Error] during process_naki: {e}")
                    continue

                # --- リーチイベント ---
                # リーチ宣言(step=1)はprocess_reach内でイベント追加
                # リーチ成立(step=2)はprocess_discard内でイベント追加
                if not processed and tag == "REACH":
                    processed = True
                    last_decision_point = None # リーチ宣言でも状態が変わるのでクリア
                    try:
                        reach_player_id = int(attrib.get("who", -1))
                        step = int(attrib.get("step", 0))
                        if reach_player_id != -1 and step == 1: # step=1 の宣言のみ処理 (step=2は打牌とセット)
                             game_state.process_reach(reach_player_id, step)
                    except ValueError: print(f"[Warning] Invalid REACH tag: {attrib}")
                    continue

                # --- 局終了イベント ---
                if not processed and (tag == "AGARI" or tag == "RYUUKYOKU"):
                    processed = True
                    last_decision_point = None
                    # スコア更新などを行う
                    if tag == "AGARI": game_state.process_agari(attrib)
                    else: game_state.process_ryuukyoku(attrib)
                    break # この局の処理は終了

                # --- その他のイベント (DORAなど) ---
                if not processed and tag == "DORA":
                    processed = True
                    try:
                        hai = int(attrib.get("hai", -1))
                        if hai != -1:
                            game_state.process_dora(hai) # process_doraを呼ぶ
                    except ValueError: print(f"[Warning] Invalid DORA tag: {attrib}")

        except Exception as e:
             print(f"[Error] Unhandled exception during round processing (Round Index {round_data.get('round_index')}): {e}")
             import traceback
             traceback.print_exc()
             last_decision_point = None # エラー発生局はクリアして次へ
             continue

    if not all_sequences:
         print("[Warning] No features extracted. Check XML logs and parsing logic.")
         return None, None, None

    # NumPy配列に変換して返す
    sequences_np = np.array(all_sequences, dtype=np.float32)
    static_features_np = np.array(all_static_features, dtype=np.float32)
    labels_np = np.array(all_labels, dtype=np.int64)

    print(f"Extraction Summary: Sequences={sequences_np.shape}, Static={static_features_np.shape}, Labels={labels_np.shape}")

    # 次元数のチェック (デバッグ用)
    if len(sequences_np) > 0:
        print(f"  Sequence Event Dim: {sequences_np.shape[-1]}")
    if len(static_features_np) > 0:
        print(f"  Static Feature Dim: {static_features_np.shape[-1]}")


    return sequences_np, static_features_np, labels_np