# feature_extractor_imitation.py (Transformer対応版)
import numpy as np
import re
from full_mahjong_parser import parse_full_mahjong_log, tile_id_to_string # xml_parser.pyからインポート
# 修正されたGameStateと、追加した定数をインポート
from game_state import GameState, NUM_TILE_TYPES, NUM_PLAYERS, MAX_EVENT_HISTORY
from tile_utils import tile_id_to_index

# feature_extractor_imitation.py の先頭に以下のコードを追加

import numpy as np
import re
from full_mahjong_parser import parse_full_mahjong_log, tile_id_to_string
from game_state import GameState, NUM_TILE_TYPES, NUM_PLAYERS, MAX_EVENT_HISTORY
from tile_utils import tile_id_to_index

# GameStateを拡張して必要なメソッドを追加
class EnhancedGameState(GameState):
    def get_event_sequence_features(self):
        """イベント履歴を固定長の数値ベクトルシーケンスに変換する"""
        sequence = []
        event_specific_dim = 2  # DISCARD(1), N(2), REACH(1) の最大値

        for event in self.event_history:
            event_vec_base = [
                event["type"],
                event["player"] + 1,
                event["tile_index"] + 1,
                event["junme"],
            ]

            event_vec_specific = [0] * event_specific_dim
            data = event.get("data", {})
            event_type_code = event["type"]

            if event_type_code == self.EVENT_TYPES["DISCARD"]:
                event_vec_specific[0] = 1 if data.get("tsumogiri", False) else 0
            elif event_type_code == self.EVENT_TYPES["N"]:
                event_vec_specific[0] = data.get("naki_type", -1) + 1
                event_vec_specific[1] = data.get("from_who", -1) + 1
            elif event_type_code == self.EVENT_TYPES["REACH"]:
                event_vec_specific[0] = data.get("step", 0)

            event_vec = event_vec_base + event_vec_specific
            sequence.append(event_vec)

        # Padding
        padding_length = MAX_EVENT_HISTORY - len(sequence)
        event_dim = 4 + event_specific_dim
        padding_vec = [self.EVENT_TYPES["PADDING"]] + [0] * (event_dim - 1)
        padded_sequence = sequence + [padding_vec] * padding_length

        try:
            return np.array(padded_sequence, dtype=np.float32)
        except ValueError as e:
            print(f"[Critical Error] Failed to convert padded_sequence to numpy array: {e}")
            fixed_sequence = []
            for vec in sequence:
                if len(vec) > event_dim:
                    vec = vec[:event_dim]
                elif len(vec) < event_dim:
                    vec = vec + [0] * (event_dim - len(vec))
                fixed_sequence.append(vec)
                
            fixed_padded = fixed_sequence + [padding_vec] * padding_length
            return np.array(fixed_padded, dtype=np.float32)

    def process_reach(self, player_id, step):
        """リーチ処理"""
        if step == 1:
            if self.current_scores[player_id] >= 1000 and self.player_reach_status[player_id] == 0:
                self.player_reach_status[player_id] = 1
                self.player_reach_junme[player_id] = self.junme
                self._add_event("REACH", player=player_id, data={"step": 1, "junme": self.junme})
            else:
                status = self.player_reach_status[player_id]
                score = self.current_scores[player_id]
                print(f"[Warn] P{player_id} reach step 1 failed: score {score}, status {status}.")
    
    def process_naki(self, naki_player_id, meld_code):
        """鳴き処理 (イベント追加、壁カウンタ更新)"""
        naki_type_str, decoded_tiles, trigger_pai_id, _ = decode_naki(meld_code)
        if naki_type_str == "不明":
            print(f"[Warn] Unknown naki m={meld_code}")
            return
        
        # 鳴きイベントをログに記録（デバッグ用）
        # 麻雀のルールに合った鳴き牌の表示 (ポンは3枚)
        clean_tiles = decoded_tiles[:3] if naki_type_str == "ポン" else decoded_tiles
        print(f"[Debug] Processing {naki_type_str} for P{naki_player_id}, decoded_tiles={[tile_id_to_string(t) for t in clean_tiles]}, trigger={tile_id_to_string(trigger_pai_id) if trigger_pai_id != -1 else 'N/A'}")
        
        # --- 加槓・暗槓 ---
        if naki_type_str == "加槓":
            # 加槓処理
            # まず、プレイヤーのポン副露を探す
            base_index = tile_id_to_index(decoded_tiles[0]) if decoded_tiles else -1
            if base_index == -1:
                print(f"[Error] Kakan decode failed with m={meld_code}")
                return
                
            # ポン面子を探す
            target_meld_index = -1
            for i, meld in enumerate(self.player_melds[naki_player_id]):
                if meld[0] == "ポン" and all(tile_id_to_index(t) == base_index for t in meld[1]):
                    target_meld_index = i
                    break
                    
            if target_meld_index == -1:
                print(f"[Error] No matching Pon found for Kakan with base_index={base_index}")
                return
                
            # 手牌から加槓牌を削除
            matching_tiles = [t for t in self.player_hands[naki_player_id] if tile_id_to_index(t) == base_index]
            if not matching_tiles:
                print(f"[Error] Kakan tile with index {base_index} not found in hand of P{naki_player_id}")
                return
                
            kakan_pai_id = matching_tiles[0]
            self.player_hands[naki_player_id].remove(kakan_pai_id)
            
            # ポン→加槓に更新 (ポンは3枚なので、+1枚で4枚)
            existing_meld = self.player_melds[naki_player_id][target_meld_index]
            new_meld_tiles = existing_meld[1] + [kakan_pai_id]
            self.player_melds[naki_player_id][target_meld_index] = ("加槓", new_meld_tiles, -1, existing_meld[3])
            
            # イベント追加
            self._add_event("N", player=naki_player_id, tile=kakan_pai_id, 
                            data={"naki_type": self.NAKI_TYPES["加槓"]})
            
            # 状態更新
            self._sort_hand(naki_player_id)
            self.current_player = naki_player_id
            self.naki_occurred_in_turn = True
            self.can_ron = False
            self.is_rinshan = True
            return
            
        elif naki_type_str == "暗槓":
            # 暗槓処理 (暗槓は4枚必要)
            base_index = tile_id_to_index(decoded_tiles[0]) if decoded_tiles else -1
            if base_index == -1:
                print(f"[Error] Ankan decode failed with m={meld_code}")
                return
                
            # 手牌から4枚揃っているか確認
            matching_tiles = [t for t in self.player_hands[naki_player_id] if tile_id_to_index(t) == base_index]
            
            if len(matching_tiles) < 4:  # 暗槓は手牌から4枚必要
                print(f"[Error] Not enough tiles for Ankan. Need 4, found {len(matching_tiles)}")
                return
                
            # 手牌から4枚削除
            for tile in matching_tiles[:4]:  # 4枚を削除
                self.player_hands[naki_player_id].remove(tile)
                
            # 暗槓を追加
            self.player_melds[naki_player_id].append(("暗槓", sorted(matching_tiles[:4]), -1, naki_player_id))
            
            # イベント追加
            self._add_event("N", player=naki_player_id, tile=matching_tiles[0], 
                            data={"naki_type": self.NAKI_TYPES["暗槓"]})
            
            # 状態更新
            self._sort_hand(naki_player_id)
            self.current_player = naki_player_id
            self.naki_occurred_in_turn = True
            self.can_ron = False
            self.is_rinshan = True
            return

        # --- チー・ポン・大明槓 ---
        trigger_player_abs = self.last_discard_event_player
        trigger_tile_id = self.last_discard_event_tile_id
        
        if trigger_player_abs == -1 or trigger_tile_id == -1:
            print(f"[Warn] Naki {naki_type_str} P{naki_player_id} has no last discard. m={meld_code}")
            return
            
        # 鳴きタイプに応じた処理
        if naki_type_str == "チー":
            # チーは相対位置が-1（上家）からのみ可能
            rel_pos = (trigger_player_abs - naki_player_id) % 4
            if rel_pos != 3:  # 上家以外からはチーできない
                print(f"[Error] Chi only allowed from player before you. P{naki_player_id} tried to Chi from P{trigger_player_abs} (rel_pos={rel_pos})")
                return
                
            # チー対象牌のインデックス
            trigger_index = tile_id_to_index(trigger_tile_id)
            if trigger_index == -1 or trigger_index >= 27:  # 字牌はチーできない
                print(f"[Error] Invalid tile for Chi: {tile_id_to_string(trigger_tile_id)}")
                return
                
            # チーの構成牌を探す（牌種が順子になるか）
            needed_indices = []
            suit = trigger_index // 9
            number = trigger_index % 9
            
            # チーの3つのケース
            if number <= 6:  # 12牌で3を鳴く
                case1 = [(suit * 9) + number - 2, (suit * 9) + number - 1]
                found1 = all(any(tile_id_to_index(t) == idx for t in self.player_hands[naki_player_id]) for idx in case1)
                if found1:
                    needed_indices = case1
                    
            if not needed_indices and number >= 1 and number <= 7:  # 13牌で2を鳴く
                case2 = [(suit * 9) + number - 1, (suit * 9) + number + 1]
                found2 = all(any(tile_id_to_index(t) == idx for t in self.player_hands[naki_player_id]) for idx in case2)
                if found2:
                    needed_indices = case2
                    
            if not needed_indices and number >= 2:  # 34牌で5を鳴く
                case3 = [(suit * 9) + number + 1, (suit * 9) + number + 2]
                found3 = all(any(tile_id_to_index(t) == idx for t in self.player_hands[naki_player_id]) for idx in case3)
                if found3:
                    needed_indices = case3
                    
            if not needed_indices:
                print(f"[Warn] Failed チー P{naki_player_id}. Trg:{tile_id_to_string(trigger_tile_id)} Hand:{[tile_id_to_string(t) for t in self.player_hands[naki_player_id]]}. m={meld_code}. Skipping.")
                return
                
            # 必要な牌を手牌から見つける
            consumed_tiles = []
            for idx in needed_indices:
                for tile in self.player_hands[naki_player_id]:
                    if tile_id_to_index(tile) == idx and tile not in consumed_tiles:
                        consumed_tiles.append(tile)
                        break
            
            if len(consumed_tiles) != len(needed_indices):
                print(f"[Error] Could not find all needed tiles for Chi. m={meld_code}")
                return
                
            # 手牌から削除
            for tile in consumed_tiles:
                self.player_hands[naki_player_id].remove(tile)
                
            # 鳴き追加 (チーは合計3枚：手牌2枚 + 捨て牌1枚)
            meld_tiles = sorted(consumed_tiles + [trigger_tile_id])
            self.player_melds[naki_player_id].append((naki_type_str, meld_tiles, trigger_tile_id, trigger_player_abs))
            
            # イベント追加
            self._add_event("N", player=naki_player_id, tile=trigger_tile_id,
                          data={"naki_type": self.NAKI_TYPES[naki_type_str], "from_who": trigger_player_abs})
        
        elif naki_type_str == "ポン":
            # ポン処理 (ポンは手牌2枚 + 捨て牌1枚 = 合計3枚)
            trigger_index = tile_id_to_index(trigger_tile_id)
            if trigger_index == -1:
                print(f"[Error] Invalid tile for Pon: {tile_id_to_string(trigger_tile_id)}")
                return
                
            # 手牌から同じ種類の牌を2枚探す
            matching_tiles = [t for t in self.player_hands[naki_player_id] if tile_id_to_index(t) == trigger_index]
            
            if len(matching_tiles) < 2:  # ポンには手牌に2枚必要
                print(f"[Error] Not enough tiles for Pon. Need 2, found {len(matching_tiles)}")
                return
                
            # 手牌から2枚削除
            for tile in matching_tiles[:2]:  # 2枚を削除
                self.player_hands[naki_player_id].remove(tile)
                
            # ポンを追加 (手牌2枚 + 捨て牌1枚 = 3枚)
            meld_tiles = sorted(matching_tiles[:2] + [trigger_tile_id])
            self.player_melds[naki_player_id].append((naki_type_str, meld_tiles, trigger_tile_id, trigger_player_abs))
            
            # イベント追加
            self._add_event("N", player=naki_player_id, tile=trigger_tile_id,
                          data={"naki_type": self.NAKI_TYPES[naki_type_str], "from_who": trigger_player_abs})
        
        elif naki_type_str == "大明槓":
            # 大明槓処理 (手牌3枚 + 捨て牌1枚 = 合計4枚)
            trigger_index = tile_id_to_index(trigger_tile_id)
            if trigger_index == -1:
                print(f"[Error] Invalid tile for Daiminkan: {tile_id_to_string(trigger_tile_id)}")
                return
                
            # 手牌から同じ種類の牌を3枚探す
            matching_tiles = [t for t in self.player_hands[naki_player_id] if tile_id_to_index(t) == trigger_index]
            
            if len(matching_tiles) < 3:  # 大明槓には手牌に3枚必要
                print(f"[Error] Not enough tiles for Daiminkan. Need 3, found {len(matching_tiles)}")
                return
                
            # 手牌から3枚削除
            for tile in matching_tiles[:3]:  # 3枚を削除
                self.player_hands[naki_player_id].remove(tile)
                
            # 大明槓を追加 (手牌3枚 + 捨て牌1枚 = 4枚)
            meld_tiles = sorted(matching_tiles[:3] + [trigger_tile_id])
            self.player_melds[naki_player_id].append((naki_type_str, meld_tiles, trigger_tile_id, trigger_player_abs))
            
            # イベント追加
            self._add_event("N", player=naki_player_id, tile=trigger_tile_id,
                          data={"naki_type": self.NAKI_TYPES[naki_type_str], "from_who": trigger_player_abs})
            
            # 嶺上牌フラグを立てる
            self.is_rinshan = True
        
        # 共通の状態更新
        self._sort_hand(naki_player_id)
        self.current_player = naki_player_id
        self.naki_occurred_in_turn = True
        self.can_ron = False
        self.last_discard_event_player = -1
        self.last_discard_event_tile_id = -1
        self.last_discard_event_tsumogiri = False

# EnhancedGameState クラスに以下のメソッドを追加

def get_static_features(self, player_id: int) -> np.ndarray:
    """
    現在の静的な状態をベクトル化する。向聴数なども含める。
    """
    from collections import defaultdict
    
    features = []

    # 1. 自分自身の手牌情報 (4チャネル: 各牌種4枚分 + 手牌にある枚数)
    my_hand_representation = np.zeros((NUM_TILE_TYPES, 5), dtype=np.int8)
    hand_indices_count = defaultdict(int)
    for tile_id in self.player_hands[player_id]:
         idx = tile_id_to_index(tile_id)
         offset = tile_id % 4
         my_hand_representation[idx, offset] = 1  # どの牌を持っているか
         hand_indices_count[idx] += 1
    for idx, count in hand_indices_count.items():
        my_hand_representation[idx, 4] = count  # 手牌にある枚数
    features.append(my_hand_representation.flatten())

    # 2. ドラ情報 (表示牌、現在のドラ牌種)
    dora_indicator_vec = np.zeros(NUM_TILE_TYPES, dtype=np.int8)
    for ind_id in self.dora_indicators:
         dora_indicator_vec[tile_id_to_index(ind_id)] = 1  # 複数ドラ表示に対応
    features.append(dora_indicator_vec)
    current_dora_indices = self.get_current_dora_indices()
    dora_tile_vec = np.zeros(NUM_TILE_TYPES, dtype=np.int8)
    for dora_idx in current_dora_indices:
         dora_tile_vec[dora_idx] = 1
    features.append(dora_tile_vec)

    # 3. 各プレイヤーの公開情報 (捨て牌、副露、リーチ状態、現物)
    # (相対プレイヤー順: 自分->下家->対面->上家)
    for p_offset in range(NUM_PLAYERS):
        target_player = (player_id + p_offset) % NUM_PLAYERS
        # 捨て牌 (各牌種が捨てられた枚数)
        discard_counts = np.zeros(NUM_TILE_TYPES, dtype=np.int8)
        # リーチ者の現物フラグ
        genbutsu_flag = np.zeros(NUM_TILE_TYPES, dtype=np.int8)
        is_target_reach = self.player_reach_status[target_player] == 2
        reach_discard_idx = self.player_reach_discard_index[target_player]
        for i, (tile, tsumogiri) in enumerate(self.player_discards[target_player]):
            tile_idx = tile_id_to_index(tile)
            if tile_idx != -1:
                discard_counts[tile_idx] += 1
                if is_target_reach and reach_discard_idx != -1 and i >= reach_discard_idx:
                     genbutsu_flag[tile_idx] = 1
        features.append(discard_counts)
        if p_offset != 0:  # 自分以外の現物情報を追加
            features.append(genbutsu_flag)

        # 副露 (種類ごとにフラグ + 構成牌) - より詳細化も可能
        meld_vec = np.zeros((len(self.NAKI_TYPES), NUM_TILE_TYPES), dtype=np.int8)
        for m_type, m_tiles, trigger_id, from_who_abs in self.player_melds[target_player]:
             if m_type in self.NAKI_TYPES:
                 naki_type_idx = self.NAKI_TYPES[m_type]
                 for tile_id in m_tiles:
                     tile_idx = tile_id_to_index(tile_id)
                     if tile_idx != -1:
                         meld_vec[naki_type_idx, tile_idx] = 1
        features.append(meld_vec.flatten())

        # リーチ状態 [status(0-2), 宣言巡目(正規化)]
        reach_stat = self.player_reach_status[target_player]
        reach_jun = self.player_reach_junme[target_player] / 18.0 if self.player_reach_junme[target_player] != -1 else 0
        features.append(np.array([reach_stat, reach_jun], dtype=np.float32))

    # 4. 場況情報 (局、本場、供託、巡目、親か、自風、残り牌山)
    round_wind_feat = self.round_num_wind / 7.0  # 東1-南4 を 0-1に正規化(近似)
    honba_feat = min(self.honba / 5.0, 1.0)  # 5本場までを正規化
    kyotaku_feat = min(self.kyotaku / 4.0, 1.0)  # 4本までを正規化
    junme_feat = min(self.junme / 18.0, 1.0)  # 18巡目までを正規化
    is_dealer_feat = 1.0 if self.dealer == player_id else 0.0
    my_wind = (player_id - self.dealer + NUM_PLAYERS) % NUM_PLAYERS  # 0:東, 1:南, 2:西, 3:北
    my_wind_vec = np.zeros(NUM_PLAYERS, dtype=np.float32)
    my_wind_vec[my_wind] = 1.0
    wall_count_feat = max(0.0, self.wall_tile_count / 70.0)  # 残り牌山割合

    ba_features = np.array([round_wind_feat, honba_feat, kyotaku_feat, junme_feat, is_dealer_feat, wall_count_feat], dtype=np.float32)
    features.append(ba_features)
    features.append(my_wind_vec)  # 自風 (one-hot)

    # 5. 点数情報 (全員の点数 - 正規化＆自分中心に回転)
    scores_feat = np.array(self.current_scores, dtype=np.float32)
    normalized_scores = scores_feat / 100000.0  # 10万点で正規化 (適宜調整)
    rotated_scores = np.roll(normalized_scores, -player_id)
    features.append(rotated_scores)

    # 6. 向聴数と受け入れ情報 (重要！)
    hand_indices = self.get_hand_indices(player_id)
    melds_info = self.get_melds_indices(player_id)

    # 向聴数計算部分（エラーハンドリングを強化）
    try:
        shanten, ukeire_indices = calculate_shanten(hand_indices, melds_info)
        if shanten < -1 or shanten > 8:  # 妥当な範囲かチェック
            print(f"[Warning] Calculated shanten value out of expected range: {shanten}. Using default.")
            shanten = 1
            ukeire_indices = []
    except Exception as e:
        # より詳細なエラー情報を出力して、デバッグしやすくする
        import traceback
        print(f"[Error] Shanten calculation failed: {e}")
        print(f"[Debug] Hand indices: {hand_indices}")
        print(f"[Debug] Melds info: {melds_info}")
        print(f"[Debug] Error traceback: {traceback.format_exc()}")
        shanten = 1  # エラー時は仮の向聴数（適当な値）
        ukeire_indices = []

    # One-hot エンコードされた向聴数ベクトル
    shanten_vec = np.zeros(9, dtype=np.float32)  # -1(和了) ~ 8向聴
    shanten_norm = max(0, min(shanten + 1, 8))  # 0(和了)~8(8向聴)に制限
    if 0 <= shanten_norm < len(shanten_vec):
        shanten_vec[shanten_norm] = 1.0
    else:
        # インデックス範囲外の場合は1向聴とする
        shanten_vec[1] = 1.0
        print(f"[Warning] Normalized shanten value {shanten_norm} out of range. Using 1.")

    # 受け入れ情報
    ukeire_vec = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
    for idx in ukeire_indices:
        if 0 <= idx < NUM_TILE_TYPES:  # 範囲チェック
            ukeire_vec[idx] = 1.0
        else:
            print(f"[Warning] Ukeire index {idx} out of range 0-{NUM_TILE_TYPES-1}")

    features.append(shanten_vec)
    features.append(ukeire_vec)
    
    # 受け入れ枚数（正規化）
    num_ukeire = len([idx for idx in ukeire_indices if 0 <= idx < NUM_TILE_TYPES])
    features.append(np.array([min(num_ukeire / 20.0, 1.0)], dtype=np.float32))

    # --- 全特徴量を結合 ---
    try:
        concatenated_features = np.concatenate([f.flatten() for f in features])
        if np.isnan(concatenated_features).any() or np.isinf(concatenated_features).any():
            print(f"[Error] NaN/Inf detected in static features for P{player_id}!")
            # NaN/Infが発生した場合のデバッグ情報を追加
            for i, f in enumerate(features):
                if np.isnan(f).any() or np.isinf(f).any():
                    print(f"  Problem in feature block {i}: {f}")
            # NaNを0で置換するなど応急処置
            concatenated_features = np.nan_to_num(concatenated_features, nan=0.0, posinf=1.0, neginf=-1.0)

        return concatenated_features
    except ValueError as e:
        print(f"[Error] Concatenating static features failed for P{player_id}.")
        # 各特徴量の形状を出力してデバッグ
        for i, f in enumerate(features):
            print(f"  Shape of feature block {i}: {np.array(f).shape}")
        raise e

def process_agari(self, attrib: dict):
    """AGARIタグの情報に基づいて状態を更新 (スコア、本場など)"""
    # とりあえず何もしない実装を追加（エラー防止用）
    print(f"[Debug] AGARI event with attributes: {attrib}")
    # TODO: 必要に応じて実装を追加
    pass
    
def process_ryuukyoku(self, attrib: dict):
    """RYUUKYOKUタグの情報に基づいて状態を更新 (スコア、本場など)"""
    # とりあえず何もしない実装を追加（エラー防止用）
    print(f"[Debug] RYUUKYOKU event with attributes: {attrib}")
    # TODO: 必要に応じて実装を追加
    pass
# extract_features_labels_for_imitation 関数内で、以下の行を変更：
# 変更前: game_state = GameState()
# 変更後: game_state = EnhancedGameState()

def extract_features_labels_for_imitation(xml_path: str):
    """
    天鳳XMLログから模倣学習用の (イベントシーケンス, 静的状態, 正解打牌) ペアを抽出する。
    """
    meta, rounds_data = parse_full_mahjong_log(xml_path)
    all_sequences = []
    all_static_features = []
    all_labels = []

    # ここを変更: GameState -> EnhancedGameState
    game_state = EnhancedGameState()
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