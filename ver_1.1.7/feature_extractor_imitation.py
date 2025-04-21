# feature_extractor_imitation.py の先頭に追加するための完全な独立実装版 EnhancedGameState

import numpy as np
from collections import defaultdict, deque
from naki_utils import decode_naki
from tile_utils import tile_id_to_index, tile_id_to_string
import full_mahjong_parser

# --- 独立実装版 GameState クラス ---
class EnhancedGameState:
    """完全に独立実装した GameState クラスの強化版"""
    
    # 定数定義
    TSUMO_TAGS = {"T": 0, "U": 1, "V": 2, "W": 3}
    DISCARD_TAGS = {"D": 0, "E": 1, "F": 2, "G": 3}
    
    # イベントタイプ定義
    EVENT_TYPES = {
        "INIT": 0,
        "TSUMO": 1,
        "DISCARD": 2,
        "N": 3,
        "REACH": 4,
        "DORA": 5,
        "PADDING": 8
    }
    
    # 鳴きタイプ定義
    NAKI_TYPES = {"チー": 0, "ポン": 1, "大明槓": 2, "加槓": 3, "暗槓": 4}
    
    def __init__(self):
        """初期化処理"""
        self.reset_state()
    
    def reset_state(self):
        """内部状態をリセット"""
        self.round_index = 0
        self.round_num_wind = 0
        self.honba = 0
        self.kyotaku = 0
        self.dealer = -1
        self.initial_scores = [25000] * NUM_PLAYERS
        self.dora_indicators = []
        self.current_scores = [25000] * NUM_PLAYERS
        self.player_hands = [[] for _ in range(NUM_PLAYERS)]
        self.player_discards = [[] for _ in range(NUM_PLAYERS)]
        self.player_melds = [[] for _ in range(NUM_PLAYERS)]
        self.player_reach_status = [0] * NUM_PLAYERS
        self.player_reach_junme = [-1] * NUM_PLAYERS
        self.player_reach_discard_index = [-1] * NUM_PLAYERS
        self.current_player = -1
        self.junme = 0
        self.last_discard_event_player = -1
        self.last_discard_event_tile_id = -1
        self.last_discard_event_tsumogiri = False
        self.can_ron = False
        self.naki_occurred_in_turn = False
        self.is_rinshan = False
        self.event_history = deque(maxlen=MAX_EVENT_HISTORY)
        self.wall_tile_count = 70
    
    def _add_event(self, event_type, player, tile=-1, data=None):
        """イベント履歴に情報を追加"""
        if data is None:
            data = {}
        
        event_code = self.EVENT_TYPES.get(event_type, -1)
        if event_code == -1:
            print(f"[Warning] Unknown event type: {event_type}")
            return
        
        # イベント情報を構造化
        event_info = {
            "type": event_code,
            "player": player,
            "tile_index": tile_id_to_index(tile) if tile != -1 else -1,
            "junme": int(np.ceil(self.junme)),
            "data": data
        }
        
        self.event_history.append(event_info)
    
    def init_round(self, round_data):
        """局の初期化"""
        self.reset_state()
        
        init_info = round_data.get("init", {})
        if not init_info:
            print("[Warning] No init info found.")
            return
        
        # 局情報の解析
        self.round_index = round_data.get("round_index", 0)
        seed_parts = init_info.get("seed", "0,0,0,0,0,0").split(",")
        
        try:
            self.round_num_wind = int(seed_parts[0])
            self.honba = int(seed_parts[1])
            self.kyotaku = int(seed_parts[2])
            dora_indicator_id = int(seed_parts[5])
            self.dora_indicators = [dora_indicator_id]
            self._add_event("DORA", player=-1, tile=dora_indicator_id)
        except (IndexError, ValueError) as e:
            print(f"[Warning] Failed parse seed: {e}")
            self.round_num_wind = 0
            self.honba = 0
            self.kyotaku = 0
            self.dora_indicators = []
        
        # 親と点数の設定
        self.dealer = int(init_info.get("oya", -1))
        self.current_player = self.dealer
        
        try:
            self.initial_scores = list(map(int, init_info.get("ten", "25000,25000,25000,25000").split(",")))
            self.current_scores = list(self.initial_scores)
        except ValueError as e:
            print(f"[Warning] Failed parse ten: {e}")
            self.initial_scores = [25000] * NUM_PLAYERS
            self.current_scores = [25000] * NUM_PLAYERS
        
        # 各プレイヤーの手牌初期化
        for p in range(NUM_PLAYERS):
            hand_str = init_info.get(f"hai{p}", "")
            self.player_hands[p] = []
            
            try:
                self.player_hands[p] = sorted(list(map(int, hand_str.split(","))) if hand_str else [])
            except ValueError as e:
                print(f"[Warning] Failed parse hai{p}: {e}")
        
        # INITイベント追加
        self._add_event("INIT", player=self.dealer, data={"round": self.round_num_wind, "honba": self.honba, "kyotaku": self.kyotaku})
        
        # 牌山の初期化
        self.wall_tile_count = 136 - 14 - (13 * NUM_PLAYERS)
    
    def _sort_hand(self, player_id):
        """手牌をソートする"""
        if 0 <= player_id < NUM_PLAYERS:
            self.player_hands[player_id].sort(key=lambda t: (tile_id_to_index(t), t))
    
    def process_tsumo(self, player_id, tile_id):
        """自摸処理"""
        if not (0 <= player_id < NUM_PLAYERS):
            print(f"[ERROR] Invalid player_id {player_id}")
            return
        
        self.current_player = player_id
        self.wall_tile_count -= 1
        
        if not self.is_rinshan:
            self.player_hands[player_id].append(tile_id)
            self._sort_hand(player_id)
            
            if not self.naki_occurred_in_turn:
                self.junme += 0.25
                
            self.naki_occurred_in_turn = False
        else:
            # 嶺上ツモの場合
            self.player_hands[player_id].append(tile_id)
            self._sort_hand(player_id)
            self.is_rinshan = False
            self.naki_occurred_in_turn = False
        
        self._add_event("TSUMO", player=player_id, tile=tile_id, data={"rinshan": self.is_rinshan})
        self.can_ron = False
        self.last_discard_event_player = -1
    
    def process_discard(self, player_id, tile_id, tsumogiri):
        """打牌処理"""
        if not (0 <= player_id < NUM_PLAYERS):
            print(f"[ERROR] Invalid player_id {player_id}")
            return
        
        try:
            self.player_hands[player_id].remove(tile_id)
            self._sort_hand(player_id)
            self.player_discards[player_id].append((tile_id, tsumogiri))
            self._add_event("DISCARD", player=player_id, tile=tile_id, data={"tsumogiri": tsumogiri})
            self.last_discard_event_player = player_id
            self.last_discard_event_tile_id = tile_id
            self.last_discard_event_tsumogiri = tsumogiri
            self.can_ron = True
            
            # リーチ成立処理
            if self.player_reach_status[player_id] == 1:
                self.player_reach_status[player_id] = 2
                self.player_reach_discard_index[player_id] = len(self.player_discards[player_id]) - 1
                self.kyotaku += 1
                self.current_scores[player_id] -= 1000
                self._add_event("REACH", player=player_id, data={"step": 2})
            
            self.current_player = (player_id + 1) % NUM_PLAYERS
            
        except ValueError:
            print(f"[ERROR] P{player_id} discard {tile_id_to_string(tile_id)} not in hand: {[tile_id_to_string(t) for t in self.player_hands[player_id]]}")
            # エラー時の状態更新
            self.player_discards[player_id].append((tile_id, tsumogiri))
            self._add_event("DISCARD", player=player_id, tile=tile_id, data={"tsumogiri": tsumogiri, "error": "not_in_hand"})
            self.last_discard_event_player = player_id
            self.last_discard_event_tile_id = tile_id
            self.last_discard_event_tsumogiri = tsumogiri
            self.can_ron = True
            self.current_player = (player_id + 1) % NUM_PLAYERS
    
    def process_naki(self, naki_player_id, meld_code):
        """鳴き処理 (イベント追加、壁カウンタ更新)"""
        naki_type_str, decoded_tiles, trigger_pai_id, _ = decode_naki(meld_code)
        if naki_type_str == "不明":
            print(f"[Warn] Unknown naki m={meld_code}")
            return
        
        # 鳴きイベントをログに記録
        clean_tiles = decoded_tiles[:3] if naki_type_str == "ポン" else decoded_tiles
        print(f"[Debug] Processing {naki_type_str} for P{naki_player_id}, decoded_tiles={[tile_id_to_string(t) for t in clean_tiles]}, trigger={tile_id_to_string(trigger_pai_id) if trigger_pai_id != -1 else 'N/A'}")
        
        # --- 加槓・暗槓 ---
        if naki_type_str == "加槓":
            # 加槓処理
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
            
            # ポン→加槓に更新
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
            # 暗槓処理
            base_index = tile_id_to_index(decoded_tiles[0]) if decoded_tiles else -1
            if base_index == -1:
                print(f"[Error] Ankan decode failed with m={meld_code}")
                return
                
            # 手牌から4枚揃っているか確認
            matching_tiles = [t for t in self.player_hands[naki_player_id] if tile_id_to_index(t) == base_index]
            
            if len(matching_tiles) < 4:
                print(f"[Error] Not enough tiles for Ankan. Need 4, found {len(matching_tiles)}")
                return
                
            # 手牌から4枚削除
            for tile in matching_tiles[:4]:
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
                
            # チーの構成牌を探す
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
                
            # 鳴き追加
            meld_tiles = sorted(consumed_tiles + [trigger_tile_id])
            self.player_melds[naki_player_id].append((naki_type_str, meld_tiles, trigger_tile_id, trigger_player_abs))
            
            # イベント追加
            self._add_event("N", player=naki_player_id, tile=trigger_tile_id,
                          data={"naki_type": self.NAKI_TYPES[naki_type_str], "from_who": trigger_player_abs})
        
        elif naki_type_str == "ポン":
            # ポン処理
            trigger_index = tile_id_to_index(trigger_tile_id)
            if trigger_index == -1:
                print(f"[Error] Invalid tile for Pon: {tile_id_to_string(trigger_tile_id)}")
                return
                
            # 手牌から同じ種類の牌を2枚探す
            matching_tiles = [t for t in self.player_hands[naki_player_id] if tile_id_to_index(t) == trigger_index]
            
            if len(matching_tiles) < 2:
                print(f"[Error] Not enough tiles for Pon. Need 2, found {len(matching_tiles)}")
                return
                
            # 手牌から2枚削除
            for tile in matching_tiles[:2]:
                self.player_hands[naki_player_id].remove(tile)
                
            # ポンを追加
            meld_tiles = sorted(matching_tiles[:2] + [trigger_tile_id])
            self.player_melds[naki_player_id].append((naki_type_str, meld_tiles, trigger_tile_id, trigger_player_abs))
            
            # イベント追加
            self._add_event("N", player=naki_player_id, tile=trigger_tile_id,
                          data={"naki_type": self.NAKI_TYPES[naki_type_str], "from_who": trigger_player_abs})
        
        elif naki_type_str == "大明槓":
            # 大明槓処理
            trigger_index = tile_id_to_index(trigger_tile_id)
            if trigger_index == -1:
                print(f"[Error] Invalid tile for Daiminkan: {tile_id_to_string(trigger_tile_id)}")
                return
                
            # 手牌から同じ種類の牌を3枚探す
            matching_tiles = [t for t in self.player_hands[naki_player_id] if tile_id_to_index(t) == trigger_index]
            
            if len(matching_tiles) < 3:
                print(f"[Error] Not enough tiles for Daiminkan. Need 3, found {len(matching_tiles)}")
                return
                
            # 手牌から3枚削除
            for tile in matching_tiles[:3]:
                self.player_hands[naki_player_id].remove(tile)
                
            # 大明槓を追加
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
    
    def process_dora(self, tile_id):
        """ドラ表示牌追加処理"""
        if tile_id != -1:
            self.dora_indicators.append(tile_id)
            self._add_event("DORA", player=-1, tile=tile_id)
    
    def process_agari(self, attrib):
        """和了処理"""
        print(f"[Debug] Processing AGARI with attributes: {attrib}")
        pass
    
    def process_ryuukyoku(self, attrib):
        """流局処理"""
        print(f"[Debug] Processing RYUUKYOKU with attributes: {attrib}")
        pass
    
    def get_current_dora_indices(self):
        """現在のドラ牌の牌種インデックスリストを取得"""
        dora_indices = []
        for indicator in self.dora_indicators:
            indicator_index = tile_id_to_index(indicator)
            dora_index = -1
            
            if 0 <= indicator_index <= 26:  # 数牌
                suit_base = (indicator_index // 9) * 9
                num = indicator_index % 9
                dora_index = suit_base + (num + 1) % 9  # 9->0
            elif 27 <= indicator_index <= 30:  # 風牌
                dora_index = 27 + (indicator_index - 27 + 1) % 4
            elif 31 <= indicator_index <= 33:  # 三元牌
                dora_index = 31 + (indicator_index - 31 + 1) % 3
                
            if dora_index != -1:
                dora_indices.append(dora_index)
                
        return dora_indices
    
    def get_hand_indices(self, player_id):
        """プレイヤーの手牌の牌種インデックスを取得"""
        return [tile_id_to_index(t) for t in self.player_hands[player_id]]
    
    def get_melds_indices(self, player_id):
        """プレイヤーの副露の牌種インデックスのリストを取得"""
        meld_indices = []
        for m_type, m_tiles, trigger_id, from_who_abs in self.player_melds[player_id]:
            meld_indices.append({
                "type": m_type,
                "tiles": [tile_id_to_index(t) for t in m_tiles],
                "trigger_index": tile_id_to_index(trigger_id) if trigger_id != -1 else -1,
                "from_who": from_who_abs
            })
        return meld_indices
    
    def get_event_sequence_features(self):
        """イベント履歴を固定長の数値ベクトルシーケンスに変換"""
        sequence = []
        event_specific_dim = 2  # DISCARD(1), N(2), REACH(1) の最大値
        
        for event in self.event_history:
            # 基本情報
            event_vec_base = [
                event["type"],
                event["player"] + 1,
                event["tile_index"] + 1,
                event["junme"],
            ]
            
            # 固有情報
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
            
            # 結合
            event_vec = event_vec_base + event_vec_specific
            sequence.append(event_vec)
        
        # パディング
        padding_length = MAX_EVENT_HISTORY - len(sequence)
        event_dim = 4 + event_specific_dim
        padding_vec = [self.EVENT_TYPES["PADDING"]] + [0] * (event_dim - 1)
        padded_sequence = sequence + [padding_vec] * padding_length
        
        try:
            return np.array(padded_sequence, dtype=np.float32)
        except ValueError as e:
            print(f"[Critical Error] Failed to convert padded_sequence to numpy array: {e}")
            
            # 次元調整
            fixed_sequence = []
            for vec in sequence:
                if len(vec) > event_dim:
                    vec = vec[:event_dim]
                elif len(vec) < event_dim:
                    vec = vec + [0] * (event_dim - len(vec))
                fixed_sequence.append(vec)
            
            fixed_padded = fixed_sequence + [padding_vec] * padding_length
            return np.array(fixed_padded, dtype=np.float32)
    
    def get_static_features(self, player_id):
        """現在の静的な状態をベクトル化"""
        features = []
        
        # 1. 自分自身の手牌情報
        my_hand_representation = np.zeros((NUM_TILE_TYPES, 5), dtype=np.int8)
        hand_indices_count = defaultdict(int)
        
        for tile_id in self.player_hands[player_id]:
            idx = tile_id_to_index(tile_id)
            offset = tile_id % 4
            my_hand_representation[idx, offset]
            offset = tile_id % 4
            my_hand_representation[idx, offset] = 1  # どの牌を持っているか
            hand_indices_count[idx] += 1
            
        for idx, count in hand_indices_count.items():
            my_hand_representation[idx, 4] = count  # 手牌にある枚数
            
        features.append(my_hand_representation.flatten())
        
        # 2. ドラ情報
        dora_indicator_vec = np.zeros(NUM_TILE_TYPES, dtype=np.int8)
        for ind_id in self.dora_indicators:
            dora_indicator_vec[tile_id_to_index(ind_id)] = 1
            
        features.append(dora_indicator_vec)
        
        current_dora_indices = self.get_current_dora_indices()
        dora_tile_vec = np.zeros(NUM_TILE_TYPES, dtype=np.int8)
        
        for dora_idx in current_dora_indices:
            dora_tile_vec[dora_idx] = 1
            
        features.append(dora_tile_vec)
        
        # 3. 各プレイヤーの公開情報
        for p_offset in range(NUM_PLAYERS):
            target_player = (player_id + p_offset) % NUM_PLAYERS
            
            # 捨て牌
            discard_counts = np.zeros(NUM_TILE_TYPES, dtype=np.int8)
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
            
            if p_offset != 0:  # 自分以外の現物情報
                features.append(genbutsu_flag)
            
            # 副露情報
            meld_vec = np.zeros((len(self.NAKI_TYPES), NUM_TILE_TYPES), dtype=np.int8)
            
            for m_type, m_tiles, trigger_id, from_who_abs in self.player_melds[target_player]:
                if m_type in self.NAKI_TYPES:
                    naki_type_idx = self.NAKI_TYPES[m_type]
                    for tile_id in m_tiles:
                        tile_idx = tile_id_to_index(tile_id)
                        if tile_idx != -1:
                            meld_vec[naki_type_idx, tile_idx] = 1
                            
            features.append(meld_vec.flatten())
            
            # リーチ状態
            reach_stat = self.player_reach_status[target_player]
            reach_jun = self.player_reach_junme[target_player] / 18.0 if self.player_reach_junme[target_player] != -1 else 0
            features.append(np.array([reach_stat, reach_jun], dtype=np.float32))
        
        # 4. 場況情報
        round_wind_feat = self.round_num_wind / 7.0
        honba_feat = min(self.honba / 5.0, 1.0)
        kyotaku_feat = min(self.kyotaku / 4.0, 1.0)
        junme_feat = min(self.junme / 18.0, 1.0)
        is_dealer_feat = 1.0 if self.dealer == player_id else 0.0
        my_wind = (player_id - self.dealer + NUM_PLAYERS) % NUM_PLAYERS
        
        my_wind_vec = np.zeros(NUM_PLAYERS, dtype=np.float32)
        my_wind_vec[my_wind] = 1.0
        
        wall_count_feat = max(0.0, self.wall_tile_count / 70.0)
        
        ba_features = np.array([round_wind_feat, honba_feat, kyotaku_feat, junme_feat, is_dealer_feat, wall_count_feat], dtype=np.float32)
        features.append(ba_features)
        features.append(my_wind_vec)
        
        # 5. 点数情報
        scores_feat = np.array(self.current_scores, dtype=np.float32)
        normalized_scores = scores_feat / 100000.0
        rotated_scores = np.roll(normalized_scores, -player_id)
        features.append(rotated_scores)
        
        # 6. 向聴数と受け入れ情報
        hand_indices = self.get_hand_indices(player_id)
        melds_info = self.get_melds_indices(player_id)
        
        try:
            shanten, ukeire_indices = calculate_shanten(hand_indices, melds_info)
            if shanten < -1 or shanten > 8:
                print(f"[Warning] Calculated shanten value out of expected range: {shanten}. Using default.")
                shanten = 1
                ukeire_indices = []
        except Exception as e:
            print(f"[Error] Shanten calculation failed: {e}")
            print(f"[Debug] Hand indices: {hand_indices}")
            print(f"[Debug] Melds info: {melds_info}")
            shanten = 1
            ukeire_indices = []
        
        # 向聴数のベクトル化
        shanten_vec = np.zeros(9, dtype=np.float32)
        shanten_norm = max(0, min(shanten + 1, 8))
        
        if 0 <= shanten_norm < len(shanten_vec):
            shanten_vec[shanten_norm] = 1.0
        else:
            shanten_vec[1] = 1.0
            
        features.append(shanten_vec)
        
        # 受け入れ情報
        ukeire_vec = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        
        for idx in ukeire_indices:
            if 0 <= idx < NUM_TILE_TYPES:
                ukeire_vec[idx] = 1.0
                
        features.append(ukeire_vec)
        
        # 受け入れ枚数
        num_ukeire = len([idx for idx in ukeire_indices if 0 <= idx < NUM_TILE_TYPES])
        features.append(np.array([min(num_ukeire / 20.0, 1.0)], dtype=np.float32))
        
        # 全特徴量を結合
        try:
            concatenated_features = np.concatenate([f.flatten() for f in features])
            
            if np.isnan(concatenated_features).any() or np.isinf(concatenated_features).any():
                print(f"[Error] NaN/Inf detected in static features for P{player_id}!")
                
                for i, f in enumerate(features):
                    if np.isnan(f).any() or np.isinf(f).any():
                        print(f"  Problem in feature block {i}: {f}")
                
                concatenated_features = np.nan_to_num(concatenated_features, nan=0.0, posinf=1.0, neginf=-1.0)
                
            return concatenated_features
            
        except ValueError as e:
            print(f"[Error] Concatenating static features failed for P{player_id}.")
            
            for i, f in enumerate(features):
                print(f"  Shape of feature block {i}: {np.array(f).shape}")
                
            raise e
    
    def get_valid_discard_options(self, player_id):
        """打牌可能な牌のリスト（インデックス 0-33）を返す。"""
        options = set()
        hand = self.player_hands[player_id]
        is_reach = self.player_reach_status[player_id] == 2
        tsumo_tile_id = hand[-1] if len(hand) % 3 == 2 else -1
        
        if is_reach:
            if tsumo_tile_id != -1:
                options.add(tile_id_to_index(tsumo_tile_id))
            else:
                print("[Warn] Reach hand invalid?")
        else:
            for tile in hand:
                options.add(tile_id_to_index(tile))
                
        return sorted(list(options))


# 以下は EnhancedGameState クラスの使用例

"""
# EnhancedGameState の使用例
game_state = EnhancedGameState()
game_state.init_round(round_data)

# イベント処理例
for event in round_data.get("events", []):
    tag = event["tag"]
    attrib = event["attrib"]
    
    # ツモ処理
    if tag.startswith("T"):
        tile_id = int(tag[1:])
        game_state.process_tsumo(0, tile_id)
    
    # 打牌処理
    elif tag.startswith("D"):
        tile_id = int(tag[1:])
        game_state.process_discard(0, tile_id, False)
    
    # 特徴量抽出
    features = game_state.get_static_features(0)
    event_seq = game_state.get_event_sequence_features()
"""

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