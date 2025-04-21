# game_state.py (Transformer対応版)
import numpy as np
from collections import defaultdict, deque  # dequeを使ってイベント履歴を管理
from full_mahjong_parser import parse_full_mahjong_log

try:
    from tile_utils import tile_id_to_index, tile_id_to_string
except ImportError:
    print("[Error] Cannot import from tile_utils.py.")
    def tile_id_to_index(tile: int) -> int: return tile // 4 if tile != -1 else -1
    def tile_id_to_string(tile: int) -> str: return f"tile_{tile}" if tile != -1 else "?"
try:
    from naki_utils import decode_naki  # tile_to_nameは不要になるかも
except ImportError:
    print("[Error] Cannot import from naki_utils.py.")
    def decode_naki(m: int) -> tuple: return "不明", [], -1, -1

# --- Shanten Calculation ---
def calculate_shanten(hand_indices, melds=[]):
    """
    手牌から向聴数と受け入れ牌を計算する関数（簡易版）
    
    Args:
        hand_indices (list): 牌種インデックス(0-33)のリスト
        melds (list): 副露情報のリスト [{type: str, tiles: list, ...}, ...]
    
    Returns:
        tuple: (向聴数, 受け入れ牌種インデックスのリスト)
    """
    try:
        # 実際の向聴数計算ライブラリが未実装の場合のフォールバック
        # エラーが出ないように、仮の向聴数と受け入れ牌を返す
        
        # 手牌のカウント（同じインデックスの牌が何枚あるか）
        hand_count = [0] * 34
        for idx in hand_indices:
            if 0 <= idx < 34:  # 範囲チェック
                hand_count[idx] += 1
        
        # 副露も考慮（暗槓・加槓・ポン・チーなど）
        melds_count = 0
        for meld in melds:
            if isinstance(meld, dict) and 'tiles' in meld:
                melds_count += 1
                for idx in meld['tiles']:
                    if 0 <= idx < 34:  # 範囲チェック
                        hand_count[idx] += 1
            
        # 副露を含めた枚数
        total_tiles = sum(hand_count)
        
        # 向聴数の簡易計算（実際は複雑な計算が必要）
        # 面子の数の近似値を計算
        pair_count = sum(1 for count in hand_count if count >= 2)
        potential_sets = sum(count // 3 for count in hand_count)
        
        # チートイツ形の向聴数（仮の計算）
        chitoitsu_shanten = 6 - min(pair_count, 7)
        
        # 通常形の向聴数（仮の計算）
        # 4面子1雀頭で和了 = 必要な面子数は4
        regular_shanten = 8 - (pair_count > 0) - (2 * potential_sets) - melds_count
        
        # 最小の向聴数を採用
        shanten = min(chitoitsu_shanten, regular_shanten)
        
        # 受け入れ牌の簡易計算（実際は向聴数減少に繋がる牌を特定する必要あり）
        ukeire = []
        
        # 各牌を1枚追加したときに向聴数が下がるか検証（単純な近似）
        for idx in range(34):
            # 既に4枚ある牌は除外
            if hand_count[idx] >= 4:
                continue
                
            # 雀頭が無い場合、雀頭候補になる可能性
            if pair_count == 0 and hand_count[idx] == 1:
                ukeire.append(idx)
            # 面子が作れる可能性（単純化）
            elif hand_count[idx] % 3 == 2:
                ukeire.append(idx)
            # 順子が作れる可能性（単純化）
            elif idx < 27:  # 数牌のみ
                suit = idx // 9
                number = idx % 9
                
                # 両面搭子
                if number <= 6 and hand_count[idx] > 0 and hand_count[idx+1] > 0:
                    if number+2 < 9:  # 範囲チェック
                        ukeire.append(suit*9 + number+2)
                if number >= 2 and hand_count[idx] > 0 and hand_count[idx-1] > 0:
                    ukeire.append(suit*9 + number-2)
                
                # 嵌張搭子
                if number <= 7 and hand_count[idx] > 0 and hand_count[idx+2] > 0:
                    ukeire.append(suit*9 + number+1)
        
        # 重複削除
        ukeire = list(set(ukeire))
        
        return max(shanten, 0), ukeire  # 向聴数が負になることは通常ない
        
    except Exception as e:
        # エラーが発生した場合はログに出力し、デフォルト値を返す
        print(f"[Debug] Shanten calculation error details: {e}")
        return 1, []  # 適当な向聴数と空の受け入れリストを返す


NUM_PLAYERS = 4
NUM_TILE_TYPES = 34
MAX_EVENT_HISTORY = 60  # 考慮する最大イベント履歴数 (Transformerのシーケンス長)

class GameState:
    TSUMO_TAGS = {"T": 0, "U": 1, "V": 2, "W": 3}
    DISCARD_TAGS = {"D": 0, "E": 1, "F": 2, "G": 3}

    # イベントタイプ定義 (数値にマッピング)
    EVENT_TYPES = {
        "INIT": 0,     # 局開始 (ダミーイベントとして履歴の開始点に使うかも)
        "TSUMO": 1,
        "DISCARD": 2,
        "N": 3,        # 鳴き (チー、ポン、大明槓、加槓、暗槓を区別するか検討)
        "REACH": 4,
        "DORA": 5,
        # "AGARI": 6,   # 予測時には不要だが履歴としてはありえる
        # "RYUUKYOKU": 7,
        "PADDING": 8   # シーケンス長を揃えるためのパディング用
    }
    # NAKIタイプも数値化 (必要なら)
    NAKI_TYPES = {"チー": 0, "ポン": 1, "大明槓": 2, "加槓": 3, "暗槓": 4}

    def __init__(self):
        self.reset_state()

    def reset_state(self):
        """GameStateの内部状態をリセット"""
        self.round_index = 0
        self.round_num_wind = 0
        self.honba = 0
        self.kyotaku = 0
        self.dealer = -1
        self.initial_scores = [25000]*NUM_PLAYERS
        self.dora_indicators = []  # ドラ表示牌IDのリスト
        self.current_scores = [25000]*NUM_PLAYERS
        self.player_hands = [[] for _ in range(NUM_PLAYERS)]  # 各プレイヤーの手牌IDリスト
        self.player_discards = [[] for _ in range(NUM_PLAYERS)]  # 各プレイヤーの捨て牌 (tile_id, tsumogiri_flag) のリスト
        self.player_melds = [[] for _ in range(NUM_PLAYERS)]  # 各プレイヤーの副露情報 [(type_str, tile_ids, trigger_id, from_who_abs)] のリスト
        self.player_reach_status = [0]*NUM_PLAYERS  # 0:未, 1:宣言中, 2:成立
        self.player_reach_junme = [-1]*NUM_PLAYERS
        self.player_reach_discard_index = [-1]*NUM_PLAYERS
        self.current_player = -1
        self.junme = 0  # 1巡を1.0とする (0.25刻み)
        self.last_discard_event_player = -1
        self.last_discard_event_tile_id = -1
        self.last_discard_event_tsumogiri = False
        self.can_ron = False
        self.naki_occurred_in_turn = False
        self.is_rinshan = False
        # --- Transformer用 ---
        self.event_history = deque(maxlen=MAX_EVENT_HISTORY)  # イベント履歴 (固定長)
        self.wall_tile_count = 70  # 牌山の残り枚数 (王牌14枚を除く)
        # TODO: 各牌の残り枚数カウンタ (見えていない牌) の実装

    def _add_event(self, event_type: str, player: int, tile: int = -1, data: dict = None):
        """イベント履歴に情報を追加"""
        if data is None: data = {}
        event_code = self.EVENT_TYPES.get(event_type, -1)
        if event_code == -1:
            print(f"[Warning] Unknown event type: {event_type}")
            return
        # イベント情報の構造化 (例)
        event_info = {
            "type": event_code,
            "player": player,  # イベント主体者 (-1なら全体イベント)
            "tile_index": tile_id_to_index(tile) if tile != -1 else -1,  # 関連牌 (-1ならなし)
            "junme": int(np.ceil(self.junme)),  # 巡目
            "data": data  # その他情報 (鳴き種類、リーチステップなど)
        }
        self.event_history.append(event_info)

    def init_round(self, round_data: dict):
        """局の初期化 (イベント履歴の初期化も含む)"""
        self.reset_state()  # 内部状態をクリア
        init_info = round_data.get("init", {})
        if not init_info: print("[Warning] No init info found."); return

        # ... (既存の初期化処理はほぼそのまま) ...
        self.round_index = round_data.get("round_index", 0)
        seed_parts = init_info.get("seed", "0,0,0,0,0,0").split(",")
        try:
            self.round_num_wind = int(seed_parts[0])
            self.honba = int(seed_parts[1])
            self.kyotaku = int(seed_parts[2])
            dora_indicator_id = int(seed_parts[5])
            self.dora_indicators = [dora_indicator_id]
            self._add_event("DORA", player=-1, tile=dora_indicator_id)  # ドラ表示イベントを追加
        except (IndexError, ValueError) as e: 
            print(f"[Warning] Failed parse seed: {e}")
            self.round_num_wind = 0
            self.honba = 0
            self.kyotaku = 0
            self.dora_indicators = []
            
        self.dealer = int(init_info.get("oya", -1))
        self.current_player = self.dealer
        try:
            self.initial_scores = list(map(int, init_info.get("ten", "25000,25000,25000,25000").split(",")))
            self.current_scores = list(self.initial_scores)
        except ValueError as e: 
            print(f"[Warning] Failed parse ten: {e}")
            self.initial_scores = [25000]*NUM_PLAYERS
            self.current_scores = [25000]*NUM_PLAYERS
            
        for p in range(NUM_PLAYERS):
            hand_str = init_info.get(f"hai{p}", "")
            self.player_hands[p] = []
            try: 
                self.player_hands[p] = sorted(list(map(int, hand_str.split(","))) if hand_str else [])
            except ValueError as e: 
                print(f"[Warning] Failed parse hai{p}: {e}")
            # 他のプレイヤー変数の初期化はreset_stateで実施済み

        # INITイベント (場の情報を含む)
        self._add_event("INIT", player=self.dealer, data={"round": self.round_num_wind, "honba": self.honba, "kyotaku": self.kyotaku})
        # 壁の初期化
        self.wall_tile_count = 136 - 14 - (13 * NUM_PLAYERS)  # (王牌14枚 + 配牌52枚)

    def _sort_hand(self, player_id):
        """手牌をソートする"""
        if 0 <= player_id < NUM_PLAYERS:
            self.player_hands[player_id].sort(key=lambda t: (tile_id_to_index(t), t))

    def process_tsumo(self, player_id: int, tile_id: int):
        """自摸処理 (イベント追加、壁カウンタ更新)"""
        if not (0 <= player_id < NUM_PLAYERS): 
            print(f"[ERROR] Invalid player_id {player_id}")
            return
            
        self.current_player = player_id
        self.wall_tile_count -= 1  # 壁牌を1枚消費

        if not self.is_rinshan:
            # ... (手牌整合性チェック) ...
            self.player_hands[player_id].append(tile_id)
            self._sort_hand(player_id)
            if not self.naki_occurred_in_turn: 
                self.junme += 0.25
            self.naki_occurred_in_turn = False
        else:  # 嶺上ツモ
            self.player_hands[player_id].append(tile_id)
            self._sort_hand(player_id)
            self.is_rinshan = False
            self.naki_occurred_in_turn = False

        self._add_event("TSUMO", player=player_id, tile=tile_id, data={"rinshan": self.is_rinshan})
        self.can_ron = False
        self.last_discard_event_player = -1  # ...リセット処理...

    def process_discard(self, player_id: int, tile_id: int, tsumogiri: bool):
        """打牌処理 (イベント追加)"""
        if not (0 <= player_id < NUM_PLAYERS): 
            print(f"[ERROR] Invalid player_id {player_id}")
            return
        # ... (手牌整合性チェック) ...

        try:
            self.player_hands[player_id].remove(tile_id)
            self._sort_hand(player_id)
            self.player_discards[player_id].append((tile_id, tsumogiri))
            self._add_event("DISCARD", player=player_id, tile=tile_id, data={"tsumogiri": tsumogiri})  # ★イベント追加
            self.last_discard_event_player = player_id
            self.last_discard_event_tile_id = tile_id
            self.last_discard_event_tsumogiri = tsumogiri
            self.can_ron = True

            # --- リーチ成立処理 (点数減算のバグ修正、イベント追加) ---
            if self.player_reach_status[player_id] == 1:
                self.player_reach_status[player_id] = 2
                self.player_reach_discard_index[player_id] = len(self.player_discards[player_id]) - 1
                self.kyotaku += 1
                self.current_scores[player_id] -= 1000  # 正しく1000点減らす
                # リーチ成立イベントも追加
                self._add_event("REACH", player=player_id, data={"step": 2})

            self.current_player = (player_id + 1) % NUM_PLAYERS
        except ValueError:
            print(f"[ERROR] P{player_id} discard {tile_id_to_string(tile_id)} not in hand: {[tile_id_to_string(t) for t in self.player_hands[player_id]]}")
            # エラーでもイベントは記録し、状態を進める
            self.player_discards[player_id].append((tile_id, tsumogiri))
            self._add_event("DISCARD", player=player_id, tile=tile_id, data={"tsumogiri": tsumogiri, "error": "not_in_hand"})
            self.last_discard_event_player = player_id
            self.last_discard_event_tile_id = tile_id
            self.last_discard_event_tsumogiri = tsumogiri
            self.can_ron = True
            self.current_player = (player_id + 1) % NUM_PLAYERS

    # game_state.py の process_naki メソッドの修正版

def process_naki(self, naki_player_id: int, meld_code: int):
    """鳴き処理 (イベント追加、壁カウンタ更新)。麻雀ルールに合わせて修正。"""
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

    def process_reach(self, player_id: int, step: int):
        """リーチ処理 (イベント追加はstep=1の宣言時とstep=2の成立時(discard内))"""
        if step == 1:
            if self.current_scores[player_id] >= 1000 and self.player_reach_status[player_id] == 0:
                self.player_reach_status[player_id] = 1  # 宣言中
                self.player_reach_junme[player_id] = self.junme
                # リーチ宣言イベントを追加
                self._add_event("REACH", player=player_id, data={"step": 1, "junme": self.junme})
            else:
                status = self.player_reach_status[player_id]
                score = self.current_scores[player_id]
                print(f"[Warn] P{player_id} reach step 1 failed: score {score}, status {status}.")
        # step 2 (成立) は process_discard 内で処理 & イベント追加

    def process_dora(self, tile_id: int):
        """ドラ表示牌追加処理"""
        if tile_id != -1:
             self.dora_indicators.append(tile_id)
             self._add_event("DORA", player=-1, tile=tile_id)
             # 新ドラ処理の場合、壁カウンタも減らす (槓ドラ)
             # self.wall_tile_count -= 1 # 必要に応じて

    def process_agari(self, attrib: dict):
        """AGARIタグの情報に基づいて状態を更新 (スコア、本場など)"""
        # ... (既存の処理) ...
        # 必要ならAGARIイベントを履歴に追加
        # who = int(attrib.get("who", -1))
        # fromWho = int(attrib.get("fromWho", -1))
        # self._add_event("AGARI", player=who, data={"from": fromWho, ...})

    def process_ryuukyoku(self, attrib: dict):
        """RYUUKYOKUタグの情報に基づいて状態を更新 (スコア、本場など)"""
        # ... (既存の処理) ...
        # 必要ならRYUUKYOKUイベントを履歴に追加
        # self._add_event("RYUUKYOKU", player=-1, data={...})

    def get_current_dora_indices(self) -> list:
        """現在のドラ牌の牌種インデックス(0-33)リストを取得"""
        dora_indices = []
        for indicator in self.dora_indicators:
            indicator_index = tile_id_to_index(indicator)
            dora_index = -1
            if 0 <= indicator_index <= 26:  # 数牌
                suit_base = (indicator_index // 9) * 9
                num = indicator_index % 9
                dora_index = suit_base + (num + 1) % 9  # 9->0
            elif 27 <= indicator_index <= 30:  # 風牌 (東南西北)
                dora_index = 27 + (indicator_index - 27 + 1) % 4
            elif 31 <= indicator_index <= 33:  # 三元牌 (白発中)
                dora_index = 31 + (indicator_index - 31 + 1) % 3
            if dora_index != -1:
                dora_indices.append(dora_index)
        return dora_indices

    def get_hand_indices(self, player_id):
        """プレイヤーの手牌を牌種インデックス(0-33)のリストで取得"""
        return [tile_id_to_index(t) for t in self.player_hands[player_id]]

    def get_melds_indices(self, player_id):
        """プレイヤーの副露を牌種インデックス(0-33)のリストのリストで取得"""
        meld_indices = []
        for m_type, m_tiles, trigger_id, from_who_abs in self.player_melds[player_id]:
             meld_indices.append({
                 "type": m_type,  # 文字列のまま or 数値コード
                 "tiles": [tile_id_to_index(t) for t in m_tiles],
                 "trigger_index": tile_id_to_index(trigger_id) if trigger_id != -1 else -1,
                 "from_who": from_who_abs
             })
        return meld_indices

    def get_event_sequence_features(self) -> np.ndarray:
        """
        イベント履歴を固定長の数値ベクトルシーケンスに変換する。
        各イベントベクトルは [type, player, tile_index, junme, data...] の形式。
        """
        sequence = []
        # イベント固有情報の最大長を定義
        event_specific_dim = 2  # DISCARD(1), N(2), REACH(1) の最大値

        for event in self.event_history:
            # 基本情報 (4次元)
            event_vec_base = [
                event["type"],
                event["player"] + 1,  # 0は全体イベント用
                event["tile_index"] + 1,  # 0は牌なし用
                event["junme"],
            ]

            # イベントタイプ固有情報 (最大 event_specific_dim 次元)
            event_vec_specific = [0] * event_specific_dim  # まず0で初期化
            data = event.get("data", {})  # dataがない場合も考慮

            event_type_code = event["type"]  # コードで比較

            if event_type_code == self.EVENT_TYPES["DISCARD"]:
                event_vec_specific[0] = 1 if data.get("tsumogiri", False) else 0
                # 残り event_specific_dim - 1 は 0 のまま
            elif event_type_code == self.EVENT_TYPES["N"]:
                event_vec_specific[0] = data.get("naki_type", -1) + 1
                event_vec_specific[1] = data.get("from_who", -1) + 1
            elif event_type_code == self.EVENT_TYPES["REACH"]:
                event_vec_specific[0] = data.get("step", 0)
                # 残り event_specific_dim - 1 は 0 のまま

            # 基本情報と固有情報を結合
            event_vec = event_vec_base + event_vec_specific
            sequence.append(event_vec)

        # Padding
        padding_length = MAX_EVENT_HISTORY - len(sequence)
        # event_vec の次元数を確定させる (基本4次元 + 固有情報 event_specific_dim 次元)
        event_dim = 4 + event_specific_dim  # この例だと 4 + 2 = 6 次元
        padding_vec = [self.EVENT_TYPES["PADDING"]] + [0] * (event_dim - 1)
        padded_sequence = sequence + [padding_vec] * padding_length

        try:
            # NumPy配列に変換
            return np.array(padded_sequence, dtype=np.float32)
        except ValueError as e:
            print(f"[Critical Error] Failed to convert padded_sequence to numpy array: {e}")
            # 次元が揃っていない場合に強制的に同じ長さにする
            fixed_sequence = []
            for vec in sequence:
                if len(vec) > event_dim:
                    vec = vec[:event_dim]  # 長すぎる場合は切り詰める
                elif len(vec) < event_dim:
                    vec = vec + [0] * (event_dim - len(vec))  # 短すぎる場合は0でパディング
                fixed_sequence.append(vec)
                
            fixed_padded = fixed_sequence + [padding_vec] * padding_length
            return np.array(fixed_padded, dtype=np.float32)

    def get_static_features(self, player_id: int) -> np.ndarray:
        """
        現在の静的な状態をベクトル化する。向聴数なども含める。
        """
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

    def get_valid_discard_options(self, player_id: int) -> list:
        """打牌可能な牌のリスト（インデックス 0-33）を返す。(既存のままでOK)"""
        options = set()
        hand = self.player_hands[player_id]
        is_reach = self.player_reach_status[player_id] == 2
        tsumo_tile_id = hand[-1] if len(hand) % 3 == 2 else -1  # ツモ牌を特定

        if is_reach:
            # リーチ後はツモ切り固定 (暗槓可能な場合は選択肢が増えるが、ここでは単純化)
            # TODO: リーチ後の暗槓判断を追加する場合、ここを修正
            if tsumo_tile_id != -1:
                 options.add(tile_id_to_index(tsumo_tile_id))
            else: 
                print("[Warn] Reach hand invalid?")  # 手牌が13枚など異常
        else:
             # リーチ中でなければ手牌から選択可能
             for tile in hand:
                  options.add(tile_id_to_index(tile))
        return sorted(list(options))