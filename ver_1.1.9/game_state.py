# /ver_1.1.9/game_state.py
import numpy as np
from collections import defaultdict, deque
import sys
import traceback # For detailed error reporting
import os # For checking file existence if needed

# --- Dependency Imports ---
try:
    # Ensure these are in the same directory (ver_1.1.9) or Python path
    from tile_utils import tile_id_to_index, tile_id_to_string, tile_index_to_id
    from naki_utils import decode_naki
except ImportError as e:
    print(f"[FATAL ERROR in game_state.py] Cannot import from tile_utils/naki_utils: {e}")
    print("Ensure tile_utils.py and naki_utils.py are in the same directory or Python path.")
    # Attempting to continue with dummy functions if imports fail, but functionality will be broken.
    # It's better to fix the imports.
    def tile_id_to_index(tile: int) -> int: return tile // 4 if tile >= 0 else -1
    def tile_id_to_string(tile: int) -> str: return f"ERR_TILE_{tile}"
    def tile_index_to_id(index: int, prefer_red: bool = False) -> int: return index * 4 if index >= 0 else -1
    def decode_naki(m: int) -> dict: return {"type": "不明", "tiles": [], "from_who_relative": -1, "consumed": [], "raw_value": m}
    # sys.exit(1) # Consider exiting if imports fail critically
# --- End Dependency Imports ---

# --- Constants ---
NUM_PLAYERS = 4
NUM_TILE_TYPES = 34         # 0-33 for different tile kinds
MAX_EVENT_HISTORY = 60      # Max sequence length for Transformer input
STATIC_FEATURE_DIM = 157    # Updated dimension after removing shanten/ukeire
# Event types for event history sequence encoding
EVENT_TYPES = {
    "INIT": 0, "TSUMO": 1, "DISCARD": 2, "N": 3, "REACH": 4,
    "DORA": 5, "AGARI": 6, "RYUUKYOKU": 7, "PADDING": 8
}
# Naki types for feature encoding (consistent with naki_utils)
NAKI_TYPES = {"チー": 0, "ポン": 1, "大明槓": 2, "加槓": 3, "暗槓": 4, "不明": -1}
# --- End Constants ---

class GameState:
    """
    Manages the state of a Mahjong game round, processes events parsed
    from XML, and generates feature vectors for ML models (without shanten).
    Includes enhanced logging for debugging state inconsistencies.
    """
    TSUMO_TAGS = {"T": 0, "U": 1, "V": 2, "W": 3}
    DISCARD_TAGS = {"D": 0, "E": 1, "F": 2, "G": 3}

    def __init__(self):
        """Initializes the GameState."""
        self.reset_state()

    def reset_state(self):
        """Resets all internal state variables to default values."""
        self.round_index: int = 0
        self.round_num_wind: int = 0
        self.honba: int = 0
        self.kyotaku: int = 0
        self.dealer: int = -1
        self.initial_scores: list[int] = [25000] * NUM_PLAYERS
        self.dora_indicators: list[int] = []
        self.current_scores: list[int] = [25000] * NUM_PLAYERS
        self.player_hands: list[list[int]] = [[] for _ in range(NUM_PLAYERS)]
        self.player_discards: list[list[tuple[int, bool]]] = [[] for _ in range(NUM_PLAYERS)]
        self.player_melds: list[list[dict]] = [[] for _ in range(NUM_PLAYERS)]
        self.player_reach_status: list[int] = [0] * NUM_PLAYERS
        self.player_reach_junme: list[float] = [-1.0] * NUM_PLAYERS
        self.player_reach_discard_index: list[int] = [-1] * NUM_PLAYERS
        self.current_player: int = -1
        self.junme: float = 0.0
        self.last_discard_event_player: int = -1
        self.last_discard_event_tile_id: int = -1
        self.last_discard_event_tsumogiri: bool = False
        self.can_ron: bool = False
        self.naki_occurred_in_turn: bool = False
        self.is_rinshan: bool = False
        self.event_history: deque = deque(maxlen=MAX_EVENT_HISTORY)
        self.wall_tile_count: int = 70

    def _add_event(self, event_type: str, player: int, tile: int = -1, data: dict = None):
        """Adds a structured event to the event history deque."""
        if data is None: data = {}
        event_code = EVENT_TYPES.get(event_type, -1)
        if event_code == -1: return

        # Determine the correct dimension based on event type for specific data
        event_specific_dim = 0
        if event_type == "DISCARD": event_specific_dim = 1 # tsumogiri
        elif event_type == "N": event_specific_dim = 2 # naki_type, from_who
        elif event_type == "REACH": event_specific_dim = 2 # step, junme
        # Add more cases if needed

        event_info = {
            "type": event_code,
            "player": player,
            "tile_index": tile_id_to_index(tile),
            "junme": int(np.ceil(self.junme)),
            "data": data, # Store raw data dict
            "_event_specific_dim": event_specific_dim # Store expected dim for later use
        }
        self.event_history.append(event_info)

    def _sort_hand(self, player_id):
        """Sorts the hand of the specified player."""
        if 0 <= player_id < NUM_PLAYERS:
            self.player_hands[player_id].sort(key=lambda t: (tile_id_to_index(t), t))

    def init_round(self, round_data: dict):
        """Initializes the game state for a new round based on parsed data."""
        self.reset_state()
        init_info = round_data.get("init", {})
        if not init_info:
            print("[Error] No 'init' info found in round_data for init_round.")
            return

        self.round_index = round_data.get("round_index", 0)
        seed_parts = init_info.get("seed", "0,0,0,0,0,0").split(",")
        try:
            if len(seed_parts) >= 6:
                self.round_num_wind = int(seed_parts[0]); self.honba = int(seed_parts[1]); self.kyotaku = int(seed_parts[2])
                dora_indicator_id = int(seed_parts[5])
                if 0 <= dora_indicator_id <= 135: self.dora_indicators = [dora_indicator_id]; self._add_event("DORA", player=-1, tile=dora_indicator_id)
                else: print(f"[Warning] Invalid initial Dora ID: {dora_indicator_id}"); self.dora_indicators = []
            else: raise ValueError("Seed string too short")
        except (IndexError, ValueError) as e:
            print(f"[Warning] Failed parse seed '{init_info.get('seed')}': {e}"); self.round_num_wind=0; self.honba=0; self.kyotaku=0; self.dora_indicators=[]

        self.dealer = int(init_info.get("oya", -1))
        if not (0 <= self.dealer < NUM_PLAYERS): print(f"[Warning] Invalid dealer index '{init_info.get('oya')}'. Setting to 0."); self.dealer = 0
        self.current_player = self.dealer

        try:
            raw_scores = init_info.get("ten", "250,250,250,250").split(",") # Default to x100 scores if format unclear
            if len(raw_scores) == 4: self.initial_scores = [int(float(s)) * 100 for s in raw_scores]; self.current_scores = list(self.initial_scores)
            else: raise ValueError("Incorrect score count")
        except (ValueError, TypeError) as e:
            print(f"[Warning] Failed parse 'ten' '{init_info.get('ten')}': {e}"); self.initial_scores=[25000]*NUM_PLAYERS; self.current_scores=[25000]*NUM_PLAYERS

        for p in range(NUM_PLAYERS):
            hand_str = init_info.get(f"hai{p}", ""); self.player_hands[p] = []
            try:
                if hand_str:
                    hand_ids = [int(h) for h in hand_str.split(',') if h.strip()] # Handle empty strings from split
                    valid_hand_ids = [tid for tid in hand_ids if 0 <= tid <= 135]
                    if len(valid_hand_ids) != len(hand_ids): print(f"[Warning] Invalid tile IDs in initial hand P{p}: {hand_str}")
                    self.player_hands[p] = valid_hand_ids; self._sort_hand(p)
            except ValueError as e: print(f"[Warning] Failed parse initial hand 'hai{p}': {e}")

        initial_hand_sum = sum(len(h) for h in self.player_hands)
        if initial_hand_sum > 52: print(f"[Warning] Initial hands sum {initial_hand_sum} > 52."); initial_hand_sum = 52
        self.wall_tile_count = 136 - 14 - initial_hand_sum; self.junme = 0.0
        init_data = {"round": self.round_num_wind, "honba": self.honba, "kyotaku": self.kyotaku}
        self._add_event("INIT", player=self.dealer, data=init_data)

    def process_tsumo(self, player_id: int, tile_id: int):
        if not (0 <= player_id < NUM_PLAYERS): print(f"[ERROR] Invalid player_id {player_id} in process_tsumo"); return
        if not (0 <= tile_id <= 135): print(f"[ERROR] Invalid tile_id {tile_id} in process_tsumo"); return
        self.current_player = player_id
        is_first_round = self.junme < 1.0; is_dealer_turn = player_id == self.dealer
        if not self.is_rinshan:
            if is_first_round and is_dealer_turn and self.junme == 0.0: self.junme = 0.1
            elif not is_first_round and player_id == 0: self.junme = np.floor(self.junme) + 1.0
            elif is_first_round and not is_dealer_turn and self.junme < 1.0:
                 if self.junme == 0.1: self.junme = 1.0
        rinshan_draw = self.is_rinshan
        if rinshan_draw: self.is_rinshan = False
        else:
             if self.wall_tile_count > 0: self.wall_tile_count -= 1
             else: print("[Warning] Tsumo with 0 wall tiles remaining.")
        self.naki_occurred_in_turn = False
        self.player_hands[player_id].append(tile_id); self._sort_hand(player_id)
        tsumo_data = {"rinshan": rinshan_draw}; self._add_event("TSUMO", player=player_id, tile=tile_id, data=tsumo_data)
        self.can_ron = False

    def process_discard(self, player_id: int, tile_id: int, tsumogiri: bool):
        """Processes a Discard event with enhanced logging."""
        reach_declared = self.player_reach_status[player_id] == 1
        if not (0 <= player_id < NUM_PLAYERS): print(f"[ERROR] Invalid player_id {player_id} in process_discard"); return
        if not (0 <= tile_id <= 135): print(f"[ERROR] Invalid tile_id {tile_id} in process_discard"); return

        hand_before_discard = list(self.player_hands[player_id]) # Copy hand before removal

        if tile_id in self.player_hands[player_id]:
            self.player_hands[player_id].remove(tile_id)
            self._sort_hand(player_id)
        else:
            # Log the hand state WHEN the error occurs
            print(f"[Warning] P{player_id} discarding {tile_id_to_string(tile_id)} ({tile_id}) not found.")
            print(f"  Hand Before Discard Attempt: {[tile_id_to_string(t) for t in hand_before_discard]}")

        self.player_discards[player_id].append((tile_id, tsumogiri))
        discard_data = {"tsumogiri": int(tsumogiri)}
        self._add_event("DISCARD", player=player_id, tile=tile_id, data=discard_data)
        self.last_discard_event_player = player_id
        self.last_discard_event_tile_id = tile_id
        self.last_discard_event_tsumogiri = tsumogiri
        self.can_ron = True

        if reach_declared:
            self.player_reach_status[player_id] = 2
            self.player_reach_discard_index[player_id] = len(self.player_discards[player_id]) - 1
            self.player_reach_junme[player_id] = self.junme
            if self.current_scores[player_id] >= 1000: self.kyotaku += 1; self.current_scores[player_id] -= 1000
            else: print(f"[Info] P{player_id} reached score {self.current_scores[player_id]} < 1000. Allowing."); self.kyotaku += 1; self.current_scores[player_id] -= 1000
            reach_data = {"step": 2, "junme": int(np.ceil(self.junme))}; self._add_event("REACH", player=player_id, data=reach_data)

    def process_naki(self, naki_player_id: int, meld_code: int):
        """Processes a Naki (call) event using naki_utils, with robust handling and logging."""
        if not (0 <= naki_player_id < NUM_PLAYERS): print(f"[ERROR] Invalid naki_player_id {naki_player_id}"); return

        naki_info = decode_naki(meld_code)
        naki_type = naki_info.get("type", "不明")
        decoded_tiles_structure = naki_info.get("tiles", []) # Use for hints
        from_who_relative = naki_info.get("from_who_relative", -1)

        if naki_type == "不明": print(f"[Warning] decode_naki failed for m={meld_code}. Skipping process_naki."); return

        from_who_player_abs = -1; called_tile_id = -1; tiles_to_remove = []; actual_meld_tiles = []

        # --- Handle calls based on type, trusting game state more ---
        if naki_type in ["チー", "ポン", "大明槓"]:
            called_tile_id = self.last_discard_event_tile_id; discarder_player_id = self.last_discard_event_player
            if discarder_player_id == -1 or called_tile_id == -1: print(f"[Warning] Naki {naki_type} P{naki_player_id} (m={meld_code}) without preceding discard. Skipping."); return
            if discarder_player_id == naki_player_id: print(f"[Warning] Naki {naki_type} P{naki_player_id} (m={meld_code}) from self discard. Skipping."); return
            from_who_player_abs = discarder_player_id
            called_tile_kind = tile_id_to_index(called_tile_id)
            if called_tile_kind == -1: print(f"[Error] Invalid called_tile_id {called_tile_id} for Naki {naki_type}. Aborting."); return

            if naki_type == "チー":
                is_from_kamicha = (discarder_player_id - naki_player_id + NUM_PLAYERS) % NUM_PLAYERS == 3
                if not is_from_kamicha: print(f"[Warning] Chi P{naki_player_id} from P{discarder_player_id} not from Kamicha. Skipping."); return
                possible_seqs = []; seq_base = called_tile_kind - (called_tile_kind % 9) # Ensure sequence is within suit
                if called_tile_kind % 9 <= 6: possible_seqs.append([called_tile_kind, called_tile_kind + 1, called_tile_kind + 2])
                if called_tile_kind % 9 >= 1 and called_tile_kind % 9 <= 7: possible_seqs.append([called_tile_kind - 1, called_tile_kind, called_tile_kind + 1])
                if called_tile_kind % 9 >= 2: possible_seqs.append([called_tile_kind - 2, called_tile_kind - 1, called_tile_kind])
                found_seq = False
                player_hand_kinds = defaultdict(list); [player_hand_kinds[tile_id_to_index(tid)].append(tid) for tid in self.player_hands[naki_player_id]]
                for seq_kinds in possible_seqs:
                    # Check if all kinds are within the same suit
                    if not all(k // 9 == called_tile_kind // 9 for k in seq_kinds): continue
                    needed_kinds = [k for k in seq_kinds if k != called_tile_kind]
                    if len(needed_kinds) == 2 and player_hand_kinds[needed_kinds[0]] and player_hand_kinds[needed_kinds[1]]:
                        # Prioritize using non-red fives if possible, unless red five is needed
                        tile1_id = player_hand_kinds[needed_kinds[0]].pop(0)
                        tile2_id = player_hand_kinds[needed_kinds[1]].pop(0)
                        tiles_to_remove = [tile1_id, tile2_id]; actual_meld_tiles = sorted([called_tile_id] + tiles_to_remove); found_seq = True; break
                if not found_seq: print(f"[Warning] Chi P{naki_player_id} (m={meld_code}): No valid sequence in hand for called {tile_id_to_string(called_tile_id)}. Skipping."); return

            elif naki_type in ["ポン", "大明槓"]:
                needed_count = 2 if naki_type == "ポン" else 3
                matching_tiles_in_hand = [tid for tid in self.player_hands[naki_player_id] if tile_id_to_index(tid) == called_tile_kind]
                if len(matching_tiles_in_hand) < needed_count: print(f"[Warning] {naki_type} P{naki_player_id} (m={meld_code}): Need {needed_count}, have {len(matching_tiles_in_hand)} of kind {called_tile_kind} for called {tile_id_to_string(called_tile_id)}. Skipping."); return
                # Prioritize non-red? Simple: take first N.
                tiles_to_remove = matching_tiles_in_hand[:needed_count]
                actual_meld_tiles = sorted([called_tile_id] + tiles_to_remove)
                if naki_type == "大明槓": self.is_rinshan = True

        elif naki_type == "加槓":
            t = meld_code >> 9; t //= 3; kakan_tile_kind = t if 0 <= t <= 33 else -1
            if kakan_tile_kind == -1: print(f"[Error] Kakan m={meld_code}: Invalid tile kind {t}. Aborting."); return
            found_added_tile_id = -1
            # Find the specific tile ID in hand (handle red fives potentially)
            possible_ids_in_hand = [tid for tid in self.player_hands[naki_player_id] if tile_id_to_index(tid) == kakan_tile_kind]
            if not possible_ids_in_hand: print(f"[Error] Kakan P{naki_player_id} (m={meld_code}): Cannot find tile of kind {kakan_tile_kind} in hand. Aborting."); return
            # Heuristic: use the first one found. A better way might be needed if multiple reds exist.
            found_added_tile_id = possible_ids_in_hand[0]
            tiles_to_remove = [found_added_tile_id]
            from_who_player_abs = naki_player_id; called_tile_id = -1; self.is_rinshan = True

        elif naki_type == "暗槓":
            tile_id_raw = meld_code >> 8; ankan_tile_kind = tile_id_raw // 4
            if not (0 <= ankan_tile_kind <= 33): print(f"[Error] Ankan m={meld_code}: Invalid tile kind {ankan_tile_kind}. Aborting."); return
            matching_tiles_in_hand = [tid for tid in self.player_hands[naki_player_id] if tile_id_to_index(tid) == ankan_tile_kind]
            if len(matching_tiles_in_hand) < 4: print(f"[Error] Ankan P{naki_player_id} (m={meld_code}): Need 4, have {len(matching_tiles_in_hand)} of kind {ankan_tile_kind}. Aborting."); return
            tiles_to_remove = matching_tiles_in_hand[:4]
            actual_meld_tiles = sorted(tiles_to_remove)
            from_who_player_abs = naki_player_id; called_tile_id = -1; self.is_rinshan = True

        # --- Perform State Update (Common Logic) ---
        removed_count = 0; temp_hand = list(self.player_hands[naki_player_id]); indices_to_pop = []

        # --- Enhanced Logging ---
        print(f"--- Naki Debug P{naki_player_id} ({naki_type}, m={meld_code}) ---")
        print(f"  Called Tile: {tile_id_to_string(called_tile_id)} ({called_tile_id}) from P{from_who_player_abs}")
        print(f"  Attempting to remove: {[tile_id_to_string(t) for t in tiles_to_remove]} from hand")
        print(f"  Hand Before Removal: {[tile_id_to_string(t) for t in self.player_hands[naki_player_id]]}")
        # --- End Enhanced Logging ---

        for tile_to_remove in tiles_to_remove:
            found = False
            for i in range(len(temp_hand)):
                if temp_hand[i] == tile_to_remove and i not in indices_to_pop: indices_to_pop.append(i); removed_count += 1; found = True; break
            if not found:
                 print(f"[Error] Naki P{naki_player_id} {naki_type}: Could not find tile {tile_id_to_string(tile_to_remove)} to remove.")
                 print(f"  Required removal list: {[tile_id_to_string(t) for t in tiles_to_remove]}")
                 print(f"  Hand state at failure: {[tile_id_to_string(tid) for tid in self.player_hands[naki_player_id]]}")
                 return # Abort if tile missing

        if removed_count == len(tiles_to_remove):
             for i in sorted(indices_to_pop, reverse=True): self.player_hands[naki_player_id].pop(i)
             self._sort_hand(naki_player_id)

             if naki_type == "加槓":
                 updated = False; pon_index = tile_id_to_index(tiles_to_remove[0])
                 for i, existing_meld in enumerate(self.player_melds[naki_player_id]):
                      if existing_meld['type'] == "ポン" and tile_id_to_index(existing_meld['tiles'][0]) == pon_index:
                          actual_meld_tiles = sorted(existing_meld['tiles'] + tiles_to_remove) # Use actual removed tile
                          self.player_melds[naki_player_id][i]['type'] = "加槓"; self.player_melds[naki_player_id][i]['tiles'] = actual_meld_tiles; self.player_melds[naki_player_id][i]['jun'] = self.junme; updated = True; break
                 if not updated: print(f"[Error] Kakan P{naki_player_id}: Corresponding Pon not found to update.")
             else:
                 new_meld = {'type': naki_type, 'tiles': actual_meld_tiles, 'from_who': from_who_player_abs, 'called_tile': called_tile_id, 'm': meld_code, 'jun': self.junme}
                 self.player_melds[naki_player_id].append(new_meld)

             self.current_player = naki_player_id; self.naki_occurred_in_turn = True; self.can_ron = False
             self.last_discard_event_player = -1; self.last_discard_event_tile_id = -1; self.last_discard_event_tsumogiri = False

             naki_event_data = {"naki_type": NAKI_TYPES.get(naki_type, -1), "from_who": from_who_player_abs}
             event_tile = called_tile_id if called_tile_id != -1 else actual_meld_tiles[0]
             self._add_event("N", player=naki_player_id, tile=event_tile, data=naki_event_data)
        else: print(f"[Error] Naki P{naki_player_id} {naki_type}: Mismatch in removable tiles count. Aborting naki process.")

    def process_reach(self, player_id: int, step: int):
        if not (0 <= player_id < NUM_PLAYERS): print(f"[ERROR] Invalid player_id {player_id} in process_reach"); return
        if step == 1:
            if self.player_reach_status[player_id] != 0 or self.current_scores[player_id] < 1000: return
            self.player_reach_status[player_id] = 1
            reach_data = {"step": 1}; self._add_event("REACH", player=player_id, data=reach_data)

    def process_dora(self, tile_id: int):
        if not (0 <= tile_id <= 135): print(f"[ERROR] Invalid tile_id {tile_id} in process_dora"); return
        self.dora_indicators.append(tile_id); self._add_event("DORA", player=-1, tile=tile_id)

    def process_agari(self, attrib: dict):
        try:
            winner = int(attrib.get("who", -1)); sc_str = attrib.get("sc"); ba_str = attrib.get("ba")
            if not (0 <= winner < NUM_PLAYERS): raise ValueError("Invalid winner")
            if sc_str: sc_values = [int(float(s)) for s in sc_str.split(",")]; self.current_scores = [v * 100 for v in sc_values[0::2]] if len(sc_values) == 8 else self.current_scores
            if ba_str: ba_values = [int(float(s)) for s in ba_str.split(",")]; self.honba, self.kyotaku = (ba_values[0] + 1 if winner == self.dealer else 0, 0) if len(ba_values) == 2 else (self.honba, self.kyotaku)
            else: self.honba = self.honba + 1 if winner == self.dealer else 0; self.kyotaku = 0
            agari_data = {k: v for k, v in attrib.items()}; machi_tile = int(attrib.get('machi', -1))
            self._add_event("AGARI", player=winner, tile=machi_tile, data=agari_data)
        except (ValueError, KeyError, TypeError, IndexError) as e: print(f"[Warning] Failed AGARI processing: {e}")

    def process_ryuukyoku(self, attrib: dict):
        try:
            sc_str = attrib.get("sc", attrib.get("owari")); ba_str = attrib.get("ba")
            if sc_str: sc_values = [int(float(s)) for s in sc_str.split(",")]; self.current_scores = [v * 100 for v in sc_values[0::2]] if len(sc_values) == 8 else self.current_scores
            if ba_str: ba_values = [int(float(s)) for s in ba_str.split(",")]; self.honba, self.kyotaku = (ba_values[0] + 1, ba_values[1]) if len(ba_values) == 2 else (self.honba, self.kyotaku)
            else: self.honba += 1
            ry_data = {k: v for k, v in attrib.items()}
            self._add_event("RYUUKYOKU", player=-1, data=ry_data)
        except (ValueError, KeyError, TypeError, IndexError) as e: print(f"[Warning] Failed RYUUKYOKU processing: {e}")

    # --- Feature Extraction Methods ---
    def get_current_dora_indices(self) -> list[int]:
        dora_indices = [];
        for ind_id in self.dora_indicators:
            ind_idx = tile_id_to_index(ind_id); dr_idx = -1
            if ind_idx == -1: continue
            if 0 <= ind_idx <= 26: dr_idx = (ind_idx // 9)*9 + (ind_idx % 9 + 1) % 9
            elif 27 <= ind_idx <= 30: dr_idx = 27 + (ind_idx - 27 + 1) % 4
            elif 31 <= ind_idx <= 33: dr_idx = 31 + (ind_idx - 31 + 1) % 3
            if dr_idx != -1: dora_indices.append(dr_idx)
        return dora_indices

    def get_hand_indices(self, player_id: int) -> list[int]:
        if 0 <= player_id < NUM_PLAYERS: return [tile_id_to_index(t) for t in self.player_hands[player_id] if tile_id_to_index(t) != -1]
        return []

    def get_event_sequence_features(self) -> np.ndarray:
        """Converts event history to a padded sequence of numerical vectors."""
        sequence = []
        # Calculate expected dimension based on _add_event logic
        event_dims = {"DISCARD": 1, "N": 2, "REACH": 2} # Max specific data dims
        event_specific_dim = max(event_dims.values()) if event_dims else 0
        event_base_dim = 4
        event_total_dim = event_base_dim + event_specific_dim

        for event in self.event_history:
            event_vec = np.zeros(event_total_dim, dtype=np.float32)
            event_vec[0]=float(event["type"]); event_vec[1]=float(event["player"]+1)
            event_vec[2]=float(event["tile_index"]+1); event_vec[3]=float(event["junme"])
            data = event.get("data", {}); event_type_code = event["type"]
            current_specific_dim = event.get("_event_specific_dim", 0) # Get expected dim for this event

            if event_type_code == EVENT_TYPES["DISCARD"]:
                 if current_specific_dim >= 1: event_vec[event_base_dim+0]=float(data.get("tsumogiri", 0))
            elif event_type_code == EVENT_TYPES["N"]:
                 if current_specific_dim >= 1: event_vec[event_base_dim+0]=float(data.get("naki_type", -1)+1)
                 if current_specific_dim >= 2: event_vec[event_base_dim+1]=float(data.get("from_who", -1)+1)
            elif event_type_code == EVENT_TYPES["REACH"]:
                 if current_specific_dim >= 1: event_vec[event_base_dim+0]=float(data.get("step", 0))
                 if current_specific_dim >= 2: event_vec[event_base_dim+1]=float(data.get("junme", 0))
            sequence.append(event_vec)

        padding_length = MAX_EVENT_HISTORY - len(sequence)
        padding_vec = np.zeros(event_total_dim, dtype=np.float32); padding_vec[0]=float(EVENT_TYPES["PADDING"])
        # Use list() on deque before slicing/padding
        padded_sequence_list = list(sequence)[-MAX_EVENT_HISTORY:] + [padding_vec] * max(0, padding_length)

        try:
             final_array = np.array(padded_sequence_list, dtype=np.float32)
             # Final shape check
             if final_array.shape != (MAX_EVENT_HISTORY, event_total_dim):
                  print(f"[ERROR] Final sequence shape mismatch: {final_array.shape}, expected {(MAX_EVENT_HISTORY, event_total_dim)}")
                  return np.zeros((MAX_EVENT_HISTORY, event_total_dim), dtype=np.float32) # Fallback
             return final_array
        except ValueError as e:
             print(f"[ERROR] Event sequence conversion failed. Vector lengths might be inconsistent.")
             # Print lengths for debugging
             # for i, vec in enumerate(padded_sequence_list): print(f" Vec {i}, Len {len(vec) if isinstance(vec, np.ndarray) else 'N/A'}")
             raise e


    def get_static_features(self, player_id: int) -> np.ndarray:
        """Generates a static feature vector (dim=157) for the given player's perspective."""
        if not (0 <= player_id < NUM_PLAYERS): raise ValueError("Invalid player_id")
        features = np.zeros(STATIC_FEATURE_DIM, dtype=np.float32); idx = 0
        DIM = {"CONTEXT": 8, "PLAYER": 5, "HAND": 34, "DORA_IND": 34, "DISCARDS": 34, "VISIBLE": 34, "POS_REACH": 8}
        try:
            f=features[idx:idx+DIM["CONTEXT"]]; idx+=DIM["CONTEXT"]; f[:]=[self.round_num_wind, self.honba, self.kyotaku, self.dealer, self.wall_tile_count, float(player_id == self.dealer), self.junme, len(self.dora_indicators)]
            f=features[idx:idx+DIM["PLAYER"]]; idx+=DIM["PLAYER"]; f[:]=[self.player_reach_status[player_id], self.player_reach_junme[player_id], len(self.player_discards[player_id]), len(self.player_melds[player_id]), len(self.player_hands[player_id])]
            start_idx=idx; idx+=DIM["HAND"]; hand_counts=defaultdict(int); [hand_counts.__setitem__(i, hand_counts[i]+1) for i in self.get_hand_indices(player_id)]; [features.__setitem__(start_idx + k, v) for k,v in hand_counts.items() if 0<=k<NUM_TILE_TYPES]
            start_idx=idx; idx+=DIM["DORA_IND"]; [features.__setitem__(start_idx + tile_id_to_index(ind_id), features[start_idx + tile_id_to_index(ind_id)]+1.0) for ind_id in self.dora_indicators if 0 <= tile_id_to_index(ind_id) < NUM_TILE_TYPES]
            start_idx=idx; idx+=DIM["DISCARDS"]; [features.__setitem__(start_idx + tile_id_to_index(t_id), features[start_idx + tile_id_to_index(t_id)]+1.0) for t_id, _ in self.player_discards[player_id] if 0 <= tile_id_to_index(t_id) < NUM_TILE_TYPES]
            start_idx=idx; idx+=DIM["VISIBLE"]
            for p in range(NUM_PLAYERS):
                [features.__setitem__(start_idx + tile_id_to_index(t_id), features[start_idx + tile_id_to_index(t_id)]+1.0) for t_id, _ in self.player_discards[p] if 0 <= tile_id_to_index(t_id) < NUM_TILE_TYPES]
                for meld in self.player_melds[p]: [features.__setitem__(start_idx + tile_id_to_index(t_id), features[start_idx + tile_id_to_index(t_id)]+1.0) for t_id in meld.get("tiles", []) if 0 <= tile_id_to_index(t_id) < NUM_TILE_TYPES]
            start_idx=idx; idx+=DIM["POS_REACH"]
            for p_offset in range(NUM_PLAYERS): p_abs = (player_id + p_offset) % NUM_PLAYERS; features[start_idx + p_offset*2 + 0] = float(p_abs == player_id); features[start_idx + p_offset*2 + 1] = float(self.player_reach_status[p_abs] == 2)
        except Exception as e: print(f"[ERROR] Static feature generation P{player_id}: {e}"); return np.zeros(STATIC_FEATURE_DIM, dtype=np.float32)
        if idx != STATIC_FEATURE_DIM: print(f"[CRITICAL ERROR] Final static idx {idx} != {STATIC_FEATURE_DIM}!"); return np.zeros(STATIC_FEATURE_DIM, dtype=np.float32)
        if np.isnan(features).any() or np.isinf(features).any(): print(f"[Error] NaN/Inf in static P{player_id}! Replacing."); features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    def get_valid_discard_options(self, player_id: int) -> list[int]:
        if not (0 <= player_id < NUM_PLAYERS): return []
        hand=self.player_hands[player_id]; is_reach=self.player_reach_status[player_id]==2; has_drawn=len(hand)%3==2
        if is_reach:
            if has_drawn and hand: idx=tile_id_to_index(hand[-1]); return [idx] if idx!=-1 else []
            else: return []
        else:
            if not has_drawn: return []
            options = sorted(list(set(tile_id_to_index(t) for t in hand if tile_id_to_index(t) != -1)))
            return options