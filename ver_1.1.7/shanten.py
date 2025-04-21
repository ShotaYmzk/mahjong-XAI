# game_state.py (Integrated version based on user provided snippets)
import numpy as np
from collections import defaultdict, deque
import sys # For error exit if imports fail

# --- Dependency Imports ---
try:
    from tile_utils import tile_id_to_index, tile_id_to_string
except ImportError as e:
    print(f"[FATAL ERROR in game_state.py] Cannot import from tile_utils.py: {e}")
    print("Ensure tile_utils.py is in the same directory or Python path.")
    sys.exit(1)
try:
    # Use the latest naki_utils provided by the user
    from naki_utils import decode_naki
except ImportError as e:
    print(f"[FATAL ERROR in game_state.py] Cannot import from naki_utils.py: {e}")
    print("Ensure naki_utils.py is in the same directory or Python path.")
    sys.exit(1)
# --- End Dependency Imports ---

# --- Shanten Calculation Placeholder ---
def calculate_shanten(hand_indices, melds=[]):
    """Placeholder for shanten calculation."""
    # TODO: Implement or import a real shanten calculator
    # print("[Warning] Using dummy shanten calculation.")
    num_tiles = len(hand_indices) + sum(len(m.get("tiles", [])) for m in melds) # Adjust based on melds structure
    shanten = 8 # Default to a high value
    if num_tiles == 14: shanten = 0 # Crude approximation: 14 tiles might be tenpai/agari
    elif num_tiles == 13: shanten = 0 # Crude approximation
    ukeire = [] # Cannot determine ukeire with dummy logic
    return shanten, ukeire
# --- End Shanten Calculation Placeholder ---

# --- Constants ---
NUM_PLAYERS = 4
NUM_TILE_TYPES = 34
MAX_EVENT_HISTORY = 60 # Sequence length for Transformer
# --- End Constants ---

class GameState:
    """Manages the state of a Mahjong game round."""
    TSUMO_TAGS = {"T": 0, "U": 1, "V": 2, "W": 3}
    DISCARD_TAGS = {"D": 0, "E": 1, "F": 2, "G": 3}

    # Event types for event history (numerical mapping)
    EVENT_TYPES = {
        "INIT": 0, "TSUMO": 1, "DISCARD": 2, "N": 3, "REACH": 4,
        "DORA": 5, "AGARI": 6, "RYUUKYOKU": 7, "PADDING": 8
    }
    # Naki types for event history data (numerical mapping)
    NAKI_TYPES = {"チー": 0, "ポン": 1, "大明槓": 2, "加槓": 3, "暗槓": 4, "不明": -1}

    def __init__(self):
        """Initializes the GameState."""
        self.reset_state()

    def reset_state(self):
        """Resets all internal state variables to default values."""
        self.round_index: int = 0          # Internal round index from parser
        self.round_num_wind: int = 0       # Round wind (0:E1, 1:E2... 4:S1...)
        self.honba: int = 0
        self.kyotaku: int = 0              # Number of riichi sticks on table
        self.dealer: int = -1              # Dealer player index (0-3)
        self.initial_scores: list[int] = [25000] * NUM_PLAYERS
        self.dora_indicators: list[int] = [] # List of Dora indicator tile IDs
        self.current_scores: list[int] = [25000] * NUM_PLAYERS
        # Hands store tile IDs (0-135)
        self.player_hands: list[list[int]] = [[] for _ in range(NUM_PLAYERS)]
        # Discards store tuples: (tile_id, tsumogiri_flag)
        self.player_discards: list[list[tuple[int, bool]]] = [[] for _ in range(NUM_PLAYERS)]
        # Melds store dicts (more structured than previous tuple):
        # {'type': str, 'tiles': list[int], 'trigger': int, 'from_who': int}
        self.player_melds: list[list[dict]] = [[] for _ in range(NUM_PLAYERS)]
        self.player_reach_status: list[int] = [0] * NUM_PLAYERS # 0: No, 1: Declared, 2: Accepted
        self.player_reach_junme: list[float] = [-1.0] * NUM_PLAYERS # Junme when reach accepted
        self.player_reach_discard_index: list[int] = [-1] * NUM_PLAYERS # Index in discard list when reach accepted
        self.current_player: int = -1      # Index of the player whose turn it is
        self.junme: float = 0.0            # Current turn number (increments by 0.25 per tsumo)
        self.last_discard_event_player: int = -1 # Player who made the last discard
        self.last_discard_event_tile_id: int = -1 # Tile ID of the last discard
        self.last_discard_event_tsumogiri: bool = False
        self.can_ron: bool = False         # Flag indicating if Ron is possible on the last discard
        self.naki_occurred_in_turn: bool = False # Flag if naki happened after last tsumo (affects junme)
        self.is_rinshan: bool = False      # Flag if the next tsumo is from rinshanpai
        # --- Transformer specific ---
        self.event_history: deque = deque(maxlen=MAX_EVENT_HISTORY) # Fixed-size event history
        self.wall_tile_count: int = 70     # Estimated remaining tiles in wall (excluding dead wall)
        # TODO: Implement remaining tile counter per type
        # self.remaining_tiles = defaultdict(lambda: 4)

    def _add_event(self, event_type: str, player: int, tile: int = -1, data: dict = None):
        """Adds a structured event to the event history deque."""
        if data is None: data = {}
        event_code = self.EVENT_TYPES.get(event_type, -1)
        if event_code == -1:
            # print(f"[WARN in GameState._add_event] Unknown event type: {event_type}") # Optional warning
            return # Don't add unknown events

        # Basic event info structure
        event_info = {
            "type": event_code,
            "player": player, # Player index associated with event (-1 for global)
            "tile_index": tile_id_to_index(tile), # Tile type index (0-33, -1 if no tile)
            "junme": int(np.ceil(self.junme)), # Current junme (integer part)
            "data": data # Additional data (e.g., naki type, reach step, tsumogiri flag)
        }
        self.event_history.append(event_info)

    def init_round(self, round_data: dict):
        """Initializes the game state for a new round based on parsed data."""
        self.reset_state() # Clear previous round state
        init_info = round_data.get("init", {})
        if not init_info: print("[Warning] No init info found in round_data."); return

        self.round_index = round_data.get("round_index", 0)
        seed_parts = init_info.get("seed", "0,0,0,0,0,0").split(",")
        try:
            self.round_num_wind = int(seed_parts[0])
            self.honba = int(seed_parts[1])
            self.kyotaku = int(seed_parts[2])
            dora_indicator_id = int(seed_parts[5])
            self.dora_indicators = [dora_indicator_id]
            # Add initial Dora reveal to event history
            self._add_event("DORA", player=-1, tile=dora_indicator_id)
        except (IndexError, ValueError) as e:
            print(f"[Warning] Failed to parse seed string '{init_info.get('seed')}': {e}")
            self.round_num_wind=0; self.honba=0; self.kyotaku=0; self.dora_indicators=[]

        self.dealer = int(init_info.get("oya", -1))
        self.current_player = self.dealer # Starts with dealer

        try:
            self.initial_scores = list(map(int, init_info.get("ten", "25000,25000,25000,25000").split(",")))
            # Ensure scores are integers
            self.initial_scores = [int(s) for s in self.initial_scores]
            self.current_scores = list(self.initial_scores)
        except (ValueError, TypeError) as e:
            print(f"[Warning] Failed to parse 'ten' attribute '{init_info.get('ten')}': {e}")
            self.initial_scores=[25000]*NUM_PLAYERS; self.current_scores=[25000]*NUM_PLAYERS

        for p in range(NUM_PLAYERS):
            hand_str = init_info.get(f"hai{p}", "")
            self.player_hands[p] = [] # Initialize empty first
            try:
                if hand_str: # Only parse if not empty
                    hand_ids = list(map(int, hand_str.split(",")))
                    # Validate tile IDs
                    valid_hand_ids = [tid for tid in hand_ids if 0 <= tid <= 135]
                    if len(valid_hand_ids) != len(hand_ids):
                        print(f"[Warning] Invalid tile IDs found in initial hand for P{p}: {hand_str}")
                    self.player_hands[p] = sorted(valid_hand_ids)
            except ValueError as e:
                print(f"[Warning] Failed to parse initial hand 'hai{p}' for P{p}: {e}")
            # Other lists (discards, melds) are already reset in reset_state

        # Initial wall count calculation
        self.wall_tile_count = 136 - 14 - (13 * NUM_PLAYERS) # 136 total - 14 dead wall - 52 initial hands = 70

        # Add INIT event to history
        init_data = {"round": self.round_num_wind, "honba": self.honba, "kyotaku": self.kyotaku}
        self._add_event("INIT", player=self.dealer, data=init_data)

    def _sort_hand(self, player_id):
        """Sorts the hand of the specified player."""
        if 0 <= player_id < NUM_PLAYERS:
            self.player_hands[player_id].sort(key=lambda t: (tile_id_to_index(t), t))

    def process_tsumo(self, player_id: int, tile_id: int):
        """Processes a Tsumo (draw) event."""
        if not (0 <= player_id < NUM_PLAYERS): print(f"[ERROR] Invalid player_id {player_id} in process_tsumo"); return
        if not (0 <= tile_id <= 135): print(f"[ERROR] Invalid tile_id {tile_id} in process_tsumo"); return

        self.current_player = player_id
        self.wall_tile_count -= 1 # Consume one tile from wall

        rinshan_draw = self.is_rinshan # Store rinshan state before modifying hand
        if rinshan_draw:
            self.is_rinshan = False # Reset rinshan flag after the draw
        else:
            # Increment junme only for non-rinshan draws if no naki occurred since last player's normal draw
             if not self.naki_occurred_in_turn:
                  self.junme += 0.25

        self.naki_occurred_in_turn = False # Reset naki flag

        # Add tile to hand
        self.player_hands[player_id].append(tile_id)
        self._sort_hand(player_id)

        # Add event to history
        tsumo_data = {"rinshan": rinshan_draw}
        self._add_event("TSUMO", player=player_id, tile=tile_id, data=tsumo_data)

        # Reset flags related to previous discard
        self.can_ron = False
        self.last_discard_event_player = -1
        self.last_discard_event_tile_id = -1
        self.last_discard_event_tsumogiri = False

    def process_discard(self, player_id: int, tile_id: int, tsumogiri: bool):
        """Processes a Discard event."""
        if not (0 <= player_id < NUM_PLAYERS): print(f"[ERROR] Invalid player_id {player_id} in process_discard"); return
        if not (0 <= tile_id <= 135): print(f"[ERROR] Invalid tile_id {tile_id} in process_discard"); return

        # Check if tile exists in hand before removing
        if tile_id in self.player_hands[player_id]:
            self.player_hands[player_id].remove(tile_id)
            self._sort_hand(player_id)
        else:
            print(f"[ERROR] P{player_id} discard {tile_id_to_string(tile_id)} not found in hand: {[tile_id_to_string(t) for t in self.player_hands[player_id]]}")
            # Attempt to proceed, but state might be inconsistent

        # Add to discard list
        self.player_discards[player_id].append((tile_id, tsumogiri))

        # Add event to history
        discard_data = {"tsumogiri": tsumogiri}
        self._add_event("DISCARD", player=player_id, tile=tile_id, data=discard_data)

        # Update last discard info
        self.last_discard_event_player = player_id
        self.last_discard_event_tile_id = tile_id
        self.last_discard_event_tsumogiri = tsumogiri
        self.can_ron = True # Ron is possible after a discard

        # Process reach step 2 (acceptance)
        if self.player_reach_status[player_id] == 1: # If player declared reach
            self.player_reach_status[player_id] = 2 # Set to accepted
            self.player_reach_discard_index[player_id] = len(self.player_discards[player_id]) - 1
            self.player_reach_junme[player_id] = self.junme # Record junme of reach acceptance
            self.kyotaku += 1
            self.current_scores[player_id] -= 1000 # Deduct reach stick cost
            # Add reach accepted event to history
            self._add_event("REACH", player=player_id, data={"step": 2})

        # Advance turn to next player
        self.current_player = (player_id + 1) % NUM_PLAYERS

    def process_naki(self, naki_player_id: int, meld_code: int):
        """Processes a Naki (call) event."""
        if not (0 <= naki_player_id < NUM_PLAYERS): print(f"[ERROR] Invalid naki_player_id {naki_player_id}"); return

        # Use decode_naki to get basic info (type, potential tiles)
        # We rely on internal state (last discard) for trigger/source confirmation
        naki_type_str, decoded_tiles, _, _ = decode_naki(meld_code)

        if naki_type_str == "不明":
            print(f"[Warning] decode_naki failed for m={meld_code}. Skipping process_naki.")
            return

        # --- Handle Self-Calls (Kans) First ---
        if naki_type_str == "加槓":
            # Kakan logic needs to find the Pon and the tile from hand
            added_offset = (meld_code & 0x0060) >> 5
            t = (meld_code & 0xFE00) >> 9
            t //= 3
            if not (0 <= t <= 33): print(f"[ERROR] Invalid tile index {t} in Kakan m={meld_code}"); return
            base_id = t * 4
            kakan_pai_id_from_offset = base_id + added_offset # Tile added from hand

            target_meld_index = -1
            original_pon_tiles = []
            original_from_who = -1
            for i, meld_data in enumerate(self.player_melds[naki_player_id]):
                if meld_data['type'] == "ポン" and tile_id_to_index(meld_data['tiles'][0]) == t:
                    target_meld_index = i
                    original_pon_tiles = meld_data['tiles']
                    original_from_who = meld_data['from_who']
                    break

            if target_meld_index == -1:
                print(f"[ERROR] Kakan P{naki_player_id}: Corresponding Pon for index {t} (m={meld_code}) not found.")
                return

            # Find the exact tile ID in hand (consider red fives)
            kakan_pai_id_in_hand = -1
            kakan_pai_hand_idx = -1
            for idx, hid in enumerate(self.player_hands[naki_player_id]):
                 # Check if tile index matches and ID matches the one derived from offset
                 # Or just check index if offset logic is unreliable for red fives
                 if tile_id_to_index(hid) == t:
                      # Heuristic: Assume the first matching tile index is the one used for kakan.
                      # A more robust way might be needed if player has multiple identical tiles.
                      kakan_pai_id_in_hand = hid
                      kakan_pai_hand_idx = idx
                      break # Found potential tile

            if kakan_pai_id_in_hand == -1:
                 print(f"[ERROR] Kakan P{naki_player_id}: Tile with index {t} not found in hand for m={meld_code}. Hand: {[tile_id_to_string(tid) for tid in self.player_hands[naki_player_id]]}")
                 return

            # Perform state update
            self.player_hands[naki_player_id].pop(kakan_pai_hand_idx)
            new_meld_tiles = sorted(original_pon_tiles + [kakan_pai_id_in_hand])
            # Update the meld entry
            self.player_melds[naki_player_id][target_meld_index] = {
                'type': "加槓", 'tiles': new_meld_tiles, 'trigger': -1, 'from_who': original_from_who
            }
            self._sort_hand(naki_player_id)
            self.current_player = naki_player_id
            self.naki_occurred_in_turn = True
            self.can_ron = False
            self.is_rinshan = True # Set rinshan flag for next draw
            # Add event history
            self._add_event("N", player=naki_player_id, tile=kakan_pai_id_in_hand,
                            data={"naki_type": self.NAKI_TYPES["加槓"], "from_who": naki_player_id})
            return # Kakan processed

        if naki_type_str == "暗槓":
            # Ankan logic needs to find 4 tiles in hand
            t = (meld_code & 0xFF00) >> 8
            ankan_tile_index = t // 4
            if not (0 <= ankan_tile_index <= 33): print(f"[ERROR] Invalid tile index {ankan_tile_index} in Ankan m={meld_code}"); return

            # Find 4 tiles of the same index in hand
            matching_indices_in_hand = [i for i, tid in enumerate(self.player_hands[naki_player_id]) if tile_id_to_index(tid) == ankan_tile_index]

            if len(matching_indices_in_hand) < 4:
                print(f"[ERROR] Ankan P{naki_player_id}: Found only {len(matching_indices_in_hand)} tiles with index {ankan_tile_index} in hand for m={meld_code}. Hand: {[tile_id_to_string(tid) for tid in self.player_hands[naki_player_id]]}")
                return

            # Remove 4 tiles and record their exact IDs
            consumed_tile_ids = []
            indices_to_remove = matching_indices_in_hand[:4] # Take the first 4 found
            for i in sorted(indices_to_remove, reverse=True):
                consumed_tile_ids.append(self.player_hands[naki_player_id].pop(i))

            # Add meld
            new_meld = {'type': "暗槓", 'tiles': sorted(consumed_tile_ids), 'trigger': -1, 'from_who': naki_player_id}
            self.player_melds[naki_player_id].append(new_meld)
            self._sort_hand(naki_player_id)
            self.current_player = naki_player_id
            self.naki_occurred_in_turn = True
            self.can_ron = False
            self.is_rinshan = True # Set rinshan flag
             # Add event history
            self._add_event("N", player=naki_player_id, tile=consumed_tile_ids[0], # Use one tile as representative
                            data={"naki_type": self.NAKI_TYPES["暗槓"], "from_who": naki_player_id})
            return # Ankan processed

        # --- Handle Calls from Others (Chi, Pon, Daiminkan) ---
        trigger_player_abs = self.last_discard_event_player
        trigger_tile_id = self.last_discard_event_tile_id

        if trigger_player_abs == -1 or trigger_tile_id == -1:
            print(f"[Warning] Naki {naki_type_str} P{naki_player_id} (m={meld_code}) occurred without preceding discard info. Skipping.")
            return
        if trigger_player_abs == naki_player_id:
            print(f"[Warning] Naki {naki_type_str} P{naki_player_id} (m={meld_code}) called from self (invalid). Skipping.")
            return

        # Validate call is possible based on type and relative position (basic check)
        # More complex validation (e.g., actual tiles in hand) is needed for full accuracy
        possible = False
        consumed_hand_ids = []
        final_meld_tiles = []

        if naki_type_str == "チー":
            # Basic check: Chi must be from Kamicha (player to the left)
            if (trigger_player_abs - naki_player_id + NUM_PLAYERS) % NUM_PLAYERS == 3: # Check if discarder is Kamicha
                 # Need to check if player actually has the other 2 tiles for the sequence
                 # Use decoded_tiles from decode_naki as the target sequence
                 hand_indices = defaultdict(int)
                 for tid in self.player_hands[naki_player_id]: hand_indices[tile_id_to_index(tid)] += 1
                 trigger_idx = tile_id_to_index(trigger_tile_id)
                 needed = []
                 seq_tiles = sorted([tile_id_to_index(t) for t in decoded_tiles]) # Get the indices [a, a+1, a+2]
                 if len(seq_tiles) == 3:
                     for idx in seq_tiles:
                         if idx != trigger_idx: needed.append(idx)
                     if len(needed) == 2 and hand_indices[needed[0]] > 0 and hand_indices[needed[1]] > 0:
                         # Player has the tiles, now find the specific IDs to remove
                         consumed_hand_ids_temp = []
                         found1, found2 = False, False
                         indices_to_remove = []
                         for i, tid in enumerate(self.player_hands[naki_player_id]):
                             t_idx = tile_id_to_index(tid)
                             if not found1 and t_idx == needed[0]:
                                 consumed_hand_ids_temp.append(tid); indices_to_remove.append(i); found1 = True
                             elif not found2 and t_idx == needed[1]:
                                 consumed_hand_ids_temp.append(tid); indices_to_remove.append(i); found2 = True
                             if found1 and found2: break
                         if found1 and found2:
                             possible = True
                             consumed_hand_ids = consumed_hand_ids_temp
                             final_meld_tiles = sorted([trigger_tile_id] + consumed_hand_ids)
                             # Use precise tiles from decode_naki if validation passes, otherwise use calculated
                             if all(t in final_meld_tiles for t in decoded_tiles) and all(t in decoded_tiles for t in final_meld_tiles):
                                 final_meld_tiles = sorted(decoded_tiles) # Trust decode_naki if consistent


            else: print(f"[Warning] Chi P{naki_player_id} from P{trigger_player_abs} is not from Kamicha.")

        elif naki_type_str in ["ポン", "大明槓"]:
            needed_count = 2 if naki_type_str == "ポン" else 3
            trigger_idx = tile_id_to_index(trigger_tile_id)
            # Find exact tiles in hand
            matching_tiles_in_hand = [tid for tid in self.player_hands[naki_player_id] if tile_id_to_index(tid) == trigger_idx]
            if len(matching_tiles_in_hand) >= needed_count:
                possible = True
                consumed_hand_ids = matching_tiles_in_hand[:needed_count] # Take the first N found
                final_meld_tiles = sorted([trigger_tile_id] + consumed_hand_ids)
                # Try to use decode_naki's tiles if they match for Pon (more accurate for red fives)
                if naki_type_str == "ポン" and len(decoded_tiles) == 3 and set(tile_id_to_index(t) for t in decoded_tiles) == {trigger_idx}:
                    final_meld_tiles = sorted(decoded_tiles) # Use decode_naki result

        # If call is determined possible, update state
        if possible:
            # Remove consumed tiles from hand
            temp_hand = list(self.player_hands[naki_player_id])
            indices_removed = []
            for tile_to_remove in consumed_hand_ids:
                try:
                    idx_to_remove = -1
                    # Find index that hasn't been marked for removal yet
                    for current_idx, hand_tile in enumerate(temp_hand):
                         if hand_tile == tile_to_remove and current_idx not in indices_removed:
                              idx_to_remove = current_idx
                              break
                    if idx_to_remove != -1:
                         temp_hand.pop(idx_to_remove) # Remove from temp list first
                         indices_removed.append(idx_to_remove)
                    else: raise ValueError # Tile not found (should not happen if possible=True)
                except ValueError:
                    print(f"[ERROR] Naki P{naki_player_id}: Failed to remove tile {tile_id_to_string(tile_to_remove)} from hand during {naki_type_str}.")
                    return # Abort state update

            # Update actual hand if all removals were successful
            self.player_hands[naki_player_id] = temp_hand

            # Add the meld
            new_meld = {'type': naki_type_str, 'tiles': final_meld_tiles, 'trigger': trigger_tile_id, 'from_who': trigger_player_abs}
            self.player_melds[naki_player_id].append(new_meld)

            self._sort_hand(naki_player_id)
            self.current_player = naki_player_id
            self.naki_occurred_in_turn = True
            self.can_ron = False
            self.last_discard_event_player = -1 # Reset last discard
            self.last_discard_event_tile_id = -1
            self.last_discard_event_tsumogiri = False
            if naki_type_str == "大明槓":
                self.is_rinshan = True # Set rinshan flag

            # Add event to history
            self._add_event("N", player=naki_player_id, tile=trigger_tile_id,
                             data={"naki_type": self.NAKI_TYPES.get(naki_type_str, -1), "from_who": trigger_player_abs})
        else:
             print(f"[Warning] Failed {naki_type_str} P{naki_player_id} (m={meld_code}). Hand: {[tile_id_to_string(tid) for tid in self.player_hands[naki_player_id]]}. Trigger: {tile_id_to_string(trigger_tile_id)} from P{trigger_player_abs}. Skipping.")


    def process_reach(self, player_id: int, step: int):
        """Processes a Reach declaration (step 1)."""
        if not (0 <= player_id < NUM_PLAYERS): print(f"[ERROR] Invalid player_id {player_id} in process_reach"); return

        if step == 1:
            # Check if already reached or no points
            if self.player_reach_status[player_id] != 0:
                print(f"[Warning] P{player_id} Reach step 1 called but already reach status {self.player_reach_status[player_id]}.")
                return
            if self.current_scores[player_id] < 1000:
                print(f"[Warning] P{player_id} Reach step 1 called but score {self.current_scores[player_id]} < 1000.")
                return

            # Set state to declared
            self.player_reach_status[player_id] = 1
            # Junme will be recorded upon successful discard (step 2)
            # Add reach declared event to history
            self._add_event("REACH", player=player_id, data={"step": 1})
        elif step == 2:
             # Step 2 is handled within process_discard, do nothing here
             pass

    def process_dora(self, tile_id: int):
        """Processes a Dora indicator reveal event."""
        if not (0 <= tile_id <= 135): print(f"[ERROR] Invalid tile_id {tile_id} in process_dora"); return

        self.dora_indicators.append(tile_id)
        # Add Dora event to history
        self._add_event("DORA", player=-1, tile=tile_id)
        # Decrement wall count if it's a Kan Dora (revealed after Kan)
        # Need a reliable way to know if this DORA tag corresponds to a Kan.
        # For simplicity, we might ignore wall count adjustment here or
        # assume all DORA tags after turn 1 potentially reduce wall count.
        # Let's assume wall count is only decremented by Tsumo for now.

    def process_agari(self, attrib: dict):
        """Processes an Agari (win) event, updating scores etc."""
        try:
            winner = int(attrib.get("who", -1))
            # Update scores based on 'sc' attribute
            sc_str = attrib.get("sc")
            if sc_str:
                sc_values = list(map(int, sc_str.split(",")))
                if len(sc_values) == 8:
                    # Scores in 'sc' are final scores *after* payment
                    self.current_scores = [sc_values[i*2] for i in range(NUM_PLAYERS)]
                else: print(f"[Warning] Invalid 'sc' length in AGARI: {sc_str}")
            # Update honba/kyotaku based on 'ba' attribute
            ba_str = attrib.get("ba")
            if ba_str:
                ba_values = list(map(int, ba_str.split(",")))
                if len(ba_values) == 2:
                    # After agari, kyotaku becomes 0
                    self.kyotaku = 0
                    # Honba increments only if dealer wins
                    current_honba = ba_values[0] # This value reflects honba *before* increment/reset
                    self.honba = current_honba + 1 if winner == self.dealer else 0
                else: print(f"[Warning] Invalid 'ba' length in AGARI: {ba_str}")
            else: # If 'ba' is missing, reset based on dealer win
                 self.kyotaku = 0
                 self.honba = self.honba + 1 if winner == self.dealer else 0

            # Add AGARI event to history
            agari_data = {k: v for k, v in attrib.items()} # Copy relevant attributes
            self._add_event("AGARI", player=winner, data=agari_data)

        except (ValueError, KeyError, TypeError, IndexError) as e:
             print(f"[Warning] Failed to fully process AGARI attributes: {attrib}. Error: {e}")

    def process_ryuukyoku(self, attrib: dict):
        """Processes a Ryuukyoku (draw) event."""
        try:
            # Update scores based on 'sc' or 'owari' attribute (owari might be more reliable for ryuukyoku payments)
            sc_str = attrib.get("sc", attrib.get("owari")) # Prefer 'owari' if exists
            if sc_str:
                sc_values = list(map(int, sc_str.split(",")))
                if len(sc_values) == 8:
                    # Scores reflect state after noten payments
                    self.current_scores = [sc_values[i*2] for i in range(NUM_PLAYERS)]
                else: print(f"[Warning] Invalid 'sc/owari' length in RYUUKYOKU: {sc_str}")

            # Update honba/kyotaku based on 'ba' attribute
            ba_str = attrib.get("ba")
            if ba_str:
                ba_values = list(map(int, ba_str.split(",")))
                if len(ba_values) == 2:
                    # After ryuukyoku, kyotaku remains
                    self.kyotaku = ba_values[1]
                    # Honba increments
                    current_honba = ba_values[0]
                    self.honba = current_honba + 1
                else: print(f"[Warning] Invalid 'ba' length in RYUUKYOKU: {ba_str}")
            else: # If 'ba' missing, assume honba increments
                 self.honba += 1
                 # Kyotaku state is unknown without 'ba'

            # Add RYUUKYOKU event to history
            ry_data = {k: v for k, v in attrib.items()}
            self._add_event("RYUUKYOKU", player=-1, data=ry_data)

        except (ValueError, KeyError, TypeError, IndexError) as e:
             print(f"[Warning] Failed to fully process RYUUKYOKU attributes: {attrib}. Error: {e}")

    # --- Feature Extraction Methods (from user provided code) ---

    def get_current_dora_indices(self) -> list[int]:
        """Gets the tile type indices (0-33) of the current dora tiles."""
        dora_indices = []
        for indicator_id in self.dora_indicators:
            indicator_index = tile_id_to_index(indicator_id)
            dora_index = -1
            if 0 <= indicator_index <= 26: # Number tiles
                suit_base = (indicator_index // 9) * 9
                num = indicator_index % 9
                dora_index = suit_base + (num + 1) % 9
            elif 27 <= indicator_index <= 30: # Wind tiles (E,S,W,N -> S,W,N,E)
                dora_index = 27 + (indicator_index - 27 + 1) % 4
            elif 31 <= indicator_index <= 33: # Dragon tiles (W,G,R -> G,R,W)
                dora_index = 31 + (indicator_index - 31 + 1) % 3
            if dora_index != -1:
                dora_indices.append(dora_index)
        return dora_indices

    def get_hand_indices(self, player_id: int) -> list[int]:
        """Gets the tile type indices (0-33) of a player's hand."""
        if 0 <= player_id < NUM_PLAYERS:
            return [tile_id_to_index(t) for t in self.player_hands[player_id]]
        return []

    def get_melds_indices(self, player_id: int) -> list[dict]:
        """Gets structured meld information for a player."""
        if 0 <= player_id < NUM_PLAYERS:
             # Return the stored meld dictionaries directly
             # Ensure the structure matches what calculate_shanten expects
             # Example structure assumed by placeholder: {'type': str, 'tiles': list[int], ...}
             # Convert tile IDs to indices if needed by shanten calculator
             return [{'type': m['type'], 'tiles': [tile_id_to_index(t) for t in m['tiles']]}
                     for m in self.player_melds[player_id]]
        return []


    def get_event_sequence_features(self) -> np.ndarray:
        """Converts event history to a padded sequence of numerical vectors."""
        sequence = []
        # Define max dimension for event-specific data (adjust as needed)
        # Example: Discard(tsumogiri=1), Naki(type=1, from=1), Reach(step=1) -> Max=2
        event_specific_dim = 2

        for event in self.event_history:
            # Base vector: type, player+1, tile_index+1, junme
            event_vec_base = [
                event["type"],
                event["player"] + 1,
                event["tile_index"] + 1,
                event["junme"],
            ]
            # Specific vector: initialize with zeros
            event_vec_specific = [0.0] * event_specific_dim
            data = event.get("data", {})
            event_type_code = event["type"]

            # Populate specific vector based on type
            if event_type_code == self.EVENT_TYPES["DISCARD"]:
                event_vec_specific[0] = 1.0 if data.get("tsumogiri", False) else 0.0
            elif event_type_code == self.EVENT_TYPES["N"]:
                # Use numerical naki type code, handle '不明' or missing
                naki_type_code = data.get("naki_type", -1)
                event_vec_specific[0] = float(naki_type_code + 1) # Map -1 (不明) to 0, 0 (チー) to 1, etc.
                event_vec_specific[1] = float(data.get("from_who", -1) + 1) # Map -1 to 0
            elif event_type_code == self.EVENT_TYPES["REACH"]:
                event_vec_specific[0] = float(data.get("step", 0))

            # Combine and append
            sequence.append(event_vec_base + event_vec_specific)

        # Padding
        padding_length = MAX_EVENT_HISTORY - len(sequence)
        event_dim = 4 + event_specific_dim
        # Padding vector: Use PADDING type code, rest zeros
        padding_vec = [float(self.EVENT_TYPES["PADDING"])] + [0.0] * (event_dim - 1)
        padded_sequence = sequence + [padding_vec] * padding_length

        try:
            return np.array(padded_sequence, dtype=np.float32)
        except ValueError as e:
             print(f"[ERROR] Failed to convert event sequence to NumPy array. Check vector lengths.")
             # Print details for debugging
             for i, vec in enumerate(padded_sequence): print(f"  Vec {i}, Len {len(vec)}: {vec}")
             raise e


    def get_static_features(self, player_id: int) -> np.ndarray:
        """Generates a static feature vector for the given player's perspective."""
        if not (0 <= player_id < NUM_PLAYERS): raise ValueError("Invalid player_id")
        features = []

        # 1. Own Hand Representation (Tile ID presence + Count)
        my_hand_representation = np.zeros((NUM_TILE_TYPES, 5), dtype=np.int8)
        hand_indices_count = defaultdict(int)
        for tile_id in self.player_hands[player_id]:
             idx = tile_id_to_index(tile_id)
             if idx != -1:
                 offset = tile_id % 4
                 if 0 <= offset <= 3: my_hand_representation[idx, offset] = 1
                 hand_indices_count[idx] += 1
        for idx, count in hand_indices_count.items():
            my_hand_representation[idx, 4] = count
        features.append(my_hand_representation.flatten())

        # 2. Dora Information (Indicators + Dora Tiles)
        dora_indicator_vec = np.zeros(NUM_TILE_TYPES, dtype=np.int8)
        for ind_id in self.dora_indicators:
             idx = tile_id_to_index(ind_id)
             if idx != -1: dora_indicator_vec[idx] = 1
        features.append(dora_indicator_vec)
        current_dora_indices = self.get_current_dora_indices()
        dora_tile_vec = np.zeros(NUM_TILE_TYPES, dtype=np.int8)
        for dora_idx in current_dora_indices:
             if dora_idx != -1: dora_tile_vec[dora_idx] = 1
        features.append(dora_tile_vec)

        # 3. Public Information per Player (Relative Order: Self -> Shimo -> Toimen -> Kami)
        for p_offset in range(NUM_PLAYERS):
            target_player = (player_id + p_offset) % NUM_PLAYERS
            # Discards (Counts per tile type)
            discard_counts = np.zeros(NUM_TILE_TYPES, dtype=np.int8)
            # Genbutsu (Tiles safe against target player if they are in reach)
            genbutsu_flag = np.zeros(NUM_TILE_TYPES, dtype=np.int8)
            is_target_reach = self.player_reach_status[target_player] == 2
            reach_discard_idx = self.player_reach_discard_index[target_player]

            for i, (tile, _) in enumerate(self.player_discards[target_player]):
                tile_idx = tile_id_to_index(tile)
                if tile_idx != -1:
                    discard_counts[tile_idx] += 1
                    # Mark genbutsu if discarded at or after reach declaration discard
                    if is_target_reach and reach_discard_idx != -1 and i >= reach_discard_idx:
                         genbutsu_flag[tile_idx] = 1
            features.append(discard_counts)
            if p_offset != 0: # Only add genbutsu for opponents
                features.append(genbutsu_flag)

            # Melds (Simplified: One-hot vector per tile type present in any meld)
            meld_presence_vec = np.zeros(NUM_TILE_TYPES, dtype=np.int8)
            # More detailed meld features could be added here (type, which specific tiles, etc.)
            for meld_data in self.player_melds[target_player]:
                 for tile_id in meld_data.get('tiles', []):
                      idx = tile_id_to_index(tile_id)
                      if idx != -1: meld_presence_vec[idx] = 1
            features.append(meld_presence_vec)

            # Reach Status & Junme (Status 0-2, Normalized Junme 0-1)
            reach_stat = float(self.player_reach_status[target_player])
            reach_jun = float(self.player_reach_junme[target_player] / 18.0) if self.player_reach_junme[target_player] > 0 else 0.0
            features.append(np.array([reach_stat, reach_jun], dtype=np.float32))

        # 4. Field Information (Round, Honba, Kyotaku, Junme, Dealer, Own Wind, Wall Count)
        round_wind_feat = float(self.round_num_wind / 15.0) # Normalize E1-N4 (0-15) approx
        honba_feat = min(float(self.honba / 5.0), 1.0) # Cap normalization at 5 honba
        kyotaku_feat = min(float(self.kyotaku / 4.0), 1.0) # Cap normalization at 4 sticks
        junme_feat = min(float(self.junme / 18.0), 1.0) # Normalize against typical max turns
        is_dealer_feat = 1.0 if self.dealer == player_id else 0.0
        # Own wind (0=E, 1=S, 2=W, 3=N relative to dealer)
        my_wind_rel = (player_id - self.dealer + NUM_PLAYERS) % NUM_PLAYERS
        my_wind_vec = np.zeros(4, dtype=np.float32); my_wind_vec[my_wind_rel] = 1.0
        # Wall count (Normalized 0-1)
        wall_count_feat = max(0.0, float(self.wall_tile_count / 70.0)) # Based on initial 70 tiles

        field_features = np.array([round_wind_feat, honba_feat, kyotaku_feat, junme_feat, is_dealer_feat, wall_count_feat], dtype=np.float32)
        features.append(field_features)
        features.append(my_wind_vec) # Add own wind one-hot vector

        # 5. Scores (Normalized and Rotated relative to player)
        scores_feat = np.array(self.current_scores, dtype=np.float32)
        # Normalize scores (e.g., divide by average starting score or a max value)
        normalized_scores = scores_feat / 30000.0 # Normalize against 30k? Adjust as needed
        rotated_scores = np.roll(normalized_scores, -player_id) # Put own score first
        features.append(rotated_scores)

        # 6. Shanten & Ukeire (Using placeholder)
        hand_indices = self.get_hand_indices(player_id)
        melds_info_for_shanten = self.get_melds_indices(player_id) # Get melds in expected format
        try:
             shanten, ukeire_indices = calculate_shanten(hand_indices, melds_info_for_shanten)
        except Exception as e:
             print(f"[Error] Shanten calculation failed for P{player_id}: {e}")
             shanten = 8 # Fallback shanten
             ukeire_indices = []

        # Shanten (One-hot encoding: index 0=Agari, 1=Tenpai, ..., 9=8+ shanten)
        shanten_vec = np.zeros(10, dtype=np.float32)
        shanten_idx = min(max(0, shanten + 1), 9) # Clamp index between 0 and 9
        shanten_vec[shanten_idx] = 1.0
        features.append(shanten_vec)

        # Ukeire (Which tile types improve shanten - multi-hot)
        ukeire_vec = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        for idx in ukeire_indices:
             if 0 <= idx < NUM_TILE_TYPES: ukeire_vec[idx] = 1.0
        features.append(ukeire_vec)
        # Number of ukeire tiles (Normalized)
        # TODO: Calculate actual number of ukeire tiles considering remaining wall count
        num_ukeire_types = len(ukeire_indices)
        features.append(np.array([min(num_ukeire_types / 15.0, 1.0)], dtype=np.float32)) # Normalize against arbitrary max

        # Concatenate all feature blocks
        try:
            concatenated_features = np.concatenate([f.flatten() for f in features])
            # Check for NaN/Inf values
            if np.isnan(concatenated_features).any() or np.isinf(concatenated_features).any():
                print(f"[Error] NaN/Inf detected in static features for P{player_id}! Replacing with 0.")
                concatenated_features = np.nan_to_num(concatenated_features, nan=0.0, posinf=0.0, neginf=0.0) # Replace with 0
            return concatenated_features
        except ValueError as e:
            print(f"[Error] Concatenating static features failed for P{player_id}.")
            for i, f_block in enumerate(features): print(f"  Shape of feature block {i}: {np.array(f_block).shape}")
            raise e

    def get_valid_discard_options(self, player_id: int) -> list[int]:
        """Returns a list of valid discard tile type indices (0-33) for the player."""
        if not (0 <= player_id < NUM_PLAYERS): return []
        options = set()
        hand = self.player_hands[player_id]
        is_reach = self.player_reach_status[player_id] == 2

        if is_reach:
            # In reach, must discard the drawn tile unless Ankan/Kakan is possible
            # Simple version: only allow discarding the drawn tile
            # Assumes hand has 14 tiles (13 + drawn)
            if len(hand) % 3 == 2 and hand: # Hand size check for drawn tile
                 # The last tile added is the one drawn (assuming process_tsumo adds to end)
                 drawn_tile_id = hand[-1]
                 drawn_tile_index = tile_id_to_index(drawn_tile_id)
                 if drawn_tile_index != -1: options.add(drawn_tile_index)
                 # TODO: Add logic for Ankan/Kakan option during reach if needed
            # else: print(f"[Warning] Reach P{player_id} hand size {len(hand)} is not as expected (14).")
        else:
             # Not in reach, any tile in hand can be discarded
             for tile_id in hand:
                  idx = tile_id_to_index(tile_id)
                  if idx != -1: options.add(idx)

        return sorted(list(options))