# /ver_1.1.9/debug_and_process.py
import sys
import argparse
import os
import traceback # For detailed error reporting
import numpy as np # For feature inspection
import xml.etree.ElementTree as ET
import urllib.parse
from typing import List, Dict, Any, Tuple
from collections import defaultdict, deque

# --- Dependency Imports ---
# Ensure naki_utils.py and tile_utils.py are in the same directory
try:
    from tile_utils import tile_id_to_index, tile_id_to_string, tile_index_to_id
    from naki_utils import decode_naki
    print("Successfully imported from tile_utils and naki_utils.")
except ImportError as e:
    print(f"[FATAL ERROR] Cannot import from tile_utils/naki_utils: {e}")
    print("Ensure tile_utils.py and naki_utils.py are in the ./ver_1.1.9/ directory.")
    sys.exit(1)
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

# ==============================================================================
# == XML Parsing Logic (from full_mahjong_parser.py) ==
# ==============================================================================

def parse_full_mahjong_log(xml_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Parses a Tenhou XML log file into game metadata and a list of round data.

    Args:
        xml_path: Path to the XML log file.

    Returns:
        A tuple containing:
            - meta: Dictionary with overall game info (<GO>, <UN>, <TAIKYOKU> attributes).
            - rounds: List of dictionaries, each representing a round.
                      Each round dict contains 'round_index', 'init' (attributes),
                      'events' (list of tags/attributes), and 'result' (final event).
    """
    meta = {}
    rounds = []
    player_name_map = {} # Maps player index (0-3) to decoded name

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"[Error] Failed to parse XML file: {xml_path} - {e}")
        return {}, []
    except FileNotFoundError:
        print(f"[Error] XML file not found: {xml_path}")
        return {}, []
    except Exception as e:
        print(f"[Error] Unexpected error reading XML file {xml_path}: {str(e)}")
        return {}, []

    current_round_data = None
    round_index_counter = 0 # Internal counter, 0-based for list index

    for elem in root:
        tag = elem.tag
        attrib = elem.attrib

        # --- Metadata Tags ---
        if tag == "GO":
            meta['go'] = attrib
        elif tag == "UN":
            meta['un'] = attrib
            # Decode player names
            for i in range(4):
                name_key = f'n{i}'
                if name_key in attrib:
                    try:
                        player_name = urllib.parse.unquote(attrib[name_key])
                        player_name_map[i] = player_name
                    except Exception as e:
                        print(f"[Warning] Could not decode player name {attrib[name_key]}: {e}")
                        player_name_map[i] = f"player_{i}_undecoded"
                else:
                    player_name_map[i] = f"player_{i}" # Default if name missing
            meta['player_names'] = [player_name_map.get(i, f'p{i}') for i in range(4)]
        elif tag == "TAIKYOKU":
            meta['taikyoku'] = attrib
            # Note: Events within TAIKYOKU but before INIT are currently ignored.

        # --- Round Start ---
        elif tag == "INIT":
            round_index_counter += 1
            current_round_data = {
                "round_index": round_index_counter, # 1-based index for user reference
                "init": attrib,
                "events": [],
                "result": None
            }
            rounds.append(current_round_data)

        # --- Events Within a Round ---
        elif current_round_data is not None:
            event_data = {"tag": tag, "attrib": attrib} # Store raw tag and attributes
            current_round_data["events"].append(event_data)

            # Check for round end tags
            if tag in ["AGARI", "RYUUKYOKU"]:
                current_round_data["result"] = event_data # Store the result event
                # Don't reset current_round_data here, wait for next INIT or end of file

        # --- Other Top-Level Tags (Optional Handling) ---
        elif tag == "Owari":
             meta['owari'] = attrib
             if current_round_data: # Sometimes Owari appears after last AGARI/RYUUKYOKU
                  # Mark the last round ended if it wasn't explicitly done
                  if not current_round_data.get("result"):
                       # This case is odd, maybe log a warning?
                       print(f"[Warning] Encountered <Owari> tag before explicit round end (AGARI/RYUUKYOKU) in round {current_round_data.get('round_index')}")
                       current_round_data["result"] = {"tag": "Owari", "attrib": attrib} # Mark with Owari?
                  current_round_data = None # End processing after Owari

    # Add player names to meta if not already present from UN tag
    if 'player_names' not in meta:
         meta['player_names'] = [player_name_map.get(i, f'p{i}') for i in range(4)]

    # print(f"Parsed {len(rounds)} rounds from {os.path.basename(xml_path)}.") # Moved to calling function
    return meta, rounds

# ==============================================================================
# == Game State Management Logic (from game_state.py) ==
# ==============================================================================

class GameState:
    """
    Manages the state of a Mahjong game round, processes events parsed
    from XML, and generates feature vectors for ML models (without shanten).
    """
    TSUMO_TAGS = {"T": 0, "U": 1, "V": 2, "W": 3}
    DISCARD_TAGS = {"D": 0, "E": 1, "F": 2, "G": 3}

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
        self.player_hands: list[list[int]] = [[] for _ in range(NUM_PLAYERS)]
        self.player_discards: list[list[tuple[int, bool]]] = [[] for _ in range(NUM_PLAYERS)]
        self.player_melds: list[list[dict]] = [[] for _ in range(NUM_PLAYERS)]
        self.player_reach_status: list[int] = [0] * NUM_PLAYERS # 0: No, 1: Declared, 2: Accepted
        self.player_reach_junme: list[float] = [-1.0] * NUM_PLAYERS # Junme when reach accepted
        self.player_reach_discard_index: list[int] = [-1] * NUM_PLAYERS # Index in discard list when reach accepted
        self.current_player: int = -1      # Index of the player whose turn it is
        self.junme: float = 0.0            # Current turn number (increments per player turn)
        self.last_discard_event_player: int = -1 # Player who made the last discard
        self.last_discard_event_tile_id: int = -1 # Tile ID of the last discard
        self.last_discard_event_tsumogiri: bool = False
        self.can_ron: bool = False         # Flag indicating if Ron is possible on the last discard
        self.naki_occurred_in_turn: bool = False # Flag if naki happened after last tsumo
        self.is_rinshan: bool = False      # Flag if the next tsumo is from rinshanpai
        self.event_history: deque = deque(maxlen=MAX_EVENT_HISTORY) # Fixed-size event history
        self.wall_tile_count: int = 70     # Estimated remaining tiles in wall (excluding dead wall)

    def _add_event(self, event_type: str, player: int, tile: int = -1, data: dict = None):
        """Adds a structured event to the event history deque."""
        if data is None: data = {}
        event_code = EVENT_TYPES.get(event_type, -1)
        if event_code == -1: return

        event_info = {
            "type": event_code,
            "player": player,
            "tile_index": tile_id_to_index(tile),
            "junme": int(np.ceil(self.junme)),
            "data": data
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
                self.round_num_wind = int(seed_parts[0])
                self.honba = int(seed_parts[1])
                self.kyotaku = int(seed_parts[2])
                dora_indicator_id = int(seed_parts[5])
                if 0 <= dora_indicator_id <= 135:
                    self.dora_indicators = [dora_indicator_id]
                    self._add_event("DORA", player=-1, tile=dora_indicator_id)
                else:
                    print(f"[Warning] Invalid initial Dora indicator ID: {dora_indicator_id}")
                    self.dora_indicators = []
            else: raise ValueError("Seed string too short")
        except (IndexError, ValueError) as e:
            print(f"[Warning] Failed to parse seed string '{init_info.get('seed')}': {e}")
            self.round_num_wind=0; self.honba=0; self.kyotaku=0; self.dora_indicators=[]

        self.dealer = int(init_info.get("oya", -1))
        if not (0 <= self.dealer < NUM_PLAYERS):
             print(f"[Warning] Invalid dealer index '{init_info.get('oya')}' found. Setting to 0.")
             self.dealer = 0
        self.current_player = self.dealer

        try:
            raw_scores = init_info.get("ten", "25000,25000,25000,25000").split(",")
            if len(raw_scores) == 4:
                self.initial_scores = [int(float(s)) * 100 for s in raw_scores]
                self.current_scores = list(self.initial_scores)
            else: raise ValueError("Incorrect number of scores")
        except (ValueError, TypeError) as e:
            print(f"[Warning] Failed to parse 'ten' attribute '{init_info.get('ten')}': {e}")
            self.initial_scores=[25000]*NUM_PLAYERS; self.current_scores=[25000]*NUM_PLAYERS

        for p in range(NUM_PLAYERS):
            hand_str = init_info.get(f"hai{p}", "")
            self.player_hands[p] = []
            try:
                if hand_str:
                    hand_ids = [int(h) for h in hand_str.split(',') if h]
                    valid_hand_ids = [tid for tid in hand_ids if 0 <= tid <= 135]
                    if len(valid_hand_ids) != len(hand_ids):
                        print(f"[Warning] Invalid tile IDs found in initial hand for P{p}: {hand_str}")
                    self.player_hands[p] = valid_hand_ids
                    self._sort_hand(p)
            except ValueError as e:
                print(f"[Warning] Failed to parse initial hand 'hai{p}' for P{p}: {e}")

        self.wall_tile_count = 136 - 14 - sum(len(h) for h in self.player_hands)
        self.junme = 0.0

        init_data = {"round": self.round_num_wind, "honba": self.honba, "kyotaku": self.kyotaku}
        self._add_event("INIT", player=self.dealer, data=init_data)

    def process_tsumo(self, player_id: int, tile_id: int):
        """Processes a Tsumo (draw) event."""
        if not (0 <= player_id < NUM_PLAYERS): print(f"[ERROR] Invalid player_id {player_id} in process_tsumo"); return
        if not (0 <= tile_id <= 135): print(f"[ERROR] Invalid tile_id {tile_id} in process_tsumo"); return

        self.current_player = player_id

        # --- Junme Update Logic ---
        # Junme increments based on player turns, roughly 1 per full cycle.
        # Let's use a simpler approach: increment after each set of 4 tsumo/discard cycles
        # More accurately: increment when player 0 draws (after first round)
        is_first_round = self.junme < 1.0
        is_dealer_turn = player_id == self.dealer

        if not self.is_rinshan:
            if is_first_round and is_dealer_turn and self.junme == 0.0:
                 self.junme = 0.1 # Dealer's first turn starts
            elif not is_first_round and player_id == 0:
                 self.junme = np.floor(self.junme) + 1.0 # New full round starts
            elif is_first_round and not is_dealer_turn and self.junme < 1.0:
                 # Check if all players have had their first turn
                 # This is tricky without tracking turns explicitly.
                 # Approximate: assume first round ends when player 0 draws next.
                 # Let's simplify: We'll use the floor of junme for features.
                 # Increment junme fractionally for now.
                 if self.junme < 1.0:
                      self.junme = 1.0 # Consider first round done after first non-dealer draw
            # No junme increment during rinshan

        rinshan_draw = self.is_rinshan
        if rinshan_draw:
            self.is_rinshan = False
        else:
             if self.wall_tile_count > 0: self.wall_tile_count -= 1
             else: print("[Warning] Tsumo occurred with 0 wall tiles remaining.")


        self.naki_occurred_in_turn = False

        self.player_hands[player_id].append(tile_id)
        self._sort_hand(player_id)

        tsumo_data = {"rinshan": rinshan_draw}
        self._add_event("TSUMO", player=player_id, tile=tile_id, data=tsumo_data)

        self.can_ron = False

    def process_discard(self, player_id: int, tile_id: int, tsumogiri: bool):
        """Processes a Discard event."""
        if not (0 <= player_id < NUM_PLAYERS): print(f"[ERROR] Invalid player_id {player_id} in process_discard"); return
        if not (0 <= tile_id <= 135): print(f"[ERROR] Invalid tile_id {tile_id} in process_discard"); return

        if tile_id in self.player_hands[player_id]:
            self.player_hands[player_id].remove(tile_id)
            self._sort_hand(player_id)
        else:
            print(f"[Warning] P{player_id} discarding {tile_id_to_string(tile_id)} ({tile_id}) not in hand: {[tile_id_to_string(t) for t in self.player_hands[player_id]]}")

        self.player_discards[player_id].append((tile_id, tsumogiri))

        discard_data = {"tsumogiri": int(tsumogiri)}
        self._add_event("DISCARD", player=player_id, tile=tile_id, data=discard_data)

        self.last_discard_event_player = player_id
        self.last_discard_event_tile_id = tile_id
        self.last_discard_event_tsumogiri = tsumogiri
        self.can_ron = True

        if self.player_reach_status[player_id] == 1:
            self.player_reach_status[player_id] = 2
            self.player_reach_discard_index[player_id] = len(self.player_discards[player_id]) - 1
            self.player_reach_junme[player_id] = self.junme
            if self.current_scores[player_id] >= 1000: # Check score again just before deduction
                 self.kyotaku += 1
                 self.current_scores[player_id] -= 1000
            else:
                 print(f"[Warning] P{player_id} reached but score {self.current_scores[player_id]} became < 1000 before deduction.")
                 self.player_reach_status[player_id] = 0 # Cancel reach? Or allow negative? Tenhou allows it.
                 self.kyotaku += 1
                 self.current_scores[player_id] -= 1000


            reach_data = {"step": 2, "junme": int(np.ceil(self.junme))}
            self._add_event("REACH", player=player_id, data=reach_data)

    def process_naki(self, naki_player_id: int, meld_code: int):
        """Processes a Naki (call) event using naki_utils."""
        if not (0 <= naki_player_id < NUM_PLAYERS): print(f"[ERROR] Invalid naki_player_id {naki_player_id}"); return

        # Use the unchanged naki_utils.py provided by the user
        # Expected output: dict with 'type', 'tiles', 'consumed', 'from_who_relative' etc.
        naki_info = decode_naki(meld_code)
        naki_type = naki_info.get("type", "不明")
        decoded_tiles = naki_info.get("tiles", []) # Tiles in the meld
        consumed_tiles_from_naki_util = naki_info.get("consumed", []) # Tiles from hand used in naki
        from_who_relative = naki_info.get("from_who_relative", -1)

        if naki_type == "不明" or not decoded_tiles:
            print(f"[Warning] decode_naki failed or returned empty tiles for m={meld_code}. Skipping process_naki.")
            return

        from_who_player_abs = -1
        called_tile_id = -1
        tiles_to_remove = []

        # Determine absolute 'from_who' and 'called_tile' for external calls
        if naki_type in ["チー", "ポン", "大明槓"]:
            called_tile_id = self.last_discard_event_tile_id
            discarder_player_id = self.last_discard_event_player

            if discarder_player_id == -1 or called_tile_id == -1:
                print(f"[Warning] Naki {naki_type} P{naki_player_id} (m={meld_code}) occurred without valid preceding discard info. Skipping.")
                return
            if discarder_player_id == naki_player_id:
                print(f"[Warning] Naki {naki_type} P{naki_player_id} (m={meld_code}) called from self discard (invalid). Skipping.")
                return

            from_who_player_abs = discarder_player_id # Trust actual discarder

            # Determine tiles to remove: all decoded tiles MINUS the called tile
            temp_remove_candidates = list(decoded_tiles)
            try:
                 temp_remove_candidates.remove(called_tile_id)
                 tiles_to_remove = temp_remove_candidates
            except ValueError:
                 print(f"[Error] Naki P{naki_player_id} {naki_type}: Called tile {tile_id_to_string(called_tile_id)} not found in decoded meld {[tile_id_to_string(t) for t in decoded_tiles]}. Aborting naki.")
                 return

            if naki_type == "大明槓": self.is_rinshan = True

        elif naki_type == "加槓":
            # Kakan: Tile added is in 'consumed'. Find original Pon to update.
            if not consumed_tiles_from_naki_util:
                 print(f"[Error] Kakan m={meld_code}: decode_naki did not return consumed tile. Aborting.")
                 return
            added_tile_id = consumed_tiles_from_naki_util[0]
            tiles_to_remove = [added_tile_id] # Remove the added tile from hand
            from_who_player_abs = naki_player_id # From self
            called_tile_id = -1 # No external called tile
            self.is_rinshan = True

        elif naki_type == "暗槓":
            # Ankan: All 4 tiles are 'consumed' from hand.
            tiles_to_remove = list(decoded_tiles) # Should be 4 tiles
            from_who_player_abs = naki_player_id # From self
            called_tile_id = -1 # No external called tile
            self.is_rinshan = True

        # --- Perform State Update ---
        # Verify and remove tiles from hand
        removed_count = 0
        temp_hand = list(self.player_hands[naki_player_id])
        indices_to_pop = []
        for tile_to_remove in tiles_to_remove:
            found = False
            for i in range(len(temp_hand)):
                # Check ID match, ensure index not already marked
                if temp_hand[i] == tile_to_remove and i not in indices_to_pop:
                    indices_to_pop.append(i)
                    removed_count += 1
                    found = True
                    break
            if not found:
                 print(f"[Error] Naki P{naki_player_id} {naki_type}: Could not find tile {tile_id_to_string(tile_to_remove)} in hand to remove. Hand: {[tile_id_to_string(t) for t in self.player_hands[naki_player_id]]}")
                 return # Abort if tile missing

        # Apply removals if all necessary tiles were found
        if removed_count == len(tiles_to_remove):
             # Remove by index in reverse order
             for i in sorted(indices_to_pop, reverse=True):
                 self.player_hands[naki_player_id].pop(i)
             self._sort_hand(naki_player_id)

             # Add or update meld in player_melds
             if naki_type == "加槓":
                 # Find and update the existing Pon meld
                 updated = False
                 pon_index = tile_id_to_index(tiles_to_remove[0]) # Index of the added tile
                 for i, existing_meld in enumerate(self.player_melds[naki_player_id]):
                      if existing_meld['type'] == "ポン" and tile_id_to_index(existing_meld['tiles'][0]) == pon_index:
                          self.player_melds[naki_player_id][i]['type'] = "加槓"
                          self.player_melds[naki_player_id][i]['tiles'] = sorted(existing_meld['tiles'] + tiles_to_remove)
                          self.player_melds[naki_player_id][i]['jun'] = self.junme # Update junme
                          updated = True
                          break
                 if not updated: print(f"[Error] Kakan P{naki_player_id}: Corresponding Pon not found to update.")
             else:
                 # Add new meld for Chi, Pon, Daiminkan, Ankan
                 new_meld = {
                     'type': naki_type,
                     'tiles': decoded_tiles,
                     'from_who': from_who_player_abs,
                     'called_tile': called_tile_id,
                     'm': meld_code,
                     'jun': self.junme
                 }
                 self.player_melds[naki_player_id].append(new_meld)

             # Update game flow state
             self.current_player = naki_player_id
             self.naki_occurred_in_turn = True
             self.can_ron = False
             self.last_discard_event_player = -1
             self.last_discard_event_tile_id = -1
             self.last_discard_event_tsumogiri = False

             # Add event to history
             naki_event_data = {
                 "naki_type": NAKI_TYPES.get(naki_type, -1),
                 "from_who": from_who_player_abs # Store absolute index
             }
             event_tile = called_tile_id if naki_type in ["チー", "ポン", "大明槓"] else decoded_tiles[0]
             self._add_event("N", player=naki_player_id, tile=event_tile, data=naki_event_data)
        else:
             print(f"[Error] Naki P{naki_player_id} {naki_type}: Mismatch in removable tiles count ({removed_count} vs {len(tiles_to_remove)}). Aborting naki process.")


    def process_reach(self, player_id: int, step: int):
        """Processes a Reach declaration (step 1). Step 2 is handled by discard."""
        if not (0 <= player_id < NUM_PLAYERS): print(f"[ERROR] Invalid player_id {player_id} in process_reach"); return

        if step == 1:
            if self.player_reach_status[player_id] != 0: return
            if self.current_scores[player_id] < 1000: return

            self.player_reach_status[player_id] = 1
            reach_data = {"step": 1}
            self._add_event("REACH", player=player_id, data=reach_data)

    def process_dora(self, tile_id: int):
        """Processes a Dora indicator reveal event."""
        if not (0 <= tile_id <= 135): print(f"[ERROR] Invalid tile_id {tile_id} in process_dora"); return
        self.dora_indicators.append(tile_id)
        self._add_event("DORA", player=-1, tile=tile_id)

    def process_agari(self, attrib: dict):
        """Processes an Agari (win) event, updating scores etc."""
        try:
            winner = int(attrib.get("who", -1))
            if not (0 <= winner < NUM_PLAYERS): raise ValueError("Invalid winner")

            sc_str = attrib.get("sc")
            if sc_str:
                sc_values = [int(float(s)) for s in sc_str.split(",")]
                if len(sc_values) == 8: self.current_scores = [v * 100 for v in sc_values[0::2]]
                else: print(f"[Warning] Invalid 'sc' length in AGARI: {sc_str}")

            ba_str = attrib.get("ba")
            if ba_str:
                ba_values = [int(float(s)) for s in ba_str.split(",")]
                if len(ba_values) == 2:
                    self.honba = ba_values[0] + 1 if winner == self.dealer else 0
                    self.kyotaku = 0
                else: print(f"[Warning] Invalid 'ba' length in AGARI: {ba_str}")
            else:
                 self.honba = self.honba + 1 if winner == self.dealer else 0
                 self.kyotaku = 0

            agari_data = {k: v for k, v in attrib.items()}
            machi_tile = int(attrib.get('machi', -1))
            self._add_event("AGARI", player=winner, tile=machi_tile, data=agari_data)

        except (ValueError, KeyError, TypeError, IndexError) as e:
             print(f"[Warning] Failed to fully process AGARI attributes: {attrib}. Error: {e}")

    def process_ryuukyoku(self, attrib: dict):
        """Processes a Ryuukyoku (draw) event."""
        try:
            sc_str = attrib.get("sc", attrib.get("owari"))
            if sc_str:
                sc_values = [int(float(s)) for s in sc_str.split(",")]
                if len(sc_values) == 8: self.current_scores = [v * 100 for v in sc_values[0::2]]
                else: print(f"[Warning] Invalid 'sc/owari' length in RYUUKYOKU: {sc_str}")

            ba_str = attrib.get("ba")
            if ba_str:
                ba_values = [int(float(s)) for s in ba_str.split(",")]
                if len(ba_values) == 2:
                    self.honba = ba_values[0] + 1
                    self.kyotaku = ba_values[1]
                else: print(f"[Warning] Invalid 'ba' length in RYUUKYOKU: {ba_str}")
            else: self.honba += 1

            ry_data = {k: v for k, v in attrib.items()}
            self._add_event("RYUUKYOKU", player=-1, data=ry_data)

        except (ValueError, KeyError, TypeError, IndexError) as e:
             print(f"[Warning] Failed to fully process RYUUKYOKU attributes: {attrib}. Error: {e}")

    # --- Feature Extraction Methods ---
    def get_current_dora_indices(self) -> list[int]:
        """Gets the tile type indices (0-33) of the actual dora tiles."""
        dora_indices = []
        for indicator_id in self.dora_indicators:
            indicator_index = tile_id_to_index(indicator_id)
            dora_index = -1
            if indicator_index == -1: continue
            if 0 <= indicator_index <= 26:
                suit_base = (indicator_index // 9) * 9
                num = indicator_index % 9
                dora_index = suit_base + (num + 1) % 9
            elif 27 <= indicator_index <= 30:
                dora_index = 27 + (indicator_index - 27 + 1) % 4
            elif 31 <= indicator_index <= 33:
                dora_index = 31 + (indicator_index - 31 + 1) % 3
            if dora_index != -1: dora_indices.append(dora_index)
        return dora_indices

    def get_hand_indices(self, player_id: int) -> list[int]:
        """Gets the tile type indices (0-33) of a player's hand."""
        if 0 <= player_id < NUM_PLAYERS:
            return [tile_id_to_index(t) for t in self.player_hands[player_id] if tile_id_to_index(t) != -1]
        return []

    def get_event_sequence_features(self) -> np.ndarray:
        """Converts event history to a padded sequence of numerical vectors."""
        sequence = []
        event_specific_dim = 2
        event_base_dim = 4
        event_total_dim = event_base_dim + event_specific_dim

        for event in self.event_history:
            event_vec = np.zeros(event_total_dim, dtype=np.float32)
            event_vec[0] = float(event["type"])
            event_vec[1] = float(event["player"] + 1)
            event_vec[2] = float(event["tile_index"] + 1)
            event_vec[3] = float(event["junme"])
            data = event.get("data", {})
            event_type_code = event["type"]
            if event_type_code == EVENT_TYPES["DISCARD"]:
                event_vec[event_base_dim + 0] = float(data.get("tsumogiri", 0))
            elif event_type_code == EVENT_TYPES["N"]:
                naki_type_code = data.get("naki_type", -1)
                event_vec[event_base_dim + 0] = float(naki_type_code + 1)
                event_vec[event_base_dim + 1] = float(data.get("from_who", -1) + 1)
            elif event_type_code == EVENT_TYPES["REACH"]:
                event_vec[event_base_dim + 0] = float(data.get("step", 0))
                event_vec[event_base_dim + 1] = float(data.get("junme", 0))
            sequence.append(event_vec)

        padding_length = MAX_EVENT_HISTORY - len(sequence)
        padding_vec = np.zeros(event_total_dim, dtype=np.float32)
        padding_vec[0] = float(EVENT_TYPES["PADDING"])
        padded_sequence = sequence + [padding_vec] * padding_length
        try: return np.array(padded_sequence[-MAX_EVENT_HISTORY:], dtype=np.float32) # Ensure fixed size
        except ValueError as e: print(f"[ERROR] Event sequence conversion failed."); raise e

    def get_static_features(self, player_id: int) -> np.ndarray:
        """Generates a static feature vector (dim=157) for the given player's perspective."""
        if not (0 <= player_id < NUM_PLAYERS): raise ValueError("Invalid player_id")
        features = np.zeros(STATIC_FEATURE_DIM, dtype=np.float32)
        idx = 0
        DIM_GAME_CONTEXT = 8; DIM_PLAYER_SPECIFIC = 5; DIM_HAND_COUNTS = 34
        DIM_DORA_INDICATORS = 34; DIM_PLAYER_DISCARDS = 34; DIM_ALL_VISIBLE = 34
        DIM_PLAYER_POS_REACH = 8

        try: # Wrap major blocks in try-except
            # 1. Game context (8 features)
            features[idx:idx+DIM_GAME_CONTEXT] = [
                self.round_num_wind, self.honba, self.kyotaku, self.dealer,
                self.wall_tile_count, float(player_id == self.dealer), self.junme,
                len(self.dora_indicators)
            ]; idx += DIM_GAME_CONTEXT

            # 2. Player specific data (5 features)
            features[idx:idx+DIM_PLAYER_SPECIFIC] = [
                self.player_reach_status[player_id], self.player_reach_junme[player_id],
                len(self.player_discards[player_id]), len(self.player_melds[player_id]),
                len(self.player_hands[player_id])
            ]; idx += DIM_PLAYER_SPECIFIC

            # 3. Hand tiles counts (34 features)
            start_idx = idx
            hand_indices = self.get_hand_indices(player_id)
            for tile_idx in hand_indices:
                if 0 <= tile_idx < NUM_TILE_TYPES: features[start_idx + tile_idx] += 1.0
            idx = start_idx + DIM_HAND_COUNTS

            # 4. Dora indicators counts (34 features)
            start_idx = idx
            for indicator_id in self.dora_indicators:
                 indicator_idx = tile_id_to_index(indicator_id)
                 if 0 <= indicator_idx < NUM_TILE_TYPES: features[start_idx + indicator_idx] += 1.0
            idx = start_idx + DIM_DORA_INDICATORS

            # 5. Player discards counts (34 features)
            start_idx = idx
            for tile_id, _ in self.player_discards[player_id]:
                 discard_idx = tile_id_to_index(tile_id)
                 if 0 <= discard_idx < NUM_TILE_TYPES: features[start_idx + discard_idx] += 1.0
            idx = start_idx + DIM_PLAYER_DISCARDS

            # 6. All visible discards and melds (34 features)
            start_idx = idx
            for p in range(NUM_PLAYERS):
                for tile_id, _ in self.player_discards[p]:
                     discard_idx = tile_id_to_index(tile_id)
                     if 0 <= discard_idx < NUM_TILE_TYPES: features[start_idx + discard_idx] += 1.0
                for meld in self.player_melds[p]:
                     for tile_id in meld.get("tiles", []):
                          meld_tile_idx = tile_id_to_index(tile_id)
                          if 0 <= meld_tile_idx < NUM_TILE_TYPES: features[start_idx + meld_tile_idx] += 1.0
            idx = start_idx + DIM_ALL_VISIBLE

            # 7. Player positions relative + Reach status (8 features)
            start_idx = idx
            for p_offset in range(NUM_PLAYERS):
                p_abs = (player_id + p_offset) % NUM_PLAYERS
                features[idx] = float(p_abs == player_id) # Is Self
                idx += 1
                features[idx] = float(self.player_reach_status[p_abs] == 2) # Is Reach Accepted
                idx += 1
            # idx = start_idx + DIM_PLAYER_POS_REACH # This line was causing index mismatch

        except Exception as e:
            print(f"[ERROR] Exception during static feature generation for P{player_id}: {e}")
            traceback.print_exc(limit=1)
            # Return zeros if error occurs during generation
            return np.zeros(STATIC_FEATURE_DIM, dtype=np.float32)

        # Final check
        if idx != STATIC_FEATURE_DIM:
            print(f"[CRITICAL ERROR] Final static feature index {idx} != STATIC_FEATURE_DIM {STATIC_FEATURE_DIM}!")
            return np.zeros(STATIC_FEATURE_DIM, dtype=np.float32) # Return zeros

        if np.isnan(features).any() or np.isinf(features).any():
            print(f"[Error] NaN/Inf detected in static features for P{player_id}! Replacing with 0.")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    def get_valid_discard_options(self, player_id: int) -> list[int]:
        """Returns a list of valid discard tile type indices (0-33) for the player."""
        if not (0 <= player_id < NUM_PLAYERS): return []
        hand = self.player_hands[player_id]
        is_reach = self.player_reach_status[player_id] == 2
        has_drawn_tile = len(hand) % 3 == 2

        if is_reach:
            if has_drawn_tile and hand:
                 drawn_tile_id = hand[-1]
                 drawn_tile_index = tile_id_to_index(drawn_tile_id)
                 return [drawn_tile_index] if drawn_tile_index != -1 else []
            else: return [] # Cannot discard if not holding 14 tiles in reach
        else:
             if not has_drawn_tile: return [] # Cannot discard before drawing
             options = set(tile_id_to_index(t) for t in hand if tile_id_to_index(t) != -1)
             return sorted(list(options))

# ==============================================================================
# == Debugging Framework Logic (from debug_gamestate.py) ==
# ==============================================================================

# --- Formatting Functions ---
def format_hand(hand_ids: list) -> str:
    """Formats a list of tile IDs into a readable hand string."""
    try:
        valid_ids = [t for t in hand_ids if isinstance(t, int) and 0 <= t <= 135]
        # Sort primarily by index, secondarily by ID
        return " ".join(sorted([tile_id_to_string(t) for t in valid_ids],
                                key=lambda x_str: (
                                    tile_id_to_index(int(x_str)) if x_str.replace('0','').isdigit() else 99, # Handle '0m' etc.
                                    int(x_str) if x_str.replace('0','').isdigit() else -1
                                )))
    except Exception as e: return str(hand_ids)

def format_discards(discard_list: list) -> str:
    """Formats a list of (tile_id, tsumogiri_flag) into a discard string."""
    try: return " ".join([f"{tile_id_to_string(t)}{'*' if f else ''}" for t, f in discard_list])
    except Exception: return str(discard_list)

def format_melds(meld_list: list) -> str:
    """Formats the GameState's meld list with readable tile strings."""
    meld_strings = []
    if not isinstance(meld_list, list): return str(meld_list)
    for meld_info in meld_list:
        try:
            if not isinstance(meld_info, dict):
                 meld_strings.append(str(meld_info)); continue
            m_type = meld_info.get('type', '不明')
            m_tiles_ids = meld_info.get('tiles', [])
            if not m_tiles_ids: tiles_str = ""
            else:
                 valid_ids = [t for t in m_tiles_ids if isinstance(t, int) and 0 <= t <= 135]
                 sorted_tile_ids = sorted(valid_ids, key=lambda t: (tile_id_to_index(t), t))
                 tiles_str = " ".join([tile_id_to_string(t) for t in sorted_tile_ids])
            meld_strings.append(f"{m_type}[{tiles_str}]")
        except Exception as e:
            print(f"[WARN] Error formatting meld item: {meld_info} - {e}")
            meld_strings.append(str(meld_info))
    return " | ".join(meld_strings)
# --- End Formatting Functions ---

def process_event(game_state: GameState, tag: str, attrib: dict, event_index: int, all_events: list, process_only: bool = False) -> tuple[str, bool, int]:
    """
    Processes a single event tag using the GameState object.
    Includes logic to print features after Tsumo.
    """
    description = ""
    processed = False
    event_player_id = -1 # Player ID associated with THIS action

    def call_gs_method(method_name, *args):
        nonlocal processed, description
        if process_only: processed = True; return
        if hasattr(game_state, method_name):
            try: getattr(game_state, method_name)(*args); processed = True
            except Exception as e:
                print(f"    [ERROR] GameState.{method_name}({args}) failed at event {event_index+1}: {e}"); traceback.print_exc(limit=1)
                description += f" [ERROR in {method_name}]"; processed = True
        else: print(f"    [WARN] GameState missing method: {method_name}"); description += f" [WARN: Method {method_name} missing]"; processed = True

    try:
        is_tsumo_event = False
        for t_tag, p_id in GameState.TSUMO_TAGS.items():
            if tag.startswith(t_tag) and tag[1:].isdigit():
                event_player_id = p_id; pai_id = int(tag[1:])
                description = f"P{event_player_id} ツモ {tile_id_to_string(pai_id)}"
                call_gs_method('process_tsumo', event_player_id, pai_id); is_tsumo_event = True; break
        if is_tsumo_event:
            if not process_only: # --- FEATURE EXTRACTION DEBUG ---
                print("    --- Feature Extraction Point ---")
                try:
                    seq_features = game_state.get_event_sequence_features()
                    static_features = game_state.get_static_features(event_player_id)
                    print(f"    Sequence Features Shape: {seq_features.shape}")
                    print(f"    Static Features Shape: {static_features.shape}")
                    if static_features.shape[0] != STATIC_FEATURE_DIM: print(f"    [WARN] Static feature dimension mismatch! Expected {STATIC_FEATURE_DIM}")
                    # Peek at next event for label
                    if event_index + 1 < len(all_events):
                        next_event = all_events[event_index + 1]; next_tag = next_event["tag"]
                        expected_label = -1
                        for d_tag, p_id in GameState.DISCARD_TAGS.items():
                            if next_tag.startswith(d_tag) and next_tag[1:].isdigit() and p_id == event_player_id:
                                try: discard_pai_id = int(next_tag[1:]); expected_label = tile_id_to_index(discard_pai_id)
                                except ValueError: pass; break
                        if expected_label != -1: print(f"    Expected Next Discard (Label Index): {expected_label} ({tile_id_to_string(tile_index_to_id(expected_label))})")
                        else: print(f"    Expected Next Action: Not a direct discard by P{event_player_id} (Tag: {next_tag})")
                except Exception as e: print(f"    [ERROR] Feature extraction/printing failed: {e}"); traceback.print_exc(limit=1)
            return description, True, event_player_id # Return after Tsumo

        is_discard_event = False
        for d_tag, p_id in GameState.DISCARD_TAGS.items():
            if tag.startswith(d_tag) and tag[1:].isdigit():
                event_player_id = p_id; pai_id = int(tag[1:]); tsumogiri = tag[0].islower()
                description = f"P{event_player_id} 打 {tile_id_to_string(pai_id)}{'*' if tsumogiri else ''}"
                call_gs_method('process_discard', event_player_id, pai_id, tsumogiri); is_discard_event = True; break
        if is_discard_event: return description, True, event_player_id

        if tag == "N":
            naki_player_id = int(attrib.get("who", -1)); meld_code = int(attrib.get("m", "0"))
            if naki_player_id != -1:
                event_player_id = naki_player_id
                naki_info = decode_naki(meld_code); naki_type_desc = naki_info.get("type", "不明")
                description = f"P{naki_player_id} 鳴き {naki_type_desc} (m={meld_code})"
                call_gs_method('process_naki', naki_player_id, meld_code)
            else: description = "[Skipped Naki: 'who' missing]"; processed = True
            return description, processed, event_player_id

        if tag == "REACH":
            reach_player_id = int(attrib.get("who", -1)); step = int(attrib.get("step", -1))
            if reach_player_id != -1 and step != -1:
                event_player_id = reach_player_id
                description = f"P{reach_player_id} リーチ (step {step})"
                if step == 1: call_gs_method('process_reach', reach_player_id, step)
                else: processed = True # Step 2 handled by discard
            else: description = "[Skipped Reach: Invalid attrs]"; processed = True
            return description, processed, event_player_id

        if tag == "DORA":
            hai_attr = attrib.get("hai");
            if hai_attr is not None and hai_attr.isdigit():
                hai = int(hai_attr); description = f"新ドラ表示: {tile_id_to_string(hai)}"
                call_gs_method('process_dora', hai)
            else: description = "[Skipped Dora: Invalid attrs]"; processed = True
            return description, processed, -1 # No specific player

        if tag == "AGARI":
            who = attrib.get('who', '?'); fromWho = attrib.get('fromWho', '?')
            event_player_id = int(who) if who.isdigit() else -1
            description = f"和了 P{who} (from P{fromWho})"
            call_gs_method('process_agari', attrib)
            return description, processed, event_player_id

        if tag == "RYUUKYOKU":
            ry_type = attrib.get('type', 'unknown'); description = f"流局 (Type: {ry_type})"
            call_gs_method('process_ryuukyoku', attrib)
            return description, processed, -1

    except (ValueError, KeyError, IndexError, TypeError) as e:
        print(f"    [ERROR] Parsing event <{tag} {attrib}> failed at event {event_index+1}: {e}"); traceback.print_exc(limit=1)
        processed = True; description = f"[ERROR parsing {tag}]"
    except Exception as e:
         print(f"    [FATAL ERROR] Unexpected exception during event processing for <{tag} {attrib}> at event {event_index+1}: {e}"); traceback.print_exc()
         processed = True; description = f"[FATAL ERROR processing {tag}]"

    if not processed: description = f"[Skipped Tag: {tag}]"
    return description, processed, event_player_id if processed else -1

def print_game_state_summary(game_state: GameState):
    """Prints a summary of the current GameState."""
    if not isinstance(game_state, GameState): print("[Error] Invalid GameState object."); return
    print(f"  Junme: {getattr(game_state, 'junme', '?'):.2f}")
    print(f"  Current Player: P{getattr(game_state, 'current_player', '?')}")
    lp = getattr(game_state, 'last_discard_event_player', -1); lt = getattr(game_state, 'last_discard_event_tile_id', -1)
    print(f"  Last Discard: P{lp} -> {tile_id_to_string(lt) if lt != -1 else 'None'}")
    print(f"  Dora Indicators: {[tile_id_to_string(t) for t in getattr(game_state, 'dora_indicators', [])]}")
    print(f"  Scores: {getattr(game_state, 'current_scores', ['?']*NUM_PLAYERS)}")
    print(f"  Kyotaku/Honba: {getattr(game_state, 'kyotaku', '?')} / {getattr(game_state, 'honba', '?')}")
    hands = getattr(game_state, 'player_hands', [[] for _ in range(NUM_PLAYERS)])
    discards = getattr(game_state, 'player_discards', [[] for _ in range(NUM_PLAYERS)])
    melds = getattr(game_state, 'player_melds', [[] for _ in range(NUM_PLAYERS)])
    reach = getattr(game_state, 'player_reach_status', [0]*NUM_PLAYERS)
    reach_j = getattr(game_state, 'player_reach_junme', [-1]*NUM_PLAYERS)
    for p in range(NUM_PLAYERS):
        r_info = ""
        if p < len(reach):
            if reach[p] == 1: r_info = "(リーチ宣言)"
            elif reach[p] == 2: r_jun = reach_j[p] if p < len(reach_j) else -1; r_info = f"* (リーチ@{r_jun:.2f}巡)" if r_jun!=-1 else "* (リーチ)"
        print(f"  --- Player {p} {r_info} ---")
        print(f"    Hand: {format_hand(hands[p] if p < len(hands) else [])} ({len(hands[p] if p < len(hands) else [])}枚)")
        print(f"    Discards: {format_discards(discards[p] if p < len(discards) else [])}")
        print(f"    Melds: {format_melds(melds[p] if p < len(melds) else [])}")

# ==============================================================================
# == Main Execution Logic (from debug_gamestate.py) ==
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Debug GameState and Feature Extraction by stepping through a Tenhou XML log round.")
    parser.add_argument("xml_file", help="Path to the Tenhou XML log file.")
    parser.add_argument("round_index", type=int, help="Round index to debug (1-based).")
    parser.add_argument("--start", type=int, default=1, metavar='EVENT_NUM',
                        help="Event index to start displaying state (1-based, default: 1).")
    parser.add_argument("--count", type=int, default=999, metavar='NUM_EVENTS',
                        help="Maximum number of events to display state for (default: all).")
    args = parser.parse_args()

    if not os.path.exists(args.xml_file): print(f"Error: XML file not found at {args.xml_file}"); sys.exit(1)
    if args.start < 1: print("Error: --start event index must be 1 or greater."); sys.exit(1)

    try:
        print(f"Parsing log file: {args.xml_file}...")
        # Use the integrated parser
        meta, rounds_data = parse_full_mahjong_log(args.xml_file)
        print(f"Found {len(rounds_data)} rounds.")

        if not (1 <= args.round_index <= len(rounds_data)):
            print(f"Error: Invalid round_index {args.round_index}. Must be between 1 and {len(rounds_data)}.")
            sys.exit(1)

        round_data = rounds_data[args.round_index - 1]
        events = round_data.get("events", [])
        num_events_in_round = len(events)
        print(f"Targeting Round {args.round_index} (Index {args.round_index - 1}) which has {num_events_in_round} events.")

        # Use the integrated GameState
        game_state = GameState()
        required_methods = ['init_round', 'process_tsumo', 'process_discard', 'process_naki', 'process_reach', 'process_dora', 'process_agari', 'process_ryuukyoku', 'get_event_sequence_features', 'get_static_features']
        missing_methods = [m for m in required_methods if not hasattr(game_state, m)]
        if missing_methods: print(f"[FATAL ERROR] GameState missing methods: {', '.join(missing_methods)}!"); sys.exit(1)

        print("Initializing game state for the round...")
        game_state.init_round(round_data)
        print("--- Initial State (After INIT) ---")
        print_game_state_summary(game_state); print("-" * 30)

        start_index_0based = max(0, args.start - 1)

        # Fast-forward
        if start_index_0based > 0:
            print(f"Fast-forwarding through the first {start_index_0based} events...")
            for i in range(start_index_0based):
                if i < num_events_in_round: process_event(game_state, events[i]["tag"], events[i]["attrib"], i, events, process_only=True)
                else: break
            print("Done fast-forwarding."); print("--- State Before Display Start ---")
            print_game_state_summary(game_state); print("-" * 30)
        else: print("Starting state display from the first event.")

        # Process and display target range
        display_count = 0
        end_index_0based = min(start_index_0based + args.count, num_events_in_round)
        print(f"--- Displaying States for Events {start_index_0based + 1} to {end_index_0based} ---")

        for i in range(start_index_0based, end_index_0based):
            event = events[i]; tag = event["tag"]; attrib = event["attrib"]
            print(f"\n>>> Processing Event {i+1}/{num_events_in_round}: <{tag}> {attrib}")
            event_description, processed_flag, event_player_id = process_event(game_state, tag, attrib, i, events, process_only=False)
            if event_description: print(f"    Action: {event_description}")
            print("--- State After Event ---")
            print_game_state_summary(game_state); print("-" * 30)
            display_count += 1
            if tag == "AGARI" or tag == "RYUUKYOKU": print("\n--- Round End Detected During Display ---"); break

        if i == end_index_0based - 1 and end_index_0based < num_events_in_round : print(f"\n--- Reached Event Count Limit ({args.count}) ---")
        elif i == num_events_in_round -1: print("\n--- Reached End of Events for the Round ---")

    except FileNotFoundError: print(f"Error: XML file not found at {args.xml_file}"); sys.exit(1)
    except ImportError: print(f"Error: Failed to import necessary modules."); sys.exit(1)
    except Exception as e: print(f"\n[FATAL ERROR] An unexpected error occurred:"); traceback.print_exc(); sys.exit(1)

if __name__ == "__main__":
    main()