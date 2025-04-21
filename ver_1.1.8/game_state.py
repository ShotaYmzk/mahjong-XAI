# game_state.py (Complete Integrated Version)
import numpy as np
from collections import defaultdict, deque
import sys
import traceback # For detailed error reporting in naki

# --- Dependency Imports ---
try:
    from tile_utils import tile_id_to_index, tile_id_to_string
except ImportError as e:
    print(f"[FATAL ERROR in game_state.py] Cannot import from tile_utils.py: {e}")
    print("Ensure tile_utils.py is in the same directory or Python path.")
    sys.exit(1)
try:
    from naki_utils import decode_naki
except ImportError as e:
    print(f"[FATAL ERROR in game_state.py] Cannot import from naki_utils.py: {e}")
    print("Ensure naki_utils.py is in the same directory or Python path.")
    sys.exit(1)
# --- End Dependency Imports ---

# --- Custom Shanten Calculation ---
try:
    from custom_shanten import calculate_shanten_and_ukeire
    CUSTOM_SHANTEN_AVAILABLE = True
except ImportError:
    print("[Warning in game_state.py] custom_shanten.py not found. Using dummy shanten calculation.")
    CUSTOM_SHANTEN_AVAILABLE = False
    # Dummy function if custom_shanten not found
    def calculate_shanten_and_ukeire(hand_tile_ids, melds_data=None):
        """Dummy shanten calculation"""
        # Return a reasonable default (8 = worst case, empty hand)
        num_tiles = len(hand_tile_ids) + sum(len(m.get("tiles", [])) for m in (melds_data or []))
        shanten = 8
        if num_tiles == 14 or num_tiles == 13: 
            shanten = 0  # Crude guess for tenpai
        return shanten, []  # No ukeire information

def calculate_shanten(hand_tile_ids: list[int], melds_data: list[dict]) -> tuple[int, list[int]]:
    """
    Calculates shanten and ukeire using our custom implementation.
    """
    if not CUSTOM_SHANTEN_AVAILABLE:
        # Use our dummy fallback
        shanten, ukeire = calculate_shanten_and_ukeire([], [])
        return shanten, ukeire

    try:
        # Call our custom implementation
        return calculate_shanten_and_ukeire(hand_tile_ids, melds_data)
    except Exception as e:
        print(f"[Error calculate_shanten] Calculation failed: {e}")
        return 8, []  # Return worst case on error
# --- End Custom Shanten Calculation ---

# --- Constants ---
NUM_PLAYERS = 4
NUM_TILE_TYPES = 34
MAX_EVENT_HISTORY = 60 # Sequence length for Transformer

# Event types for encoding
EVENT_TYPES = {
    "DRAW": 0,
    "DISCARD": 1,
    "CALL": 2,
    "RIICHI": 3,
    "DORA_REVEALED": 4,
    "WIN": 5,
    "DRAW_GAME": 6,
    "NEW_ROUND": 7,
    "PADDING": 8
}

class GameState:
    """Manages the state of a Mahjong game round, including event history and feature generation."""
    TSUMO_TAGS = {"T": 0, "U": 1, "V": 2, "W": 3}
    DISCARD_TAGS = {"D": 0, "E": 1, "F": 2, "G": 3}

    NAKI_TYPES = {"チー": 0, "ポン": 1, "大明槓": 2, "加槓": 3, "暗槓": 4, "不明": -1}

    def __init__(self):
        """Initializes the GameState."""
        self.reset_state()

    def reset_state(self):
        """Resets all internal state variables."""
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
        self.player_melds: list[list[dict]] = [[] for _ in range(NUM_PLAYERS)] # Using dict format
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
        """Adds a structured event to the event history."""
        if data is None: data = {}
        event_code = self.EVENT_TYPES.get(event_type, -1)
        if event_code == -1: return

        event_info = {
            "type": event_code, "player": player,
            "tile_index": tile_id_to_index(tile),
            "junme": int(np.ceil(self.junme)), "data": data
        }
        self.event_history.append(event_info)

    def init_round(self, round_data: dict):
        """Initializes state for a new round."""
        self.reset_state()
        init_info = round_data.get("init", {})
        if not init_info: print("[Warning] No init info found."); return

        self.round_index = round_data.get("round_index", 0)
        seed_parts = init_info.get("seed", "0,0,0,0,0,0").split(",")
        try:
            self.round_num_wind = int(seed_parts[0]); self.honba = int(seed_parts[1]); self.kyotaku = int(seed_parts[2])
            dora_indicator_id = int(seed_parts[5]); self.dora_indicators = [dora_indicator_id]
            self._add_event("DORA", player=-1, tile=dora_indicator_id)
        except (IndexError, ValueError, TypeError) as e: print(f"[Warning] Failed parse seed: {e}"); self.round_num_wind=0; self.honba=0; self.kyotaku=0; self.dora_indicators=[]

        self.dealer = int(init_info.get("oya", -1)); self.current_player = self.dealer

        try:
            self.initial_scores = [int(s) for s in init_info.get("ten", "25000,25000,25000,25000").split(",")]
            self.current_scores = list(self.initial_scores)
        except (ValueError, TypeError) as e: print(f"[Warning] Failed parse ten: {e}"); self.initial_scores=[25000]*NUM_PLAYERS; self.current_scores=[25000]*NUM_PLAYERS

        for p in range(NUM_PLAYERS):
            hand_str = init_info.get(f"hai{p}", ""); self.player_hands[p] = []
            try:
                if hand_str:
                    hand_ids = list(map(int, hand_str.split(",")))
                    valid_hand_ids = [tid for tid in hand_ids if 0 <= tid <= 135]
                    self.player_hands[p] = sorted(valid_hand_ids)
            except (ValueError, TypeError) as e: print(f"[Warning] Failed parse hai{p}: {e}")

        self.wall_tile_count = 136 - 14 - (13 * NUM_PLAYERS)
        init_data = {"round": self.round_num_wind, "honba": self.honba, "kyotaku": self.kyotaku}
        self._add_event("INIT", player=self.dealer, data=init_data)

    def _sort_hand(self, player_id):
        """Sorts player's hand."""
        if 0 <= player_id < NUM_PLAYERS:
            self.player_hands[player_id].sort(key=lambda t: (tile_id_to_index(t), t))

    def process_tsumo(self, player_id: int, tile_id: int):
        """Processes Tsumo."""
        # Update game state
        self.current_player = player_id
        if player_id == self.dealer and self.junme == 0:
            self.junme = 0.1 # First turn of dealer
        elif player_id == 0 and self.junme < 1.0:
            self.junme = 1.0 # First round completed
        elif player_id == 0 and self.junme >= 1.0:
            self.junme += 1.0 # Increment on each new round
        
        # Update wall count and check rinshan flag
        if not self.is_rinshan:
            self.wall_tile_count -= 1
        self.is_rinshan = False  # Reset rinshan flag
        
        # Add tile to player's hand and sort
        if 0 <= player_id < NUM_PLAYERS:
            self.player_hands[player_id].append(tile_id)
            self._sort_hand(player_id)
            
        # Add to event history
        self._add_event("TSUMO", player=player_id, tile=tile_id)
        
        # Reset discard event info and turn state
        self.naki_occurred_in_turn = False
        self.can_ron = False

    def process_discard(self, player_id: int, tile_id: int, tsumogiri: bool):
        """Processes Discard."""
        # Update game state
        self.last_discard_event_player = player_id
        self.last_discard_event_tile_id = tile_id
        self.last_discard_event_tsumogiri = tsumogiri
        self.can_ron = True
        
        # Remove from player's hand
        if 0 <= player_id < NUM_PLAYERS:
            if tile_id in self.player_hands[player_id]:
                self.player_hands[player_id].remove(tile_id)
            else:
                print(f"[Warning process_discard] Tile {tile_id} not in player {player_id}'s hand: {self.player_hands[player_id]}")
                
            # Add to discard pile
            self.player_discards[player_id].append((tile_id, tsumogiri))
            
        # Add to event history
        self._add_event("DISCARD", player=player_id, tile=tile_id, data={"tsumogiri": int(tsumogiri)})
        
        # Update riichi status for step 2 (actual riichi discard)
        if self.player_reach_status[player_id] == 1:  # Step 1 was previously processed
            self.player_reach_status[player_id] = 2  # Confirm riichi with discard
            self.player_reach_discard_index[player_id] = len(self.player_discards[player_id]) - 1

    def process_naki(self, naki_player_id: int, meld_code: int):
        """Processes Naki (call) event."""
        import traceback
        
        # Flag that a naki occurred this turn
        self.naki_occurred_in_turn = True
        self.can_ron = False  # Can't ron after a call on this discard
        
        # Try to decode the meld information
        meld_info = {}
        try:
            meld_info = decode_naki(meld_code, self.last_discard_event_tile_id, self.last_discard_event_player, naki_player_id)
        except Exception as e:
            print(f"[Error process_naki] Failed to decode naki: {e}")
            traceback.print_exc()
            return
        
        meld_type = meld_info.get("type", "不明")
        meld_tiles = meld_info.get("tiles", [])
        
        # Update player's hand and melds
        if 0 <= naki_player_id < NUM_PLAYERS:
            # Handle the called tile (last discard) separately
            called_tile_id = self.last_discard_event_tile_id
            
            # For each tile in the meld that is not the called tile, try to remove it from hand
            for tile_id in meld_tiles:
                # Don't try to remove the called tile from hand (it comes from discard)
                if tile_id == called_tile_id:
                    continue
                    
                # Try to remove the tile from hand if it exists
                if tile_id in self.player_hands[naki_player_id]:
                    self.player_hands[naki_player_id].remove(tile_id)
                else:
                    # This warning is expected in some cases:
                    # 1. For kans, all 4 tiles are included but only 3 might be in hand
                    # 2. For logs with incomplete information
                    # 3. When red five replacements occur
                    if meld_type == "暗槓" or meld_type == "加槓":
                        # For kans, missing tiles are expected
                        pass
                    else:
                        print(f"[Warning process_naki] Meld tile {tile_id} not found in player {naki_player_id}'s hand - this may be normal for some XML logs")
            
            # Add the meld to player's melds
            self.player_melds[naki_player_id].append(meld_info)
            
            # Check if kan (4-tile set)
            is_kan = meld_type in ["大明槓", "加槓", "暗槓"]
            
            # For kans, take a tile from the dead wall
            if is_kan:
                self.is_rinshan = True
                # Kandora is handled in process_dora
                
            # Transfer current turn to the naki player
            self.current_player = naki_player_id
            
        # Add to event history
        self._add_event("N", player=naki_player_id, data={"meld_type": self.NAKI_TYPES.get(meld_type, -1), "from_player": self.last_discard_event_player})

    def process_reach(self, player_id: int, step: int):
        """Processes Riichi declaration."""
        if 0 <= player_id < NUM_PLAYERS and step == 1:
            self.player_reach_status[player_id] = 1  # Step 1 of riichi
            self.player_reach_junme[player_id] = self.junme
            # Step 2 is processed during discard
            
            # Add to event history
            self._add_event("REACH", player=player_id, data={"step": step, "junme": int(self.junme)})

    def process_dora(self, tile_id: int):
        """Processes new Dora indicator revealed."""
        self.dora_indicators.append(tile_id)
        
        # Add to event history
        self._add_event("DORA", player=-1, tile=tile_id)

    def process_agari(self, attrib: dict):
        """Processes Agari (win)."""
        # Round is ending with a win
        who = int(attrib.get("who", -1))
        from_who = int(attrib.get("fromWho", -1))
        
        # Add to event history
        self._add_event("AGARI", player=who, data={"from_player": from_who})

    def process_ryuukyoku(self, attrib: dict):
        """Processes Ryuukyoku (draw)."""
        # Round is ending without a win
        reason = attrib.get("type", "")
        
        # Add to event history
        self._add_event("RYUUKYOKU", player=-1, data={"reason": reason})

    def get_current_dora_indices(self) -> list[int]:
        """
        Returns a list of all dora tile indices based on indicators.
        For simplicity, returns the indicators themselves as dora.
        """
        dora_indices = []
        for indicator_id in self.dora_indicators:
            dora_idx = tile_id_to_index(indicator_id)
            if dora_idx != -1:
                dora_indices.append(dora_idx)
        return dora_indices

    def get_hand_indices(self, player_id: int) -> list[int]:
        """Get the tile indices for a player's hand."""
        if 0 <= player_id < NUM_PLAYERS:
            return [tile_id_to_index(tile_id) for tile_id in self.player_hands[player_id]]
        return []

    def get_melds_indices(self, player_id: int) -> list[dict]:
        """Get a player's melds with tile indices."""
        if 0 <= player_id < NUM_PLAYERS:
            return self.player_melds[player_id]
        return []

    def get_event_sequence_features(self) -> np.ndarray:
        """
        Generate feature vectors for event sequence.
        Each event is converted to a fixed-dimension vector.
        """
        # Define the event feature dimension (adapt if needed)
        event_feature_dim = 32
        
        # Initialize with padding events
        padding_code = float(self.EVENT_TYPES["PADDING"])
        features = np.zeros((MAX_EVENT_HISTORY, event_feature_dim), dtype=np.float32)
        features[:, 0] = padding_code  # Mark all as padding initially
        
        # Fill in actual events
        for i, event in enumerate(self.event_history):
            if i >= MAX_EVENT_HISTORY: break  # Don't exceed max length
            
            # Base features
            features[i, 0] = float(event["type"])  # Event type code
            features[i, 1] = float(event["player"] if event["player"] >= 0 else -1)  # Player ID
            features[i, 2] = float(event["tile_index"] if event["tile_index"] >= 0 else -1)  # Tile index
            features[i, 3] = float(event["junme"])  # Turn number
            
            # Additional features (based on event type)
            data = event["data"]
            if event["type"] == self.EVENT_TYPES["INIT"]:
                features[i, 4] = float(data.get("round", 0))
                features[i, 5] = float(data.get("honba", 0))
            elif event["type"] == self.EVENT_TYPES["DISCARD"]:
                features[i, 4] = float(data.get("tsumogiri", 0))
            elif event["type"] == self.EVENT_TYPES["N"]:
                features[i, 4] = float(data.get("meld_type", -1))
                features[i, 5] = float(data.get("from_player", -1))
            elif event["type"] == self.EVENT_TYPES["REACH"]:
                features[i, 4] = float(data.get("step", 0))
                features[i, 5] = float(data.get("junme", 0))
            elif event["type"] == self.EVENT_TYPES["AGARI"]:
                features[i, 4] = float(data.get("from_player", -1))
            
        return features

    def get_static_features(self, player_id: int) -> np.ndarray:
        """
        Generate static features for the current game state, from player_id's perspective.
        This includes hand tiles, dora, discards, etc.
        """
        # Define feature dimensions
        static_feature_dim = 192
        features = np.zeros(static_feature_dim, dtype=np.float32)
        
        if not (0 <= player_id < NUM_PLAYERS):
            return features
        
        idx = 0  # Current index in the feature vector
        
        # 1. Game context (round, honba, etc.) - 8 features
        features[idx] = self.round_num_wind; idx += 1
        features[idx] = self.honba; idx += 1
        features[idx] = self.kyotaku; idx += 1
        features[idx] = self.dealer; idx += 1
        features[idx] = self.wall_tile_count; idx += 1
        features[idx] = (player_id == self.dealer); idx += 1  # Is dealer flag
        features[idx] = self.junme; idx += 1
        features[idx] = len(self.dora_indicators); idx += 1
        
        # 2. Player specific data - 5 features
        features[idx] = self.player_reach_status[player_id]; idx += 1
        features[idx] = self.player_reach_junme[player_id]; idx += 1
        features[idx] = len(self.player_discards[player_id]); idx += 1
        features[idx] = len(self.player_melds[player_id]); idx += 1
        features[idx] = len(self.player_hands[player_id]); idx += 1
        
        # 3. Hand tiles one-hot encoding - 34 features
        try:
            hand_indices = self.get_hand_indices(player_id)
            for i in range(NUM_TILE_TYPES):
                features[idx + i] = hand_indices.count(i)
        except Exception as e:
            print(f"[Warning get_static_features] Error processing hand indices: {e}")
        idx += NUM_TILE_TYPES
        
        # 4. Dora indicators - 34 features
        try:
            dora_indices = self.get_current_dora_indices()
            for i in range(NUM_TILE_TYPES):
                features[idx + i] = dora_indices.count(i)
        except Exception as e:
            print(f"[Warning get_static_features] Error processing dora indices: {e}")
        idx += NUM_TILE_TYPES
        
        # 5. Player discards - 34 features
        try:
            discard_indices = [tile_id_to_index(tile_id) for tile_id, _ in self.player_discards[player_id]]
            for i in range(NUM_TILE_TYPES):
                features[idx + i] = discard_indices.count(i)
        except Exception as e:
            print(f"[Warning get_static_features] Error processing discard indices: {e}")
        idx += NUM_TILE_TYPES
        
        # 6. All visible discards and melds (excluding hand) - 34 features
        try:
            # Combine all discards and melds to see what's visible
            visible_indices = []
            for p in range(NUM_PLAYERS):
                visible_indices.extend([tile_id_to_index(tile_id) for tile_id, _ in self.player_discards[p]])
                for meld in self.player_melds[p]:
                    # Safely extract tiles from meld
                    meld_tiles = meld.get("tiles", [])
                    for tile_id in meld_tiles:
                        tile_idx = tile_id_to_index(tile_id)
                        if 0 <= tile_idx < NUM_TILE_TYPES:  # Ensure valid index
                            visible_indices.append(tile_idx)
            
            for i in range(NUM_TILE_TYPES):
                features[idx + i] = visible_indices.count(i)
        except Exception as e:
            print(f"[Warning get_static_features] Error processing visible indices: {e}")
        idx += NUM_TILE_TYPES
        
        # 7. Current waiting tiles (ukeire) based on shanten - 34 features
        try:
            shanten, ukeire = calculate_shanten(self.player_hands[player_id], self.player_melds[player_id])
            for i in range(NUM_TILE_TYPES):
                features[idx + i] = 1.0 if i in ukeire else 0.0
        except Exception as e:
            print(f"[Warning get_static_features] Error calculating shanten: {e}")
        idx += NUM_TILE_TYPES
        
        # 8. Player positions relative to current player - 8 features
        try:
            for p in range(NUM_PLAYERS):
                # Is this player position
                features[idx] = 1.0 if p == player_id else 0.0
                idx += 1
                # Is in riichi
                features[idx] = 1.0 if self.player_reach_status[p] >= 2 else 0.0
                idx += 1
        except Exception as e:
            print(f"[Warning get_static_features] Error setting player position features: {e}")
        
        return features

    def get_valid_discard_options(self, player_id: int) -> list[int]:
        """Return the tile indices that player can discard."""
        if 0 <= player_id < NUM_PLAYERS:
            return [tile_id_to_index(tile_id) for tile_id in self.player_hands[player_id]]
        return []