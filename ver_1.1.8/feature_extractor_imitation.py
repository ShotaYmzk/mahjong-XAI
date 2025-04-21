# feature_extractor_imitation.py
import numpy as np
import sys
import traceback

# --- Import necessary components ---
try:
    # Use the completed GameState class from game_state.py
    from game_state import GameState, NUM_PLAYERS, MAX_EVENT_HISTORY, NUM_TILE_TYPES
    # Import utilities
    from tile_utils import tile_id_to_index, tile_id_to_string
    # Import the parser function (only needed if not called from model.py directly)
    # from full_mahjong_parser import parse_full_mahjong_log
except ImportError as e:
    print(f"[FATAL ERROR in feature_extractor_imitation.py] Failed to import modules: {e}")
    print("Ensure game_state.py and tile_utils.py are accessible.")
    sys.exit(1)
# ----------------------------------

def extract_features_labels_for_imitation(round_data: dict, game_state: GameState) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Extracts (sequence, static_features, label) triples for imitation learning from a single round's data.
    Uses the provided GameState object.

    Args:
        round_data: Dictionary containing data for one round (from parse_full_mahjong_log).
        game_state: An initialized GameState object to process the round.

    Returns:
        Tuple containing NumPy arrays for sequences, static features, and labels for the round,
        or (None, None, None) if no valid data points are generated.
    """
    sequences_list = []
    static_features_list = []
    labels_list = []
    last_decision_point = None # Store state before discard decision

    try:
        # Initialize GameState for the round (caller should handle game_state creation)
        # game_state.init_round(round_data) # init_round should be called *outside* this function per file

        events = round_data.get("events", [])
        if not events:
             return None, None, None # Skip if no events

        for event in events:
            tag = event["tag"]
            attrib = event["attrib"]
            processed = False # Flag if event was handled by a specific block

            # --- Tsumo Event ---
            # Check if the tag represents a tsumo action
            tsumo_player_id = -1
            tsumo_pai_id = -1
            for t_tag, p_id in GameState.TSUMO_TAGS.items():
                if tag.startswith(t_tag) and tag[1:].isdigit():
                    try:
                        tsumo_pai_id = int(tag[1:])
                        tsumo_player_id = p_id
                        processed = True
                        break
                    except ValueError: continue

            if processed:
                # --- Decision Point: Player has drawn a tile ---
                try:
                    # 1. Get current event sequence features BEFORE processing the tsumo
                    current_sequence = game_state.get_event_sequence_features()

                    # 2. Process the tsumo to update the hand state
                    game_state.process_tsumo(tsumo_player_id, tsumo_pai_id)

                    # 3. Get static features AFTER processing tsumo (includes the drawn tile)
                    current_static = game_state.get_static_features(tsumo_player_id)

                    # Store the state and sequence for the upcoming discard decision
                    last_decision_point = {
                        "sequence": current_sequence,
                        "static": current_static,
                        "player": tsumo_player_id
                    }
                except Exception as e:
                    print(f"[Error] Feature extraction failed at Tsumo (P{tsumo_player_id}, Tile:{tsumo_pai_id}): {e}")
                    # traceback.print_exc() # Uncomment for full traceback
                    last_decision_point = None # Clear state if error occurs
                continue # Move to the next event (expecting a discard)

            # --- Discard Event ---
            discard_player_id = -1
            discard_pai_id = -1
            tsumogiri = False
            for d_tag, p_id in GameState.DISCARD_TAGS.items():
                if tag.startswith(d_tag) and tag[1:].isdigit():
                    try:
                        discard_pai_id = int(tag[1:])
                        discard_player_id = p_id
                        tsumogiri = tag[0].islower()
                        processed = True
                        break
                    except ValueError: continue

            if processed:
                # --- Action Taken: Player discarded a tile ---
                # Check if this discard corresponds to the last decision point
                if last_decision_point and last_decision_point["player"] == discard_player_id:
                    label = tile_id_to_index(discard_pai_id)
                    if label != -1 and last_decision_point["sequence"] is not None and last_decision_point["static"] is not None:
                        sequences_list.append(last_decision_point["sequence"])
                        static_features_list.append(last_decision_point["static"])
                        labels_list.append(label)
                    # else: print(f"[Warning] Invalid label or missing features for discard P{discard_player_id}, Tile:{discard_pai_id}") # Optional warning
                    last_decision_point = None # Clear after use
                # else: Discard occurred without matching Tsumo state (e.g., after Naki) - Ignore for simple imitation learning

                # Process the discard to update the game state for subsequent events
                try:
                    game_state.process_discard(discard_player_id, discard_pai_id, tsumogiri)
                except Exception as e:
                    print(f"[Error] GameState update failed at Discard (P{discard_player_id}, Tile:{discard_pai_id}): {e}")
                    # traceback.print_exc()
                continue

            # --- Naki Event ---
            if not processed and tag == "N":
                last_decision_point = None # Naki invalidates previous Tsumo decision state
                try:
                    naki_player_id = int(attrib.get("who", -1))
                    meld_code = int(attrib.get("m", "0"))
                    if naki_player_id != -1:
                        game_state.process_naki(naki_player_id, meld_code)
                        processed = True
                except (ValueError, KeyError, AttributeError, Exception) as e: # Catch more potential errors
                    print(f"[Error] GameState update failed at Naki (Attrib:{attrib}): {e}")
                    # traceback.print_exc()
                continue

            # --- Reach Event ---
            if not processed and tag == "REACH":
                last_decision_point = None # Reach declaration might change state/strategy
                try:
                    reach_player_id = int(attrib.get("who", -1))
                    step = int(attrib.get("step", -1))
                    # Process only step 1 declaration; step 2 is handled by discard
                    if reach_player_id != -1 and step == 1:
                        game_state.process_reach(reach_player_id, step)
                        processed = True
                    elif step == 2: # Mark as processed but don't call method
                         processed = True
                except (ValueError, KeyError, AttributeError, Exception) as e:
                    print(f"[Error] GameState update failed at Reach (Attrib:{attrib}): {e}")
                    # traceback.print_exc()
                continue

            # --- Dora Event ---
            if not processed and tag == "DORA":
                try:
                    hai_attr = attrib.get("hai")
                    if hai_attr is not None and hai_attr.isdigit():
                        dora_indicator_id = int(hai_attr)
                        game_state.process_dora(dora_indicator_id)
                        processed = True
                except (ValueError, KeyError, AttributeError, Exception) as e:
                    print(f"[Error] GameState update failed at Dora (Attrib:{attrib}): {e}")
                    # traceback.print_exc()
                continue

            # --- Round End Events ---
            if not processed and (tag == "AGARI" or tag == "RYUUKYOKU"):
                last_decision_point = None # End of round, clear any pending state
                try:
                    if tag == "AGARI": game_state.process_agari(attrib)
                    else: game_state.process_ryuukyoku(attrib)
                    processed = True
                except Exception as e:
                     print(f"[Error] GameState update failed at {tag} (Attrib:{attrib}): {e}")
                break # Stop processing events for this round

        # End of event loop for the round

    except Exception as e:
         round_idx = round_data.get('round_index', 'Unknown')
         print(f"[Error] Unhandled exception during round {round_idx} processing: {e}")
         traceback.print_exc()
         # Return None to indicate failure for this round
         return None, None, None

    # Convert collected lists to NumPy arrays if data exists
    if sequences_list:
        try:
            sequences_np = np.array(sequences_list, dtype=np.float32)
            static_features_np = np.array(static_features_list, dtype=np.float32)
            labels_np = np.array(labels_list, dtype=np.int64)
            return sequences_np, static_features_np, labels_np
        except ValueError as e:
             print(f"[Error] Failed to convert extracted data to NumPy arrays for round {round_data.get('round_index', 'Unknown')}: {e}")
             # Attempt to find inconsistent shapes
             seq_shapes = {s.shape for s in sequences_list}
             sta_shapes = {s.shape for s in static_features_list}
             print(f"  Unique Sequence Shapes: {seq_shapes}")
             print(f"  Unique Static Shapes: {sta_shapes}")
             return None, None, None # Return None on conversion error
    else:
        return None, None, None # No data points generated for this round