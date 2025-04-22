# debug_gamestate.py (Modified for Feature Debugging)
import sys
import argparse
import os
import traceback # For detailed error reporting
import numpy as np # For feature inspection

# --- Path setup ---
# Ensure other modules can be imported (adjust if your structure differs)
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path: sys.path.insert(0, script_dir)
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path: sys.path.insert(0, parent_dir)
# --- End Path setup ---

try:
    from full_mahjong_parser import parse_full_mahjong_log
    # Ensure game_state.py is accessible and defines GameState, constants
    from game_state import GameState, NUM_PLAYERS, STATIC_FEATURE_DIM, MAX_EVENT_HISTORY, EVENT_TYPES
    from tile_utils import tile_id_to_string, tile_id_to_index
    from naki_utils import decode_naki
except ImportError as e:
    print(f"[FATAL ERROR] Failed to import required modules: {e}")
    print("Please ensure full_mahjong_parser.py, game_state.py, tile_utils.py, and naki_utils.py are accessible.")
    sys.exit(1)

# --- Formatting Functions (Copied from original debug_gamestate.py) ---
def format_hand(hand_ids: list) -> str:
    """Formats a list of tile IDs into a readable hand string."""
    try:
        # Sort by index first, then ID for standard mahjong order
        valid_ids = [t for t in hand_ids if isinstance(t, int) and 0 <= t <= 135]
        return " ".join(sorted([tile_id_to_string(t) for t in valid_ids], key=lambda x_str: (tile_id_to_index(int(x_str.split('_')[-1])) if '_' in x_str else tile_id_to_index(int(x_str)) if x_str.isdigit() else 99, int(x_str.split('_')[-1]) if '_' in x_str else int(x_str) if x_str.isdigit() else -1)))
    except Exception as e: # Fallback for unexpected hand data
        # print(f"Debug format_hand error: {e}, Hand: {hand_ids}")
        return str(hand_ids)

def format_discards(discard_list: list) -> str:
    """Formats a list of (tile_id, tsumogiri_flag) into a discard string."""
    try:
        return " ".join([f"{tile_id_to_string(t)}{'*' if f else ''}" for t, f in discard_list])
    except Exception: # Fallback for unexpected discard data
        return str(discard_list)

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
    Returns a description, processed flag, and the player_id if applicable.
    Handles potential errors during GameState method calls.
    Now includes logic to print features after Tsumo.
    """
    description = ""
    processed = False
    player_id = -1 # Initialize player_id for the current event
    event_player_id = -1 # Player ID specifically associated with THIS action (Tsumo/Discard)

    # --- Helper to safely call GameState methods ---
    def call_gs_method(method_name, *args):
        nonlocal processed, description
        if process_only:
            processed = True # Assume it would be processed
            return
        if hasattr(game_state, method_name):
            try:
                getattr(game_state, method_name)(*args)
                processed = True
            except Exception as e:
                print(f"    [ERROR] GameState.{method_name}({args}) failed at event {event_index+1}: {e}")
                traceback.print_exc(limit=1)
                description += f" [ERROR in {method_name}]"
                processed = True # Let it continue if possible
        else:
            print(f"    [WARN] GameState is missing method: {method_name}")
            description += f" [WARN: Method {method_name} missing]"
            processed = True # Mark as processed but method was missing
    # -------------------------------------------------

    try:
        # Tsumo Event
        is_tsumo_event = False
        for t_tag, p_id in GameState.TSUMO_TAGS.items():
            if tag.startswith(t_tag) and tag[1:].isdigit():
                event_player_id = p_id
                pai_id = int(tag[1:])
                description = f"P{event_player_id} ツモ {tile_id_to_string(pai_id)}"
                call_gs_method('process_tsumo', event_player_id, pai_id)
                is_tsumo_event = True
                break
        if is_tsumo_event:
            # --- FEATURE EXTRACTION DEBUG ---
            if not process_only:
                print("    --- Feature Extraction Point ---")
                try:
                    # 1. Get Sequence Features (Represents history UP TO this draw)
                    seq_features = game_state.get_event_sequence_features()
                    print(f"    Sequence Features Shape: {seq_features.shape}")
                    # Print last 5 non-padding events
                    padding_code = float(EVENT_TYPES["PADDING"])
                    last_events_idx = np.where(seq_features[:, 0] != padding_code)[0]
                    last_events = seq_features[last_events_idx][-5:] if len(last_events_idx) > 0 else []
                    print(f"    Last {len(last_events)} Sequence Events (Type, Player+1, TileIdx+1, Junme, ...):")
                    for ev in last_events: print(f"      {ev.round(1)}")

                    # 2. Get Static Features (Represents state AFTER this draw)
                    static_features = game_state.get_static_features(event_player_id)
                    print(f"    Static Features Shape: {static_features.shape}")
                    if static_features.shape[0] == STATIC_FEATURE_DIM:
                         # Print some key slices (adjust indices based on final feature definition)
                         hand_slice_end = 34 * 5 # Example: Hand representation size
                         dora_ind_slice_end = hand_slice_end + 34
                         dora_tile_slice_end = dora_ind_slice_end + 34
                         print(f"    Static Hand Rep (Non-zero): {np.where(static_features[:hand_slice_end] != 0)[0]}")
                         print(f"    Static Dora Indicators: {np.where(static_features[hand_slice_end:dora_ind_slice_end] != 0)[0]}")
                         print(f"    Static Dora Tiles: {np.where(static_features[dora_ind_slice_end:dora_tile_slice_end] != 0)[0]}")
                         # Add more slices as needed
                    else:
                         print(f"    [WARN] Static feature dimension mismatch! Expected {STATIC_FEATURE_DIM}, Got {static_features.shape[0]}")

                    # 3. (Optional) Peek at next event for expected label
                    if event_index + 1 < len(all_events):
                        next_event = all_events[event_index + 1]
                        next_tag = next_event["tag"]
                        next_attrib = next_event["attrib"]
                        expected_label = -1
                        for d_tag, p_id in GameState.DISCARD_TAGS.items():
                            if next_tag.startswith(d_tag) and next_tag[1:].isdigit() and p_id == event_player_id:
                                try:
                                    discard_pai_id = int(next_tag[1:])
                                    expected_label = tile_id_to_index(discard_pai_id)
                                    print(f"    Expected Next Discard (Label): P{p_id} discards {tile_id_to_string(discard_pai_id)} (Index: {expected_label})")
                                except ValueError: pass
                                break
                        if expected_label == -1:
                             print(f"    Expected Next Action: Not a direct discard by P{event_player_id} (Tag: {next_tag})")
                    else:
                         print("    Expected Next Action: End of round events.")

                except Exception as e:
                    print(f"    [ERROR] Feature extraction/printing failed: {e}")
                    traceback.print_exc(limit=1)
            # --- END FEATURE EXTRACTION DEBUG ---
            return description, True, event_player_id # Return after Tsumo

        # Discard Event
        for d_tag, p_id in GameState.DISCARD_TAGS.items():
            if tag.startswith(d_tag) and tag[1:].isdigit():
                event_player_id = p_id
                pai_id = int(tag[1:])
                tsumogiri = tag[0].islower()
                description = f"P{event_player_id} 打 {tile_id_to_string(pai_id)}{'*' if tsumogiri else ''}"
                call_gs_method('process_discard', event_player_id, pai_id, tsumogiri)
                break
        if processed and event_player_id != -1: return description, True, event_player_id # Return if discard

        # Naki Event
        if tag == "N":
            naki_player_id = int(attrib.get("who", -1))
            meld_code = int(attrib.get("m", "0"))
            if naki_player_id != -1:
                event_player_id = naki_player_id # The player performing the naki
                # Use decode_naki for description hint (type only)
                naki_info = decode_naki(meld_code) # Use latest naki_utils version
                naki_type_desc = naki_info.get("type", "不明")
                description = f"P{naki_player_id} 鳴き {naki_type_desc} (m={meld_code})"
                call_gs_method('process_naki', naki_player_id, meld_code)
            else:
                description = "[Skipped Naki: 'who' missing]"
                processed = True # Mark as processed/skipped
        if processed and tag == "N": return description, True, event_player_id

        # Reach Event
        if tag == "REACH":
            reach_player_id = int(attrib.get("who", -1))
            step = int(attrib.get("step", -1))
            if reach_player_id != -1 and step != -1:
                event_player_id = reach_player_id
                description = f"P{reach_player_id} リーチ (step {step})"
                if step == 1: # Only process step 1 explicitly
                    call_gs_method('process_reach', reach_player_id, step)
                else: # Step 2 is handled by discard, just mark as processed
                    processed = True
            else:
                description = "[Skipped Reach: Invalid attrs]"
                processed = True
        if processed and tag == "REACH": return description, True, event_player_id

        # Dora Event
        if tag == "DORA":
            hai_attr = attrib.get("hai")
            if hai_attr is not None and hai_attr.isdigit():
                hai = int(hai_attr)
                description = f"新ドラ表示: {tile_id_to_string(hai)}"
                call_gs_method('process_dora', hai)
            else:
                description = "[Skipped Dora: Invalid attrs]"
                processed = True
        if processed and tag == "DORA": return description, True, -1 # No specific player

        # Agari Event
        if tag == "AGARI":
            who = attrib.get('who', '?'); fromWho = attrib.get('fromWho', '?')
            event_player_id = int(who) if who.isdigit() else -1
            description = f"和了 P{who} (from P{fromWho})"
            call_gs_method('process_agari', attrib)
        if processed and tag == "AGARI": return description, True, event_player_id

        # Ryuukyoku Event
        if tag == "RYUUKYOKU":
            ry_type = attrib.get('type', 'unknown')
            description = f"流局 (Type: {ry_type})"
            call_gs_method('process_ryuukyoku', attrib)
        if processed and tag == "RYUUKYOKU": return description, True, -1 # No specific player

    except (ValueError, KeyError, IndexError, TypeError) as e:
        print(f"    [ERROR] Parsing event <{tag} {attrib}> failed at event {event_index+1}: {e}")
        traceback.print_exc(limit=1)
        processed = True
        description = f"[ERROR parsing {tag}]"
    except Exception as e: # Catch any other unexpected errors
         print(f"    [FATAL ERROR] Unexpected exception during event processing for <{tag} {attrib}> at event {event_index+1}: {e}")
         traceback.print_exc()
         processed = True
         description = f"[FATAL ERROR processing {tag}]"

    if not processed:
        description = f"[Skipped Tag: {tag}]"

    # Return -1 for player_id if event wasn't player-specific or failed
    return description, processed, event_player_id if processed else -1


def print_game_state_summary(game_state: GameState):
    """Prints a summary of the current GameState."""
    if not isinstance(game_state, GameState):
        print("[Error] Invalid GameState object passed to print_game_state_summary.")
        return

    print(f"  Junme: {getattr(game_state, 'junme', '?'):.2f}")
    current_player = getattr(game_state, 'current_player', '?')
    print(f"  Current Player: P{current_player}")
    last_discard_player = getattr(game_state, 'last_discard_event_player', -1)
    last_discard_tile = getattr(game_state, 'last_discard_event_tile_id', -1)
    print(f"  Last Discard: P{last_discard_player} -> {tile_id_to_string(last_discard_tile) if last_discard_tile != -1 else 'None'}")
    dora_indicators = getattr(game_state, 'dora_indicators', [])
    print(f"  Dora Indicators: {[tile_id_to_string(t) for t in dora_indicators]}")
    scores = getattr(game_state, 'current_scores', ['?']*NUM_PLAYERS)
    print(f"  Scores: {scores}")
    kyotaku = getattr(game_state, 'kyotaku', '?')
    honba = getattr(game_state, 'honba', '?')
    print(f"  Kyotaku/Honba: {kyotaku} / {honba}")

    player_hands = getattr(game_state, 'player_hands', [[] for _ in range(NUM_PLAYERS)])
    player_discards = getattr(game_state, 'player_discards', [[] for _ in range(NUM_PLAYERS)])
    player_melds = getattr(game_state, 'player_melds', [[] for _ in range(NUM_PLAYERS)])
    reach_status = getattr(game_state, 'player_reach_status', [0]*NUM_PLAYERS)
    reach_junme = getattr(game_state, 'player_reach_junme', [-1]*NUM_PLAYERS)

    for p in range(NUM_PLAYERS):
        reach_info = ""
        p_reach_stat = reach_status[p] if isinstance(reach_status, list) and len(reach_status) > p else 0
        p_reach_jun = reach_junme[p] if isinstance(reach_junme, list) and len(reach_junme) > p else -1
        if p_reach_stat == 1: reach_info = "(リーチ宣言)"
        elif p_reach_stat == 2:
             reach_junme_str = f"{p_reach_jun:.2f}" if p_reach_jun != -1 else "?"
             reach_info = f"* (リーチ@{reach_junme_str}巡)"

        print(f"  --- Player {p} {reach_info} ---")
        hand_p = player_hands[p] if isinstance(player_hands, list) and len(player_hands) > p else []
        discards_p = player_discards[p] if isinstance(player_discards, list) and len(player_discards) > p else []
        melds_p = player_melds[p] if isinstance(player_melds, list) and len(player_melds) > p else []

        print(f"    Hand: {format_hand(hand_p)} ({len(hand_p)}枚)")
        print(f"    Discards: {format_discards(discards_p)}")
        print(f"    Melds: {format_melds(melds_p)}")

def main():
    parser = argparse.ArgumentParser(description="Debug GameState and Feature Extraction by stepping through a Tenhou XML log round.")
    parser.add_argument("xml_file", help="Path to the Tenhou XML log file.")
    parser.add_argument("round_index", type=int, help="Round index to debug (1-based).")
    parser.add_argument("--start", type=int, default=1, metavar='EVENT_NUM',
                        help="Event index to start displaying state (1-based, default: 1).")
    parser.add_argument("--count", type=int, default=999, metavar='NUM_EVENTS',
                        help="Maximum number of events to display state for (default: all).")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.xml_file):
        print(f"Error: XML file not found at {args.xml_file}")
        sys.exit(1)
    if args.start < 1:
        print("Error: --start event index must be 1 or greater.")
        sys.exit(1)

    try:
        print(f"Parsing log file: {args.xml_file}...")
        meta, rounds_data = parse_full_mahjong_log(args.xml_file)
        print(f"Found {len(rounds_data)} rounds.")

        if args.round_index < 1 or args.round_index > len(rounds_data):
            print(f"Error: Invalid round_index {args.round_index}. Must be between 1 and {len(rounds_data)}.")
            sys.exit(1)

        round_data = rounds_data[args.round_index - 1]
        events = round_data.get("events", [])
        num_events_in_round = len(events)
        print(f"Targeting Round {args.round_index} (Index {args.round_index - 1}) which has {num_events_in_round} events.")

        game_state = GameState() # Create instance
        # Verify necessary methods exist
        required_methods = ['init_round', 'process_tsumo', 'process_discard', 'process_naki', 'process_reach', 'process_dora', 'process_agari', 'process_ryuukyoku', 'get_event_sequence_features', 'get_static_features']
        missing_methods = [m for m in required_methods if not hasattr(game_state, m)]
        if missing_methods:
             print(f"[FATAL ERROR] GameState object is missing required methods: {', '.join(missing_methods)}! Check game_state.py.")
             sys.exit(1)

        print("Initializing game state for the round...")
        game_state.init_round(round_data)
        print("--- Initial State (After INIT) ---")
        print_game_state_summary(game_state)
        print("-" * 30)

        start_index_0based = max(0, args.start - 1) # 0-based start index for loop

        # Fast-forward: Process events before the start index without printing state details
        if start_index_0based > 0:
            print(f"Fast-forwarding through the first {start_index_0based} events...")
            for i in range(start_index_0based):
                if i < num_events_in_round: # Check bounds
                    process_event(game_state, events[i]["tag"], events[i]["attrib"], i, events, process_only=True)
                else: break
            print("Done fast-forwarding.")
            print("--- State Before Display Start ---")
            print_game_state_summary(game_state)
            print("-" * 30)
        else:
             print("Starting state display from the first event.")

        # Process and display state for the target range of events
        display_count = 0
        end_index_0based = min(start_index_0based + args.count, num_events_in_round)

        print(f"--- Displaying States for Events {start_index_0based + 1} to {end_index_0based} ---")

        for i in range(start_index_0based, end_index_0based):
            event = events[i]; tag = event["tag"]; attrib = event["attrib"]

            print(f"\n>>> Processing Event {i+1}/{num_events_in_round}: <{tag}> {attrib}")

            # Process the event (process_only=False) and get description
            # Pass the full event list 'events' to allow peeking ahead
            event_description, processed_flag, event_player_id = process_event(game_state, tag, attrib, i, events, process_only=False)

            if event_description:
                print(f"    Action: {event_description}")

            # Always print state after attempting to process, unless just fast-forwarding
            if not process_only:
                print("--- State After Event ---")
                print_game_state_summary(game_state)
                print("-" * 30)
                display_count += 1

            # Stop if round ends
            if tag == "AGARI" or tag == "RYUUKYOKU":
                print("\n--- Round End Detected During Display ---")
                break

        if i == end_index_0based - 1 and end_index_0based < num_events_in_round :
             print(f"\n--- Reached Event Count Limit ({args.count}) ---")
        elif i == num_events_in_round -1:
             print("\n--- Reached End of Events for the Round ---")

    except FileNotFoundError:
        print(f"Error: XML file not found at {args.xml_file}")
        sys.exit(1)
    except ImportError:
         print(f"Error: Failed to import necessary modules.")
         sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred during execution:")
        traceback.print_exc() # Print full traceback
        sys.exit(1)

if __name__ == "__main__":
    main()