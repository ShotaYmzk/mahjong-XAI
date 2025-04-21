# full_mahjong_parser.py
# (Keep the existing code as provided in the prompt)
import xml.etree.ElementTree as ET

import os
try:
    from tile_utils import tile_id_to_string, tile_id_to_index # Keep using tile_utils
    from naki_utils import decode_naki, tile_to_name # Keep using naki_utils
except ImportError:
    print("[Error] Cannot import from tile_utils or naki_utils in full_mahjong_parser.")
    # Minimal fallbacks for basic functionality if needed for standalone testing
    def tile_id_to_string(tile: int) -> str: return f"tile_{tile}" if tile is not None else "?"
    def tile_id_to_index(tile: int) -> int: return tile // 4 if tile is not None else -1
    def decode_naki(m: int) -> tuple: return "Unknown", [], -1, -1
    def tile_to_name(tile: int) -> str: return tile_id_to_string(tile)


# 赤牌判定フラグ（GameState側で処理する方針ならここは不要かも）
hongpai = False # Set to True if this script needs to independently identify red fives

def sort_hand(hand: list) -> list:
    """
    手牌（整数のリスト）を、牌タイプインデックスおよび牌IDでソートして返す関数です。
    Handles potential None or invalid values gracefully.
    """
    valid_hand = [t for t in hand if isinstance(t, int) and 0 <= t <= 135]
    # Sort first by type index (0-33), then by ID (0-135) for consistency
    # Handle potential errors in tile_id_to_index
    return sorted(valid_hand, key=lambda t: (tile_id_to_index(t) if tile_id_to_index(t) != -1 else 99, t))

def parse_full_mahjong_log(xml_path: str):
    """
    XMLファイルをパースし、全局（半荘分）のデータを抽出する関数です。
    Returns:
        tuple: (meta, rounds)
            meta: 全体情報の辞書（<GO>, <UN>, <TAIKYOKU>）
            rounds: 各局ごとのデータリスト（各局は辞書）
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"[Error] Failed to parse XML file: {xml_path} - {e}")
        return {}, [] # Return empty data on parse error
    except FileNotFoundError:
        print(f"[Error] XML file not found: {xml_path}")
        return {}, []

    meta = {}
    rounds = []
    current_round = None
    round_index = 0 # Will be 1-based

    for elem in root:
        # Metadata tags
        if elem.tag in ["GO", "UN", "TAIKYOKU"]:
            meta[elem.tag] = elem.attrib
        # Start of a new round
        elif elem.tag == "INIT":
            round_index += 1
            # Reset current round data for the new round
            current_round = {"round_index": round_index, "init": elem.attrib, "events": []}
            rounds.append(current_round)
        # Events within a round (or potentially before first INIT like DORA?)
        else:
            event_data = {"tag": elem.tag, "attrib": elem.attrib, "text": elem.text}
            if current_round is not None:
                current_round["events"].append(event_data)
                # Check for round end tags
                if elem.tag in ["AGARI", "RYUUKYOKU"]:
                    current_round["result"] = event_data # Store the result event
                    # current_round = None # Keep current_round active until next INIT
            else:
                # Handle tags appearing before the first INIT (e.g., global DORA?)
                # This is less common but possible. Append to the last round's events if available?
                # Or maybe store in a separate pre-game event list in meta?
                # For now, let's ignore events before the first INIT or append to last if exists.
                if rounds:
                     # Append to the very last round's events if no current round is active
                     # This might happen for tags after RYUUKYOKU/AGARI but before next INIT
                     rounds[-1]["events"].append(event_data)
                else:
                    # Or store in meta if no rounds yet?
                    if 'pre_init_events' not in meta: meta['pre_init_events'] = []
                    meta['pre_init_events'].append(event_data)
                    # print(f"[Debug] Tag '{elem.tag}' found before first INIT.")

    return meta, rounds


# extract_round_features remains useful for debugging and understanding data
def extract_round_features(round_data: dict) -> dict:
    """
    １局分のデータから各種特徴量を抽出し、辞書として返す関数です。
    （主にデバッグや分析用途。GameStateで状態は管理）
    """
    features = {}
    features["round_index"] = round_data.get("round_index", "N/A")

    init_data = round_data.get("init", {})
    features["dealer"] = int(init_data.get("oya", -1)) # Oya is 0-3 index
    try:
        features["init_scores"] = list(map(int, init_data.get("ten", "25000,25000,25000,25000").split(",")))
    except ValueError:
        features["init_scores"] = [25000] * 4 # Fallback

    seed_str = init_data.get("seed")
    if seed_str:
        seed_fields = seed_str.split(",")
        if len(seed_fields) >= 6:
            try:
                features["seed_round_wind_raw"] = int(seed_fields[0]) # 0=E1, 1=E2,... 4=S1...
                features["seed_honba"] = int(seed_fields[1])
                features["seed_kyotaku"] = int(seed_fields[2])
                # Dice are 0-5 in log, add 1 for 1-6
                features["dice1"] = int(seed_fields[3]) + 1
                features["dice2"] = int(seed_fields[4]) + 1
                features["dorahaip_indicator"] = int(seed_fields[5]) # Initial Dora indicator ID
            except (ValueError, IndexError):
                 features["seed_round_wind_raw"] = features["seed_honba"] = features["seed_kyotaku"] = features["dice1"] = features["dice2"] = features["dorahaip_indicator"] = None
                 print(f"[Warning] Could not parse seed string: {seed_str}")
        else:
            features["seed_round_wind_raw"] = features["seed_honba"] = features["seed_kyotaku"] = features["dice1"] = features["dice2"] = features["dorahaip_indicator"] = None
            print(f"[Warning] Incomplete seed string: {seed_str}")
    else:
        features["seed_round_wind_raw"] = features["seed_honba"] = features["seed_kyotaku"] = features["dice1"] = features["dice2"] = features["dorahaip_indicator"] = None

    player_hands = {}
    for pid in range(4):
        key = f"hai{pid}"
        hand_str = init_data.get(key)
        if hand_str:
            try:
                hand = list(map(int, hand_str.split(",")))
                player_hands[pid] = sort_hand(hand) # Use the sorting function
            except ValueError:
                player_hands[pid] = []
                print(f"[Warning] Could not parse hand string for player {pid}: {hand_str}")
        else:
            player_hands[pid] = [] # No hand info (shouldn't happen in valid logs)
    features["player_hands_initial"] = player_hands

    events = round_data.get("events", [])
    features["events_raw"] = events # Store raw events for reference

    # --- Processed Events (Example - GameState handles the detailed logic) ---
    tsumo_events = []
    discard_events = []
    reach_events = []
    naki_events = []
    dora_events = []
    # Maps tag prefix to player index (0-3)
    tsumo_map = { "T": 0, "U": 1, "V": 2, "W": 3 }
    discard_map = { "D": 0, "E": 1, "F": 2, "G": 3 } # Includes lower case tsumogiri

    for ev in events:
        tag = ev["tag"]
        attrib = ev["attrib"]
        processed_event = {"raw_tag": tag, "attrib": attrib} # Start with basic info

        # Tsumo/Discard events (Format: [TUVWXYZDEFG][tile_id])
        if len(tag) > 1 and tag[0].upper() in tsumo_map:
            prefix = tag[0].upper()
            player_id = tsumo_map[prefix]
            try:
                tile_id = int(tag[1:])
                processed_event["type"] = "TSUMO"
                processed_event["player"] = player_id
                processed_event["tile_id"] = tile_id
                tsumo_events.append(processed_event)
            except ValueError:
                print(f"[Warning] Invalid Tsumo tag format: {tag}")
        elif len(tag) > 1 and tag[0].upper() in discard_map:
            prefix_upper = tag[0].upper()
            player_id = discard_map[prefix_upper]
            try:
                tile_id = int(tag[1:])
                processed_event["type"] = "DISCARD"
                processed_event["player"] = player_id
                processed_event["tile_id"] = tile_id
                processed_event["tsumogiri"] = tag[0].islower() # Check original tag case
                discard_events.append(processed_event)
            except ValueError:
                print(f"[Warning] Invalid Discard tag format: {tag}")
        # Naki event
        elif tag == "N":
            try:
                naki_player_id = int(attrib.get("who", -1))
                meld_code = int(attrib.get("m", "0"))
                if naki_player_id != -1:
                    naki_type, naki_tiles, trigger_id, _ = decode_naki(meld_code) # Use updated naki_utils
                    processed_event["type"] = "NAKI"
                    processed_event["player"] = naki_player_id
                    processed_event["meld_code"] = meld_code
                    processed_event["naki_type"] = naki_type
                    processed_event["naki_tiles_decoded"] = naki_tiles # Decoded tile IDs
                    processed_event["trigger_tile_id_decoded"] = trigger_id # Decoded trigger ID
                    naki_events.append(processed_event)
                else:
                    print(f"[Warning] Naki tag missing 'who' attribute: {attrib}")
            except (ValueError, KeyError) as e:
                print(f"[Warning] Could not parse N tag: {attrib} - {e}")
        # Reach event
        elif tag == "REACH":
            try:
                reach_player_id = int(attrib.get("who", -1))
                step = int(attrib.get("step", -1)) # 1: Declare, 2: Stick accepted
                if reach_player_id != -1 and step != -1:
                    processed_event["type"] = "REACH"
                    processed_event["player"] = reach_player_id
                    processed_event["step"] = step
                    reach_events.append(processed_event)
                else:
                    print(f"[Warning] Invalid REACH tag attributes: {attrib}")
            except (ValueError, KeyError):
                print(f"[Warning] Could not parse REACH tag: {attrib}")
        # Dora event (revealing new indicator)
        elif tag == "DORA":
             try:
                dora_indicator_id = int(attrib.get("hai", -1))
                if dora_indicator_id != -1:
                    processed_event["type"] = "DORA"
                    processed_event["dora_indicator_id"] = dora_indicator_id
                    dora_events.append(processed_event)
                else:
                    print(f"[Warning] Invalid DORA tag attribute: {attrib}")
             except (ValueError, KeyError):
                  print(f"[Warning] Could not parse DORA tag: {attrib}")
        # Agari/Ryuukyoku handled separately below

    features["tsumo_events"] = tsumo_events
    features["discard_events"] = discard_events
    features["reach_events"] = reach_events
    features["naki_events"] = naki_events
    features["dora_events"] = dora_events

    # Result processing
    result_event = round_data.get("result")
    features["result_raw"] = result_event
    if result_event:
        res_tag = result_event.get("tag")
        res_attr = result_event.get("attrib", {})
        features["result_type"] = res_tag

        try:
            # Score changes (sc format: p0_score,p0_change,p1_score,p1_change,...)
            sc_str = res_attr.get("sc")
            if sc_str:
                sc_values = list(map(int, sc_str.split(",")))
                if len(sc_values) == 8:
                    features["final_scores"] = [sc_values[i*2] for i in range(4)]
                    features["score_changes"] = [sc_values[i*2+1] for i in range(4)]
                else:
                     print(f"[Warning] Invalid 'sc' attribute length in {res_tag}: {sc_str}")
            # Honba/Kyotaku (ba format: honba,kyotaku)
            ba_str = res_attr.get("ba")
            if ba_str:
                 ba_values = list(map(int, ba_str.split(",")))
                 if len(ba_values) == 2:
                     features["final_honba"] = ba_values[0]
                     features["final_kyotaku"] = ba_values[1] # Kyotaku usually becomes 0 after agari
                 else:
                     print(f"[Warning] Invalid 'ba' attribute length in {res_tag}: {ba_str}")

        except (ValueError, KeyError):
            print(f"[Warning] Could not parse result attributes for {res_tag}: {res_attr}")

        # Agari specific info
        if res_tag == "AGARI":
            try:
                features["winner"] = int(res_attr.get("who", -1))
                features["loser"] = int(res_attr.get("fromWho", -1)) # Player who dealt in (ron) or -1 (tsumo)
                # Yaku/Score info (ten format: points,fu,limit_flag?)
                ten_str = res_attr.get("ten")
                if ten_str:
                    ten_values = list(map(int, ten_str.split(",")))
                    features["agari_points"] = ten_values[0] if len(ten_values)>0 else None
                    features["agari_fu"] = ten_values[1] if len(ten_values)>1 else None
                    features["agari_limit"] = ten_values[2] if len(ten_values)>2 else None # e.g., 1 for mangan
                # Yaku list (yaku format: yaku_id,han_value,yaku_id,han_value,...)
                # Yakuman list (yakuman format: yakuman_id,yakuman_id,...)
                features["agari_yaku"] = res_attr.get("yaku")
                features["agari_yakuman"] = res_attr.get("yakuman")
                features["agari_dora"] = res_attr.get("doraHai") # Dora tiles in hand/melds
                features["agari_ura_dora"] = res_attr.get("doraHaiUra") # Ura dora indicators
                features["agari_final_hand"] = res_attr.get("hai") # Winner's final hand
                features["agari_final_melds_code"] = res_attr.get("m") # Winner's melds (codes)
                features["agari_winning_tile"] = int(res_attr.get("machi", -1)) # The winning tile ID

            except (ValueError, KeyError):
                 print(f"[Warning] Could not parse AGARI attributes: {res_attr}")

        # Ryuukyoku specific info
        elif res_tag == "RYUUKYOKU":
            features["ryuukyoku_type"] = res_attr.get("type", "unknown") # e.g., yao9, kaze4, reach4, etc.
            # Tenpai status (owari format similar to sc, but shows tenpai/noten payment)

    else:
        features["result_type"] = None

    return features


if __name__ == "__main__":
    # Make sure the path is correct relative to where you run the script
    default_xml = "../xml_logs/2009022011gm-00a9-0000-d7935c6d.xml"
    # Check if the default file exists, otherwise prompt or use a fixed path
    import sys
    if len(sys.argv) > 1:
        xml_file = sys.argv[1]
    else:
        xml_file = default_xml
        if not os.path.exists(xml_file):
             # Try finding it relative to this script's location if default fails
             script_dir = os.path.dirname(__file__)
             xml_file = os.path.join(script_dir, default_xml)
             if not os.path.exists(xml_file):
                 print(f"Error: Default XML log not found at {default_xml} or relative path.")
                 print("Usage: python full_mahjong_parser.py [path/to/your/log.xml]")
                 sys.exit(1)


    print(f"Parsing XML: {xml_file}")
    meta, rounds = parse_full_mahjong_log(xml_file)

    if not rounds:
        print("No rounds found in the XML file.")
        sys.exit(1)

    print("\n【Overall Meta Information】")
    for key, val in meta.items():
        # Avoid printing potentially long pre-init events list directly
        if key == 'pre_init_events':
            print(f"  {key}: {len(val)} events")
        else:
            print(f"  {key}: {val}")

    print("\n【Features for Each Round】")
    for r_idx, r in enumerate(rounds):
        print("-" * 50)
        print(f"Processing Round {r_idx + 1} (Data Index: {r.get('round_index')})")
        try:
            feat = extract_round_features(r)

            print(f"Round Index (from data): {feat.get('round_index', 'N/A')}")
            dealer_idx = feat.get('dealer', -1)
            dealer_str = f"Player {dealer_idx}" if dealer_idx != -1 else "Unknown"
            print(f"Dealer: {dealer_str}")
            print(f"Initial Scores: {feat.get('init_scores', 'N/A')}")

            print("Seed Info:")
            if feat.get('seed_round_wind_raw') is not None:
                 round_winds = ["East 1", "East 2", "East 3", "East 4",
                                "South 1", "South 2", "South 3", "South 4",
                                "West 1", "West 2", "West 3", "West 4", # Extend if needed
                                "North 1", "North 2", "North 3", "North 4"]
                 wind_idx = feat['seed_round_wind_raw']
                 round_wind_str = round_winds[wind_idx] if 0 <= wind_idx < len(round_winds) else f"Raw {wind_idx}"
                 print(f"  Round/Wind: {round_wind_str}, Honba: {feat.get('seed_honba', '?')}, Kyotaku: {feat.get('seed_kyotaku', '?')}")
                 print(f"  Dice: {feat.get('dice1', '?')} + {feat.get('dice2', '?')}")
                 dora_ind = feat.get('dorahaip_indicator')
                 dora_str = tile_id_to_string(dora_ind) if dora_ind is not None else '?'
                 print(f"  Initial Dora Indicator: {dora_str} (ID: {dora_ind})")
            else:
                print("  Seed info not available.")

            print("Initial Hands (Sorted):")
            hands = feat.get("player_hands_initial", {})
            for pid in range(4):
                hand = hands.get(pid, [])
                hand_str = " ".join([tile_id_to_string(t) for t in hand]) if hand else "N/A"
                print(f"  Player {pid}: {hand_str}")

            print(f"Total Events Processed: {len(feat.get('events_raw', []))}")
            print(f"  Tsumo Events: {len(feat.get('tsumo_events', []))}")
            print(f"  Discard Events: {len(feat.get('discard_events', []))}")
            print(f"  Naki Events: {len(feat.get('naki_events', []))}")
            # Example: Print first few discards
            print("  First 5 Discards:")
            for d_ev in feat.get('discard_events', [])[:5]:
                 p = d_ev['player']
                 t_str = tile_id_to_string(d_ev['tile_id'])
                 tsumo_marker = "*" if d_ev['tsumogiri'] else ""
                 print(f"    P{p}: {t_str}{tsumo_marker}")

            print("  Naki Details:")
            for n_ev in feat.get('naki_events', []):
                 p = n_ev['player']
                 n_type = n_ev['naki_type']
                 tiles_str = " ".join([tile_id_to_string(t) for t in n_ev['naki_tiles_decoded']])
                 trigger_str = tile_id_to_string(n_ev['trigger_tile_id_decoded']) if n_ev.get('trigger_tile_id_decoded', -1) != -1 else ""
                 # We need GameState context to know *who* it was from
                 print(f"    P{p} called {n_type}: [{tiles_str}] (Trigger: {trigger_str})")


            print(f"Result: {feat.get('result_type', 'N/A')}")
            if feat.get('result_type') == 'AGARI':
                print(f"  Winner: P{feat.get('winner', '?')}, Loser: P{feat.get('loser', '?')}")
                print(f"  Score: {feat.get('agari_points', '?')}pts, Fu: {feat.get('agari_fu', '?')}")
                # Print yaku/yakuman if needed
            elif feat.get('result_type') == 'RYUUKYOKU':
                 print(f"  Type: {feat.get('ryuukyoku_type', 'unknown')}")

            if feat.get('final_scores'):
                 print(f"Final Scores: {feat.get('final_scores')}")
                 print(f"Score Changes: {feat.get('score_changes')}")

        except Exception as e:
            print(f"!! Error processing round {r_idx + 1}: {e}")
            import traceback
            traceback.print_exc()