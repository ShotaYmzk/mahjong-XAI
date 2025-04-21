# full_mahjong_parser.py (Revised to use previous discard event for naki source)
import xml.etree.ElementTree as ET
import os
import sys
import re

# --- Path setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path: sys.path.insert(0, script_dir)
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path: sys.path.insert(0, parent_dir)
# --- End Path setup ---

try:
    from tile_utils import tile_id_to_string, tile_id_to_index
    # Keep using the latest decode_naki, but rely on parser for 'from_who' and 'trigger'
    from naki_utils import decode_naki
except ImportError as e:
    # ... (Fallback functions remain the same) ...
    print(f"[Error] Cannot import from tile_utils or naki_utils: {e}")
    print("[Info] Using minimal fallback functions for basic parsing.")
    def tile_id_to_string(tile: int) -> str: return f"t{tile}" if tile is not None and tile != -1 else "?"
    def tile_id_to_index(tile: int) -> int: return tile // 4 if tile is not None and isinstance(tile, int) and tile !=-1 else -1
    def decode_naki(m: int) -> tuple: return "Unknown", [], -1, -1


# ... (sort_hand, parse_full_mahjong_log remain the same) ...
def sort_hand(hand: list) -> list:
    """
    手牌（整数のリスト）を、牌タイプインデックスおよび牌IDでソートして返す関数です。
    Handles potential None or invalid values gracefully.
    """
    valid_hand = [t for t in hand if isinstance(t, int) and 0 <= t <= 135]
    # Sort first by type index (0-33), then by ID (0-135) for consistency
    return sorted(valid_hand, key=lambda t: (tile_id_to_index(t) if tile_id_to_index(t) != -1 else 99, t))

def clean_xml_string(xml_string):
    """Clean an XML string to handle common issues like duplicate attributes."""
    # Handle duplicate attributes by keeping the first occurrence
    # This uses a regex to find attributes with the same name and keeps only the first one
    pattern = r'(\w+)=("[^"]*")(\s+\1="[^"]*")+' 
    while re.search(pattern, xml_string):
        xml_string = re.sub(pattern, r'\1=\2', xml_string)
    return xml_string

def parse_full_mahjong_log(xml_path: str):
    """
    XMLファイルをパースし、全局（半荘分）のデータを抽出する関数です。
    Returns:
        tuple: (meta, rounds)
            meta: 全体情報の辞書（<GO>, <UN>, <TAIKYOKU>）
            rounds: 各局ごとのデータリスト（各局は辞書）
    """
    try:
        # First, read the file as a string
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        
        # Clean the XML content to handle duplicate attributes
        cleaned_xml = clean_xml_string(xml_content)
        
        # Parse the cleaned XML
        root = ET.fromstring(cleaned_xml)
    except ET.ParseError as e:
        print(f"[Error] Failed to parse XML file: {xml_path} - {e}")
        return {}, [] # Return empty data on parse error
    except FileNotFoundError:
        print(f"[Error] XML file not found: {xml_path}")
        return {}, []
    except Exception as e:
        print(f"[Error] Unexpected error parsing XML file {xml_path}: {str(e)}")
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
        # Events within a round
        elif current_round is not None: # Ensure we are inside a round
            event_data = {"tag": elem.tag, "attrib": elem.attrib, "text": elem.text}
            current_round["events"].append(event_data)
            # Check for round end tags
            if elem.tag in ["AGARI", "RYUUKYOKU"]:
                current_round["result"] = event_data # Store the result event
        # Handle tags appearing before the first INIT or after last round (less common)
        else:
            event_data = {"tag": elem.tag, "attrib": elem.attrib, "text": elem.text}
            if rounds:
                 # Append to the very last round's events if no current round is active
                 rounds[-1]["events"].append(event_data)
            else:
                # Or store in meta if no rounds yet?
                if 'pre_init_events' not in meta: meta['pre_init_events'] = []
                meta['pre_init_events'].append(event_data)

    return meta, rounds


def extract_round_features(round_data: dict) -> dict:
    """
    １局分のデータから各種特徴量を抽出し、辞書として返す関数です。
    鳴きの「誰から」「どの牌」は直前の打牌イベントから特定します。
    """
    features = {}
    # ... (Initial feature extraction like round_index, init_scores, seed remains the same) ...
    features["round_index"] = round_data.get("round_index", "N/A")
    init_data = round_data.get("init", {})
    features["dealer"] = int(init_data.get("oya", -1))
    try: features["init_scores"] = list(map(int, init_data.get("ten", "25000,25000,25000,25000").split(",")))
    except ValueError: features["init_scores"] = [25000] * 4
    seed_str = init_data.get("seed")
    if seed_str:
        seed_fields = seed_str.split(",")
        if len(seed_fields) >= 6:
            try:
                features["seed_round_wind_raw"] = int(seed_fields[0]); features["seed_honba"] = int(seed_fields[1]); features["seed_kyotaku"] = int(seed_fields[2])
                features["dice1"] = int(seed_fields[3]) + 1; features["dice2"] = int(seed_fields[4]) + 1; features["dorahaip_indicator"] = int(seed_fields[5])
            except (ValueError, IndexError): features["seed_round_wind_raw"] = features["seed_honba"] = features["seed_kyotaku"] = features["dice1"] = features["dice2"] = features["dorahaip_indicator"] = None; print(f"[Warning] Could not parse seed string: {seed_str}")
        else: features["seed_round_wind_raw"] = features["seed_honba"] = features["seed_kyotaku"] = features["dice1"] = features["dice2"] = features["dorahaip_indicator"] = None; print(f"[Warning] Incomplete seed string: {seed_str}")
    else: features["seed_round_wind_raw"] = features["seed_honba"] = features["seed_kyotaku"] = features["dice1"] = features["dice2"] = features["dorahaip_indicator"] = None
    player_hands = {};
    for pid in range(4):
        key = f"hai{pid}"; hand_str = init_data.get(key)
        if hand_str:
            try: player_hands[pid] = sort_hand(list(map(int, hand_str.split(","))))
            except ValueError: player_hands[pid] = []; print(f"[Warning] Could not parse hand string for player {pid}: {hand_str}")
        else: player_hands[pid] = []
    features["player_hands_initial"] = player_hands


    events = round_data.get("events", [])
    features["events_raw"] = events # Store raw events for reference

    tsumo_events = []
    discard_events = []
    reach_events = []
    naki_events = []
    dora_events = []
    tsumo_map = { "T": 0, "U": 1, "V": 2, "W": 3 }
    discard_map = { "D": 0, "E": 1, "F": 2, "G": 3 }

    # --- MODIFICATION: Keep track of the last discard event ---
    last_discard_info = {"player": -1, "tile_id": -1}
    # ---------------------------------------------------------

    for ev in events:
        tag = ev["tag"]
        attrib = ev["attrib"]
        processed_event = {"raw_tag": tag, "attrib": attrib}

        # Tsumo event
        if len(tag) > 1 and tag[0].upper() in tsumo_map and tag[1:].isdigit():
            prefix = tag[0].upper()
            player_id = tsumo_map[prefix]
            try:
                tile_id = int(tag[1:])
                processed_event["type"] = "TSUMO"; processed_event["player"] = player_id; processed_event["tile_id"] = tile_id
                tsumo_events.append(processed_event)
                last_discard_info = {"player": -1, "tile_id": -1} # Reset last discard on tsumo
            except ValueError: print(f"[Warning] Invalid Tsumo tag format despite check: {tag}")

        # Discard event
        elif len(tag) > 1 and tag[0].upper() in discard_map and tag[1:].isdigit():
            prefix_upper = tag[0].upper()
            player_id = discard_map[prefix_upper]
            try:
                tile_id = int(tag[1:])
                processed_event["type"] = "DISCARD"; processed_event["player"] = player_id; processed_event["tile_id"] = tile_id
                processed_event["tsumogiri"] = tag[0].islower()
                discard_events.append(processed_event)
                # --- MODIFICATION: Update last discard info ---
                last_discard_info = {"player": player_id, "tile_id": tile_id}
                # ----------------------------------------------
            except ValueError: print(f"[Warning] Invalid Discard tag format despite check: {tag}")

        # Naki event
        elif tag == "N":
            try:
                naki_player_id = int(attrib.get("who", -1)) # Player who calls
                meld_code = int(attrib.get("m", "0"))
                if naki_player_id != -1:
                    # Decode type and tiles using decode_naki
                    naki_type, naki_tiles, _, _ = decode_naki(meld_code) # Ignore trigger/kui from decode_naki

                    # --- MODIFICATION: Determine source and trigger from last discard ---
                    from_who_player_id = -1
                    trigger_tile_id = -1
                    is_valid_naki_source = False

                    # Check if there was a preceding discard event
                    if last_discard_info["player"] != -1:
                        # Check if the caller is different from the discarder (cannot call from self except Kan)
                        if naki_player_id != last_discard_info["player"]:
                            # Check if the naki type allows calling from others
                            if naki_type in ["チー", "ポン", "大明槓"]:
                                from_who_player_id = last_discard_info["player"]
                                trigger_tile_id = last_discard_info["tile_id"]
                                is_valid_naki_source = True
                                # Further validation: check relative position for Chi? (Optional here)
                                # expected_kui_for_chi = (naki_player_id - from_who_player_id + 4) % 4 # Calculate expected relative pos
                                # if naki_type == "チー" and expected_kui_for_chi != 2: print("[Warn] Unexpected relative position for Chi")

                        # Handle Kans from self (Ankan/Kakan)
                        elif naki_type in ["暗槓", "加槓"]:
                            from_who_player_id = naki_player_id # Source is self
                            trigger_tile_id = -1 # No external trigger tile for Ankan/Kakan (trigger is internal)
                            is_valid_naki_source = True # Assuming Ankan/Kakan are valid calls from self
                    # -----------------------------------------------------------------

                    if is_valid_naki_source or naki_type in ["暗槓", "加槓"]: # Proceed if source is valid or it's a self-kan
                        processed_event["type"] = "NAKI"
                        processed_event["player"] = naki_player_id
                        processed_event["meld_code"] = meld_code
                        processed_event["naki_type"] = naki_type
                        processed_event["naki_tiles_decoded"] = naki_tiles # From decode_naki

                        # --- MODIFICATION: Store determined source ---
                        processed_event["from_who_player_id"] = from_who_player_id
                        processed_event["trigger_tile_id"] = trigger_tile_id # This is now the reliable trigger ID
                        # --------------------------------------------
                        naki_events.append(processed_event)

                        # Reset last discard as it was called
                        last_discard_info = {"player": -1, "tile_id": -1}
                    else:
                        # This case means Naki tag appeared without a valid preceding discard
                        print(f"[Warning] Naki event for P{naki_player_id} (type: {naki_type}, m={meld_code}) occurred without a valid preceding discard. Last discard: {last_discard_info}. Skipping naki processing.")

                else:
                    print(f"[Warning] Naki tag missing 'who' attribute: {attrib}")
            except (ValueError, KeyError, TypeError) as e:
                meld_code_str = attrib.get("m", "N/A")
                print(f"[Warning] Could not parse N tag or decode meld (m={meld_code_str}): {attrib} - {e}")

        # Reach event
        elif tag == "REACH":
            # ... (Reach processing remains the same) ...
            try:
                reach_player_id = int(attrib.get("who", -1)); step = int(attrib.get("step", -1))
                if reach_player_id != -1 and step != -1:
                    processed_event["type"] = "REACH"; processed_event["player"] = reach_player_id; processed_event["step"] = step
                    reach_events.append(processed_event)
                    # Reset last discard if reach is declared (step 1)? Usually discard follows immediately.
                    # if step == 1: last_discard_info = {"player": -1, "tile_id": -1}
                else: print(f"[Warning] Invalid REACH tag attributes: {attrib}")
            except (ValueError, KeyError): print(f"[Warning] Could not parse REACH tag: {attrib}")


        # Dora event
        elif tag == "DORA":
            # ... (Dora processing remains the same) ...
             try:
                hai_attr = attrib.get("hai")
                if hai_attr is not None and hai_attr.isdigit():
                    dora_indicator_id = int(hai_attr)
                    processed_event["type"] = "DORA"; processed_event["dora_indicator_id"] = dora_indicator_id
                    dora_events.append(processed_event)
                else: print(f"[Warning] Invalid or missing 'hai' attribute in DORA tag: {attrib}")
             except (KeyError): print(f"[Warning] Could not parse DORA tag: {attrib}")

        # Handle the specific warning case seen in the log
        elif tag.upper() == "DORA" and not tag[1:].isdigit():
             print(f"[Info] Skipping non-standard tag that looks like DORA: {tag}")
             pass # Ignore this specific case

        # Agari/Ryuukyoku will be handled below the loop

    features["tsumo_events"] = tsumo_events
    features["discard_events"] = discard_events
    features["reach_events"] = reach_events
    features["naki_events"] = naki_events
    features["dora_events"] = dora_events

    # Result processing
    result_event = round_data.get("result")
    features["result_raw"] = result_event
    # ... (Result processing remains the same) ...
    if result_event:
        res_tag = result_event.get("tag"); res_attr = result_event.get("attrib", {})
        features["result_type"] = res_tag
        try:
            sc_str = res_attr.get("sc")
            if sc_str: sc_values = list(map(int, sc_str.split(","))); features["final_scores"] = [sc_values[i*2] for i in range(4)]; features["score_changes"] = [sc_values[i*2+1] for i in range(4)]
            ba_str = res_attr.get("ba")
            if ba_str: ba_values = list(map(int, ba_str.split(","))); features["final_honba"] = ba_values[0]; features["final_kyotaku"] = ba_values[1]
        except (ValueError, KeyError, IndexError) as e: print(f"[Warning] Could not parse result attributes for {res_tag}: {res_attr} - {e}")
        if res_tag == "AGARI":
            try:
                features["winner"] = int(res_attr.get("who", -1)); features["loser"] = int(res_attr.get("fromWho", -1))
                if features["winner"] == features["loser"]: features["loser"] = -1
                ten_str = res_attr.get("ten");
                if ten_str: ten_values = list(map(int, ten_str.split(","))); features["agari_fu"] = ten_values[0] if len(ten_values)>0 else None; features["agari_points"] = ten_values[1] if len(ten_values)>1 else None; features["agari_limit"] = ten_values[2] if len(ten_values)>2 else None
                features["agari_yaku"] = res_attr.get("yaku"); features["agari_yakuman"] = res_attr.get("yakuman"); features["agari_dora"] = res_attr.get("doraHai"); features["agari_ura_dora"] = res_attr.get("doraHaiUra"); features["agari_final_hand"] = res_attr.get("hai"); features["agari_final_melds_code"] = res_attr.get("m"); features["agari_winning_tile"] = int(res_attr.get("machi", -1))
            except (ValueError, KeyError) as e: print(f"[Warning] Could not parse AGARI attributes: {res_attr} - {e}")
        elif res_tag == "RYUUKYOKU":
            features["ryuukyoku_type"] = res_attr.get("type", "unknown")
            owari_str = res_attr.get("owari");
            if owari_str:
                 features["ryuukyoku_owari_raw"] = owari_str
                 try:
                     owari_values = list(map(int, owari_str.split(",")))
                     if len(owari_values) == 8: features["final_scores"] = [owari_values[i*2] for i in range(4)]; features["score_changes"] = [owari_values[i*2+1] for i in range(4)]; features["tenpai_status"] = [sc > 0 for sc in features["score_changes"]]
                     else: print(f"[Warning] Invalid 'owari' attribute length in RYUUKYOKU: {owari_str}")
                 except ValueError: print(f"[Warning] Could not parse 'owari' attribute in RYUUKYOKU: {owari_str}")
            for pid in range(4): key = f"hai{pid}";
            if key in res_attr:
                if "ryuukyoku_hands" not in features: features["ryuukyoku_hands"] = {}
                features["ryuukyoku_hands"][pid] = res_attr[key]
    else: features["result_type"] = None

    return features

if __name__ == "__main__":
    # ... (XML file finding logic remains the same) ...
    default_xml = "../xml_logs/2009022011gm-00a9-0000-d7935c6d.xml"
    if len(sys.argv) > 1: xml_file = sys.argv[1]
    else:
        xml_file = default_xml
        if not os.path.exists(xml_file):
             xml_file_rel = os.path.join(script_dir, default_xml)
             if os.path.exists(xml_file_rel): xml_file = xml_file_rel
             else:
                 abs_log_path = os.path.abspath(os.path.join(script_dir, default_xml))
                 xml_file_up = abs_log_path
                 if os.path.exists(xml_file_up): xml_file = xml_file_up
                 else:
                     print(f"Error: Default XML log not found at specified relative path: {default_xml}")
                     print(f"       Tried absolute path: {abs_log_path}")
                     print(f"       Also checked relative to script: {xml_file_rel}")
                     search_dirs = [".", "./xml_logs", "../xml_logs", os.path.join(script_dir,"xml_logs"), os.path.join(parent_dir,"xml_logs")]
                     found_log = None
                     for sdir in search_dirs:
                         abs_sdir = os.path.abspath(sdir);
                         if os.path.isdir(abs_sdir):
                             try:
                                 for fname in os.listdir(abs_sdir):
                                     if fname.lower().endswith(".xml"): found_log = os.path.join(abs_sdir, fname); print(f"Info: Found an XML file, using: {found_log}"); xml_file = found_log; break
                             except OSError as e: print(f"Warning: Could not access directory {abs_sdir}: {e}")
                         if found_log: break
                     if not found_log: print("Error: No XML file found in common search directories."); sys.exit(1)

    print(f"Parsing XML: {xml_file}")

    try:
        from tile_utils import tile_id_to_string, tile_id_to_index
        from naki_utils import decode_naki
    except ImportError as e: print(f"[Error] Re-import failed: {e}")

    meta, rounds = parse_full_mahjong_log(xml_file)

    if not rounds: print("No rounds found in the XML file."); sys.exit(1)

    print("\n【Overall Meta Information】")
    for key, val in meta.items():
        if key == 'pre_init_events': print(f"  {key}: {len(val)} events")
        else: print(f"  {key}: {val}")

    print("\n【Features for Each Round】")
    for r_idx, r in enumerate(rounds):
        print("-" * 50)
        print(f"Processing Round {r_idx + 1} (Data Index: {r.get('round_index')})")
        try:
            feat = extract_round_features(r)

            # ... (Round info printing remains the same) ...
            print(f"Round Index (from data): {feat.get('round_index', 'N/A')}")
            dealer_idx = feat.get('dealer', -1); dealer_str = f"Player {dealer_idx}" if dealer_idx != -1 else "Unknown"; print(f"Dealer: {dealer_str}")
            print(f"Initial Scores: {feat.get('init_scores', 'N/A')}")
            print("Seed Info:")
            if feat.get('seed_round_wind_raw') is not None:
                 round_winds = ["East 1", "East 2", "East 3", "East 4", "South 1", "South 2", "South 3", "South 4", "West 1", "West 2", "West 3", "West 4", "North 1", "North 2", "North 3", "North 4"]
                 wind_idx = feat['seed_round_wind_raw']; round_wind_str = round_winds[wind_idx] if 0 <= wind_idx < len(round_winds) else f"Raw {wind_idx}"
                 print(f"  Round/Wind: {round_wind_str}, Honba: {feat.get('seed_honba', '?')}, Kyotaku: {feat.get('seed_kyotaku', '?')}")
                 print(f"  Dice: {feat.get('dice1', '?')} + {feat.get('dice2', '?')}")
                 dora_ind = feat.get('dorahaip_indicator'); dora_str = tile_id_to_string(dora_ind) if dora_ind is not None else '?'; print(f"  Initial Dora Indicator: {dora_str} (ID: {dora_ind})")
            else: print("  Seed info not available.")
            print("Initial Hands (Sorted):")
            hands = feat.get("player_hands_initial", {})
            for pid in range(4): hand = hands.get(pid, []); hand_str = " ".join([tile_id_to_string(t) for t in hand]) if hand else "N/A"; print(f"  Player {pid}: {hand_str}")
            print(f"Total Events Processed: {len(feat.get('events_raw', []))}")
            print(f"  Tsumo Events: {len(feat.get('tsumo_events', []))}")
            print(f"  Discard Events: {len(feat.get('discard_events', []))}")
            print(f"  Naki Events: {len(feat.get('naki_events', []))}")
            print(f"  Reach Events: {len(feat.get('reach_events', []))}")
            print(f"  Dora Events: {len(feat.get('dora_events', []))}")
            print("  First 5 Discards:")
            for d_ev in feat.get('discard_events', [])[:5]: p = d_ev['player']; t_str = tile_id_to_string(d_ev['tile_id']); tsumo_marker = "*" if d_ev.get('tsumogiri', False) else ""; print(f"    P{p}: {t_str}{tsumo_marker}")


            # --- MODIFIED NAKI DISPLAY ---
            print("  Naki Details:")
            for n_ev in feat.get('naki_events', []):
                 p = n_ev['player'] # Who called
                 n_type = n_ev.get('naki_type', 'Unknown')
                 tiles_decoded = n_ev.get('naki_tiles_decoded', [])
                 tiles_str = " ".join([tile_id_to_string(t) for t in tiles_decoded])
                 meld_code = n_ev.get('meld_code', 0)
                 # Get the determined source player and trigger tile
                 from_who_pid = n_ev.get('from_who_player_id', -1)
                 trigger_tid = n_ev.get('trigger_tile_id', -1)

                 from_str = ""
                 trigger_str = ""

                 # Determine relative position string (optional but helpful)
                 relative_pos_str = ""
                 if from_who_pid != -1 and from_who_pid != p:
                     diff = (from_who_pid - p + 4) % 4
                     if diff == 1: relative_pos_str = "下家"
                     elif diff == 2: relative_pos_str = "対面"
                     elif diff == 3: relative_pos_str = "上家"

                 # Construct display strings based on naki type and determined source
                 if n_type in ["チー", "ポン", "大明槓"]:
                     if from_who_pid != -1:
                         from_str = f" from P{from_who_pid} ({relative_pos_str})"
                         if trigger_tid != -1:
                             trigger_str = f" on {tile_id_to_string(trigger_tid)}"
                         else:
                             # Should not happen if logic is correct and discard existed
                             trigger_str = " on ? (Trigger missing!)"
                     else:
                         # Should not happen if logic is correct (Naki without discard)
                         from_str = " from ? (Source unknown!)"
                 elif n_type == "加槓":
                     from_str = " (加槓)" # From self
                     # Trigger is internal, maybe show added tile if decode_naki provides it?
                 elif n_type == "暗槓":
                     from_str = " (暗槓)" # From self
                     # No external trigger
                 elif n_type == "不明":
                     from_str = " (Error decoding type)"

                 else:
                     from_str = f" (Type={n_type})"

                 # Check tile list length again for display consistency
                 expected_len = 4 if "槓" in n_type else 3
                 if n_type != "不明" and len(tiles_decoded) != expected_len:
                      tiles_str += f" [WARN: {len(tiles_decoded)} tiles]"

                 # Include meld_code (m) in output for debugging
                 print(f"    P{p} called {n_type}{from_str}{trigger_str}: [{tiles_str}] (m={meld_code})")
            # --- END MODIFIED NAKI DISPLAY ---


            # ... (Result printing remains the same) ...
            print(f"Result: {feat.get('result_type', 'N/A')}")
            if feat.get('result_type') == 'AGARI':
                winner = feat.get('winner', -1); loser = feat.get('loser', -1)
                winner_str = f"P{winner}" if winner != -1 else "?"; loser_str = f"P{loser}" if loser != -1 else "Tsumo"
                print(f"  Winner: {winner_str}, From: {loser_str}")
                print(f"  Score: {feat.get('agari_points', '?')}pts, Fu: {feat.get('agari_fu', '?')}")
            elif feat.get('result_type') == 'RYUUKYOKU':
                 print(f"  Type: {feat.get('ryuukyoku_type', 'unknown')}")
                 if "tenpai_status" in feat: tenpai_players = [f"P{i}" for i, status in enumerate(feat["tenpai_status"]) if status]; print(f"  Tenpai: {', '.join(tenpai_players) if tenpai_players else 'None'}")
            if feat.get('final_scores'): print(f"Final Scores: {feat.get('final_scores')}"); print(f"Score Changes: {feat.get('score_changes')}")

        except Exception as e:
            print(f"!! Error processing round {r_idx + 1}: {e}")
            import traceback
            traceback.print_exc()