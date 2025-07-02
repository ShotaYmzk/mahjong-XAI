# /ver_1.1.9/full_mahjong_parser.py
import xml.etree.ElementTree as ET
import urllib.parse
import os
import sys
from typing import List, Dict, Any, Tuple

# --- Path setup for potential utility imports if needed later ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path: sys.path.insert(0, script_dir)
# --- End Path setup ---

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
        print(f"[Parser Error] Failed to parse XML file: {xml_path} - {e}")
        return {}, []
    except FileNotFoundError:
        print(f"[Parser Error] XML file not found: {xml_path}")
        return {}, []
    except Exception as e:
        print(f"[Parser Error] Unexpected error reading XML file {xml_path}: {str(e)}")
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
                        print(f"[Parser Warning] Could not decode player name {attrib[name_key]}: {e}")
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
                  if not current_round_data.get("result"):
                       print(f"[Parser Warning] Encountered <Owari> tag before explicit round end in round {current_round_data.get('round_index')}")
                       current_round_data["result"] = {"tag": "Owari", "attrib": attrib}
                  current_round_data = None # End processing after Owari

    # Add player names to meta if not already present from UN tag
    if 'player_names' not in meta:
         meta['player_names'] = [player_name_map.get(i, f'p{i}') for i in range(4)]

    # print(f"Parsed {len(rounds)} rounds from {os.path.basename(xml_path)}.") # Moved to calling function
    return meta, rounds

# --- Example Usage (for testing this script directly) ---
# (Keep the __main__ block from the previous version if you want to test this file alone)