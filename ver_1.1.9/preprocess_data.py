# /ver_1.1.9/preprocess_data.py
import os
import numpy as np
from multiprocessing import Pool, cpu_count
import logging
from tqdm import tqdm
import sys
import time
import glob
import gc # Garbage collection
from typing import Tuple, List, Dict, Any # <<< --- ADD THIS IMPORT

# --- Ensure other modules can be imported ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path: sys.path.insert(0, script_dir)
# ---------------------------------------------

try:
    from full_mahjong_parser import parse_full_mahjong_log
    # GameState imports tile_utils and naki_utils internally
    from game_state import GameState, NUM_PLAYERS, STATIC_FEATURE_DIM, MAX_EVENT_HISTORY, EVENT_TYPES
    from tile_utils import tile_id_to_index # Import directly if needed here
    print("Successfully imported helper modules.")
except ImportError as e:
    print(f"[FATAL ERROR in preprocess_data.py] Failed to import modules: {e}")
    print("Ensure full_mahjong_parser.py, game_state.py, tile_utils.py, naki_utils.py are accessible.")
    sys.exit(1)

# --- Configuration ---
XML_LOG_DIR = "/home/ubuntu/Documents/xml_logs/"        # <<< ADJUST: Directory containing Tenhou XML logs
OUTPUT_DIR = "./training_data/"     # Directory to save the processed NPZ files
NUM_PROCESSES = max(1, cpu_count() // 2) # Use half the CPU cores, minimum 1
FILES_PER_BATCH = 50        # Number of XML files processed before saving an NPZ batch
OUTPUT_NPZ_PREFIX = "mahjong_imitation_data_v119" # Prefix for saved NPZ files
SKIP_EXISTING_BATCHES = True # Set to False to re-process all XMLs
# -------------------------------------------

# --- Logging Setup ---
LOG_FILE = "data_processing_v119.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s/%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)
# ---------------------

# Note: The function signature below uses the imported 'Tuple', 'List'
def extract_features_for_file(xml_path: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Processes a single XML log file and extracts (sequence, static, label) triples.
    Uses the GameState class to manage state and generate features.
    """
    filename = os.path.basename(xml_path)
    file_sequences = []
    file_static_features = []
    file_labels = []
    last_decision_points = {} # Store state keyed by player_id {player_id: {"sequence": ..., "static": ...}}

    try:
        meta, rounds_data = parse_full_mahjong_log(xml_path)
        if not rounds_data:
            logging.debug(f"No rounds found in {filename}.")
            return [], [], []

        game_state = GameState() # Create GameState instance for this file

        for round_idx, round_data in enumerate(rounds_data):
            try:
                game_state.init_round(round_data)
                events = round_data.get("events", [])
                if not events: continue # Skip round if no events

                last_decision_points.clear() # Reset for new round

                for event in events:
                    tag = event["tag"]
                    attrib = event["attrib"]
                    processed_by_gamestate = False # Flag if GameState handled this

                    # --- Tsumo Event ---
                    tsumo_player_id = -1
                    tsumo_pai_id = -1
                    is_tsumo = False
                    for t_tag, p_id in GameState.TSUMO_TAGS.items():
                        if tag.startswith(t_tag) and tag[1:].isdigit():
                             is_tsumo = True; tsumo_player_id = p_id
                             try: tsumo_pai_id = int(tag[1:])
                             except ValueError: is_tsumo = False; break # Invalid tile ID
                             break # Found Tsumo tag

                    if is_tsumo:
                        try:
                            # Get features BEFORE processing tsumo
                            current_sequence = game_state.get_event_sequence_features()
                            # Process tsumo AFTER getting sequence features
                            game_state.process_tsumo(tsumo_player_id, tsumo_pai_id)
                            processed_by_gamestate = True
                            # Get static features AFTER processing tsumo
                            current_static = game_state.get_static_features(tsumo_player_id)
                            # Store state for the upcoming discard decision
                            last_decision_points[tsumo_player_id] = {
                                "sequence": current_sequence,
                                "static": current_static,
                            }
                        except Exception as e:
                            logging.warning(f"Error during Tsumo processing/feature extraction in {filename}, Round {round_idx+1}, Tag {tag}: {e}", exc_info=False)
                            last_decision_points.pop(tsumo_player_id, None) # Clear invalid state
                        continue # Move to next event

                    # --- Discard Event ---
                    discard_player_id = -1
                    discard_pai_id = -1
                    tsumogiri = False
                    is_discard = False
                    for d_tag, p_id in GameState.DISCARD_TAGS.items():
                        if tag.startswith(d_tag) and tag[1:].isdigit():
                            is_discard = True; discard_player_id = p_id
                            try: discard_pai_id = int(tag[1:])
                            except ValueError: is_discard = False; break # Invalid tile ID
                            tsumogiri = tag[0].islower()
                            break # Found Discard tag

                    if is_discard:
                        try:
                            # Check if this discard corresponds to a stored decision point
                            if discard_player_id in last_decision_points:
                                decision_state = last_decision_points.pop(discard_player_id) # Use and remove
                                label = tile_id_to_index(discard_pai_id)
                                if label != -1 and decision_state["sequence"] is not None and decision_state["static"] is not None:
                                    # Check dimensions before appending
                                    # Adjust expected event dim if _add_event changed (e.g., 6 -> 8)
                                    expected_event_dim = 6 # Check this value based on _add_event in game_state.py
                                    if decision_state["sequence"].shape == (MAX_EVENT_HISTORY, expected_event_dim) and \
                                       decision_state["static"].shape == (STATIC_FEATURE_DIM,):
                                        file_sequences.append(decision_state["sequence"])
                                        file_static_features.append(decision_state["static"])
                                        file_labels.append(label)
                                    else:
                                        logging.warning(f"Dimension mismatch skipped sample in {filename}. Seq: {decision_state['sequence'].shape} (expected {MAX_EVENT_HISTORY},{expected_event_dim}), Static: {decision_state['static'].shape} (expected {STATIC_FEATURE_DIM})")
                            # Process the discard AFTER pairing features/label
                            game_state.process_discard(discard_player_id, discard_pai_id, tsumogiri)
                            processed_by_gamestate = True
                        except Exception as e:
                            logging.warning(f"Error during Discard processing/feature pairing in {filename}, Round {round_idx+1}, Tag {tag}: {e}", exc_info=False)
                        continue # Move to next event

                    # --- Other Events (Naki, Reach, Dora, End Round) ---
                    event_processed_by_other = False
                    if tag == "N":
                        naki_player_id = int(attrib.get("who", -1)); meld_code = int(attrib.get("m", "0"))
                        if naki_player_id != -1: game_state.process_naki(naki_player_id, meld_code); last_decision_points.clear(); event_processed_by_other = True
                    elif tag == "REACH":
                         reach_player_id = int(attrib.get("who", -1)); step = int(attrib.get("step", -1))
                         if reach_player_id != -1 and step == 1: game_state.process_reach(reach_player_id, step); event_processed_by_other = True
                         elif step == 2: event_processed_by_other = True # Handled by discard
                    elif tag == "DORA":
                         hai_attr = attrib.get("hai")
                         if hai_attr is not None and hai_attr.isdigit(): game_state.process_dora(int(hai_attr)); event_processed_by_other = True
                    elif tag == "AGARI":
                         game_state.process_agari(attrib); last_decision_points.clear(); event_processed_by_other = True; break # End round
                    elif tag == "RYUUKYOKU":
                         game_state.process_ryuukyoku(attrib); last_decision_points.clear(); event_processed_by_other = True; break # End round

            except Exception as e:
                logging.warning(f"Error processing round {round_idx+1} in {filename}: {e}", exc_info=False)
                continue # Skip to next round

        # Cleanup after processing file
        del game_state, rounds_data, meta
        gc.collect()

        if file_labels:
            # logging.debug(f"Extracted {len(file_labels)} samples from {filename}.") # Reduce log noise
            return file_sequences, file_static_features, file_labels
        else:
            # logging.debug(f"No samples extracted from {filename}.") # Reduce log noise
            return [], [], []

    except Exception as e:
        logging.error(f"Critical error processing file {filename}: {e}", exc_info=True)
        return [], [], []

# (Keep save_batch and main functions as they were in the previous response)
# ... rest of the save_batch and main functions ...

def save_batch(batch_data, batch_idx):
    """Saves collected batch data to an NPZ file."""
    if not batch_data["labels"]:
        logging.info(f"Skipping empty batch {batch_idx}.")
        return 0

    output_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NPZ_PREFIX}_batch_{batch_idx}.npz")
    try:
        sequences_np = np.array(batch_data["sequences"], dtype=np.float32)
        static_np = np.array(batch_data["static"], dtype=np.float32)
        labels_np = np.array(batch_data["labels"], dtype=np.int64)

        # Final dimension check before saving
        # Check expected event dim based on _add_event in game_state.py
        expected_event_dim = 6 # Adjust if game_state._add_event changed
        if sequences_np.ndim != 3 or static_np.ndim != 2 or labels_np.ndim != 1:
             raise ValueError("Incorrect array dimensions before saving.")
        if sequences_np.shape[1:] != (MAX_EVENT_HISTORY, expected_event_dim):
             raise ValueError(f"Incorrect sequence dimensions: {sequences_np.shape} (expected {(..., MAX_EVENT_HISTORY, expected_event_dim)})")
        if static_np.shape[1] != STATIC_FEATURE_DIM:
             raise ValueError(f"Incorrect static dimensions: {static_np.shape} (expected {(..., STATIC_FEATURE_DIM)})")

        np.savez_compressed(
            output_path,
            sequences=sequences_np,
            static_features=static_np, # Use standard name
            labels=labels_np
        )
        num_samples = len(labels_np)
        logging.info(f"Saved batch {batch_idx} ({num_samples} samples) to {output_path}")
        return num_samples
    except ValueError as e:
        logging.error(f"Error creating NumPy arrays for batch {batch_idx}: {e}")
        logging.error(f"  Shapes - Seq: {len(batch_data['sequences'])}, Static: {len(batch_data['static'])}, Labels: {len(batch_data['labels'])}")
        if batch_data['sequences']: logging.error(f"  First Seq Shape: {batch_data['sequences'][0].shape}")
        if batch_data['static']: logging.error(f"  First Static Shape: {batch_data['static'][0].shape}")
        return 0
    except Exception as e:
        logging.error(f"Error saving batch {batch_idx} to {output_path}: {e}")
        return 0

def main():
    """Main function to orchestrate XML processing."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.info(f"Created output directory: {OUTPUT_DIR}")

    if not os.path.isdir(XML_LOG_DIR):
         logging.error(f"XML log directory not found: {XML_LOG_DIR}")
         sys.exit(1)

    xml_files = sorted(glob.glob(os.path.join(XML_LOG_DIR, "*.xml"))) # Ensure sorting
    total_files = len(xml_files)
    logging.info(f"Found {total_files} XML files in {XML_LOG_DIR}")

    if total_files == 0:
        logging.warning("No XML files found. Exiting.")
        return

    start_batch_idx = 0
    if SKIP_EXISTING_BATCHES:
        existing_batches = glob.glob(os.path.join(OUTPUT_DIR, f"{OUTPUT_NPZ_PREFIX}_batch_*.npz"))
        if existing_batches:
            try:
                last_batch_num = max(int(f.split('_batch_')[-1].split('.npz')[0]) for f in existing_batches)
                start_batch_idx = last_batch_num + 1
                logging.info(f"Found existing batches up to {last_batch_num}. Starting from batch {start_batch_idx}.")
            except ValueError:
                logging.warning("Could not determine last batch number from filenames. Starting from batch 0.")

    files_to_process = xml_files[(start_batch_idx * FILES_PER_BATCH):]
    num_files_to_process = len(files_to_process)
    logging.info(f"Processing {num_files_to_process} XML files (starting from file index {start_batch_idx * FILES_PER_BATCH}).")

    if num_files_to_process == 0:
        logging.info("No new files to process based on existing batches. Exiting.")
        return

    total_samples_processed = 0
    start_time_all = time.time()

    logging.info(f"Starting data extraction with {NUM_PROCESSES} processes...")
    with Pool(processes=NUM_PROCESSES) as pool:
        batch_data = {"sequences": [], "static": [], "labels": []}
        batch_file_count = 0
        current_batch_idx = start_batch_idx

        results_iterator = pool.imap_unordered(extract_features_for_file, files_to_process)

        for i, result in enumerate(tqdm(results_iterator, total=num_files_to_process, desc="Processing XML Files")):
            seq_list, static_list, label_list = result
            if label_list: # Only add if data was extracted
                batch_data["sequences"].extend(seq_list)
                batch_data["static"].extend(static_list)
                batch_data["labels"].extend(label_list)
            batch_file_count += 1

            if batch_file_count >= FILES_PER_BATCH or (i + 1) == num_files_to_process:
                num_saved = save_batch(batch_data, current_batch_idx)
                total_samples_processed += num_saved
                batch_data = {"sequences": [], "static": [], "labels": []}
                batch_file_count = 0
                current_batch_idx += 1
                gc.collect() # Force garbage collection between saving batches

    end_time_all = time.time()
    logging.info("="*30)
    logging.info("Data Processing Complete.")
    logging.info(f"Total XML files processed in this run: {num_files_to_process}")
    logging.info(f"Total samples extracted in this run: {total_samples_processed}")
    logging.info(f"Saved data up to batch index {current_batch_idx - 1} in: {OUTPUT_DIR}")
    logging.info(f"Total time: {end_time_all - start_time_all:.2f} seconds")
    logging.info(f"Log file saved to: {LOG_FILE}")
    logging.info("Next step: Run train.py to train the model.")
    logging.info("="*30)

if __name__ == "__main__":
    main()