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
import h5py # HDF5 ファイル操作用 ★追加
from typing import Tuple, List, Dict, Any # <<< --- ADD THIS IMPORT

# --- Ensure other modules can be imported ---
# このスクリプトが実行されるディレクトリをPythonの検索パスに追加
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
# ---------------------------------------------

try:
    # 同じディレクトリにある他のモジュールをインポート
    from full_mahjong_parser import parse_full_mahjong_log
    # GameState imports tile_utils and naki_utils internally
    from game_state import GameState, NUM_PLAYERS, STATIC_FEATURE_DIM, MAX_EVENT_HISTORY, EVENT_TYPES
    from tile_utils import tile_id_to_index, tile_id_to_string # Import directly if needed here
    logging.info("Successfully imported helper modules.")
except ImportError as e:
    logging.critical(f"[FATAL ERROR in preprocess_data.py] Failed to import modules: {e}")
    logging.critical("Ensure full_mahjong_parser.py, game_state.py, tile_utils.py, naki_utils.py are in the same directory.")
    sys.exit(1)

# --- Configuration ---
XML_LOG_DIR = "/home/ubuntu/Documents/XML/xml_logs" # <<< ADJUST: Directory containing Tenhou XML logs
OUTPUT_DIR = "./training_data/"     # Directory to save the processed HDF5 file
OUTPUT_HDF5_FILENAME = "mahjong_imitation_data_v1110.hdf5" # ★変更点: 出力HDF5ファイル名
OUTPUT_HDF5_PATH = os.path.join(OUTPUT_DIR, OUTPUT_HDF5_FILENAME) # ★変更点: 出力HDF5ファイルのフルパス

NUM_PROCESSES = max(1, cpu_count() // 2) # Use half the CPU cores, minimum 1
FILES_PER_HDF5_WRITE = 50 # ★変更点: NPZバッチからHDF5書き込みトリガーへ。何ファイル処理ごとにHDF5に追記するか
# SKIP_EXISTING_BATCHES = True # NPZバッチ用だったのでHDF5では不要（上書き）

# -------------------------------------------

# --- Logging Setup ---
LOG_FILE = "data_processing_v119.log"
# logging.basicConfig は一度しか設定できないため、既に設定されている場合はスキップ
if not logging.getLogger('').handlers:
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
    Returns lists of numpy arrays and labels.
    This function runs in a separate process.
    """
    filename = os.path.basename(xml_path)
    file_sequences = []
    file_static_features = []
    file_labels = []
    # Store state keyed by player_id for the decision point *before* discard
    # {player_id: {"sequence": np.ndarray, "static": np.ndarray, "event_index": int}}
    last_decision_points: Dict[int, Dict[str, Any]] = {}

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

                for event_index, event in enumerate(events):
                    tag = event["tag"]
                    attrib = event["attrib"]

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
                            # Process tsumo
                            game_state.process_tsumo(tsumo_player_id, tsumo_pai_id)

                            # Get features *after* tsumo (this is the state where discard decision is made)
                            current_sequence = game_state.get_event_sequence_features()
                            current_static = game_state.get_static_features(tsumo_player_id)

                            # Store state for the upcoming discard decision
                            last_decision_points[tsumo_player_id] = {
                                "sequence": current_sequence,
                                "static": current_static,
                                "event_index": event_index # Store index for debugging
                            }
                            # logging.debug(f"R{round_idx+1} E{event_index+1} (<{tag}>): Stored decision point for P{tsumo_player_id}")

                        except Exception as e:
                            # Log the error but allow processing of other players/rounds in the same file
                            logging.warning(f"Error during Tsumo processing/feature extraction in {filename}, Round {round_idx+1}, Event {event_index+1} (<{tag}>): {e}", exc_info=False)
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

                                if label == -1:
                                     logging.warning(f"R{round_idx+1} E{event_index+1} (<{tag}>): Invalid discard tile ID {discard_pai_id}. Skipping sample.")
                                elif decision_state["sequence"] is None or decision_state["static"] is None:
                                     logging.warning(f"R{round_idx+1} E{event_index+1} (<{tag}>): Stored decision state was None. Skipping sample.")
                                else:
                                    # Check dimensions before appending
                                    # The event_total_dim is calculated dynamically in GameState.get_event_sequence_features
                                    # We need to get the expected dim from a dummy call or GameState instance
                                    # A safer way is to check the shape against the constant STATIC_FEATURE_DIM
                                    # and the sequence length MAX_EVENT_HISTORY, and assume the event dim is consistent.
                                    expected_seq_shape = (MAX_EVENT_HISTORY, decision_state["sequence"].shape[1]) # Check only length, trust event dim from GS
                                    expected_static_shape = (STATIC_FEATURE_DIM,)

                                    if decision_state["sequence"].shape == expected_seq_shape and \
                                       decision_state["static"].shape == expected_static_shape:
                                        file_sequences.append(decision_state["sequence"])
                                        file_static_features.append(decision_state["static"])
                                        file_labels.append(label)
                                        # logging.debug(f"R{round_idx+1} E{event_index+1} (<{tag}>): Added sample for P{discard_player_id} (Label: {label}, {tile_id_to_string(discard_pai_id)})")
                                    else:
                                        logging.warning(f"R{round_idx+1} E{event_index+1} (<{tag}>): Dimension mismatch skipped sample for P{discard_player_id}. "
                                                        f"Seq: {decision_state['sequence'].shape} (expected {expected_seq_shape}), "
                                                        f"Static: {decision_state['static'].shape} (expected {expected_static_shape})")
                            else:
                                # This discard did not follow a Tsumo event for this player where we stored state.
                                # This can happen after Naki, or if the previous Tsumo event was skipped due to error.
                                # logging.debug(f"R{round_idx+1} E{event_index+1} (<{tag}>): No decision point found for P{discard_player_id}. Skipping sample extraction for this discard.")
                                pass # This is expected behavior for discards not immediately after a Tsumo

                            # Process the discard AFTER pairing features/label
                            game_state.process_discard(discard_player_id, discard_pai_id, tsumogiri)
                        except Exception as e:
                            logging.warning(f"Error during Discard processing/feature pairing in {filename}, Round {round_idx+1}, Event {event_index+1} (<{tag}>): {e}", exc_info=False)
                        continue # Move to next event

                    # --- Other Events (Naki, Reach, Dora, End Round) ---
                    # These events clear any pending decision points as the turn structure changes
                    if tag == "N":
                        naki_player_id = int(attrib.get("who", -1)); meld_code = int(attrib.get("m", "0"))
                        if naki_player_id != -1:
                             try:
                                game_state.process_naki(naki_player_id, meld_code)
                                # Naki changes turn order, clear pending discard decisions
                                last_decision_points.clear()
                                # logging.debug(f"R{round_idx+1} E{event_index+1} (<{tag}>): Processed Naki by P{naki_player_id}. Cleared decision points.")
                             except Exception as e:
                                logging.warning(f"Error processing Naki in {filename}, Round {round_idx+1}, Event {event_index+1} (<{tag}>): {e}", exc_info=False)
                                last_decision_points.clear() # Clear state on Naki error
                    elif tag == "REACH":
                         reach_player_id = int(attrib.get("who", -1)); step = int(attrib.get("step", -1))
                         if reach_player_id != -1 and step == 1:
                             try:
                                game_state.process_reach(reach_player_id, step)
                                # Reach step 1 is declaration, discard follows immediately, state is captured *before* discard
                                # No need to clear decision points here, the discard event will use it.
                                # logging.debug(f"R{round_idx+1} E{event_index+1} (<{tag}>): Processed Reach step 1 by P{reach_player_id}.")
                             except Exception as e:
                                logging.warning(f"Error processing Reach step 1 in {filename}, Round {round_idx+1}, Event {event_index+1} (<{tag}>): {e}", exc_info=False)
                         elif step == 2:
                             # Reach step 2 is acceptance, happens *after* the reach discard.
                             # This event itself doesn't create a new decision point or change turn structure unexpectedly.
                             # It's handled implicitly by the discard event that completes the reach.
                             # logging.debug(f"R{round_idx+1} E{event_index+1} (<{tag}>): Processed Reach step 2.")
                             pass # Handled by discard
                    elif tag == "DORA":
                         hai_attr = attrib.get("hai")
                         if hai_attr is not None and hai_attr.isdigit():
                             try:
                                game_state.process_dora(int(hai_attr))
                                # Dora reveal doesn't change turn structure or create a discard decision point
                                # logging.debug(f"R{round_idx+1} E{event_index+1} (<{tag}>): Processed Dora.")
                             except Exception as e:
                                logging.warning(f"Error processing Dora in {filename}, Round {round_idx+1}, Event {event_index+1} (<{tag}>): {e}", exc_info=False)
                    elif tag == "AGARI":
                         try:
                             game_state.process_agari(attrib)
                             last_decision_points.clear() # Round ends, clear any pending decisions
                             # logging.debug(f"R{round_idx+1} E{event_index+1} (<{tag}>): Processed Agari. Cleared decision points. Ending round.")
                         except Exception as e:
                             logging.warning(f"Error processing Agari in {filename}, Round {round_idx+1}, Event {event_index+1} (<{tag}>): {e}", exc_info=False)
                         break # End round processing
                    elif tag == "RYUUKYOKU":
                         try:
                             game_state.process_ryuukyoku(attrib)
                             last_decision_points.clear() # Round ends, clear any pending decisions
                             # logging.debug(f"R{round_idx+1} E{event_index+1} (<{tag}>): Processed Ryuukyoku. Cleared decision points. Ending round.")
                         except Exception as e:
                             logging.warning(f"Error processing Ryuukyoku in {filename}, Round {round_idx+1}, Event {event_index+1} (<{tag}>): {e}", exc_info=False)
                         break # End round processing
                    # else:
                        # logging.debug(f"R{round_idx+1} E{event_index+1} (<{tag}>): Unhandled or non-state-changing event.")


            except Exception as e:
                # This catches errors within a specific round's event loop, after init_round
                logging.warning(f"Error processing events in round {round_idx+1} of {filename}: {e}", exc_info=False)
                last_decision_points.clear() # Clear state on round error
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
        # This catches errors during parse_full_mahjong_log or init_round
        logging.error(f"Critical error processing file {filename}: {e}", exc_info=True)
        return [], [], []

def append_to_hdf5(hdf5_file: h5py.File, sequences: List[np.ndarray], static_features: List[np.ndarray], labels: List[int]):
    """Appends data to the HDF5 datasets."""
    if not labels: # Nothing to append
        return 0

    num_new_samples = len(labels)
    current_size = hdf5_file['labels'].shape[0]
    new_size = current_size + num_new_samples

    try:
        # Resize datasets
        hdf5_file['sequences'].resize((new_size, hdf5_file['sequences'].shape[1], hdf5_file['sequences'].shape[2]))
        hdf5_file['static_features'].resize((new_size, hdf5_file['static_features'].shape[1]))
        hdf5_file['labels'].resize((new_size,))

        # Append data
        hdf5_file['sequences'][current_size:new_size] = np.array(sequences, dtype=np.float32)
        hdf5_file['static_features'][current_size:new_size] = np.array(static_features, dtype=np.float32)
        hdf5_file['labels'][current_size:new_size] = np.array(labels, dtype=np.int64)

        # Ensure data is written to disk (important for SWMR read)
        hdf5_file.flush()

        return num_new_samples
    except Exception as e:
        logging.error(f"Error appending {num_new_samples} samples to HDF5 file: {e}", exc_info=True)
        # Attempt to revert resize if possible (complex, maybe just log error)
        # For simplicity, we just log and return 0, assuming the file might be corrupted from this point.
        # A more robust solution might involve temporary files or more complex HDF5 handling.
        return 0


def main():
    """Main function to orchestrate XML processing and HDF5 saving."""
    logging.info("Starting data preprocessing to HDF5...")

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

    # --- Initialize HDF5 File ---
    # Delete existing file if it exists, to start fresh
    if os.path.exists(OUTPUT_HDF5_PATH):
        logging.warning(f"Existing HDF5 file found at {OUTPUT_HDF5_PATH}. Deleting and creating a new one.")
        try:
            os.remove(OUTPUT_HDF5_PATH)
        except Exception as e:
            logging.error(f"Failed to delete existing HDF5 file {OUTPUT_HDF5_PATH}: {e}")
            sys.exit(1)

    # Determine dimensions from the first successfully processed file
    # This requires processing at least one file sequentially first
    logging.info("Processing first XML file to determine data dimensions for HDF5...")
    first_seq_data, first_static_data, first_label_data = [], [], []
    first_file_processed = False
    first_xml_path = None

    # Iterate through files until one successfully yields data
    for xml_file in xml_files:
        temp_seq, temp_static, temp_labels = extract_features_for_file(xml_file)
        if temp_labels:
            first_seq_data = temp_seq
            first_static_data = temp_static
            first_label_data = temp_labels
            first_file_processed = True
            first_xml_path = xml_file # Store the path of the first processed file
            logging.info(f"Successfully processed {os.path.basename(xml_file)} to get dimensions ({len(temp_labels)} samples).")
            break # Found dimensions, exit loop

    if not first_file_processed:
        logging.error("Could not extract any samples from the provided XML files to determine data dimensions. Aborting.")
        sys.exit(1)

    # Get dimensions from the first sample
    # Assuming all samples will have consistent dimensions after processing
    if not first_seq_data or not first_static_data or not first_label_data:
         logging.error("Internal error: First file processing returned empty lists despite first_file_processed being True. Aborting.")
         sys.exit(1)

    sample_seq_shape = first_seq_data[0].shape # (MAX_EVENT_HISTORY, event_dim)
    sample_static_shape = first_static_data[0].shape # (STATIC_FEATURE_DIM,)
    sample_label_shape = () # label is scalar

    # Validate dimensions against constants
    if sample_seq_shape[0] != MAX_EVENT_HISTORY:
         logging.error(f"Dimension mismatch from first sample! Sequence length: {sample_seq_shape[0]} (expected {MAX_EVENT_HISTORY}). Aborting.")
         sys.exit(1)
    if sample_static_shape[0] != STATIC_FEATURE_DIM:
         logging.error(f"Dimension mismatch from first sample! Static dimension: {sample_static_shape[0]} (expected {STATIC_FEATURE_DIM}). Aborting.")
         sys.exit(1)
    # event_dim (sample_seq_shape[1]) is dynamic, no constant check needed here

    logging.info(f"Data dimensions determined: Seq shape {sample_seq_shape}, Static shape {sample_static_shape}, Label shape {sample_label_shape}")

    # Create HDF5 file and datasets with the first batch of data
    hdf5_file = None # Initialize to None
    try:
        # Use 'w' mode to create a new file (overwrites if exists)
        hdf5_file = h5py.File(OUTPUT_HDF5_PATH, 'w', libver='latest') # libver='latest' for SWMR compatibility
        # Create datasets with initial data and enable resizing (chunks=True)
        hdf5_file.create_dataset('sequences', data=np.array(first_seq_data, dtype=np.float32),
                                 maxshape=(None, sample_seq_shape[0], sample_seq_shape[1]), chunks=True, dtype=np.float32)
        hdf5_file.create_dataset('static_features', data=np.array(first_static_data, dtype=np.float32),
                                 maxshape=(None, sample_static_shape[0]), chunks=True, dtype=np.float32)
        hdf5_file.create_dataset('labels', data=np.array(first_label_data, dtype=np.int64),
                                 maxshape=(None,), chunks=True, dtype=np.int64)
        logging.info(f"Created HDF5 file and initial datasets at {OUTPUT_HDF5_PATH} with {len(first_label_data)} samples.")

        # Enable Single-Writer Multiple-Reader (SWMR) mode
        # This allows the training script to potentially read while writing is in progress,
        # although the MahjongHdf5Dataset opens/closes per __getitem__ which is safer.
        # SWMR is good practice for large, growing HDF5 files.
        hdf5_file.swmr_mode = True
        logging.info("HDF5 SWMR mode enabled.")

    except Exception as e:
        logging.error(f"Failed to create HDF5 file or initial datasets: {e}", exc_info=True)
        # Ensure file is closed if creation failed partially
        if hdf5_file:
            try:
                hdf5_file.close()
            except Exception as close_e:
                logging.error(f"Error closing HDF5 file after creation failure: {close_e}")
        sys.exit(1)

    # Prepare the list of files to process in the pool (all files EXCEPT the first one processed)
    files_to_process_in_pool = xml_files[xml_files.index(first_xml_path) + 1:]
    num_files_to_process_in_pool = len(files_to_process_in_pool)
    logging.info(f"Processing remaining {num_files_to_process_in_pool} XML files using multiprocessing pool.")

    total_samples_processed = len(first_label_data) # Start count with samples from the first file
    start_time_all = time.time()

    if num_files_to_process_in_pool > 0:
        logging.info(f"Starting data extraction with {NUM_PROCESSES} processes...")
        # Use imap_unordered for results as they complete
        # Create the Pool within the scope where it's used
        with Pool(processes=NUM_PROCESSES) as pool:
            results_iterator = pool.imap_unordered(extract_features_for_file, files_to_process_in_pool)

            # Collect data in the main process and append to HDF5 periodically
            collected_sequences = []
            collected_static_features = []
            collected_labels = []
            files_processed_in_batch = 0 # This counter is for the pool processing part

            try:
                # Iterate over results from the pool
                for i, result in enumerate(tqdm(results_iterator, total=num_files_to_process_in_pool, desc="Processing XML Files")):
                    seq_list, static_list, label_list = result
                    if label_list: # Only add if data was extracted
                        # Optional: Add dimension check here again for safety, though it should be consistent
                        # if seq_list[0].shape != sample_seq_shape or static_list[0].shape != sample_static_shape:
                        #     logging.warning(f"Dimension mismatch in pool result for file {files_to_process_in_pool[i]}! Skipping data from this file.")
                        #     continue # Skip data from this file if dimensions are wrong

                        collected_sequences.extend(seq_list)
                        collected_static_features.extend(static_list)
                        collected_labels.extend(label_list)

                    files_processed_in_batch += 1

                    # Append to HDF5 if enough files processed or it's the last result
                    if files_processed_in_batch >= FILES_PER_HDF5_WRITE or (i + 1) == num_files_to_process_in_pool:
                        num_appended = append_to_hdf5(hdf5_file, collected_sequences, collected_static_features, collected_labels)
                        total_samples_processed += num_appended
                        # Reset collected data for the next batch
                        collected_sequences = []
                        collected_static_features = []
                        collected_labels = []
                        files_processed_in_batch = 0
                        gc.collect() # Force garbage collection

            except Exception as e:
                # This catches errors during the iteration over pool results
                logging.error(f"Critical error during multiprocessing pool processing: {e}", exc_info=True)
                # Re-raise the exception to be caught by the outer finally block
                raise

        # The Pool is automatically closed by the 'with' statement here

    else:
        logging.info("No more XML files to process after the first one.")


    # Ensure the HDF5 file is closed
    # Use a finally block to ensure closing even if errors occurred
    try:
        # Append any remaining collected data before closing (if pool was used and data is left)
        if collected_labels: # Check if there's data left after the loop
             num_appended = append_to_hdf5(hdf5_file, collected_sequences, collected_static_features, collected_labels)
             total_samples_processed += num_appended
             logging.info(f"Appended remaining {num_appended} samples before closing HDF5.")

        if hdf5_file: # Check if the file object was successfully created
            hdf5_file.close()
            logging.info("HDF5 file closed.")
        else:
            logging.warning("HDF5 file object was not created successfully, nothing to close.")

    except Exception as e:
        logging.error(f"Error closing HDF5 file in finally block: {e}")


    end_time_all = time.time()
    logging.info("="*30)
    logging.info("Data Processing Complete.")
    logging.info(f"Total XML files processed in this run: {total_files}") # Total files including the first one
    logging.info(f"Total samples extracted and saved to HDF5: {total_samples_processed}")
    logging.info(f"Output HDF5 file: {OUTPUT_HDF5_PATH}")
    logging.info(f"Total time: {end_time_all - start_time_all:.2f} seconds")
    logging.info(f"Log file saved to: {LOG_FILE}")
    logging.info("Next step: Run train.py to train the model using the HDF5 file.")
    logging.info("="*30)

if __name__ == "__main__":
    # It's generally recommended to protect the main execution block
    # when using multiprocessing on some platforms (like Windows).
    # This is already done by the `if __name__ == "__main__":` block.
    main()