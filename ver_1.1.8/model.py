# model.py (Revised for advanced Transformer-based architecture)
import os
import numpy as np
from multiprocessing import Pool, cpu_count
import logging
from tqdm import tqdm
import sys
from naki_utils import tile_to_name, tile_id_to_index, decode_naki
import time

# --- Ensure other modules can be imported ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path: sys.path.insert(0, script_dir)
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path: sys.path.insert(0, parent_dir)
# ---------------------------------------------

try:
    from full_mahjong_parser import parse_full_mahjong_log
    # Import the feature extraction function
    from feature_extractor_imitation import extract_features_labels_for_imitation
    # Import GameState class itself
    from game_state import GameState
except ImportError as e:
    print(f"[FATAL ERROR in model.py] Failed to import modules: {e}")
    sys.exit(1)

# --- Configuration ---
# !!! ADJUST THESE PATHS FOR YOUR ENVIRONMENT !!!
XML_LOG_DIR = "../xml_logs/"  # Directory containing your Tenhou XML logs
OUTPUT_DIR = "./training_data/" # Directory to save the processed NPZ files
# -------------------------------------------
NUM_PROCESSES = 2  # Using fewer processes to reduce memory usage
BATCH_SIZE = 10    # Smaller batch size to reduce memory footprint
OUTPUT_NPZ_PREFIX = "mahjong_imitation_data" # Prefix for saved NPZ files
DEBUG_MODE = False # Set to True for more verbose logging, False to filter warnings

# --- Memory Management Settings ---
FORCE_GC_BETWEEN_BATCHES = True  # Force garbage collection between batches
DISABLE_AUGMENTATION = True      # Disable augmentation to save memory
SKIP_ALREADY_PROCESSED = True    # Skip files that already have data in output folder
# -------------------------------------------

# --- Advanced Settings for Data Quality ---
FILTER_INCOMPLETE_GAMES = True  # Skip games that don't have complete data
FILTER_LOW_RANK_GAMES = True    # Only use high-ranked player games
MIN_PLAYER_RANK = 15            # Minimum player rank (higher is better)
DISCARD_AUGMENTATION = False if DISABLE_AUGMENTATION else True  # Enable data augmentation for discards
AUGMENTATION_FACTOR = 2         # Number of augmentations per original example
# -------------------------------------------

# --- Logging Setup ---
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO, # DEBUG level if DEBUG_MODE is True
    format="%(asctime)s [%(processName)s/%(levelname)s] %(message)s", # Include process name
    handlers=[
        logging.FileHandler("data_processing.log", mode='w'), # Overwrite log each run
        logging.StreamHandler()
    ]
)

# --- Warning Filter (Optional) ---
class WarningFilter(logging.Filter):
    def __init__(self, patterns): self.patterns = patterns; super().__init__()
    def filter(self, record): return not any(p in record.getMessage() for p in self.patterns)

# Filter common warnings if not in debug mode
if not DEBUG_MODE:
    # FILTERED_WARNINGS = ["Using dummy shanten calculation", "Shanten calculation failed"] # Example
    FILTERED_WARNINGS = [] # Add patterns here if needed
    warning_filter = WarningFilter(FILTERED_WARNINGS)
    for handler in logging.getLogger().handlers: handler.addFilter(warning_filter)
# ----------------------------------

def augment_discard_data(sequences, static_features, labels, factor=AUGMENTATION_FACTOR):
    """
    Augments training data by creating variations of existing examples.
    
    This can include:
    1. Small random noise to numeric features
    2. Shuffling equivalent positions in the data
    3. Creating mirror positions where applicable
    
    Returns augmented data arrays.
    """
    if not DISCARD_AUGMENTATION or factor <= 1:
        return sequences, static_features, labels
    
    aug_sequences = [sequences]
    aug_static = [static_features]
    aug_labels = [labels]
    
    # Create augmented copies
    for i in range(factor - 1):
        # Create a copy with small random noise (±2%)
        seq_noise = sequences * np.random.uniform(0.98, 1.02, sequences.shape)
        static_noise = static_features * np.random.uniform(0.98, 1.02, static_features.shape)
        
        # Only add noise to non-zero elements to preserve structure
        seq_noise = np.where(sequences != 0, seq_noise, sequences)
        static_noise = np.where(static_features != 0, static_noise, static_features)
        
        aug_sequences.append(seq_noise)
        aug_static.append(static_noise)
        aug_labels.append(labels)
    
    # Concatenate all augmented data
    return np.concatenate(aug_sequences), np.concatenate(aug_static), np.concatenate(aug_labels)

def filter_high_quality_games(meta_data):
    """
    Filters games based on quality criteria like player ranks.
    Returns True if the game meets quality standards.
    """
    if not FILTER_LOW_RANK_GAMES:
        return True
        
    try:
        # Get player ranks from meta data (format depends on your parser)
        ranks = meta_data.get('ranks', [])
        if not ranks:
            return True  # If rank info is missing, include by default
            
        # Check if at least 2 players have the minimum rank
        high_rank_players = sum(1 for rank in ranks if rank >= MIN_PLAYER_RANK)
        return high_rank_players >= 2
    except Exception as e:
        logging.warning(f"Error filtering by rank: {e}")
        return True  # Include in case of errors

def process_xml_file(xml_path):
    """Processes a single XML log file to extract features and labels."""
    filename = os.path.basename(xml_path)
    all_round_sequences = []
    all_round_static = []
    all_round_labels = []
    
    # Memory management
    import gc
    
    try:
        logging.debug(f"Processing {filename}...")
        meta, rounds_data = parse_full_mahjong_log(xml_path)
        
        # Filter games based on quality criteria
        if FILTER_LOW_RANK_GAMES and not filter_high_quality_games(meta):
            logging.debug(f"Skipping {filename} due to low player ranks")
            # Clean up before returning
            del meta
            gc.collect()
            return None, None, None
            
        if not rounds_data:
            logging.debug(f"No rounds found in {filename}.")
            # Clean up before returning
            del meta
            gc.collect()
            return None, None, None

        game_state = GameState() # Create GameState instance for this file

        # Process each round with memory management
        for round_idx, round_data in enumerate(rounds_data):
            try:
                game_state.init_round(round_data) # Initialize state for the round
                # Call the extraction function, passing the initialized GameState
                sequences, static_features, labels = extract_features_labels_for_imitation(round_data, game_state)

                if sequences is not None and static_features is not None and labels is not None:
                    all_round_sequences.append(sequences)
                    all_round_static.append(static_features)
                    all_round_labels.append(labels)
                
                # Clean up each round to prevent memory leaks
                del round_data
                if round_idx % 5 == 0:  # Periodically collect garbage during round processing
                    gc.collect()
                    
            except Exception as e:
                logging.warning(f"Error processing round {round_idx} in {filename}: {e}")
                continue

        # Free memory from rounds_data
        del rounds_data
        del meta
        del game_state
        
        # Combine data from all rounds in this file
        if all_round_sequences:
            try:
                file_sequences = np.concatenate(all_round_sequences, axis=0)
                file_static = np.concatenate(all_round_static, axis=0)
                file_labels = np.concatenate(all_round_labels, axis=0)
                
                # Clean up individual round data
                del all_round_sequences
                del all_round_static
                del all_round_labels
                gc.collect()
                
                # Apply data augmentation
                if DISCARD_AUGMENTATION and AUGMENTATION_FACTOR > 1:
                    file_sequences, file_static, file_labels = augment_discard_data(
                        file_sequences, file_static, file_labels, AUGMENTATION_FACTOR
                    )
                    
                logging.debug(f"Extracted {len(file_labels)} samples from {filename}.")
                return file_sequences, file_static, file_labels
            except Exception as e:
                logging.error(f"Error concatenating data from {filename}: {e}")
                # Clean up on error
                del all_round_sequences
                del all_round_static
                del all_round_labels
                gc.collect()
                return None, None, None
        else:
            logging.debug(f"No samples extracted from {filename}.")
            # Clean up
            del all_round_sequences
            del all_round_static
            del all_round_labels
            gc.collect()
            return None, None, None

    except Exception as e:
        logging.error(f"Error processing {filename}: {e}", exc_info=False)
        # Clean up on error
        try:
            del all_round_sequences
            del all_round_static
            del all_round_labels
        except:
            pass
        gc.collect()
        return None, None, None

def process_batch(xml_batch, batch_idx):
    """Process a batch of XML files and save the results."""
    total_batches = len(xml_batch) // BATCH_SIZE + (1 if len(xml_batch) % BATCH_SIZE > 0 else 0)
    start_time = time.time()
    sequences_list = []
    static_list = []
    labels_list = []
    processed_files = 0
    skipped_files = 0
    
    # Track memory usage
    import psutil
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    try:
        logging.info(f"Batch {batch_idx+1}/{total_batches}: Processing {len(xml_batch)} files")
        
        # Process the batch data directly - xml_batch contains numpy arrays, not file paths
        if all(isinstance(item, np.ndarray) for item in xml_batch):
            # This is a list of numpy arrays, not file paths
            sequences_list = xml_batch
            
            # Save combined batch data
            try:
                all_sequences = np.concatenate(sequences_list, axis=0)
                
                # For static features and labels, use the corresponding data from main() function
                output_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NPZ_PREFIX}_batch_{batch_idx}.npz")
                np.savez_compressed(
                    output_path,
                    sequences=all_sequences
                )
                processed_files = len(sequences_list)
                logging.info(f"Saved batch data to {output_path} with {len(all_sequences)} samples")
            except Exception as e:
                logging.error(f"Error saving batch data: {e}")
        else:
            # Process as file paths (original implementation)
            for i, xml_path in enumerate(xml_batch):
                # Skip already processed files if enabled
                if SKIP_ALREADY_PROCESSED:
                    output_file = os.path.join(
                        OUTPUT_DIR, 
                        os.path.basename(xml_path).replace('.xml', '.npz')
                    )
                    if os.path.exists(output_file):
                        skipped_files += 1
                        continue
                
                # Process the file
                seq, static, labels = process_xml_file(xml_path)
                
                # Save individual file data separately for memory efficiency
                if seq is not None and static is not None and labels is not None:
                    try:
                        output_file = os.path.join(
                            OUTPUT_DIR, 
                            os.path.basename(xml_path).replace('.xml', '.npz')
                        )
                        np.savez_compressed(
                            output_file,
                            sequences=seq,
                            static=static,
                            labels=labels
                        )
                        processed_files += 1
                        
                        # Optional: don't accumulate all data in memory
                        if not FORCE_GC_BETWEEN_BATCHES:
                            # Just count that we processed it successfully
                            pass
                        else:
                            # Add to batch lists (higher memory usage)
                            sequences_list.append(seq)
                            static_list.append(static)
                            labels_list.append(labels)
                    except Exception as e:
                        logging.error(f"Error saving data for {xml_path}: {e}")
                
                # Progress reporting
                if (i + 1) % 10 == 0 or i == len(xml_batch) - 1:
                    elapsed = time.time() - start_time
                    files_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                    current_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_diff = current_memory - initial_memory
                    
                    logging.info(f"Batch {batch_idx+1}/{total_batches}: "
                                f"Processed {i+1}/{len(xml_batch)} files "
                                f"({files_per_sec:.2f} files/sec), "
                                f"Memory: {current_memory:.1f}MB ({memory_diff:+.1f}MB)")
                
                # Force garbage collection between files if enabled
                if FORCE_GC_BETWEEN_BATCHES and (i + 1) % 5 == 0:
                    import gc
                    gc.collect()
            
            # Save combined batch data if enabled
            if FORCE_GC_BETWEEN_BATCHES and sequences_list:
                try:
                    all_sequences = np.concatenate(sequences_list, axis=0)
                    all_static = np.concatenate(static_list, axis=0)
                    all_labels = np.concatenate(labels_list, axis=0)
                    
                    output_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NPZ_PREFIX}_batch_{batch_idx}.npz")
                    np.savez_compressed(
                        output_path,
                        sequences=all_sequences,
                        static=all_static,
                        labels=all_labels
                    )
                    logging.info(f"Saved batch data to {output_path} with {len(all_labels)} samples")
                except Exception as e:
                    logging.error(f"Error saving batch data: {e}")
        
        # Clear memory
        import gc
        del sequences_list
        del static_list
        del labels_list
        gc.collect()
        
        end_time = time.time()
        elapsed = end_time - start_time
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_change = final_memory - initial_memory
        
        logging.info(f"Batch {batch_idx+1}/{total_batches} completed in {elapsed:.2f}s. "
                    f"Processed {processed_files} files, skipped {skipped_files} files. "
                    f"Final memory: {final_memory:.1f}MB ({memory_change:+.1f}MB)")
                    
    except Exception as e:
        logging.error(f"Error processing batch {batch_idx}: {e}", exc_info=True)
        
        # Emergency memory cleanup
        import gc
        del sequences_list
        del static_list
        del labels_list
        gc.collect()

def main():
    """Main function to orchestrate XML processing."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.info(f"Created output directory: {OUTPUT_DIR}")

    if not os.path.isdir(XML_LOG_DIR):
         logging.error(f"XML log directory not found: {XML_LOG_DIR}")
         sys.exit(1)

    xml_files = [os.path.join(XML_LOG_DIR, f) for f in os.listdir(XML_LOG_DIR) if f.lower().endswith(".xml")]
    total_files = len(xml_files)
    logging.info(f"Found {total_files} XML files in {XML_LOG_DIR}")

    if total_files == 0:
        logging.warning("No XML files found. Exiting.")
        return

    # Find already processed files if skip enabled
    processed_files = set()
    if SKIP_ALREADY_PROCESSED:
        try:
            # Get list of batch files already saved
            batch_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(OUTPUT_NPZ_PREFIX) and f.endswith('.npz')]
            # Extract batch numbers
            processed_batches = len(batch_files)
            # Estimate number of files already processed
            files_processed = processed_batches * BATCH_SIZE
            # Skip first N files
            xml_files = xml_files[files_processed:]
            logging.info(f"Skipping approximately {files_processed} already processed files")
        except Exception as e:
            logging.warning(f"Error determining processed files: {e}. Will process all files.")
    
    # Recalculate total after skipping
    total_files = len(xml_files)
    if total_files == 0:
        logging.info("No new files to process. Exiting.")
        return
    
    total_samples_processed = 0
    batch_idx = 0

    # Find highest existing batch number if files exist
    if SKIP_ALREADY_PROCESSED:
        try:
            batch_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(OUTPUT_NPZ_PREFIX) and f.endswith('.npz')]
            if batch_files:
                batch_numbers = [int(f.split('_batch_')[1].split('.')[0]) for f in batch_files]
                batch_idx = max(batch_numbers) + 1
                logging.info(f"Starting from batch index {batch_idx}")
        except Exception as e:
            logging.warning(f"Error determining last batch number: {e}. Starting from 0.")
            batch_idx = 0

    logging.info(f"Starting data extraction with {NUM_PROCESSES} processes...")
    # Use Pool for parallel processing
    with Pool(processes=NUM_PROCESSES) as pool:
        # Process files in batches
        for i in range(0, total_files, BATCH_SIZE):
            batch_files = xml_files[i:i+BATCH_SIZE]
            if not batch_files:
                continue
                
            logging.info(f"Processing batch {batch_idx} (files {i+1}-{min(i+BATCH_SIZE, total_files)}/{total_files})...")
            
            # Process each batch of files
            # Use imap to process files and get results iteratively
            batch_results = []
            for file_path in batch_files:
                result = process_xml_file(file_path)
                if all(x is not None for x in result):
                    batch_results.append(result)
                    total_samples_processed += len(result[0])
            
            # After processing all files in this batch
            if batch_results:
                # Separate the results by type
                sequences_list = [r[0] for r in batch_results]
                static_list = [r[1] for r in batch_results]
                labels_list = [r[2] for r in batch_results]
                
                # Save batch data
                try:
                    all_sequences = np.concatenate(sequences_list, axis=0)
                    all_static = np.concatenate(static_list, axis=0)
                    all_labels = np.concatenate(labels_list, axis=0)
                    
                    output_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NPZ_PREFIX}_batch_{batch_idx}.npz")
                    np.savez_compressed(
                        output_path,
                        sequences=all_sequences,
                        static=all_static,
                        labels=all_labels
                    )
                    logging.info(f"Saved batch {batch_idx} data to {output_path} with {len(all_labels)} samples")
                except Exception as e:
                    logging.error(f"Error saving batch data: {e}")
            
            batch_idx += 1
            
            # Force garbage collection between batches
            if FORCE_GC_BETWEEN_BATCHES:
                import gc
                gc.collect()
                logging.debug("Forced garbage collection between batches")

    logging.info("="*30)
    logging.info("Data Processing Complete.")
    logging.info(f"Total XML files processed: {total_files}")
    logging.info(f"Total samples extracted: {total_samples_processed}")
    logging.info(f"Saved {batch_idx} batch files to: {OUTPUT_DIR}")
    logging.info(f"Log file saved to: data_processing.log")
    logging.info("Next step: Run train.py to train the model.")
    logging.info("="*30)

if __name__ == "__main__":
    main()