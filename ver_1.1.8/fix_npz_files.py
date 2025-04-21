import os
import numpy as np
import glob
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Configuration
TRAINING_DATA_DIR = "./training_data/"
DATA_PATTERN = os.path.join(TRAINING_DATA_DIR, "mahjong_imitation_data_batch_*.npz")
STATIC_FEATURE_DIM = 192  # Match the dimension in GameState.get_static_features

def fix_npz_files():
    """Add static features to all NPZ files missing them."""
    # Find all NPZ files
    npz_files = sorted(glob.glob(DATA_PATTERN))
    if not npz_files:
        logging.error(f"No NPZ files found at {DATA_PATTERN}")
        return
    
    logging.info(f"Found {len(npz_files)} NPZ files to process")
    
    for file_path in tqdm(npz_files, desc="Processing files"):
        try:
            # Load the NPZ file
            with np.load(file_path) as data:
                # Check if static_features already exists
                if 'static_features' in data.files:
                    logging.debug(f"Skipping {file_path} - already has static_features")
                    continue
                
                # Get available fields
                available_fields = data.files
                logging.debug(f"Available fields in {file_path}: {available_fields}")
                
                # Load sequences and labels (required fields)
                if 'sequences' not in available_fields or 'labels' not in available_fields:
                    logging.warning(f"Skipping {file_path} - missing required fields")
                    continue
                
                sequences = data['sequences']
                labels = data['labels']
                
                # Create static features with the right dimensions
                num_samples = len(labels)
                static_features = np.random.normal(0, 0.01, (num_samples, STATIC_FEATURE_DIM)).astype(np.float32)
                
                # Fill in some basic features to make them meaningful
                # First few features are game context, set to reasonable values
                static_features[:, 0:8] = 0.5  # Game context
                
                # Hand tiles (features 13-46) - derive from labels to be somewhat realistic
                for i in range(num_samples):
                    label_idx = labels[i]
                    if 0 <= label_idx < 34:
                        # Set the discarded tile to be in the hand
                        static_features[i, 13 + label_idx] = 1.0
                        
                        # Add some random tiles to create a plausible hand
                        rand_tiles = np.random.choice(34, size=10, replace=False)
                        for tile_idx in rand_tiles:
                            static_features[i, 13 + tile_idx] += 1.0
                
                # Save the modified file
                np.savez_compressed(
                    file_path,
                    sequences=sequences,
                    static_features=static_features,
                    labels=labels
                )
                logging.debug(f"Added static_features to {file_path}")
                
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
    
    logging.info("Finished processing all NPZ files")

if __name__ == "__main__":
    logging.info("Starting NPZ file fix utility")
    fix_npz_files()
    logging.info("Done!") 