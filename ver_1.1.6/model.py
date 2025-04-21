# model.py (Transformerデータ準備版)
import os
import numpy as np
from multiprocessing import Pool, cpu_count
# 修正された feature_extractor をインポート
from feature_extractor_imitation import extract_features_labels_for_imitation
import logging
from tqdm import tqdm

# --- Configuration ---
XML_LOG_DIR = "../xml_logs/"
OUTPUT_DIR = "./training_data/"
NUM_PROCESSES = min(cpu_count(), 8)
BATCH_SIZE = 50 # バッチファイルあたりのXMLファイル数 (メモリ使用量に応じて調整)
# 出力ファイル名 (複数の配列を含むのでnpzが良い)
OUTPUT_NPZ_PREFIX = "mahjong_transformer_data"

# Set up logging (変更なし)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("mahjong_transformer_processing.log"),
        logging.StreamHandler()
    ]
)

def process_xml_file(xml_path):
    """Process a single XML file and extract features and labels for Transformer."""
    try:
        # 3種類のデータを取得するように変更
        sequences, static_features, labels = extract_features_labels_for_imitation(xml_path)
        if sequences is not None and static_features is not None and labels is not None and len(sequences) > 0:
            # Noneでないことと、データ数が0でないことを確認
            return sequences, static_features, labels
        else:
            # logging.warning(f"No valid data extracted from {os.path.basename(xml_path)}")
            return None, None, None
    except Exception as e:
        logging.error(f"Error processing {os.path.basename(xml_path)}: {e}", exc_info=False) # exc_info=Falseでトレースバック抑制
        return None, None, None

def save_batch_data(batch_sequences, batch_static, batch_labels, batch_idx, output_dir):
    """Save a batch of sequence, static features, and labels to disk."""
    if batch_sequences: # いずれか一つが空でなければデータがあるとみなす
        try:
            # 各リスト内のNumPy配列を結合
            sequences_array = np.concatenate(batch_sequences, axis=0)
            static_array = np.concatenate(batch_static, axis=0)
            labels_array = np.concatenate(batch_labels, axis=0)

            output_path = os.path.join(output_dir, f"{OUTPUT_NPZ_PREFIX}_batch_{batch_idx}.npz")
            # savez_compressed で複数の配列をキーワード付きで保存
            np.savez_compressed(output_path,
                                sequences=sequences_array,
                                static_features=static_array,
                                labels=labels_array)
            num_samples = len(sequences_array)
            logging.info(f"Saved batch {batch_idx} to {output_path} with {num_samples} samples")
            return num_samples
        except ValueError as e:
             logging.error(f"Error concatenating batch {batch_idx}: {e}")
             # 各要素の形状を出力してデバッグ
             for i in range(len(batch_sequences)):
                 logging.error(f"  Seq shape {i}: {batch_sequences[i].shape}, Static shape {i}: {batch_static[i].shape}, Label shape {i}: {batch_labels[i].shape}")
             return 0
        except Exception as e:
             logging.error(f"Error saving batch {batch_idx}: {e}", exc_info=True)
             return 0
    return 0

def main():
    """Main function to process all XML logs and prepare training data for Transformer."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    xml_files = [os.path.join(XML_LOG_DIR, f) for f in os.listdir(XML_LOG_DIR) if f.endswith(".xml")]
    total_files = len(xml_files)
    logging.info(f"Found {total_files} XML files to process")

    if total_files == 0:
        logging.error("No XML files found in the specified directory")
        return

    # Initialize data storage for batches
    batch_sequences_list = []
    batch_static_list = []
    batch_labels_list = []
    total_samples_processed = 0
    batch_idx = 0

    # Process files in parallel
    logging.info(f"Starting processing with {NUM_PROCESSES} workers...")
    with Pool(processes=NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap(process_xml_file, xml_files), total=total_files, desc="Processing XML files"))

    # Collect results and save in batches
    logging.info("Collecting results and saving batches...")
    for i, result_tuple in enumerate(tqdm(results, desc="Saving batches")):
        sequences, static_features, labels = result_tuple
        if sequences is not None and static_features is not None and labels is not None:
            batch_sequences_list.append(sequences)
            batch_static_list.append(static_features)
            batch_labels_list.append(labels)

            # Save batch if we reach the batch size or it's the last file
            if len(batch_sequences_list) >= BATCH_SIZE or i == len(results) - 1:
                 num_saved = save_batch_data(batch_sequences_list, batch_static_list, batch_labels_list, batch_idx, OUTPUT_DIR)
                 total_samples_processed += num_saved
                 # Clear lists for the next batch
                 batch_sequences_list = []
                 batch_static_list = []
                 batch_labels_list = []
                 batch_idx += 1

    # Final logging
    logging.info(f"Processing complete.")
    logging.info(f"Processed {total_files} files.")
    logging.info(f"Extracted and saved {total_samples_processed} samples in {batch_idx} batches.")
    logging.info(f"Data saved in {OUTPUT_DIR} with prefix '{OUTPUT_NPZ_PREFIX}'")
    logging.info("Next step: Combine batches if needed and run train.py")

if __name__ == "__main__":
    main()