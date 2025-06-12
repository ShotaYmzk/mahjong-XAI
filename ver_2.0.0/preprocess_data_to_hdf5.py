# preprocess_data_to_hdf5.py
import os
import numpy as np
import pandas as pd
import h5py
from multiprocessing import Pool, cpu_count
import logging
from tqdm import tqdm
import sys
import time
import glob
import gc
from typing import Tuple, List, Dict, Any, Union

# --- プロジェクトモジュールのインポート ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from full_mahjong_parser import parse_full_mahjong_log
    from game_state import GameState, STATIC_FEATURE_DIM, MAX_EVENT_HISTORY
    from tile_utils import tile_id_to_index
    # train.pyから定数をインポート
    from train import EVENT_FEATURE_DIM
    logging.info("プロジェクトモジュールを正常にインポートしました。")
    logging.info(f"静的特徴量次元: {STATIC_FEATURE_DIM}, イベント特徴量次元: {EVENT_FEATURE_DIM}")
except ImportError as e:
    logging.critical(f"[致命的エラー] モジュールのインポートに失敗: {e}")
    sys.exit(1)

# --- 設定 ---
XML_LOG_DIR = "/home/ubuntu/Documents/xml_logs_2023"
OUTPUT_DIR = "./training_data/"
OUTPUT_HDF5_FILENAME = "mahjong_imitation_data.hdf5"
OUTPUT_HDF5_PATH = os.path.join(OUTPUT_DIR, OUTPUT_HDF5_FILENAME)

NUM_PROCESSES = max(1, cpu_count() - 2)
FILES_PER_CHUNK = 200  # HDF5に書き込む単位となるファイル数

# --- ログ設定 ---
LOG_FILE = "data_processing_to_hdf5.log"
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(processName)s/%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler()])

WorkerResult = Union[pd.DataFrame, Tuple[str, str]]

# extract_features_for_file関数はParquet版とほぼ同じ
def extract_features_for_file(xml_path: str) -> WorkerResult:
    filename = os.path.basename(xml_path)
    try:
        all_samples = []
        last_decision_points: Dict[int, Dict[str, Any]] = {}
        
        _, rounds_data = parse_full_mahjong_log(xml_path)
        if not rounds_data:
            return pd.DataFrame()

        game_state = GameState()

        for round_idx, round_data in enumerate(rounds_data):
            try:
                game_state.init_round(round_data)
                events = round_data.get("events", [])
                if not events: continue
                last_decision_points.clear()

                for event in events:
                    tag, attrib = event["tag"], event["attrib"]
                    tag_upper = tag.upper()
                    
                    is_tsumo = False
                    for t_tag, p_id in GameState.TSUMO_TAGS.items():
                        if tag.startswith(t_tag) and tag[1:].isdigit():
                            tsumo_player_id, tsumo_pai_id = p_id, int(tag[1:])
                            game_state.process_tsumo(tsumo_player_id, tsumo_pai_id)
                            context = {
                                "sequences": game_state.get_event_sequence_features(),
                                "static_features": game_state.get_static_features(tsumo_player_id),
                            }
                            last_decision_points[tsumo_player_id] = context
                            is_tsumo = True
                            break
                    if is_tsumo: continue

                    is_discard = False
                    for d_tag, p_id in GameState.DISCARD_TAGS.items():
                        if tag_upper.startswith(d_tag) and tag_upper[1:].isdigit():
                            discard_player_id, discard_pai_id = p_id, int(tag_upper[1:])
                            tsumogiri = tag[0].islower()
                            if discard_player_id in last_decision_points:
                                decision_state = last_decision_points.pop(discard_player_id)
                                label = tile_id_to_index(discard_pai_id)
                                if label != -1:
                                    sample = {
                                        'sequences_flat': decision_state["sequences"].flatten(),
                                        'static_features': decision_state["static_features"], 
                                        'labels': label
                                    }
                                    all_samples.append(sample)
                            game_state.process_discard(discard_player_id, discard_pai_id, tsumogiri)
                            is_discard = True
                            break
                    if is_discard: continue

                    if tag == "N":
                        game_state.process_naki(int(attrib['who']), int(attrib['m']))
                        last_decision_points.clear()
                    elif tag == "REACH" and int(attrib.get('step', 0)) == 1:
                        game_state.process_reach(int(attrib['who']), 1)
                    elif tag in ["AGARI", "RYUUKYOKU"]:
                        last_decision_points.clear()
                        break
            except Exception:
                continue
        
        if not all_samples:
            return pd.DataFrame()
        return pd.DataFrame(all_samples)

    except Exception as e:
        error_message = f"ファイル {filename} の処理中に致命的なエラー: {e.__class__.__name__}: {e}"
        return (xml_path, error_message)

def main():
    logging.info("最強AI向けデータセット(HDF5)の生成を開始します...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_HDF5_PATH):
        logging.warning(f"既存のHDF5ファイル {OUTPUT_HDF5_PATH} を削除します。")
        os.remove(OUTPUT_HDF5_PATH)

    xml_files = sorted(glob.glob(os.path.join(XML_LOG_DIR, "*.xml")))
    if not xml_files:
        logging.error(f"XMLログファイルがディレクトリに見つかりません: {XML_LOG_DIR}"); return

    logging.info(f"{len(xml_files)}個のXMLファイルを検出しました。{NUM_PROCESSES}個のプロセスで処理を開始します。")

    total_samples_processed = 0
    failed_files_count = 0
    start_time_all = time.time()
    
    # HDF5ファイルを開き、データセットを初期化
    with h5py.File(OUTPUT_HDF5_PATH, 'w') as hf:
        # データセットをリサイズ可能(maxshape=(None,...))、チャンク指定、圧縮有効で作成
        dset_seq = hf.create_dataset('sequences', 
                                     shape=(0, MAX_EVENT_HISTORY * EVENT_FEATURE_DIM), 
                                     maxshape=(None, MAX_EVENT_HISTORY * EVENT_FEATURE_DIM), 
                                     dtype='f4', chunks=True, compression="gzip")
        dset_static = hf.create_dataset('statics', 
                                        shape=(0, STATIC_FEATURE_DIM), 
                                        maxshape=(None, STATIC_FEATURE_DIM), 
                                        dtype='f4', chunks=True, compression="gzip")
        dset_labels = hf.create_dataset('labels', 
                                        shape=(0,), 
                                        maxshape=(None,), 
                                        dtype='i1', chunks=True, compression="gzip")
        
        logging.info("HDF5ファイルのデータセットを初期化しました。")

        with Pool(processes=NUM_PROCESSES, maxtasksperchild=1) as pool:
            results_iterator = pool.imap_unordered(extract_features_for_file, xml_files)
            
            batch_dfs = []
            files_processed_since_write = 0

            for result in tqdm(results_iterator, total=len(xml_files), desc="XMLファイルを処理中"):
                if isinstance(result, pd.DataFrame) and not result.empty:
                    batch_dfs.append(result)
                elif isinstance(result, tuple):
                    failed_path, error_msg = result
                    logging.error(f"ワーカプロセスでエラー: {error_msg} (ファイル: {os.path.basename(failed_path)})")
                    failed_files_count += 1
                
                files_processed_since_write += 1

                # 一定数のファイルを処理したら、またはバッチが溜まったらHDF5に書き込む
                if (files_processed_since_write >= FILES_PER_CHUNK) and batch_dfs:
                    combined_df = pd.concat(batch_dfs, ignore_index=True)
                    num_new_samples = len(combined_df)
                    
                    if num_new_samples > 0:
                        # 1. データセットのサイズをリサイズ
                        dset_seq.resize(total_samples_processed + num_new_samples, axis=0)
                        dset_static.resize(total_samples_processed + num_new_samples, axis=0)
                        dset_labels.resize(total_samples_processed + num_new_samples, axis=0)

                        # 2. NumPy配列に変換して書き込み
                        dset_seq[total_samples_processed:] = np.stack(combined_df['sequences_flat'].values)
                        dset_static[total_samples_processed:] = np.stack(combined_df['static_features'].values)
                        dset_labels[total_samples_processed:] = combined_df['labels'].to_numpy(dtype=np.int8)
                        
                        total_samples_processed += num_new_samples
                        logging.info(f" {num_new_samples}件のサンプルをHDF5ファイルに書き込みました。(合計: {total_samples_processed})")

                    batch_dfs = []
                    files_processed_since_write = 0
                    gc.collect()
            
            # ループ終了後、残りのバッチを書き込む
            if batch_dfs:
                combined_df = pd.concat(batch_dfs, ignore_index=True)
                num_new_samples = len(combined_df)

                if num_new_samples > 0:
                    dset_seq.resize(total_samples_processed + num_new_samples, axis=0)
                    dset_static.resize(total_samples_processed + num_new_samples, axis=0)
                    dset_labels.resize(total_samples_processed + num_new_samples, axis=0)

                    dset_seq[total_samples_processed:] = np.stack(combined_df['sequences_flat'].values)
                    dset_static[total_samples_processed:] = np.stack(combined_df['static_features'].values)
                    dset_labels[total_samples_processed:] = combined_df['labels'].to_numpy(dtype=np.int8)

                    total_samples_processed += num_new_samples
                    logging.info(f"残りの {num_new_samples}件のサンプルを書き込みました。(合計: {total_samples_processed})")

    end_time_all = time.time()
    logging.info("="*30)
    logging.info("データ前処理(HDF5)が完了しました。")
    logging.info(f"処理したXMLファイル数: {len(xml_files)}")
    logging.info(f"失敗したファイル数: {failed_files_count}")
    logging.info(f"抽出・保存した総サンプル数: {total_samples_processed}")
    logging.info(f"出力HDF5ファイル: {OUTPUT_HDF5_PATH}")
    logging.info(f"総処理時間: {end_time_all - start_time_all:.2f} 秒")
    logging.info(f"ログファイル: {LOG_FILE}")
    logging.info("="*30)

if __name__ == "__main__":
    main()