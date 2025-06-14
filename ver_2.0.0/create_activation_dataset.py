import torch
import numpy as np
import os
import glob
import time
import logging
import h5py
# ★★★★★★★★★★★★★★★★★★★★★★★ 修正箇所1 ★★★★★★★★★★★★★★★★★★★★★★★
from multiprocessing import Pool, cpu_count, set_start_method 
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
from tqdm import tqdm
import random

# --- プロジェクトモジュールのインポート ---
from predict import load_trained_model, DEFAULT_MODEL_PATH
from full_mahjong_parser import parse_full_mahjong_log
from game_state import GameState, STATIC_FEATURE_DIM, NUM_PLAYERS, EVENT_TYPES
import shanten_analyzer

# --- 設定 ---
XML_DIR = "/home/ubuntu/Documents/tenhou_xml_2023"
OUTPUT_HDF5_PATH = "./activation_dataset.hdf5"
MODEL_PATH = DEFAULT_MODEL_PATH
# テスト用に処理するファイル数を制限。全ファイルを実行する場合は None にする
MAX_FILES_TO_PROCESS = None
PROCESS_COUNT = max(1, cpu_count() - 2) # CPUコア数に応じて調整

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')

# --- グローバル変数 (マルチプロセス用) ---
model_global = None
event_feature_dim_global = None

def initialize_worker(model_path):
    """各ワーカープロセスの初期化関数"""
    global model_global, event_feature_dim_global
    # logging.info(f"Worker {os.getpid()} is initializing...") # ログが多すぎるのでコメントアウト
    try:
        event_feature_dim_global = GameState().get_event_sequence_features().shape[1]
        model_global = load_trained_model(model_path, event_feature_dim_global, STATIC_FEATURE_DIM)
        # logging.info(f"Worker {os.getpid()} model loaded successfully.")
    except Exception as e:
        # logging.error(f"Worker {os.getpid()} failed to load model: {e}", exc_info=True)
        model_global = None

def process_xml_file(xml_path):
    """単一のXMLファイルを処理し、データポイントを抽出する"""
    if model_global is None:
        # logging.error(f"Model is not loaded in worker {os.getpid()}. Skipping file.")
        return []

    try:
        _, rounds_data = parse_full_mahjong_log(xml_path)
    except Exception as e:
        return []

    data_points = []
    for round_data in rounds_data:
        try:
            events = round_data.get("events", [])
            if not events: continue
            
            game_state = GameState()
            game_state.init_round(round_data)

            for i, event_xml in enumerate(events):
                is_discard = False
                discard_player_id, discard_pai_id = -1, -1
                tag = event_xml["tag"]
                for d_tag, p_id in GameState.DISCARD_TAGS.items():
                    if tag.startswith(d_tag) and tag[1:].isdigit():
                        discard_player_id, discard_pai_id = p_id, int(tag[1:])
                        is_discard = True
                        break
                
                if is_discard:
                    hand_14 = game_state.player_hands[discard_player_id]
                    if len(hand_14) % 3 != 2:
                        game_state.process_event(event_xml)
                        continue

                    _, _, _, _, activation_vector = game_state.predict_discard_with_model(
                        model_global, discard_player_id
                    )

                    if activation_vector is not None:
                        shanten_change, ukeire_count = shanten_analyzer.analyze_speed_metrics(
                            hand_14, discard_pai_id, game_state
                        )
                        is_deal_in = 0
                        if i + 1 < len(events) and events[i+1]['tag'] == 'AGARI' and events[i+1]['attrib'].get('fromWho') == str(discard_player_id):
                            is_deal_in = 1
                        dora_count = game_state.count_dora_in_hand(discard_player_id)

                        data_points.append({
                            "activation": activation_vector,
                            "shanten_change": shanten_change,
                            "ukeire_count": ukeire_count,
                            "is_deal_in": is_deal_in,
                            "dora_count": dora_count
                        })
                game_state.process_event(event_xml)
        except Exception:
            continue
    return data_points


if __name__ == "__main__":
    # ★★★★★★★★★★★★★★★★★★★★★★★ 修正箇所2 ★★★★★★★★★★★★★★★★★★★★★★★
    # PyTorch + CUDA + multiprocessing を使うためのおまじない
    # 必ず if __name__ == "__main__": の直下に書く
    try:
        set_start_method('spawn')
        logging.info("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        # すでに設定されている場合は何もしない
        pass
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

    logging.info("Starting dataset creation process...")
    xml_files = glob.glob(os.path.join(XML_DIR, "**", "*.mjlog"), recursive=True)
    xml_files += glob.glob(os.path.join(XML_DIR, "**", "*.xml"), recursive=True)
    
    if not xml_files:
        logging.error(f"No XML/MJLOG files found in {XML_DIR}. Please check the path.")
        exit()

    if MAX_FILES_TO_PROCESS is not None:
        random.shuffle(xml_files)
        xml_files = xml_files[:MAX_FILES_TO_PROCESS]
    
    logging.info(f"Found {len(xml_files)} XML/MJLOG files to process.")
    logging.info(f"Using {PROCESS_COUNT} worker processes.")

    ACTIVATION_DIM = 512
    
    if os.path.exists(OUTPUT_HDF5_PATH):
        logging.warning(f"既存のデータセットファイル {OUTPUT_HDF5_PATH} を削除します。")
        os.remove(OUTPUT_HDF5_PATH)

    with h5py.File(OUTPUT_HDF5_PATH, 'w') as hf:
        dset_activations = hf.create_dataset('activations', (0, ACTIVATION_DIM), maxshape=(None, ACTIVATION_DIM), dtype='f4', chunks=(1024, ACTIVATION_DIM))
        dset_shanten = hf.create_dataset('shanten_changes', (0,), maxshape=(None,), dtype='i1', chunks=(8192,))
        dset_ukeire = hf.create_dataset('ukeire_counts', (0,), maxshape=(None,), dtype='i2', chunks=(8192,))
        dset_deal_in = hf.create_dataset('is_deal_ins', (0,), maxshape=(None,), dtype='i1', chunks=(8192,))
        dset_dora = hf.create_dataset('dora_counts', (0,), maxshape=(None,), dtype='i1', chunks=(8192,))

        total_data_points = 0
        with Pool(processes=PROCESS_COUNT, initializer=initialize_worker, initargs=(MODEL_PATH,)) as pool:
            with tqdm(total=len(xml_files), desc="Processing XML files") as pbar:
                for result_list in pool.imap_unordered(process_xml_file, xml_files):
                    if result_list:
                        num_new_points = len(result_list)
                        current_size = dset_activations.shape[0]
                        dset_activations.resize(current_size + num_new_points, axis=0)
                        dset_shanten.resize(current_size + num_new_points, axis=0)
                        dset_ukeire.resize(current_size + num_new_points, axis=0)
                        dset_deal_in.resize(current_size + num_new_points, axis=0)
                        dset_dora.resize(current_size + num_new_points, axis=0)

                        dset_activations[current_size:] = np.array([d['activation'] for d in result_list], dtype=np.float32)
                        dset_shanten[current_size:] = [d['shanten_change'] for d in result_list]
                        dset_ukeire[current_size:] = [d['ukeire_count'] for d in result_list]
                        dset_deal_in[current_size:] = [d['is_deal_in'] for d in result_list]
                        dset_dora[current_size:] = [d['dora_count'] for d in result_list]

                        total_data_points += num_new_points
                    pbar.update(1)
                    pbar.set_postfix(data_points=total_data_points)

    logging.info(f"Finished processing. Total data points collected: {total_data_points}")
    if total_data_points == 0:
        logging.error("データポイントが1つも収集できませんでした。牌譜ファイルのパスや内容を確認してください。")
    else:
        logging.info(f"Dataset saved to {OUTPUT_HDF5_PATH}")