# preprocess_data_to_parquet.py (BrokenPipeError対策版)

import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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
    from game_state import GameState, NUM_PLAYERS, STATIC_FEATURE_DIM, MAX_EVENT_HISTORY
    from tile_utils import tile_id_to_index
    logging.info("プロジェクトモジュールを正常にインポートしました。")
    logging.info(f"静的特徴量次元: {STATIC_FEATURE_DIM}")
except ImportError as e:
    logging.critical(f"[致命的エラー] モジュールのインポートに失敗: {e}")
    sys.exit(1)

# --- 設定 ---
XML_LOG_DIR = "/home/ubuntu/Documents/xml_logs_2023"
OUTPUT_DIR = "./training_data/"
OUTPUT_PARQUET_FILENAME = "mahjong_imitation_data_v_strong_flat.parquet"
OUTPUT_PARQUET_PATH = os.path.join(OUTPUT_DIR, OUTPUT_PARQUET_FILENAME)

NUM_PROCESSES = max(1, cpu_count() - 2)
FILES_PER_WRITE = 100

# --- ログ設定 ---
LOG_FILE = "data_processing_to_parquet_v_strong_flat.log"
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(processName)s/%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler()])

# ★★★ ワーカ関数の戻り値の型を定義 ★★★
WorkerResult = Union[pd.DataFrame, Tuple[str, str]]

def extract_features_for_file(xml_path: str) -> WorkerResult:
    """
    単一のXMLログファイルを処理し、DataFrameまたはエラー情報を返す。
    """
    filename = os.path.basename(xml_path)
    # ★★★ 関数全体をtry...exceptで囲み、予期せぬエラーをキャッチ ★★★
    try:
        all_samples = []
        last_decision_points: Dict[int, Dict[str, Any]] = {}
        
        _, rounds_data = parse_full_mahjong_log(xml_path)
        if not rounds_data:
            return pd.DataFrame()

        game_state = GameState()

        for round_idx, round_data in enumerate(rounds_data):
            # ラウンドごとのエラーはスキップして継続
            try:
                game_state.init_round(round_data)
                events = round_data.get("events", [])
                if not events: continue
                last_decision_points.clear()

                for event in events:
                    tag, attrib = event["tag"], event["attrib"]
                    # (内部のロジックは変更なし)
                    is_tsumo = False
                    for t_tag, p_id in GameState.TSUMO_TAGS.items():
                        if tag.startswith(t_tag) and tag[1:].isdigit():
                            tsumo_player_id, tsumo_pai_id = p_id, int(tag[1:])
                            game_state.process_tsumo(tsumo_player_id, tsumo_pai_id)
                            context = {
                                "round_wind": game_state.round_num_wind, "honba": game_state.honba,
                                "kyotaku": game_state.kyotaku, "junme": game_state.junme,
                                "player_id": tsumo_player_id, "dealer_id": game_state.dealer,
                                "scores": game_state.current_scores,
                                "dora_indicators": game_state.dora_indicators,
                                "reach_status": game_state.player_reach_status,
                                "sequences": game_state.get_event_sequence_features(),
                                "static_features": game_state.get_static_features(tsumo_player_id),
                            }
                            last_decision_points[tsumo_player_id] = context
                            is_tsumo = True
                            break
                    if is_tsumo: continue

                    is_discard = False
                    for d_tag, p_id in GameState.DISCARD_TAGS.items():
                        if tag.startswith(d_tag) and tag[1:].isdigit():
                            discard_player_id, discard_pai_id, tsumogiri = p_id, int(tag[1:]), tag[0].islower()
                            if discard_player_id in last_decision_points:
                                decision_state = last_decision_points.pop(discard_player_id)
                                label = tile_id_to_index(discard_pai_id)
                                if label != -1:
                                    sample = {
                                        'round_wind': decision_state['round_wind'], 'honba': decision_state['honba'],
                                        'kyotaku': decision_state['kyotaku'], 'junme': decision_state['junme'],
                                        'player_id': decision_state['player_id'], 'dealer_id': decision_state['dealer_id'],
                                        'scores': decision_state['scores'], 'dora_indicators': decision_state['dora_indicators'],
                                        'reach_status': decision_state['reach_status'],
                                        'sequences_flat': decision_state["sequences"].flatten(),
                                        'static_features': decision_state["static_features"], 'labels': label
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
                # ラウンドレベルのエラーはログに残さず、ファイルレベルで集約
                continue
        
        if not all_samples:
            return pd.DataFrame()
        return pd.DataFrame(all_samples)

    except Exception as e:
        # ★★★ 致命的なエラーが発生した場合、ファイル名とエラーメッセージを返す ★★★
        error_message = f"ファイル {filename} の処理中に致命的なエラー: {e.__class__.__name__}: {e}"
        return (xml_path, error_message)

def get_pyarrow_schema() -> pa.Schema:
    EVENT_FEATURE_DIM = 6 
    sequences_flat_type = pa.list_(pa.float32(), list_size=MAX_EVENT_HISTORY * EVENT_FEATURE_DIM)
    static_type = pa.list_(pa.float32(), list_size=STATIC_FEATURE_DIM)
    schema = pa.schema([
        pa.field('round_wind', pa.int16()), pa.field('honba', pa.int16()),
        pa.field('kyotaku', pa.int16()), pa.field('junme', pa.float32()),
        pa.field('player_id', pa.int8()), pa.field('dealer_id', pa.int8()),
        pa.field('scores', pa.list_(pa.int32(), list_size=NUM_PLAYERS)),
        pa.field('dora_indicators', pa.list_(pa.int32())),
        pa.field('reach_status', pa.list_(pa.int8(), list_size=NUM_PLAYERS)),
        pa.field('sequences_flat', sequences_flat_type),
        pa.field('static_features', static_type), pa.field('labels', pa.int8())
    ])
    return schema

def main():
    logging.info("最強AI向けデータセットの生成を開始します... (BrokenPipeError対策版)")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_PARQUET_PATH):
        logging.warning(f"既存のParquetファイル {OUTPUT_PARQUET_PATH} を削除します。")
        os.remove(OUTPUT_PARQUET_PATH)

    xml_files = sorted(glob.glob(os.path.join(XML_LOG_DIR, "*.xml")))
    if not xml_files:
        logging.error(f"XMLログファイルがディレクトリに見つかりません: {XML_LOG_DIR}"); return

    logging.info(f"{len(xml_files)}個のXMLファイルを検出しました。{NUM_PROCESSES}個のプロセスで処理を開始します。")

    total_samples_processed = 0
    failed_files_count = 0
    start_time_all = time.time()
    
    writer = None
    schema = get_pyarrow_schema()
    logging.info("PyArrowスキーマを定義しました。")

    # maxtasksperchild=1 は、ワーカプロセスが1つのタスクを終えるごとに新しいプロセスに置き換える設定。
    # メモリリーク対策として非常に有効。
    with Pool(processes=NUM_PROCESSES, maxtasksperchild=1) as pool:
        # imap_unordered はタスクが完了した順に結果を返すため、効率的。
        results_iterator = pool.imap_unordered(extract_features_for_file, xml_files)
        
        batch_dfs = []
        files_processed_since_write = 0

        for result in tqdm(results_iterator, total=len(xml_files), desc="XMLファイルを処理中"):
            # ★★★ 親プロセスでのエラーハンドリング ★★★
            if isinstance(result, pd.DataFrame):
                if not result.empty:
                    batch_dfs.append(result)
            elif isinstance(result, tuple):
                # エラータプル (xml_path, error_message) を受け取った場合
                failed_path, error_msg = result
                logging.error(f"ワーカプロセスでエラー: {error_msg} (ファイル: {os.path.basename(failed_path)})")
                failed_files_count += 1
            
            files_processed_since_write += 1

            if (files_processed_since_write >= FILES_PER_WRITE or (len(batch_dfs) > 0 and (total_samples_processed + sum(len(df) for df in batch_dfs)) % (FILES_PER_WRITE * 1000) == 0)) and batch_dfs:
                combined_df = pd.concat(batch_dfs, ignore_index=True)
                if not combined_df.empty:
                    try:
                        if writer is None:
                            writer = pq.ParquetWriter(OUTPUT_PARQUET_PATH, schema, compression='ZSTD')
                        table = pa.Table.from_pandas(combined_df, schema=schema, preserve_index=False)
                        writer.write_table(table)
                        num_written = len(combined_df)
                        total_samples_processed += num_written
                        logging.info(f" {num_written}件のサンプルをParquetファイルに書き込みました。(合計: {total_samples_processed})")
                    except Exception as e:
                        logging.error(f"Parquetへの書き込み中にエラーが発生しました: {e}", exc_info=True)
                batch_dfs = []
                files_processed_since_write = 0
                gc.collect()
        
        # ★★★ ループ終了後、残りのバッチを書き込む ★★★
        if batch_dfs:
            combined_df = pd.concat(batch_dfs, ignore_index=True)
            if not combined_df.empty:
                try:
                    if writer is None:
                        writer = pq.ParquetWriter(OUTPUT_PARQUET_PATH, schema, compression='ZSTD')
                    table = pa.Table.from_pandas(combined_df, schema=schema, preserve_index=False)
                    writer.write_table(table)
                    num_written = len(combined_df)
                    total_samples_processed += num_written
                    logging.info(f"残りの {num_written}件のサンプルを書き込みました。(合計: {total_samples_processed})")
                except Exception as e:
                    logging.error(f"最後のバッチの書き込み中にエラーが発生しました: {e}", exc_info=True)


    if writer:
        writer.close()
        logging.info("Parquetライターを正常にクローズしました。")

    end_time_all = time.time()
    logging.info("="*30)
    logging.info("データ前処理が完了しました。")
    logging.info(f"処理したXMLファイル数: {len(xml_files)}")
    logging.info(f"失敗したファイル数: {failed_files_count}")
    logging.info(f"抽出・保存した総サンプル数: {total_samples_processed}")
    logging.info(f"出力Parquetファイル: {OUTPUT_PARQUET_PATH}")
    logging.info(f"総処理時間: {end_time_all - start_time_all:.2f} 秒")
    logging.info(f"ログファイル: {LOG_FILE}")
    logging.info("="*30)

if __name__ == "__main__":
    main()