# preprocess_data_to_npy.py (メモリ効率改善版)
import pyarrow.parquet as pq
import numpy as np
import os
import logging
from tqdm import tqdm

# --- 設定 ---
INPUT_PARQUET_PATH = "./training_data/mahjong_imitation_data_v_strong_flat.parquet"
OUTPUT_DIR = "./training_data_npy/"
OUTPUT_SEQUENCES_PATH = os.path.join(OUTPUT_DIR, "sequences.npy")
OUTPUT_STATICS_PATH = os.path.join(OUTPUT_DIR, "statics.npy")
OUTPUT_LABELS_PATH = os.path.join(OUTPUT_DIR, "labels.npy")

# --- ログ設定 ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def convert_parquet_to_npy_chunked():
    logging.info(f"Parquetファイル {INPUT_PARQUET_PATH} のチャンクごとの変換を開始します...")
    if not os.path.exists(INPUT_PARQUET_PATH):
        logging.error("入力Parquetファイルが見つかりません。")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # まずはParquetファイルを開き、メタデータを取得
    pq_file = pq.ParquetFile(INPUT_PARQUET_PATH)
    num_row_groups = pq_file.num_row_groups
    total_rows = pq_file.metadata.num_rows
    logging.info(f"{total_rows}行、{num_row_groups}個の行グループを検出しました。")

    # .npyファイルを書き込みモード('w+')で初期化し、メモリマップオブジェクトを取得
    # これにより、巨大な空ファイルをディスク上に確保し、メモリを消費せずにアクセスできる
    # shapeの最初の次元はtotal_rows、2番目以降はParquetのスキーマから推測
    # 例: sequences_flat は (total_rows, 360), statics は (total_rows, 543)
    # shapeを決め打ちする（要確認）
    sequences_shape = (total_rows, 360) # MAX_EVENT_HISTORY(60) * EVENT_FEATURE_DIM(6)
    statics_shape = (total_rows, 543)    # STATIC_FEATURE_DIM
    labels_shape = (total_rows,)

    sequences_npy = np.lib.format.open_memmap(OUTPUT_SEQUENCES_PATH, mode='w+', dtype=np.float32, shape=sequences_shape)
    statics_npy = np.lib.format.open_memmap(OUTPUT_STATICS_PATH, mode='w+', dtype=np.float32, shape=statics_shape)
    labels_npy = np.lib.format.open_memmap(OUTPUT_LABELS_PATH, mode='w+', dtype=np.int8, shape=labels_shape)
    
    logging.info(".npyファイルのメモリマップを初期化しました。")

    processed_rows = 0
    # 行グループごとにループ処理
    for i in tqdm(range(num_row_groups), desc="行グループを処理中"):
        # 1つの行グループだけをメモリに読み込む
        table = pq_file.read_row_group(i)
        df_chunk = table.to_pandas()
        
        chunk_size = len(df_chunk)
        
        # データを抽出し、メモリマップされた.npyファイルに書き込む
        sequences_npy[processed_rows : processed_rows + chunk_size] = np.stack(df_chunk['sequences_flat'].values)
        statics_npy[processed_rows : processed_rows + chunk_size] = np.stack(df_chunk['static_features'].values)
        labels_npy[processed_rows : processed_rows + chunk_size] = df_chunk['labels'].to_numpy(dtype=np.int8)

        processed_rows += chunk_size

    logging.info("全ての行グループの処理と保存が完了しました。")
    logging.info(f"出力ファイル形状:")
    logging.info(f"  sequences: {sequences_npy.shape}")
    logging.info(f"  statics:   {statics_npy.shape}")
    logging.info(f"  labels:    {labels_npy.shape}")

if __name__ == "__main__":
    # 関数名を変更したため、呼び出しも変更
    convert_parquet_to_npy_chunked()