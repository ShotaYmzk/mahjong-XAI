# data_preparation.py
import glob
import numpy as np
from feature_extractor import extract_sequences  # 先ほどの抽出コードをfeature_extractor.pyにまとめたもの

if __name__ == "__main__":
    # 1000個のXMLファイルのみを対象
    xml_files = glob.glob('xml_logs/*.xml')[:1000]
    print(f"処理するXMLファイル数: {len(xml_files)}")
    sequences = extract_sequences(xml_files)
    
    # 各シーケンスから特徴量とラベルを取得
    X = np.stack([seq['features'] for seq in sequences])  # (num_sequences, max_len, 802)
    y = np.stack([seq['labels'] for seq in sequences])    # (num_sequences, max_len, 34)
    masks = (X.sum(axis=-1) != 0).astype(np.int32)          # (num_sequences, max_len)

    np.save('X.npy', X)
    np.save('y.npy', y)
    np.save('masks.npy', masks)
    print("Data saved as X.npy, y.npy, and masks.npy")