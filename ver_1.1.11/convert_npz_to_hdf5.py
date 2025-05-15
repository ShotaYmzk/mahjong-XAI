# scripts/convert_npz_to_hdf5.py
import h5py
import glob
import numpy as np
from tqdm import tqdm

# 元 NPZ ファイルのパターン
npz_pattern = "ver_1.1.10/training_data/mahjong_imitation_data_v119_batch_*.npz"
# 出力先 HDF5 ファイル
h5_path = "ver_1.1.10/training_data/mahjong_data_v119.h5"

npz_files = sorted(glob.glob(npz_pattern))
assert npz_files, "No NPZ files found"

# まずは総サンプル数と各次元を調べる
total = 0
for f in npz_files:
    with np.load(f) as d:
        total += len(d["labels"])
# サンプル例から次元を取得
with np.load(npz_files[0]) as d:
    seq_shape = d["sequences"].shape[1:]       # (SeqLen, EventDim)
    static_dim = d["static_features"].shape[1] # StaticDim

print(f"Total samples: {total}, seq_shape={seq_shape}, static_dim={static_dim}")

# HDF5 を書き込みモードで開く
with h5py.File(h5_path, "w") as h5:
    # チャンクサイズはバッチサイズに合わせるのがおすすめ
    chunk_size = 1024
    seq_ds    = h5.create_dataset("sequences",
                                  shape=(total, *seq_shape),
                                  dtype="f4",
                                  chunks=(chunk_size, *seq_shape),
                                  compression="lzf")
    static_ds = h5.create_dataset("static_features",
                                  shape=(total, static_dim),
                                  dtype="f4",
                                  chunks=(chunk_size, static_dim),
                                  compression="lzf")
    label_ds  = h5.create_dataset("labels",
                                  shape=(total,),
                                  dtype="i8",
                                  chunks=(chunk_size,),
                                  compression="lzf")

    idx = 0
    for f in tqdm(npz_files, desc="Converting"):
        with np.load(f) as d:
            n = len(d["labels"])
            seq_ds   [idx:idx+n] = d["sequences"]
            static_ds[idx:idx+n] = d["static_features"]
            label_ds [idx:idx+n] = d["labels"]
            idx += n

print(f"Written HDF5 to {h5_path}")
