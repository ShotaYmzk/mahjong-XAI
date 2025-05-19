import os
import gzip
from glob import glob

# 元のファイルがあるディレクトリ
base_dir = os.path.expanduser('~/Documents/Tenhou_logs')

# 新しく整理したファイルを保存するディレクトリ
target_base_dir = os.path.expanduser('~/Documents/tenhou_dataset')
os.makedirs(target_base_dir, exist_ok=True)

# 各年度ごとのフォルダを処理
for year in range(2019, 2024):
    source_dir = os.path.join(base_dir, str(year))
    target_dir = os.path.join(target_base_dir, str(year))
    os.makedirs(target_dir, exist_ok=True)

    # sccで始まるHTMLファイルのみを取得
    pattern = os.path.join(source_dir, 'scc*.html.gz')
    for gz_path in glob(pattern):
        # .gz拡張子を外してHTMLファイル名を生成
        file_name = os.path.basename(gz_path)
        html_name = file_name[:-3]  # .gzを削除
        target_path = os.path.join(target_dir, html_name)

        # gzipから読み込み、解凍して保存
        with gzip.open(gz_path, 'rt', encoding='utf-8', errors='ignore') as f_in:
            content = f_in.read()
        with open(target_path, 'w', encoding='utf-8') as f_out:
            f_out.write(content)

        print(f"Decompressed {gz_path} to {target_path}")
