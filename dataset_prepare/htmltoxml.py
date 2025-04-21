import os
import re
import glob
import requests

# HTMLファイルが保存されているディレクトリ（onedrive内のフォルダ）を指定
html_dir = r"/Users/yamazakiakirafutoshi/Library/CloudStorage/OneDrive-芝浦工業大学教研テナント(SIC)/上岡研/天鳳データ/2009"  # 適宜変更してください

# ダウンロードしたXMLファイルの保存先ディレクトリを作成
xml_dir = os.path.join(html_dir, "downloaded_xml")
os.makedirs(xml_dir, exist_ok=True)

# ログID抽出用の正規表現パターン（例：href="http://tenhou.net/0/?log=2010010100gm-00b9-0000-b32902e7"）
pattern = r'href="http://tenhou\.net/0/\?log=([^"]+)"'

# ディレクトリ内の全HTMLファイルに対して処理
for filepath in glob.glob(os.path.join(html_dir, "*.html")):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 「四鳳」が含まれているかチェック
    if "四鳳" not in content:
        continue

    # 正規表現でログIDを抽出
    log_ids = re.findall(pattern, content)
    
    for log_id in log_ids:
        # 新しいURL形式に変換
        new_url = f"http://tenhou.net/0/log/?{log_id}"
        print(f"Downloading XML for log id {log_id} from {new_url} ...")
        try:
            response = requests.get(new_url)
            if response.status_code == 200:
                # XMLファイル名はログID.xmlとする
                xml_filename = os.path.join(xml_dir, f"{log_id}.xml")
                with open(xml_filename, "wb") as xml_file:
                    xml_file.write(response.content)
                print(f"Saved XML to {xml_filename}")
            else:
                print(f"Failed to download XML for {log_id}. HTTP status: {response.status_code}")
        except Exception as e:
            print(f"Error downloading XML for {log_id}: {e}")