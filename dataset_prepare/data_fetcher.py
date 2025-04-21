#このコードでは、天鳳のログをダウンロードするための関数を定義しています。
#天鳳のログは、特定のURLからXML形式で取得されます。
#バックグラウンドでSeleniumを使用して、指定されたURLからXMLログをダウンロードし、指定されたディレクトリに保存します。
import re
import os
import glob
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options # ChromeOptionsをインポート

def download_xml_selenium(url, save_dir="../xml_logs"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_id = url.split('?')[-1].replace('log=', '')
    file_path = os.path.join(save_dir, f"{log_id}.xml")
    if os.path.exists(file_path):
        print(f"既に存在: {file_path}")
        return file_path

    # --- ここから変更 ---
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # ヘッドレスモードを有効にする
    chrome_options.add_argument("--no-sandbox") # Sandboxなしで実行（ヘッドレスでよく必要）
    chrome_options.add_argument("--disable-dev-shm-usage") # /dev/shmの使用を無効化（リソース制限のある環境で有効）
    chrome_options.add_argument("--window-size=1920,1080") # ウィンドウサイズを指定（一部サイトで必要）
    # --- ここまで変更 ---

    # ChromeDriverのパスはご自身の環境に合わせてください
    service = Service('/Users/yamazakiakirafutoshi/chromedriver-mac-arm64/chromedriver')
    # driverの初期化時にoptionsを追加
    driver = webdriver.Chrome(service=service, options=chrome_options)
    try:
        print(f"アクセス開始 (ヘッドレス): {url}")
        driver.get(url)
        # ページの<pre>タグが表示されるまで最大15秒待機
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "pre")))
        pre_tag = driver.find_element(By.TAG_NAME, "pre")
        raw_log = pre_tag.text
        print(f"取得したログの長さ: {len(raw_log)} 文字")
        if not raw_log.strip():
            print(f"警告: {url} のログが空です")
            return None
        # 保存前にログの一部を表示して確認（デバッグ用）
        print(f"保存前確認: ログの先頭100文字: {raw_log[:100]}")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(raw_log)
        # ファイルが正しく保存されたか確認
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            print(f"ダウンロード成功: {file_path}")
            return file_path
        else:
            print(f"エラー: ファイルが生成されなかったか空です: {file_path}")
            return None
    except Exception as e:
        print(f"ダウンロード失敗 {url}: {e}")
        # エラー発生時もNoneを返すようにする
        return None
    finally:
        # ブラウザを必ず閉じる
        driver.quit()

def fetch_xml_files(base_dir="/Users/yamazakiakirafutoshi/Library/CloudStorage/OneDrive-芝浦工業大学教研テナント(SIC)/上岡研/天鳳データ"):
    # 2009年から2024年までのフォルダを検索（2025は含まない）
    html_files = []
    for year in range(2009, 2015): # 2009年から2024年まで
        year_dir = os.path.join(base_dir, str(year))
        if os.path.exists(year_dir):
            # 指定された年のディレクトリ内のすべてのscc*.htmlファイルを取得
            html_files.extend(glob.glob(os.path.join(year_dir, "scc*.html")))

    print(f"発見したHTMLファイル数: {len(html_files)}")
    xml_files = []

    # 各HTMLファイルを処理
    for html_file in html_files:
        try:
            with open(html_file, "r", encoding="utf-8") as f:
                data = f.read()
        except Exception as e:
            print(f"ファイル読み込みエラー {html_file}: {e}")
            continue # 次のファイルへ

        # HTMLデータを行に分割
        lines = data.split('<br>')
        four_player_urls = []
        # 各行をチェックして「四鳳」を含む行からURLを抽出
        for line in lines:
            if '四鳳' in line:
                # href属性からURLを正規表現で抽出
                match = re.search(r'href=["\'](http://tenhou\.net/0/\?log=[^"\']+)["\']', line)
                if match:
                    # URLの形式を変換
                    url = match.group(1).replace('http://tenhou.net/0/?log=', 'http://tenhou.net/0/log/?')
                    four_player_urls.append(url)
        print(f"{html_file} から抽出された四鳳URL数: {len(four_player_urls)}")

        # 抽出したURLごとにXMLファイルをダウンロード
        for url in four_player_urls:
            xml_file = download_xml_selenium(url)
            # ダウンロードが成功した場合のみリストに追加
            if xml_file:
                xml_files.append(xml_file)

    return xml_files

# スクリプトが直接実行された場合にのみ実行
if __name__ == "__main__":
    xml_files = fetch_xml_files()
    print(f"最終的にダウンロードされた有効なXMLファイル数: {len(xml_files)}")