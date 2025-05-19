#data_fetcher_ubuntu.py
import re
import os
import glob
import requests
from time import sleep
import html # ← これを追加！

# --- 設定 ---
# ... (以下、設定や関数の定義が続く) ...
# --- 設定 ---
# ★重要: 事前に展開した .html ファイルが年ごとに入っている親ディレクトリのパスを指定★
# 例: /home/ubuntu/Documents/Tenhou-dataset
HTML_BASE_DIR = "/home/ubuntu/Documents/tenhou_dataset"
# XMLファイルを保存するディレクトリ
XML_SAVE_DIR = "/home/ubuntu/Documents/xml_logs"
# 処理したい年の範囲
YEAR_RANGE = range(2024, 2022, -1) # ★必要に応じて変更★
# --- 設定ここまで ---

def download_xml_requests(url, save_dir=XML_SAVE_DIR):
    """
    指定されたURLからrequestsを使い、<mjloggm> タグブロックのみを検出・抽出し、
    HTMLエンティティをデコードしてXMLファイルとして保存する。
    <pre> タグの検出は行わない。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # ログID抽出 (ファイル名用)
    log_id_match = re.search(r'\?(\d+gm-[\w-]+)', url)
    if not log_id_match:
        print(f"エラー: URLからログIDを抽出できません: {url}")
        return None
    log_id = log_id_match.group(1)
    file_path = os.path.join(save_dir, f"{log_id}.xml")

    if os.path.exists(file_path):
        print(f"既に存在: {file_path}")
        return file_path

    try:
        print(f"リクエスト開始: {url}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status() # エラーがあれば例外発生

        # 文字コードを推定させる
        response.encoding = response.apparent_encoding
        print(f"推定された文字コード: {response.encoding}")
        html_text = response.text # or response.content.decode(response.encoding)

        # <mjloggm ...> から </mjloggm> までを抽出する正規表現 (これのみ使用)
        # re.DOTALL: 改行を含む任意文字にマッチ
        # re.IGNORECASE: 大文字小文字を区別しない
        mjlog_match = re.search(r'(<mjloggm[^>]*>.*?</mjloggm>)', html_text, re.DOTALL | re.IGNORECASE)

        # mjloggm タグが見つからなければエラー終了
        if not mjlog_match:
            print(f"エラー: <mjloggm>...</mjloggm> タグが見つかりませんでした: {url}")
            print(f"HTML内容抜粋: {html_text[:500]}") # デバッグ用にHTMLの先頭を表示
            return None # フォールバックせずに終了

        # <mjloggm> ブロック全体を抽出
        raw_log = mjlog_match.group(1).strip()
        print("<mjloggm> タグブロックを抽出成功。")

        # 抽出した内容の確認
        print(f"抽出したログの長さ: {len(raw_log)} 文字")
        if not raw_log:
            print(f"警告: {url} のログ内容が空です (mjloggm抽出後)")
            return None

        # HTMLエンティティ (&lt; &gt; など) をデコード
        decoded_log = html.unescape(raw_log)

        # デコード後の内容をUTF-8でファイルに書き込む
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(decoded_log)

        # 保存確認
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            print(f"ダウンロード成功: {file_path}")
            return file_path
        else:
            print(f"エラー: ファイルが生成されなかったか空です: {file_path}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"リクエスト失敗 {url}: {e}")
        return None
    except Exception as e:
        print(f"予期せぬエラー {url}: {e}")
        return None
    finally:
        # サーバー負荷軽減のため、リクエスト間に少し待機を入れる
        sleep(0.5)


def fetch_xml_files(base_dir=HTML_BASE_DIR, year_range=YEAR_RANGE):
    """
    展開済みの .html ファイルを読み込み、URLを抽出してXMLをダウンロードする。
    (この関数は requests 版の download_xml_requests を呼び出す)
    """
    # ★前提: base_dir には展開済みの .html ファイルが入っていること★
    html_files = []
    for year in year_range:
        year_dir = os.path.join(base_dir, str(year))
        if os.path.exists(year_dir):
            # 展開済みの .html ファイルを検索
            found = glob.glob(os.path.join(year_dir, "scc*.html"))
            print(f"年 {year}: {len(found)} 個の .html ファイルを発見")
            html_files.extend(found)
        else:
            print(f"ディレクトリが見つかりません: {year_dir}")

    print(f"発見したHTMLファイル総数: {len(html_files)}")
    if not html_files:
        print(".html ファイルが見つからないため、処理を終了します。")
        print(f"検索したベースディレクトリ: {base_dir}")
        print(f"検索した年範囲: {year_range}")
        return []

    xml_files_downloaded = []

    # 各HTMLファイルを処理
    for i, html_file in enumerate(html_files):
        print(f"\n--- HTML処理中 ({i+1}/{len(html_files)}): {html_file} ---")
        try:
            # 通常の open で .html ファイルを読み込む
            with open(html_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"ファイル読み込みエラー {html_file}: {e}")
            continue

        four_player_urls = []
        # 各行をチェックして「四鳳」を含む行からURLを抽出
        for line_num, line in enumerate(lines):
            if '四鳳' in line:
                match = re.search(r'href=["\'](http://tenhou\.net/0/\?log=([^"\']+))["\']', line)
                if match:
                    original_url = match.group(1)
                    log_param = match.group(2)
                    # URLの形式をログ表示用 'http://tenhou.net/0/log/?...' に変換
                    display_url = f'http://tenhou.net/0/log/?{log_param}'
                    four_player_urls.append(display_url)

        print(f"{os.path.basename(html_file)} から抽出された四鳳URL数: {len(four_player_urls)}")
        if not four_player_urls:
             print(f"URLが見つかりませんでした。HTMLの内容を確認してください。")

        # 抽出したURLごとにXMLファイルをダウンロード
        for url in four_player_urls:
            # ★requests版のダウンロード関数を呼び出す★
            xml_file = download_xml_requests(url, save_dir=XML_SAVE_DIR)
            if xml_file:
                xml_files_downloaded.append(xml_file)

    return xml_files_downloaded

# スクリプトが直接実行された場合にのみ実行
if __name__ == "__main__":
    print("--- Requests版 XMLダウンロードスクリプト開始 ---")
    print(f"HTMLベースディレクトリ (展開済): {HTML_BASE_DIR}")
    print(f"XML保存ディレクトリ: {XML_SAVE_DIR}")
    print(f"処理対象年: {list(YEAR_RANGE)}")
    print("------------")

    if not os.path.exists(HTML_BASE_DIR):
        print(f"エラー: 指定された展開済みHTMLベースディレクトリが存在しません: {HTML_BASE_DIR}")
        print("先に展開用スクリプトを実行したか、パスが正しいか確認してください。")
    else:
        xml_files = fetch_xml_files()
        print(f"\n--- 処理完了 ---")
        print(f"最終的にダウンロード/確認された有効なXMLファイル数: {len(xml_files)}")
        print(f"XMLファイルは {os.path.abspath(XML_SAVE_DIR)} に保存されています。")