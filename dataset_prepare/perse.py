from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import torch

# ChromeDriverのパスを指定（実際のパスに変更）
service = Service('/Users/yamazakiakirafutoshi/chromedriver')
driver = webdriver.Chrome(service=service)

# 対象の牌譜URL
url = "https://tenhou.net/0/log/?2009022011gm-00a9-0000-d7935c6d"

# ページにアクセス
driver.get(url)
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

# 生の牌譜ログを取得
pre_tag = driver.find_element(By.TAG_NAME, "pre")
raw_log = pre_tag.text

# ログをテキストファイルに保存
with open("tenhou_log.txt", "w", encoding="utf-8") as f:
    f.write(raw_log)
print("牌譜ログを 'tenhou_log.txt' に保存しました。")

# ブラウザを閉じる
driver.quit()

# NN用に牌譜を前処理
def preprocess_mahjong_log(raw_log):
    # プレイヤー情報と初期設定を取得
    un_match = re.search(r'<UN n0="([^"]+)" n1="([^"]+)" n2="([^"]+)" n3="([^"]+)"', raw_log)
    players = [un_match.group(i + 1) for i in range(4)] if un_match else ["P0", "P1", "P2", "P3"]

    # 局の開始（<INIT>）と行動（<T>, <D>）を抽出
    rounds = re.split(r'<INIT ', raw_log)  # 各局を分割
    player_data = {i: [] for i in range(4)}  # プレイヤーごとの行動シーケンス
    round_info = []  # 局ごとの情報

    for i, round_data in enumerate(rounds[1:], 1):  # rounds[0]はヘッダー部分なのでスキップ
        # 局番号を特定（東1局、東2局、...、南4局など）
        round_num = i  # 簡易的に連番で管理（実際は<INIT>のseedから計算可能）
        is_first_round = (i == 1)  # 東1局
        is_last_round = (i == len(rounds) - 1)  # オーラス

        # 行動を抽出
        actions = re.findall(r'<[TD][0-9]+>', round_data)
        current_player = 0  # 親（0）から順番に仮定（実際は<INIT>のoyaで調整）

        for action in actions:
            if action.startswith('<T'):
                # ツモ（誰かが牌を取る）
                tile = int(action[2:-1])
            elif action.startswith('<D'):
                # 捨て牌（誰かが牌を切る）
                tile = int(action[2:-1])
                player_data[current_player].append({
                    'round': round_num,
                    'is_first': is_first_round,
                    'is_last': is_last_round,
                    'tile': tile
                })
                current_player = (current_player + 1) % 4  # 次のプレイヤーへ

        round_info.append({'round': round_num, 'is_first': is_first_round, 'is_last': is_last_round})

    return players, player_data, round_info

# 前処理のテスト
with open("tenhou_log.txt", "r", encoding="utf-8") as f:
    log_data = f.read()
    
  