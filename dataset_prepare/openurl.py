import webbrowser
import re

# ファイルから全文を読み込む
with open("scc20200220_four_player.html", "r", encoding="utf-8") as f:
    data = f.read()

# NN用URL（http://tenhou.net/0/log/?以降の文字列）の抽出
match = re.search(r'href=["\'](http://tenhou\.net/0/log/\?[^"\']+)["\']', data)
if match:
    url = match.group(1)
    print("開くURL:", url)
    webbrowser.open(url)
else:
    print("URLが見つかりませんでした。")