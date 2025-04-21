import re

# 入力ファイルと出力ファイルのパス（必要に応じて変更してください）
input_file = 'scc20200220.html'
output_file = 'scc20200220_four_player.html'

# リンク中のURLを変換するための正規表現パターン
# http://tenhou.net/0/?log=XXXXXXXX → http://tenhou.net/0/log/?XXXXXXXX
url_pattern = re.compile(r'href="http://tenhou\.net/0/\?log=([^"]+)"')

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

filtered_lines = []
for line in lines:
    # パイプ(|)で区切って各項目に分割
    parts = line.split('|')
    # 項目数が不足している場合はスキップ
    if len(parts) < 3:
        continue

    # ３つ目の項目（インデックス2）をトリムして確認
    third_field = parts[2].strip()
    # 「四鳳」で始まる行のみ対象（例："四鳳南喰赤"、"四鳳東喰赤" など）
    if not third_field.startswith("四鳳"):
        continue

    # 対象の行の中で、リンクURLを NN 用に変換する
    new_line = url_pattern.sub(r'href="http://tenhou.net/0/log/?\1"', line)
    filtered_lines.append(new_line)

# 変換後の行を出力ファイルへ書き出す
with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(filtered_lines)

print(f"Processed {len(filtered_lines)} lines. Output written to {output_file}")