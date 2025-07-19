import json

def remove_newlines_from_jsonl(input_file, output_file):
    """
    JSONLファイル内の改行を削除して、各JSONオブジェクトを1行にまとめる
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            line = line.strip()
            if line:  # 空行をスキップ
                try:
                    # JSONを解析
                    data = json.loads(line)
                    
                    # contentsの各部分のtextフィールドから改行を削除
                    if 'contents' in data:
                        for content in data['contents']:
                            if 'parts' in content:
                                for part in content['parts']:
                                    if 'text' in part:
                                        # 改行を空白に置換（完全に削除すると単語がくっついてしまう場合があるため）
                                        part['text'] = part['text'].replace('\n', ' ').replace('\r', ' ')
                                        # 複数の空白を1つにまとめる
                                        import re
                                        part['text'] = re.sub(r'\s+', ' ', part['text']).strip()
                    
                    # JSONを1行で出力
                    json.dump(data, outfile, ensure_ascii=False, separators=(',', ':'))
                    outfile.write('\n')
                    
                except json.JSONDecodeError as e:
                    print(f"JSON解析エラー: {e}")
                    print(f"問題のある行: {line}")

if __name__ == "__main__":
    input_file = "ver_2.1.0/mahjong_dataset_converted.jsonl"
    output_file = "ver_2.1.0/mahjong_dataset_converted_fixed.jsonl"
    
    print(f"処理開始: {input_file} -> {output_file}")
    remove_newlines_from_jsonl(input_file, output_file)
    print("処理完了") 