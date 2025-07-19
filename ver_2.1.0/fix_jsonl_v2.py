import json
import re

def remove_newlines_from_jsonl(input_file, output_file):
    """
    JSONLファイル内の改行を削除して、各JSONオブジェクトを1行にまとめる
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        # ファイル全体を読み込む
        content = infile.read()
        
        # JSONオブジェクトを分割する
        # '}}\n{' のパターンで分割
        json_objects = content.split('}\n{')
        
        for i, json_str in enumerate(json_objects):
            # 最初と最後のオブジェクトの括弧を修正
            if i == 0:
                # 最初のオブジェクト：最後に } を追加
                if not json_str.endswith('}'):
                    json_str += '}'
            elif i == len(json_objects) - 1:
                # 最後のオブジェクト：最初に { を追加
                if not json_str.startswith('{'):
                    json_str = '{' + json_str
            else:
                # 中間のオブジェクト：最初に {、最後に } を追加
                json_str = '{' + json_str + '}'
            
            # 不正な改行を除去してJSONとして解析を試行
            try:
                # JSONを解析
                data = json.loads(json_str)
                
                # contentsの各部分のtextフィールドから改行を削除
                if 'contents' in data:
                    for content in data['contents']:
                        if 'parts' in content:
                            for part in content['parts']:
                                if 'text' in part:
                                    # 改行を空白に置換
                                    part['text'] = part['text'].replace('\n', ' ').replace('\r', ' ')
                                    # 複数の空白を1つにまとめる
                                    part['text'] = re.sub(r'\s+', ' ', part['text']).strip()
                
                # JSONを1行で出力
                json.dump(data, outfile, ensure_ascii=False, separators=(',', ':'))
                outfile.write('\n')
                
            except json.JSONDecodeError as e:
                print(f"JSON解析エラー: {e}")
                print(f"問題のある文字列の最初の100文字: {json_str[:100]}")

def alternative_approach(input_file, output_file):
    """
    代替アプローチ：行ベースでJSONオブジェクトを再構築
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        json_buffer = ""
        brace_count = 0
        
        for line in infile:
            json_buffer += line
            
            # 括弧の数をカウント
            brace_count += line.count('{') - line.count('}')
            
            # 括弧が閉じられた場合、完全なJSONオブジェクトとして処理
            if brace_count == 0 and json_buffer.strip():
                try:
                    # JSONを解析
                    data = json.loads(json_buffer.strip())
                    
                    # contentsの各部分のtextフィールドから改行を削除
                    if 'contents' in data:
                        for content in data['contents']:
                            if 'parts' in content:
                                for part in content['parts']:
                                    if 'text' in part:
                                        # 改行を空白に置換
                                        part['text'] = part['text'].replace('\n', ' ').replace('\r', ' ')
                                        # 複数の空白を1つにまとめる
                                        part['text'] = re.sub(r'\s+', ' ', part['text']).strip()
                    
                    # JSONを1行で出力
                    json.dump(data, outfile, ensure_ascii=False, separators=(',', ':'))
                    outfile.write('\n')
                    
                    # バッファをリセット
                    json_buffer = ""
                    
                except json.JSONDecodeError as e:
                    print(f"JSON解析エラー: {e}")
                    print(f"問題のあるJSONの最初の100文字: {json_buffer[:100]}")
                    json_buffer = ""

if __name__ == "__main__":
    input_file = "mahjong_dataset_converted.jsonl"
    output_file = "mahjong_dataset_for_gemini.jsonl"
    
    print(f"処理開始（代替アプローチ）: {input_file} -> {output_file}")
    alternative_approach(input_file, output_file)
    print("処理完了") 