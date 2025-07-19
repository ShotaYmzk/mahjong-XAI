import json

def convert_format(input_file, output_file):
    """
    mahjong_dataset.jsonlをsft_train_data.jsonlの形式に変換する
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            # 元のデータを読み込み
            data = json.loads(line.strip())
            
            # 新しい形式に変換
            converted_data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": data["instruction"]}]
                    },
                    {
                        "role": "model", 
                        "parts": [{"text": data["output"]}]
                    }
                ]
            }
            
            # 変換後のデータを書き込み
            f_out.write(json.dumps(converted_data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    input_file = "mahjong_dataset.jsonl"
    output_file = "mahjong_dataset_converted.jsonl"
    
    convert_format(input_file, output_file)
    print(f"変換完了: {input_file} -> {output_file}") 