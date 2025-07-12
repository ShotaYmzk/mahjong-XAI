from google import genai
from google.genai import types
import base64
import os
import sys
import argparse
from pathlib import Path

def generate_for_prompt(prompt_text, client, model):
    """プロンプトテキストからLLMの出力を生成"""
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(text=prompt_text)
            ]
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=1,
        seed=0,
        max_output_tokens=65535,
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
            )
        ],
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1,
        ),
    )

    # ストリーミングで出力を取得
    full_response = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.text:
            full_response += chunk.text
            print(chunk.text, end="")
    
    return full_response

def process_analysis_results(analysis_results_path):
    """analysis_resultsディレクトリの処理"""
    client = genai.Client(
        vertexai=True,
        project="635939494644",
        location="us-central1",
    )

    model = "projects/635939494644/locations/us-central1/endpoints/429703437786021888"
    
    analysis_path = Path(analysis_results_path)
    if not analysis_path.exists():
        print(f"[エラー] 指定されたパスが存在しません: {analysis_results_path}")
        return
    
    # tsumo_xxフォルダを探す
    tsumo_dirs = sorted([d for d in analysis_path.iterdir() 
                        if d.is_dir() and d.name.startswith("tsumo_")])
    
    if not tsumo_dirs:
        print(f"[エラー] tsumo_xxフォルダが見つかりません: {analysis_results_path}")
        return
    
    print(f"発見したtsumo_xxフォルダ: {len(tsumo_dirs)}個")
    
    for i, tsumo_dir in enumerate(tsumo_dirs):
        prompt_file = tsumo_dir / "prompt.txt"
        output_file = tsumo_dir / "output.txt"
        
        if not prompt_file.exists():
            print(f"[警告] prompt.txtが見つかりません: {prompt_file}")
            continue
        
        if output_file.exists():
            print(f"[スキップ] output.txtが既に存在します: {output_file}")
            continue
        
        print(f"\n=== 処理中: {tsumo_dir.name} ({i+1}/{len(tsumo_dirs)}) ===")
        
        try:
            # prompt.txtの読み込み
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_text = f.read()
            
            print(f"プロンプトファイル読み込み完了: {len(prompt_text)}文字")
            print("LLMによる分析を開始...")
            
            # LLMで分析
            response = generate_for_prompt(prompt_text, client, model)
            
            # output.txtに保存
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response)
            
            print(f"\n出力保存完了: {output_file}")
            
        except Exception as e:
            print(f"[エラー] {tsumo_dir.name}の処理に失敗: {e}")
            continue
    
    print(f"\n=== 処理完了 ===")

def main():
    parser = argparse.ArgumentParser(description="analysis_resultsからプロンプトを読み込んでLLM分析を実行")
    parser.add_argument("analysis_dir", help="analysis_resultsの中の特定のフォルダパス (例: analysis_results/kawanowa_R2_P0)")
    parser.add_argument("--force", action='store_true', help="既存のoutput.txtを上書きする")
    
    args = parser.parse_args()
    
    if args.force:
        # 強制上書きモードの場合、既存のoutput.txtを削除
        analysis_path = Path(args.analysis_dir)
        if analysis_path.exists():
            for output_file in analysis_path.rglob("output.txt"):
                output_file.unlink()
                print(f"削除: {output_file}")
    
    process_analysis_results(args.analysis_dir)

if __name__ == "__main__":
    main()