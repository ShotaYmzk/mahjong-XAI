import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import os

# デバッグ用の環境変数を設定
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# --- 1. 環境設定 & モデルのロード ---
# このスクリプトを実行する前に、ターミナルで `huggingface-cli login` を実行して
# Hugging Face Hubにログインしておいてください。

# モデルID
model_id = "google/gemma-2-2b-it"  # より安定したモデルに変更

# 4ビット量子化の設定（より保守的な設定に変更）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # float16からbfloat16に変更（数値安定性向上）
    bnb_4bit_use_double_quant=True,  # 二重量子化を有効化
)

print(f"--- ファインチューニング前のモデル ({model_id}) をロード中... ---")

# モデルのロード
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",  # "cuda"から"auto"に変更
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # データ型を明示的に指定
)

# トークナイザのロード
tokenizer = AutoTokenizer.from_pretrained(model_id)

# パディングトークンが設定されていない場合は設定
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("--- モデルのロードが完了しました ---")

# --- 2. 推論の実行 ---
print("\n--- 推論を実行します ---")

# Hugging Faceの推論パイプラインを作成
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# モデルに聞きたい質問
question = "麻雀とはなに？"

# プロンプトの作成 (Gemma-2が学習した対話形式に合わせる)
prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"

try:
    # 推論の実行（より保守的なパラメータに変更）
    outputs = pipe(
        prompt,
        max_new_tokens=256,  # トークン数を減らして安定性向上
        do_sample=True,
        temperature=0.3,  # より低い温度で安定性向上
        top_k=20,  # より小さな値で安定性向上
        top_p=0.9,  # より保守的な値
        pad_token_id=tokenizer.eos_token_id,  # パディングトークンを明示的に指定
        repetition_penalty=1.1,  # 繰り返し防止
    )
    
    # --- 3. 結果の表示 ---
    print("\n" + "="*20 + " 推論結果 " + "="*20)
    
    # モデルが生成した回答部分だけを綺麗に抽出して表示
    generated_text = outputs[0]["generated_text"]
    
    # "<start_of_turn>model\n" の後がモデルの回答
    if "<start_of_turn>model\n" in generated_text:
        answer = generated_text.split("<start_of_turn>model\n")[1]
        # 不要な終了タグがあれば除去
        if "<end_of_turn>" in answer:
            answer = answer.split("<end_of_turn>")[0]
    else:
        answer = generated_text
    
    print(f"質問: {question}\n")
    print("モデルの回答:")
    print(answer.strip())
    print("="*52)

except Exception as e:
    print(f"エラーが発生しました: {e}")
    print("以下の解決策を試してください：")
    print("1. より小さなモデル（gemma-2-2b-it）を使用")
    print("2. GPUメモリをクリア: torch.cuda.empty_cache()")
    print("3. より保守的な生成パラメータを使用")
    
    # GPUメモリをクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()