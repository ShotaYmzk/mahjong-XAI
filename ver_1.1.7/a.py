import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time # 時間計測用 (任意)

# --- 設定 ---
# 使用するモデルのID (Hugging Face Hub上の名前)
# Instructモデル (対話や指示応答向け) を使うのがおすすめです
model_id = "meta-llama/Llama-3.1-8B-Instruct"


# --- トークナイザーとモデルのロード ---
print(f"Loading tokenizer for {model_id}...")
# トークナイザー（テキストと数値IDを相互変換するツール）をロード
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(f"Loading model {model_id}...")
start_time = time.time()
# モデル本体をロード
# torch_dtype=torch.bfloat16 は比較的新しいGPUで高速。古いGPUでは torch.float16 を試す
# device_map="auto" で自動的にGPU (利用可能なら) にモデルを配置
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
end_time = time.time()
print(f"Model loaded in {end_time - start_time:.2f} seconds.")

# --- プロンプトの準備 ---
# Llama 3 Instruct モデルに適したチャット形式でプロンプトを作成
messages = [
    # システムメッセージ (モデルの役割を与える)
    {"role": "system", "content": "あなたは親切なAIアシスタントです。"},
    # ユーザーメッセージ (モデルへの指示や質問)
    {"role": "user", "content": "あなたの研究テーマである麻雀AIのXAIについて、面白い応用例を一つ提案してください。"},
]

# トークナイザーを使って、チャット形式をモデルが理解できる数値IDのリストに変換
# add_generation_prompt=True で、モデルが応答を生成し始めるための指示を追加
# return_tensors="pt" でPyTorchテンソル形式に
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device) # モデルと同じデバイス (GPU/CPU) に配置

# --- テキスト生成の実行 ---
print("Generating response...")
start_time = time.time()

# テキスト生成の終了条件となるトークンIDを設定
# Llama 3 では、通常の終了トークン (eos_token_id) と <|eot_id|> が使われる
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# モデルにテキスト生成を実行させる
# max_new_tokens: 生成する最大のトークン数
# do_sample=True: ランダム性を持たせる (Falseだと毎回同じ結果になりやすい)
# temperature: ランダム性の度合い (低いほど決定的、高いほど多様)
# top_p: 生成候補を絞る方法の一つ (0.9程度が一般的)
outputs = model.generate(
    input_ids,
    max_new_tokens=512, # 生成する最大トークン数を増やす
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.7, # 少し高めて多様性を出す
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id # pad_token_id を設定
)
end_time = time.time()
print(f"Response generated in {end_time - start_time:.2f} seconds.")

# --- 結果の表示 ---
# 生成されたIDリストから、入力部分を除いた応答部分だけを取り出す
response_ids = outputs[0][input_ids.shape[-1]:]
# 応答部分のIDリストを、トークナイザーで人間が読めるテキストに戻す
response = tokenizer.decode(response_ids, skip_special_tokens=True)

print("\n--- AI Response ---")
print(response)
print("------------------")
