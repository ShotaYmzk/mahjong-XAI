import torch
import transformers
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# GPUメモリクリア
torch.cuda.empty_cache()

# --- 設定 ---ここを変える
#model_id = "google/gemma-2-2b-it"  # ベースモデル
#adapter_path = "./gemma2-2b-mahjong-qa-improved/final_adapter"  # ファインチューニング済みアダプタのパス

model_id = "google/gemma-3-4b-pt"
adapter_path = "./gemma3-4b-mahjong-qa-pt/final_adapter"


print("--- ファインチューニング済みモデルをロード中... ---")

# 4ビット量子化の設
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ベースモデルのロード
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)

# トークナイザのロード
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ファインチューニング済みアダプタをロード（比較用にコメントアウト）
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.to(torch.bfloat16)

# 普通のGemmaモデルを使用（ファインチューニングなし）
#model = base_model

print("--- モデルロード完了！ ---")

# --- 推論関数 ---
def ask_mahjong_question(question, temperature=0.7, max_tokens=1500):
    """
    麻雀に関する質問をモデルに送信して回答を取得
    
    Args:
        question (str): 質問文
        temperature (float): 生成の温度 (0.1-1.0, 低いほど決定的)
        max_tokens (int): 最大生成トークン数
    
    Returns:
        str: モデルの回答
    """
    # プロンプトの作成
    prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
    
    # トークナイズ
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 推論実行
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            use_cache=False,
        )
    
    # 結果をデコード
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 回答部分を抽出
    if "<start_of_turn>model\n" in generated_text:
        answer = generated_text.split("<start_of_turn>model\n")[1]
        if "<end_of_turn>" in answer:
            answer = answer.split("<end_of_turn>")[0]
    else:
        answer = generated_text
    
    return answer.strip()

# --- 使用例 ---
if __name__ == "__main__":
    print("\n🀄 麻雀AI助手へようこそ！")
    print("麻雀に関する質問をしてください。'quit'で終了します。\n")
    
    # 対話形式で質問を受け付け
    while True:
        try:
            # ユーザーからの入力
            user_question = input("🙋 質問: ")
            
            # 終了チェック
            if user_question.lower() in ['quit', 'exit', 'q', '終了']:
                print("👋 ありがとうございました！")
                break
            
            # 空の入力チェック
            if not user_question.strip():
                continue
            
            # 推論実行
            print("🤖 考え中...")
            answer = ask_mahjong_question(user_question)
            
            # 結果表示
            print(f"💬 回答: {answer}\n")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 ありがとうございました！")
            break
        except Exception as e:
            print(f"❌ エラーが発生しました: {e}")
            continue

# --- 事前定義済み質問でのテスト ---
def test_predefined_questions():
    """事前に定義された質問でテスト"""
    print("\n--- 事前定義質問テスト ---")
    
    test_questions = [
        "麻雀とは何ですか？",
        "立直（リーチ）について教えてください。",
        "役牌について説明してください。",
        "ツモとロンの違いは何ですか？",
        "チーとポンの違いを教えてください。",
        "麻雀の点数計算方法を教えてください。",
        "一発について説明してください。",
        "ドラとは何ですか？"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📋 質問{i}: {question}")
        
        # 複数の温度設定でテスト
        temperatures = [0.1, 0.3, 0.7]
        
        for temp in temperatures:
            answer = ask_mahjong_question(question, temperature=temp, max_tokens=150)
            print(f"🌡️ 温度{temp}: {answer}")
            print("-" * 30)

# テスト実行用（コメントアウトを外すとテスト実行）
# test_predefined_questions()