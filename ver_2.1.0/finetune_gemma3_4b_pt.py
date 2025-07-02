import torch
import transformers
from datasets import load_dataset, Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import os
import gc

# GPUメモリクリア
torch.cuda.empty_cache()

# --- 1. 環境設定 & モデルのロード ---
print("--- ベースモデルとトークナイザをロード中... ---")

model_id = "google/gemma-3-4b-pt"

# 4ビット量子化の設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# モデルのロード
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",  # Gemma2推奨設定
)

# トークナイザのロード
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("--- モデルとトークナイザのロードが完了しました ---")

# --- 2. データセットの前処理と拡張 ---
print("\n--- データセットをロード中... ---")

try:
    dataset = load_dataset("json", data_files="/home/ubuntu/Documents/mahjong-XAI/ver_2.1.0/mahjong_dataset.jsonl", split="train")
    print(f"元データセット読み込み完了: {len(dataset)} サンプル")
except Exception as e:
    print(f"データセット読み込みエラー: {e}")
    exit(1)

# データ拡張関数
def augment_dataset(dataset):
    """データを拡張して学習データを増やす"""
    augmented_data = []
    
    for example in dataset:
        # 元データ
        augmented_data.append({
            'instruction': example['instruction'],
            'output': example['output']
        })
        
        # パラフレーズによる拡張
        instruction = example['instruction']
        output = example['output']
        
        # 質問のバリエーション追加
        variations = [
            f"{instruction}について詳しく説明してください。",
            f"{instruction}とは何ですか？",
            f"{instruction}について教えて。",
            f"{instruction}を説明してください。",
            f"{instruction}に関して教えてください。"
        ]
        
        for var in variations:
            if var != instruction and len(var) > 5:  # 重複回避と最小長チェック
                augmented_data.append({
                    'instruction': var,
                    'output': output
                })
    
    return Dataset.from_list(augmented_data)

# データセット拡張
print("--- データセットを拡張中... ---")
expanded_dataset = augment_dataset(dataset)
print(f"拡張後のデータセット: {len(expanded_dataset)} サンプル")

# データを前処理する関数
def preprocess_function(examples):
    inputs = []
    for i in range(len(examples['instruction'])):
        # Gemma-2の対話形式に合わせる
        text = f"<start_of_turn>user\n{examples['instruction'][i]}<end_of_turn>\n<start_of_turn>model\n{examples['output'][i]}<end_of_turn>"
        inputs.append(text)
    
    # トークナイズ
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding=False,  # データコレーターでパディング
        return_tensors=None
    )
    
    # ラベルを設定（言語モデリングのため、入力と同じ）
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    
    return model_inputs

# データセットを前処理
print("--- データセットを前処理中... ---")
tokenized_dataset = expanded_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=expanded_dataset.column_names,
    desc="Tokenizing dataset"
)

print(f"前処理完了: {len(tokenized_dataset)} サンプル")

# --- 3. ファインチューニングの実行 ---
print("\n--- ファインチューニングを開始します ---")

# 改善されたLoRAの設定
lora_config = LoraConfig(
    r=16,  # 8→16に増加（表現力向上）
    lora_alpha=32,  # 16→32に増加（学習の強化）
    lora_dropout=0.05,  # 0.1→0.05に減少（小データセット用）
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# 量子化モデルをPEFTで扱えるように準備
model_for_training = prepare_model_for_kbit_training(base_model)
model_for_training = get_peft_model(model_for_training, lora_config)

# トレーニング可能パラメータ数を表示
model_for_training.print_trainable_parameters()

# データコレーターの設定
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # 因果言語モデリング
)

# 改善されたトレーニング引数の設定
output_dir = "./gemma3-4b-mahjong-qa-pt"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,  # 1e-4→2e-4に増加（小データセット用）
    num_train_epochs=10,  # 3→10に大幅増加
    logging_steps=2,  # より頻繁にログ
    save_steps=25,
    save_total_limit=3,
    report_to="none",
    bf16=True,
    warmup_steps=5,  # 10→5に減少（小データセット用）
    max_grad_norm=1.0,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    logging_dir=f"{output_dir}/logs",
)

# 詳細な学習監視用コールバック
class DetailedCallback:
    """詳細な学習監視用コールバック"""
    def __init__(self, tokenizer, test_questions):
        self.tokenizer = tokenizer
        self.test_questions = test_questions
        self.losses = []
        self.step_count = 0
    
    def on_log(self, logs):
        self.step_count += 1
        if 'loss' in logs:
            self.losses.append(logs['loss'])
            print(f"Step {self.step_count}: Loss = {logs['loss']:.4f}")
            
            # 定期的にサンプル推論（10ステップごと）
            if self.step_count % 10 == 0:
                self.test_inference()
    
    def test_inference(self):
        """学習中のサンプル推論テスト"""
        if self.test_questions:
            question = self.test_questions[0]  # 最初の質問でテスト
            print(f"中間テスト - 質問: {question}")

# テスト用質問
test_questions = [
    "麻雀とは何ですか？",
    "立直（リーチ）について教えてください。",
    "役牌について説明してください。"
]

# コールバック初期化
callback = DetailedCallback(tokenizer, test_questions)

# Trainerの初期化
trainer = Trainer(
    model=model_for_training,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
)

# ファインチューニングの開始
try:
    print("--- トレーニング開始 ---")
    print(f"総ステップ数: {len(tokenized_dataset) // training_args.per_device_train_batch_size // training_args.gradient_accumulation_steps * training_args.num_train_epochs}")
    trainer.train()
    print("--- ファインチューニング完了！ ---")
except Exception as e:
    print(f"トレーニング中にエラーが発生しました: {e}")
    print("バッチサイズやシーケンス長を更に小さくしてみてください")
    import traceback
    traceback.print_exc()

# 最終的なLoRAアダプタの保存
final_adapter_dir = os.path.join(output_dir, "final_adapter")
trainer.save_model(final_adapter_dir)
print(f"アダプタを {final_adapter_dir} に保存しました")

# メモリクリア
del trainer, model_for_training
gc.collect()
torch.cuda.empty_cache()

# --- 4. 詳細な推論による動作確認 ---
print("\n--- ファインチューニング済みモデルで詳細推論を実行します ---")

def detailed_inference_test(model, tokenizer, questions):
    """詳細な推論テストと評価"""
    results = []
    
    for question in questions:
        prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 複数の温度設定でテスト
        temperatures = [0.1, 0.3, 0.7]
        
        for temp in temperatures:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=temp,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    use_cache=False,
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            if "<start_of_turn>model\n" in generated_text:
                answer = generated_text.split("<start_of_turn>model\n")[1]
                if "<end_of_turn>" in answer:
                    answer = answer.split("<end_of_turn>")[0]
            else:
                answer = generated_text
            
            results.append({
                'question': question,
                'temperature': temp,
                'answer': answer.strip()
            })
    
    return results

try:
    # 学習済みアダプタをロードしてベースモデルに適用
    model_for_inference = PeftModel.from_pretrained(base_model, final_adapter_dir)
    
    # データ型を統一
    model_for_inference = model_for_inference.to(torch.bfloat16)
    
    print("\n--- 詳細推論結果 ---")
    
    # 詳細な評価を実行
    test_results = detailed_inference_test(model_for_inference, tokenizer, test_questions)
    
    # 結果を整理して表示
    for question in test_questions:
        print(f"\n📋 質問: {question}")
        print("=" * 60)
        
        question_results = [r for r in test_results if r['question'] == question]
        for result in question_results:
            print(f"🌡️ 温度{result['temperature']}: {result['answer']}")
            print("-" * 40)

except Exception as e:
    print(f"推論中にエラーが発生しました: {e}")
    print("推論はスキップされました")
    import traceback
    traceback.print_exc()

print("\n--- 処理完了 ---")
print(f"最終モデルは {final_adapter_dir} に保存されています")
print("学習ログは ./gemma3-4b-mahjong-qa-pt/logs で確認できます")