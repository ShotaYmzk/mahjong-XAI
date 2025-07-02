# validation.py
from peft import PeftModel
import torch

# ベースモデルとトークナイザを再ロード（推論用）
base_model_id = "google/gemma-3-4b-it"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# 学習したLoRAアダプタをロードしてベースモデルに適用
model = PeftModel.from_pretrained(base_model, "./gemma3-4b-mahjong-final")

# 推論パイプラインの作成
pipe = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

# テスト用の局面情報
test_instruction = """
麻雀とは？
"""

# プロンプトの作成 (学習時と全く同じ形式にすることが重要！)
prompt = f"<start_of_turn>user\n{test_instruction}<end_of_turn>\n<start_of_turn>model\n"

# 推論の実行
outputs = pipe(
    prompt,
    max_new_tokens=512, # 生成する最大トークン数
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
)

# 結果の表示
print(outputs[0]["generated_text"])