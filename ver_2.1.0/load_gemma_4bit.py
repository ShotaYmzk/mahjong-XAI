# load_gemma_4bit.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# モデルID
model_id = "google/gemma-3-4b-it" # 指示チューニング済みモデル

# 4ビット量子化の設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# モデルのロード
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="cuda", # 自動的にGPUに配置
    trust_remote_code=True,
)

# トークナイザのロード
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token # パディングトークンを設定