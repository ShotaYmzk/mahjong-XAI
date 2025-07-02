#!/usr/bin/env python3
"""
麻雀AI分析説明生成システム (Gemma Powered)
========================================

このモジュールは、prompt.pyで生成された分析プロンプトを使用して、
ファインチューニング済みGemmaモデルで高品質な麻雀戦術説明を生成します。

Features:
- 複数のプロンプトスタイル対応 (tactical, quantitative, comparative)
- バッチ処理による効率的な説明生成
- 高度なメモリ管理とパフォーマンス最適化
- 詳細なログとエラーハンドリング
- 多様な出力フォーマット (JSON, Markdown, HTML)
- GPU最適化と量子化対応

Author: Mahjong XAI Research Team
Version: 2.1.0 Enhanced
"""

import os
import sys
import json
import logging
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import time

import torch
import transformers
from peft import PeftModel
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    GenerationConfig
)
import yaml

# カスタムロガー設定
def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """高性能ロガーセットアップ"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        # コンソールハンドラ
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # ファイルハンドラ
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"explain_gemma_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

logger = setup_logger(__name__)

@dataclass
class ModelConfig:
    """モデル設定クラス"""
    model_id: str = "google/gemma-3-4b-pt"
    adapter_path: str = "./gemma3-4b-mahjong-qa-pt/final_adapter"
    device_map: str = "auto"
    torch_dtype: torch.dtype = torch.bfloat16
    attn_implementation: str = "eager"
    use_quantization: bool = True
    trust_remote_code: bool = True

@dataclass
class GenerationConfig:
    """生成設定クラス"""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 2000
    do_sample: bool = True
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    use_cache: bool = True
    pad_token_id: Optional[int] = None

@dataclass
class ProcessingStats:
    """処理統計クラス"""
    total_prompts: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    total_tokens_generated: int = 0
    total_processing_time: float = 0.0
    average_generation_time: float = 0.0
    memory_usage_peak: float = 0.0

class MahjongExplainerGemma:
    """
    麻雀AI分析説明生成システム
    
    ファインチューニング済みGemmaモデルを使用して、
    高品質な麻雀戦術説明を生成するクラス。
    """
    
    def __init__(
        self, 
        model_config: ModelConfig,
        generation_config: GenerationConfig,
        cache_dir: Optional[str] = None
    ):
        """
        初期化
        
        Args:
            model_config: モデル設定
            generation_config: 生成設定
            cache_dir: キャッシュディレクトリ
        """
        self.model_config = model_config
        self.generation_config = generation_config
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.stats = ProcessingStats()
        
        logger.info("MahjongExplainerGemma initialized")
    
    def setup_quantization(self) -> Optional[BitsAndBytesConfig]:
        """量子化設定のセットアップ"""
        if not self.model_config.use_quantization:
            return None
            
        logger.info("Setting up 4-bit quantization")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.model_config.torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
    
    def load_model(self) -> None:
        """モデルとトークナイザーの読み込み"""
        try:
            logger.info(f"Loading base model: {self.model_config.model_id}")
            
            # GPU メモリクリア
            torch.cuda.empty_cache()
            
            # 量子化設定
            bnb_config = self.setup_quantization()
            
            # ベースモデル読み込み
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_id,
                quantization_config=bnb_config,
                device_map=self.model_config.device_map,
                trust_remote_code=self.model_config.trust_remote_code,
                torch_dtype=self.model_config.torch_dtype,
                attn_implementation=self.model_config.attn_implementation,
                cache_dir=str(self.cache_dir),
            )
            
            # トークナイザー読み込み
            logger.info("Loading tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_id,
                cache_dir=str(self.cache_dir)
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            
            # ファインチューニング済みアダプタ読み込み（比較用にコメントアウト）
            # if os.path.exists(self.model_config.adapter_path):
            #     logger.info(f"Loading fine-tuned adapter: {self.model_config.adapter_path}")
            #     self.model = PeftModel.from_pretrained(
            #         self.model, 
            #         self.model_config.adapter_path
            #     )
            #     self.model = self.model.to(self.model_config.torch_dtype)
            # else:
            #     logger.warning(f"Adapter path not found: {self.model_config.adapter_path}")
            
            logger.info("Using base model without fine-tuned adapter for comparison")
            
            # 生成設定のpad_token_idを設定
            self.generation_config.pad_token_id = self.tokenizer.eos_token_id
            
            logger.info("Model loading completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def create_prompt(self, user_input: str) -> str:
        """Gemma用プロンプト形式の作成"""
        return f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"
    
    def extract_response(self, generated_text: str) -> str:
        """生成されたテキストから回答部分を抽出"""
        if "<start_of_turn>model\n" in generated_text:
            answer = generated_text.split("<start_of_turn>model\n")[1]
            if "<end_of_turn>" in answer:
                answer = answer.split("<end_of_turn>")[0]
        else:
            answer = generated_text
        
        return answer.strip()
    
    def generate_explanation(
        self, 
        prompt_text: str,
        custom_generation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        麻雀戦術説明の生成
        
        Args:
            prompt_text: 入力プロンプト
            custom_generation_config: カスタム生成設定
            
        Returns:
            生成結果の辞書
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # プロンプト準備
            formatted_prompt = self.create_prompt(prompt_text)
            
            # トークナイズ
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=4096  # コンテキスト長制限
            ).to(self.model.device)
            
            input_length = inputs['input_ids'].shape[1]
            
            # 生成設定の準備
            gen_config = self.generation_config.__dict__.copy()
            if custom_generation_config:
                gen_config.update(custom_generation_config)
            
            logger.info(f"Generating explanation (input tokens: {input_length})")
            
            # 説明生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_config,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # 結果のデコード
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=False
            )
            
            # 回答抽出
            explanation = self.extract_response(generated_text)
            
            # 統計更新
            generation_time = time.time() - start_time
            output_length = len(outputs[0]) - input_length
            
            self.stats.successful_generations += 1
            self.stats.total_tokens_generated += output_length
            self.stats.total_processing_time += generation_time
            
            # メモリ使用量記録
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
                self.stats.memory_usage_peak = max(self.stats.memory_usage_peak, memory_used)
            
            result = {
                "explanation": explanation,
                "input_tokens": input_length,
                "output_tokens": output_length,
                "generation_time": generation_time,
                "success": True,
                "error": None,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Generated explanation ({output_length} tokens in {generation_time:.2f}s)")
            return result
            
        except Exception as e:
            self.stats.failed_generations += 1
            logger.error(f"Generation failed: {e}")
            
            return {
                "explanation": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "generation_time": time.time() - start_time,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def process_prompt_file(
        self, 
        prompt_file: Path,
        output_dir: Optional[Path] = None,
        format_type: str = "markdown"
    ) -> Dict[str, Any]:
        """
        プロンプトファイルの処理
        
        Args:
            prompt_file: プロンプトファイルのパス
            output_dir: 出力ディレクトリ
            format_type: 出力フォーマット ('markdown', 'json', 'html')
            
        Returns:
            処理結果
        """
        try:
            # プロンプト読み込み
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_text = f.read()
            
            logger.info(f"Processing prompt file: {prompt_file}")
            
            # 説明生成
            result = self.generate_explanation(prompt_text)
            
            if result["success"]:
                # 出力ファイル準備
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(exist_ok=True)
                    
                    base_name = prompt_file.stem
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # フォーマット別出力
                    if format_type == "markdown":
                        output_file = output_dir / f"{base_name}_explanation_{timestamp}.md"
                        self._save_markdown(result, output_file, prompt_file)
                    elif format_type == "json":
                        output_file = output_dir / f"{base_name}_explanation_{timestamp}.json"
                        self._save_json(result, output_file, prompt_file)
                    elif format_type == "html":
                        output_file = output_dir / f"{base_name}_explanation_{timestamp}.html"
                        self._save_html(result, output_file, prompt_file)
                    
                    result["output_file"] = str(output_file)
                    logger.info(f"Saved explanation to: {output_file}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process prompt file {prompt_file}: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _save_markdown(self, result: Dict[str, Any], output_file: Path, prompt_file: Path) -> None:
        """Markdown形式で保存"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 麻雀AI分析説明\n\n")
            f.write(f"**生成日時**: {result['timestamp']}\n")
            f.write(f"**元プロンプト**: {prompt_file.name}\n")
            f.write(f"**生成時間**: {result['generation_time']:.2f}秒\n")
            f.write(f"**出力トークン数**: {result['output_tokens']}\n\n")
            f.write("## AI分析による戦術解説\n\n")
            f.write(result['explanation'])
            f.write("\n\n---\n")
            f.write("*Generated by MahjongExplainerGemma v2.1.0*\n")
    
    def _save_json(self, result: Dict[str, Any], output_file: Path, prompt_file: Path) -> None:
        """JSON形式で保存"""
        output_data = {
            "metadata": {
                "source_prompt_file": str(prompt_file),
                "generation_timestamp": result['timestamp'],
                "model_version": "MahjongExplainerGemma v2.1.0"
            },
            "generation_stats": {
                "input_tokens": result['input_tokens'],
                "output_tokens": result['output_tokens'],
                "generation_time": result['generation_time'],
                "success": result['success']
            },
            "explanation": result['explanation']
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    def _save_html(self, result: Dict[str, Any], output_file: Path, prompt_file: Path) -> None:
        """HTML形式で保存"""
        html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>麻雀AI分析説明</title>
    <style>
        body {{ font-family: 'Noto Sans JP', sans-serif; line-height: 1.6; margin: 40px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }}
        .content {{ background: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 20px; }}
        .stats {{ background: #e9ecef; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .explanation {{ white-space: pre-wrap; background: white; padding: 20px; border-radius: 5px; border-left: 4px solid #667eea; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🀄 麻雀AI分析説明</h1>
        <p>Generated by MahjongExplainerGemma v2.1.0</p>
    </div>
    
    <div class="content">
        <div class="stats">
            <h3>📊 生成統計</h3>
            <p><strong>生成日時:</strong> {result['timestamp']}</p>
            <p><strong>元プロンプト:</strong> {prompt_file.name}</p>
            <p><strong>生成時間:</strong> {result['generation_time']:.2f}秒</p>
            <p><strong>出力トークン数:</strong> {result['output_tokens']}</p>
        </div>
        
        <h2>🧠 AI分析による戦術解説</h2>
        <div class="explanation">{result['explanation']}</div>
    </div>
</body>
</html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def batch_process(
        self, 
        prompt_files: List[Path],
        output_dir: Path,
        format_type: str = "markdown"
    ) -> Dict[str, Any]:
        """
        複数プロンプトファイルのバッチ処理
        
        Args:
            prompt_files: プロンプトファイルのリスト
            output_dir: 出力ディレクトリ
            format_type: 出力フォーマット
            
        Returns:
            バッチ処理結果
        """
        self.stats.total_prompts = len(prompt_files)
        batch_results = []
        
        logger.info(f"Starting batch processing of {len(prompt_files)} files")
        
        for i, prompt_file in enumerate(prompt_files, 1):
            logger.info(f"Processing {i}/{len(prompt_files)}: {prompt_file.name}")
            
            result = self.process_prompt_file(prompt_file, output_dir, format_type)
            batch_results.append({
                "file": str(prompt_file),
                "result": result
            })
            
            # メモリクリーンアップ
            if i % 5 == 0:  # 5ファイルごと
                torch.cuda.empty_cache()
        
        # 統計計算
        self.stats.average_generation_time = (
            self.stats.total_processing_time / max(self.stats.successful_generations, 1)
        )
        
        batch_summary = {
            "total_files": len(prompt_files),
            "successful": self.stats.successful_generations,
            "failed": self.stats.failed_generations,
            "total_tokens_generated": self.stats.total_tokens_generated,
            "average_generation_time": self.stats.average_generation_time,
            "peak_memory_usage_gb": self.stats.memory_usage_peak,
            "results": batch_results
        }
        
        logger.info(f"Batch processing completed: {self.stats.successful_generations}/{len(prompt_files)} successful")
        return batch_summary
    
    def cleanup(self) -> None:
        """リソースクリーンアップ"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        torch.cuda.empty_cache()
        logger.info("Model cleanup completed")

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="麻雀AI分析説明生成システム (Gemma Powered)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 単一ファイル処理
  python explain_gemma.py prompt_tactical_analysis_20241229.txt
  
  # バッチ処理
  python explain_gemma.py prompt_*.txt --batch --output-dir ./explanations
  
  # 対話モード
  python explain_gemma.py --interactive
        """
    )
    
    parser.add_argument(
        "input_files", 
        nargs="*", 
        help="プロンプトファイル(複数可)"
    )
    parser.add_argument(
        "--batch", 
        action="store_true", 
        help="バッチ処理モード"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default="./explanations",
        help="出力ディレクトリ (default: ./explanations)"
    )
    parser.add_argument(
        "--format", 
        choices=["markdown", "json", "html"], 
        default="markdown",
        help="出力フォーマット (default: markdown)"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true", 
        help="対話モード"
    )
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO",
        help="ログレベル (default: INFO)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="生成の温度設定 (default: 0.7)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2000,
        help="最大生成トークン数 (default: 2000)"
    )
    
    args = parser.parse_args()
    
    # ログレベル設定
    global logger
    logger = setup_logger(__name__, args.log_level)
    
    try:
        # モデル設定
        model_config = ModelConfig()
        generation_config = GenerationConfig(
            temperature=args.temperature,
            max_new_tokens=args.max_tokens
        )
        
        # システム初期化
        explainer = MahjongExplainerGemma(
            model_config=model_config,
            generation_config=generation_config
        )
        
        # モデル読み込み
        logger.info("Initializing Mahjong Explainer System...")
        explainer.load_model()
        
        if args.interactive:
            # 対話モード
            interactive_mode(explainer)
        
        elif args.input_files:
            # ファイル処理モード
            input_files = []
            for pattern in args.input_files:
                if "*" in pattern or "?" in pattern:
                    import glob
                    input_files.extend([Path(f) for f in glob.glob(pattern)])
                else:
                    input_files.append(Path(pattern))
            
            if not input_files:
                logger.error("No input files found")
                return 1
            
            # 出力ディレクトリ作成
            args.output_dir.mkdir(exist_ok=True)
            
            if args.batch or len(input_files) > 1:
                # バッチ処理
                results = explainer.batch_process(
                    input_files, 
                    args.output_dir, 
                    args.format
                )
                
                # バッチ結果保存
                summary_file = args.output_dir / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Batch summary saved to: {summary_file}")
                print(f"\n🎯 バッチ処理完了!")
                print(f"📁 結果サマリー: {summary_file}")
                print(f"✅ 成功: {results['successful']}/{results['total_files']}")
                
            else:
                # 単一ファイル処理
                result = explainer.process_prompt_file(
                    input_files[0], 
                    args.output_dir, 
                    args.format
                )
                
                if result["success"]:
                    print(f"\n✅ 説明生成完了!")
                    print(f"📁 出力ファイル: {result.get('output_file', 'N/A')}")
                    print(f"📊 統計: {result['output_tokens']}トークン, {result['generation_time']:.2f}秒")
                else:
                    print(f"\n❌ 説明生成失敗: {result.get('error', 'Unknown error')}")
        
        else:
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return 1
    
    finally:
        # クリーンアップ
        if 'explainer' in locals():
            explainer.cleanup()
    
    return 0

def interactive_mode(explainer: MahjongExplainerGemma) -> None:
    """対話モード"""
    print("\n🀄 麻雀AI分析説明システム (対話モード)")
    print("プロンプトを入力してください。'quit'で終了します。\n")
    
    while True:
        try:
            user_input = input("🎯 プロンプト: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q', '終了']:
                print("👋 ありがとうございました！")
                break
            
            if not user_input:
                continue
            
            print("🤖 分析中...")
            result = explainer.generate_explanation(user_input)
            
            if result["success"]:
                print(f"\n💡 AI解説:\n{result['explanation']}")
                print(f"\n📊 生成統計: {result['output_tokens']}トークン, {result['generation_time']:.2f}秒")
            else:
                print(f"\n❌ エラー: {result['error']}")
            
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n👋 ありがとうございました！")
            break
        except Exception as e:
            print(f"❌ エラー: {e}")

if __name__ == "__main__":
    sys.exit(main()) 