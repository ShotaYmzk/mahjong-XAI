# llm_integration.py - LLM連携・自動解説生成システム
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import requests
from typing import List, Dict, Optional

class LLMIntegration:
    """LLMとの連携クラス"""
    
    def __init__(self, api_key: str = None, model_name: str = "gpt-3.5-turbo", base_url: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.base_url = base_url or "https://api.openai.com/v1"
        self.request_count = 0
        self.total_tokens = 0
        
        if not self.api_key:
            print("[警告] APIキーが設定されていません。環境変数 OPENAI_API_KEY を設定してください")
    
    def call_llm(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> Optional[str]:
        """LLMを呼び出して応答を取得"""
        if not self.api_key:
            return None
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "あなたは麻雀の専門コーチです。分析結果に基づいて、わかりやすく実戦的な解説を提供してください。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                self.request_count += 1
                if 'usage' in result:
                    self.total_tokens += result['usage']['total_tokens']
                
                return result['choices'][0]['message']['content']
            else:
                print(f"[エラー] LLM API呼び出しエラー: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"[エラー] LLM呼び出し中にエラー: {e}")
            return None
    
    def get_usage_stats(self) -> Dict:
        """使用統計を取得"""
        return {
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.total_tokens * 0.002 / 1000  # GPT-3.5-turbo概算コスト
        }

class BatchExplanationSystem:
    """バッチ解説生成システム"""
    
    def __init__(self, llm_integration: LLMIntegration, delay_seconds: float = 1.0):
        self.llm = llm_integration
        self.delay_seconds = delay_seconds
    
    def process_analysis_directory(self, analysis_dir: Path, output_dir: Path = None) -> Dict:
        """分析ディレクトリを処理して解説を生成"""
        analysis_path = Path(analysis_dir)
        
        if output_dir is None:
            output_dir = analysis_path / "llm_explanations"
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(exist_ok=True)
        
        # ツモ局面ディレクトリを取得
        moment_dirs = sorted([d for d in analysis_path.iterdir() 
                            if d.is_dir() and d.name.startswith("tsumo_")])
        
        if not moment_dirs:
            print(f"[エラー] 分析結果が見つかりません: {analysis_path}")
            return {"success": False, "processed": 0}
        
        print(f"処理対象: {len(moment_dirs)}個の局面")
        print(f"出力先: {output_dir}")
        
        processed_count = 0
        failed_count = 0
        explanations = []
        
        for i, moment_dir in enumerate(moment_dirs):
            print(f"処理中 [{i+1}/{len(moment_dirs)}]: {moment_dir.name}")
            
            try:
                result = self._process_single_moment(moment_dir, output_dir)
                if result:
                    explanations.append(result)
                    processed_count += 1
                else:
                    failed_count += 1
                    
                # レート制限対策
                if i < len(moment_dirs) - 1:  # 最後でなければ待機
                    time.sleep(self.delay_seconds)
                    
            except Exception as e:
                print(f"  [エラー] {moment_dir.name}の処理中にエラー: {e}")
                failed_count += 1
        
        # 統合レポートの生成
        self._create_integrated_report(explanations, output_dir, analysis_path)
        
        # 統計情報
        stats = self.llm.get_usage_stats()
        
        result = {
            "success": True,
            "processed": processed_count,
            "failed": failed_count,
            "total_moments": len(moment_dirs),
            "output_directory": str(output_dir),
            "llm_stats": stats
        }
        
        # 結果サマリの保存
        with open(output_dir / "batch_summary.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return result
    
    def _process_single_moment(self, moment_dir: Path, output_dir: Path) -> Optional[Dict]:
        """単一局面の解説生成"""
        prompt_file = moment_dir / "prompt.txt"
        summary_file = moment_dir / "summary.json"
        
        if not prompt_file.exists() or not summary_file.exists():
            print(f"  [警告] 必要なファイルが見つかりません: {moment_dir}")
            return None
        
        # プロンプトとサマリの読み込み
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read()
            
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        # LLM呼び出し
        explanation = self.llm.call_llm(prompt, max_tokens=1500, temperature=0.7)
        
        if explanation:
            # 解説の保存
            explanation_file = output_dir / f"{moment_dir.name}_explanation.txt"
            with open(explanation_file, 'w', encoding='utf-8') as f:
                f.write(f"=== {moment_dir.name} 解説 ===\n")
                f.write(f"生成日時: {datetime.now().isoformat()}\n")
                f.write(f"予測: {summary['predicted_tile']} (信頼度: {summary['predicted_probability']:.1%})\n")
                f.write(f"実際: {summary['actual_tile']}\n")
                f.write(f"正解: {'○' if summary['match'] else '×'}\n")
                f.write("="*50 + "\n\n")
                f.write(explanation)
            
            # JSON形式でも保存
            explanation_data = {
                "moment_info": summary,
                "explanation": explanation,
                "generated_at": datetime.now().isoformat(),
                "model_used": self.llm.model_name
            }
            
            explanation_json = output_dir / f"{moment_dir.name}_explanation.json"
            with open(explanation_json, 'w', encoding='utf-8') as f:
                json.dump(explanation_data, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ 解説生成完了: {explanation_file.name}")
            return explanation_data
        else:
            print(f"  ✗ 解説生成失敗")
            return None
    
    def _create_integrated_report(self, explanations: List[Dict], output_dir: Path, analysis_path: Path):
        """統合レポートの作成"""
        if not explanations:
            return
        
        # HTMLレポートの生成
        html_content = self._generate_html_report(explanations, analysis_path)
        html_file = output_dir / "integrated_report.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # テキストレポートの生成
        text_content = self._generate_text_report(explanations, analysis_path)
        text_file = output_dir / "integrated_report.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        print(f"統合レポートを生成しました:")
        print(f"  HTML: {html_file}")
        print(f"  テキスト: {text_file}")
    
    def _generate_html_report(self, explanations: List[Dict], analysis_path: Path) -> str:
        """HTMLレポートの生成"""
        html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>麻雀AI分析解説レポート</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background-color: #f4f4f4; padding: 20px; margin-bottom: 30px; }}
        .moment {{ border: 1px solid #ddd; margin: 20px 0; padding: 20px; }}
        .moment-header {{ background-color: #e8f4fd; padding: 10px; margin-bottom: 15px; }}
        .correct {{ color: green; font-weight: bold; }}
        .incorrect {{ color: red; font-weight: bold; }}
        .explanation {{ background-color: #f9f9f9; padding: 15px; border-left: 4px solid #007cba; }}
        .stats {{ background-color: #fff3cd; padding: 15px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>麻雀AI分析解説レポート</h1>
        <p>生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
        <p>ソース: {analysis_path.name}</p>
        <p>総局面数: {len(explanations)}局面</p>
    </div>
"""
        
        correct_count = sum(1 for exp in explanations if exp["moment_info"]["match"])
        accuracy = correct_count / len(explanations) if explanations else 0
        
        html += f"""
    <div class="stats">
        <h2>分析統計</h2>
        <ul>
            <li>正解数: {correct_count} / {len(explanations)}</li>
            <li>正解率: {accuracy:.1%}</li>
            <li>平均信頼度: {sum(exp["moment_info"]["predicted_probability"] for exp in explanations) / len(explanations):.1%}</li>
        </ul>
    </div>
"""
        
        for i, exp in enumerate(explanations):
            moment_info = exp["moment_info"]
            explanation = exp["explanation"]
            
            correct_class = "correct" if moment_info["match"] else "incorrect"
            correct_symbol = "○" if moment_info["match"] else "×"
            
            html += f"""
    <div class="moment">
        <div class="moment-header">
            <h3>局面 {i+1}: ツモ{moment_info["tsumo_count"]}</h3>
            <p>
                予測: <strong>{moment_info["predicted_tile"]}</strong> 
                (信頼度: {moment_info["predicted_probability"]:.1%}) | 
                実際: <strong>{moment_info["actual_tile"]}</strong> | 
                <span class="{correct_class}">正解: {correct_symbol}</span>
            </p>
        </div>
        <div class="explanation">
            <h4>AI解説:</h4>
            <div style="white-space: pre-line;">{explanation}</div>
        </div>
    </div>
"""
        
        html += """
</body>
</html>
"""
        return html
    
    def _generate_text_report(self, explanations: List[Dict], analysis_path: Path) -> str:
        """テキストレポートの生成"""
        report = f"""=== 麻雀AI分析解説レポート ===
生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
ソース: {analysis_path.name}
総局面数: {len(explanations)}局面

"""
        
        correct_count = sum(1 for exp in explanations if exp["moment_info"]["match"])
        accuracy = correct_count / len(explanations) if explanations else 0
        avg_confidence = sum(exp["moment_info"]["predicted_probability"] for exp in explanations) / len(explanations)
        
        report += f"""=== 分析統計 ===
正解数: {correct_count} / {len(explanations)}
正解率: {accuracy:.1%}
平均信頼度: {avg_confidence:.1%}

{"="*50}

"""
        
        for i, exp in enumerate(explanations):
            moment_info = exp["moment_info"]
            explanation = exp["explanation"]
            
            correct_symbol = "○" if moment_info["match"] else "×"
            
            report += f"""=== 局面 {i+1}: ツモ{moment_info["tsumo_count"]} ===
予測: {moment_info["predicted_tile"]} (信頼度: {moment_info["predicted_probability"]:.1%})
実際: {moment_info["actual_tile"]}
正解: {correct_symbol}

【AI解説】
{explanation}

{"="*50}

"""
        
        return report

def main():
    parser = argparse.ArgumentParser(description="LLM連携・自動解説生成システム")
    parser.add_argument("analysis_dir", help="分析結果ディレクトリのパス")
    parser.add_argument("--output_dir", help="解説出力ディレクトリ（未指定時は analysis_dir/llm_explanations）")
    parser.add_argument("--api_key", help="OpenAI API Key（環境変数 OPENAI_API_KEY でも設定可能）")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="使用するLLMモデル名")
    parser.add_argument("--delay", type=float, default=1.0, help="API呼び出し間の待機時間（秒）")
    parser.add_argument("--base_url", help="API Base URL（デフォルト: OpenAI）")
    parser.add_argument("--dry_run", action='store_true', help="実際にAPIを呼び出さずにテストする")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.analysis_dir):
        print(f"[エラー] 分析ディレクトリが見つかりません: {args.analysis_dir}")
        exit(1)
    
    # LLM統合クラスの初期化
    if args.dry_run:
        print("[テストモード] 実際のAPI呼び出しは行いません")
        llm = None
    else:
        llm = LLMIntegration(
            api_key=args.api_key,
            model_name=args.model,
            base_url=args.base_url
        )
        
        if not llm.api_key:
            print("[エラー] API キーが設定されていません")
            print("--api_key オプションで指定するか、環境変数 OPENAI_API_KEY を設定してください")
            exit(1)
    
    # バッチ処理システムの初期化
    if args.dry_run:
        # ドライランモード用のモックLLM
        class MockLLM:
            def __init__(self):
                self.request_count = 0
                self.total_tokens = 0
                self.model_name = "mock"
                
            def call_llm(self, prompt, **kwargs):
                self.request_count += 1
                self.total_tokens += len(prompt) // 4  # 簡易トークン数
                return "[テストモード] ここに解説が生成されます..."
                
            def get_usage_stats(self):
                return {
                    "request_count": self.request_count,
                    "total_tokens": self.total_tokens,
                    "estimated_cost_usd": 0.0
                }
        
        llm = MockLLM()
    
    batch_system = BatchExplanationSystem(llm, delay_seconds=args.delay)
    
    try:
        print(f"解説生成を開始します...")
        print(f"対象ディレクトリ: {args.analysis_dir}")
        
        result = batch_system.process_analysis_directory(
            analysis_dir=args.analysis_dir,
            output_dir=args.output_dir
        )
        
        if result["success"]:
            print(f"\n=== 解説生成完了 ===")
            print(f"処理済み: {result['processed']}/{result['total_moments']} 局面")
            print(f"失敗: {result['failed']} 局面")
            print(f"出力先: {result['output_directory']}")
            
            if not args.dry_run:
                stats = result["llm_stats"]
                print(f"\nLLM使用統計:")
                print(f"  リクエスト数: {stats['request_count']}")
                print(f"  総トークン数: {stats['total_tokens']:,}")
                print(f"  推定コスト: ${stats['estimated_cost_usd']:.4f}")
            
            print("\n生成されたファイル:")
            print("  - integrated_report.html: ブラウザで閲覧可能な統合レポート")
            print("  - integrated_report.txt: テキスト形式の統合レポート")
            print("  - tsumo_XX_explanation.txt: 各局面の解説")
            print("  - batch_summary.json: 処理結果サマリ")
        else:
            print("解説生成に失敗しました")
            
    except KeyboardInterrupt:
        print("\n\n処理を中断しました")
    except Exception as e:
        print(f"[エラー] 予期せぬエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 