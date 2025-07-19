# XAI麻雀分析システム v2.1.0

**XAIを用いた麻雀における打牌根拠の可視化と自然言語説明システム**

## 📋 概要

本システムは、麻雀AIの打牌判断を**Explainable AI (XAI)** の手法を用いて解析し、人間が理解しやすい自然言語での説明を生成するツールです。Transformerベースの麻雀AIと大規模言語モデル（LLM）を組み合わせ、以下の機能を提供します：

### ✨ 主な機能

- **🧠 高精度打牌予測**: Transformerベースのモデルによる麻雀AI予測
- **🔍 アテンション可視化**: どの局面要素に注目して判断したかを可視化
- **📊 SHAP特徴量重要度**: 各特徴量が予測に与える影響度を定量化
- **🏷️ 概念ラベリング**: PCAとクラスタリングによる戦略概念の自動抽出
- **💬 自然言語説明**: LLMによる人間が理解しやすい戦術解説生成
- **📈 統計分析・可視化**: 予測精度や傾向の詳細分析
- **🚀 バッチ処理**: 1局分全てのツモ局面を一括分析

## 🏗️ システム構成

```
ver_2.1.0/
├── batch_analysis.py          # 📦 1局分全打牌バッチ分析システム
├── interactive_analysis.py    # 🖥️ インタラクティブ分析・可視化ツール
├── llm_integration.py         # 🤖 LLM連携・自動解説生成システム
├── predict_enhanced.py        # 🔮 強化版予測・XAI分析ツール
├── predict.py                 # 🎯 基本予測・分析ツール（Legacy）
├── prompt.py                  # 📝 プロンプト生成ユーティリティ
└── README.md                  # 📚 このファイル
```

## 🚀 クイックスタート

### 1. 基本的な使用方法

```bash
# インタラクティブモードで分析実行
python interactive_analysis.py sample_game.xml

# 特定の局・プレイヤーを指定してバッチ分析
python batch_analysis.py sample_game.xml 1 --player_id 0

# LLMによる自動解説生成
python llm_integration.py analysis_results/sample_game_R1_P0 --api_key YOUR_OPENAI_API_KEY
```

### 2. 研究用の完全な分析パイプライン

```bash
# ステップ1: 1局分の全打牌を分析
python batch_analysis.py your_game.xml 2 --player_id 1 --output_dir research_results

# ステップ2: 結果の可視化と統計分析
python interactive_analysis.py your_game.xml --auto --round 2 --player 1

# ステップ3: LLMによる解説生成
python llm_integration.py research_results/your_game_R2_P1

# ステップ4: 統合レポートの確認
open research_results/your_game_R2_P1/llm_explanations/integrated_report.html
```

## 📊 研究計画との対応

本システムは以下の研究計画に基づいて設計されています：

### 🎯 研究目的
- **目的1**: AIの打牌根拠を自然言語で説明 → `llm_integration.py`
- **目的2**: プレイヤーのスキル向上支援 → 統合レポートとプロンプト生成
- **目的3**: 多様な説明スタイルの提供 → 概念ラベリングと3つの観点別解説

### 🔬 提案手法の実装

#### 5.1 全体フロー
```
局面入力 → Transformer (predict_enhanced.py) → 重要度計算 → LLM (llm_integration.py) → 3種説明出力
```

#### 5.2 Attention による局所重要度
- `MahjongTransformerV2WithAttention` クラス
- `analyze_attention_weights()` 関数
- 層別アテンション可視化

#### 5.3 Activation による戦術概念スコア
- PCA次元削減 → k-meansクラスタリング
- Safety/Speed/HighDora等の概念自動抽出
- `analyze_with_concept_labels()` 関数

#### 5.4 LLM説明生成
- 構造化プロンプト生成
- Few-shot + Chain-of-Thought
- 3つの観点（即効性・戦術・代替案）

#### 5.5 多様な説明スタイル
- **定量的**: SHAP値による数値説明
- **定性的**: 概念ラベルによる戦略説明  
- **比較的**: 代替選択肢との比較

## 🛠️ 詳細な使用方法

### batch_analysis.py - バッチ分析システム

1局分の全てのツモ局面を一括で分析し、各局面のプロンプトを生成します。

```bash
# 基本的な使用法
python batch_analysis.py <XML_FILE> <ROUND_INDEX> [OPTIONS]

# 例: sample.xmlの第2局、プレイヤー0の全打牌を分析
python batch_analysis.py sample.xml 2 --player_id 0 --output_dir results

# 全プレイヤーを対象とする場合
python batch_analysis.py sample.xml 2 --output_dir results
```

**主要オプション:**
- `--player_id`: 対象プレイヤー (0-3, 未指定時は全プレイヤー)
- `--model_path`: 学習済みモデルのパス
- `--output_dir`: 結果出力ディレクトリ

**出力ファイル構造:**
```
results/
└── sample_R2_P0/
    ├── tsumo_01/
    │   ├── prompt.txt          # LLM用プロンプト
    │   ├── analysis_data.json  # 詳細分析データ
    │   └── summary.json        # 局面サマリ
    ├── tsumo_02/
    │   └── ...
    └── overall_summary.json    # 全体統計
```

### interactive_analysis.py - インタラクティブ分析

プレイヤー選択から結果可視化まで、対話的に分析を実行できます。

```bash
# インタラクティブモード（推奨）
python interactive_analysis.py sample.xml

# 自動モード（スクリプト実行用）
python interactive_analysis.py sample.xml --auto --round 1 --player 0
```

**機能:**
- 📋 プレイヤー・局選択メニュー
- ⏱️ 推定処理時間の表示
- 📊 結果の自動可視化（正解率推移、信頼度分布等）
- 📄 LLM用プロンプトファイルの一括エクスポート

### llm_integration.py - LLM連携システム

生成されたプロンプトを自動的にLLMに送信し、解説を生成します。

```bash
# 基本的な使用法
python llm_integration.py <ANALYSIS_DIR> [OPTIONS]

# OpenAI GPT-3.5-turbo使用
export OPENAI_API_KEY="your-api-key"
python llm_integration.py results/sample_R2_P0

# 他のモデルを使用
python llm_integration.py results/sample_R2_P0 --model gpt-4 --delay 2.0

# テストモード（API呼び出しなし）
python llm_integration.py results/sample_R2_P0 --dry_run
```

**主要オプション:**
- `--api_key`: OpenAI API Key
- `--model`: 使用するモデル名 (gpt-3.5-turbo, gpt-4等)
- `--delay`: API呼び出し間隔（秒）
- `--dry_run`: テストモード

**出力ファイル:**
```
results/sample_R2_P0/llm_explanations/
├── integrated_report.html     # ブラウザで閲覧可能な統合レポート
├── integrated_report.txt      # テキスト形式統合レポート
├── tsumo_01_explanation.txt   # 各局面の解説
├── tsumo_01_explanation.json  # JSON形式解説データ
└── batch_summary.json         # 処理結果サマリ
```

### predict_enhanced.py - 単発予測ツール

個別の局面を詳細分析する場合に使用します。

```bash
# 基本的な予測・分析
python predict_enhanced.py sample.xml 2 5

# 全機能有効での分析
python predict_enhanced.py sample.xml 2 5 \
  --visualize_attention \
  --output_json detailed_analysis.json
```

## 📈 生成される分析レポート

### 1. 統合HTMLレポート

**内容:**
- 📊 全体統計（正解率、平均信頼度等）
- 🎯 各局面の予測結果とAI解説
- 🎨 見やすいHTML形式

### 2. 可視化グラフ

**生成されるグラフ:**
- `accuracy_analysis.png`: 正解率推移と信頼度分布
- `confidence_analysis.png`: 信頼度の詳細分析
- `tile_analysis.png`: 牌種別予測分析

### 3. LLM解説例

```
=== 即効性判断 ===
孤立した役牌の西は手作りに最も不要です。これを先に切ることで、手牌を最も広く受けられる形に保てます。

=== 戦術的根拠 ===
この手牌はマンズ・ピンズで複数の面子候補があり、まだ発展途上です。西は自風でも場風でもない役牌（オタ風）で、重ねて刻子にする価値が低く、完全に孤立しています。序盤なので、こうした不要な字牌から整理し、有効牌を引く確率を最大化するのがセオリーです。

=== 代替案検討 ===
実戦で打たれた9p切りは、ドラ7pを引いた際の7p8p9pという面子の可能性を消してしまいます。4pや9mも、それぞれ優秀な搭子（ターツ）や雀頭候補を崩すことになり、受け入れ枚数を大きく損します。
```

## ⚙️ 環境設定

### 必要なパッケージ

```bash
# Python 3.8以上
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn
pip install shap h5py joblib
pip install requests  # LLM連携用
```

### モデルファイル

以下のファイルが必要です：
- `../ver_1.1.11/trained_model/mahjong_transformer_v1111_large_compiled_2.pth`
- `../ver_2.0.0/pca_model_2.joblib`
- `../ver_2.0.0/kmeans_model_2.joblib` 
- `../ver_2.0.0/concept_labels_2.joblib`

### 環境変数

```bash
# LLM使用時
export OPENAI_API_KEY="your-openai-api-key"

# 他のLLMサービス使用時
export LLM_BASE_URL="https://api.your-llm-service.com/v1"
```

## 🔬 研究での活用方法

### 実験設計例

1. **被験者実験準備**
```bash
# 複数局面の分析
for round in {1..4}; do
  python batch_analysis.py experiment_game.xml $round --player_id 0
done

# 解説生成
for dir in experiment_game_R*_P0; do
  python llm_integration.py $dir
done
```

2. **パフォーマンス評価**
```bash
# 予測精度分析
python interactive_analysis.py experiment_game.xml --auto --round 1 --player 0

# 統計的分析（overall_summary.jsonを使用）
python -c "
import json
with open('results/experiment_game_R1_P0/overall_summary.json', 'r') as f:
    data = json.load(f)
    print(f'Accuracy: {data[\"overview\"][\"accuracy\"]:.3f}')
    print(f'Confidence: {data[\"overview\"][\"average_confidence\"]:.3f}')
"
```

### 評価指標の取得

システムが自動的に生成する評価指標：

- **予測精度**: 正解率、信頼度
- **説明品質**: SHAP重要度、アテンション分析
- **計算効率**: 処理時間、トークン使用量

## 🤝 貢献・カスタマイズ

### 新しい説明手法の追加

1. `predict_enhanced.py`に分析関数を追加
2. `batch_analysis.py`の`_create_comprehensive_prompt()`を修正
3. プロンプトテンプレートをカスタマイズ

### 他のLLMとの連携

`llm_integration.py`の`LLMIntegration`クラスを継承して、新しいAPI に対応:

```python
class CustomLLMIntegration(LLMIntegration):
    def call_llm(self, prompt, **kwargs):
        # カスタムAPI呼び出しを実装
        pass
```

## ⚠️ 注意事項・制限事項

### API使用コスト

- GPT-3.5-turbo: 約$0.002/1000トークン
- 1局面あたり約1000-2000トークン
- 20局面 → 約$0.04-0.08

### 処理時間

- 1局面の分析: 約5-10秒
- 1局分（20-30局面）: 約5-10分
- LLM解説生成: 約1-2分/局面

### 計算リソース

- GPU推奨（CUDA対応）
- メモリ: 8GB以上
- ストレージ: 分析結果用に1-2GB

## 📚 関連論文・参考文献

- Li, X., et al. (2024). "Tjong: A transformer-based Mahjong AI via hierarchical decision-making"
- Kim, J., et al. (2024). "Bridging the Gap between Expert and Language Models: Concept-guided Chess Commentary Generation"

## 📞 サポート・問い合わせ

問題や質問がある場合は、以下の情報とともに報告してください：

- 使用したコマンド
- エラーメッセージ
- Python・パッケージのバージョン
- システム環境（OS、GPU等）

---

**麻雀XAI研究プロジェクト v2.1.0**  
*XAIを用いた麻雀における打牌根拠の可視化と自然言語説明* 