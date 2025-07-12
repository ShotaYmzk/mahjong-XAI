# 修正版データセット作成・分析システム

このドキュメントでは、手牌にない牌が推奨打牌として表示される問題を修正したシステムの使用方法を説明します。

## 問題の概要

既存のシステムでは以下の問題がありました：

1. **データセット作成時の問題**: 学習データで手牌にない牌がターゲットとして設定されている
2. **推奨打牌表示の問題**: Top5推奨打牌が手牌の有効性チェックを受けていない

## 修正内容

### 1. `create_training_dataset.py` - 正しいデータセット作成

**主な機能:**
- 手牌にない牌をターゲットとして除外
- リーチ状態での打牌制限チェック
- 有効な打牌選択肢のみを学習データに含める
- 詳細な統計情報とエラーログ

**使用方法:**
```bash
# 基本的な使用
python create_training_dataset.py /path/to/xml_files --output training_data/fixed_dataset.hdf5

# オプション付き
python create_training_dataset.py /path/to/xml_files \
    --output training_data/fixed_dataset.hdf5 \
    --max-files 100 \
    --workers 8 \
    --debug
```

**検証機能:**
- `validate_sample()`: 各サンプルの有効性を厳密チェック
- 手牌数確認（14枚）
- 実際の打牌が手牌に存在するかチェック
- リーチ後のツモ切り制限チェック

### 2. `fixed_batch_analysis.py` - 修正版分析システム

**主な修正:**
- Top5推奨打牌を有効な選択肢（手牌にある牌）のみに制限
- `format_valid_top_predictions()` 関数で適切なフィルタリング
- 無効な推奨を統計から除外

**使用方法:**
```bash
# 特定プレイヤーの分析
python fixed_batch_analysis.py juni.xml 1 --player 3 --output fixed_results

# 全プレイヤーの分析
python fixed_batch_analysis.py juni.xml 1 --output fixed_results
```

## データセット品質向上

### 検証項目

1. **基本バリデーション**
   - プレイヤーID範囲チェック (0-3)
   - 牌ID範囲チェック (0-135)

2. **手牌整合性**
   - 手牌が空でないかチェック
   - ツモ後14枚であることを確認
   - 打牌する牌が実際に手牌にあるかチェック

3. **ゲームルール準拠**
   - リーチ後のツモ切り制限
   - 有効打牌選択肢との一致確認

### 統計出力例

```
=== データセット統計 ===
総ゲーム数: 150
総局数: 1800
有効サンプル数: 45230
無効サンプル数: 5420
無効サンプルの理由:
  tile_not_in_hand: 3210 (59.2%)
  invalid_hand_size_13: 1500 (27.7%)
  reach_violation: 410 (7.6%)
  not_in_valid_options: 300 (5.5%)
```

## 推奨打牌表示の修正

### 修正前の問題

```
【推奨打牌Top5】
  1位: 中 (17.1%)  ← 手牌にない！
  2位: 西 (9.2%)   ← 手牌にない！
  3位: 4s (6.8%)
  4位: 東 (6.1%)   ← 手牌にない！
  5位: 1p (4.5%)   ← 手牌にない！
```

### 修正後

```
【推奨打牌Top5】
  1位: 4s (6.8%)
  2位: 8s (4.2%)
  3位: 3m (3.1%)
  4位: 7s (2.9%)
  5位: 1s (2.3%)
```

## ファイル説明

### `create_training_dataset.py`

- **MahjongDatasetCreator**: メインのデータセット作成クラス
- **validate_sample()**: サンプル有効性検証
- **extract_samples_from_round()**: 1局からの有効サンプル抽出
- マルチプロセシング対応
- 詳細な統計情報出力

### `fixed_batch_analysis.py`

- **FixedBatchAnalysisSystem**: 修正版分析システム
- **format_valid_top_predictions()**: 有効な推奨打牌のみ表示
- **analyze_single_moment()**: 局面分析（有効性チェック付き）
- プロンプト生成とJSON出力

## 使用例

### 1. 新しいデータセット作成

```bash
# XMLファイルから正しいデータセットを作成
python create_training_dataset.py ../xml_logs \
    --output training_data/mahjong_valid_v3.hdf5 \
    --max-files 50 \
    --workers 4 \
    --debug
```

### 2. 修正版で分析実行

```bash
# 修正版システムで分析
python fixed_batch_analysis.py juni.xml 1 \
    --player 3 \
    --output fixed_analysis_results \
    --model ../trained_models/fixed_model.pth
```

### 3. 結果確認

```bash
# 出力結果の確認
ls fixed_analysis_results/juni_R1_P3/
# tsumo_1/ tsumo_2/ ... summary.json

# 個別局面の確認
cat fixed_analysis_results/juni_R1_P3/tsumo_56/prompt.txt
```

## 期待される改善効果

1. **学習データ品質向上**
   - 無効なターゲットの除去
   - より現実的な学習データ

2. **予測精度向上**
   - 手牌制約を満たす予測のみ
   - ルール準拠の推奨

3. **分析結果の信頼性向上**
   - 実際に打てる牌のみの推奨
   - より実用的な戦術分析

## 注意事項

- 既存のモデルは新しいデータセットで再訓練が必要
- 古い分析結果との比較時は手牌制約の違いに注意
- デバッグモードでは詳細ログが出力されるためディスク容量に注意

## トラブルシューティング

### よくあるエラー

1. **ImportError**: プロジェクトモジュールが見つからない
   - パスが正しく設定されているか確認
   - 必要なファイルが同じディレクトリにあるか確認

2. **メモリエラー**: 大量のXMLファイル処理時
   - `--max-files` で処理ファイル数を制限
   - `--workers` でプロセス数を調整

3. **データセットサイズが小さい**: 有効サンプルが少ない
   - `--debug` で無効理由を確認
   - XMLファイルの品質を確認 