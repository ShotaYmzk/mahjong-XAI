# 麻雀AI推論スクリプト (Mahjong AI Inference Script)

## 概要

`suiron.py` は訓練済みの麻雀Transformerモデルを使用して、XMLログファイルから麻雀の次の牌を予測する推論スクリプトです。

## 使用方法

### 基本的な使用方法

```bash
# 最終モデルを使用してtest_log.xmlを解析
python suiron.py --model_path trained_model/mahjong_transformer_v_strong_flat_e500.pth --xml_path test_log.xml

# チェックポイントモデルを使用
python suiron.py --model_path checkpoints_v_strong_flat_e500/checkpoint_epoch_214.pth --xml_path test_log.xml

# 特定の局のみを解析
python suiron.py --model_path trained_model/mahjong_transformer_v_strong_flat_e500.pth --xml_path test_log.xml --round 1

# 特定のイベントまで解析
python suiron.py --model_path trained_model/mahjong_transformer_v_strong_flat_e500.pth --xml_path test_log.xml --round 1 --event 10

# 結果をJSONファイルに保存
python suiron.py --model_path trained_model/mahjong_transformer_v_strong_flat_e500.pth --xml_path test_log.xml --output results.json
```

### コマンドライン引数

- `--model_path`: 訓練済みモデルのパス（必須）
- `--xml_path`: 解析するXMLログファイルのパス（必須）
- `--round`: 解析対象の局番号（1から開始、指定なしで全局）
- `--event`: 解析対象のイベント番号（指定なしで局の最後まで）
- `--device`: 使用デバイス (cuda/cpu、指定なしで自動選択)
- `--output`: 結果出力ファイル（JSON形式）

## 出力例

```
================================================================================
麻雀AI推論結果
================================================================================
対局者: 三河屋, 三田村茜, チャオリ, LITHIUM
解析局数: 1/8
予測実行回数: 5

--- 第1局 ---
処理イベント数: 10
予測実行回数: 5

  予測 1: イベント 0 (T123)
    予測牌: 西 (確率: 0.578)
    Top-5予測:
      1. 西 (0.578)
      2. 南 (0.229)
      3. 東 (0.038)
    コンテキスト: プレイヤー0, 巡目0.0
```

## モデル情報

- **モデル名**: MahjongTransformer
- **アーキテクチャ**: Transformer-based
- **パラメータ数**: 3,438,243個
- **最高検証精度**: 約69.86%
- **出力**: 34種類の牌の予測（NUM_TILE_TYPES=34）

## 利用可能なモデル

1. **最終モデル**: `trained_model/mahjong_transformer_v_strong_flat_e500.pth`
   - 最高性能のモデル（推奨）

2. **チェックポイント**: `checkpoints_v_strong_flat_e500/checkpoint_epoch_*.pth`
   - 各エポックのチェックポイント（214エポックまで利用可能）

## 技術詳細

- **特徴量**: イベントシーケンス + 静的特徴量
- **イベントシーケンス**: 最大60イベントの履歴
- **静的特徴量**: 157次元のゲーム状態情報
- **予測対象**: 次の牌の種類（34種類）
- **推論タイミング**: ツモイベント（T, U, V, W）の後

## 注意事項

- XMLログファイルは天鳳形式である必要があります
- モデルはCUDA対応（CPUでも動作）
- 大量の予測を実行する場合は時間がかかる場合があります
