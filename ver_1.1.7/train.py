# train.py (Transformer版)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset, random_split
import numpy as np
import os
import glob
import math
from tqdm import tqdm
from sklearn.model_selection import train_test_split # オプション: 全データをメモリに乗せられる場合

# --- 設定 ---
# データセットパス (ワイルドカードで複数バッチを読み込む)
MAX_EVENT_HISTORY = 500  # 適切なシーケンス長を設定
DATA_DIR = "./training_data/"
DATA_PATTERN = os.path.join(DATA_DIR, "mahjong_transformer_data_batch_*.npz")
MODEL_SAVE_PATH = "./trained_model/mahjong_transformer_model.pth"
BATCH_SIZE = 128    # GPUメモリに合わせて調整 (Transformerはメモリ消費が大きい傾向)
NUM_EPOCHS = 30      # エポック数 (データ量や収束具合で調整)
LEARNING_RATE = 1e-4 # Transformer向けの学習率 (AdamWが推奨されることも)
VALIDATION_SPLIT = 0.1 # 検証データの割合
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
# DEVICE = torch.device("cpu") # CPUで強制実行する場合

print(f"Using device: {DEVICE}")

# --- Transformer モデル定義 ---
NUM_TILE_TYPES = 34 # 打牌の選択肢

class PositionalEncoding(nn.Module):
    """Transformer用の位置エンコーディング (batch_first=True 対応)"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500): # max_lenはMAX_EVENT_HISTORYと一致させる
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # peの形状を [max_len, d_model] に変更し、後でunsqueezeする
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # peを [1, max_len, d_model] の形状で登録 (バッチ次元を追加)
        self.register_buffer('pe', pe.unsqueeze(0)) # バッチ次元を追加

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] (batch_first=Trueのため)
        """
        # xのシーケンス長に合わせてpeをスライスし、加算する
        # self.pe の形状は [1, max_len, d_model]
        # x の形状は [batch_size, seq_len, d_model]
        # スライス: self.pe[:, :x.size(1)] -> [1, seq_len, d_model]
        # これをxに加算すると、バッチ次元(0)がブロードキャストされる
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MahjongTransformerModel(nn.Module):
    def __init__(self,
                 event_feature_dim: int, # イベントシーケンスの各要素の次元
                 static_feature_dim: int, # 静的特徴ベクトルの次元
                 d_model: int = 128,      # Transformer内部の次元 (要調整)
                 nhead: int = 4,          # Multi-head Attentionのヘッド数 (d_modelを割り切れる数)
                 d_hid: int = 128,       # Transformer EncoderのFFN中間層次元 (d_model*4程度)
                 nlayers: int = 2,        # Transformer Encoderの層数 (要調整)
                 dropout: float = 0.1,    # ドロップアウト率
                 output_dim: int = NUM_TILE_TYPES,
                 max_seq_len: int = MAX_EVENT_HISTORY): # game_stateと合わせる
        super().__init__()
        self.d_model = d_model

        # 1. イベントシーケンス用Embedding + Positional Encoding
        self.event_encoder = nn.Linear(event_feature_dim, d_model) # 線形層でEmbedding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)

        # 2. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True) # batch_first=Trueに注意
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        # 3. 静的特徴量用エンコーダー (MLP)
        self.static_encoder = nn.Sequential(
            nn.Linear(static_feature_dim, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 2) # 次元数をd_model//2に
        )

        # 4. 結合層と最終出力層 (MLP)
        # Transformer出力(d_model)と静的特徴出力(d_model//2)を結合
        self.decoder = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.event_encoder.weight.data.uniform_(-initrange, initrange)
        self.event_encoder.bias.data.zero_()
        # static_encoderとdecoder内のLinear層も初期化 (省略)

    def forward(self, event_seq: torch.Tensor, static_feat: torch.Tensor, src_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            event_seq: イベントシーケンス, shape [batch_size, seq_len, event_feature_dim]
            static_feat: 静的特徴量, shape [batch_size, static_feature_dim]
            src_padding_mask: イベントシーケンスのパディングマスク, shape [batch_size, seq_len]
                              (値がTrueの位置は無視される)
        """
        # 1. イベントシーケンス処理
        # [batch_size, seq_len, event_feature_dim] -> [batch_size, seq_len, d_model]
        embedded_seq = self.event_encoder(event_seq) * math.sqrt(self.d_model)
        # Positional Encodingのために [seq_len, batch_size, d_model] に変換してから戻す場合もあるが、
        # nn.TransformerEncoderLayer (batch_first=True) と PositionalEncoding の実装に依存
        # ここでは PositionalEncoding が [seq_len, batch, dim] を期待すると仮定し、batch_firstに合わせる
        # -> PositionalEncoding を修正するか、ここでtranspose/untransposeする
        # ---- PositionalEncoding修正 or Transpose ----
        # 例: transposeする場合
        # embedded_seq = embedded_seq.permute(1, 0, 2) # [seq_len, batch_size, d_model]
        # pos_encoded_seq = self.pos_encoder(embedded_seq)
        # transformer_output = self.transformer_encoder(pos_encoded_seq, src_key_padding_mask=src_padding_mask) # マスクを使用
        # transformer_output = transformer_output.permute(1, 0, 2) # [batch_size, seq_len, d_model]
        # ---- PositionalEncoding を batch_first=True に合わせる方が楽 ----
        # (上のPositionalEncodingクラスのforwardを修正する必要がある: shape [batch, seq, dim]を受け取るように)
        # 以下はPositionalEncodingが batch_first 対応済みと仮定
        pos_encoded_seq = self.pos_encoder(embedded_seq) # 要修正: PositionalEncodingの実装確認

        # Transformer Encoderに通す
        # src_key_padding_mask は [batch_size, seq_len]
        transformer_output = self.transformer_encoder(pos_encoded_seq, src_key_padding_mask=src_padding_mask)

        # Transformerの出力から代表ベクトルを取得 (例: 最初のトークン or 平均)
        # ここではシーケンス全体の平均プーリングを使用 (マスク考慮)
        if src_padding_mask is not None:
            # マスクされていない部分の長さを計算
            seq_len = (~src_padding_mask).sum(dim=1, keepdim=True).float() # [batch_size, 1]
            seq_len = torch.max(seq_len, torch.tensor(1.0, device=DEVICE)) # 0除算防止
            # マスク部分を0にして合計し、長さで割る
            masked_output = transformer_output.masked_fill(src_padding_mask.unsqueeze(-1), 0.0)
            transformer_pooled = masked_output.sum(dim=1) / seq_len
        else:
            transformer_pooled = transformer_output.mean(dim=1) # [batch_size, d_model]


        # 2. 静的特徴量処理
        # [batch_size, static_feature_dim] -> [batch_size, d_model // 2]
        encoded_static = self.static_encoder(static_feat)

        # 3. 結合して最終予測
        # [batch_size, d_model + d_model // 2]
        combined_features = torch.cat((transformer_pooled, encoded_static), dim=1)
        # [batch_size, output_dim]
        output = self.decoder(combined_features)

        return output

# --- PyTorch Dataset クラス (バッチファイル読み込み対応) ---
class MahjongNpzDataset(Dataset):
    def __init__(self, npz_files):
        self.npz_files = npz_files
        self.file_lengths = [len(np.load(f)['labels']) for f in npz_files]
        self.cumulative_lengths = np.cumsum(self.file_lengths)
        self.total_length = self.cumulative_lengths[-1]
        # データ次元数を最初のファイルから取得 (全ファイルで同じ前提)
        with np.load(npz_files[0]) as data:
            self.seq_len = data['sequences'].shape[1]
            self.event_dim = data['sequences'].shape[2]
            self.static_dim = data['static_features'].shape[1]
        print(f"Dataset initialized with {len(npz_files)} files, total {self.total_length} samples.")
        print(f"  Sequence Length: {self.seq_len}, Event Dim: {self.event_dim}, Static Dim: {self.static_dim}")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # idxがどのファイルに属するかを見つける
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        # ファイル内でのインデックスを計算
        local_idx = idx - (self.cumulative_lengths[file_idx-1] if file_idx > 0 else 0)

        # 対応するnpzファイルをロード (毎回ロードするのは非効率だが、メモリ節約になる)
        # TODO: キャッシュ機構を追加すると高速化できる
        with np.load(self.npz_files[file_idx]) as data:
            sequence = torch.tensor(data['sequences'][local_idx], dtype=torch.float32)
            static_features = torch.tensor(data['static_features'][local_idx], dtype=torch.float32)
            label = torch.tensor(data['labels'][local_idx], dtype=torch.long)

            # パディングマスク生成 (イベントタイプがPADDING_CODEでない箇所がTrue)
            # GameState.EVENT_TYPES["PADDING"] を参照できるようにする
            padding_code = 8 # GameStateからインポート or ここで定義
            # sequenceの最初の要素 (イベントタイプ) を見て判断
            # 重要: src_key_padding_mask は True の位置が無視される
            src_padding_mask = (sequence[:, 0] == padding_code) # [seq_len]

        return sequence, static_features, label, src_padding_mask

# --- メイン処理 ---
if __name__ == "__main__":
    # 1. データセットファイルのリストを取得
    npz_files = sorted(glob.glob(DATA_PATTERN))
    if not npz_files:
        print(f"Error: No data files found matching pattern: {DATA_PATTERN}")
        exit()
    print(f"Found {len(npz_files)} data files.")

    # 2. Dataset の作成 (全ファイルを扱う)
    full_dataset = MahjongNpzDataset(npz_files)
    event_feature_dim = full_dataset.event_dim
    static_feature_dim = full_dataset.static_dim
    max_seq_len = full_dataset.seq_len # データから取得したシーケンス長

    # 3. 訓練データと検証データに分割
    total_size = len(full_dataset)
    val_size = int(total_size * VALIDATION_SPLIT)
    train_size = total_size - val_size
    # random_splitを使ってインデックスを分割
    train_indices, val_indices = random_split(range(total_size), [train_size, val_size])

    # Subsetを使ってデータセットを分割 (データをメモリにコピーしない)
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    print(f"Data split: Training samples={len(train_dataset)}, Validation samples={len(val_dataset)}")

    # 4. DataLoader の作成
    # num_workersは環境に合わせて調整 (MPS/CUDAでは0が良い場合も)
    num_workers = 2 if DEVICE == torch.device("cpu") else 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True if DEVICE != torch.device("cpu") else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True if DEVICE != torch.device("cpu") else False)
    print("DataLoader created.")

    # 5. モデル、損失関数、最適化アルゴリズムの定義
    model = MahjongTransformerModel(
        event_feature_dim=event_feature_dim,
        static_feature_dim=static_feature_dim,
        max_seq_len=max_seq_len,
        # 他のハイパーパラメータはクラス定義のデフォルト値を使用 or ここで指定
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    # AdamWがTransformerで推奨されることが多い
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    # 学習率スケジューラ (例: コサインアニーリング)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(train_loader), eta_min=LEARNING_RATE / 10)

    print("Model, Criterion, Optimizer defined.")
    # print(model) # モデル構造の確認

    # 6. 訓練ループ
    print("Starting training...")
    best_val_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        # --- 訓練フェーズ ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # tqdmを使った進捗表示
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for sequences, static_features, labels, masks in train_pbar:
            sequences, static_features, labels, masks = sequences.to(DEVICE), static_features.to(DEVICE), labels.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(sequences, static_features, masks)
            loss = criterion(outputs, labels)
            loss.backward()
            # 勾配クリッピング (Transformerで有効な場合がある)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step() # ステップごとにスケジューラ更新

            running_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # tqdmに進捗情報を表示
            train_pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        epoch_loss_train = running_loss / total_train
        epoch_acc_train = 100 * correct_train / total_train
        print(f'\n--- Epoch {epoch+1} Training --- Loss: {epoch_loss_train:.4f}, Accuracy: {epoch_acc_train:.2f}%')

        # --- 検証フェーズ ---
        model.eval()
        running_loss_val = 0.0
        correct_val = 0
        total_val = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
        with torch.no_grad():
            for sequences, static_features, labels, masks in val_pbar:
                sequences, static_features, labels, masks = sequences.to(DEVICE), static_features.to(DEVICE), labels.to(DEVICE), masks.to(DEVICE)
                outputs = model(sequences, static_features, masks)
                loss = criterion(outputs, labels)
                running_loss_val += loss.item() * sequences.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                val_pbar.set_postfix(loss=loss.item())

        epoch_loss_val = running_loss_val / total_val
        epoch_acc_val = 100 * correct_val / total_val
        print(f'--- Epoch {epoch+1} Validation --- Loss: {epoch_loss_val:.4f}, Accuracy: {epoch_acc_val:.2f}%')

        # --- モデルの保存 ---
        if epoch_acc_val > best_val_accuracy:
            best_val_accuracy = epoch_acc_val
            save_dir = os.path.dirname(MODEL_SAVE_PATH)
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Best model saved to {MODEL_SAVE_PATH} (Validation Accuracy: {best_val_accuracy:.2f}%)")

    print("Training finished.")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    
