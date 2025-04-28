import os
import sys
import glob
import time
import random
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# --- 環境・デバイス設定 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device} {torch.cuda.get_device_name(device) if device.type=='cuda' else ''}")

# --- 再現性のためのシード設定 ---
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)

# --- モジュールと定数 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
from train import MahjongNpzDataset
from game_state import NUM_TILE_TYPES, MAX_EVENT_HISTORY, EVENT_TYPES, STATIC_FEATURE_DIM

# --- RNNモデル定義 ---
class MahjongRNNModel(nn.Module):
    """LSTMを用いた時系列処理モデル"""
    def __init__(self,
                 event_dim: int,
                 static_dim: int,
                 hidden_size: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 output_dim: int = NUM_TILE_TYPES):
        super().__init__()
        self.event_encoder = nn.Linear(event_dim, hidden_size)
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self,
                event_seq: torch.Tensor,
                static_feat: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = event_seq.size()
        x = self.event_encoder(event_seq)  # (B, S, hidden)

        if attention_mask is not None:
            lengths = (attention_mask == False).sum(dim=1).cpu()
        else:
            lengths = torch.full((batch_size,), seq_len, dtype=torch.long)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, _) = self.rnn(packed)
        final_hidden = h_n[-1]  # (B, hidden)
        static_emb = self.static_encoder(static_feat)  # (B, hidden)
        combined = torch.cat([final_hidden, static_emb], dim=1)  # (B, hidden*2)
        logits = self.classifier(combined)
        return logits

# --- ハイパーパラメータ ---
DATA_PATTERN       = os.path.join("./training_data/", "mahjong_imitation_data_v119_batch_*.npz")
VALIDATION_SPLIT   = 0.05
SUBSET_RATIO       = 0.05   # データセット全体から使用する割合 (0.05→5%)
BATCH_SIZE         = 1024
NUM_EPOCHS         = 50
LEARNING_RATE      = 5e-4
WEIGHT_DECAY       = 0.05
CLIP_GRAD_NORM     = 1.0
ACCUMULATION_STEPS = 4
NUM_WORKERS        = min(8, os.cpu_count() or 1)
PREFETCH_FACTOR    = 4

# --- データロードと最初の分割 ---
logging.info("Scanning NPZ files...")
npz_files = sorted(glob.glob(DATA_PATTERN))
if not npz_files:
    raise FileNotFoundError(f"No files found: {DATA_PATTERN}")

# 元のデータセットをロードし、属性を保持
orig_dataset = MahjongNpzDataset(npz_files)
event_dim = orig_dataset.event_dim
static_dim = orig_dataset.static_dim

# データセットをSUBSET_RATIOだけサブセット化
if 0.0 < SUBSET_RATIO < 1.0:
    subset_size = int(len(orig_dataset) * SUBSET_RATIO)
    logging.info(f"Subsampling dataset: {SUBSET_RATIO*100:.1f}% → {subset_size} samples")
    all_indices = list(range(len(orig_dataset)))
    subset_indices = random.sample(all_indices, subset_size)
    dataset = Subset(orig_dataset, subset_indices)
else:
    logging.info("Using full dataset")
    dataset = orig_dataset

# トレイン/バリデーション分割
val_size   = int(len(dataset) * VALIDATION_SPLIT)
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

loader_kwargs = dict(
    num_workers=NUM_WORKERS,
    pin_memory=(device.type=='cuda'),
    prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=(NUM_WORKERS>0)
)
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **loader_kwargs
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, **loader_kwargs
)
logging.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# --- モデル・最適化・スケジューラ・AMPスケーラーの初期化 ---
model = MahjongRNNModel(
    event_dim=event_dim,
    static_dim=static_dim
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=max(1, NUM_EPOCHS//5), T_mult=1, eta_min=LEARNING_RATE/100
)
scaler = GradScaler(enabled=(device.type=='cuda'))

# --- トレーニングと評価関数 ---
def train_epoch(loader, epoch):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]")

    for step, (seqs, statics, labels, mask) in enumerate(pbar, start=1):
        seqs    = seqs.to(device, non_blocking=True)
        statics = statics.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)
        mask    = mask.to(device, non_blocking=True)

        with autocast():
            logits = model(seqs, statics, attention_mask=mask)
            loss = criterion(logits, labels) / ACCUMULATION_STEPS
        scaler.scale(loss).backward()

        if step % ACCUMULATION_STEPS == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item() * ACCUMULATION_STEPS
        pbar.set_postfix(loss=f"{total_loss/step:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    return total_loss / len(loader)


def evaluate(loader, epoch):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]  ")
        for seqs, statics, labels, mask in pbar:
            seqs    = seqs.to(device, non_blocking=True)
            statics = statics.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)
            mask    = mask.to(device, non_blocking=True)

            logits = model(seqs, statics, attention_mask=mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{total_loss/len(loader):.4f}", acc=f"{correct/total:.4f}")

    return total_loss / len(loader), (correct / total) if total>0 else 0.0

# --- メインループ ---
def main():
    best_acc = 0.0

    for epoch in range(1, NUM_EPOCHS+1):
        start_time = time.time()

        train_loss = train_epoch(train_loader, epoch)
        val_loss, val_acc = evaluate(val_loader, epoch)

        elapsed = time.time() - start_time
        logging.info(
            f"Epoch {epoch}/{NUM_EPOCHS} - "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("./trained_model", exist_ok=True)
            torch.save(model.state_dict(), os.path.join("./trained_model", "best_rnn.pth"))
            logging.info(f"New best model saved. Val Acc: {best_acc:.4f}")

    logging.info("Training complete.")

if __name__ == "__main__":
    main()
