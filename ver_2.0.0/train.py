# /ver_1.1.11/train_strong.py (最強AI向け学習スクリプト - メトリクス保存強化版)
# ===============================================================================
# =                              ライブラリのインポート                              =
# ===============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import numpy as np
import os
import glob
import math
from tqdm import tqdm
import logging
import sys
import time
import random
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import gc
import pandas as pd
import pyarrow.parquet as pq
from torch.amp import autocast
from torch.cuda.amp import autocast, GradScaler
import argparse

# ===============================================================================
# =                         プロジェクトモジュールのインポート                           =
# ===============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path: sys.path.insert(0, script_dir)
try:
    from game_state import NUM_TILE_TYPES, MAX_EVENT_HISTORY, EVENT_TYPES, STATIC_FEATURE_DIM
    logging.info(f"インポートされた定数: NUM_TILE_TYPES={NUM_TILE_TYPES}, STATIC_FEATURE_DIM={STATIC_FEATURE_DIM}")
except ImportError as e:
    logging.critical(f"[致命的エラー] game_state.pyからのインポートに失敗: {e}")
    sys.exit(1)

# ===============================================================================
# =                                  設定値                                   =
# ===============================================================================
# --- データ関連 ---
DATA_PARQUET_PATH = "./training_data/mahjong_imitation_data_v_strong_flat.parquet"
VALIDATION_SPLIT = 0.05

# --- モデル保存関連 ---
MODEL_SAVE_PATH = "./trained_model/mahjong_transformer_v_strong_flat.pth"
CHECKPOINT_DIR = "./checkpoints_v_strong_flat/"

# --- ログ・プロット関連 ---
LOG_DIR = "./logs_v_strong_flat"
PLOT_DIR = "./plots_v_strong_flat"
PLOT_EVERY_EPOCH = 1
INTERACTIVE_PLOT = False

# --- トレーニングハイパーパラメータ ---
BATCH_SIZE = 1024
NUM_EPOCHS = 100
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.05
CLIP_GRAD_NORM = 1.0
ACCUMULATION_STEPS = 2

# --- Transformerモデルハイパーパラメータ ---
D_MODEL = 256
NHEAD = 4
D_HID = 1024
NLAYERS = 4
DROPOUT = 0.1
ACTIVATION = 'gelu'

# --- 高度なトレーニング機能 ---
USE_AMP = True
USE_TORCH_COMPILE = True
COMPILE_MODE = "reduce-overhead"
LABEL_SMOOTHING = 0.1

# --- その他 ---
EARLY_STOPPING_PATIENCE = 10
SEED = 42
EVENT_FEATURE_DIM = 6

# ===============================================================================
# =                         各種セットアップ (ログ、デバイスなど)                        =
# ===============================================================================
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

log_file_path = os.path.join(LOG_DIR, f"training_{os.path.basename(MODEL_SAVE_PATH).replace('.pth','')}.log")
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(log_file_path, mode='w'), logging.StreamHandler()])

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    logging.info(f"CUDA Device: {torch.cuda.get_device_name(DEVICE)}, BF16 Supported: {bf16_supported}")
else:
    DEVICE = torch.device("cpu")
    USE_AMP = False; USE_TORCH_COMPILE = False
logging.info(f"使用デバイス: {DEVICE}")

# ==============================================================================
# =                       モデルとデータセットのクラス定義                        =
# ==============================================================================

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_EVENT_HISTORY):
        super().__init__()
        if d_model % 2 != 0: raise ValueError("d_modelは2で割り切れる必要があります。")
        freqs = 1.0 / (10000 ** (torch.arange(0, d_model // 2).float() / (d_model // 2)))
        self.register_buffer('freqs', freqs)
        self.register_buffer('pos_seq', torch.arange(max_len).float())

    def forward(self, x):
        seq_len = x.shape[1]
        positions = self.pos_seq[:seq_len].unsqueeze(0).to(x.device)
        angles = positions.unsqueeze(-1) * self.freqs.unsqueeze(0).to(x.device)
        sin_angles, cos_angles = torch.sin(angles), torch.cos(angles)
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = x_even * cos_angles - x_odd * sin_angles
        x_rotated[..., 1::2] = x_even * sin_angles + x_odd * cos_angles
        return x_rotated

class MahjongTransformer(nn.Module):
    def __init__(self, event_feature_dim, static_feature_dim):
        super().__init__()
        self.d_model = D_MODEL
        self.event_encoder = nn.Sequential(nn.Linear(event_feature_dim, D_MODEL), nn.LayerNorm(D_MODEL), nn.Dropout(DROPOUT))
        self.pos_encoder = RotaryPositionalEncoding(D_MODEL)
        encoder_layer = TransformerEncoderLayer(d_model=D_MODEL, nhead=NHEAD, dim_feedforward=D_HID, dropout=DROPOUT, activation=ACTIVATION, batch_first=True, norm_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, NLAYERS)
        self.static_encoder = nn.Sequential(nn.Linear(static_feature_dim, D_MODEL), nn.LayerNorm(D_MODEL), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(D_MODEL, D_MODEL), nn.LayerNorm(D_MODEL))
        self.attention_pool = nn.Sequential(nn.Linear(D_MODEL, 1), nn.Softmax(dim=1))
        self.output_head = nn.Sequential(nn.Linear(D_MODEL * 2, D_MODEL), nn.LayerNorm(D_MODEL), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(D_MODEL, D_MODEL // 2), nn.LayerNorm(D_MODEL // 2), nn.GELU(), nn.Dropout(DROPOUT * 0.5), nn.Linear(D_MODEL // 2, NUM_TILE_TYPES))
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=nn.init.calculate_gain('gelu'))
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, event_seq, static_feat, attention_mask=None):
        event_encoded = self.event_encoder(event_seq)
        pos_encoded = self.pos_encoder(event_encoded)
        transformer_output = self.transformer_encoder(pos_encoded, src_key_padding_mask=attention_mask)
        attn_weights = self.attention_pool(transformer_output)
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(-1), 0.0)
        context_vector = torch.sum(attn_weights * transformer_output, dim=1)
        static_encoded = self.static_encoder(static_feat)
        combined = torch.cat([context_vector, static_encoded], dim=1)
        return self.output_head(combined)

class MahjongParquetDataset(Dataset):
    def __init__(self, parquet_path):
        self.parquet_path = parquet_path
        try:
            self.table = pq.read_table(parquet_path, memory_map=True)
            self.length = len(self.table)
            if self.length == 0: raise RuntimeError(f"Parquetファイルにサンプルが含まれていません: {parquet_path}")
            logging.info(f"Parquet Dataset initialized: {self.length} samples.")
        except Exception as e:
            logging.critical(f"Parquetファイルの読み込みまたは初期化に失敗しました: {parquet_path} - {e}")
            raise

    def __len__(self): return self.length

    def __getitem__(self, idx):
        row = self.table.slice(idx, 1).to_pydict()
        seq_flat_np = np.array(row['sequences_flat'][0], dtype=np.float32)
        static_np = np.array(row['static_features'][0], dtype=np.float32)
        label = row['labels'][0]
        try:
            seq_np = seq_flat_np.reshape(MAX_EVENT_HISTORY, EVENT_FEATURE_DIM)
        except ValueError as e:
            logging.error(f"サンプル {idx} のシーケンス形状の復元に失敗。Error: {e}"); seq_np = np.zeros((MAX_EVENT_HISTORY, EVENT_FEATURE_DIM), dtype=np.float32)
        seq, static, label_tensor = torch.from_numpy(seq_np), torch.from_numpy(static_np), torch.tensor(label, dtype=torch.long)
        padding_mask = (seq[:, 0] == float(EVENT_TYPES["PADDING"]))
        return seq, static, label_tensor, padding_mask

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, pred_logits, target):
        pred_log_softmax = torch.log_softmax(pred_logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.full_like(pred_log_softmax, self.smoothing / (NUM_TILE_TYPES - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return self.criterion(pred_log_softmax, true_dist)

# ==============================================================================
# =                          トレーニング関連のヘルパー関数                          =
# ==============================================================================
def calculate_accuracy(predictions, targets):
    with torch.no_grad():
        _, predicted_indices = torch.max(predictions, 1)
        accuracy = (predicted_indices == targets).float().mean().item()
        _, top3_indices = torch.topk(predictions, 3, dim=1)
        top3_correct = torch.any(top3_indices == targets.unsqueeze(1), dim=1).float()
        top3_accuracy = top3_correct.mean().item()
    return accuracy, top3_accuracy

def update_plots(fig, axs, epoch, metrics):
    epochs = list(range(1, epoch + 2))
    for key in metrics:
        min_len = min(len(metrics[key]), len(epochs))
        metrics[key], current_epochs = metrics[key][:min_len], epochs[:min_len]
    axs[0,0].clear(); axs[0,0].plot(current_epochs, metrics['train_loss'], 'b-', label='Train'); axs[0,0].plot(current_epochs, metrics['val_loss'], 'r-', label='Val'); axs[0,0].legend(); axs[0,0].grid(True); axs[0,0].set_title('Loss')
    axs[0,1].clear(); axs[0,1].plot(current_epochs, metrics['train_acc'], 'b-', label='Train'); axs[0,1].plot(current_epochs, metrics['val_acc'], 'r-', label='Val'); axs[0,1].legend(); axs[0,1].grid(True); axs[0,1].set_title('Accuracy')
    axs[1,0].clear(); axs[1,0].plot(current_epochs, metrics['train_top3'], 'b-', label='Train'); axs[1,0].plot(current_epochs, metrics['val_top3'], 'r-', label='Val'); axs[1,0].legend(); axs[1,0].grid(True); axs[1,0].set_title('Top-3 Accuracy')
    axs[1,1].clear(); axs[1,1].plot(current_epochs, metrics['lr'], 'g-'); axs[1,1].grid(True); axs[1,1].set_title('Learning Rate'); axs[1,1].set_yscale('log')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'latest_training_metrics.png'))

def find_latest_checkpoint(checkpoint_dir):
    """指定されたディレクトリから最新のチェックポイントを見つける"""
    try:
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
        if not checkpoints: return None
        return max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    except Exception as e:
        logging.error(f"最新チェックポイントの検索中にエラー: {e}"); return None

# ==============================================================================
# =                            メイン学習ループ                            =
# ==============================================================================
def train_model(resume=False):
    global USE_TORCH_COMPILE
    logging.info("最強AIモデルの学習プロセスを開始します...")

    if not os.path.exists(DATA_PARQUET_PATH):
        logging.critical(f"データファイルが見つかりません: {DATA_PARQUET_PATH}"); sys.exit(1)
    
    full_dataset = MahjongParquetDataset(DATA_PARQUET_PATH)
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    logging.info(f"データセット分割: Train={len(train_dataset)}, Validation={len(val_dataset)}")

    num_workers = min(os.cpu_count() // 2, 8)
    pin_memory = (DEVICE.type == 'cuda')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0), drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))
    logging.info(f"データローダー準備完了 (ワーカー数: {num_workers})")

    model = MahjongTransformer(event_feature_dim=EVENT_FEATURE_DIM, static_feature_dim=STATIC_FEATURE_DIM).to(DEVICE)
    logging.info(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    if USE_TORCH_COMPILE and hasattr(torch, 'compile'):
        try: logging.info(f"torch.compileを適用中 (モード: {COMPILE_MODE})..."); model = torch.compile(model, mode=COMPILE_MODE)
        except Exception as e: logging.warning(f"torch.compile失敗: {e}"); USE_TORCH_COMPILE = False

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LEARNING_RATE / 100)
    criterion = LabelSmoothingLoss(smoothing=LABEL_SMOOTHING) if LABEL_SMOOTHING > 0 else nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=USE_AMP)
    amp_dtype = torch.bfloat16 if bf16_supported else torch.float16
    logging.info(f"初期設定完了 (Scheduler: CosineAnnealingLR, Loss: {type(criterion).__name__}, AMP: {USE_AMP})")

    start_epoch = 0
    metrics = {k: [] for k in ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_top3', 'val_top3', 'lr']}
    best_val_acc = 0.0
    epochs_without_improvement = 0

    if resume:
        latest_checkpoint_path = find_latest_checkpoint(CHECKPOINT_DIR)
        if latest_checkpoint_path:
            logging.info(f"チェックポイントから学習を再開します: {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path, map_location=DEVICE)
            model_state_dict = checkpoint['model_state_dict']
            if USE_TORCH_COMPILE: model._orig_mod.load_state_dict(model_state_dict)
            else: model.load_state_dict(model_state_dict)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch']
            metrics = checkpoint.get('metrics', metrics)
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
            logging.info(f"エポック {start_epoch + 1} から学習を再開します。")
        else:
            logging.warning("再開が要求されましたが、チェックポイントが見つかりません。最初から学習を開始します。")

    fig, axs = plt.subplots(2, 2, figsize=(15, 10)); plt.tight_layout()
    total_start_time = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start_time = time.time()
        
        model.train()
        train_loss_accum, train_acc_accum, train_top3_accum, num_train_samples = 0.0, 0.0, 0.0, 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
        for i, (seq, static, labels, mask) in enumerate(pbar):
            seq, static, labels, mask = seq.to(DEVICE), static.to(DEVICE), labels.to(DEVICE), mask.to(DEVICE)
            with autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=USE_AMP):
                outputs = model(seq, static, mask)
                loss = criterion(outputs, labels) / ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                if CLIP_GRAD_NORM > 0: scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            with torch.no_grad(): acc1, acc3 = calculate_accuracy(outputs, labels)
            bs = labels.size(0); train_loss_accum += loss.item() * ACCUMULATION_STEPS; train_acc_accum += acc1 * bs; train_top3_accum += acc3 * bs; num_train_samples += bs
            pbar.set_postfix({'Loss': f'{train_loss_accum / (i+1):.4f}', 'Acc': f'{train_acc_accum / num_train_samples:.3f}'})

        model.eval()
        val_loss_accum, val_acc_accum, val_top3_accum, num_val_samples = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for seq, static, labels, mask in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=False):
                seq, static, labels, mask = seq.to(DEVICE), static.to(DEVICE), labels.to(DEVICE), mask.to(DEVICE)
                with autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=USE_AMP):
                    outputs = model(seq, static, mask); loss = criterion(outputs, labels)
                acc1, acc3 = calculate_accuracy(outputs, labels)
                bs = labels.size(0); val_loss_accum += loss.item() * bs; val_acc_accum += acc1 * bs; val_top3_accum += acc3 * bs; num_val_samples += bs

        epoch_train_loss = train_loss_accum / len(train_loader); epoch_train_acc = train_acc_accum / num_train_samples; epoch_train_top3 = train_top3_accum / num_train_samples
        epoch_val_loss = val_loss_accum / num_val_samples; epoch_val_acc = val_acc_accum / num_val_samples; epoch_val_top3 = val_top3_accum / num_val_samples
        metrics['train_loss'].append(epoch_train_loss); metrics['val_loss'].append(epoch_val_loss); metrics['train_acc'].append(epoch_train_acc); metrics['val_acc'].append(epoch_val_acc); metrics['train_top3'].append(epoch_train_top3); metrics['val_top3'].append(epoch_val_top3); metrics['lr'].append(optimizer.param_groups[0]['lr'])
        
        logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS} - Time: {time.time()-epoch_start_time:.1f}s - "
                     f"Train L:{epoch_train_loss:.4f} A:{epoch_train_acc:.4f} T3:{epoch_train_top3:.4f} | "
                     f"Val L:{epoch_val_loss:.4f} A:{epoch_val_acc:.4f} T3:{epoch_val_top3:.4f}")
        
        if (epoch + 1) % PLOT_EVERY_EPOCH == 0: update_plots(fig, axs, epoch, metrics)

        is_best = epoch_val_acc > best_val_acc
        if is_best:
            best_val_acc = epoch_val_acc; epochs_without_improvement = 0
            save_model_state = model._orig_mod.state_dict() if USE_TORCH_COMPILE else model.state_dict()
            torch.save({'model_state_dict': save_model_state}, MODEL_SAVE_PATH)
            logging.info(f"*** New best model saved at epoch {epoch+1} with Val Acc: {best_val_acc:.4f} ***")
        else:
            epochs_without_improvement += 1
        
        # ★★★ チェックポイントに全ての情報を保存 ★★★
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model._orig_mod.state_dict() if USE_TORCH_COMPILE else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'metrics': metrics,  # 全てのメトリクス履歴を保存
            'best_val_acc': best_val_acc,
            'epochs_without_improvement': epochs_without_improvement,
            'val_acc': epoch_val_acc # このエポックのval_accも直接保存
        }, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth"))
        
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            logging.info(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
            break
        
        lr_scheduler.step()

    logging.info("="*30 + "\n学習プロセス完了。"); logging.info(f"総学習時間: {(time.time() - total_start_time)/60:.2f} 分"); logging.info(f"最良検証精度: {best_val_acc:.4f}")
    fig.savefig(os.path.join(PLOT_DIR, 'final_training_curves.png')); plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Mahjong Transformer model from Parquet data.')
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint')
    args = parser.parse_args()
    train_model(resume=args.resume)