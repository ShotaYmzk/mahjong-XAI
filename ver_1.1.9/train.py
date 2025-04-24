# /ver_1.1.9/train.py
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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler # For mixed precision
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# --- Ensure game_state constants can be imported if needed ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path: sys.path.insert(0, script_dir)
try:
    from game_state import NUM_TILE_TYPES, MAX_EVENT_HISTORY, EVENT_TYPES, STATIC_FEATURE_DIM
    print(f"Imported constants: NUM_TILE_TYPES={NUM_TILE_TYPES}, MAX_EVENT_HISTORY={MAX_EVENT_HISTORY}, STATIC_FEATURE_DIM={STATIC_FEATURE_DIM}")
except ImportError as e:
     print(f"[FATAL ERROR in train.py] Cannot import constants from game_state.py: {e}")
     NUM_TILE_TYPES = 34; MAX_EVENT_HISTORY = 60; STATIC_FEATURE_DIM = 157; EVENT_TYPES = {"PADDING": 8}
     print(f"[Warning] Using fallback constants: NUM_TILE_TYPES={NUM_TILE_TYPES}, MAX_EVENT_HISTORY={MAX_EVENT_HISTORY}, STATIC_FEATURE_DIM={STATIC_FEATURE_DIM}")
# -----------------------------------------------------

# --- Configuration ---
DATA_DIR = "./training_data/"
DATA_PATTERN = os.path.join(DATA_DIR, "mahjong_imitation_data_v119_batch_*.npz")
MODEL_SAVE_PATH = "./trained_model/mahjong_transformer_v119_large.pth"
CHECKPOINT_DIR = "./checkpoints_v119_large/"
LOG_DIR = "./logs"
PLOT_DIR = "./plots_v119_large"

# Training hyperparameters
BATCH_SIZE = 4096
NUM_EPOCHS = 50
LEARNING_RATE = 5e-4
VALIDATION_SPLIT = 0.05
WEIGHT_DECAY = 0.05
CLIP_GRAD_NORM = 1.0
WARMUP_STEPS = 1000
ACCUMULATION_STEPS = 1

# Transformer Model Hyperparameters
D_MODEL = 512
NHEAD = 8
D_HID = 2048
NLAYERS = 6
DROPOUT = 0.1
ACTIVATION = 'gelu' # Using GELU

# Advanced Training Features
USE_AMP = True
USE_EMA = False
EMA_DECAY = 0.999
LABEL_SMOOTHING = 0.1

# Plot settings
PLOT_EVERY_EPOCH = 1
INTERACTIVE_PLOT = False

# Early stopping settings
EARLY_STOPPING_PATIENCE = 7

# --- Device Configuration ---
if torch.cuda.is_available():
     DEVICE = torch.device("cuda"); torch.backends.cudnn.benchmark = True; torch.set_float32_matmul_precision('high')
     print(f"CUDA Device: {torch.cuda.get_device_name(DEVICE)}"); print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}"); print(f"TF32 Matmul Precision: {torch.get_float32_matmul_precision()}")
else: DEVICE = torch.device("cpu"); print("[Warning] CUDA not available, using CPU.")
print(f"Using device: {DEVICE}")

# Seeds
SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(SEED)

# Create directories
os.makedirs(LOG_DIR, exist_ok=True); os.makedirs(CHECKPOINT_DIR, exist_ok=True); os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True); os.makedirs(PLOT_DIR, exist_ok=True)

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(os.path.join(LOG_DIR, "model_training_v119_large.log"), mode='w'), logging.StreamHandler()])

# --- Positional Encoding ---
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_EVENT_HISTORY):
        super().__init__(); assert d_model % 2 == 0; self.d_model = d_model; self.max_len = max_len; self.dim_half = d_model // 2
        freqs = 1.0 / (10000 ** (torch.arange(0, self.dim_half).float() / self.dim_half)); self.register_buffer('freqs', freqs)
        pos_seq = torch.arange(max_len).float(); self.register_buffer('pos_seq', pos_seq)
    def forward(self, x):
        seq_len = x.shape[1]; assert seq_len <= self.max_len
        positions = self.pos_seq[:seq_len].unsqueeze(0).to(x.device); angles = positions.unsqueeze(-1) * self.freqs.unsqueeze(0).to(x.device)
        sin = torch.sin(angles); cos = torch.cos(angles); x_even, x_odd = x[..., 0::2], x[..., 1::2]
        x_even_rotated = x_even * cos - x_odd * sin; x_odd_rotated = x_odd * cos + x_even * sin
        x_rotated = torch.zeros_like(x); x_rotated[..., 0::2] = x_even_rotated; x_rotated[..., 1::2] = x_odd_rotated
        return x_rotated

# --- Transformer Model Definition ---
class MahjongTransformerV2(nn.Module):
    def __init__(self, event_feature_dim, static_feature_dim, d_model=D_MODEL, nhead=NHEAD, d_hid=D_HID, nlayers=NLAYERS, dropout=DROPOUT, activation=ACTIVATION, output_dim=NUM_TILE_TYPES):
        super().__init__(); self.d_model = d_model
        self.event_encoder = nn.Sequential(nn.Linear(event_feature_dim, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout))
        self.pos_encoder = RotaryPositionalEncoding(d_model)
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout, activation=activation, batch_first=True, norm_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.static_encoder = nn.Sequential(nn.Linear(static_feature_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        self.attention_pool = nn.Sequential(nn.Linear(d_model, 1), nn.Softmax(dim=1))
        self.output_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2), nn.LayerNorm(d_model // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, output_dim))
        self._init_weights() # Call initialization

    def _init_weights(self):
        """Initialize weights using Xavier normal"""
        for name, p in self.named_parameters():
            if p.dim() > 1: # Weight matrices
                # Use gain=1.0 for most activations including GELU, specific gain for ReLU
                gain = 1.0 if ACTIVATION != 'relu' else nn.init.calculate_gain('relu') # <<< FIX HERE
                nn.init.xavier_normal_(p, gain=gain)
            elif 'bias' in name: # Bias terms
                nn.init.zeros_(p)

    def forward(self, event_seq, static_feat, attention_mask=None):
        event_encoded = self.event_encoder(event_seq); pos_encoded = self.pos_encoder(event_encoded)
        transformer_output = self.transformer_encoder(pos_encoded, src_key_padding_mask=attention_mask)
        attn_weights = self.attention_pool(transformer_output)
        if attention_mask is not None: attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(-1), 0.0)
        context_vector = torch.sum(attn_weights * transformer_output, dim=1)
        static_encoded = self.static_encoder(static_feat); combined = torch.cat([context_vector, static_encoded], dim=1)
        return self.output_head(combined)

# --- EMA --- (Optional)
class EMA:
    def __init__(self, model, decay): self.model=model; self.decay=decay; self.shadow={}; self.backup={}; self.register()
    def register(self): self.shadow = {name: param.data.clone() for name, param in self.model.named_parameters() if param.requires_grad}
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad: self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
    def apply_shadow(self): self.backup = {name: param.data.clone() for name, param in self.model.named_parameters() if param.requires_grad}; self.model.load_state_dict(self.shadow, strict=False)
    def restore(self): self.model.load_state_dict(self.backup, strict=False); self.backup = {}

# --- Dataset ---
class MahjongNpzDataset(Dataset):
    def __init__(self, npz_files):
        self.npz_files = npz_files; self.file_metadata = []; self.cumulative_lengths = [0]; self.total_length = 0
        self.seq_len = -1; self.event_dim = -1; self.static_dim = -1
        self.padding_code = float(EVENT_TYPES["PADDING"]); self.sequences_field = 'sequences'; self.static_field = 'static_features'; self.labels_field = 'labels'
        logging.info("Scanning NPZ files for metadata..."); first_file = True
        skipped_count = 0
        for f in tqdm(self.npz_files, desc="Scanning files", leave=False):
            try:
                with np.load(f, allow_pickle=False) as data:
                    if not all(k in data for k in [self.sequences_field, self.static_field, self.labels_field]):
                        # logging.warning(f"Skipping {f} - missing fields. Found: {list(data.files)}") # Reduce log noise
                        skipped_count += 1
                        continue
                    length = len(data[self.labels_field])
                    if length == 0: skipped_count += 1; continue # Skip empty
                    self.file_metadata.append({'path': f, 'length': length}); self.total_length += length; self.cumulative_lengths.append(self.total_length)
                    if first_file:
                        self.seq_len, self.event_dim = data[self.sequences_field].shape[1:]; loaded_static_dim = data[self.static_field].shape[1]
                        if loaded_static_dim != STATIC_FEATURE_DIM: raise ValueError(f"Static dim mismatch! Expected {STATIC_FEATURE_DIM}, got {loaded_static_dim} in {f}")
                        self.static_dim = loaded_static_dim; first_file = False
            except Exception as e: logging.error(f"Error reading metadata from {f}: {e}"); skipped_count += 1
        if self.total_length == 0: raise RuntimeError("No valid data found in NPZ files.")
        logging.info(f"Dataset initialized: {len(self.file_metadata)} files ({skipped_count} skipped), {self.total_length} samples.")
        logging.info(f"Dims: SeqLen={self.seq_len}, Event={self.event_dim}, Static={self.static_dim}")
    def __len__(self): return self.total_length
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_length: raise IndexError("Index out of bounds")
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        file_info = self.file_metadata[file_idx]; offset = idx - self.cumulative_lengths[file_idx]
        try:
            with np.load(file_info['path'], allow_pickle=False) as data:
                seq = data[self.sequences_field][offset].astype(np.float32); static = data[self.static_field][offset].astype(np.float32)
                label = data[self.labels_field][offset].astype(np.int64); padding_mask = (seq[:, 0] == self.padding_code)
                if seq.shape != (self.seq_len, self.event_dim): seq = np.zeros((self.seq_len, self.event_dim), dtype=np.float32)
                if static.shape != (self.static_dim,): static = np.zeros((self.static_dim,), dtype=np.float32)
                return seq, static, label, padding_mask
        except Exception as e:
            logging.error(f"Error loading sample {idx} from {file_info['path']}: {e}")
            return np.zeros((self.seq_len, self.event_dim), dtype=np.float32), np.zeros((self.static_dim,), dtype=np.float32), np.zeros((), dtype=np.int64), np.ones((self.seq_len,), dtype=bool)

# --- Loss function ---
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0, num_classes=NUM_TILE_TYPES):
        super().__init__(); self.smoothing=smoothing; self.num_classes=num_classes; self.confidence=1.0-smoothing; self.criterion=nn.KLDivLoss(reduction='batchmean')
    def forward(self, pred, target):
        pred = torch.log_softmax(pred, dim=-1);
        with torch.no_grad(): true_dist = torch.zeros_like(pred); true_dist.fill_(self.smoothing / (self.num_classes - 1)); true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) # Use target.data for scatter_
        return self.criterion(pred, true_dist)

# --- Metrics ---
def calculate_accuracy(predictions, targets):
    with torch.no_grad(): _, predicted = torch.max(predictions, 1); accuracy = (predicted == targets).float().mean().item(); _, top3_indices = torch.topk(predictions, 3, dim=1); top3_correct = torch.any(top3_indices == targets.unsqueeze(1), dim=1).float().mean().item()
    return accuracy, top3_correct

# --- Plotting ---
def init_plots():
    fig, axs = plt.subplots(2, 2, figsize=(15, 10)); axs[0, 0].set_title('Loss'); axs[0, 1].set_title('Accuracy'); axs[1, 0].set_title('Top-3 Accuracy'); axs[1, 1].set_title('Learning Rate'); plt.tight_layout(); return fig, axs
def update_plots(fig, axs, epoch, metrics):
    epochs = list(range(1, epoch + 2)); axs[0,0].clear(); axs[0,0].plot(epochs, metrics['train_loss'], 'b-', label='Train'); axs[0,0].plot(epochs, metrics['val_loss'], 'r-', label='Val'); axs[0,0].legend(); axs[0,0].grid(True); axs[0,0].set_title('Loss')
    axs[0,1].clear(); axs[0,1].plot(epochs, metrics['train_acc'], 'b-', label='Train'); axs[0,1].plot(epochs, metrics['val_acc'], 'r-', label='Val'); axs[0,1].legend(); axs[0,1].grid(True); axs[0,1].set_title('Accuracy')
    axs[1,0].clear(); axs[1,0].plot(epochs, metrics['train_top3'], 'b-', label='Train'); axs[1,0].plot(epochs, metrics['val_top3'], 'r-', label='Val'); axs[1,0].legend(); axs[1,0].grid(True); axs[1,0].set_title('Top-3 Accuracy')
    axs[1,1].clear(); axs[1,1].plot(epochs, metrics['lr'], 'g-'); axs[1,1].grid(True); axs[1,1].set_title('Learning Rate'); plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, f'training_metrics_epoch_{epoch+1}.png'); latest_path = os.path.join(PLOT_DIR, 'latest_training_metrics.png')
    try: fig.savefig(plot_path); fig.savefig(latest_path)
    except Exception as e: logging.warning(f"Failed to save plot: {e}")
    if INTERACTIVE_PLOT: plt.pause(0.1)

# --- Training Loop ---
def train_model():
    npz_files = sorted(glob.glob(DATA_PATTERN))
    if not npz_files: logging.error(f"No NPZ files found: {DATA_PATTERN}"); return
    logging.info(f"Found {len(npz_files)} NPZ files")

    full_dataset = MahjongNpzDataset(npz_files)
    event_dim = full_dataset.event_dim; static_dim = full_dataset.static_dim

    val_size = int(len(full_dataset) * VALIDATION_SPLIT); train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
    logging.info(f"Train samples: {train_size}, Validation samples: {val_size}")

    num_workers = min(16, os.cpu_count() if os.cpu_count() else 4)
    logging.info(f"Using {num_workers} DataLoader workers.")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True if DEVICE.type=='cuda' else False, prefetch_factor=4 if num_workers > 0 else None, persistent_workers=num_workers > 0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=num_workers, pin_memory=True if DEVICE.type=='cuda' else False, prefetch_factor=4 if num_workers > 0 else None, persistent_workers=num_workers > 0)

    # --- Model, Optimizer, Scheduler, Loss ---
    model = MahjongTransformerV2(event_feature_dim=event_dim, static_feature_dim=static_dim).to(DEVICE)
    logging.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    ema = EMA(model, decay=EMA_DECAY) if USE_EMA else None
    criterion = LabelSmoothingLoss(smoothing=LABEL_SMOOTHING) if LABEL_SMOOTHING > 0 else nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98), eps=1e-6)
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, NUM_EPOCHS // 5), T_mult=1, eta_min=LEARNING_RATE / 50)
    scaler = GradScaler(enabled=USE_AMP and DEVICE.type == 'cuda')
    # --- End Initialization ---

    metrics = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_top3': [], 'val_top3': [], 'lr': []}
    best_val_acc = 0.0; best_epoch = 0; epochs_without_improvement = 0
    if INTERACTIVE_PLOT: plt.ion()
    fig, axs = init_plots()

    logging.info(f"Starting training for {NUM_EPOCHS} epochs on {DEVICE}...")
    for epoch in range(NUM_EPOCHS):
        model.train(); train_loss = 0.0; train_acc = 0.0; train_top3 = 0.0; train_batches = 0; start_time = time.time()
        optimizer.zero_grad(set_to_none=True) # More efficient zeroing

        for i, (seq, static, labels, padding_mask) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)):
            seq, static, labels, padding_mask = seq.to(DEVICE, non_blocking=True), static.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True), padding_mask.to(DEVICE, non_blocking=True)

            with autocast(device_type=DEVICE.type, enabled=scaler.is_enabled()):
                outputs = model(seq, static, padding_mask)
                loss = criterion(outputs, labels)
                if ACCUMULATION_STEPS > 1: loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                if CLIP_GRAD_NORM > 0: scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                if ema: ema.update()

            acc, acc_top3 = calculate_accuracy(outputs.detach(), labels)
            train_loss += loss.item() * ACCUMULATION_STEPS; train_acc += acc; train_top3 += acc_top3; train_batches += 1

        # --- End of Training Epoch ---
        epoch_train_loss = train_loss / train_batches; epoch_train_acc = train_acc / train_batches; epoch_train_acc_top3 = train_acc_top3 / train_batches
        metrics['train_loss'].append(epoch_train_loss); metrics['train_acc'].append(epoch_train_acc); metrics['train_top3'].append(epoch_train_acc_top3)
        current_lr = optimizer.param_groups[0]['lr']; metrics['lr'].append(current_lr)

        # --- Validation ---
        model.eval(); val_loss = 0.0; val_acc = 0.0; val_top3 = 0.0; val_batches = 0
        if ema: ema.apply_shadow()
        with torch.no_grad():
            for seq, static, labels, padding_mask in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=False):
                seq, static, labels, padding_mask = seq.to(DEVICE, non_blocking=True), static.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True), padding_mask.to(DEVICE, non_blocking=True)
                with autocast(device_type=DEVICE.type, enabled=scaler.is_enabled()):
                    outputs = model(seq, static, padding_mask)
                    loss = criterion(outputs, labels)
                acc, acc_top3 = calculate_accuracy(outputs, labels)
                val_loss += loss.item(); val_acc += acc; val_top3 += acc_top3; val_batches += 1
        epoch_val_loss = val_loss / val_batches if val_batches > 0 else 0.0; epoch_val_acc = val_acc / val_batches if val_batches > 0 else 0.0; epoch_val_acc_top3 = val_top3 / val_batches if val_batches > 0 else 0.0
        metrics['val_loss'].append(epoch_val_loss); metrics['val_acc'].append(epoch_val_acc); metrics['val_top3'].append(epoch_val_acc_top3)
        if ema: ema.restore()
        # --- End Validation ---

        epoch_time = time.time() - start_time
        logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS} - Time: {epoch_time:.1f}s - Train Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.4f} - Top3: {epoch_train_acc_top3:.4f} | Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc:.4f} - Top3: {epoch_val_acc_top3:.4f} | LR: {current_lr:.6f}")

        lr_scheduler.step() # Step scheduler every epoch for CosineAnnealing

        if (epoch + 1) % PLOT_EVERY_EPOCH == 0 or epoch == NUM_EPOCHS - 1: update_plots(fig, axs, epoch, metrics)

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_acc': epoch_val_acc}, checkpoint_path)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc; best_epoch = epoch + 1; epochs_without_improvement = 0
            save_model_state = ema.shadow if ema else model.state_dict()
            torch.save(save_model_state, MODEL_SAVE_PATH); logging.info(f"*** New best model saved at epoch {best_epoch} with Val Acc: {best_val_acc:.4f} ***")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE: logging.info(f"Early stopping at epoch {epoch+1}."); break

    # Final save and cleanup
    update_plots(fig, axs, epoch, metrics); plt.figure(fig); plt.savefig(os.path.join(LOG_DIR, 'final_training_curves_v119_large.png'))
    if INTERACTIVE_PLOT: plt.ioff(); plt.show()
    else: plt.close(fig)
    logging.info(f"Training completed. Best model from epoch {best_epoch} (Val Acc: {best_val_acc:.4f}) saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()