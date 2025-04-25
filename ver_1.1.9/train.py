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
import gc # ガベージコレクタをインポート

# --- Ensure game_state constants can be imported if needed ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path: sys.path.insert(0, script_dir)
try:
    # game_state から定数をインポート (エラー処理はそのまま)
    from game_state import NUM_TILE_TYPES, MAX_EVENT_HISTORY, EVENT_TYPES, STATIC_FEATURE_DIM
    print(f"Imported constants: NUM_TILE_TYPES={NUM_TILE_TYPES}, MAX_EVENT_HISTORY={MAX_EVENT_HISTORY}, STATIC_FEATURE_DIM={STATIC_FEATURE_DIM}")
except ImportError as e:
     print(f"[FATAL ERROR in train.py] Cannot import constants from game_state.py: {e}")
     NUM_TILE_TYPES = 34; MAX_EVENT_HISTORY = 60; STATIC_FEATURE_DIM = 157; EVENT_TYPES = {"PADDING": 8}
     print(f"[Warning] Using fallback constants: NUM_TILE_TYPES={NUM_TILE_TYPES}, MAX_EVENT_HISTORY={MAX_EVENT_HISTORY}, STATIC_FEATURE_DIM={STATIC_FEATURE_DIM}")
# -----------------------------------------------------

# --- Configuration (変更なし) ---
DATA_DIR = "./training_data/"
DATA_PATTERN = os.path.join(DATA_DIR, "mahjong_imitation_data_v119_batch_*.npz")
MODEL_SAVE_PATH = "./trained_model/mahjong_transformer_v119_large.pth"
CHECKPOINT_DIR = "./checkpoints_v119_large/"
LOG_DIR = "./logs"
PLOT_DIR = "./plots_v119_large"

# Training hyperparameters (変更なし)
BATCH_SIZE = 4096 # VRAMが許す限り大きく保つ
NUM_EPOCHS = 50
LEARNING_RATE = 5e-4
VALIDATION_SPLIT = 0.05
WEIGHT_DECAY = 0.05
CLIP_GRAD_NORM = 1.0
WARMUP_STEPS = 1000 # エポックあたりのステップ数に基づいて調整検討
ACCUMULATION_STEPS = 1 # メモリには大きすぎる場合に増やす

# Transformer Model Hyperparameters (変更なし)
D_MODEL = 512
NHEAD = 8
D_HID = 2048 # D_MODEL * 4
NLAYERS = 6
DROPOUT = 0.1
ACTIVATION = 'gelu'

# Advanced Training Features (変更なし)
USE_AMP = True # Ampere/Turing+ GPUでは必須
USE_EMA = False
EMA_DECAY = 0.999
LABEL_SMOOTHING = 0.1

# Plot settings (変更なし)
PLOT_EVERY_EPOCH = 1
INTERACTIVE_PLOT = False

# Early stopping settings (変更なし)
EARLY_STOPPING_PATIENCE = 7

# --- Device Configuration (変更なし) ---
if torch.cuda.is_available():
     DEVICE = torch.device("cuda"); torch.backends.cudnn.benchmark = True; torch.set_float32_matmul_precision('high')
     print(f"CUDA Device: {torch.cuda.get_device_name(DEVICE)}"); print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}"); print(f"TF32 Matmul Precision: {torch.get_float32_matmul_precision()}")
else: DEVICE = torch.device("cpu"); print("[Warning] CUDA not available, using CPU.")
print(f"Using device: {DEVICE}")

# Seeds (変更なし)
SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(SEED)

# Create directories (変更なし)
os.makedirs(LOG_DIR, exist_ok=True); os.makedirs(CHECKPOINT_DIR, exist_ok=True); os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True); os.makedirs(PLOT_DIR, exist_ok=True)

# Logging Setup (変更なし)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(os.path.join(LOG_DIR, "model_training_v119_large.log"), mode='w'), logging.StreamHandler()])

# --- Positional Encoding (変更なし) ---
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_EVENT_HISTORY):
        super().__init__(); assert d_model % 2 == 0; self.d_model = d_model; self.max_len = max_len; self.dim_half = d_model // 2
        freqs = 1.0 / (10000 ** (torch.arange(0, self.dim_half).float() / self.dim_half)); self.register_buffer('freqs', freqs)
        pos_seq = torch.arange(max_len).float(); self.register_buffer('pos_seq', pos_seq)
    def forward(self, x):
        seq_len = x.shape[1]; assert seq_len <= self.max_len, f"Sequence length {seq_len} exceeds max_len {self.max_len}"
        if seq_len > self.pos_seq.size(0):
             logging.warning(f"RoPE: Input sequence length {seq_len} > precomputed positions {self.pos_seq.size(0)}. Recomputing.")
             self.pos_seq = torch.arange(seq_len, device=x.device).float()

        positions = self.pos_seq[:seq_len].unsqueeze(0).to(x.device); angles = positions.unsqueeze(-1) * self.freqs.unsqueeze(0).to(x.device)
        sin = torch.sin(angles); cos = torch.cos(angles)
        x_even, x_odd = x[..., 0:self.dim_half], x[..., self.dim_half:self.d_model] # Correct slicing
        x_even_rotated = x_even * cos - x_odd * sin; x_odd_rotated = x_odd * cos + x_even * sin
        x_rotated = torch.zeros_like(x);
        x_rotated[..., 0:self.dim_half] = x_even_rotated; x_rotated[..., self.dim_half:self.d_model] = x_odd_rotated # Correct assignment
        return x_rotated

# --- Transformer Model Definition (変更なし) ---
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
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                gain = nn.init.calculate_gain(ACTIVATION) if ACTIVATION in ['relu', 'leaky_relu'] else 1.0
                nn.init.xavier_normal_(p, gain=gain)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, event_seq, static_feat, attention_mask=None):
        event_encoded = self.event_encoder(event_seq);
        pos_encoded = self.pos_encoder(event_encoded)
        transformer_output = self.transformer_encoder(pos_encoded, src_key_padding_mask=attention_mask)

        attn_weights = self.attention_pool(transformer_output)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1)
            attn_weights = attn_weights.masked_fill(mask_expanded, 0.0)

        context_vector = torch.sum(attn_weights * transformer_output, dim=1)
        static_encoded = self.static_encoder(static_feat);
        combined = torch.cat([context_vector, static_encoded], dim=1)
        return self.output_head(combined)

# --- EMA (変更なし) ---
class EMA:
    def __init__(self, model, decay): self.model=model; self.decay=decay; self.shadow={}; self.backup={}; self.register()
    def register(self): self.shadow = {name: param.data.clone() for name, param in self.model.named_parameters() if param.requires_grad}
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad: self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
    def apply_shadow(self): self.backup = {name: param.data.clone() for name, param in self.model.named_parameters() if param.requires_grad}; self.model.load_state_dict(self.shadow, strict=False)
    def restore(self): self.model.load_state_dict(self.backup, strict=False); self.backup = {}


# --- Dataset (修正箇所) ---
class MahjongNpzDataset(Dataset):
    def __init__(self, npz_files):
        self.npz_files = npz_files; self.file_metadata = []; self.cumulative_lengths = [0]; self.total_length = 0
        self.seq_len = -1; self.event_dim = -1; self.static_dim = -1
        self.padding_code = float(EVENT_TYPES["PADDING"]); self.sequences_field = 'sequences'; self.static_field = 'static_features'; self.labels_field = 'labels'

        # ===== キャッシュ用属性 =====
        self.cache_file_idx = -1
        self.cached_data = None
        # ==========================

        logging.info("Scanning NPZ files for metadata..."); first_file = True
        skipped_count = 0
        valid_files = [] # 有効なファイルパスのみを保存
        for f in tqdm(self.npz_files, desc="Scanning files", leave=False):
            try:
                # オプション: 小さすぎるファイルをスキップ
                if os.path.getsize(f) < 100: # 100バイト未満のファイルをスキップ
                    skipped_count += 1
                    continue

                with np.load(f, allow_pickle=False) as data:
                    # 必要なフィールドが存在するか確認
                    if not all(k in data for k in [self.sequences_field, self.static_field, self.labels_field]):
                        skipped_count += 1; continue
                    # サンプル数を取得
                    length = len(data[self.labels_field])
                    if length == 0: skipped_count += 1; continue # 空のファイルはスキップ

                    # --- 次元チェック ---
                    current_seq_shape = data[self.sequences_field].shape
                    current_static_shape = data[self.static_field].shape

                    # 配列の次元数が正しいか確認 (Sequenceは3次元, Staticは2次元)
                    if len(current_seq_shape) != 3 or len(current_static_shape) != 2:
                         logging.warning(f"Skipping {f} - incorrect array dimensions. Seq: {current_seq_shape}, Static: {current_static_shape}")
                         skipped_count += 1; continue

                    # 最初の有効なファイルで次元を記録
                    if first_file:
                        # shape[0]はサンプル数なので、[1:]で特徴次元を取得
                        self.seq_len, self.event_dim = current_seq_shape[1:]
                        loaded_static_dim = current_static_shape[1]
                        # STATIC_FEATURE_DIM と一致するか確認
                        if loaded_static_dim != STATIC_FEATURE_DIM:
                             raise ValueError(f"Static dim mismatch! Expected {STATIC_FEATURE_DIM}, got {loaded_static_dim} in {f}")
                        self.static_dim = loaded_static_dim; first_file = False
                    else:
                        # 2番目以降のファイルが最初のファイルと次元が一致するか検証
                        if current_seq_shape[1:] != (self.seq_len, self.event_dim) or current_static_shape[1] != self.static_dim:
                            logging.warning(f"Skipping {f} - dimension mismatch. Seq: {current_seq_shape[1:]} (exp {self.seq_len, self.event_dim}), Static: {current_static_shape[1]} (exp {self.static_dim})")
                            skipped_count += 1; continue
                    # --- 次元チェック終了 ---

                    # 全てのチェックをパスした場合、メタデータを追加
                    self.file_metadata.append({'path': f, 'length': length})
                    self.total_length += length
                    self.cumulative_lengths.append(self.total_length)
                    valid_files.append(f) # 実際に使用するファイルのリストに追加

            except Exception as e: logging.error(f"Error reading metadata from {f}: {e}"); skipped_count += 1

        self.npz_files = valid_files # ファイルリストを有効なもののみに更新
        if self.total_length == 0: raise RuntimeError("No valid data found in NPZ files after scanning.")
        logging.info(f"Dataset initialized: {len(self.file_metadata)} files ({skipped_count} skipped), {self.total_length} samples.")
        logging.info(f"Dims: SeqLen={self.seq_len}, Event={self.event_dim}, Static={self.static_dim}")

    def __len__(self): return self.total_length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_length: raise IndexError("Index out of bounds")

        # idxがどのファイルに属するか検索
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        file_info = self.file_metadata[file_idx]
        # そのファイル内でのインデックス（オフセット）を計算
        offset = idx - self.cumulative_lengths[file_idx]

        try:
            # ===== キャッシュロジック =====
            if file_idx != self.cache_file_idx:
                # キャッシュミス：新しいファイルを読み込む
                # logging.debug(f"Cache miss for idx {idx}. Loading file {file_idx}: {file_info['path']}") # オプションのデバッグログ
                # 以前のキャッシュが存在すればメモリを解放
                if self.cached_data is not None:
                    del self.cached_data
                    gc.collect() # ガベージコレクションを促す
                with np.load(file_info['path'], allow_pickle=False) as data:
                    # 必要な配列をすべてメモリに読み込む
                    self.cached_data = {
                        self.sequences_field: data[self.sequences_field],
                        self.static_field: data[self.static_field],
                        self.labels_field: data[self.labels_field]
                    }
                self.cache_file_idx = file_idx
            # else:
                # logging.debug(f"Cache hit for idx {idx}. Using file {file_idx}") # オプションのデバッグログ
            # ==========================

            # キャッシュされたデータから特定のサンプルを抽出
            seq = self.cached_data[self.sequences_field][offset].astype(np.float32)
            static = self.cached_data[self.static_field][offset].astype(np.float32)
            label = self.cached_data[self.labels_field][offset].astype(np.int64)

            # パディングマスクを生成 (True の場所がパディング)
            padding_mask = (seq[:, 0] == self.padding_code) # 最初の特徴量がパディングを示すと仮定

            # 基本的な形状チェック（初期化された次元と一致するはず）
            if seq.shape != (self.seq_len, self.event_dim) or static.shape != (self.static_dim,):
                logging.error(f"Shape mismatch for sample {idx} from {file_info['path']}! Seq: {seq.shape}, Static: {static.shape}. Returning zeros.")
                # バッチの照合をクラッシュさせないためのフォールバック
                seq = np.zeros((self.seq_len, self.event_dim), dtype=np.float32)
                static = np.zeros((self.static_dim,), dtype=np.float32)
                label = np.zeros((), dtype=np.int64) # ラベル0を返す？
                padding_mask = np.ones((self.seq_len,), dtype=bool) # すべてマスク

            return seq, static, label, padding_mask

        except Exception as e:
            # 特定のサンプルで致命的なエラーが発生した場合
            logging.error(f"CRITICAL Error loading sample {idx} (offset {offset}) from file {file_idx} ({file_info['path']}): {e}", exc_info=True)
            # トレーニングループをクラッシュさせないためにゼロデータを返す
            return np.zeros((self.seq_len, self.event_dim), dtype=np.float32), \
                   np.zeros((self.static_dim,), dtype=np.float32), \
                   np.zeros((), dtype=np.int64), \
                   np.ones((self.seq_len,), dtype=bool) # すべてマスク

# --- Loss function (変更なし) ---
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0, num_classes=NUM_TILE_TYPES):
        super().__init__(); self.smoothing=smoothing; self.num_classes=num_classes; self.confidence=1.0-smoothing; self.criterion=nn.KLDivLoss(reduction='batchmean')
    def forward(self, pred, target):
        pred = torch.log_softmax(pred, dim=-1);
        with torch.no_grad(): true_dist = torch.zeros_like(pred); true_dist.fill_(self.smoothing / (self.num_classes - 1)); true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return self.criterion(pred, true_dist)

# --- Metrics (変更なし) ---
def calculate_accuracy(predictions, targets):
    with torch.no_grad(): _, predicted = torch.max(predictions, 1); accuracy = (predicted == targets).float().mean().item(); _, top3_indices = torch.topk(predictions, 3, dim=1); top3_correct = torch.any(top3_indices == targets.unsqueeze(1), dim=1).float().mean().item()
    return accuracy, top3_correct

# --- Plotting (変更なし) ---
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

# --- Training Loop (修正箇所あり) ---
def train_model():
    npz_files = sorted(glob.glob(DATA_PATTERN))
    if not npz_files: logging.error(f"No NPZ files found: {DATA_PATTERN}"); return
    logging.info(f"Found {len(npz_files)} NPZ files")

    full_dataset = MahjongNpzDataset(npz_files)
    # --- データセットの次元を確認 ---
    if full_dataset.event_dim <= 0 or full_dataset.static_dim <= 0:
        logging.error("Failed to determine feature dimensions from dataset files.")
        return
    event_dim = full_dataset.event_dim; static_dim = full_dataset.static_dim
    logging.info(f"Dataset dimensions determined: Event={event_dim}, Static={static_dim}, SeqLen={full_dataset.seq_len}")
    # --------------------------------

    val_size = int(len(full_dataset) * VALIDATION_SPLIT); train_size = len(full_dataset) - val_size
    # VALIDATION_SPLITが小さく、データセットも小さい場合に val_size が0にならないようにする
    if val_size == 0 and len(full_dataset) > 0 and VALIDATION_SPLIT > 0:
        val_size = 1
        train_size = len(full_dataset) - 1
        logging.warning(f"Validation split resulted in 0 samples, using 1 validation sample.")
    if train_size <= 0:
         logging.error("No training samples after split. Check dataset size and validation split.")
         return

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
    logging.info(f"Train samples: {train_size}, Validation samples: {val_size}")

    # --- num_workers をシステムとバッチサイズに基づいて調整 ---
    # I/Oがまだ制限要因であるか、RAMが限られている場合、少ないワーカーの方が良い場合がある
    # まずは中程度の数から始める
    num_workers = min(8, os.cpu_count() // 2 if os.cpu_count() else 4)
    # 明示的に意図しない限り、num_workersが0にならないようにする
    if num_workers <= 0: num_workers = 1
    logging.info(f"Using {num_workers} DataLoader workers.")
    # -------------------------------------------------------

    # num_workers > 0 の場合、オーバーヘッド削減のために persistent_workers=True を検討
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=(DEVICE.type=='cuda'), prefetch_factor=2 if num_workers > 0 else None, persistent_workers=(num_workers > 0), drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=num_workers, pin_memory=(DEVICE.type=='cuda'), prefetch_factor=2 if num_workers > 0 else None, persistent_workers=(num_workers > 0))

    # --- Model, Optimizer, Scheduler, Loss (変更なし、ただしSchedulerのT_0は調整) ---
    model = MahjongTransformerV2(event_feature_dim=event_dim, static_feature_dim=static_dim).to(DEVICE)
    logging.info(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    ema = EMA(model, decay=EMA_DECAY) if USE_EMA else None
    criterion = LabelSmoothingLoss(smoothing=LABEL_SMOOTHING) if LABEL_SMOOTHING > 0 else nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98), eps=1e-6)
    # エポックあたりのステップ数に基づいて T_0 を調整？ 例: T_0 = len(train_loader)
    steps_per_epoch = len(train_loader) // ACCUMULATION_STEPS
    # CosineAnnealingWarmRestartsのT_0をエポック数の1/5またはステップ数に基づいて調整
    # eta_minも調整 (LEARNING_RATE / 100 など)
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, steps_per_epoch * (NUM_EPOCHS // 5)), T_mult=1, eta_min=LEARNING_RATE / 100)
    scaler = GradScaler(enabled=USE_AMP and DEVICE.type == 'cuda')
    # --- End Initialization ---

    metrics = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_top3': [], 'val_top3': [], 'lr': []}
    best_val_acc = 0.0; best_epoch = 0; epochs_without_improvement = 0
    if INTERACTIVE_PLOT: plt.ion()
    fig, axs = init_plots()

    logging.info(f"Starting training for {NUM_EPOCHS} epochs on {DEVICE}...")
    for epoch in range(NUM_EPOCHS):
        model.train(); train_loss = 0.0; train_acc = 0.0; train_top3 = 0.0; train_steps = 0; start_time = time.time()
        optimizer.zero_grad(set_to_none=True) # より効率的なゼロ化

        # train_loaderをtqdmでラップ
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
        for i, batch in pbar:
            try:
                 # バッチのアンパックとデバイスへの移動
                 seq, static, labels, padding_mask = batch
                 seq, static, labels, padding_mask = seq.to(DEVICE, non_blocking=True), static.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True), padding_mask.to(DEVICE, non_blocking=True)
            except Exception as e:
                 logging.error(f"Error unpacking or moving batch {i} to device: {e}")
                 # データロードが致命的に失敗した場合、このバッチをスキップ
                 # 詳細については __getitem__ のエラーログを確認
                 continue

            # 入力形状チェック (オプション、デバッグ用)
            # if i == 0:
            #     logging.debug(f"Batch shapes - Seq: {seq.shape}, Static: {static.shape}, Labels: {labels.shape}, Mask: {padding_mask.shape}")

            with autocast(device_type=DEVICE.type, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, enabled=scaler.is_enabled()):
                outputs = model(seq, static, padding_mask)
                loss = criterion(outputs, labels)
                if ACCUMULATION_STEPS > 1: loss = loss / ACCUMULATION_STEPS

            # NaN損失のチェック
            if torch.isnan(loss):
                 logging.error(f"NaN loss detected at epoch {epoch+1}, batch {i}. Skipping batch.")
                 # ステップをスキップしてもスケーラーを更新する必要がある
                 scaler.update()
                 optimizer.zero_grad(set_to_none=True)
                 continue

            scaler.scale(loss).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                if CLIP_GRAD_NORM > 0:
                    scaler.unscale_(optimizer) # クリッピング前にアンスケール
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                # スケジューラは通常、エポックごとまたは固定ステップごとにステップする
                # lr_scheduler.step() # <<< 内部ループの外に移動

                if ema: ema.update()

            # メトリクス計算前にdetach()を確認
            with torch.no_grad():
                acc, acc_top3 = calculate_accuracy(outputs.detach(), labels.detach())
                # 累積除算前の実際の損失を取得
                current_loss = loss.item() * ACCUMULATION_STEPS

            train_loss += current_loss; train_acc += acc; train_top3 += acc_top3; train_steps += 1
            # tqdmプログレスバーを更新
            pbar.set_postfix({'Loss': f'{current_loss:.4f}', 'Acc': f'{acc:.3f}'})


        # --- Training Epoch 終了 ---
        # トレーニングステップが実行されたか確認
        if train_steps == 0:
             logging.warning(f"Epoch {epoch+1} completed with 0 training steps. Check data loading.")
             # ゼロ除算を避ける
             epoch_train_loss = 0.0; epoch_train_acc = 0.0; epoch_train_acc_top3 = 0.0
        else:
             epoch_train_loss = train_loss / train_steps
             epoch_train_acc = train_acc / train_steps
             epoch_train_acc_top3 = train_acc_top3 / train_steps

        metrics['train_loss'].append(epoch_train_loss); metrics['train_acc'].append(epoch_train_acc); metrics['train_top3'].append(epoch_train_acc_top3)
        current_lr = optimizer.param_groups[0]['lr']; metrics['lr'].append(current_lr)

        # --- Validation ---
        model.eval(); val_loss = 0.0; val_acc = 0.0; val_top3 = 0.0; val_batches = 0
        if ema: ema.apply_shadow()
        # val_loaderをtqdmでラップ
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                try:
                    # バリデーションバッチのアンパックと移動
                    seq, static, labels, padding_mask = batch
                    seq, static, labels, padding_mask = seq.to(DEVICE, non_blocking=True), static.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True), padding_mask.to(DEVICE, non_blocking=True)
                except Exception as e:
                    logging.error(f"Error unpacking or moving validation batch to device: {e}")
                    continue

                with autocast(device_type=DEVICE.type, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, enabled=scaler.is_enabled()): # バリデーションでもAMPを使用
                    outputs = model(seq, static, padding_mask)
                    loss = criterion(outputs, labels)

                # バリデーション中のNaN損失チェック
                if torch.isnan(loss):
                    logging.error(f"NaN loss detected during validation epoch {epoch+1}. Skipping batch.")
                    continue

                acc, acc_top3 = calculate_accuracy(outputs, labels)
                val_loss += loss.item(); val_acc += acc; val_top3 += acc_top3; val_batches += 1
                # tqdmプログレスバーを更新
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{acc:.3f}'})

        # バリデーションステップが実行されたか確認
        if val_batches == 0:
             logging.warning(f"Epoch {epoch+1} completed with 0 validation steps. Check data loading.")
             epoch_val_loss = 0.0; epoch_val_acc = 0.0; epoch_val_acc_top3 = 0.0
        else:
             epoch_val_loss = val_loss / val_batches
             epoch_val_acc = val_acc / val_batches
             epoch_val_acc_top3 = val_top3 / val_batches

        metrics['val_loss'].append(epoch_val_loss); metrics['val_acc'].append(epoch_val_acc); metrics['val_top3'].append(epoch_val_acc_top3)
        if ema: ema.restore()
        # --- End Validation ---

        epoch_time = time.time() - start_time
        logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS} - Time: {epoch_time:.1f}s - Train Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.4f} - Top3: {epoch_train_acc_top3:.4f} | Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc:.4f} - Top3: {epoch_val_acc_top3:.4f} | LR: {current_lr:.6f}")

        # バリデーションフェーズの後にスケジューラをステップ
        lr_scheduler.step()

        # プロットを更新
        if (epoch + 1) % PLOT_EVERY_EPOCH == 0 or epoch == NUM_EPOCHS - 1: update_plots(fig, axs, epoch, metrics)

        # チェックポイントを保存
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
        save_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_acc': epoch_val_acc,
            'event_dim': event_dim, # 再ロード用に次元を保存
            'static_dim': static_dim
        }
        if ema: save_dict['ema_state_dict'] = ema.shadow
        torch.save(save_dict, checkpoint_path)

        # バリデーション精度に基づいて最良モデルを保存
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc; best_epoch = epoch + 1; epochs_without_improvement = 0
            save_model_state = ema.shadow if ema else model.state_dict()
            # 最良モデルの状態と一緒に次元も保存
            best_model_save_dict = {
                 'model_state_dict': save_model_state,
                 'event_dim': event_dim,
                 'static_dim': static_dim
            }
            torch.save(best_model_save_dict, MODEL_SAVE_PATH);
            logging.info(f"*** New best model saved at epoch {best_epoch} with Val Acc: {best_val_acc:.4f} ***")
        else:
            epochs_without_improvement += 1
            # アーリーストッピング
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                logging.info(f"Validation accuracy did not improve for {EARLY_STOPPING_PATIENCE} epochs. Early stopping at epoch {epoch+1}.")
                break

    # 最終保存とクリーンアップ
    logging.info("Saving final plots...")
    update_plots(fig, axs, epoch, metrics); # 最終エポックのデータで更新
    plt.figure(fig); # 正しいフィギュアがアクティブであることを確認
    plt.savefig(os.path.join(PLOT_DIR, 'final_training_curves_v119_large.png'))
    if INTERACTIVE_PLOT: plt.ioff(); plt.show()
    else: plt.close(fig)
    logging.info(f"Training completed. Best model from epoch {best_epoch} (Val Acc: {best_val_acc:.4f}) saved to: {MODEL_SAVE_PATH}")
    logging.info(f"Checkpoints saved in: {CHECKPOINT_DIR}")
    logging.info(f"Logs saved in: {LOG_DIR}")
    logging.info(f"Plots saved in: {PLOT_DIR}")


if __name__ == "__main__":
    train_model()