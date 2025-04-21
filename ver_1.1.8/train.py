# train.py (Advanced Transformer architecture for mahjong AI)
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
from torch.amp import autocast, GradScaler
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# --- Ensure game_state can be imported for constants ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path: sys.path.insert(0, script_dir)
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path: sys.path.insert(0, parent_dir)
try:
    from game_state import NUM_TILE_TYPES, MAX_EVENT_HISTORY, EVENT_TYPES
except ImportError as e:
     print(f"[FATAL ERROR in train.py] Cannot import constants from game_state.py: {e}")
     # Define fallbacks if game_state import fails, but this indicates a problem
     NUM_TILE_TYPES = 34
     MAX_EVENT_HISTORY = 60 # Make sure this matches data generation
     EVENT_TYPES = {"PADDING": 8} # Need at least PADDING code
# -----------------------------------------------------

# --- Configuration ---
DATA_DIR = "./training_data/" # Directory containing the NPZ batch files
DATA_PATTERN = os.path.join(DATA_DIR, "mahjong_imitation_data_batch_*.npz")
MODEL_SAVE_PATH = "./trained_model/mahjong_transformer_v2.pth"
CHECKPOINT_DIR = "./checkpoints/"
LOG_DIR = "./logs"
PLOT_DIR = "./plots"  # Directory to save plots during training
# -------------------------------------------

# トレーニングハイパーパラメータ ― スピード重視で最適化
BATCH_SIZE = 1024        # バッチサイズ（さらに大きく設定）
NUM_EPOCHS = 50          # エポック数（最大50回まで学習）
LEARNING_RATE = 2e-4     # 学習率（高めに設定）
VALIDATION_SPLIT = 0.05  # 検証データの割合（5%のみ使用）
WEIGHT_DECAY = 0.01      # AdamW の L2 正則化係数
CLIP_GRAD_NORM = 1.0     # 勾配クリッピング閾値
WARMUP_STEPS = 1000      # ウォームアップステップ数（少なめ）

# データセットサイズの上限（-1 で全サンプルを使用）
MAX_SAMPLES = 500000     # サンプル数を絞って高速化

# サンプリング設定 ― 学習ファイルを一部だけ使用
MAX_FILES_PERCENT = 0.5  # 全ファイルのうち 50% を使用

# Transformer モデルハイパーパラメータ ― スピード重視で小型化
D_MODEL = 128            # 埋め込み次元（小さめ）
NHEAD = 4                # アテンションヘッド数（少なめ）
D_HID = 512              # FFN 中間層の次元（小さめ）
NLAYERS = 2              # Transformer 層数（2層のみ）
DROPOUT = 0.1            # ドロップアウト率
ACTIVATION = 'relu'      # 活性化関数（高速な ReLU を使用）

# 高度なトレーニング機能
USE_AMP = True              # 自動混合精度（AMP）
USE_EMA = False             # EMA（指数移動平均）を無効化
EMA_DECAY = 0.999           # EMA 減衰率
LABEL_SMOOTHING = 0.0       # ラベル平滑化を無効化
USE_WEIGHT_AVERAGING = False# 重み平均化を無効化


# Plot settings
PLOT_EVERY_EPOCH = 2        # Update plots less frequently
INTERACTIVE_PLOT = False    # Don't show interactive plots

# Early stopping settings - Disabled
EARLY_STOPPING_PATIENCE = 999  # Effectively disable early stopping

# --- Device Configuration ---
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
     DEVICE = torch.device("mps")
elif torch.cuda.is_available():
     DEVICE = torch.device("cuda")
else:
     DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# --------------------------

# --- Create necessary directories ---
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)  # Create plots directory

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "model_training.log"), mode='w'),
        logging.StreamHandler()
    ]
)
# ---------------------

# --- Advanced Positional Encoding ---
class RotaryPositionalEncoding(nn.Module):
    """Rotary positional encoding for better performance than standard sin/cos"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        # Only need frequencies for half the dimensions
        self.dim_half = d_model // 2
        # Generate rotation frequencies for half the dimensions
        freqs = 1.0 / (10000 ** (torch.arange(0, self.dim_half).float() / self.dim_half))
        self.register_buffer('freqs', freqs)
        pos_seq = torch.arange(max_len).float()
        self.register_buffer('pos_seq', pos_seq)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        positions = self.pos_seq[:seq_len].unsqueeze(0)  # [1, seq_len]
        
        # Create rotation angles for half dimensions
        # [1, seq_len, dim_half]
        angles = positions.unsqueeze(-1) * self.freqs.unsqueeze(0)
        sin = torch.sin(angles)  # [1, seq_len, dim_half]
        cos = torch.cos(angles)  # [1, seq_len, dim_half]
        
        # Reshape x to separate even and odd dimensions
        x_even = x[:, :, 0::2]  # [batch_size, seq_len, dim_half]
        x_odd = x[:, :, 1::2]   # [batch_size, seq_len, dim_half]
        
        # Apply rotary encoding to each half
        x_even_rotated = x_even * cos - x_odd * sin
        x_odd_rotated = x_odd * cos + x_even * sin
        
        # Interleave the rotated dimensions back
        x_rotated = torch.zeros_like(x)
        x_rotated[:, :, 0::2] = x_even_rotated
        x_rotated[:, :, 1::2] = x_odd_rotated
        
        return x_rotated

# --- Advanced Transformer Model Definition ---
class MahjongTransformerV2(nn.Module):
    """Advanced Transformer model with rotary position encoding and multi-head attention"""
    def __init__(self, event_feature_dim, static_feature_dim,
                 d_model=D_MODEL, nhead=NHEAD, d_hid=D_HID,
                 nlayers=NLAYERS, dropout=DROPOUT, activation=ACTIVATION,
                 output_dim=NUM_TILE_TYPES):
        super().__init__()
        
        # Input embeddings with layer norm
        self.event_encoder = nn.Sequential(
            nn.Linear(event_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Advanced positional encoding
        self.pos_encoder = RotaryPositionalEncoding(d_model)
        
        # Transformer with pre-norm architecture
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        # Static features encoder with residual connections
        self.static_encoder = nn.Sequential(
            nn.Linear(static_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Attention pooling for sequence output
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # Multi-head output projection
        self.output_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, output_dim)
        )
        
        # Initialize weights with improved scheme
        self._init_weights()
        
    def _init_weights(self):
        """Custom Xavier initialization with gain for better convergence"""
        for name, p in self.named_parameters():
            if p.dim() > 1:  # Weight matrices
                if 'output_head' in name:
                    # Use smaller initialization for final layers
                    nn.init.xavier_normal_(p, gain=0.7)
                else:
                    nn.init.xavier_normal_(p, gain=1.0)
            else:  # Bias terms
                nn.init.zeros_(p)
    
    def forward(self, event_seq, static_feat, attention_mask=None):
        # Process event sequence
        event_encoded = self.event_encoder(event_seq)
        pos_encoded = self.pos_encoder(event_encoded)
        
        # Create padding mask for transformer
        if attention_mask is not None:
            # Convert boolean mask to float with -inf for masked positions
            attn_mask = attention_mask.float().masked_fill(attention_mask, float('-inf'))
        else:
            attn_mask = None
        
        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(
            pos_encoded,
            src_key_padding_mask=attention_mask
        )
        
        # Attention pooling of transformer output
        attn_weights = self.attention_pool(transformer_output)
        if attention_mask is not None:
            # Zero out attention on padding tokens
            attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(-1), 0)
        context_vector = torch.sum(attn_weights * transformer_output, dim=1)
        
        # Process static features and combine
        static_encoded = self.static_encoder(static_feat)
        combined = torch.cat([context_vector, static_encoded], dim=1)
        
        # Output logits
        return self.output_head(combined)

# --- Exponential Moving Average ---
class EMA:
    """Maintains moving averages of model parameters with decay"""
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# --- Dataset with padding masks ---
class MahjongNpzDataset(Dataset):
    """Dataset to load data from multiple NPZ files with padding mask generation"""
    def __init__(self, npz_files):
        """Initialize dataset with optimized loading."""
        self.npz_files = npz_files
        
        # Limit to 50% of files for much faster training
        total_files = len(self.npz_files)
        max_files = int(total_files * MAX_FILES_PERCENT)
        logging.info(f"Limiting dataset to {max_files} files ({MAX_FILES_PERCENT*100:.0f}% of {total_files}) for faster training")
        
        # Use random sampling instead of just taking the first files
        if max_files < total_files:
            file_indices = np.random.choice(total_files, size=max_files, replace=False)
            self.npz_files = [self.npz_files[i] for i in file_indices]
        
        self.file_metadata = []
        self.cumulative_lengths = [0]
        self.total_length = 0
        self.seq_len = -1
        self.event_dim = -1
        self.static_dim = -1
        self.padding_code = float(EVENT_TYPES["PADDING"])
        
        # Field names mapping - will be determined from first file
        self.sequences_field = 'sequences'  # Default field name
        self.static_field = 'static_features'  # Default field name
        self.labels_field = 'labels'  # Default field name

        logging.info("Scanning NPZ files for metadata...")
        first_file = True
        
        for f in tqdm(self.npz_files, desc="Scanning files"):
            try:
                with np.load(f) as data:
                    # For the first file, detect available fields
                    if first_file:
                        available_fields = data.files
                        logging.info(f"Available fields in NPZ files: {available_fields}")
                        
                        # Check and map field names
                        if 'sequences' not in available_fields and 'sequence' in available_fields:
                            self.sequences_field = 'sequence'
                        
                        # Try several possible names for static features
                        if 'static_features' not in available_fields:
                            if 'static' in available_fields:
                                self.static_field = 'static'
                                logging.info(f"Using 'static' instead of 'static_features'")
                            elif 'features' in available_fields:
                                self.static_field = 'features'
                                logging.info(f"Using 'features' instead of 'static_features'")
                        
                        # Try different label field names
                        if 'labels' not in available_fields and 'label' in available_fields:
                            self.labels_field = 'label'
                        
                        logging.info(f"Using fields: {self.sequences_field}, {self.static_field}, {self.labels_field}")
                        first_file = False

                    # Check if required fields exist
                    if self.sequences_field not in data or self.labels_field not in data:
                        missing_fields = []
                        if self.sequences_field not in data:
                            missing_fields.append(self.sequences_field)
                        if self.labels_field not in data:
                            missing_fields.append(self.labels_field)
                        logging.warning(f"Skipping file {f} - missing required fields: {missing_fields}")
                        continue
                    
                    # Handle the case where static features might not exist
                    if self.static_field not in data:
                        logging.warning(f"File {f} doesn't have static features. Using dummy values.")
                    
                    length = len(data[self.labels_field])
                    if length == 0:
                        logging.warning(f"Skipping empty file: {f}")
                        continue
                    
                    self.file_metadata.append({'path': f, 'length': length})
                    self.total_length += length
                    self.cumulative_lengths.append(self.total_length)
                    
                    # Get dimensions from the first valid file
                    if self.seq_len == -1:
                        self.seq_len = data[self.sequences_field].shape[1]
                        self.event_dim = data[self.sequences_field].shape[2]
                        
                        # Get static dim if it exists, otherwise use dummy value
                        if self.static_field in data:
                            self.static_dim = data[self.static_field].shape[1]
                        else:
                            # Use a small dummy dimension if static features aren't available
                            self.static_dim = 1
                            logging.warning(f"Using dummy static dimension: {self.static_dim}")
            except Exception as e:
                logging.error(f"Error reading metadata from {f}: {e}")

        if self.total_length == 0:
            raise RuntimeError("No valid data found in any NPZ files.")

        logging.info(f"Dataset initialized: {len(self.file_metadata)} files, {self.total_length} samples.")
        logging.info(f"Dims: SeqLen={self.seq_len}, Event={self.event_dim}, Static={self.static_dim}")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_length:
            raise IndexError("Index out of bounds")
            
        # Find which file contains the index
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        file_info = self.file_metadata[file_idx]
        # Calculate offset within the file
        offset = idx - self.cumulative_lengths[file_idx]
        
        # Load the specific sample
        try:
            with np.load(file_info['path']) as data:
                # Get sequences (required)
                seq = data[self.sequences_field][offset].astype(np.float32)
                
                # Get static features if available, otherwise use dummy values
                if self.static_field in data:
                    static = data[self.static_field][offset].astype(np.float32)
                else:
                    # Create dummy static features
                    static = np.zeros((self.static_dim,), dtype=np.float32)
                
                # Get labels (required)
                label = data[self.labels_field][offset].astype(np.int64)
                
                # Generate attention mask
                # Identify padding tokens in first feature dimension
                padding_mask = (seq[:, 0] == self.padding_code)
                
                # After loading the data, truncate sequences to make training faster
                MAX_SEQ_LENGTH = 30  # Shorter sequences for faster training
                if seq.shape[0] > MAX_SEQ_LENGTH:
                    # Keep only the last MAX_SEQ_LENGTH events
                    seq = seq[-MAX_SEQ_LENGTH:, :]
                    padding_mask = padding_mask[-MAX_SEQ_LENGTH:]
                
                return seq, static, label, padding_mask
        except Exception as e:
            logging.error(f"Error loading sample {idx} from {file_info['path']}: {e}")
            # Return zeros as fallback (this shouldn't be needed in production)
            dummy_seq = np.zeros((self.seq_len, self.event_dim), dtype=np.float32)
            dummy_static = np.zeros((self.static_dim,), dtype=np.float32)
            dummy_label = np.zeros((), dtype=np.int64)
            dummy_mask = np.ones((self.seq_len,), dtype=bool)
            return dummy_seq, dummy_static, dummy_label, dummy_mask

# --- Loss function with label smoothing ---
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0, num_classes=NUM_TILE_TYPES):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, pred, target):
        pred = torch.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return self.criterion(pred, true_dist)

# --- Metrics calculation ---
def calculate_accuracy(predictions, targets):
    """Calculate accuracy and top-k accuracy"""
    _, predicted = torch.max(predictions, 1)
    accuracy = (predicted == targets).float().mean().item()
    
    # Calculate top-3 accuracy
    _, top3_indices = torch.topk(predictions, 3, dim=1)
    top3_correct = torch.any(top3_indices == targets.unsqueeze(1), dim=1).float().mean().item()
    
    return accuracy, top3_correct

# --- Plotting Functions ---
def init_plots():
    """Initialize the plots for training metrics."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot (top left)
    axs[0, 0].set_title('Training and Validation Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    
    # Accuracy plot (top right)
    axs[0, 1].set_title('Training and Validation Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    
    # Top-3 Accuracy plot (bottom left)
    axs[1, 0].set_title('Training and Validation Top-3 Accuracy')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Top-3 Accuracy')
    
    # Learning Rate plot (bottom right)
    axs[1, 1].set_title('Learning Rate')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Learning Rate')
    
    plt.tight_layout()
    return fig, axs

def update_plots(fig, axs, epoch, train_losses, val_losses, train_accs, val_accs, 
                train_acc_top3s, val_acc_top3s, learning_rates):
    """Update the plots with current metrics."""
    epochs = list(range(1, epoch + 2))
    
    # Clear existing plots
    for ax in axs.flatten():
        ax.clear()
    
    # Loss plot (top left)
    axs[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axs[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axs[0, 0].set_title('Training and Validation Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Accuracy plot (top right)
    axs[0, 1].plot(epochs, train_accs, 'b-', label='Train Accuracy')
    axs[0, 1].plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    axs[0, 1].set_title('Training and Validation Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Top-3 Accuracy plot (bottom left)
    axs[1, 0].plot(epochs, train_acc_top3s, 'b-', label='Train Top-3 Accuracy')
    axs[1, 0].plot(epochs, val_acc_top3s, 'r-', label='Validation Top-3 Accuracy') 
    axs[1, 0].set_title('Training and Validation Top-3 Accuracy')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Top-3 Accuracy')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Learning Rate plot (bottom right)
    axs[1, 1].plot(epochs, learning_rates, 'g-')
    axs[1, 1].set_title('Learning Rate')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Learning Rate')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(PLOT_DIR, f'training_metrics_epoch_{epoch+1}.png')
    fig.savefig(plot_path)
    
    # Save the latest plot as a fixed name for easy access
    latest_path = os.path.join(PLOT_DIR, 'latest_training_metrics.png')
    fig.savefig(latest_path)
    
    # Display if interactive plotting is enabled
    if INTERACTIVE_PLOT:
        plt.pause(0.1)  # Small pause to update the plot window

# --- Training loop ---
def train_model():
    # Find all NPZ files
    npz_files = sorted(glob.glob(DATA_PATTERN))
    if not npz_files:
        logging.error(f"No NPZ files found at {DATA_PATTERN}")
        return
    
    logging.info(f"Found {len(npz_files)} NPZ files")
    
    # Create dataset and split into train/validation
    full_dataset = MahjongNpzDataset(npz_files)
    
    # Store dimensions from the original dataset before subsetting
    event_dim = full_dataset.event_dim
    static_dim = full_dataset.static_dim
    
    # Limit overall dataset size for faster training
    limited_dataset = full_dataset
    if MAX_SAMPLES > 0 and len(full_dataset) > MAX_SAMPLES:
        logging.info(f"Limiting dataset to {MAX_SAMPLES} samples for faster training")
        indices = np.random.choice(len(full_dataset), size=MAX_SAMPLES, replace=False)
        limited_dataset = Subset(full_dataset, indices)
    
    # Calculate split sizes
    val_size = int(len(limited_dataset) * VALIDATION_SPLIT)
    train_size = len(limited_dataset) - val_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(
        limited_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    # Determine optimal number of workers based on CPU count
    num_workers = min(12, os.cpu_count() or 4)  # More workers
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=3,           # More prefetching
        persistent_workers=True,
        drop_last=True               # Drop incomplete batches for speed
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True,
        drop_last=True
    )
    
    # Initialize model
    model = MahjongTransformerV2(
        event_feature_dim=event_dim,
        static_feature_dim=static_dim
    )
    model.to(DEVICE)
    
    # Initialize EMA if enabled
    ema = EMA(model, decay=EMA_DECAY) if USE_EMA else None
    
    # Initialize loss function
    criterion = LabelSmoothingLoss(smoothing=LABEL_SMOOTHING) if LABEL_SMOOTHING > 0 else nn.CrossEntropyLoss()
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart period
        T_mult=2,  # Multiply period by this factor after each restart
        eta_min=LEARNING_RATE / 10  # Minimum learning rate
    )
    
    # Initialize gradient scaler for AMP
    scaler = GradScaler() if USE_AMP else None
    
    # Initialize metrics tracking
    best_val_acc = 0.0
    best_epoch = 0
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    train_acc_top3s = []
    val_acc_top3s = []
    learning_rates = []
    epochs_without_improvement = 0
    
    # Initialize plots
    if INTERACTIVE_PLOT:
        plt.ion()  # Turn on interactive mode
    fig, axs = init_plots()
    
    logging.info(f"Starting training with {train_size:,} training samples, {val_size:,} validation samples")
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_acc_top3 = 0.0
        train_batches = 0
        
        start_time = time.time()
        
        for seq, static, labels, padding_mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            # Move data to device
            seq = seq.to(DEVICE)
            static = static.to(DEVICE)
            labels = labels.to(DEVICE)
            padding_mask = padding_mask.to(DEVICE)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with AMP if enabled
            if scaler is not None:
                with autocast(device_type=DEVICE.type):
                    outputs = model(seq, static, padding_mask)
                    loss = criterion(outputs, labels)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                if CLIP_GRAD_NORM > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward/backward without AMP
                outputs = model(seq, static, padding_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                
                if CLIP_GRAD_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                
                optimizer.step()
            
            # Update EMA if enabled
            if ema is not None:
                ema.update()
            
            # Calculate metrics
            acc, acc_top3 = calculate_accuracy(outputs, labels)
            
            # Update running metrics
            train_loss += loss.item()
            train_acc += acc
            train_acc_top3 += acc_top3
            train_batches += 1
        
        # Update learning rate
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / train_batches
        epoch_train_acc = train_acc / train_batches
        epoch_train_acc_top3 = train_acc_top3 / train_batches
        
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        train_acc_top3s.append(epoch_train_acc_top3)
        
        # Validation phase
        model.eval()
        if ema is not None:
            ema.apply_shadow()  # Use EMA weights for validation
            
        val_loss = 0.0
        val_acc = 0.0
        val_acc_top3 = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for seq, static, labels, padding_mask in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                # Move data to device
                seq = seq.to(DEVICE)
                static = static.to(DEVICE)
                labels = labels.to(DEVICE)
                padding_mask = padding_mask.to(DEVICE)
                
                # Forward pass
                outputs = model(seq, static, padding_mask)
                loss = criterion(outputs, labels)
                
                # Calculate metrics
                acc, acc_top3 = calculate_accuracy(outputs, labels)
                
                # Update running metrics
                val_loss += loss.item()
                val_acc += acc
                val_acc_top3 += acc_top3
                val_batches += 1
        
        # Calculate epoch metrics
        epoch_val_loss = val_loss / val_batches
        epoch_val_acc = val_acc / val_batches
        epoch_val_acc_top3 = val_acc_top3 / val_batches
        
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        val_acc_top3s.append(epoch_val_acc_top3)
        
        # Revert to training weights if using EMA
        if ema is not None:
            ema.restore()
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Log metrics
        logging.info(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - "
            f"Time: {epoch_time:.1f}s - "
            f"Train Loss: {epoch_train_loss:.4f} - "
            f"Train Acc: {epoch_train_acc:.4f} - "
            f"Train Top-3: {epoch_train_acc_top3:.4f} - "
            f"Val Loss: {epoch_val_loss:.4f} - "
            f"Val Acc: {epoch_val_acc:.4f} - "
            f"Val Top-3: {epoch_val_acc_top3:.4f} - "
            f"LR: {current_lr:.6f}"
        )
        
        # Update plots at every PLOT_EVERY_EPOCH
        if (epoch + 1) % PLOT_EVERY_EPOCH == 0:
            update_plots(fig, axs, epoch, train_losses, val_losses, train_accs, val_accs, 
                      train_acc_top3s, val_acc_top3s, learning_rates)
        
        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'train_acc': epoch_train_acc,
            'val_acc': epoch_val_acc,
            'train_top3_acc': epoch_train_acc_top3,
            'val_top3_acc': epoch_val_acc_top3,
            'learning_rate': current_lr,
        }, checkpoint_path)
        
        # Check if this is the best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch + 1
            
            # Save best model
            if ema is not None:
                ema.apply_shadow()  # Use EMA weights for best model
                
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
            if ema is not None:
                ema.restore()
                
            logging.info(f"New best model saved with validation accuracy: {best_val_acc:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Remove early stopping - train for all epochs
        # if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        #     logging.info(f"Early stopping triggered after {epoch+1} epochs without improvement")
        #     break
    
    # Final plot update
    update_plots(fig, axs, epoch, train_losses, val_losses, train_accs, val_accs, 
               train_acc_top3s, val_acc_top3s, learning_rates)
    
    # Save final plots with both metrics
    plt.figure(figsize=(12, 10))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.plot(val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(2, 2, 2)
    plt.plot(train_accs, 'b-', label='Train Accuracy')
    plt.plot(val_accs, 'r-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    
    # Top-3 Accuracy plot
    plt.subplot(2, 2, 3)
    plt.plot(train_acc_top3s, 'b-', label='Train Top-3 Accuracy')
    plt.plot(val_acc_top3s, 'r-', label='Validation Top-3 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Top-3 Accuracy')
    plt.legend()
    plt.title('Training and Validation Top-3 Accuracy')
    plt.grid(True)
    
    # Learning rate plot
    plt.subplot(2, 2, 4)
    plt.plot(learning_rates, 'g-')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, 'final_training_curves.png'))
    
    if INTERACTIVE_PLOT:
        plt.ioff()  # Turn off interactive mode
        plt.show()  # Show the final plot
    
    logging.info(f"Training completed after {NUM_EPOCHS} epochs. Best model from epoch {best_epoch} with validation accuracy: {best_val_acc:.4f}")
    logging.info(f"Training plots saved to {PLOT_DIR} and {LOG_DIR}")
    return best_val_acc

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')  # Use TF32 for faster training on NVIDIA GPUs
    torch.backends.cudnn.benchmark = True       # Enable cudnn auto-tuner
    train_model()