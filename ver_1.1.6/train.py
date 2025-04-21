# train.py (Complete version with plotting)
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
import matplotlib.pyplot as plt # Import Matplotlib

# --- Ensure game_state can be imported for constants ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path: sys.path.insert(0, script_dir)
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path: sys.path.insert(0, parent_dir)
try:
    # Import constants needed from game_state
    from game_state import NUM_TILE_TYPES, MAX_EVENT_HISTORY, EVENT_TYPES
except ImportError as e:
    print(f"[FATAL ERROR in train.py] Cannot import constants from game_state.py: {e}")
    print("Ensure game_state.py is accessible.")
    # Define fallbacks if game_state import fails, but this indicates a setup problem
    NUM_TILE_TYPES = 34
    MAX_EVENT_HISTORY = 60 # !!! IMPORTANT: This MUST match the sequence length in your data !!!
    EVENT_TYPES = {"PADDING": 8} # Need at least PADDING code
    print("[Warning] Using fallback constants due to game_state import failure.")
# -----------------------------------------------------

# --- Configuration ---
# !!! ADJUST THESE PATHS FOR YOUR ENVIRONMENT !!!
DATA_DIR = "./training_data/" # Directory containing the NPZ batch files from model.py
DATA_PATTERN = os.path.join(DATA_DIR, "mahjong_transformer_data_batch_*.npz") # Pattern to find data files
MODEL_SAVE_PATH = "./trained_model/mahjong_transformer_imitation.pth" # Path to save the best model
PLOT_SAVE_PATH = "./trained_model/training_curves.png" # Path to save the training curves plot
# -------------------------------------------

# Training Hyperparameters (Adjust as needed)
BATCH_SIZE = 64      # Reduce if GPU memory is limited
NUM_EPOCHS = 100      # Increase for larger datasets or better convergence (was 100, reduced for initial testing)
LEARNING_RATE = 1e-4 # Learning rate for AdamW
VALIDATION_SPLIT = 0.1 # Use 10% of data for validation
WEIGHT_DECAY = 0.01   # Weight decay for AdamW
CLIP_GRAD_NORM = 1.0  # Max norm for gradient clipping

# Transformer Hyperparameters (Adjust based on resources and desired model capacity)
D_MODEL = 128          # Internal dimension of the Transformer
NHEAD = 4              # Number of attention heads (must divide D_MODEL evenly)
D_HID = 256            # Dimension of the feedforward network model in nn.TransformerEncoderLayer
NLAYERS = 2            # Number of nn.TransformerEncoderLayer layers
DROPOUT = 0.1          # Dropout probability

# --- Device Configuration ---
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")
# --------------------------

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("model_training.log", mode='w'), # Overwrite log each run
        logging.StreamHandler(sys.stdout) # Also print logs to console
    ]
)
# ---------------------


# --- Transformer Model Definition ---
class PositionalEncoding(nn.Module):
    """Positional encoding module (batch_first=True)."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000): # Increased default max_len
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model) # Add batch dimension directly
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: Tensor, shape [batch_size, seq_len, embedding_dim]"""
        # Add positional encoding up to the sequence length of the input
        # Ensure pe slicing doesn't go out of bounds if max_len is smaller than expected
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
             raise ValueError(f"Input sequence length ({seq_len}) exceeds PositionalEncoding max_len ({self.pe.size(1)})")
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)

class MahjongTransformerModel(nn.Module):
    """Transformer model for Mahjong discard prediction."""
    def __init__(self, event_feature_dim: int, static_feature_dim: int,
                 d_model: int = D_MODEL, nhead: int = NHEAD, d_hid: int = D_HID,
                 nlayers: int = NLAYERS, dropout: float = DROPOUT,
                 output_dim: int = NUM_TILE_TYPES, max_seq_len: int = MAX_EVENT_HISTORY):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"nhead ({nhead}) must divide d_model ({d_model}) evenly.")
        self.d_model = d_model
        # Input embedding/projection for event sequence
        self.event_encoder = nn.Linear(event_feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len) # Pass max_seq_len
        # Standard Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        # Encoder for static features (MLP)
        self.static_encoder = nn.Sequential(
            nn.Linear(static_feature_dim, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 2)
        )
        # Decoder to combine features and predict output probabilities
        self.decoder = nn.Sequential(
            nn.Linear(d_model + d_model // 2, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        self.init_weights() # Initialize weights

    def init_weights(self) -> None:
        initrange = 0.1
        # Initialize linear layers (example for event_encoder)
        self.event_encoder.weight.data.uniform_(-initrange, initrange)
        if self.event_encoder.bias is not None: self.event_encoder.bias.data.zero_()
        # Initialize other linear layers in static_encoder and decoder
        for layer in self.static_encoder:
             if isinstance(layer, nn.Linear):
                 layer.weight.data.uniform_(-initrange, initrange)
                 if layer.bias is not None: layer.bias.data.zero_()
        for layer in self.decoder:
             if isinstance(layer, nn.Linear):
                 layer.weight.data.uniform_(-initrange, initrange)
                 if layer.bias is not None: layer.bias.data.zero_()

    def forward(self, event_seq: torch.Tensor, static_feat: torch.Tensor, src_padding_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        # Process event sequence: Embed -> PosEncode -> Transformer
        # src_padding_mask should be True where padded
        embedded_seq = self.event_encoder(event_seq) * math.sqrt(self.d_model)
        pos_encoded_seq = self.pos_encoder(embedded_seq)
        # Transformer expects mask where True indicates masking
        transformer_output = self.transformer_encoder(pos_encoded_seq, src_key_padding_mask=src_padding_mask)

        # Pooling: Use mean pooling, ignoring padding
        # Invert mask for summing non-padded elements (True where not padded)
        non_padding_mask = ~src_padding_mask
        seq_len = non_padding_mask.sum(dim=1, keepdim=True).float()
        seq_len = torch.max(seq_len, torch.tensor(1.0, device=seq_len.device)) # Avoid division by zero
        # Mask out padded values (where src_padding_mask is True) before summing
        masked_output = transformer_output.masked_fill(src_padding_mask.unsqueeze(-1), 0.0)
        transformer_pooled = masked_output.sum(dim=1) / seq_len # [batch_size, d_model]

        # Process static features
        encoded_static = self.static_encoder(static_feat) # [batch_size, d_model // 2]

        # Combine and decode
        combined_features = torch.cat((transformer_pooled, encoded_static), dim=1)
        output_logits = self.decoder(combined_features) # [batch_size, output_dim]
        return output_logits
# --- End Model Definition ---


# --- Dataset Definition ---
class MahjongNpzDataset(Dataset):
    """Dataset to load data lazily from multiple NPZ files."""
    def __init__(self, npz_files):
        self.npz_files = npz_files
        self.file_metadata = [] # Stores {'path': filepath, 'length': length}
        self.cumulative_lengths = [0]
        self.total_length = 0
        self.seq_len = -1
        self.event_dim = -1
        self.static_dim = -1
        try:
            self.padding_code = float(EVENT_TYPES["PADDING"])
        except KeyError:
             logging.error("EVENT_TYPES['PADDING'] not found in game_state constants! Using default 8.")
             self.padding_code = 8.0 # Fallback padding code

        logging.info("Scanning NPZ files for metadata...")
        for f in tqdm(npz_files, desc="Scanning files"):
            try:
                # Use 'allow_pickle=False' for security if possible, but default should be fine
                with np.load(f) as data:
                    # Check if required keys exist
                    if not all(key in data for key in ['sequences', 'static_features', 'labels']):
                        logging.warning(f"Skipping file {f} due to missing keys.")
                        continue
                    length = len(data['labels'])
                    if length == 0:
                        logging.warning(f"Skipping empty file: {f}")
                        continue
                    self.file_metadata.append({'path': f, 'length': length})
                    self.total_length += length
                    self.cumulative_lengths.append(self.total_length)
                    # Get dimensions from the first valid file
                    if self.seq_len == -1:
                        seq_shape = data['sequences'].shape
                        static_shape = data['static_features'].shape
                        if len(seq_shape) != 3 or len(static_shape) != 2:
                            logging.warning(f"Skipping file {f} due to unexpected data dimensions: seq={seq_shape}, static={static_shape}")
                            # Remove the just added metadata and adjust length
                            self.total_length -= length
                            self.cumulative_lengths.pop()
                            self.file_metadata.pop()
                            continue
                        self.seq_len = seq_shape[1]
                        self.event_dim = seq_shape[2]
                        self.static_dim = static_shape[1]
                        # Check consistency with MAX_EVENT_HISTORY from game_state
                        if self.seq_len != MAX_EVENT_HISTORY:
                            logging.warning(f"Sequence length from data ({self.seq_len}) does not match MAX_EVENT_HISTORY ({MAX_EVENT_HISTORY}). Using {self.seq_len}.")
            except Exception as e:
                logging.error(f"Error reading metadata or data from {f}: {e}")

        if self.total_length == 0:
            raise RuntimeError("No valid data found in any NPZ files. Check NPZ file contents and paths.")
        if self.seq_len == -1:
            raise RuntimeError("Could not determine data dimensions from NPZ files.")

        logging.info(f"Dataset initialized: {len(self.file_metadata)} valid files, {self.total_length} samples.")
        logging.info(f"Dims: SeqLen={self.seq_len}, Event={self.event_dim}, Static={self.static_dim}")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_length: raise IndexError("Index out of bounds")
        # Find which file contains the index using cumulative lengths
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        local_idx = idx - self.cumulative_lengths[file_idx]
        metadata = self.file_metadata[file_idx]
        filepath = metadata['path']

        try:
            # Load data for the specific index from the identified file
            with np.load(filepath) as data:
                sequence = torch.tensor(data['sequences'][local_idx], dtype=torch.float32)
                static_features = torch.tensor(data['static_features'][local_idx], dtype=torch.float32)
                label = torch.tensor(data['labels'][local_idx], dtype=torch.long)

                # Create padding mask (True where padded)
                # Assumes the first feature in the sequence vector is the event type code
                src_padding_mask = (sequence[:, 0] == self.padding_code) # Shape: [seq_len]

            return sequence, static_features, label, src_padding_mask
        except Exception as e:
            logging.error(f"Error loading data at index {idx} (file: {filepath}, local_idx: {local_idx}): {e}")
            raise RuntimeError(f"Failed to load data for index {idx}") from e
# --- End Dataset Definition ---


# --- Main Training Function ---
def run_training():
    logging.info("--- Starting Training Process ---")
    # 1. Find Data Files
    npz_files = sorted(glob.glob(DATA_PATTERN))
    if not npz_files:
        logging.error(f"Error: No data files found matching pattern: {DATA_PATTERN}")
        return
    logging.info(f"Found {len(npz_files)} data files.")

    # 2. Create Dataset
    try:
        full_dataset = MahjongNpzDataset(npz_files)
        if full_dataset.total_length == 0: return # Exit if dataset is empty
        if full_dataset.seq_len <= 0 or full_dataset.event_dim <= 0 or full_dataset.static_dim <= 0:
            raise ValueError("Invalid data dimensions detected in dataset.")
    except (RuntimeError, ValueError) as e:
        logging.error(f"Failed to initialize dataset: {e}")
        return

    # Use actual sequence length from data for the model
    effective_max_seq_len = full_dataset.seq_len

    # 3. Split Data
    total_size = len(full_dataset)
    val_size = int(total_size * VALIDATION_SPLIT)
    train_size = total_size - val_size
    if train_size <= 0 or val_size <= 0:
        logging.error(f"Dataset too small for validation split (Total: {total_size}). Need at least {1/VALIDATION_SPLIT:.0f} samples.")
        return
    logging.info(f"Splitting data: {train_size} train, {val_size} validation.")
    # Use torch's random_split for splitting indices
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 4. Create DataLoaders
    # Determine num_workers based on platform and CPU count
    num_workers = min(4, os.cpu_count()) if DEVICE == torch.device("cpu") and os.cpu_count() else 0 # Use 0 for GPU/MPS or if cpu_count fails
    logging.info(f"Using {num_workers} workers for DataLoaders.")
    try:
        # Use persistent_workers=True if num_workers > 0 to speed up epoch starts
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers,
                                  pin_memory=DEVICE != torch.device("cpu"), persistent_workers=num_workers > 0, drop_last=True) # drop_last to smooth training stats
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers,
                                pin_memory=DEVICE != torch.device("cpu"), persistent_workers=num_workers > 0)
        logging.info("DataLoaders created.")
    except Exception as e:
        logging.error(f"Failed to create DataLoaders: {e}")
        return


    # 5. Initialize Model, Loss, Optimizer, Scheduler
    logging.info("Initializing model...")
    try:
        model = MahjongTransformerModel(
            event_feature_dim=full_dataset.event_dim,
            static_feature_dim=full_dataset.static_dim,
            max_seq_len=effective_max_seq_len, # Use sequence length from data
            d_model=D_MODEL, nhead=NHEAD, d_hid=D_HID, nlayers=NLAYERS, dropout=DROPOUT
        ).to(DEVICE)
    except ValueError as e: # Catch nhead/d_model mismatch
        logging.error(f"Failed to initialize model: {e}")
        return

    logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Scheduler updates per step
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(train_loader), eta_min=LEARNING_RATE / 100) # Smaller eta_min
    logging.info("Loss function, optimizer, and scheduler initialized.")

    # 6. Training Loop
    logging.info(f"--- Starting Training for {NUM_EPOCHS} epochs ---")
    best_val_accuracy = 0.0
    # --- Lists to store metrics for plotting ---
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    # ------------------------------------------

    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0; correct_train = 0; total_train = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False, unit="batch")
        for batch_idx, batch in enumerate(train_pbar):
            try:
                sequences, static_features, labels, masks = [b.to(DEVICE, non_blocking=True) for b in batch] # Use non_blocking for potential speedup

                optimizer.zero_grad()
                outputs = model(sequences, static_features, masks)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM) # Clip gradients
                optimizer.step()
                scheduler.step() # Step scheduler

                train_loss += loss.item() # Store raw loss
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                # Update progress bar postfix less frequently for performance
                if batch_idx % 50 == 0:
                     train_pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1E}")

            except Exception as e:
                logging.error(f"Error during training batch {batch_idx} in epoch {epoch+1}: {e}", exc_info=True)
                continue # Skip this batch on error

        # Calculate average epoch metrics
        if total_train > 0:
            avg_train_loss = train_loss / len(train_loader) # Average loss per batch
            avg_train_acc = 100 * correct_train / total_train
            logging.info(f"Epoch {epoch+1} Train Summary: Loss={avg_train_loss:.4f}, Acc={avg_train_acc:.2f}%")
            train_losses.append(avg_train_loss)
            train_accuracies.append(avg_train_acc)
        else:
             logging.warning(f"Epoch {epoch+1} Training: No samples processed successfully.")
             train_losses.append(float('nan'))
             train_accuracies.append(float('nan'))


        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0; correct_val = 0; total_val = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]  ", leave=False, unit="batch")
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                try:
                    sequences, static_features, labels, masks = [b.to(DEVICE, non_blocking=True) for b in batch]
                    outputs = model(sequences, static_features, masks)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() # Store raw loss
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
                    if batch_idx % 50 == 0:
                        val_pbar.set_postfix(loss=f"{loss.item():.4f}")
                except Exception as e:
                    logging.error(f"Error during validation batch {batch_idx} in epoch {epoch+1}: {e}", exc_info=True)
                    continue # Skip this batch on error

        # Calculate average epoch metrics
        if total_val > 0:
            avg_val_loss = val_loss / len(val_loader) # Average loss per batch
            avg_val_acc = 100 * correct_val / total_val
            logging.info(f"Epoch {epoch+1} Valid Summary: Loss={avg_val_loss:.4f}, Acc={avg_val_acc:.2f}%")
            val_losses.append(avg_val_loss)
            val_accuracies.append(avg_val_acc)

            # --- Save Best Model based on Validation Accuracy ---
            if avg_val_acc > best_val_accuracy:
                best_val_accuracy = avg_val_acc
                save_dir = os.path.dirname(MODEL_SAVE_PATH)
                if save_dir and not os.path.exists(save_dir): os.makedirs(save_dir)
                try:
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    logging.info(f"** Best model saved to {MODEL_SAVE_PATH} (Epoch {epoch+1}, Val Acc: {best_val_accuracy:.2f}%) **")
                except Exception as e:
                    logging.error(f"Failed to save model: {e}")
        else:
             logging.warning(f"Epoch {epoch+1} Validation: No samples processed successfully.")
             val_losses.append(float('nan'))
             val_accuracies.append(float('nan'))


    logging.info("--- Training Finished ---")
    logging.info(f"Best validation accuracy achieved: {best_val_accuracy:.2f}%")

    # --- Plotting ---
    if train_losses and val_losses and train_accuracies and val_accuracies:
        try:
            # Filter out NaN values for plotting if any epoch failed completely
            valid_epochs = [i + 1 for i, (tl, vl, ta, va) in enumerate(zip(train_losses, val_losses, train_accuracies, val_accuracies))
                            if not (np.isnan(tl) or np.isnan(vl) or np.isnan(ta) or np.isnan(va))]
            valid_train_losses = [train_losses[i-1] for i in valid_epochs]
            valid_val_losses = [val_losses[i-1] for i in valid_epochs]
            valid_train_accuracies = [train_accuracies[i-1] for i in valid_epochs]
            valid_val_accuracies = [val_accuracies[i-1] for i in valid_epochs]

            if not valid_epochs:
                 logging.warning("No valid epoch data to plot.")
                 return

            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(valid_epochs, valid_train_losses, 'bo-', label='Training Loss')
            plt.plot(valid_epochs, valid_val_losses, 'ro-', label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(valid_epochs, valid_train_accuracies, 'bo-', label='Training Accuracy')
            plt.plot(valid_epochs, valid_val_accuracies, 'ro-', label='Validation Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plot_save_dir = os.path.dirname(PLOT_SAVE_PATH)
            if plot_save_dir and not os.path.exists(plot_save_dir): os.makedirs(plot_save_dir)
            plt.savefig(PLOT_SAVE_PATH)
            logging.info(f"Training curves saved to {PLOT_SAVE_PATH}")
            # plt.show() # Uncomment to display plot interactively

        except Exception as e:
            logging.error(f"Failed to generate or save plot: {e}", exc_info=True)
    else:
        logging.warning("Metrics lists are empty or incomplete, skipping plot generation.")
    # -----------------------

# --- End Main Training Function ---

if __name__ == "__main__":
    # Check for Matplotlib requirement
    try:
        import matplotlib
    except ImportError:
        print("Error: Matplotlib is required for plotting training curves.")
        print("Please install it using: pip install matplotlib")
        sys.exit(1)

    run_training() # Run the main training process