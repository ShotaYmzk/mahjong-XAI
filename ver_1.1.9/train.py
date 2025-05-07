# /ver_1.1.9/train.py
# ==============================================================================
# =                              Import Libraries                              =
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import numpy as np
import os
import glob
import math
from tqdm import tqdm # プログレスバー表示
import logging      # ログ出力用
import sys
import time         # 時間計測用
import random       # 乱数生成用
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts # 学習率スケジューラ
import matplotlib.pyplot as plt # プロット用
from torch.amp import autocast, GradScaler # 自動混合精度 (AMP) 用
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import gc # ガベージコレクタ

# ==============================================================================
# =        Import Project Modules & Constants (エラーハンドリング強化)        =
# ==============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path: sys.path.insert(0, script_dir)
try:
    # game_state.py から必要な定数をインポート
    from game_state import NUM_TILE_TYPES, MAX_EVENT_HISTORY, EVENT_TYPES, STATIC_FEATURE_DIM
    print(f"Imported constants: NUM_TILE_TYPES={NUM_TILE_TYPES}, MAX_EVENT_HISTORY={MAX_EVENT_HISTORY}, STATIC_FEATURE_DIM={STATIC_FEATURE_DIM}")
except ImportError as e:
     # インポート失敗は致命的なのでログに残し、フォールバック値を使用する警告を出す
     print(f"[FATAL ERROR in train.py] Cannot import constants from game_state.py: {e}")
     logging.critical(f"Failed to import constants from game_state.py: {e}. Using fallback values.")
     NUM_TILE_TYPES = 34         # 牌の種類 (萬子1-9, 筒子1-9, 索子1-9, 字牌7種)
     MAX_EVENT_HISTORY = 60      # Transformerに入力するイベント系列の最大長
     STATIC_FEATURE_DIM = 157    # 1サンプルあたりの静的特徴量の次元数
     EVENT_TYPES = {"PADDING": 8} # イベントタイプ辞書（最低限PADDINGが必要）
     print(f"[Warning] Using fallback constants: NUM_TILE_TYPES={NUM_TILE_TYPES}, MAX_EVENT_HISTORY={MAX_EVENT_HISTORY}, STATIC_FEATURE_DIM={STATIC_FEATURE_DIM}")

# ==============================================================================
# =                           Configuration Settings                           =
# ==============================================================================
# --- データ関連 ---
DATA_DIR = "./training_data/"                           # NPZファイルが保存されているディレクトリ
DATA_PATTERN = os.path.join(DATA_DIR, "mahjong_imitation_data_v119_batch_*.npz") # 読み込むNPZファイルのパターン
VALIDATION_SPLIT = 0.05                                 # データセットからバリデーション用に分割する割合

# --- モデル保存関連 ---
MODEL_SAVE_PATH = "./trained_model/mahjong_transformer_v119_large_compiled.pth" # 最良モデルの保存パス (compile版を示す名前に変更)
CHECKPOINT_DIR = "./checkpoints_v119_large_compiled/"       # チェックポイントの保存ディレクトリ (compile版を示す名前に変更)

# --- ログ・プロット関連 ---
LOG_DIR = "./logs"                                      # ログファイルの保存ディレクトリ
PLOT_DIR = "./plots_v119_large_compiled"                # プロット画像の保存ディレクトリ (compile版を示す名前に変更)
PLOT_EVERY_EPOCH = 1                                    # 何エポックごとにプロットを更新・保存するか
INTERACTIVE_PLOT = False                                # プロットを対話的に表示するか (通常はFalse)

# --- トレーニングハイパーパラメータ ---
BATCH_SIZE = 2048  # Reduced from 4096 to handle memory constraints
NUM_EPOCHS = 50
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.05
CLIP_GRAD_NORM = 1.0
WARMUP_STEPS = 1000
ACCUMULATION_STEPS = 2  # Increased from 1 to handle memory constraints

# --- Transformerモデルハイパーパラメータ ---
D_MODEL = 512               # モデル内部の基本次元数
NHEAD = 8                   # Multi-Head Attentionのヘッド数 (D_MODELを割り切れる必要あり)
D_HID = 2048                # Transformer内部のFeedForward層の中間次元数 (通常 D_MODEL * 4)
NLAYERS = 6                 # Transformer Encoder Layerの数
DROPOUT = 0.1               # ドロップアウト率
ACTIVATION = 'gelu'         # Transformer内部の活性化関数 ('relu' または 'gelu')

# --- 高度なトレーニング機能 ---
USE_AMP = True              # 自動混合精度 (AMP) を使用するか (GPUが対応していればTrue推奨)
USE_TORCH_COMPILE = True    # torch.compile を使用するか (PyTorch 2.0以降、大幅な高速化が期待できる) ★追加
COMPILE_MODE = "default"    # torch.compile のモード ('default', 'reduce-overhead', 'max-autotune')
USE_EMA = False             # Exponential Moving Average を使用するか (オプション)
EMA_DECAY = 0.999           # EMAの減衰率
LABEL_SMOOTHING = 0.1       # ラベルスムージングの度合い (0.0で無効)

# --- その他 ---
EARLY_STOPPING_PATIENCE = 7 # バリデーション精度が向上しなかった場合に早期終了するまでのエポック数
SEED = 42                   # 乱数シード (再現性のため)

# ==============================================================================
# =                             Device Configuration                           =
# ==============================================================================
bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
if torch.cuda.is_available():
     DEVICE = torch.device("cuda")
     torch.backends.cudnn.benchmark = True
     torch.set_float32_matmul_precision('high')
     # Add memory optimization settings
     torch.cuda.empty_cache()
     os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
     print(f"CUDA Device: {torch.cuda.get_device_name(DEVICE)}")
     print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
     print(f"TF32 Matmul Precision: {torch.get_float32_matmul_precision()}")
     print(f"BF16 Supported: {bf16_supported}")
else:
     DEVICE = torch.device("cpu")
     print("[Warning] CUDA not available, using CPU.")
     USE_AMP = False # CPUではAMPは無効
     USE_TORCH_COMPILE = False # CPUでのtorch.compileはまだ実験的
print(f"Using device: {DEVICE}")

# ==============================================================================
# =                                Seed Setting                                =
# ==============================================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)

# ==============================================================================
# =                            Directory Creation                            =
# ==============================================================================
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ==============================================================================
# =                             Logging Setup                                =
# ==============================================================================
log_file_path = os.path.join(LOG_DIR, f"model_training_{os.path.basename(MODEL_SAVE_PATH).replace('.pth','')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, mode='w'), # ログファイルへの出力
        logging.StreamHandler()                      # コンソールへの出力
    ]
)
logging.info(f"Logging to: {log_file_path}")

# ==============================================================================
# =                        Positional Encoding Definition                      =
# ==============================================================================
class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Encoding (RoPE)の実装"""
    def __init__(self, d_model, max_len=MAX_EVENT_HISTORY):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be divisible by 2 for Rotary Positional Encoding.")
        self.d_model = d_model
        self.max_len = max_len
        self.dim_half = d_model // 2

        # 周波数を計算 (θ_i = 1 / (base^(2i / d)))
        freqs = 1.0 / (10000 ** (torch.arange(0, self.dim_half).float() / self.dim_half))
        # 計算した周波数をバッファとして登録 (モデルパラメータではないが、状態として保存される)
        self.register_buffer('freqs', freqs)

        # 位置インデックス (0, 1, ..., max_len-1) を生成し、バッファとして登録
        pos_seq = torch.arange(max_len).float()
        self.register_buffer('pos_seq', pos_seq)

    def forward(self, x):
        # 入力xの形状: (Batch, SeqLen, Dim)
        seq_len = x.shape[1]
        if seq_len > self.max_len:
             # 事前計算した最大長を超えた場合は警告し、動的に再計算（非効率なので避けるべき）
             logging.warning(f"RoPE: Input sequence length {seq_len} > precomputed max_len {self.max_len}. Recomputing positions.")
             positions = torch.arange(seq_len, device=x.device).float().unsqueeze(0)
        else:
             # 事前計算した位置インデックスを使用
             positions = self.pos_seq[:seq_len].unsqueeze(0).to(x.device) # (1, SeqLen)

        # 角度を計算: θ * m (mは位置)
        # freqs: (Dim/2), positions: (1, SeqLen) -> angles: (1, SeqLen, Dim/2)
        angles = positions.unsqueeze(-1) * self.freqs.unsqueeze(0).to(x.device)

        # sinとcosを計算
        sin_angles = torch.sin(angles) # (1, SeqLen, Dim/2)
        cos_angles = torch.cos(angles) # (1, SeqLen, Dim/2)

        # 入力xを偶数番目と奇数番目の次元に分割
        # x: (B, S, D) -> x_even, x_odd: (B, S, D/2)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # RoPEの回転を適用:
        # x_even' = x_even * cosθ - x_odd * sinθ
        # x_odd'  = x_even * sinθ + x_odd * cosθ
        x_even_rotated = x_even * cos_angles - x_odd * sin_angles
        x_odd_rotated = x_even * sin_angles + x_odd * cos_angles # 元のコードはここが + x_odd * cos になっていたが、通常は x_even * sin

        # 回転後の次元を結合して元の形状に戻す
        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = x_even_rotated
        x_rotated[..., 1::2] = x_odd_rotated

        return x_rotated

# ==============================================================================
# =                       Transformer Model Definition                       =
# ==============================================================================
class MahjongTransformerV2(nn.Module):
    """イベント系列と静的特徴を入力とするTransformerモデル"""
    def __init__(self, event_feature_dim, static_feature_dim, d_model=D_MODEL, nhead=NHEAD, d_hid=D_HID, nlayers=NLAYERS, dropout=DROPOUT, activation=ACTIVATION, output_dim=NUM_TILE_TYPES):
        super().__init__()
        self.d_model = d_model

        # 1. イベント系列特徴量をd_model次元にエンコードする層
        self.event_encoder = nn.Sequential(
            nn.Linear(event_feature_dim, d_model),
            nn.LayerNorm(d_model), # Layer Normalizationを追加
            nn.Dropout(dropout)
        )

        # 2. 位置エンコーディング層 (RoPEを使用)
        self.pos_encoder = RotaryPositionalEncoding(d_model)

        # 3. Transformer Encoder層
        #    - norm_first=True: LayerNormをAttention/FFNの前に行う (安定化しやすい)
        #    - batch_first=True: 入力テンソルの形状を (Batch, SeqLen, Dim) にする
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout,
            activation=activation, batch_first=True, norm_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # 4. 静的特徴量をd_model次元にエンコードする層
        self.static_encoder = nn.Sequential(
            nn.Linear(static_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(), # 活性化関数
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model), # もう一層追加して表現力を上げる
            nn.LayerNorm(d_model)
        )

        # 5. Attention Pooling層 (Transformerの出力系列を集約する)
        #    - 各タイムステップの重要度を学習し、重み付き和を取る
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1) # SeqLen次元でSoftmax
        )

        # 6. 最終出力ヘッド
        #    - Attention Poolingされたイベント特徴とエンコードされた静的特徴を結合
        #    - 複数層の線形層と活性化関数を通して最終的な出力次元 (牌の種類数) に変換
        self.output_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), # イベント特徴(d_model) + 静的特徴(d_model)
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5), # 出力に近い層でドロップアウト率を下げることも
            nn.Linear(d_model // 2, output_dim)
        )

        self._init_weights() # 重みの初期化を実行

    def _init_weights(self):
        """モデルの重みを初期化する"""
        for name, p in self.named_parameters():
            if p.dim() > 1: # 重み行列の場合
                # Xavier Glorot初期化 (正規分布) を使用
                # gain は活性化関数に応じて調整 (GELUは通常1.0)
                gain = nn.init.calculate_gain(ACTIVATION) if ACTIVATION in ['relu', 'leaky_relu'] else 1.0
                nn.init.xavier_normal_(p, gain=gain)
            elif 'bias' in name: # バイアス項の場合
                nn.init.zeros_(p) # ゼロで初期化

    def forward(self, event_seq, static_feat, attention_mask=None):
        """
        モデルのフォワードパス
        Args:
            event_seq (Tensor): イベント系列データ (Batch, SeqLen, EventFeatDim)
            static_feat (Tensor): 静的特徴データ (Batch, StaticFeatDim)
            attention_mask (Tensor, optional): パディング部分を示すマスク (Batch, SeqLen)。Trueの部分がパディング。
        Returns:
            Tensor: 各牌の選択確率のlogit (Batch, OutputDim)
        """
        # 1. イベント系列をエンコード
        # (B, S, E_dim) -> (B, S, D_model)
        event_encoded = self.event_encoder(event_seq)

        # 2. 位置エンコーディングを適用
        # (B, S, D_model) -> (B, S, D_model)
        pos_encoded = self.pos_encoder(event_encoded)

        # 3. Transformer Encoderに入力
        # src_key_padding_mask は True の位置を無視する
        # (B, S, D_model) -> (B, S, D_model)
        transformer_output = self.transformer_encoder(pos_encoded, src_key_padding_mask=attention_mask)

        # 4. Attention Poolingで系列を集約
        # (B, S, D_model) -> (B, S, 1)
        attn_weights = self.attention_pool(transformer_output)
        # パディング部分の重みを0にする (Softmaxの後でも適用可能)
        if attention_mask is not None:
            # マスクの形状を (B, S, 1) に拡張
            mask_expanded = attention_mask.unsqueeze(-1)
            attn_weights = attn_weights.masked_fill(mask_expanded, 0.0)

        # 重み付き和を計算してコンテキストベクトルを得る
        # (B, S, 1) * (B, S, D_model) -> sum over S -> (B, D_model)
        context_vector = torch.sum(attn_weights * transformer_output, dim=1)

        # 5. 静的特徴量をエンコード
        # (B, Static_dim) -> (B, D_model)
        static_encoded = self.static_encoder(static_feat)

        # 6. イベントコンテキストと静的特徴を結合
        # (B, D_model), (B, D_model) -> (B, D_model * 2)
        combined = torch.cat([context_vector, static_encoded], dim=1)

        # 7. 出力ヘッドを通して最終出力を得る
        # (B, D_model * 2) -> (B, OutputDim)
        return self.output_head(combined)

# ==============================================================================
# =                  Exponential Moving Average (EMA)                          =
# ==============================================================================
class EMA:
    """モデルパラメータのExponential Moving Averageを計算・適用するクラス (オプション)"""
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {} # EMAパラメータを保持する辞書
        self.backup = {} # apply_shadow時に元のパラメータを保持する辞書
        self.register()

    def register(self):
        """現在のモデルパラメータをシャドウパラメータとして初期登録"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """現在のモデルパラメータを使ってシャドウパラメータを更新"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow # 登録されているか確認
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """現在のモデルパラメータをバックアップし、シャドウパラメータをモデルに適用"""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """バックアップした元のパラメータをモデルに戻す"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup # バックアップがあるか確認
                param.data = self.backup[name]
        self.backup = {}

# ==============================================================================
# =                            Dataset Definition                            =
# ==============================================================================
class MahjongNpzDataset(Dataset):
    """
    複数のNPZファイルから麻雀データを読み込むDatasetクラス。
    ファイル単位でのキャッシング機能を持つ。
    """
    def __init__(self, npz_files):
        self.npz_files = npz_files          # NPZファイルのパスリスト
        self.file_metadata = []             # 各ファイルの {'path': str, 'length': int} を格納
        self.cumulative_lengths = [0]       # 各ファイルまでの累積サンプル数
        self.total_length = 0               # データセット全体の総サンプル数

        # データ次元（最初の有効なファイルから取得）
        self.seq_len = -1                   # イベント系列長
        self.event_dim = -1                 # 1イベントあたりの特徴量次元
        self.static_dim = -1                # 静的特徴量の次元

        # 定数とフィールド名
        self.padding_code = float(EVENT_TYPES["PADDING"]) # パディングを示すイベントコード
        self.sequences_field = 'sequences'          # NPZ内のシーケンスデータのキー名
        self.static_field = 'static_features'       # NPZ内の静的データのキー名
        self.labels_field = 'labels'                # NPZ内のラベルデータのキー名

        # --- キャッシュ用属性 ---
        self.cache_file_idx = -1            # 現在キャッシュしているファイルのインデックス
        self.cached_data = None             # キャッシュされたデータ (NumPy配列の辞書)
        # ----------------------

        logging.info("Scanning NPZ files for metadata...")
        first_file = True
        skipped_count = 0
        valid_files = [] # 実際に使用するファイルのリスト

        # 全NPZファイルをスキャンしてメタデータ（パス、サンプル数）を収集し、次元を検証
        for f in tqdm(self.npz_files, desc="Scanning files", leave=False):
            try:
                # ファイルサイズが小さすぎる場合はスキップ (オプション)
                if os.path.getsize(f) < 100:
                    skipped_count += 1; continue

                # NPZファイルを開いて中身を確認
                with np.load(f, allow_pickle=False) as data:
                    # 必要なキーが存在するか確認
                    if not all(k in data for k in [self.sequences_field, self.static_field, self.labels_field]):
                        skipped_count += 1; continue
                    # サンプル数を取得
                    length = len(data[self.labels_field])
                    if length == 0: skipped_count += 1; continue # 空のファイルはスキップ

                    # --- 次元チェック ---
                    current_seq_shape = data[self.sequences_field].shape
                    current_static_shape = data[self.static_field].shape

                    # 配列の次元数を確認 (Sequence: 3, Static: 2)
                    if len(current_seq_shape) != 3 or len(current_static_shape) != 2:
                         logging.warning(f"Skipping {f} - incorrect array dimensions. Seq: {current_seq_shape}, Static: {current_static_shape}")
                         skipped_count += 1; continue

                    # 特徴量の次元数を取得 (サンプル数を除く)
                    current_seq_len, current_event_dim = current_seq_shape[1:]
                    current_static_dim = current_static_shape[1]

                    # 最初の有効なファイルで次元を設定
                    if first_file:
                        self.seq_len = current_seq_len
                        self.event_dim = current_event_dim
                        # 静的特徴量の次元が期待通りか確認
                        if current_static_dim != STATIC_FEATURE_DIM:
                             raise ValueError(f"Static dim mismatch! Expected {STATIC_FEATURE_DIM}, got {current_static_dim} in {f}")
                        self.static_dim = current_static_dim
                        first_file = False
                    # 2番目以降のファイルで次元が一致するか検証
                    else:
                        if (current_seq_len != self.seq_len or
                            current_event_dim != self.event_dim or
                            current_static_dim != self.static_dim):
                            logging.warning(f"Skipping {f} - dimension mismatch. "
                                            f"Seq: {(current_seq_len, current_event_dim)} (exp {(self.seq_len, self.event_dim)}), "
                                            f"Static: {current_static_dim} (exp {self.static_dim})")
                            skipped_count += 1; continue
                    # --- 次元チェック終了 ---

                    # 有効なファイルとしてメタデータを記録
                    self.file_metadata.append({'path': f, 'length': length})
                    self.total_length += length
                    self.cumulative_lengths.append(self.total_length) # 累積長を更新
                    valid_files.append(f) # 有効なファイルリストに追加

            except Exception as e:
                logging.error(f"Error reading metadata from {f}: {e}")
                skipped_count += 1

        self.npz_files = valid_files # ファイルリストを有効なものだけに更新
        if self.total_length == 0:
            raise RuntimeError("No valid data found in NPZ files after scanning.")

        logging.info(f"Dataset initialized: {len(self.file_metadata)} files ({skipped_count} skipped), {self.total_length} samples.")
        logging.info(f"Dims: SeqLen={self.seq_len}, Event={self.event_dim}, Static={self.static_dim}")

    def __len__(self):
        """データセットの総サンプル数を返す"""
        return self.total_length

    def __getitem__(self, idx):
        """指定されたインデックスのサンプルを返す"""
        if not 0 <= idx < self.total_length:
            raise IndexError("Index out of bounds")

        # idxがどのファイルに属するかを二分探索で特定
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        file_info = self.file_metadata[file_idx]
        # そのファイル内でのオフセット（インデックス）を計算
        offset = idx - self.cumulative_lengths[file_idx]

        try:
            # --- キャッシュロジック ---
            if file_idx != self.cache_file_idx:
                # キャッシュミス: 新しいファイルをロード
                # logging.debug(f"Cache miss for idx {idx}. Loading file {file_idx}: {file_info['path']}")
                if self.cached_data is not None:
                    del self.cached_data # 古いキャッシュデータを削除
                    gc.collect()          # メモリ解放を試みる
                # 新しいファイルをメモリにロード
                with np.load(file_info['path'], allow_pickle=False) as data:
                    # 必要な配列のみをキャッシュ
                    self.cached_data = {
                        self.sequences_field: data[self.sequences_field],
                        self.static_field: data[self.static_field],
                        self.labels_field: data[self.labels_field]
                    }
                self.cache_file_idx = file_idx # キャッシュしたファイルインデックスを更新
            # --- キャッシュロジック終了 ---

            # キャッシュからデータを取得
            seq = self.cached_data[self.sequences_field][offset].astype(np.float32)
            static = self.cached_data[self.static_field][offset].astype(np.float32)
            label = self.cached_data[self.labels_field][offset].astype(np.int64)

            # パディングマスクを生成 (シーケンスの最初の特徴量がパディングコードかチェック)
            padding_mask = (seq[:, 0] == self.padding_code) # (SeqLen,) の boolean Tensor

            # 念のため形状をチェック (初期化時の次元と一致するはず)
            if seq.shape != (self.seq_len, self.event_dim) or static.shape != (self.static_dim,):
                logging.error(f"Shape mismatch for loaded sample {idx} from {file_info['path']}! "
                              f"Seq: {seq.shape} (expected {(self.seq_len, self.event_dim)}), "
                              f"Static: {static.shape} (expected {(self.static_dim,)}). Returning zeros.")
                # エラー時はゼロ埋めデータを返す (トレーニングを止めないため)
                seq = np.zeros((self.seq_len, self.event_dim), dtype=np.float32)
                static = np.zeros((self.static_dim,), dtype=np.float32)
                label = np.zeros((), dtype=np.int64) # ラベル0を返す
                padding_mask = np.ones((self.seq_len,), dtype=bool) # 全てパディング扱い

            return seq, static, label, padding_mask

        except Exception as e:
            # 予期せぬエラーが発生した場合
            logging.error(f"CRITICAL Error loading sample {idx} (offset {offset}) from file {file_idx} ({file_info['path']}): {e}", exc_info=True)
            # トレーニングを止めないためにゼロデータを返す
            return np.zeros((self.seq_len, self.event_dim), dtype=np.float32), \
                   np.zeros((self.static_dim,), dtype=np.float32), \
                   np.zeros((), dtype=np.int64), \
                   np.ones((self.seq_len,), dtype=bool) # 全てパディング扱い

# ==============================================================================
# =                          Loss Function Definition                          =
# ==============================================================================
class LabelSmoothingLoss(nn.Module):
    """ラベルスムージング付きクロスエントロピー損失"""
    def __init__(self, smoothing=0.0, num_classes=NUM_TILE_TYPES):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
        # KLDivLossを使用。log_target=Falseがデフォルト。reduction='batchmean'はバッチ全体で平均を取る
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, pred_logits, target):
        # pred_logits: モデルの出力 (Batch, NumClasses)
        # target: 正解ラベル (Batch,)
        # 1. モデルの出力をlog-softmaxにかける
        pred_log_softmax = torch.log_softmax(pred_logits, dim=-1)

        # 2. 正解分布を作成 (ラベルスムージング適用)
        with torch.no_grad(): # 勾配計算は不要
            true_dist = torch.zeros_like(pred_log_softmax)
            # 全てのクラスに均等に smoothing / (num_classes - 1) を割り当て
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            # 正解ラベルの位置に confidence (1.0 - smoothing) を割り当て
            # target.data.unsqueeze(1) で (Batch,) -> (Batch, 1) に変形
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        # 3. KLダイバージェンスを計算
        return self.criterion(pred_log_softmax, true_dist)

# ==============================================================================
# =                            Metrics Calculation                             =
# ==============================================================================
def calculate_accuracy(predictions, targets):
    """Top-1およびTop-3精度を計算する"""
    with torch.no_grad():
        # Top-1精度: 最も確率の高い予測が正解と一致するか
        _, predicted_indices = torch.max(predictions, 1)
        correct = (predicted_indices == targets).float()
        accuracy = correct.mean().item()

        # Top-3精度: 確率の高い上位3つの予測の中に正解が含まれるか
        _, top3_indices = torch.topk(predictions, 3, dim=1)
        # targetsを(Batch, 1)に変形して比較
        target_expanded = targets.unsqueeze(1)
        top3_correct = torch.any(top3_indices == target_expanded, dim=1).float()
        top3_accuracy = top3_correct.mean().item()

    return accuracy, top3_accuracy

# ==============================================================================
# =                             Plotting Functions                             =
# ==============================================================================
def init_plots():
    """プロット用のFigureとAxesを初期化"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10)) # 2x2のサブプロットを作成
    axs[0, 0].set_title('Loss (Train/Val)')
    axs[0, 1].set_title('Accuracy (Train/Val)')
    axs[1, 0].set_title('Top-3 Accuracy (Train/Val)')
    axs[1, 1].set_title('Learning Rate')
    plt.tight_layout() # サブプロット間のスペース調整
    return fig, axs

def update_plots(fig, axs, epoch, metrics):
    """収集したメトリクスでプロットを更新し、保存する"""
    epochs = list(range(1, epoch + 2)) # X軸用エポックリスト (1から開始)

    # Lossプロット
    axs[0,0].clear(); axs[0,0].plot(epochs, metrics['train_loss'], 'b-', label='Train'); axs[0,0].plot(epochs, metrics['val_loss'], 'r-', label='Val'); axs[0,0].legend(); axs[0,0].grid(True); axs[0,0].set_title('Loss (Train/Val)')

    # Accuracyプロット
    axs[0,1].clear(); axs[0,1].plot(epochs, metrics['train_acc'], 'b-', label='Train'); axs[0,1].plot(epochs, metrics['val_acc'], 'r-', label='Val'); axs[0,1].legend(); axs[0,1].grid(True); axs[0,1].set_title('Accuracy (Train/Val)')

    # Top-3 Accuracyプロット
    axs[1,0].clear(); axs[1,0].plot(epochs, metrics['train_top3'], 'b-', label='Train'); axs[1,0].plot(epochs, metrics['val_top3'], 'r-', label='Val'); axs[1,0].legend(); axs[1,0].grid(True); axs[1,0].set_title('Top-3 Accuracy (Train/Val)')

    # Learning Rateプロット
    axs[1,1].clear(); axs[1,1].plot(epochs, metrics['lr'], 'g-'); axs[1,1].grid(True); axs[1,1].set_title('Learning Rate')

    plt.tight_layout() # 再度レイアウト調整

    # プロットをファイルに保存
    plot_path = os.path.join(PLOT_DIR, f'training_metrics_epoch_{epoch+1}.png')
    latest_path = os.path.join(PLOT_DIR, 'latest_training_metrics.png')
    try:
        fig.savefig(plot_path)
        fig.savefig(latest_path) # 最新のプロットを別名で保存
    except Exception as e:
        logging.warning(f"Failed to save plot: {e}")

    # 対話モードなら画面に表示
    if INTERACTIVE_PLOT:
        plt.pause(0.1)

# ==============================================================================
# =                            Main Training Loop                            =
# ==============================================================================
def train_model():
    """モデルのトレーニングを実行するメイン関数"""
    logging.info("Starting training process...")

    # --- データファイルの検索 ---
    npz_files = sorted(glob.glob(DATA_PATTERN))
    if not npz_files:
        logging.error(f"No NPZ files found matching pattern: {DATA_PATTERN}")
        return
    logging.info(f"Found {len(npz_files)} NPZ files")

    # --- データセットの初期化 ---
    try:
        full_dataset = MahjongNpzDataset(npz_files)
    except (RuntimeError, ValueError) as e:
        logging.error(f"Failed to initialize dataset: {e}")
        return

    # データセットから次元情報を取得
    if full_dataset.event_dim <= 0 or full_dataset.static_dim <= 0 or full_dataset.seq_len <= 0:
        logging.error("Invalid feature dimensions determined from dataset.")
        return
    event_dim = full_dataset.event_dim
    static_dim = full_dataset.static_dim
    logging.info(f"Dataset dimensions: Event={event_dim}, Static={static_dim}, SeqLen={full_dataset.seq_len}")

    # --- データセットの分割 (Train / Validation) ---
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    if val_size == 0 and len(full_dataset) > 0 and VALIDATION_SPLIT > 0:
        val_size = max(1, int(len(full_dataset) * 0.01)) # 最低1サンプルか1%は確保
        train_size = len(full_dataset) - val_size
        logging.warning(f"Validation split resulted in 0 samples, adjusting to {val_size} validation samples.")
    if train_size <= 0 or val_size <= 0:
         logging.error(f"Invalid dataset split: Train={train_size}, Val={val_size}. Check dataset size and validation split.")
         return

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(SEED))
    logging.info(f"Dataset split: Train samples={train_size}, Validation samples={val_size}")

    # --- DataLoaderの準備 ---
    # CPUコア数に基づいてワーカー数を決定 (多すぎるとオーバーヘッド増)
    num_workers = min(os.cpu_count() // 2 if os.cpu_count() else 1, 8) # CPUコアの半分、最大8程度
    logging.info(f"Setting up DataLoaders with {num_workers} workers...")
    pin_memory_flag = (DEVICE.type == 'cuda') # GPU使用時のみ有効
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory_flag,
        prefetch_factor=4 if num_workers > 0 else None, # 先行読み込みするバッチ数 ★調整点
        persistent_workers=(num_workers > 0), # ワーカープロセスを維持してオーバーヘッド削減
        drop_last=True # バッチサイズに満たない最後のバッチを捨てる
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, # バリデーションはシャッフル不要、バッチサイズは大きくても良い
        num_workers=num_workers, pin_memory=pin_memory_flag,
        prefetch_factor=4 if num_workers > 0 else None, # ★調整点
        persistent_workers=(num_workers > 0)
    )
    logging.info(f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- モデル、オプティマイザ、スケジューラ、損失関数、スケーラーの初期化 ---
    logging.info("Initializing model, optimizer, scheduler, loss, and scaler...")
    model = MahjongTransformerV2(event_feature_dim=event_dim, static_feature_dim=static_dim).to(DEVICE)
    logging.info(f"Model Parameters (Trainable): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- torch.compile の適用 --- ★追加
    if USE_TORCH_COMPILE and hasattr(torch, 'compile'):
        logging.info(f"Applying torch.compile with mode='{COMPILE_MODE}'...")
        try:
            # モデルをコンパイル (PyTorch 2.0以降)
            model = torch.compile(model, mode=COMPILE_MODE)
            logging.info("torch.compile applied successfully.")
        except Exception as e:
            logging.warning(f"torch.compile failed: {e}. Proceeding without compile.", exc_info=True)
            # コンパイル失敗時は USE_TORCH_COMPILE を False にするなどの対応も可能
    elif USE_TORCH_COMPILE:
        logging.warning("torch.compile requested but not available (requires PyTorch 2.0+).")
    # ---------------------------

    # EMA (オプション)
    ema = EMA(model, decay=EMA_DECAY) if USE_EMA else None
    if USE_EMA: logging.info("Using Exponential Moving Average (EMA).")

    # 損失関数
    criterion = LabelSmoothingLoss(smoothing=LABEL_SMOOTHING) if LABEL_SMOOTHING > 0 else nn.CrossEntropyLoss()
    logging.info(f"Using Loss: {type(criterion).__name__}" + (f" with smoothing={LABEL_SMOOTHING}" if LABEL_SMOOTHING > 0 else ""))

    # オプティマイザ (AdamW推奨)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98), eps=1e-6)
    logging.info(f"Using Optimizer: AdamW (LR={LEARNING_RATE}, WD={WEIGHT_DECAY})")

    # 学習率スケジューラ (CosineAnnealingWarmRestarts)
    # T_0 はウォームリスタート間のエポック数 (全エポックの1/5程度)
    steps_per_epoch_for_scheduler = len(train_loader) # 1エポックあたりのDataLoaderのステップ数
    scheduler_t0 = max(1, NUM_EPOCHS // 5) # エポック単位で設定
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=scheduler_t0, T_mult=1, eta_min=LEARNING_RATE / 100)
    logging.info(f"Using LR Scheduler: CosineAnnealingWarmRestarts (T_0={scheduler_t0}, eta_min={LEARNING_RATE / 100})")

    # GradScaler (AMP用)
    # AMPが無効でも、enabled=False で GradScaler オブジェクトは作成しておく (コード分岐を減らすため)
    scaler = GradScaler(enabled=(USE_AMP and DEVICE.type == 'cuda'))
    logging.info(f"Automatic Mixed Precision (AMP) {'Enabled' if scaler.is_enabled() else 'Disabled'}")
    amp_dtype = torch.bfloat16 if bf16_supported else torch.float16
    if scaler.is_enabled(): logging.info(f"AMP dtype: {amp_dtype}")

    # --- トレーニング状態の追跡用変数 ---
    metrics = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_top3': [], 'val_top3': [], 'lr': []}
    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    if INTERACTIVE_PLOT: plt.ion() # 対話モード開始
    fig, axs = init_plots()       # プロット初期化

    # --- トレーニングループ開始 ---
    logging.info(f"Starting training for {NUM_EPOCHS} epochs on {DEVICE}...")
    logging.info(f"Effective batch size: {BATCH_SIZE * ACCUMULATION_STEPS} (Batch Size: {BATCH_SIZE}, Accumulation Steps: {ACCUMULATION_STEPS})")

    total_start_time = time.time() # 全体の開始時間

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time() # エポック開始時間

        # ============ トレーニングフェーズ ============
        model.train() # モデルをトレーニングモードに設定
        train_loss_accum = 0.0 # エポックの累積損失
        train_acc_accum = 0.0  # エポックの累積Top-1精度
        train_top3_accum = 0.0 # エポックの累積Top-3精度
        optimizer_steps = 0    # このエポックで実行されたOptimizerステップ数

        # プログレスバー設定
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)

        # --- 時間計測用 (デバッグ) ---
        # time_data_loading = 0.0
        # time_transfer = 0.0
        # time_forward = 0.0
        # time_backward = 0.0
        # time_step = 0.0
        # iter_start_time = time.time()
        # --------------------------

        optimizer.zero_grad(set_to_none=True) # エポック開始時に勾配をクリア

        for i, batch in pbar:
            # iter_data_end_time = time.time()
            # time_data_loading += (iter_data_end_time - iter_start_time)

            try:
                 seq, static, labels, padding_mask = batch
                 # データをデバイスに転送
                 seq = seq.to(DEVICE, non_blocking=pin_memory_flag)
                 static = static.to(DEVICE, non_blocking=pin_memory_flag)
                 labels = labels.to(DEVICE, non_blocking=pin_memory_flag)
                 padding_mask = padding_mask.to(DEVICE, non_blocking=pin_memory_flag)
            except Exception as e:
                 logging.error(f"Error unpacking or moving batch {i} to device: {e}", exc_info=True)
                 # iter_start_time = time.time() # 次のイテレーションの開始時間
                 continue # エラーが発生したバッチはスキップ

            # iter_transfer_end_time = time.time()
            # time_transfer += (iter_transfer_end_time - iter_data_end_time)

            # AMPコンテキスト内でフォワードパスを実行
            with autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=scaler.is_enabled()):
                outputs = model(seq, static, padding_mask)
                loss = criterion(outputs, labels)
                # 勾配累積のために損失を調整
                if ACCUMULATION_STEPS > 1:
                    loss = loss / ACCUMULATION_STEPS

            # iter_forward_end_time = time.time()
            # time_forward += (iter_forward_end_time - iter_transfer_end_time)

            # NaN損失チェック
            if torch.isnan(loss):
                 logging.warning(f"NaN loss detected at epoch {epoch+1}, batch {i}. Skipping gradient update.")
                 # NaNが発生した場合、このバッチの勾配は使わず、次のバッチへ
                 # オプティマイザステップもスキップされるようにする
                 # scaler.update() # スケーラーの状態は更新すべきか？ドキュメント確認
                 # iter_start_time = time.time() # 次のイテレーションの開始時間
                 continue

            # スケーラーを使って勾配計算
            scaler.scale(loss).backward()

            # iter_backward_end_time = time.time()
            # time_backward += (iter_backward_end_time - iter_forward_end_time)

            # ACCUMULATION_STEPSごとにパラメータ更新
            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                # 勾配クリッピング (オプション)
                if CLIP_GRAD_NORM > 0:
                    scaler.unscale_(optimizer) # step前にunscaleが必要
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)

                # オプティマイザステップとスケーラー更新
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True) # 勾配をクリア
                optimizer_steps += 1

                # EMA更新 (オプション)
                if ema:
                    ema.update()

            # iter_step_end_time = time.time()
            # time_step += (iter_step_end_time - iter_backward_end_time)

            # メトリクス計算 (勾配計算に影響しないように no_grad コンテキスト内)
            with torch.no_grad():
                acc, acc_top3 = calculate_accuracy(outputs.detach(), labels.detach())
                # 累積前の損失値を記録
                current_loss = loss.item() * ACCUMULATION_STEPS if ACCUMULATION_STEPS > 1 else loss.item()

            train_loss_accum += current_loss
            train_acc_accum += acc
            train_top3_accum += acc_top3

            # プログレスバーに情報を表示
            pbar.set_postfix({'Loss': f'{current_loss:.4f}', 'Acc': f'{acc:.3f}', 'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'})
            # iter_start_time = time.time() # 次のイテレーションの開始時間

        # エポック終了時の平均メトリクスを計算
        num_batches_processed = len(train_loader)
        if num_batches_processed > 0:
            epoch_train_loss = train_loss_accum / num_batches_processed
            epoch_train_acc = train_acc_accum / num_batches_processed
            epoch_train_acc_top3 = train_top3_accum / num_batches_processed
        else:
            logging.warning(f"Epoch {epoch+1} had no batches processed in training loop.")
            epoch_train_loss, epoch_train_acc, epoch_train_acc_top3 = 0.0, 0.0, 0.0

        # メトリクスリストに追加
        metrics['train_loss'].append(epoch_train_loss)
        metrics['train_acc'].append(epoch_train_acc)
        metrics['train_top3'].append(epoch_train_acc_top3)
        metrics['lr'].append(optimizer.param_groups[0]['lr']) # 現在の学習率を記録

        # --- 時間計測結果表示 (デバッグ用) ---
        # logging.debug(f"Epoch {epoch+1} Timing Breakdown:")
        # logging.debug(f"  Data Loading: {time_data_loading:.2f}s")
        # logging.debug(f"  Data Transfer: {time_transfer:.2f}s")
        # logging.debug(f"  Forward Pass: {time_forward:.2f}s")
        # logging.debug(f"  Backward Pass: {time_backward:.2f}s")
        # logging.debug(f"  Optimizer Step: {time_step:.2f}s")
        # ---------------------------------

        # ============ バリデーションフェーズ ============
        model.eval() # モデルを評価モードに設定
        val_loss_accum = 0.0
        val_acc_accum = 0.0
        val_top3_accum = 0.0
        val_batches_processed = 0

        # EMAを使用している場合、評価時にはEMAの重みを使用
        if ema:
            ema.apply_shadow()

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=False)
        with torch.no_grad(): # 勾配計算を無効化
            for batch in val_pbar:
                try:
                    seq, static, labels, padding_mask = batch
                    seq = seq.to(DEVICE, non_blocking=pin_memory_flag)
                    static = static.to(DEVICE, non_blocking=pin_memory_flag)
                    labels = labels.to(DEVICE, non_blocking=pin_memory_flag)
                    padding_mask = padding_mask.to(DEVICE, non_blocking=pin_memory_flag)
                except Exception as e:
                    logging.error(f"Error unpacking or moving validation batch to device: {e}", exc_info=True)
                    continue

                # AMPコンテキスト内で実行 (推論時も有効な場合がある)
                with autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=scaler.is_enabled()):
                    outputs = model(seq, static, padding_mask)
                    loss = criterion(outputs, labels)

                if torch.isnan(loss):
                    logging.warning(f"NaN loss detected during validation epoch {epoch+1}. Skipping batch.")
                    continue

                acc, acc_top3 = calculate_accuracy(outputs, labels)
                val_loss_accum += loss.item()
                val_acc_accum += acc
                val_top3_accum += acc_top3
                val_batches_processed += 1
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{acc:.3f}'})

        # EMAを使用していた場合、元の重みに戻す
        if ema:
            ema.restore()

        # バリデーションメトリクスの平均を計算
        if val_batches_processed > 0:
            epoch_val_loss = val_loss_accum / val_batches_processed
            epoch_val_acc = val_acc_accum / val_batches_processed
            epoch_val_acc_top3 = val_top3_accum / val_batches_processed
        else:
            logging.warning(f"Epoch {epoch+1} had no batches processed in validation loop.")
            epoch_val_loss, epoch_val_acc, epoch_val_acc_top3 = 0.0, 0.0, 0.0

        metrics['val_loss'].append(epoch_val_loss)
        metrics['val_acc'].append(epoch_val_acc)
        metrics['val_top3'].append(epoch_val_acc_top3)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # --- エポック結果のログ出力 ---
        logging.info(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Time: {epoch_duration:.1f}s - "
            f"Train Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.4f} - Top3: {epoch_train_acc_top3:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc:.4f} - Top3: {epoch_val_acc_top3:.4f} | "
            f"LR: {metrics['lr'][-1]:.6f}"
        )

        # --- 学習率スケジューラの更新 ---
        # ReduceLROnPlateauの場合は val_loss を渡すなど、スケジューラに合わせて調整
        lr_scheduler.step()

        # --- プロットの更新・保存 ---
        if (epoch + 1) % PLOT_EVERY_EPOCH == 0 or epoch == NUM_EPOCHS - 1:
            update_plots(fig, axs, epoch, metrics)

        # --- チェックポイントの保存 ---
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
        save_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict() if not USE_TORCH_COMPILE else model._orig_mod.state_dict(), # compile後は ._orig_mod
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_acc': epoch_val_acc,
            'event_dim': event_dim, # モデル再構築のため次元を保存
            'static_dim': static_dim
        }
        if ema:
            # EMAのシャドウパラメータも保存
            save_dict['ema_state_dict'] = ema.shadow
        torch.save(save_dict, checkpoint_path)
        logging.info(f"Checkpoint saved to {checkpoint_path}")

        # --- 最良モデルの保存 & アーリーストッピング ---
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            # EMAを使用している場合はEMAの重みを保存、そうでなければモデルの重みを保存
            save_model_state = ema.shadow if ema else (model._orig_mod.state_dict() if USE_TORCH_COMPILE else model.state_dict())
            best_model_save_dict = {
                 'model_state_dict': save_model_state,
                 'event_dim': event_dim, # 次元情報も一緒に保存
                 'static_dim': static_dim
            }
            torch.save(best_model_save_dict, MODEL_SAVE_PATH)
            logging.info(f"*** New best model saved at epoch {best_epoch} with Val Acc: {best_val_acc:.4f} to {MODEL_SAVE_PATH} ***")
        else:
            epochs_without_improvement += 1
            logging.info(f"Validation accuracy did not improve for {epochs_without_improvement} epoch(s). Best was {best_val_acc:.4f} at epoch {best_epoch}.")
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                logging.info(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
                break # トレーニングループを終了

    # --- トレーニング終了処理 ---
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    logging.info("="*30)
    logging.info("Training loop finished.")
    logging.info(f"Total training time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    logging.info(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    logging.info(f"Best model saved to: {MODEL_SAVE_PATH}")
    logging.info(f"Checkpoints saved in: {CHECKPOINT_DIR}")
    logging.info(f"Logs saved to: {log_file_path}")
    logging.info(f"Plots saved in: {PLOT_DIR}")

    # 最終プロットの保存
    final_epoch_index = epoch # ループがbreakした場合でも最後のepochを使う
    update_plots(fig, axs, final_epoch_index, metrics)
    final_plot_path = os.path.join(PLOT_DIR, 'final_training_curves.png')
    plt.figure(fig) # プロット対象のFigureを明示
    plt.savefig(final_plot_path)
    logging.info(f"Final training curves saved to: {final_plot_path}")
    if INTERACTIVE_PLOT:
        plt.ioff() # 対話モード終了
        plt.show()
    else:
        plt.close(fig) # Figureを閉じてメモリ解放

    logging.info("Training process completed.")
    logging.info("="*30)


# ==============================================================================
# =                              Script Execution                            =
# ==============================================================================
if __name__ == "__main__":
    # 環境変数の設定 (オプション、メモリ断片化対策)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    train_model()