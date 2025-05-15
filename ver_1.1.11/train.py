# /ver_1.1.9/train.py
# train.py
# ===============================================================================
# =                              Import Libraries                              =
# ===============================================================================
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
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler


# ===============================================================================
# =        Import Project Modules & Constants (エラーハンドリング強化)        =
# ===============================================================================
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

# ===============================================================================
# =                           Configuration Settings                           =
# ===============================================================================
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
BATCH_SIZE = 1024           # 1回のパラメータ更新で使うサンプル数 (メモリに応じて調整)
NUM_EPOCHS = 100             # トレーニングを行う総エポック数
LEARNING_RATE = 5e-4        # 学習率の初期値
WEIGHT_DECAY = 0.05         # AdamWのWeight Decay (正則化)
CLIP_GRAD_NORM = 1.0        # 勾配クリッピングの上限値 (0以下で無効)
ACCUMULATION_STEPS = 6      # 勾配を累積するステップ数 (実質バッチサイズ = BATCH_SIZE * ACCUMULATION_STEPS)
                            # メモリ不足時にBATCH_SIZEを減らし、これを増やす

# --- Transformerモデルハイパーパラメータ ---
D_MODEL = 256               # モデル内部の基本次元数
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

# ===============================================================================
# =                             Device Configuration                           =
# ===============================================================================
bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
if torch.cuda.is_available():
     DEVICE = torch.device("cuda")
     torch.backends.cudnn.benchmark = True # cudnnの自動チューナーを有効化 (入力サイズが固定の場合に高速化)
     torch.set_float32_matmul_precision('high') # TF32の使用を設定 ('high' or 'medium')
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
# =                            Profiler Function                               =
# ==============================================================================
def run_with_profiler(model, train_loader, device, log_dir="./profiler_logs", num_steps=20):
    """
    最初の num_steps ステップだけプロファイラを動かして、
    TensorBoard で可視化できるログを出力するサンプル関数。
    """
    # プロファイルのスケジュール: 1ステップ待機 → 1ステップウォームアップ → 5ステップ記録 → 繰り返し
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=num_steps, repeat=1),
        on_trace_ready=tensorboard_trace_handler(log_dir),
        record_shapes=True,       # 各 op の入出力サイズも記録
        profile_memory=True,      # メモリ使用量も記録
        with_stack=True           # スタックトレース付き
    )
    model.train()
    model.to(device)
    optimizer = None  # dummy
    prof.start()
    for step, batch in enumerate(train_loader):
        if step >= num_steps * 3:  # schedule(wait+warmup+active) の合計ステップ数
            break
        seq, static, labels, mask = batch
        seq = seq.to(device, non_blocking=True)
        static = static.to(device, non_blocking=True)
        with record_function("forward_and_backward"):
            out = model(seq, static, mask.to(device))
            # backward は省略して forward だけでも GPU‐CPU 間を計測できます
        prof.step()
    prof.stop()
    print(f"Profiler trace written to {log_dir}")


# ==============================================================================
# =                            DataLoader Profiler                             =
# ==============================================================================
def profile_dataloader(loader, num_batches=10):
    import time
    it = iter(loader)
    total = 0.0
    for i in range(num_batches):
        t0 = time.time()
        _ = next(it)
        torch.cuda.synchronize()  # GPU→CPU同期（pin_memory の転送含む）
        t1 = time.time()
        total += (t1 - t0)
    print(f"DataLoader avg batch time: {total/num_batches:.4f}s")



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
        # EVENT_TYPES がインポートされていることを確認
        if "PADDING" not in EVENT_TYPES:
             logging.critical("EVENT_TYPES dictionary missing 'PADDING' key. Using fallback 8.")
             self.padding_code = 8.0
        else:
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
                if not os.path.exists(f) or os.path.getsize(f) < 100:
                    if not os.path.exists(f): logging.debug(f"Skipping missing file: {f}")
                    else: logging.debug(f"Skipping small file (<100B): {f}")
                    skipped_count += 1; continue

                # NPZファイルを開いて中身を確認
                with np.load(f, allow_pickle=False) as data:
                    # 必要なキーが存在するか確認
                    if not all(k in data for k in [self.sequences_field, self.static_field, self.labels_field]):
                        logging.warning(f"Skipping {f} - missing required keys ({self.sequences_field}, {self.static_field}, {self.labels_field}). Found keys: {list(data.keys())}")
                        skipped_count += 1; continue
                    # サンプル数を取得
                    length = len(data[self.labels_field])
                    if length == 0:
                        logging.debug(f"Skipping empty file: {f}")
                        skipped_count += 1; continue # 空のファイルはスキップ

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
        # cumulative_lengths は [0, len(file1), len(file1)+len(file2), ...]
        # searchsorted(idx, side='right') は idx より大きい最初の要素のインデックスを返す
        # 例: idx=5, cumulative_lengths=[0, 10, 25]. searchsorted(5) -> 1. file_idx = 1-1 = 0. Correct.
        # 例: idx=10, cumulative_lengths=[0, 10, 25]. searchsorted(10, side='right') -> 2. file_idx = 2-1 = 1. Correct.
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
            # イベントベクトルの最初の要素がイベントタイプコード
            padding_mask = (seq[:, 0] == self.padding_code) # (SeqLen,) の boolean Tensor

            # 念のため形状をチェック (初期化時の次元と一致するはず)
            # self.event_dim が -1 のままの場合は、初期化に失敗している可能性
            if self.seq_len == -1 or self.event_dim == -1 or self.static_dim == -1:
                 logging.critical(f"Dataset dimensions not initialized correctly. Cannot check sample shape for idx {idx}.")
                 # この場合はゼロ埋めデータを返すしかない
                 return np.zeros((MAX_EVENT_HISTORY, 6), dtype=np.float32), \
                        np.zeros((STATIC_FEATURE_DIM,), dtype=np.float32), \
                        np.zeros((), dtype=np.int64), \
                        np.ones((MAX_EVENT_HISTORY,), dtype=bool)

            if seq.shape != (self.seq_len, self.event_dim) or static.shape != (self.static_dim,):
                logging.error(f"Shape mismatch for loaded sample {idx} (offset {offset}) from file {file_idx} ({file_info['path']})! "
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
            # この場合も次元が不明なので、安全なデフォルト次元を使用
            safe_event_dim = 6 # GameState._add_event の基本次元 + 最大特定次元 (4+2=6) を仮定
            safe_seq_len = MAX_EVENT_HISTORY
            safe_static_dim = STATIC_FEATURE_DIM
            logging.error(f"Returning zero data with assumed shapes: Seq=({safe_seq_len}, {safe_event_dim}), Static=({safe_static_dim},)")

            return np.zeros((safe_seq_len, safe_event_dim), dtype=np.float32), \
                   np.zeros((safe_static_dim,), dtype=np.float32), \
                   np.zeros((), dtype=np.int64), \
                   np.ones((safe_seq_len,), dtype=bool) # 全てパディング扱い

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
            # num_classes が1以下の場合は割り算でエラーになる可能性があるのでチェック
            if self.num_classes > 1:
                true_dist.fill_(self.smoothing / (self.num_classes - 1))
            else: # クラスが1つしかない場合はスムージング無効
                 true_dist.fill_(0.0)
                 self.confidence = 1.0 # スムージング無効なのでconfidenceは1.0

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
        _, top3_indices = torch.topk(predictions, min(3, predictions.size(1)), dim=1) # クラス数が3未満の場合はmin(3, num_classes)
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
    global USE_TORCH_COMPILE

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
        logging.error("Invalid feature dimensions determined from dataset. Aborting training.")
        return
    event_dim = full_dataset.event_dim
    static_dim = full_dataset.static_dim
    logging.info(f"Dataset dimensions: Event={event_dim}, Static={static_dim}, SeqLen={full_dataset.seq_len}")

    # --- データセットの分割 (Train / Validation) ---
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    if val_size < BATCH_SIZE * 2 and len(full_dataset) >= BATCH_SIZE * 4:
        val_size = max(1, int(len(full_dataset) * 0.01))
        train_size = len(full_dataset) - val_size
        logging.warning(f"Validation split too small, adjusting to {val_size} samples.")
    if train_size <= 0 or val_size <= 0:
        logging.error(f"Invalid split: Train={train_size}, Val={val_size}. Aborting.")
        return

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    logging.info(f"Dataset split: Train samples={train_size}, Validation samples={val_size}")

    # --- DataLoaderの準備 ---
    num_workers = min(os.cpu_count() // 2 if os.cpu_count() else 1, 8)
    logging.info(f"Setting up DataLoaders with {num_workers} workers...")
    pin_memory = (DEVICE.type == 'cuda')
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        prefetch_factor=max(2, num_workers*2) if num_workers>0 else None,
        persistent_workers=(num_workers>0), drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE*2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        prefetch_factor=max(2, num_workers*2) if num_workers>0 else None,
        persistent_workers=(num_workers>0)
    )
    logging.info(f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # DataLoader 作成後すぐ
    profile_dataloader(train_loader, num_batches=5)

    # --- モデル・オプティマイザ・スケジューラ・損失・スケーラー初期化 ---
    logging.info("Initializing model, optimizer, scheduler, loss, and scaler...")
    model = MahjongTransformerV2(event_feature_dim=event_dim,
                                 static_feature_dim=static_dim).to(DEVICE)
    logging.info(f"Model Parameters (Total): {sum(p.numel() for p in model.parameters()):,}")
    logging.info(f"Model Parameters (Trainable): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # torch.compile 適用
    if USE_TORCH_COMPILE and hasattr(torch, 'compile'):
        logging.info(f"Applying torch.compile with mode='{COMPILE_MODE}'...")
        try:
            model = torch.compile(model, mode=COMPILE_MODE)
            logging.info("torch.compile applied successfully.")
        except Exception as e:
            logging.warning(f"torch.compile failed: {e}. Proceeding without compile.", exc_info=True)
            USE_TORCH_COMPILE = False
    elif USE_TORCH_COMPILE:
        logging.warning("torch.compile requested but not available (requires PyTorch 2.0+).")
        USE_TORCH_COMPILE = False

    # EMA (オプション)
    ema = EMA(model, decay=EMA_DECAY) if USE_EMA else None
    if USE_EMA: logging.info("Using Exponential Moving Average (EMA).")

    # 損失関数
    criterion = (LabelSmoothingLoss(smoothing=LABEL_SMOOTHING, num_classes=NUM_TILE_TYPES)
                 if LABEL_SMOOTHING > 0 else nn.CrossEntropyLoss())
    logging.info(f"Using Loss: {type(criterion).__name__}"
                 + (f" (smoothing={LABEL_SMOOTHING})" if LABEL_SMOOTHING>0 else ""))

    # オプティマイザ
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY, betas=(0.9,0.98), eps=1e-6)
    logging.info(f"Using Optimizer: AdamW (LR={LEARNING_RATE}, WD={WEIGHT_DECAY})")

    # 学習率スケジューラ
    scheduler_t0 = max(1, NUM_EPOCHS//5)
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=scheduler_t0, T_mult=1, eta_min=LEARNING_RATE/100)
    logging.info(f"Using LR Scheduler: CosineAnnealingWarmRestarts (T_0={scheduler_t0})")

    # AMP スケーラー
    scaler = GradScaler(enabled=(USE_AMP and DEVICE.type=='cuda'))
    logging.info(f"Automatic Mixed Precision (AMP) {'Enabled' if scaler.is_enabled() else 'Disabled'}")
    amp_dtype = torch.bfloat16 if bf16_supported else torch.float16
    if scaler.is_enabled():
        logging.info(f"AMP dtype: {amp_dtype}")

    # メトリクス追跡
    metrics = {k: [] for k in ['train_loss','val_loss','train_acc','val_acc','train_top3','val_top3','lr']}
    best_val_acc = 0.0
    epochs_without_improvement = 0

    if INTERACTIVE_PLOT:
        plt.ion()
    fig, axs = init_plots()

    logging.info(f"Starting training: {NUM_EPOCHS} epochs on {DEVICE}")
    total_start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        # ============ トレーニングフェーズ ============
        model.train()
        train_loss_accum = train_acc_accum = train_top3_accum = 0.0
        num_train_samples = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
        optimizer.zero_grad(set_to_none=True)

        for i, (seq, static, labels, mask) in enumerate(pbar):
            try:
                seq = seq.to(DEVICE, non_blocking=pin_memory)
                static = static.to(DEVICE, non_blocking=pin_memory)
                labels = labels.to(DEVICE, non_blocking=pin_memory)
                mask = mask.to(DEVICE, non_blocking=pin_memory)
            except Exception as e:
                logging.error(f"Batch transfer error: {e}", exc_info=True)
                continue

            with autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=scaler.is_enabled()):
                outputs = model(seq, static, mask)
                loss = criterion(outputs, labels)
                if ACCUMULATION_STEPS > 1:
                    loss = loss / ACCUMULATION_STEPS

            if torch.isnan(loss):
                logging.warning(f"NaN loss at epoch {epoch+1}, batch {i}")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                if CLIP_GRAD_NORM > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema:
                    ema.update()

            with torch.no_grad():
                acc1, acc3 = calculate_accuracy(outputs, labels)

            bs = labels.size(0)
            train_loss_accum += loss.item() * (ACCUMULATION_STEPS if ACCUMULATION_STEPS>1 else 1)
            train_acc_accum += acc1 * bs
            train_top3_accum += acc3 * bs
            num_train_samples += bs

            avg_loss = train_loss_accum / len(train_loader)
            avg_acc = train_acc_accum / num_train_samples if num_train_samples else 0.0
            pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Acc': f'{avg_acc:.3f}',
                              'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'})

        # エポックごとの平均メトリクス記録
        epoch_train_loss = train_loss_accum / len(train_loader) if num_train_samples else 0.0
        epoch_train_acc = train_acc_accum / num_train_samples if num_train_samples else 0.0
        epoch_train_top3 = train_top3_accum / num_train_samples if num_train_samples else 0.0
        metrics['train_loss'].append(epoch_train_loss)
        metrics['train_acc'].append(epoch_train_acc)
        metrics['train_top3'].append(epoch_train_top3)
        metrics['lr'].append(optimizer.param_groups[0]['lr'])

        # ============ バリデーションフェーズ ============
        model.eval()
        if ema: ema.apply_shadow()
        val_loss_accum = val_acc_accum = val_top3_accum = 0.0
        num_val_samples = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=False)
        with torch.no_grad():
            for seq, static, labels, mask in val_pbar:
                try:
                    seq = seq.to(DEVICE, non_blocking=pin_memory)
                    static = static.to(DEVICE, non_blocking=pin_memory)
                    labels = labels.to(DEVICE, non_blocking=pin_memory)
                    mask = mask.to(DEVICE, non_blocking=pin_memory)
                except Exception as e:
                    logging.error(f"Val batch transfer error: {e}", exc_info=True)
                    continue

                with autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=scaler.is_enabled()):
                    outputs = model(seq, static, mask)
                    loss = criterion(outputs, labels)

                if torch.isnan(loss):
                    logging.warning(f"NaN loss in validation epoch {epoch+1}")
                    continue

                acc1, acc3 = calculate_accuracy(outputs, labels)
                bs = labels.size(0)
                val_loss_accum += loss.item() * bs
                val_acc_accum += acc1 * bs
                val_top3_accum += acc3 * bs
                num_val_samples += bs
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{acc1:.3f}'})

        if ema: ema.restore()
        epoch_val_loss = val_loss_accum / num_val_samples if num_val_samples else 0.0
        epoch_val_acc = val_acc_accum / num_val_samples if num_val_samples else 0.0
        epoch_val_top3 = val_top3_accum / num_val_samples if num_val_samples else 0.0
        metrics['val_loss'].append(epoch_val_loss)
        metrics['val_acc'].append(epoch_val_acc)
        metrics['val_top3'].append(epoch_val_top3)

        epoch_duration = time.time() - epoch_start_time
        logging.info(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Time: {epoch_duration:.1f}s - "
            f"Train L:{epoch_train_loss:.4f} A:{epoch_train_acc:.4f} T3:{epoch_train_top3:.4f} | "
            f"Val L:{epoch_val_loss:.4f} A:{epoch_val_acc:.4f} T3:{epoch_val_top3:.4f}"
        )

        lr_scheduler.step()
        if (epoch+1) % PLOT_EVERY_EPOCH == 0 or epoch == NUM_EPOCHS-1:
            update_plots(fig, axs, epoch, metrics)

        # チェックポイント保存
        ckpt = {
            'epoch': epoch+1,
            'model_state_dict': (model._orig_mod.state_dict() if USE_TORCH_COMPILE else model.state_dict()),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_acc': epoch_val_acc,
            'event_dim': event_dim,
            'static_dim': static_dim
        }
        if ema:
            ckpt['ema_state_dict'] = ema.shadow
        torch.save(ckpt, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth"))

        # ベストモデル＆アーリーストップ
        if num_val_samples and epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            save_state = (ema.shadow if ema else
                          (model._orig_mod.state_dict() if USE_TORCH_COMPILE else model.state_dict()))
            torch.save({
                'model_state_dict': save_state,
                'event_dim': event_dim,
                'static_dim': static_dim
            }, MODEL_SAVE_PATH)
            epochs_without_improvement = 0
        else:
            if num_val_samples:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    logging.info("Early stopping triggered.")
                    break

    # --- 終了処理 ---
    total_duration = time.time() - total_start_time
    logging.info(f"Training completed in {total_duration/60:.2f} minutes. Best Val Acc: {best_val_acc:.4f}")

    update_plots(fig, axs, epoch, metrics)
    final_plot_path = os.path.join(PLOT_DIR, 'final_training_curves.png')
    fig.savefig(final_plot_path)
    plt.close(fig)


# ==============================================================================
# =                              Script Execution                            =
# ==============================================================================
if __name__ == "__main__":
    # 環境変数の設定 (オプション、メモリ断片化対策)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    train_model()