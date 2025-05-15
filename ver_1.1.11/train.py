# /ver_1.1.10/train.py
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
# from torch.amp import autocast, GradScaler # 自動混合精度 (AMP) 用 # この行をコメントアウトまたは削除
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import gc # ガベージコレクタ
import h5py # HDF5 ファイル操作用 ★追加
from torch.amp import autocast  # 自動混合精度 (AMP) のコンテキストマネージャー
from torch.cuda.amp import GradScaler  # CUDA用のGradScaler
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler # プロファイラ

# ===============================================================================
# =        Import Project Modules & Constants (エラーハンドリング強化)        =
# ===============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path: sys.path.insert(0, script_dir)
try:
    # game_state.py から必要な定数をインポート
    # ここでインポートが成功しても、naki_utils内でtile_utilsのインポートに失敗する可能性はある
    from game_state import NUM_TILE_TYPES, MAX_EVENT_HISTORY, EVENT_TYPES, STATIC_FEATURE_DIM
    logging.info(f"Imported constants: NUM_TILE_TYPES={NUM_TILE_TYPES}, MAX_EVENT_HISTORY={MAX_EVENT_HISTORY}, STATIC_FEATURE_DIM={STATIC_FEATURE_DIM}")
except ImportError as e:
     # インポート失敗は致命的なのでログに残し、フォールバック値を使用する警告を出す
     logging.critical(f"[FATAL ERROR in train.py] Cannot import constants from game_state.py: {e}")
     logging.critical("Ensure game_state.py (and its dependencies tile_utils.py, naki_utils.py) are in the same directory.")
     # フォールバック値は最低限の動作を保証するためのものであり、正確性は保証されません
     NUM_TILE_TYPES = 34         # 牌の種類 (萬子1-9, 筒子1-9, 索子1-9, 字牌7種)
     MAX_EVENT_HISTORY = 60      # Transformerに入力するイベント系列の最大長
     STATIC_FEATURE_DIM = 157    # 1サンプルあたりの静的特徴量の次元数
     EVENT_TYPES = {"PADDING": 8} # イベントタイプ辞書（最低限PADDINGが必要）
     logging.warning(f"[Warning] Using fallback constants due to import error: NUM_TILE_TYPES={NUM_TILE_TYPES}, MAX_EVENT_HISTORY={MAX_EVENT_HISTORY}, STATIC_FEATURE_DIM={STATIC_FEATURE_DIM}")
     # 実際にはsys.exit(1)で終了させるべきですが、デバッグのために続行を許可

# ===============================================================================
# =                           Configuration Settings                           =
# ===============================================================================
# --- データ関連 ---
# HDF5 ファイルのパスを指定
DATA_HDF5_PATH = "/home/ubuntu/Documents/mahjong-XAI/ver_1.1.10/training_data/mahjong_imitation_data_v1110.hdf5" # ★変更点: NPZパターンからHDF5パスへ
##/home/ubuntu/Documents/mahjong-XAI/ver_1.1.10/training_data
VALIDATION_SPLIT = 0.05                                 # データセットからバリデーション用に分割する割合

# --- モデル保存関連 ---
MODEL_SAVE_PATH = "./trained_model/mahjong_transformer_v1111_large_compiled.pth" # 最良モデルの保存パス (compile版を示す名前に変更)
CHECKPOINT_DIR = "./checkpoints_v1111_large_compiled/"       # チェックポイントの保存ディレクトリ (compile版を示す名前に変更)

# --- ログ・プロット関連 ---
LOG_DIR = "./logs"                                      # ログファイルの保存ディレクトリ
PLOT_DIR = "./plots_v1111_large_compiled"               # プロット画像の保存ディレクトリ (compile版を示す名前に変更)
PLOT_EVERY_EPOCH = 1                                    # 何エポックごとにプロットを更新・保存するか
INTERACTIVE_PLOT = False                                # プロットを対話的に表示するか (通常はFalse)
PROFILER_LOG_DIR = "./profiler_logs"                    # プロファイラログの出力先 ★追加

# --- トレーニングハイパーパラメータ ---
BATCH_SIZE = 2048           # 1回のパラメータ更新で使うサンプル数 (メモリに応じて調整)
NUM_EPOCHS = 125             # トレーニングを行う総エポック数
LEARNING_RATE = 5e-4        # 学習率の初期値
WEIGHT_DECAY = 0.05         # AdamWのWeight Decay (正則化)
CLIP_GRAD_NORM = 1.0        # 勾配クリッピングの上限値 (0以下で無効)
ACCUMULATION_STEPS = 2      # 勾配を累積するステップ数 (実質バッチサイズ = BATCH_SIZE * ACCUMULATION_STEPS)
                            # メモリ不足時にBATCH_SIZEを減らし、これを増やす

# --- Transformerモデルハイパーパラメータ ---
D_MODEL = 256               # モデル内部の基本次元数 ★変更点: ユーザー提供コードに合わせた
NHEAD = 4                   # Multi-Head Attentionのヘッド数 (D_MODELを割り切れる必要あり)
D_HID = 1024                # Transformer内部のFeedForward層の中間次元数 (通常 D_MODEL * 4)
NLAYERS = 4                 # Transformer Encoder Layerの数
DROPOUT = 0.1               # ドロップアウト率
ACTIVATION = 'relu'         # Transformer内部の活性化関数 ('relu' または 'gelu')

# --- 高度なトレーニング機能 ---
USE_AMP = True              # 自動混合精度 (AMP) を使用するか (GPUが対応していればTrue推奨)
USE_TORCH_COMPILE = True    # torch.compile を使用するか (PyTorch 2.0以降、大幅な高速化が期待できる)
COMPILE_MODE = "default"    # torch.compile のモード ('default', 'reduce-overhead', 'max-autotune')
USE_EMA = False             # Exponential Moving Average を使用するか (オプション)
EMA_DECAY = 0.999           # EMAの減衰率
LABEL_SMOOTHING = 0.1       # ラベルスムージングの度合い (0.0で無効)
PROFILE_DATALOADER_STEPS = 10 # DataLoaderのプロファイルを行うステップ数 (0で無効) ★追加

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
     logging.info(f"CUDA Device: {torch.cuda.get_device_name(DEVICE)}")
     logging.info(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
     logging.info(f"TF32 Matmul Precision: {torch.get_float32_matmul_precision()}")
     logging.info(f"BF16 Supported: {bf16_supported}")
else:
     DEVICE = torch.device("cpu")
     logging.warning("[Warning] CUDA not available, using CPU.")
     USE_AMP = False # CPUではAMPは無効
     USE_TORCH_COMPILE = False # CPUでのtorch.compileはまだ実験的
logging.info(f"Using device: {DEVICE}")


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
os.makedirs(PROFILER_LOG_DIR, exist_ok=True) # プロファイラログディレクトリ作成 ★追加

# ==============================================================================
# =                             Logging Setup                                =
# ==============================================================================
log_file_path = os.path.join(LOG_DIR, f"model_training_{os.path.basename(MODEL_SAVE_PATH).replace('.pth','')}.log")
# logging.basicConfig は一度しか設定できないため、既に設定されている場合はスキップ
if not logging.getLogger('').handlers:
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
        x_odd_rotated = x_even * sin_angles + x_odd * cos_angles

        # 回転後の次元を結合して元の形状に戻す
        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = x_even_rotated
        x_rotated[..., 1::2] = x_odd_rotated

        return x_rotated

# ==============================================================================
# =                            Profiler Function                               =
# ==============================================================================
# run_with_profiler 関数を train_model 関数より前に移動
def run_with_profiler(model, train_loader, device, log_dir=PROFILER_LOG_DIR, num_steps=20): # log_dir を設定から参照
    """
    最初の num_steps ステップだけプロファイラを動かして、
    TensorBoard で可視化できるログを出力するサンプル関数。
    """
    # プロファイルのスケジュール: 1ステップ待機 → 1ステップウォームアップ → num_stepsステップ記録 → 繰り返し
    schedule_obj = schedule(wait=1, warmup=1, active=num_steps, repeat=1)

    # on_trace_ready コールバック関数
    # 各ステップのトレースを TensorBoard 形式で保存
    def trace_handler(p):
        output_dir = os.path.join(log_dir, time.strftime("%Y%m%d_%H%M%S"))
        p.export_chrome_trace(os.path.join(output_dir, f"trace_{p.step_num}.json"))
        # p.export_stacks(os.path.join(output_dir, f"stacks_{p.step_num}.txt")) # オプション: スタックトレースも保存
        print(f"Profiler trace saved to {output_dir}")

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule_obj,
        on_trace_ready=trace_handler,
        record_shapes=True,       # 各 op の入出力サイズも記録
        profile_memory=True,      # メモリ使用量も記録
        with_stack=True           # スタックトレース付き
    )

    model.train()
    model.to(device)
    # オプティマイザはプロファイル対象外とするため、ここではダミーまたは省略
    # optimizer = optim.AdamW(model.parameters(), lr=1e-3) # 例: ダミーオプティマイザ

    print(f"Starting profiler for {num_steps} steps...")
    prof.start()

    # DataLoaderからデータを取得し、モデルをフォワードパスするループ
    # schedule の合計ステップ数 (wait + warmup + active) だけループを回す
    total_profiler_steps = schedule_obj.total_steps(1) # repeat=1 の場合
    try:
        for step, batch in enumerate(train_loader):
            if step >= total_profiler_steps:
                 break

            # record_function で特定のコードブロックを計測
            with record_function("DataLoader_to_Device"): # データ転送の計測
                 seq, static, labels, mask = batch
                 seq = seq.to(device, non_blocking=True)
                 static = static.to(device, non_blocking=True)
                 labels = labels.to(device, non_blocking=True) # ラベルも転送
                 mask = mask.to(device, non_blocking=True)

            with record_function("Forward_Pass"): # フォワードパスの計測
                 out = model(seq, static, mask)
                 # 損失計算もフォワードパスの一部として計測
                 # loss = nn.CrossEntropyLoss()(out, labels) # 例: ダミー損失計算

            # Backward Pass (勾配計算) も計測したい場合はここに記述
            # with record_function("Backward_Pass"):
            #     loss.backward()

            # オプティマイザステップやスケーラーステップはプロファイルの目的に応じて含めるか判断
            # optimizer.step() # 例: ダミー更新
            # optimizer.zero_grad() # 例: 勾配クリア

            prof.step() # プロファイラのステップを進める

    except Exception as e:
        print(f"Error during profiler run: {e}")
        import traceback
        traceback.print_exc()

    finally:
        prof.stop() # プロファイラを停止
        print("Profiler stopped.")
        print(f"Profiler trace written to {log_dir}. Use 'tensorboard --logdir {log_dir}' to view.")


# ==============================================================================
# =                            DataLoader Profiler                             =
# ==============================================================================
# profile_dataloader 関数を train_model 関数より前に移動
def profile_dataloader(loader, num_batches=10):
    """
    DataLoaderのデータ読み込み速度を計測する関数。
    GPU転送時間も含む。
    """
    if num_batches <= 0:
        logging.info("DataLoader profiling skipped (num_batches <= 0).")
        return

    logging.info(f"Profiling DataLoader for {num_batches} batches...")
    it = iter(loader)
    total_time = 0.0
    # GPUが利用可能な場合のみCUDAイベントを使用
    if DEVICE.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    # ウォームアップ
    logging.info("DataLoader profiling: Warming up...")
    for _ in range(min(5, num_batches)):
        try:
            # next(it) はCPUで実行されるデータローディング部分
            _ = next(it)
            if DEVICE.type == 'cuda':
                 # CPUからGPUへの転送完了を待つ
                 torch.cuda.synchronize()
        except StopIteration:
            logging.warning("DataLoader exhausted during warm-up.")
            return
        except Exception as e:
            logging.error(f"Error during DataLoader warm-up: {e}", exc_info=True)
            return

    logging.info(f"DataLoader profiling: Measuring {num_batches} batches...")
    start_time_cpu = time.time() # CPU側の時間計測も開始
    for i in tqdm(range(num_batches), desc="Profiling DataLoader", leave=False):
        try:
            if DEVICE.type == 'cuda':
                 start_event.record()

            batch = next(it) # CPUでのデータ読み込み
            # データをデバイスに転送（pin_memoryが有効なら非同期転送）
            seq, static, labels, mask = batch
            seq = seq.to(DEVICE, non_blocking=True)
            static = static.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)

            if DEVICE.type == 'cuda':
                 # GPUへの転送完了を待つ（pin_memory使用時は重要）
                 # または、GPUでの計算開始直前に同期する
                 # ここでは転送完了までを計測するため、明示的に待つ
                 torch.cuda.current_stream().synchronize()
                 end_event.record()
                 end_event.synchronize() # 計測終了イベントの完了を待つ
                 total_time += start_event.elapsed_time(end_event) / 1000.0 # ミリ秒 -> 秒

            # CPU側の時間計測はループの外で行うか、個別に計測する
            # ここではDataLoaderの__iter__と__next__にかかる時間 + to(DEVICE) の時間を計測したいので、
            # CPU時間とGPUイベント時間を併用する。
            # 簡単のため、ここではGPUイベント時間（転送完了まで）をメインとする。
            # CPU側の純粋な読み込み時間を見たい場合は、next(it) の前後でtime.time()を使う。

        except StopIteration:
            logging.warning(f"DataLoader exhausted after {i} batches during profiling.")
            num_batches = i # 実際に処理したバッチ数に更新
            break
        except Exception as e:
            logging.error(f"Error during DataLoader profiling batch {i}: {e}", exc_info=True)
            # エラーが発生したバッチは計測に含めない
            num_batches = i # 実際に処理したバッチ数に更新
            break

    end_time_cpu = time.time()
    total_time_cpu = end_time_cpu - start_time_cpu

    if num_batches > 0:
        if DEVICE.type == 'cuda':
             avg_time_per_batch = total_time / num_batches # GPUイベント時間での平均
             logging.info(f"DataLoader avg batch loading + transfer time (GPU Event) over {num_batches} batches: {avg_time_per_batch:.4f}s")
        else:
             avg_time_per_batch_cpu = total_time_cpu / num_batches # CPU時間での平均
             logging.info(f"DataLoader avg batch loading time (CPU Time) over {num_batches} batches: {avg_time_per_batch_cpu:.4f}s")

    else:
        logging.warning("DataLoader profiling failed or processed 0 batches.")


# ==============================================================================
# =                       Transformer Model Definition                       =
# ==============================================================================
# MahjongTransformerV2 クラス定義を train_model 関数より前に移動
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
class EMA(object): # object を継承 (古いスタイルだが互換性のため)
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
                # assert name in self.shadow # 登録されているか確認 - 初期化で登録済みのはず
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
                # assert name in self.backup # バックアップがあるか確認 - apply_shadowでバックアップ済みのはず
                param.data = self.backup[name]
        self.backup = {}

# ==============================================================================
# =                            Dataset Definition                            =
# ==============================================================================

class MahjongHdf5Dataset(Dataset):
    """HDF5 から直接シーケンスを読み込む Dataset (__getitem__でファイル開閉)"""
    def __init__(self, h5_path):
        self.h5_path = h5_path

        # ファイルを開かずにパスだけ保持し、メタデータ（長さ、次元）を取得
        try:
            with h5py.File(self.h5_path, "r", swmr=True) as hf:
                # データセットの長さを取得
                if "labels" not in hf:
                     raise KeyError(f"Required dataset 'labels' not found in {h5_path}")
                self.length = hf["labels"].shape[0]

                if self.length == 0:
                     raise RuntimeError(f"HDF5 file {h5_path} contains no samples.")

                # データ次元（最初のサンプルから取得）
                if "sequences" not in hf or "static_features" not in hf:
                      raise KeyError("Required datasets 'sequences' or 'static_features' not found.")
                first_seq = hf["sequences"][0]
                first_static = hf["static_features"][0]
                self.seq_len = first_seq.shape[0]
                self.event_dim = first_seq.shape[1]
                self.static_dim = first_static.shape[0]
                # 定数 STATIC_FEATURE_DIM との整合性チェック
                if self.static_dim != STATIC_FEATURE_DIM:
                     raise ValueError(f"Static dim mismatch in HDF5! Expected {STATIC_FEATURE_DIM}, got {self.static_dim} in {h5_path}")

        except FileNotFoundError:
             raise FileNotFoundError(f"HDF5 file not found at {h5_path}")
        except Exception as e:
            raise RuntimeError(f"Error determining dataset metadata from {h5_path}: {e}") from e

        # 定数 EVENT_TYPES["PADDING"] の確認
        # EVENT_TYPES がインポートされていることを確認
        if "PADDING" not in EVENT_TYPES:
             logging.critical("EVENT_TYPES dictionary missing 'PADDING' key. Using fallback 8.0.")
             self.padding_code = 8.0
        else:
             self.padding_code = float(EVENT_TYPES["PADDING"])

        logging.info(f"HDF5 Dataset initialized: {self.length} samples.")
        logging.info(f"Dims: SeqLen={self.seq_len}, Event={self.event_dim}, Static={self.static_dim}")


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not 0 <= idx < self.length:
            raise IndexError("Index out of bounds")

        try:
            # __getitem__ が呼ばれるたびにファイルを開閉する
            # DataLoader の num_workers > 0 で安全に動作させるための簡易策
            # SWMR=True であれば、読み込みは並列に行えるはず
            with h5py.File(self.h5_path, "r", swmr=True) as hf:
                # HDF5 からスライス取得 → numpy
                # データセットが存在するか再チェック (念のため)
                if "sequences" not in hf or "static_features" not in hf or "labels" not in hf:
                     raise KeyError(f"Required datasets missing in HDF5 file {self.h5_path} during __getitem__.")

                # データ読み込み
                seq_np    = hf["sequences"][idx]           # shape: (SeqLen, EventDim)
                static_np = hf["static_features"][idx]        # shape: (StaticDim,)
                label     = int(hf["labels"][idx])    # scalar

            # numpy → torch
            seq    = torch.from_numpy(seq_np).float()
            static = torch.from_numpy(static_np).float()
            label  = torch.tensor(label, dtype=torch.long)

            # パディングマスクの生成 (シーケンスの最初の特徴量がパディングコードかチェック)
            # イベントベクトルの最初の要素がイベントタイプコード
            padding_mask = (seq[:, 0] == self.padding_code) # (SeqLen,) の boolean Tensor

            # 念のため形状チェック (初期化時の次元と一致するはず)
            # self.event_dim などが -1 のままの場合は、初期化に失敗している可能性
            # 初期化で次元を取得しているので、ここでは取得した次元と比較
            expected_seq_shape = (self.seq_len, self.event_dim)
            expected_static_shape = (self.static_dim,)

            if seq.shape != expected_seq_shape or static.shape != expected_static_shape:
                logging.error(f"Shape mismatch for loaded sample {idx} from {self.h5_path}! "
                              f"Seq: {seq.shape} (expected {expected_seq_shape}), "
                              f"Static: {static.shape} (expected {expected_static_shape}). Returning zeros.")
                # エラー時はゼロ埋めデータを返す (トレーニングを止めないため)
                seq = torch.zeros(expected_seq_shape, dtype=torch.float32)
                static = torch.zeros(expected_static_shape, dtype=torch.float32)
                label = torch.tensor(0, dtype=torch.long) # ラベル0を返す
                padding_mask = torch.ones(self.seq_len, dtype=torch.bool) # 全てパディング扱い

            return seq, static, label, padding_mask

        except Exception as e:
             logging.error(f"CRITICAL Error loading sample {idx} from {self.h5_path}: {e}", exc_info=True)
             # トレーニングを止めないためにゼロデータを返す
             # この場合も次元が不明なので、初期化で取得した次元を使う
             safe_event_dim = getattr(self, 'event_dim', 6)
             safe_seq_len = getattr(self, 'seq_len', MAX_EVENT_HISTORY)
             safe_static_dim = getattr(self, 'static_dim', STATIC_FEATURE_DIM)
             logging.error(f"Returning zero data with assumed shapes: Seq=({safe_seq_len}, {safe_event_dim}), Static=({safe_static_dim},)")

             return torch.zeros((safe_seq_len, safe_event_dim), dtype=torch.float32), \
                    torch.zeros((safe_static_dim,), dtype=torch.float32), \
                    torch.tensor(0, dtype=torch.long), \
                    torch.ones((safe_seq_len,), dtype=torch.bool) # 全てパディング扱い


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
    global USE_TORCH_COMPILE # グローバル変数を変更する可能性があるため宣言

    """モデルのトレーニングを実行するメイン関数"""
    logging.info("Starting training process...")

    # --- HDF5 データファイルの確認 ---
    if not os.path.exists(DATA_HDF5_PATH):
        logging.error(f"HDF5 data file not found: {DATA_HDF5_PATH}")
        logging.error("Please run preprocess_data.py first to generate the HDF5 file, or check the DATA_HDF5_PATH setting.")
        return
    logging.info(f"Found HDF5 data file: {DATA_HDF5_PATH}")

    # --- データセットの初期化 ---
    try:
        # HDF5 Dataset を使用
        full_dataset = MahjongHdf5Dataset(DATA_HDF5_PATH)
    except (RuntimeError, ValueError, FileNotFoundError, KeyError) as e: # HDF5関連のエラーを追加
        logging.error(f"Failed to initialize dataset from {DATA_HDF5_PATH}: {e}")
        return

    # データセットから次元情報を取得
    # Dataset初期化時に取得済み
    event_dim = full_dataset.event_dim
    static_dim = full_dataset.static_dim
    seq_len = full_dataset.seq_len # Sequence lengthもDatasetから取得
    logging.info(f"Dataset dimensions: Event={event_dim}, Static={static_dim}, SeqLen={seq_len}")

    # --- データセットの分割 (Train / Validation) ---
    total_samples = len(full_dataset)
    val_size = int(total_samples * VALIDATION_SPLIT)
    train_size = total_samples - val_size

    # バリデーションセットが小さすぎる場合の調整
    if val_size < BATCH_SIZE * 2 and total_samples >= BATCH_SIZE * 4:
         val_size = max(1, int(total_samples * 0.01)) # 最低1サンプルか1%は確保
         train_size = total_samples - val_size
         logging.warning(f"Validation split too small (< {BATCH_SIZE*2}), adjusting to {val_size} samples.")

    if train_size <= 0 or val_size <= 0:
         logging.error(f"Invalid dataset split: Train={train_size}, Val={val_size}. Check dataset size ({total_samples}) and validation split ({VALIDATION_SPLIT}). Aborting training.")
         return

    # インデックスをシャッフルして分割
    indices = list(range(total_samples))
    random.seed(SEED) # 再現性のためにシードを設定
    random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Subset を使用してデータセットを分割
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    logging.info(f"Dataset split: Train samples={len(train_dataset)}, Validation samples={len(val_dataset)}")

    # --- DataLoaderの準備 ---
    # CPUコア数に基づいてワーカー数を決定 (多すぎるとオーバーヘッド増)
    # HDF5を__getitem__で開閉する場合、ワーカー数が多いとファイルI/Oがボトルネックになりやすい
    # まずは num_workers=0 や少ない数で試すのが良いかもしれない
    num_workers = min(os.cpu_count() // 2 if os.cpu_count() else 1, 8)
    logging.info(f"Setting up DataLoaders with {num_workers} workers...")
    pin_memory = (DEVICE.type == 'cuda') # GPU使用時のみ有効
    # prefetch_factor: ワーカーがメインプロセスに先行して準備しておくバッチ数
    # persistent_workers: エポック間でワーカープロセスを維持するか
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        prefetch_factor=max(2, num_workers*2) if num_workers>0 else None, # ワーカー数に応じて調整
        persistent_workers=(num_workers > 0), # ワーカープロセスを維持してオーバーヘッド削減
        drop_last=True # バッチサイズに満たない最後のバッチを捨てる
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, # バリデーションはシャッフル不要、バッチサイズは大きくても良い
        num_workers=num_workers, pin_memory=pin_memory,
        prefetch_factor=max(2, num_workers*2) if num_workers>0 else None,
        persistent_workers=(num_workers > 0)
    )
    logging.info(f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # DataLoader 作成後すぐ、データローディング速度をプロファイル
    # profile_dataloader 関数が train_model 関数より前に定義されていることを確認
    if PROFILE_DATALOADER_STEPS > 0:
        profile_dataloader(train_loader, num_batches=PROFILE_DATALOADER_STEPS)


    # --- モデル・オプティマイザ・スケジューラ・損失・スケーラー初期化 ---
    logging.info("Initializing model, optimizer, scheduler, loss, and scaler...")
    # モデル初期化時に Dataset から取得した次元情報を渡す
    # MahjongTransformerV2 クラスが train_model 関数より前に定義されていることを確認
    model = MahjongTransformerV2(event_feature_dim=event_dim,
                                 static_feature_dim=static_dim,
                                 d_model=D_MODEL, # 設定から参照
                                 nhead=NHEAD,
                                 d_hid=D_HID,
                                 nlayers=NLAYERS,
                                 dropout=DROPOUT,
                                 activation=ACTIVATION,
                                 output_dim=NUM_TILE_TYPES # 定数から参照
                                 ).to(DEVICE)
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
            USE_TORCH_COMPILE = False # コンパイル失敗時はフラグをFalseにする
    elif USE_TORCH_COMPILE:
        logging.warning("torch.compile requested but not available (requires PyTorch 2.0+).")
        USE_TORCH_COMPILE = False # 利用できない場合はフラグをFalseにする

    # EMA (オプション)
    # EMA クラスが train_model 関数より前に定義されていることを確認
    ema = EMA(model, decay=EMA_DECAY) if USE_EMA else None
    if USE_EMA: logging.info("Using Exponential Moving Average (EMA).")

    # 損失関数
    # LabelSmoothingLoss クラスが train_model 関数より前に定義されていることを確認
    criterion = (LabelSmoothingLoss(smoothing=LABEL_SMOOTHING, num_classes=NUM_TILE_TYPES)
                 if LABEL_SMOOTHING > 0 else nn.CrossEntropyLoss())
    logging.info(f"Using Loss: {type(criterion).__name__}"
                 + (f" (smoothing={LABEL_SMOOTHING})" if LABEL_SMOOTHING>0 else ""))

    # オプティマイザ
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY, betas=(0.9,0.98), eps=1e-6)
    logging.info(f"Using Optimizer: AdamW (LR={LEARNING_RATE}, WD={WEIGHT_DECAY})")

    # 学習率スケジューラ
    scheduler_t0 = max(1, NUM_EPOCHS)
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
    best_epoch = 0 # Best epoch index (0-based)
    epochs_without_improvement = 0

    if INTERACTIVE_PLOT:
        plt.ion()
    # init_plots 関数が train_model 関数より前に定義されていることを確認
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
                # データを指定されたデバイス (GPU) に転送
                seq = seq.to(DEVICE, non_blocking=pin_memory)
                static = static.to(DEVICE, non_blocking=pin_memory)
                labels = labels.to(DEVICE, non_blocking=pin_memory)
                mask = mask.to(DEVICE, non_blocking=pin_memory)
            except Exception as e:
                logging.error(f"Batch transfer error at epoch {epoch+1}, batch {i}: {e}", exc_info=True)
                continue # エラーが発生したバッチはスキップ

            with autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=scaler.is_enabled()):
                outputs = model(seq, static, mask)
                loss = criterion(outputs, labels)
                if ACCUMULATION_STEPS > 1:
                    loss = loss / ACCUMULATION_STEPS

            if torch.isnan(loss):
                logging.warning(f"NaN loss at epoch {epoch+1}, batch {i}. Skipping gradient update.")
                optimizer.zero_grad(set_to_none=True) # 念のため勾配をクリア
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
                # calculate_accuracy 関数が train_model 関数より前に定義されていることを確認
                acc1, acc3 = calculate_accuracy(outputs, labels)

            bs = labels.size(0)
            train_loss_accum += loss.item() * (ACCUMULATION_STEPS if ACCUMULATION_STEPS>1 else 1)
            train_acc_accum += acc1 * bs
            train_top3_accum += acc3 * bs
            num_train_samples += bs

            # プログレスバーに情報を表示 (現在のバッチではなく、累積平均を表示)
            # ロスはバッチ数で平均、精度はサンプル数で平均
            avg_loss_display = train_loss_accum / (i + 1)
            avg_acc_display = train_acc_accum / num_train_samples if num_train_samples > 0 else 0.0
            pbar.set_postfix({'Loss': f'{avg_loss_display:.4f}', 'Acc': f'{avg_acc_display:.3f}',
                              'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'})

        # エポック終了時の平均メトリクスを計算
        epoch_train_loss = train_loss_accum / len(train_loader) if len(train_loader) > 0 else 0.0 # ロスはバッチ数で平均
        epoch_train_acc = train_acc_accum / num_train_samples if num_train_samples > 0 else 0.0
        epoch_train_top3 = train_top3_accum / num_train_samples if num_train_samples > 0 else 0.0
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
                    logging.error(f"Val batch transfer error at epoch {epoch+1}: {e}", exc_info=True)
                    continue

                with autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=scaler.is_enabled()):
                    outputs = model(seq, static, mask)
                    loss = criterion(outputs, labels)

                if torch.isnan(loss):
                    logging.warning(f"NaN loss in validation epoch {epoch+1}. Skipping batch.")
                    continue

                acc1, acc3 = calculate_accuracy(outputs, labels)
                bs = labels.size(0)
                val_loss_accum += loss.item() * bs # サンプル数で重み付け
                val_acc_accum += acc1 * bs # サンプル数で重み付け
                val_top3_accum += acc3 * bs # サンプル数で重み付け
                num_val_samples += bs # 処理したサンプル数を加算
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{acc1:.3f}'})

        if ema: ema.restore()
        epoch_val_loss = val_loss_accum / num_val_samples if num_val_samples > 0 else 0.0
        epoch_val_acc = val_acc_accum / num_val_samples if num_val_samples > 0 else 0.0
        epoch_val_top3 = val_top3_accum / num_val_samples if num_val_samples > 0 else 0.0
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
        # update_plots 関数が train_model 関数より前に定義されていることを確認
        if (epoch+1) % PLOT_EVERY_EPOCH == 0 or epoch == NUM_EPOCHS-1:
            update_plots(fig, axs, epoch, metrics)

        # チェックポイント保存
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
        save_dict = {
            'epoch': epoch+1,
            # torch.compile を使っている場合、元のモデルの状態を保存するには ._orig_mod を使う
            'model_state_dict': model.state_dict() if not USE_TORCH_COMPILE else model._orig_mod.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_acc': epoch_val_acc,
            'event_dim': event_dim, # モデル再構築のため次元を保存
            'static_dim': static_dim,
            'seq_len': seq_len # モデル再構築のためシーケンス長を保存
        }
        if ema:
            # EMAのシャドウパラメータも保存
            save_dict['ema_state_dict'] = ema.shadow
        try:
            torch.save(save_dict, checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")
        except Exception as e:
             logging.error(f"Failed to save checkpoint {checkpoint_path}: {e}", exc_info=True)


        # --- 最良モデルの保存 & アーリーストッピング ---
        # バリデーションセットが0サンプルの場合はスキップ
        if num_val_samples > 0 and epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch + 1 # 1-based index
            epochs_without_improvement = 0
            # EMAを使用している場合はEMAのシャドウ重みを保存、そうでなければモデルの元の重みを保存
            save_model_state = ema.shadow if ema else (model._orig_mod.state_dict() if USE_TORCH_COMPILE else model.state_dict())
            best_model_save_dict = {
                 'model_state_dict': save_model_state,
                 'event_dim': event_dim, # 次元情報も一緒に保存
                 'static_dim': static_dim,
                 'seq_len': seq_len # シーケンス長も保存
            }
            try:
                torch.save(best_model_save_dict, MODEL_SAVE_PATH)
                logging.info(f"*** New best model saved at epoch {best_epoch} with Val Acc: {best_val_acc:.4f} to {MODEL_SAVE_PATH} ***")
            except Exception as e:
                 logging.error(f"Failed to save best model to {MODEL_SAVE_PATH}: {e}", exc_info=True)

        else:
            if num_val_samples > 0: # バリデーションサンプルがある場合のみカウント
                epochs_without_improvement += 1
                logging.info(f"Validation accuracy did not improve for {epochs_without_improvement} epoch(s). Best was {best_val_acc:.4f} at epoch {best_epoch}.")
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    logging.info(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
                    break # トレーニングループを終了
            else:
                 # バリデーションサンプルがない場合は早期終了のカウントはしないが、警告は出す
                 logging.warning(f"No validation samples processed in epoch {epoch+1}. Cannot evaluate for early stopping.")


    # --- トレーニング終了処理 ---
    total_duration = time.time() - total_start_time
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
    try:
        plt.savefig(final_plot_path)
        logging.info(f"Final training curves saved to: {final_plot_path}")
    except Exception as e:
        logging.warning(f"Failed to save final plot: {e}")

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
    # os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE' # 一部の環境でHDF5のロック問題を回避

    # プロファイラを実行したい場合は以下のコメントアウトを外す
    # 注意: プロファイラ実行中はトレーニングループは実行されません
    # print("\n--- Running Profiler ---")
    # # まずダミーのDataLoaderを作成し、Datasetから次元を取得する必要がある
    # # HDF5ファイルが先に生成されている前提
    # try:
    #      dummy_dataset = MahjongHdf5Dataset(DATA_HDF5_PATH)
    #      # プロファイラ時はワーカー0でシンプルに（マルチプロセスだとプロファイルが複雑になる）
    #      dummy_loader = DataLoader(dummy_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    #      # MahjongTransformerV2 クラスが train_model 関数より前に定義されていることを確認
    #      dummy_model = MahjongTransformerV2(event_feature_dim=dummy_dataset.event_dim, static_feature_dim=dummy_dataset.static_dim).to(DEVICE)
    #      # run_with_profiler 関数が train_model 関数より前に定義されていることを確認
    #      run_with_profiler(dummy_model, dummy_loader, DEVICE, num_steps=10) # プロファイルするステップ数を調整
    #      print("--- Profiler Finished ---")
    # except Exception as e:
    #      print(f"Error during profiler setup/run: {e}")
    #      import traceback
    #      traceback.print_exc()
    # print("\n--- Starting Training ---") # プロファイラ後にトレーニングを実行する場合

    train_model()