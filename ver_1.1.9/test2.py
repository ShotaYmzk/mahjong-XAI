import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import platform # OS情報取得のため
from tqdm import tqdm # tqdmをインポート

print("--- PyTorch GPU Check with CNN ---")
print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")
print(f"OS: {platform.system()} {platform.release()}")

# --- 1. デバイスの設定 ---
bf16_supported = False # デフォルト値
amp_dtype = torch.float32 # デフォルト値
USE_AMP = False # デフォルト値

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU is available! Using device: {device}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    try:
        # 簡単なテンソル演算を試す
        x = torch.tensor([100, 200]).to(device)
        y = x * 2
        print(f"Simple GPU tensor operation successful: {y}")
        # 可能であればBF16のサポートを確認
        bf16_supported = torch.cuda.is_bf16_supported()
        print(f"BF16 Supported: {bf16_supported}")
        # 可能であればAMPを使ってみる（必須ではない）
        USE_AMP = True
        amp_dtype = torch.bfloat16 if bf16_supported else torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
        print(f"AMP Enabled: {USE_AMP}, Dtype: {amp_dtype}")
    except Exception as e:
        print(f"[ERROR] Could not perform simple GPU operation or check capabilities: {e}")
        print("Falling back to CPU.")
        device = torch.device("cpu")
        USE_AMP = False # CPUではAMP無効
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP) # Dummy scaler
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU.")
    USE_AMP = False
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP) # Dummy scaler

# --- 2. データセットの準備 (MNIST) ---
print("\n--- Preparing MNIST Dataset ---")
try:
    # 画像データをテンソルに変換し、正規化する
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNISTの標準的な平均・標準偏差
    ])

    # 訓練データセットのダウンロードと読み込み
    print("Downloading/Loading training data...")
    # GPUメモリに余裕があればバッチサイズを増やす (e.g., 128 or 256)
    BATCH_SIZE = 128
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    # DataLoaderのnum_workersも適切に設定
    # Windowsでは0が良い場合が多い、LinuxではCPUコア数に応じて増やす
    num_workers = 2 if platform.system() != 'Windows' else 0
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)

    print(f"Dataset prepared successfully. Batch size: {BATCH_SIZE}, Num workers: {num_workers}")
except Exception as e:
    print(f"[ERROR] Failed to prepare dataset: {e}")
    exit() # データセットが準備できない場合は終了

# --- 3. 少し大規模なモデルの定義 (CNN) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 畳み込み層 (Convolutional Layers)
        self.conv_block1 = nn.Sequential(
            # Input: (N, 1, 28, 28)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1), # Output: (N, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: (N, 32, 14, 14)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # Output: (N, 64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: (N, 64, 7, 7)
        )
        # 全結合層 (Fully Connected Layers)
        self.fc_block = nn.Sequential(
            nn.Flatten(), # Input: (N, 64 * 7 * 7) = (N, 3136)
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # Dropoutを追加して過学習を抑制
            nn.Linear(128, 10) # Output: (N, 10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.fc_block(x)
        return x

model = SimpleCNN().to(device) # モデルを定義し、指定したデバイスに送る
print(f"\nModel loaded on device: {next(model.parameters()).device}")
# モデルのパラメータ数を表示
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model Parameters: {num_params:,}")

# --- 4. 損失関数とオプティマイザ ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 5. 簡単な訓練ループ (AMP対応) ---
num_epochs = 100 # 少しだけエポック数を増やす
print(f"\n--- Starting CNN Training ({num_epochs} epochs) ---")
start_time = time.time()

for epoch in range(num_epochs):
    running_loss = 0.0
    num_batches = 0
    model.train() # モデルを訓練モードに設定

    # tqdmを使って進捗を表示
    pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch + 1}/{num_epochs}")

    for i, data in pbar:
        # データを取得し、デバイスに送る
        inputs, labels = data
        # non_blocking=True は pin_memory=True と合わせて使うと効果的
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # オプティマイザの勾配をリセット
        optimizer.zero_grad(set_to_none=True) # set_to_none=Trueでメモリ効率向上

        # AMPコンテキスト内でフォワードパスと損失計算
        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=amp_dtype if device.type=='cuda' else torch.float32):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # スケーラーを使ってバックワードパス
        scaler.scale(loss).backward()

        # スケーラーを使ってオプティマイザステップ
        scaler.step(optimizer)

        # スケーラーの状態を更新
        scaler.update()

        running_loss += loss.item()
        num_batches += 1

        # tqdmの表示を更新 (修正箇所)
        # 計算結果を先に求めてからf-stringでフォーマットする
        current_avg_loss = running_loss / num_batches
        pbar.set_postfix({'Loss': f'{current_avg_loss:.4f}'}) # f-stringで変数を指定

    # エポックごとの平均損失
    epoch_loss = running_loss / num_batches
    # エポック終了時の損失をprint (tqdmのバーが消えるので)
    print(f'\nEpoch {epoch + 1} finished. Average Loss: {epoch_loss:.4f}') # 改行を追加

end_time = time.time()
total_time = end_time - start_time
print(f"Training finished in {total_time:.2f} seconds.")
if num_epochs > 0:
    print(f"Average time per epoch: {total_time / num_epochs:.2f} seconds.")


# --- 6. 結果の表示 ---
print("\n--- Final Check ---")
if device.type == 'cuda':
    print("✅ GPU was successfully used for CNN training!")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Final Loss (approx): {epoch_loss:.4f}")
    # 使用したメモリ情報を表示 (あくまで目安)
    # torch.cuda.max_memory_allocated はPyTorch 1.1以降で利用可能
    if hasattr(torch.cuda, 'max_memory_allocated'):
        print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
    if hasattr(torch.cuda, 'max_memory_reserved'):
        print(f"Max Memory Reserved: {torch.cuda.max_memory_reserved(device) / 1024**2:.2f} MB")
    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
        torch.cuda.reset_peak_memory_stats(device) # 次回のためにリセット
else:
    print("⚠️ Training was performed on the CPU.")
    print(f"Final Loss (approx): {epoch_loss:.4f}")

print("\nIf 'GPU is available!' was printed and no errors occurred during training,")
print("your GPU environment for PyTorch is likely set up correctly for more complex models.")
print("Consider increasing BATCH_SIZE or model complexity further if your GPU has more memory.")