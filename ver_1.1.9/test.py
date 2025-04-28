import torch
import time
import math

print("PyTorch CUDA Information:")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("CUDA is not available in your PyTorch installation")
    exit(1)

# デバイス情報
print(f"  CUDA version: {torch.version.cuda}")
print(f"  Current CUDA device: {torch.cuda.current_device()}")
print(f"  Device name: {torch.cuda.get_device_name()}")

# メモリ状況
total_memory = torch.cuda.get_device_properties(0).total_memory
allocated_before = torch.cuda.memory_allocated(0)
print("\nBefore stress:")
print(f"  Total GPU memory:    {total_memory/1024**3:.2f} GB")
print(f"  Allocated before:    {allocated_before/1024**3:.2f} GB")

# ───────────────────────────────────────────────────────────
# 1) メモリを先に確保して負荷をかける
#    → 全体の50%を占める大きさのテンソルを作成
# ───────────────────────────────────────────────────────────
fill_bytes = int(total_memory * 0.9)
# float32 1要素4バイト なので要素数 = fill_bytes / 4
num_elems = fill_bytes // 4
fill_dim = int(math.sqrt(num_elems))
print(f"\nAllocating a {fill_dim}×{fill_dim} tensor (~50% GPU memory)...")
fill_tensor = torch.randn(fill_dim, fill_dim, device='cuda')
torch.cuda.synchronize()

allocated_mid = torch.cuda.memory_allocated(0)
print(f"  Allocated after fill: {allocated_mid/1024**3:.2f} GB")

# ───────────────────────────────────────────────────────────
# 2) 約20秒間、繰り返し大きめの行列演算を実行
# ───────────────────────────────────────────────────────────
stress_duration = 20.0
mat_dim = 6000  # 6K×6K 行列で回してみる
print(f"\nStarting stress test: {stress_duration:.1f} seconds of matmul({mat_dim}×{mat_dim}) loops...")
start_time = time.time()
iter_count = 0

while time.time() - start_time < stress_duration:
    # 毎回新規に大きな乱数行列を作って matmul
    a = torch.randn(mat_dim, mat_dim, device='cuda')
    b = torch.randn(mat_dim, mat_dim, device='cuda')
    c = torch.matmul(a, b)
    torch.cuda.synchronize()  # 演算完了を待つ
    iter_count += 1

end_time = time.time()
elapsed = end_time - start_time
print(f"Completed {iter_count} iterations in {elapsed:.2f} seconds")
print(f"Average time per matmul: {elapsed/iter_count:.4f} s")

# ───────────────────────────────────────────────────────────
# 3) 終了後のメモリ状況とキャッシュ開放
# ───────────────────────────────────────────────────────────
allocated_after = torch.cuda.memory_allocated(0)
print(f"\nAllocated after stress: {allocated_after/1024**3:.2f} GB")
torch.cuda.empty_cache()
print(f"Allocated after empty_cache: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
