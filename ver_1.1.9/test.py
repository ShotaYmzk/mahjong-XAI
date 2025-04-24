# Add this at the top of your training script after imports
import torch
import time

print("PyTorch CUDA Information:")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    
    # GPU使用率とメモリ使用量の確認
    print("\nGPU Utilization Test:")
    # GPUメモリの総量と空き容量を確認
    total_memory = torch.cuda.get_device_properties(0).total_memory
    reserved_memory = torch.cuda.memory_reserved(0)
    allocated_memory = torch.cuda.memory_allocated(0)
    free_memory = total_memory - allocated_memory
    
    print(f"Total GPU memory: {total_memory/1024**3:.2f} GB")
    print(f"Reserved GPU memory: {reserved_memory/1024**3:.2f} GB")
    print(f"Allocated GPU memory: {allocated_memory/1024**3:.2f} GB")
    print(f"Free GPU memory: {free_memory/1024**3:.2f} GB")
    
    # GPUの使用率をテスト - 大きな行列を作成して演算する
    print("\nTesting GPU performance with matrix operations...")
    
    # ウォームアップ
    x = torch.randn(5000, 5000, device='cuda')
    y = torch.randn(5000, 5000, device='cuda')
    torch.matmul(x, y)
    torch.cuda.synchronize()
    
    # パフォーマンステスト
    start_time = time.time()
    for _ in range(10):
        x = torch.randn(5000, 5000, device='cuda')
        y = torch.randn(5000, 5000, device='cuda')
        z = torch.matmul(x, y)
        torch.cuda.synchronize()  # GPU処理の完了を待つ
    end_time = time.time()
    
    print(f"Matrix multiplication time: {(end_time-start_time)/10:.4f} seconds per operation")
    print("If the test completed successfully, your GPU is working and can be utilized for ML tasks")
    
    # テスト後のメモリ使用状況
    allocated_memory_after = torch.cuda.memory_allocated(0)
    print(f"GPU memory allocated after test: {allocated_memory_after/1024**3:.2f} GB")
    
    # メモリをクリア
    torch.cuda.empty_cache()
    print(f"GPU memory after clearing cache: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
else:
    print("CUDA is not available in your PyTorch installation")