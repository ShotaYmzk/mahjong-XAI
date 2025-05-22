import os
import glob
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime

def get_absolute_path(relative_path):
    """相対パスを絶対パスに変換"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, relative_path)

def load_checkpoint_metrics(checkpoint_path):
    """チェックポイントからメトリクスを読み込む"""
    try:
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("Available keys in checkpoint:", list(checkpoint.keys()))
        
        # チェックポイントの内容を詳細に表示
        print("\nCheckpoint contents:")
        for key, value in checkpoint.items():
            if isinstance(value, dict):
                print(f"{key}: dict with keys {list(value.keys())}")
                if key == 'state_dict':
                    print("State dict keys:", list(value.keys())[:10], "...")
            elif isinstance(value, list):
                print(f"{key}: list of length {len(value)}")
            else:
                print(f"{key}: {type(value)}")
        
        # メトリクスの取得を試みる
        metrics = {}
        if 'metrics' in checkpoint:
            # train2.pyの形式: メトリクスが直接保存されている
            metrics = checkpoint['metrics']
            print("\nFound metrics in checkpoint['metrics']")
        elif 'state_dict' in checkpoint:
            # モデルの状態辞書からメトリクスを抽出
            state_dict = checkpoint['state_dict']
            print("\nTrying to extract metrics from state_dict")
            metrics = {
                'train_loss': [state_dict.get('train_loss', 0.0)],
                'val_loss': [state_dict.get('val_loss', 0.0)],
                'train_acc': [state_dict.get('train_acc', 0.0)],
                'val_acc': [state_dict.get('val_acc', 0.0)],
                'train_top3': [state_dict.get('train_top3', 0.0)],
                'val_top3': [state_dict.get('val_top3', 0.0)],
                'lr': [state_dict.get('lr', 0.0)]
            }
        elif 'train_loss' in checkpoint:
            # チェックポイント自体がメトリクスを含んでいる場合
            print("\nFound metrics directly in checkpoint")
            metrics = {
                'train_loss': [checkpoint.get('train_loss', 0.0)],
                'val_loss': [checkpoint.get('val_loss', 0.0)],
                'train_acc': [checkpoint.get('train_acc', 0.0)],
                'val_acc': [checkpoint.get('val_acc', 0.0)],
                'train_top3': [checkpoint.get('train_top3', 0.0)],
                'val_top3': [checkpoint.get('val_top3', 0.0)],
                'lr': [checkpoint.get('lr', 0.0)]
            }
        elif 'model' in checkpoint:
            # 新しい形式のチェックポイントの場合
            print("\nFound new format checkpoint with 'model' key")
            model_state = checkpoint['model']
            if isinstance(model_state, dict):
                metrics = {
                    'train_loss': [model_state.get('train_loss', 0.0)],
                    'val_loss': [model_state.get('val_loss', 0.0)],
                    'train_acc': [model_state.get('train_acc', 0.0)],
                    'val_acc': [model_state.get('val_acc', 0.0)],
                    'train_top3': [model_state.get('train_top3', 0.0)],
                    'val_top3': [model_state.get('val_top3', 0.0)],
                    'lr': [model_state.get('lr', 0.0)]
                }
        
        epoch = checkpoint.get('epoch', 0)
        val_acc = checkpoint.get('val_acc', 0.0)
        
        # メトリクスの存在確認とデバッグ出力
        print(f"\nLoading metrics from epoch {epoch}:")
        print("Available keys in metrics:", list(metrics.keys()))
        
        # 各メトリクスの値を確認
        for key in ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_top3', 'val_top3', 'lr']:
            if key in metrics:
                values = metrics[key]
                if isinstance(values, list) and len(values) > 0:
                    print(f"{key}: {len(values)} values, first value: {values[0]:.4f}, last value: {values[-1]:.4f}")
                else:
                    print(f"{key}: Invalid data format or empty list")
            else:
                print(f"{key}: Not found in metrics")
        
        return metrics, epoch, val_acc
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, 0.0

def plot_training_curves(checkpoint_dir, save_dir=None):
    """チェックポイントから学習曲線をプロット"""
    # パスの確認
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for checkpoints in: {checkpoint_dir}")
    
    # チェックポイントを探す
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        print("Directory contents:")
        try:
            print(os.listdir(checkpoint_dir))
        except Exception as e:
            print(f"Error listing directory: {e}")
        return

    # エポック番号でソート
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f"Found {len(checkpoints)} checkpoints")
    
    # すべてのチェックポイントからメトリクスを読み込む
    all_metrics = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
        'train_top3': [], 'val_top3': [], 'lr': []
    }
    final_epoch = 0
    
    for checkpoint_path in checkpoints:
        metrics, epoch, _ = load_checkpoint_metrics(checkpoint_path)
        if metrics is None:
            continue
            
        final_epoch = max(final_epoch, epoch)
        
        # 各メトリクスの最新値を追加
        for key in all_metrics.keys():
            if key in metrics and isinstance(metrics[key], list) and len(metrics[key]) > 0:
                all_metrics[key].append(metrics[key][-1])
            else:
                all_metrics[key].append(None)
    
    # プロットの準備
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    epochs = list(range(1, final_epoch + 1))
    print(f"Plotting from epoch 1 to {final_epoch}")

    # Lossプロット
    if all_metrics['train_loss'] and all_metrics['val_loss']:
        train_loss = [x for x in all_metrics['train_loss'] if x is not None]
        val_loss = [x for x in all_metrics['val_loss'] if x is not None]
        if train_loss and val_loss:
            print(f"Plotting loss: train={len(train_loss)} points, val={len(val_loss)} points")
            axs[0,0].plot(range(1, len(train_loss) + 1), train_loss, 'b-', label='Train')
            axs[0,0].plot(range(1, len(val_loss) + 1), val_loss, 'r-', label='Val')
            axs[0,0].set_title('Loss (Train/Val)')
            axs[0,0].set_xlabel('Epoch')
            axs[0,0].set_ylabel('Loss')
            axs[0,0].legend()
            axs[0,0].grid(True)
            axs[0,0].set_xlim(1, final_epoch)

    # Accuracyプロット
    if all_metrics['train_acc'] and all_metrics['val_acc']:
        train_acc = [x for x in all_metrics['train_acc'] if x is not None]
        val_acc = [x for x in all_metrics['val_acc'] if x is not None]
        if train_acc and val_acc:
            print(f"Plotting accuracy: train={len(train_acc)} points, val={len(val_acc)} points")
            axs[0,1].plot(range(1, len(train_acc) + 1), train_acc, 'b-', label='Train')
            axs[0,1].plot(range(1, len(val_acc) + 1), val_acc, 'r-', label='Val')
            axs[0,1].set_title('Accuracy (Train/Val)')
            axs[0,1].set_xlabel('Epoch')
            axs[0,1].set_ylabel('Accuracy')
            axs[0,1].legend()
            axs[0,1].grid(True)
            axs[0,1].set_xlim(1, final_epoch)
            axs[0,1].set_ylim(0, 1)

    # Top-3 Accuracyプロット
    if all_metrics['train_top3'] and all_metrics['val_top3']:
        train_top3 = [x for x in all_metrics['train_top3'] if x is not None]
        val_top3 = [x for x in all_metrics['val_top3'] if x is not None]
        if train_top3 and val_top3:
            print(f"Plotting top3: train={len(train_top3)} points, val={len(val_top3)} points")
            axs[1,0].plot(range(1, len(train_top3) + 1), train_top3, 'b-', label='Train')
            axs[1,0].plot(range(1, len(val_top3) + 1), val_top3, 'r-', label='Val')
            axs[1,0].set_title('Top-3 Accuracy (Train/Val)')
            axs[1,0].set_xlabel('Epoch')
            axs[1,0].set_ylabel('Top-3 Accuracy')
            axs[1,0].legend()
            axs[1,0].grid(True)
            axs[1,0].set_xlim(1, final_epoch)
            axs[1,0].set_ylim(0, 1)

    # Learning Rateプロット
    if all_metrics['lr']:
        lr = [x for x in all_metrics['lr'] if x is not None]
        if lr:
            print(f"Plotting learning rate: {len(lr)} points")
            axs[1,1].plot(range(1, len(lr) + 1), lr, 'g-')
            axs[1,1].set_title('Learning Rate')
            axs[1,1].set_xlabel('Epoch')
            axs[1,1].set_ylabel('Learning Rate')
            axs[1,1].grid(True)
            axs[1,1].set_xlim(1, final_epoch)

    plt.tight_layout()

    # プロットを保存
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f'training_curves_{timestamp}.png')
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
            import traceback
            traceback.print_exc()
    else:
        plt.show()

def print_accuracy_summary(checkpoint_dir):
    """チェックポイントのaccuracyサマリーを表示"""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    # エポック番号でソート
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    print("\n=== Training Progress Summary ===")
    print("Epoch\tVal Acc\tTrain Acc\tTop-3 Val\tTop-3 Train")
    print("-" * 50)
    
    for checkpoint_path in checkpoints:
        metrics, epoch, val_acc = load_checkpoint_metrics(checkpoint_path)
        if metrics is None:
            continue
            
        # メトリクスの最後の値を取得
        train_acc = metrics.get('train_acc', [0.0])[-1] if metrics.get('train_acc') else 0.0
        val_top3 = metrics.get('val_top3', [0.0])[-1] if metrics.get('val_top3') else 0.0
        train_top3 = metrics.get('train_top3', [0.0])[-1] if metrics.get('train_top3') else 0.0
        
        print(f"{epoch:5d}\t{val_acc:.4f}\t{train_acc:.4f}\t{val_top3:.4f}\t{train_top3:.4f}")
    
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description='Visualize training metrics from checkpoints')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_v1111_large_compiled',
                      help='Directory containing checkpoints')
    parser.add_argument('--save-dir', type=str, default='plots_v1111_large_compiled_new',
                      help='Directory to save plots')
    parser.add_argument('--summary-only', action='store_true',
                      help='Only print accuracy summary without plotting')
    
    args = parser.parse_args()
    
    # 相対パスを絶対パスに変換
    checkpoint_dir = get_absolute_path(args.checkpoint_dir)
    save_dir = get_absolute_path(args.save_dir)
    
    print(f"Using checkpoint directory: {checkpoint_dir}")
    print(f"Using save directory: {save_dir}")
    
    # サマリーの表示
    print_accuracy_summary(checkpoint_dir)
    
    # プロットの作成（summary-onlyがFalseの場合）
    if not args.summary_only:
        plot_training_curves(checkpoint_dir, save_dir)

if __name__ == "__main__":
    main()
