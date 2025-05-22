import torch
import matplotlib.pyplot as plt
import os
import glob
import re
import numpy as np

# ===============================================================================
# =                           Configuration Settings                           =
# ===============================================================================
CHECKPOINT_DIR = "./checkpoints_v1111_large_compiled" # チェックポイントが保存されているディレクトリ
OUTPUT_PLOT_FILE = "./plots/loaded_metrics_plot.png" # 生成するプロット画像の保存先

# ===============================================================================
# =                             Helper Functions                              =
# ===============================================================================
def get_sorted_checkpoint_files(checkpoint_dir):
    """指定されたディレクトリからエポック番号順にソートされたチェックポイントファイルのリストを取得する"""
    if not os.path.isdir(checkpoint_dir):
        print(f"エラー: チェックポイントディレクトリが見つかりません: {checkpoint_dir}")
        return []

    # 'checkpoint_epoch_数字.pth' のパターンに一致するファイルを検索
    # 例: checkpoint_epoch_1.pth, checkpoint_epoch_10.pth
    file_pattern = os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth")
    checkpoint_files = glob.glob(file_pattern)

    if not checkpoint_files:
        print(f"エラー: {checkpoint_dir} にチェックポイントファイルが見つかりません。")
        print(f"検索パターン: {file_pattern}")
        return []

    # エポック番号を抽出してソートするためのキー関数
    def sort_key(filename):
        match = re.search(r'checkpoint_epoch_(\d+)\.pth$', os.path.basename(filename))
        return int(match.group(1)) if match else -1

    checkpoint_files.sort(key=sort_key)
    
    # ソートキーが-1（パターンに一致しない）ファイルをフィルタリング（念のため）
    checkpoint_files = [f for f in checkpoint_files if sort_key(f) != -1]

    print(f"{len(checkpoint_files)} 個のチェックポイントファイルが見つかりました。")
    return checkpoint_files

def load_metrics_from_checkpoints(checkpoint_files):
    """チェックポイントファイルからメトリクスを読み込む"""
    epochs = []
    val_accuracies = []
    learning_rates = []

    print("チェックポイントファイルからメトリクスを読み込んでいます...")
    for cp_file in checkpoint_files:
        try:
            # CPUにロード (GPUがなくても動作するように)
            checkpoint = torch.load(cp_file, map_location=torch.device('cpu'))
            
            # epoch: チェックポイントは1-basedで保存されていると仮定
            if 'epoch' not in checkpoint:
                print(f"警告: {cp_file} に 'epoch' キーがありません。スキップします。")
                continue
            current_epoch = checkpoint['epoch']
            
            # val_acc: 検証精度
            if 'val_acc' not in checkpoint:
                print(f"警告: {cp_file} に 'val_acc' キーがありません。スキップします。")
                continue
            val_acc = checkpoint['val_acc']

            # lr: 学習率 (optimizer_state_dictから取得)
            if 'optimizer_state_dict' in checkpoint and \
               'param_groups' in checkpoint['optimizer_state_dict'] and \
               len(checkpoint['optimizer_state_dict']['param_groups']) > 0 and \
               'lr' in checkpoint['optimizer_state_dict']['param_groups'][0]:
                lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
            else:
                print(f"警告: {cp_file} から学習率を読み取れませんでした。NaNとして記録します。")
                lr = float('nan') # 読み取れない場合は NaN

            epochs.append(current_epoch)
            val_accuracies.append(val_acc)
            learning_rates.append(lr)
            
            if current_epoch % 10 == 0 or current_epoch == 1: # 10エポックごと、または最初のエポックでログ表示
                 print(f"  Epoch {current_epoch}: Val Acc = {val_acc:.4f}, LR = {lr:.2e}")

        except Exception as e:
            print(f"エラー: {cp_file} の読み込み中にエラーが発生しました: {e}")
            continue
    
    if not epochs:
        print("エラー: 有効なメトリクスをどのチェックポイントからも読み込めませんでした。")
        return None, None, None

    return epochs, val_accuracies, learning_rates

# ===============================================================================
# =                               Plotting Function                             =
# ===============================================================================
def plot_metrics(epochs, val_accuracies, learning_rates, output_file):
    """読み込んだメトリクスをプロットし、ファイルに保存する"""
    if not epochs:
        print("プロットするデータがありません。")
        return

    num_plots = 0
    if val_accuracies and not all(np.isnan(val_accuracies)):
        num_plots += 1
    if learning_rates and not all(np.isnan(learning_rates)):
        num_plots += 1

    if num_plots == 0:
        print("プロット可能な有効なデータがありません。")
        return

    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots), squeeze=False)
    plot_idx = 0

    # 検証精度のプロット
    if val_accuracies and not all(np.isnan(val_accuracies)):
        ax = axs[plot_idx, 0]
        ax.plot(epochs, val_accuracies, marker='o', linestyle='-', color='r', label='Validation Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Validation Accuracy vs. Epoch')
        ax.legend()
        ax.grid(True)
        plot_idx += 1

    # 学習率のプロット
    if learning_rates and not all(np.isnan(learning_rates)):
        ax = axs[plot_idx, 0]
        ax.plot(epochs, learning_rates, marker='.', linestyle='-', color='g', label='Learning Rate')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate vs. Epoch')
        ax.set_yscale('log') # 学習率は対数スケールの方が見やすいことが多い
        ax.legend()
        ax.grid(True)
        plot_idx += 1

    plt.tight_layout()
    
    try:
        plt.savefig(output_file)
        print(f"プロットが {output_file} に保存されました。")
    except Exception as e:
        print(f"エラー: プロットの保存中にエラーが発生しました: {e}")
    
    # 対話的に表示する場合は plt.show() をコール
    # plt.show()

    # 注意喚起メッセージ
    print("\n--- 注意 ---")
    print("このプロットは、チェックポイントファイルから直接読み取れた情報に基づいています。")
    print("具体的には、検証精度 (Validation Accuracy) と学習率 (Learning Rate) のみです。")
    print("損失曲線 (Loss curves)、訓練精度 (Train Accuracy)、Top-3 精度 (Top-3 Accuracy) の完全な履歴は、")
    print("チェックポイントファイルには通常保存されていません。")
    print(f"これらの詳細なメトリクスについては、トレーニングスクリプトが生成したプロット画像が保存されているディレクトリ")
    print(f"（例: {os.path.dirname(CHECKPOINT_DIR)}/plots_v1111_large_compiled/ など）をご確認ください。")
    print("多くの場合、'latest_training_metrics.png' や 'training_metrics_epoch_X.png' といったファイル名で保存されています。")

# ===============================================================================
# =                               Main Execution                                =
# ===============================================================================
if __name__ == "__main__":
    print(f"指定されたチェックポイントディレクトリ: {CHECKPOINT_DIR}")
    
    # 出力ディレクトリが存在しない場合は作成
    output_dir = os.path.dirname(OUTPUT_PLOT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"出力ディレクトリ {output_dir} を作成しました。")

    checkpoint_files_list = get_sorted_checkpoint_files(CHECKPOINT_DIR)

    if checkpoint_files_list:
        epochs_data, val_acc_data, lr_data = load_metrics_from_checkpoints(checkpoint_files_list)
        
        if epochs_data: # 有効なデータが読み込めた場合のみプロット
            plot_metrics(epochs_data, val_acc_data, lr_data, OUTPUT_PLOT_FILE)
        else:
            print("メトリクスの読み込みに失敗したため、プロットをスキップします。")
    else:
        print("チェックポイントファイルが見つからなかったため、処理を終了します。")