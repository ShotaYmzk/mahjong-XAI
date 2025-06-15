import numpy as np
import h5py
import logging
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib # モデルの保存用

# --- 設定 ---
HDF5_PATH = './activation_dataset.hdf5'
N_COMPONENTS_PCA = 50  # PCAで削減する次元数
N_CLUSTERS = 25        # k-meansのクラスタ数
BATCH_SIZE_KMEANS = 4096 # MiniBatchKMeansのバッチサイズ
Z_SCORE_THRESHOLD = 1.0 # 概念ラベルを付与するZスコアのしきい値
SAFETY_Z_SCORE_THRESHOLD = 0.8 # Safetyラベル用のZスコアのしきい値
CHUNK_SIZE = 1000000   # 一度に処理するデータポイント数

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data_in_chunks(path, chunk_size=CHUNK_SIZE):
    """HDF5ファイルからデータセットをチャンク単位でロードする"""
    logging.info(f"HDF5ファイルからデータをチャンク単位でロード中: {path}")
    with h5py.File(path, 'r') as hf:
        total_size = hf['activations'].shape[0]
        for i in range(0, total_size, chunk_size):
            end = min(i + chunk_size, total_size)
            logging.info(f"チャンク {i//chunk_size + 1} をロード中... ({i} から {end} まで)")
            yield (
                hf['activations'][i:end],
                hf['shanten_changes'][i:end],
                hf['ukeire_counts'][i:end],
                hf['is_deal_ins'][i:end],
                hf['dora_counts'][i:end]
            )

def main():
    # 1. データのチャンク単位での処理
    logging.info("PCAの学習を開始...")
    pca = PCA(n_components=N_COMPONENTS_PCA)
    
    # 最初のチャンクでPCAを初期化
    first_chunk = True
    total_samples = 0
    
    for chunk_idx, (activations_chunk, shanten_changes_chunk, ukeire_counts_chunk, 
                   is_deal_ins_chunk, dora_counts_chunk) in enumerate(load_data_in_chunks(HDF5_PATH)):
        
        if first_chunk:
            # 最初のチャンクでPCAを初期化
            pca.fit(activations_chunk)
            first_chunk = False
            logging.info(f"PCAの初期化完了。入力次元: {activations_chunk.shape[1]}, 削減後: {N_COMPONENTS_PCA}")
        
        # チャンクごとにPCA変換を実行
        activations_pca_chunk = pca.transform(activations_chunk)
        total_samples += len(activations_chunk)
        
        # チャンクごとのデータを保存
        np.save(f'chunk_{chunk_idx}_pca.npy', activations_pca_chunk)
        np.save(f'chunk_{chunk_idx}_metadata.npy', {
            'shanten_changes': shanten_changes_chunk,
            'ukeire_counts': ukeire_counts_chunk,
            'is_deal_ins': is_deal_ins_chunk,
            'dora_counts': dora_counts_chunk
        })
        
        logging.info(f"チャンク {chunk_idx + 1} の処理完了")
    
    logging.info(f"PCA完了。累積寄与率: {np.sum(pca.explained_variance_ratio_):.4f}")
    joblib.dump(pca, 'pca_model_2.joblib')
    logging.info("PCAモデルを 'pca_model_2.joblib' に保存しました。")

    # 2. k-meansによるクラスタリング
    logging.info(f"MiniBatchKMeansによるクラスタリングを開始... (クラスタ数: {N_CLUSTERS})")
    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=BATCH_SIZE_KMEANS, n_init='auto', random_state=42)
    
    # チャンクごとにクラスタリングを実行
    for chunk_idx in range((total_samples + CHUNK_SIZE - 1) // CHUNK_SIZE):
        activations_pca_chunk = np.load(f'chunk_{chunk_idx}_pca.npy')
        kmeans.partial_fit(activations_pca_chunk)
        logging.info(f"チャンク {chunk_idx + 1} のクラスタリング完了")
    
    joblib.dump(kmeans, 'kmeans_model_2.joblib')
    logging.info("k-meansモデルを 'kmeans_model_2.joblib' に保存しました。")

    # 3. クラスタごとの統計情報を計算
    logging.info("クラスタごとの統計情報を計算中...")
    cluster_stats = []
    
    for chunk_idx in range((total_samples + CHUNK_SIZE - 1) // CHUNK_SIZE):
        activations_pca_chunk = np.load(f'chunk_{chunk_idx}_pca.npy')
        metadata = np.load(f'chunk_{chunk_idx}_metadata.npy', allow_pickle=True).item()
        
        cluster_labels_chunk = kmeans.predict(activations_pca_chunk)
        
        df_chunk = pd.DataFrame({
            'cluster': cluster_labels_chunk,
            'shanten_change': metadata['shanten_changes'],
            'ukeire_count': metadata['ukeire_counts'],
            'deal_in_rate': metadata['is_deal_ins'],
            'dora_count': metadata['dora_counts']
        })
        
        cluster_stats.append(df_chunk)
        
        # 一時ファイルを削除
        import os
        os.remove(f'chunk_{chunk_idx}_pca.npy')
        os.remove(f'chunk_{chunk_idx}_metadata.npy')
    
    # 全チャンクの統計情報を結合
    df = pd.concat(cluster_stats, ignore_index=True)
    
    cluster_stats = df.groupby('cluster').agg(
        size=('cluster', 'size'),
        avg_shanten_change=('shanten_change', 'mean'),
        avg_ukeire_count=('ukeire_count', 'mean'),
        deal_in_rate=('deal_in_rate', 'mean'),
        avg_dora_count=('dora_count', 'mean')
    ).reset_index()

    logging.info("統計情報の計算完了。")

    # 4. Z-scoreを計算して概念ラベルを付与
    logging.info("Z-scoreを計算し、概念ラベルを付与中...")
    stats_df = cluster_stats.copy()
    metrics_to_normalize = ['avg_shanten_change', 'avg_ukeire_count', 'deal_in_rate', 'avg_dora_count']
    
    for metric in metrics_to_normalize:
        scaler = StandardScaler()
        weighted_metric = np.repeat(stats_df[metric].values, stats_df['size'].values)
        scaler.fit(weighted_metric.reshape(-1, 1))
        stats_df[f'{metric}_zscore'] = scaler.transform(stats_df[[metric]])

    # 概念ラベルを決定
    concept_labels = {}
    for i, row in stats_df.iterrows():
        cluster_id = int(row['cluster'])
        labels = []
        if row['avg_shanten_change_zscore'] > Z_SCORE_THRESHOLD or row['avg_ukeire_count_zscore'] > Z_SCORE_THRESHOLD:
            labels.append('Speed')
        if row['deal_in_rate_zscore'] < -SAFETY_Z_SCORE_THRESHOLD:
            labels.append('Safety')
        
        if not labels:
            labels.append('Normal')
        
        concept_labels[cluster_id] = labels

    stats_df['concept_labels'] = stats_df['cluster'].map(concept_labels)
    
    # 結果を表示
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', None)
    print("\n--- クラスタ分析結果 ---")
    print(stats_df[['cluster', 'size', 'avg_shanten_change', 'avg_ukeire_count', 'deal_in_rate', 'avg_dora_count', 'concept_labels']].sort_values(by='cluster'))

    # 概念ラベルを保存
    joblib.dump(concept_labels, 'concept_labels_2.joblib')
    logging.info("概念ラベルを 'concept_labels_2.joblib' に保存しました。")
    logging.info("すべての処理が完了しました。")


if __name__ == '__main__':
    main()