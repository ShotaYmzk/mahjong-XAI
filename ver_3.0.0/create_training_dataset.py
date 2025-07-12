# create_training_dataset.py - transformerモデル用の正しい麻雀データセット作成
import os
import h5py
import numpy as np
import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
import multiprocessing as mp
from functools import partial
import time
from datetime import datetime

# プロジェクトモジュール
try:
    from full_mahjong_parser import parse_full_mahjong_log
    from game_state import GameState, NUM_TILE_TYPES, MAX_EVENT_HISTORY, STATIC_FEATURE_DIM, EVENT_TYPES, NUM_PLAYERS
    from tile_utils import tile_id_to_string, tile_id_to_index, tile_index_to_id
    print("プロジェクトモジュールを正常にインポートしました。")
except ImportError as e:
    print(f"[エラー] プロジェクトモジュールのインポートに失敗しました: {e}")
    exit(1)

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MahjongDatasetCreator:
    """麻雀AI用の正しいデータセットを作成するクラス"""
    
    def __init__(self, output_path="training_data/mahjong_valid_dataset.hdf5"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 統計情報
        self.stats = {
            'total_games': 0,
            'total_rounds': 0, 
            'total_samples': 0,
            'valid_samples': 0,
            'invalid_samples': 0,
            'invalid_reasons': defaultdict(int),
            'tile_distribution': Counter(),
            'player_distribution': Counter()
        }
        
    def validate_sample(self, game_state, player_id, target_discard_tile_id):
        """サンプルの有効性を検証"""
        # 基本的なバリデーション
        if not (0 <= player_id < NUM_PLAYERS):
            return False, "invalid_player_id"
            
        if not (0 <= target_discard_tile_id <= 135):
            return False, "invalid_tile_id"
            
        # 手牌の確認
        player_hand = game_state.player_hands[player_id]
        if not player_hand:
            return False, "empty_hand"
            
        # ツモ後の手牌数確認（14枚であるべき）
        if len(player_hand) != 14:
            return False, f"invalid_hand_size_{len(player_hand)}"
            
        # 実際に捨てられる牌が手牌にあるかチェック
        if target_discard_tile_id not in player_hand:
            return False, "tile_not_in_hand"
            
        # リーチ状態のチェック
        reach_status = game_state.player_reach_status[player_id]
        if reach_status == 2:  # リーチ済み
            # リーチ後は基本的にツモ切りのみ
            last_tile = player_hand[-1] if player_hand else -1
            if last_tile != target_discard_tile_id:
                return False, "reach_violation"
                
        # 有効な打牌選択肢に含まれるかチェック
        valid_options = game_state.get_valid_discard_options(player_id)
        target_tile_index = tile_id_to_index(target_discard_tile_id)
        if target_tile_index not in valid_options:
            return False, "not_in_valid_options"
            
        return True, "valid"
    
    def extract_samples_from_round(self, round_data, game_index, round_index):
        """1局から学習サンプルを抽出"""
        samples = []
        events = round_data.get("events", [])
        
        game_state = GameState()
        try:
            game_state.init_round(round_data)
        except Exception as e:
            logger.warning(f"Game {game_index}, Round {round_index}: 初期化エラー - {e}")
            return samples
            
        sample_count = 0
        
        for i, event_xml in enumerate(events):
            tag = event_xml["tag"]
            attrib = event_xml["attrib"]
            
            # ツモイベントの検出
            tsumo_player_id = -1
            tsumo_tile_id = -1
            
            for t_tag, p_id in GameState.TSUMO_TAGS.items():
                if tag.startswith(t_tag) and tag[1:].isdigit():
                    try:
                        tsumo_tile_id = int(tag[1:])
                        tsumo_player_id = p_id
                        break
                    except (ValueError, IndexError):
                        continue
                        
            if tsumo_player_id != -1:
                # ツモを処理
                try:
                    game_state.process_tsumo(tsumo_player_id, tsumo_tile_id)
                except Exception as e:
                    logger.warning(f"Game {game_index}, Round {round_index}: ツモ処理エラー - {e}")
                    continue
                
                # 次の打牌を探す
                discard_info = self._find_next_discard(events, i, tsumo_player_id)
                
                if discard_info:
                    discard_tile_id = discard_info['tile_id']
                    
                    # サンプルの有効性を検証
                    is_valid, reason = self.validate_sample(game_state, tsumo_player_id, discard_tile_id)
                    
                    if is_valid:
                        try:
                            # 特徴量を生成
                            event_sequence = game_state.get_event_sequence_features()
                            static_features = game_state.get_static_features(tsumo_player_id)
                            target_index = tile_id_to_index(discard_tile_id)
                            
                            if target_index != -1:
                                sample = {
                                    'event_sequence': event_sequence,
                                    'static_features': static_features,
                                    'target': target_index,
                                    'game_index': game_index,
                                    'round_index': round_index,
                                    'sample_index': sample_count,
                                    'player_id': tsumo_player_id,
                                    'tsumo_tile': tile_id_to_string(tsumo_tile_id),
                                    'discard_tile': tile_id_to_string(discard_tile_id),
                                    'hand_size': len(game_state.player_hands[tsumo_player_id]),
                                    'junme': float(game_state.junme),
                                    'tsumogiri': discard_info.get('tsumogiri', False)
                                }
                                samples.append(sample)
                                sample_count += 1
                                
                                # 統計更新
                                self.stats['valid_samples'] += 1
                                self.stats['tile_distribution'][target_index] += 1
                                self.stats['player_distribution'][tsumo_player_id] += 1
                        except Exception as e:
                            logger.warning(f"Game {game_index}, Round {round_index}: 特徴量生成エラー - {e}")
                            self.stats['invalid_samples'] += 1
                            self.stats['invalid_reasons']['feature_generation_error'] += 1
                    else:
                        self.stats['invalid_samples'] += 1
                        self.stats['invalid_reasons'][reason] += 1
                        
                        # デバッグ用ログ（手牌にない牌の場合）
                        if reason == "tile_not_in_hand":
                            hand_str = [tile_id_to_string(t) for t in game_state.player_hands[tsumo_player_id]]
                            discard_str = tile_id_to_string(discard_tile_id)
                            logger.debug(f"Invalid discard: P{tsumo_player_id} tried to discard {discard_str} not in hand {hand_str}")
                
                # 実際の打牌も処理（状態の一貫性のため）
                if discard_info:
                    try:
                        game_state.process_discard(
                            tsumo_player_id, 
                            discard_info['tile_id'], 
                            discard_info.get('tsumogiri', False)
                        )
                    except Exception as e:
                        logger.warning(f"Game {game_index}, Round {round_index}: 打牌処理エラー - {e}")
            else:
                # 非ツモイベントの処理
                try:
                    game_state.process_event(event_xml)
                except Exception as e:
                    logger.warning(f"Game {game_index}, Round {round_index}: イベント処理エラー - {e}")
                    
        return samples
    
    def _find_next_discard(self, events, tsumo_index, tsumo_player_id):
        """ツモ後の次の打牌を見つける"""
        search_index = tsumo_index + 1
        
        while search_index < len(events):
            event_xml = events[search_index]
            tag = event_xml["tag"]
            
            # リーチ宣言はスキップ
            if tag == "REACH":
                search_index += 1
                continue
                
            # 打牌イベントをチェック
            for d_tag, p_id in GameState.DISCARD_TAGS.items():
                if (tag.startswith(d_tag) and 
                    tag[1:].isdigit() and 
                    p_id == tsumo_player_id):
                    try:
                        discard_tile_id = int(tag[1:])
                        tsumogiri = tag[0].islower()
                        return {
                            'tile_id': discard_tile_id,
                            'tsumogiri': tsumogiri,
                            'event_xml': event_xml
                        }
                    except (ValueError, IndexError):
                        continue
            
            # 他プレイヤーのイベントが来たら終了
            other_player_event = False
            for tag_prefix in ['T', 'U', 'V', 'W', 'D', 'E', 'F', 'G']:
                if (tag.startswith(tag_prefix) and 
                    tag[1:].isdigit()):
                    event_player = GameState.TSUMO_TAGS.get(tag_prefix, 
                                                         GameState.DISCARD_TAGS.get(tag_prefix, -1))
                    if event_player != tsumo_player_id and event_player != -1:
                        other_player_event = True
                        break
            
            if other_player_event:
                break
                
            search_index += 1
        
        return None
    
    def process_single_file(self, xml_path):
        """1つのXMLファイルを処理"""
        logger.info(f"処理開始: {xml_path}")
        
        try:
            meta, rounds_data = parse_full_mahjong_log(xml_path)
        except Exception as e:
            logger.error(f"ファイル解析エラー {xml_path}: {e}")
            return []
            
        all_samples = []
        game_index = self.stats['total_games']
        
        for round_index, round_data in enumerate(rounds_data):
            samples = self.extract_samples_from_round(round_data, game_index, round_index)
            all_samples.extend(samples)
            
        self.stats['total_games'] += 1
        self.stats['total_rounds'] += len(rounds_data)
        
        logger.info(f"完了: {xml_path} - {len(all_samples)}サンプル抽出")
        return all_samples
    
    def create_dataset(self, xml_directory, max_files=None, num_workers=4):
        """データセット作成のメイン処理"""
        logger.info("データセット作成開始")
        
        xml_files = list(Path(xml_directory).glob("*.xml"))
        if max_files:
            xml_files = xml_files[:max_files]
            
        logger.info(f"処理対象ファイル数: {len(xml_files)}")
        
        # マルチプロセシングで処理
        with mp.Pool(num_workers) as pool:
            all_samples_list = pool.map(self.process_single_file, xml_files)
            
        # サンプルを統合
        all_samples = []
        for samples in all_samples_list:
            all_samples.extend(samples)
            
        self.stats['total_samples'] = len(all_samples)
        
        if not all_samples:
            logger.error("有効なサンプルが見つかりませんでした")
            return
            
        # HDF5ファイルに保存
        self._save_to_hdf5(all_samples)
        
        # 統計情報を表示
        self._print_statistics()
        
    def _save_to_hdf5(self, samples):
        """サンプルをHDF5ファイルに保存"""
        logger.info(f"HDF5ファイルに保存中: {self.output_path}")
        
        # データを配列に変換
        event_sequences = np.array([s['event_sequence'] for s in samples], dtype=np.float32)
        static_features = np.array([s['static_features'] for s in samples], dtype=np.float32)
        targets = np.array([s['target'] for s in samples], dtype=np.int32)
        
        # メタデータ
        metadata = []
        for s in samples:
            metadata.append({
                'game_index': s['game_index'],
                'round_index': s['round_index'],
                'sample_index': s['sample_index'],
                'player_id': s['player_id'],
                'tsumo_tile': s['tsumo_tile'],
                'discard_tile': s['discard_tile'],
                'hand_size': s['hand_size'],
                'junme': s['junme'],
                'tsumogiri': s['tsumogiri']
            })
        
        with h5py.File(self.output_path, 'w') as f:
            # データセット作成
            f.create_dataset('event_sequences', data=event_sequences, compression='gzip')
            f.create_dataset('static_features', data=static_features, compression='gzip')
            f.create_dataset('targets', data=targets, compression='gzip')
            
            # メタデータ保存
            f.attrs['num_samples'] = len(samples)
            f.attrs['event_sequence_shape'] = event_sequences.shape
            f.attrs['static_features_shape'] = static_features.shape
            f.attrs['num_tile_types'] = NUM_TILE_TYPES
            f.attrs['max_event_history'] = MAX_EVENT_HISTORY
            f.attrs['static_feature_dim'] = STATIC_FEATURE_DIM
            f.attrs['creation_time'] = datetime.now().isoformat()
            
            # 統計情報保存
            stats_group = f.create_group('statistics')
            for key, value in self.stats.items():
                if isinstance(value, (dict, Counter)):
                    subgroup = stats_group.create_group(key)
                    for k, v in value.items():
                        subgroup.attrs[str(k)] = v
                else:
                    stats_group.attrs[key] = value
                    
            # メタデータをJSON文字列として保存
            metadata_str = json.dumps(metadata, ensure_ascii=False, indent=2)
            f.create_dataset('metadata', data=metadata_str.encode('utf-8'))
            
        logger.info(f"保存完了: {len(samples)}サンプル")
        
    def _print_statistics(self):
        """統計情報を表示"""
        logger.info("=== データセット統計 ===")
        logger.info(f"総ゲーム数: {self.stats['total_games']}")
        logger.info(f"総局数: {self.stats['total_rounds']}")
        logger.info(f"有効サンプル数: {self.stats['valid_samples']}")
        logger.info(f"無効サンプル数: {self.stats['invalid_samples']}")
        
        if self.stats['invalid_samples'] > 0:
            logger.info("無効サンプルの理由:")
            for reason, count in self.stats['invalid_reasons'].items():
                percentage = count / self.stats['invalid_samples'] * 100
                logger.info(f"  {reason}: {count} ({percentage:.1f}%)")
                
        # 牌の分布
        if self.stats['tile_distribution']:
            logger.info("\n牌の分布（上位10種）:")
            for tile_idx, count in self.stats['tile_distribution'].most_common(10):
                tile_name = tile_id_to_string(tile_index_to_id(tile_idx))
                percentage = count / self.stats['valid_samples'] * 100
                logger.info(f"  {tile_name}: {count} ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(
        description="麻雀AI用の正しいトレーニングデータセットを作成します"
    )
    parser.add_argument("xml_directory", help="XMLファイルが格納されたディレクトリ")
    parser.add_argument("--output", default="training_data/mahjong_valid_dataset.hdf5", 
                       help="出力HDF5ファイルパス")
    parser.add_argument("--max-files", type=int, help="処理するファイル数の上限")
    parser.add_argument("--workers", type=int, default=4, help="並列処理のワーカー数")
    parser.add_argument("--debug", action='store_true', help="デバッグログを有効にする")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    creator = MahjongDatasetCreator(args.output)
    creator.create_dataset(args.xml_directory, args.max_files, args.workers)

if __name__ == "__main__":
    main() 