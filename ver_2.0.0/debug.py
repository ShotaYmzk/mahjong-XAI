import torch
import numpy as np
import os
import glob
import logging
import random

# --- プロジェクトモジュールのインポート (修正箇所) ---

# game_state.py から定数とクラスをインポート
from game_state import GameState, STATIC_FEATURE_DIM, NUM_PLAYERS, EVENT_TYPES, MAX_EVENT_HISTORY

# predict.py から必要なものをインポート
from predict import load_trained_model, DEFAULT_MODEL_PATH

# parser と analyzer をインポート
from full_mahjong_parser import parse_full_mahjong_log
import shanten_analyzer

# tile_utils.py からは関数のみをインポート
# from tile_utils import tile_id_to_index, hand_ids_to_34_array # shanten_analyzerが内部で使うのでここでは不要

# --- 設定 ---
XML_DIR = "/home/ubuntu/Documents/tenhou_xml_2023"
MODEL_PATH = DEFAULT_MODEL_PATH

# --- ロギング設定 ---
# ログレベルをDEBUGに設定して詳細な情報を表示
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def debug_single_file():
    """単一のファイルをデバッグモードで処理する"""
    logging.info("デバッグモードでデータ収集プロセスを開始します。")

    # --- 1. モデルのロード ---
    logging.info("モデルをロード中...")
    try:
        event_feature_dim = GameState().get_event_sequence_features().shape[1]
        model = load_trained_model(MODEL_PATH, event_feature_dim, STATIC_FEATURE_DIM)
        logging.info("モデルのロード成功。")
    except Exception as e:
        logging.error(f"モデルのロードに失敗しました: {e}", exc_info=True)
        return

    # --- 2. 牌譜ファイルの選択 ---
    xml_files = glob.glob(os.path.join(XML_DIR, "**", "*.xml"), recursive=True)
    xml_files += glob.glob(os.path.join(XML_DIR, "**", "*.mjlog"), recursive=True)
    if not xml_files:
        logging.error(f"デバッグ対象のXML/MJLOGファイルが見つかりません。パスを確認してください: {XML_DIR}")
        return
    
    # ランダムに1ファイル選ぶ
    target_xml_file = random.choice(xml_files)
    logging.info(f"デバッグ対象ファイル: {target_xml_file}")

    # --- 3. 牌譜の処理 ---
    try:
        _, rounds_data = parse_full_mahjong_log(target_xml_file)
    except Exception as e:
        logging.error(f"牌譜ファイルのパースに失敗しました: {e}", exc_info=True)
        return

    total_data_points_collected = 0
    logging.info(f"{len(rounds_data)}局のデータを処理します。")

    for round_idx, round_data in enumerate(rounds_data):
        logging.debug(f"\n--- 局 {round_idx + 1} の処理開始 ---")
        try:
            events = round_data.get("events", [])
            if not events:
                logging.warning("この局にはイベントがありません。スキップします。")
                continue

            game_state = GameState()
            game_state.init_round(round_data)

            for i, event_xml in enumerate(events):
                tag = event_xml["tag"]
                # logging.debug(f"イベント {i}: {tag}") # ログが多すぎるのでコメントアウト

                is_discard = False
                discard_player_id, discard_pai_id = -1, -1
                for d_tag, p_id in GameState.DISCARD_TAGS.items():
                    if tag.startswith(d_tag) and tag[1:].isdigit():
                        discard_player_id, discard_pai_id = p_id, int(tag[1:])
                        is_discard = True
                        break
                
                if is_discard:
                    # 打牌直前の手牌は14枚のはず
                    hand_14 = game_state.player_hands[discard_player_id]
                    if len(hand_14) % 3 != 2:
                        logging.warning(f"    [警告] 打牌イベント直前ですが、P{discard_player_id}の手牌が14枚ではありません: {len(hand_14)}枚")
                        # このイベントはスキップ
                        game_state.process_event(event_xml)
                        continue

                    logging.debug(f"  [打牌イベント検出] プレイヤー: {discard_player_id}, 打牌: {discard_pai_id}")
                    
                    # モデルで予測とアクティベーション抽出
                    _, _, _, _, activation_vector = game_state.predict_discard_with_model(
                        model, discard_player_id
                    )

                    if activation_vector is not None:
                        logging.debug("    [成功] アクティベーションベクトルを抽出しました。")
                        
                        # 指標計算
                        shanten_change, ukeire_count = shanten_analyzer.analyze_speed_metrics(
                            hand_14, discard_pai_id, game_state
                        )
                        is_deal_in = 0
                        if i + 1 < len(events) and events[i+1]['tag'] == 'AGARI' and events[i+1]['attrib'].get('fromWho') == str(discard_player_id):
                            is_deal_in = 1
                        dora_count = game_state.count_dora_in_hand(discard_player_id)
                        
                        logging.info(f"    [データ収集成功] P{discard_player_id} | シャンテン変化:{shanten_change}, 受け入れ:{ukeire_count}, 放銃:{is_deal_in}, ドラ:{dora_count}")
                        total_data_points_collected += 1
                    else:
                        logging.warning("    [失敗] アクティベーションベクトルが抽出できませんでした (None)。")
                
                # GameStateを更新
                game_state.process_event(event_xml)

        except Exception as e:
            logging.error(f"局 {round_idx + 1} の処理中に予期せぬエラー: {e}", exc_info=True)
            continue
    
    logging.info("\n--- デバッグ完了 ---")
    logging.info(f"処理したファイル: {target_xml_file}")
    logging.info(f"収集できたデータポイント数: {total_data_points_collected}")
    if total_data_points_collected == 0:
        logging.error("最終的にデータポイントを1つも収集できませんでした。ログを確認して原因を特定してください。")
    else:
        logging.info("データポイントの収集に成功しました。create_activation_dataset.py のロジックは正常である可能性が高いです。")


if __name__ == '__main__':
    debug_single_file()