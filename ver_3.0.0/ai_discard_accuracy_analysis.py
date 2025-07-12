# ai_discard_accuracy_analysis.py - AIの推奨打牌と実際の打牌の一致率分析システム
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import re

try:
    from full_mahjong_parser import parse_full_mahjong_log
    from game_state import GameState, NUM_PLAYERS, NUM_TILE_TYPES
    from tile_utils import tile_id_to_string, tile_index_to_id
    from predict_enhanced import (
        load_trained_model, predict_discard, 
        DEFAULT_MODEL_PATH, DEVICE
    )
    from interactive_analysis import InteractiveAnalysisSystem
    from mahjong.shanten import Shanten
    from mahjong.tile import TilesConverter
except ImportError as e:
    print(f"[エラー] プロジェクトモジュールのインポートに失敗しました: {e}")
    exit(1)

class AccuracyAnalysisSystem:
    """AIの推奨打牌と実際の打牌の一致率分析システム"""
    
    def __init__(self):
        self.plt_japanese_support()
        self.model = None
        self.shanten_calculator = Shanten()
        
    def plt_japanese_support(self):
        """Matplotlibで日本語表示をサポート"""
        try:
            plt.rcParams['font.family'] = 'DejaVu Sans'
        except:
            print("[警告] 日本語フォントの設定に失敗しました")
    
    def robust_hand_parser(self, hand_string: str) -> list[int]:
        """手牌文字列を34種配列に変換"""
        # 正規表現を使って、各スーツの数字をすべて抽出して結合する
        man = "".join(re.findall(r'([0-9]+)m', hand_string))
        pin = "".join(re.findall(r'([0-9]+)p', hand_string))
        sou = "".join(re.findall(r'([0-9]+)s', hand_string))
        honors = "".join(re.findall(r'([0-9]+)z', hand_string))
        
        # まず has_aka_dora=True オプション付きで136種の牌に変換
        tiles_136 = TilesConverter.string_to_136_array(
            man=man, pin=pin, sou=sou, honors=honors, has_aka_dora=True
        )
        
        # 136種の牌から34種の配列に変換して返す
        return TilesConverter.to_34_array(tiles_136)
    
    def format_tiles_for_display(self, tile_indices):
        """牌のインデックスのリストを表示用の文字列に変換"""
        if not tile_indices:
            return ""
            
        man = sorted([i for i in tile_indices if 0 <= i <= 8])
        pin = sorted([i for i in tile_indices if 9 <= i <= 17])
        sou = sorted([i for i in tile_indices if 18 <= i <= 26])
        honors = sorted([i for i in tile_indices if 27 <= i <= 33])
        
        result_str = ""
        if man:
            # 赤5萬は0mと表示
            result_str += "".join(['0' if t == 4 else str(t + 1) for t in man]) + "m"
        if pin:
            # 赤5筒は0pと表示
            result_str += "".join(['0' if t == 13 else str(t - 9 + 1) for t in pin]) + "p"
        if sou:
            # 赤5索は0sと表示
            result_str += "".join(['0' if t == 22 else str(t - 18 + 1) for t in sou]) + "s"
        if honors:
            result_str += "".join([str(t - 27 + 1) for t in honors]) + "z"
            
        return result_str
    
    def format_shanten(self, shanten_value: int) -> str:
        """シャンテン数を「N向聴」または「聴牌」の文字列に変換"""
        if shanten_value == 0:
            return "聴牌"
        if shanten_value < 0:
            return "和了"
        return f"{shanten_value}向聴"
    
    def get_shanten_after_best_discard(self, tiles_14, shanten_func_name):
        """14枚の手牌から1枚捨てて13枚にした時の、最小シャンテン数を計算"""
        shanten_func = getattr(self.shanten_calculator, shanten_func_name)
        min_shanten = 8
        
        unique_tiles_in_hand = [i for i, count in enumerate(tiles_14) if count > 0]
        if not unique_tiles_in_hand:
            return min_shanten

        for discard_index in unique_tiles_in_hand:
            temp_hand_13 = list(tiles_14)
            temp_hand_13[discard_index] -= 1
            shanten = shanten_func(temp_hand_13)
            if shanten < min_shanten:
                min_shanten = shanten
                
        return min_shanten
    
    def analyze_discard_ukeire(self, hand_tiles_34, discard_tile_index):
        """特定の牌を打牌した時の受け入れ分析"""
        # 手牌から指定の牌を1枚減らす
        hand_13_tiles = list(hand_tiles_34)
        if hand_13_tiles[discard_tile_index] <= 0:
            return None  # その牌を持っていない場合
            
        hand_13_tiles[discard_tile_index] -= 1
        shanten_13 = self.shanten_calculator.calculate_shanten(hand_13_tiles)
        
        ukeire_tiles = {}
        
        if shanten_13 == 0:  # 聴牌の場合: 待ち牌を計算
            for draw_index in range(34):
                # 5枚目になる牌は引けない and 自分が捨てた牌はフリテンになるので待ちに含めない
                if hand_tiles_34[draw_index] < 4 and draw_index != discard_tile_index:
                    temp_hand_14 = list(hand_13_tiles)
                    temp_hand_14[draw_index] += 1
                    # アガリ(-1)になる牌を探す
                    if self.shanten_calculator.calculate_shanten(temp_hand_14) == -1:
                        remaining_count = 4 - hand_tiles_34[draw_index]
                        ukeire_tiles[draw_index] = remaining_count
        else:  # 聴牌していない場合: シャンテン数を進める牌を計算
            for draw_index in range(34):
                if hand_tiles_34[draw_index] < 4:
                    hand_14_after_draw = list(hand_13_tiles)
                    hand_14_after_draw[draw_index] += 1
                    shanten_after_draw_and_discard = self.get_shanten_after_best_discard(
                        hand_14_after_draw, 'calculate_shanten'
                    )
                    if shanten_after_draw_and_discard < shanten_13:
                        remaining_count = 4 - hand_tiles_34[draw_index]
                        ukeire_tiles[draw_index] = remaining_count

        total_ukeire_count = sum(ukeire_tiles.values())
        
        return {
            "discard_tile": discard_tile_index,
            "shanten_after_discard": shanten_13,
            "ukeire_tiles": ukeire_tiles,
            "total_ukeire_count": total_ukeire_count
        }
    
    def convert_game_hand_to_34_array(self, game_state, player_id):
        """GameStateの手牌を34種配列に変換"""
        hand_tiles_34 = [0] * 34
        
        player_hand = game_state.current_hands[player_id]
        for tile_id in player_hand:
            # tile_idから34種インデックスに変換
            if tile_id == -1:
                continue
                
            # 萬子 (0-8)
            if 1 <= tile_id <= 9:
                hand_tiles_34[tile_id - 1] += 1
            # 筒子 (9-17)  
            elif 10 <= tile_id <= 18:
                hand_tiles_34[tile_id - 10 + 9] += 1
            # 索子 (18-26)
            elif 19 <= tile_id <= 27:
                hand_tiles_34[tile_id - 19 + 18] += 1
            # 字牌 (27-33)
            elif 28 <= tile_id <= 34:
                hand_tiles_34[tile_id - 28 + 27] += 1
            # 赤5萬 (tile_id = 0)
            elif tile_id == 0:
                hand_tiles_34[4] += 1  # 5萬の位置
            # 赤5筒 (tile_id = 50)
            elif tile_id == 50:
                hand_tiles_34[13] += 1  # 5筒の位置
            # 赤5索 (tile_id = 51)
            elif tile_id == 51:
                hand_tiles_34[22] += 1  # 5索の位置
                
        return hand_tiles_34
    
    def tile_string_to_34_index(self, tile_string):
        """牌文字列を34種インデックスに変換"""
        if not tile_string or tile_string == "?":
            return None
            
        if tile_string[-1] == 'm':
            num = int(tile_string[0])
            if num == 0:  # 赤5萬
                return 4
            return num - 1
        elif tile_string[-1] == 'p':
            num = int(tile_string[0])
            if num == 0:  # 赤5筒
                return 13
            return num - 1 + 9
        elif tile_string[-1] == 's':
            num = int(tile_string[0])
            if num == 0:  # 赤5索
                return 22
            return num - 1 + 18
        elif tile_string[-1] == 'z':
            num = int(tile_string[0])
            return num - 1 + 27
            
        return None

    def show_main_menu(self):
        """メインメニューを表示"""
        print("\n=== AIの推奨打牌一致率分析システム ===")
        print("1. 牌譜ファイルを選択して分析")
        print("2. 既存の分析結果を可視化")
        print("3. 終了")
        
        while True:
            choice = input("\n選択してください (1-3): ").strip()
            if choice in ['1', '2', '3']:
                return int(choice)
            print("1, 2, または 3 を入力してください")
    
    def select_xml_file(self):
        """XMLファイルを選択"""
        print("\n=== 牌譜ファイル選択 ===")
        
        # 現在のディレクトリのXMLファイルを検索
        xml_files = list(Path('.').glob('*.xml'))
        
        if not xml_files:
            print("現在のディレクトリにXMLファイルが見つかりません")
            return None
            
        print("利用可能な牌譜ファイル:")
        for i, file_path in enumerate(xml_files):
            print(f"  {i+1}: {file_path.name}")
        
        while True:
            try:
                choice = input(f"\nファイルを選択してください (1-{len(xml_files)}): ")
                index = int(choice) - 1
                if 0 <= index < len(xml_files):
                    return str(xml_files[index])
                else:
                    print(f"1から{len(xml_files)}の間で入力してください")
            except ValueError:
                print("数値を入力してください")
    
    def select_target_for_analysis(self, xml_path):
        """分析対象（局・プレイヤー）を選択"""
        interactive_system = InteractiveAnalysisSystem()
        
        # プレイヤー選択（まず局を仮選択してプレイヤー名を取得）
        temp_round_index, temp_player_id, player_names = interactive_system.show_player_selection_menu(xml_path)
        
        if temp_round_index is None:
            return None, None, None, None
        
        # 局の選択（特定局 or 全局）
        print("\n=== 分析範囲選択 ===")
        print("1. 特定の局のみ")
        print("2. 全局（1試合通して）")
        
        while True:
            choice = input("\n選択してください (1-2): ").strip()
            if choice == "1":
                round_index = temp_round_index
                all_rounds = False
                break
            elif choice == "2":
                round_index = None  # 全局の場合はNone
                all_rounds = True
                break
            else:
                print("1 または 2 を入力してください")
        
        # プレイヤーの最終確認（全局の場合は再選択）
        if all_rounds:
            print(f"\n利用可能なプレイヤー:")
            for i, name in enumerate(player_names):
                print(f"  {i}: {name}")
            print(f"  4: 全プレイヤー")
            
            while True:
                try:
                    choice = input(f"\nプレイヤーを選択してください (0-4): ")
                    if choice == "4":
                        player_id = None
                        break
                    else:
                        player_id = int(choice)
                        if 0 <= player_id < len(player_names):
                            break
                        else:
                            print(f"0から4の間で入力してください")
                except ValueError:
                    print("数値を入力してください")
        else:
            player_id = temp_player_id
            
        # 分析対象の確認表示
        print("\n=== 分析対象確認 ===")
        print(f"牌譜: {os.path.basename(xml_path)}")
        if all_rounds:
            print(f"範囲: 全局（1試合通して）")
        else:
            print(f"局: 第{round_index}局")
        
        if player_id is not None:
            print(f"プレイヤー: {player_names[player_id]}")
        else:
            print("プレイヤー: 全プレイヤー")
            
        return round_index, player_id, player_names, all_rounds
    
    def extract_tsumo_moments(self, xml_path, target_round_index=None, target_player_id=None, all_rounds=False):
        """ツモ局面を抽出"""
        if all_rounds:
            print("全局の局面抽出中...")
        else:
            print("局面抽出中...")
        
        try:
            meta, rounds_data = parse_full_mahjong_log(xml_path)
        except Exception as e:
            print(f"牌譜ファイルの解析エラー: {e}")
            return []
        
        # 処理対象局の決定
        if all_rounds:
            target_rounds = list(range(len(rounds_data)))
            print(f"全{len(rounds_data)}局を処理します")
        else:
            if not (1 <= target_round_index <= len(rounds_data)):
                print(f"不正な局インデックス: {target_round_index}")
                return []
            target_rounds = [target_round_index - 1]  # 0-indexed
            
        all_tsumo_moments = []
        
        for round_idx in target_rounds:
            if all_rounds:
                print(f"第{round_idx + 1}局処理中...")
                
            round_data = rounds_data[round_idx]
            events = round_data.get("events", [])
            
            tsumo_moments = []
            game_state = GameState()
            game_state.init_round(round_data)
            
            for i, event_xml in enumerate(events):
                tag = event_xml["tag"]
                
                # ツモイベントの検出
                tsumo_player_id = -1
                tsumo_pai_id = -1
                
                for t_tag, p_id in GameState.TSUMO_TAGS.items():
                    if tag.startswith(t_tag) and tag[1:].isdigit():
                        try:
                            tsumo_pai_id = int(tag[1:])
                            tsumo_player_id = p_id
                            break
                        except (ValueError, IndexError):
                            continue
                            
                if tsumo_player_id != -1:
                    # プレイヤーフィルタリング
                    if target_player_id is None or tsumo_player_id == target_player_id:
                        # 次の打牌を探す
                        actual_discard_info = self._find_next_discard(events, i, tsumo_player_id)
                        
                        if actual_discard_info is not None:
                            # 現在の状態をコピーして保存
                            game_state_copy = self._copy_game_state(game_state)
                            
                            moment_data = {
                                "round_index": round_idx + 1,  # 1-indexed for display
                                "tsumo_info": {
                                    "player": tsumo_player_id,
                                    "pai": tsumo_pai_id
                                },
                                "actual_discard_info": actual_discard_info,
                                "game_state": game_state_copy
                            }
                            tsumo_moments.append(moment_data)
                    
                    # 状態を更新
                    game_state.process_tsumo(tsumo_player_id, tsumo_pai_id)
                else:
                    # その他のイベントを処理
                    game_state.process_event(event_xml)
                    
            all_tsumo_moments.extend(tsumo_moments)
            
        if all_rounds:
            print(f"全局で{len(all_tsumo_moments)}局面を抽出しました")
        
        return all_tsumo_moments
    
    def _find_next_discard(self, events, current_index, player_id):
        """次の打牌を探す"""
        search_index = current_index + 1
        while search_index < len(events):
            next_event = events[search_index]
            next_tag = next_event["tag"]
            
            # リーチ宣言はスキップ
            if next_tag == "REACH":
                search_index += 1
                continue
            
            # 打牌イベントをチェック
            for d_tag, p_id_next in GameState.DISCARD_TAGS.items():
                if (next_tag.startswith(d_tag) and 
                    next_tag[1:].isdigit() and 
                    p_id_next == player_id):
                    try:
                        discard_pai_id = int(next_tag[1:])
                        tsumogiri = next_tag[0].islower()
                        return {
                            "player": p_id_next,
                            "pai": discard_pai_id,
                            "tsumogiri": tsumogiri
                        }
                    except (ValueError, IndexError):
                        continue
            
            # 他プレイヤーのイベントが来たら終了
            other_player_event = False
            for tag_prefix in ['T', 'U', 'V', 'W', 'D', 'E', 'F', 'G']:
                if (next_tag.startswith(tag_prefix) and 
                    next_tag[1:].isdigit() and 
                    GameState.TSUMO_TAGS.get(tag_prefix, GameState.DISCARD_TAGS.get(tag_prefix)) != player_id):
                    other_player_event = True
                    break
            
            if other_player_event:
                break
                
            search_index += 1
        
        return None
    
    def _copy_game_state(self, game_state):
        """ゲーム状態をコピー"""
        import copy
        return copy.deepcopy(game_state)
    
    def run_accuracy_analysis(self, xml_path, round_index=None, player_id=None, all_rounds=False):
        """一致率分析を実行"""
        print("\n=== 分析実行 ===")
        
        # ツモ局面の抽出
        tsumo_moments = self.extract_tsumo_moments(xml_path, round_index, player_id, all_rounds)
        
        if not tsumo_moments:
            print("分析対象の局面が見つかりませんでした")
            return None
            
        print(f"分析対象: {len(tsumo_moments)}局面")
        
        # 各局面の分析
        analysis_results = []
        for i, moment in enumerate(tsumo_moments):
            print(f"局面分析中 [{i+1}/{len(tsumo_moments)}]", end="\r")
            
            try:
                result = self._analyze_single_moment(moment)
                if result is not None:
                    # 局情報を追加
                    result["round_index"] = moment["round_index"]
                    analysis_results.append(result)
            except Exception as e:
                print(f"局面{i+1}の分析でエラー: {e}")
                continue
        
        print()  # 改行
        
        if not analysis_results:
            print("分析できる局面がありませんでした")
            return None
        
        # 一致率の計算
        accuracy_data = self._calculate_accuracy_metrics(analysis_results)
        
        # 全局の場合は局別統計も追加
        if all_rounds:
            accuracy_data["round_statistics"] = self._calculate_round_statistics(analysis_results)
        
        # 結果を保存
        output_dir = self._save_results(accuracy_data, xml_path, round_index, player_id, all_rounds)
        
        return output_dir, accuracy_data
    
    def _analyze_single_moment(self, moment):
        """単一局面の分析"""
        game_state = moment["game_state"]
        tsumo_info = moment["tsumo_info"]
        actual_discard_info = moment["actual_discard_info"]
        player_id = tsumo_info["player"]
        
        # ツモを実行
        game_state.process_tsumo(tsumo_info["player"], tsumo_info["pai"])
        
        # 特徴量生成
        try:
            event_sequence = game_state.get_event_sequence_features()
            static_features = game_state.get_static_features(player_id)
            event_dim = event_sequence.shape[1]
            static_dim = static_features.shape[0]
        except Exception as e:
            print(f"特徴量生成エラー: {e}")
            return None
            
        # モデルロード（初回のみ）
        if self.model is None:
            print("モデルロード中...")
            self.model = load_trained_model(DEFAULT_MODEL_PATH, event_dim, static_dim)
            
        # 予測実行
        try:
            predicted_index, predicted_prob, all_probabilities, _ = predict_discard(
                self.model, game_state, player_id, return_attention=False
            )
        except Exception as e:
            print(f"予測エラー: {e}")
            return None
        
        # 結果をまとめる
        predicted_tile_id = tile_index_to_id(predicted_index)
        actual_tile_id = actual_discard_info["pai"]
        
        # トップ10予測の取得
        top_indices = np.argsort(all_probabilities)[::-1][:10]
        top_predictions = []
        for i, idx in enumerate(top_indices):
            if 0 <= idx < NUM_TILE_TYPES:
                tile_id = tile_index_to_id(idx)
                if tile_id != -1:
                    tile_str = tile_id_to_string(tile_id)
                    if tile_str != "?":
                        top_predictions.append({
                            "rank": i + 1,
                            "tile": tile_str,
                            "probability": float(all_probabilities[idx])
                        })
        
        # 実際の打牌の順位を計算
        actual_rank = None
        for i, pred in enumerate(top_predictions):
            if pred["tile"] == tile_id_to_string(actual_tile_id):
                actual_rank = i + 1
                break
        
        # 受け入れ比較分析
        ukeire_comparison = self._analyze_ukeire_comparison(
            game_state, player_id, 
            tile_id_to_string(predicted_tile_id),
            tile_id_to_string(actual_tile_id)
        )
        
        return {
            "predicted_tile": tile_id_to_string(predicted_tile_id),
            "actual_tile": tile_id_to_string(actual_tile_id),
            "confidence": float(predicted_prob),
            "correct": predicted_tile_id == actual_tile_id,
            "actual_rank": actual_rank,
            "top_predictions": top_predictions,
            "ukeire_comparison": ukeire_comparison
        }
    
    def _analyze_ukeire_comparison(self, game_state, player_id, predicted_tile_str, actual_tile_str):
        """AIの推奨打牌と実際の打牌の受け入れ比較"""
        try:
            # 現在の手牌を34種配列に変換
            hand_tiles_34 = self.convert_game_hand_to_34_array(game_state, player_id)
            
            # 予測打牌と実際打牌のインデックスを取得
            predicted_index = self.tile_string_to_34_index(predicted_tile_str)
            actual_index = self.tile_string_to_34_index(actual_tile_str)
            
            if predicted_index is None or actual_index is None:
                return {
                    "error": "牌変換エラー",
                    "predicted_tile": predicted_tile_str,
                    "actual_tile": actual_tile_str
                }
            
            # 各打牌の受け入れ分析
            predicted_analysis = self.analyze_discard_ukeire(hand_tiles_34, predicted_index)
            actual_analysis = self.analyze_discard_ukeire(hand_tiles_34, actual_index)
            
            if predicted_analysis is None or actual_analysis is None:
                return {
                    "error": "受け入れ分析エラー",
                    "predicted_tile": predicted_tile_str,
                    "actual_tile": actual_tile_str
                }
            
            # 受け入れ牌の文字列化
            predicted_ukeire_str = self.format_tiles_for_display(
                sorted(predicted_analysis["ukeire_tiles"].keys())
            )
            actual_ukeire_str = self.format_tiles_for_display(
                sorted(actual_analysis["ukeire_tiles"].keys())
            )
            
            return {
                "predicted": {
                    "tile": predicted_tile_str,
                    "shanten": self.format_shanten(predicted_analysis["shanten_after_discard"]),
                    "ukeire_tiles": predicted_ukeire_str,
                    "ukeire_count": predicted_analysis["total_ukeire_count"]
                },
                "actual": {
                    "tile": actual_tile_str,
                    "shanten": self.format_shanten(actual_analysis["shanten_after_discard"]),
                    "ukeire_tiles": actual_ukeire_str,
                    "ukeire_count": actual_analysis["total_ukeire_count"]
                },
                "comparison": {
                    "ukeire_difference": predicted_analysis["total_ukeire_count"] - actual_analysis["total_ukeire_count"],
                    "better_choice": "AI推奨" if predicted_analysis["total_ukeire_count"] > actual_analysis["total_ukeire_count"] 
                                   else "実際打牌" if actual_analysis["total_ukeire_count"] > predicted_analysis["total_ukeire_count"]
                                   else "同等"
                }
            }
            
        except Exception as e:
            return {
                "error": f"受け入れ比較エラー: {e}",
                "predicted_tile": predicted_tile_str,
                "actual_tile": actual_tile_str
            }
    
    def _calculate_accuracy_metrics(self, analysis_results):
        """一致率メトリクスを計算"""
        total = len(analysis_results)
        
        # 1位一致率
        exact_matches = sum(1 for r in analysis_results if r["correct"])
        exact_accuracy = exact_matches / total
        
        # 3位以内一致率
        top3_matches = sum(1 for r in analysis_results 
                          if r["actual_rank"] is not None and r["actual_rank"] <= 3)
        top3_accuracy = top3_matches / total
        
        # 5位以内一致率
        top5_matches = sum(1 for r in analysis_results 
                          if r["actual_rank"] is not None and r["actual_rank"] <= 5)
        top5_accuracy = top5_matches / total
        
        # 順位分布
        rank_distribution = Counter()
        for r in analysis_results:
            if r["actual_rank"] is not None:
                rank_distribution[r["actual_rank"]] += 1
            else:
                rank_distribution["圏外"] += 1
        
        # 信頼度別分析
        confidence_bins = {"高信頼度(>0.5)": 0, "中信頼度(0.2-0.5)": 0, "低信頼度(<0.2)": 0}
        confidence_correct = {"高信頼度(>0.5)": 0, "中信頼度(0.2-0.5)": 0, "低信頼度(<0.2)": 0}
        
        for r in analysis_results:
            conf = r["confidence"]
            if conf > 0.5:
                bin_name = "高信頼度(>0.5)"
            elif conf > 0.2:
                bin_name = "中信頼度(0.2-0.5)"
            else:
                bin_name = "低信頼度(<0.2)"
                
            confidence_bins[bin_name] += 1
            if r["correct"]:
                confidence_correct[bin_name] += 1
        
        confidence_accuracy = {}
        for bin_name in confidence_bins:
            if confidence_bins[bin_name] > 0:
                confidence_accuracy[bin_name] = confidence_correct[bin_name] / confidence_bins[bin_name]
            else:
                confidence_accuracy[bin_name] = 0.0
        
        return {
            "total_moments": total,
            "analysis_results": analysis_results,
            "metrics": {
                "exact_accuracy": exact_accuracy,
                "top3_accuracy": top3_accuracy,
                "top5_accuracy": top5_accuracy,
                "exact_matches": exact_matches,
                "top3_matches": top3_matches,
                "top5_matches": top5_matches,
                "rank_distribution": dict(rank_distribution),
                "confidence_accuracy": confidence_accuracy,
                "average_confidence": np.mean([r["confidence"] for r in analysis_results])
            }
        }
    
    def _calculate_round_statistics(self, analysis_results):
        """局別統計を計算"""
        round_stats = {}
        
        # 局別にグループ化
        rounds_data = {}
        for result in analysis_results:
            round_idx = result["round_index"]
            if round_idx not in rounds_data:
                rounds_data[round_idx] = []
            rounds_data[round_idx].append(result)
        
        # 各局の統計を計算
        for round_idx, round_results in rounds_data.items():
            total = len(round_results)
            exact_matches = sum(1 for r in round_results if r["correct"])
            top3_matches = sum(1 for r in round_results 
                             if r["actual_rank"] is not None and r["actual_rank"] <= 3)
            avg_confidence = np.mean([r["confidence"] for r in round_results])
            
            round_stats[f"第{round_idx}局"] = {
                "total_moments": total,
                "exact_accuracy": exact_matches / total if total > 0 else 0,
                "top3_accuracy": top3_matches / total if total > 0 else 0,
                "average_confidence": avg_confidence
            }
        
        return round_stats
    
    def _save_results(self, accuracy_data, xml_path, round_index=None, player_id=None, all_rounds=False):
        """結果を保存"""
        # 出力ディレクトリの作成
        xml_name = Path(xml_path).stem
        player_str = f"P{player_id}" if player_id is not None else "ALL"
        
        if all_rounds:
            round_str = "ALL_ROUNDS"
        else:
            round_str = f"R{round_index}"
            
        output_dir = Path("accuracy_results") / f"{xml_name}_{round_str}_{player_str}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON形式で保存
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "xml_file": xml_path,
            "round_index": round_index,
            "player_id": player_id,
            "all_rounds": all_rounds,
            **accuracy_data
        }
        
        result_file = output_dir / "accuracy_analysis.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
        
        # 概要テキスト形式で保存
        summary_file = output_dir / "分析結果概要.txt"
        self._save_summary_text(accuracy_data, summary_file, xml_path, round_index, player_id, all_rounds)
            
        print(f"結果を保存しました: {result_file}")
        print(f"概要を保存しました: {summary_file}")
        return output_dir
    
    def _save_summary_text(self, accuracy_data, summary_file, xml_path, round_index, player_id, all_rounds):
        """概要テキストファイルを保存"""
        metrics = accuracy_data["metrics"]
        total = accuracy_data["total_moments"]
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=== AIの推奨打牌と実際の打牌の一致率分析結果 ===\n")
            f.write(f"分析日時: {datetime.now().strftime('%Y年%m月%d日 %H時%M分%S秒')}\n")
            f.write(f"牌譜ファイル: {Path(xml_path).name}\n")
            
            if all_rounds:
                f.write("分析範囲: 全局（1試合通して）\n")
            else:
                f.write(f"分析範囲: 第{round_index}局\n")
            
            if player_id is not None:
                f.write(f"対象プレイヤー: プレイヤー{player_id}\n")
            else:
                f.write("対象プレイヤー: 全プレイヤー\n")
            
            f.write("\n" + "="*50 + "\n")
            
            if all_rounds:
                f.write("■ 1試合全局の分析結果\n")
                f.write(f"プレイヤー{player_id}の全打牌分析：\n")
            else:
                f.write("■ 分析結果\n")
            
            f.write(f"・総局面数: {total}局面")
            if all_rounds:
                # 局数を計算
                round_stats = accuracy_data.get("round_statistics", {})
                num_rounds = len(round_stats)
                f.write(f"（全{num_rounds}局合計）")
            f.write("\n")
            
            f.write(f"・平均信頼度: {metrics['average_confidence']:.3f}\n")
            f.write("\n")
            
            f.write("【一致率】\n")
            f.write(f"・1位一致率:     {metrics['exact_accuracy']:.1%} ({metrics['exact_matches']}/{total})\n")
            f.write(f"・3位以内一致率: {metrics['top3_accuracy']:.1%} ({metrics['top3_matches']}/{total})\n")
            f.write(f"・5位以内一致率: {metrics['top5_accuracy']:.1%} ({metrics['top5_matches']}/{total})\n")
            f.write("\n")
            
            f.write("【信頼度別正解率】\n")
            for conf_level, accuracy in metrics["confidence_accuracy"].items():
                f.write(f"・{conf_level}: {accuracy:.1%}\n")
            f.write("\n")
            
            f.write("【実際の打牌の予測順位分布】\n")
            rank_dist = metrics["rank_distribution"]
            for rank in sorted([k for k in rank_dist.keys() if isinstance(k, int)]):
                f.write(f"・{rank}位: {rank_dist[rank]}回\n")
            if "圏外" in rank_dist:
                f.write(f"・圏外: {rank_dist['圏外']}回\n")
            
            # 局別統計の保存（全局分析の場合）
            if "round_statistics" in accuracy_data:
                f.write("\n" + "="*50 + "\n")
                f.write("■ 局別詳細統計\n")
                round_stats = accuracy_data["round_statistics"]
                for round_name, stats in round_stats.items():
                    f.write(f"・{round_name}: ")
                    f.write(f"1位{stats['exact_accuracy']:.1%} / ")
                    f.write(f"3位以内{stats['top3_accuracy']:.1%} / ")
                    f.write(f"信頼度{stats['average_confidence']:.3f} ")
                    f.write(f"({stats['total_moments']}局面)\n")
            
            f.write("\n" + "="*50 + "\n")
            f.write("※ この結果は AI の予測と実際の打牌を比較したものです\n")
            f.write("※ 1位一致率: AIの最推奨打牌と実際の打牌が一致した割合\n")
            f.write("※ 3位以内一致率: 実際の打牌がAIの上位3候補以内に含まれた割合\n")
    
    def display_accuracy_results(self, accuracy_data):
        """一致率分析結果を表示"""
        metrics = accuracy_data["metrics"]
        total = accuracy_data["total_moments"]
        
        print(f"\n=== 一致率分析結果 ===")
        print(f"総局面数: {total}")
        print(f"平均信頼度: {metrics['average_confidence']:.3f}")
        print()
        
        print("【一致率】")
        print(f"1位一致率:     {metrics['exact_accuracy']:.1%} ({metrics['exact_matches']}/{total})")
        print(f"3位以内一致率: {metrics['top3_accuracy']:.1%} ({metrics['top3_matches']}/{total})")
        print(f"5位以内一致率: {metrics['top5_accuracy']:.1%} ({metrics['top5_matches']}/{total})")
        print()
        
        print("【信頼度別正解率】")
        for conf_level, accuracy in metrics["confidence_accuracy"].items():
            print(f"{conf_level}: {accuracy:.1%}")
        print()
        
        print("【実際の打牌の予測順位分布】")
        rank_dist = metrics["rank_distribution"]
        for rank in sorted([k for k in rank_dist.keys() if isinstance(k, int)]):
            print(f"{rank}位: {rank_dist[rank]}回")
        if "圏外" in rank_dist:
            print(f"圏外: {rank_dist['圏外']}回")
        
        # 局別統計の表示（全局分析の場合）
        if "round_statistics" in accuracy_data:
            print("\n【局別統計】")
            round_stats = accuracy_data["round_statistics"]
            for round_name, stats in round_stats.items():
                print(f"{round_name}: 1位{stats['exact_accuracy']:.1%} 3位以内{stats['top3_accuracy']:.1%} "
                      f"信頼度{stats['average_confidence']:.3f} ({stats['total_moments']}局面)")
    
    def create_accuracy_visualizations(self, accuracy_data, result_dir):
        """一致率分析の可視化を作成"""
        metrics = accuracy_data["metrics"]
        analysis_results = accuracy_data["analysis_results"]
        
        # 図のスタイル設定
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AI推奨打牌 一致率分析', fontsize=16, fontweight='bold')
        
        # 1. 一致率比較バープロット
        ax1 = axes[0, 0]
        categories = ['1位一致', '3位以内', '5位以内']
        accuracies = [metrics['exact_accuracy'], metrics['top3_accuracy'], metrics['top5_accuracy']]
        colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
        
        bars = ax1.bar(categories, accuracies, color=colors, alpha=0.7)
        ax1.set_ylabel('一致率')
        ax1.set_title('一致率比較')
        ax1.set_ylim(0, 1)
        
        # バーの上に数値を表示
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom')
        
        # 2. 予測順位分布
        ax2 = axes[0, 1]
        rank_dist = metrics["rank_distribution"]
        ranks = sorted([k for k in rank_dist.keys() if isinstance(k, int)])
        counts = [rank_dist[r] for r in ranks]
        
        ax2.bar(ranks, counts, alpha=0.7)
        ax2.set_xlabel('予測順位')
        ax2.set_ylabel('回数')
        ax2.set_title('実際の打牌の予測順位分布')
        ax2.set_xticks(ranks)
        
        # 3. 信頼度別正解率
        ax3 = axes[1, 0]
        conf_levels = list(metrics["confidence_accuracy"].keys())
        conf_accuracies = list(metrics["confidence_accuracy"].values())
        
        ax3.bar(conf_levels, conf_accuracies, alpha=0.7, color='purple')
        ax3.set_ylabel('正解率')
        ax3.set_title('信頼度別正解率')
        ax3.set_ylim(0, 1)
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # バーの上に数値を表示
        for i, acc in enumerate(conf_accuracies):
            ax3.text(i, acc + 0.01, f'{acc:.1%}', ha='center', va='bottom')
        
        # 4. 局面ごとの信頼度と正解の散布図
        ax4 = axes[1, 1]
        confidences = [r["confidence"] for r in analysis_results]
        correct_flags = [1 if r["correct"] else 0 for r in analysis_results]
        
        # 正解・不正解で色分け
        correct_conf = [c for c, correct in zip(confidences, correct_flags) if correct]
        incorrect_conf = [c for c, correct in zip(confidences, correct_flags) if not correct]
        
        ax4.scatter(range(len(correct_conf)), correct_conf, 
                   alpha=0.6, label='正解', color='green')
        ax4.scatter(range(len(correct_conf), len(confidences)), incorrect_conf, 
                   alpha=0.6, label='不正解', color='red')
        
        ax4.set_xlabel('局面')
        ax4.set_ylabel('信頼度')
        ax4.set_title('局面ごとの予測信頼度')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_file = Path(result_dir) / "accuracy_plots.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"可視化グラフを保存しました: {output_file}")
    
    def select_existing_results(self):
        """既存の分析結果を選択"""
        results_dir = Path("accuracy_results")
        
        if not results_dir.exists():
            print("分析結果ディレクトリが見つかりません")
            return None
            
        # 分析結果ディレクトリを検索
        result_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
        
        if not result_dirs:
            print("分析結果が見つかりません")
            return None
            
        print("\n=== 既存の分析結果 ===")
        for i, result_dir in enumerate(result_dirs):
            print(f"  {i+1}: {result_dir.name}")
        
        while True:
            try:
                choice = input(f"\n結果を選択してください (1-{len(result_dirs)}): ")
                index = int(choice) - 1
                if 0 <= index < len(result_dirs):
                    return result_dirs[index]
                else:
                    print(f"1から{len(result_dirs)}の間で入力してください")
            except ValueError:
                print("数値を入力してください")
    
    def load_existing_accuracy_data(self, result_dir):
        """既存の一致率分析データを読み込み"""
        accuracy_file = result_dir / "accuracy_analysis.json"
        
        if not accuracy_file.exists():
            print("分析結果ファイルが見つかりません")
            return None
        
        with open(accuracy_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return data
    
    def run(self):
        """メインの実行ループ"""
        while True:
            choice = self.show_main_menu()
            
            if choice == 1:
                # 新規分析
                xml_path = self.select_xml_file()
                if xml_path is None:
                    continue
                
                analysis_target = self.select_target_for_analysis(xml_path)
                if analysis_target[0] is None and not analysis_target[3]:  # round_index がNoneかつall_roundsがFalse
                    continue
                
                round_index, player_id, player_names, all_rounds = analysis_target
                
                result = self.run_accuracy_analysis(xml_path, round_index, player_id, all_rounds)
                if result is not None:
                    result_dir, accuracy_data = result
                    self.display_accuracy_results(accuracy_data)
                    self.create_accuracy_visualizations(accuracy_data, result_dir)
                    
            elif choice == 2:
                # 既存結果の可視化
                result_dir = self.select_existing_results()
                if result_dir is None:
                    continue
                
                accuracy_data = self.load_existing_accuracy_data(result_dir)
                if accuracy_data is not None:
                    self.display_accuracy_results(accuracy_data)
                    self.create_accuracy_visualizations(accuracy_data, result_dir)
                    
            elif choice == 3:
                print("終了します")
                break


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="AIの推奨打牌と実際の打牌の一致率分析")
    parser.add_argument("--xml_file", help="特定のXMLファイルを指定")
    parser.add_argument("--round_index", type=int, help="特定の局を指定")
    parser.add_argument("--player_id", type=int, choices=[0, 1, 2, 3], help="特定のプレイヤーを指定")
    parser.add_argument("--all_rounds", action="store_true", help="全局を分析")
    
    args = parser.parse_args()
    
    system = AccuracyAnalysisSystem()
    
    if args.xml_file:
        # コマンドライン引数で直接実行
        if args.all_rounds:
            result = system.run_accuracy_analysis(args.xml_file, None, args.player_id, True)
        elif args.round_index:
            result = system.run_accuracy_analysis(args.xml_file, args.round_index, args.player_id, False)
        else:
            print("--round_index または --all_rounds を指定してください")
            return
            
        if result is not None:
            result_dir, accuracy_data = result
            system.display_accuracy_results(accuracy_data)
            system.create_accuracy_visualizations(accuracy_data, result_dir)
    else:
        # インタラクティブモード
        system.run()


if __name__ == "__main__":
    main() 