# interactive_analysis.py - インタラクティブな分析・可視化システム
import os
import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict, Counter

try:
    from full_mahjong_parser import parse_full_mahjong_log
    from game_state import GameState, NUM_PLAYERS
    from tile_utils import tile_id_to_string
    from batch_analysis import BatchAnalysisSystem
except ImportError as e:
    print(f"[エラー] プロジェクトモジュールのインポートに失敗しました: {e}")
    exit(1)

class InteractiveAnalysisSystem:
    """インタラクティブな分析・可視化システム"""
    
    def __init__(self):
        self.plt_japanese_support()
        
    def plt_japanese_support(self):
        """Matplotlibで日本語表示をサポート"""
        try:
            plt.rcParams['font.family'] = 'DejaVu Sans'
            # 必要に応じて日本語フォントを設定
            # plt.rcParams['font.family'] = 'IPAexGothic'
        except:
            print("[警告] 日本語フォントの設定に失敗しました")
    
    def show_player_selection_menu(self, xml_path):
        """プレイヤー選択メニューを表示"""
        print("\n=== 麻雀AI分析システム ===")
        print(f"牌譜ファイル: {os.path.basename(xml_path)}")
        
        try:
            meta, rounds_data = parse_full_mahjong_log(xml_path)
        except Exception as e:
            print(f"[エラー] 牌譜ファイルの読み込みに失敗: {e}")
            return None, None, None
            
        # プレイヤー名の表示
        player_names = meta.get('player_names', [f"プレイヤー{i}" for i in range(NUM_PLAYERS)])
        print(f"\n利用可能な局数: {len(rounds_data)}局")
        print("\nプレイヤー一覧:")
        for i, name in enumerate(player_names):
            print(f"  {i}: {name}")
            
        # 局の選択
        while True:
            try:
                round_input = input(f"\n分析する局を選択してください (1-{len(rounds_data)}): ")
                round_index = int(round_input)
                if 1 <= round_index <= len(rounds_data):
                    break
                else:
                    print(f"1から{len(rounds_data)}の間で入力してください")
            except ValueError:
                print("数値を入力してください")
                
        # プレイヤーの選択
        print("\nプレイヤー選択:")
        print("  0-3: 特定のプレイヤー")
        print("  A: 全プレイヤー")
        
        while True:
            player_input = input("選択してください: ").strip().upper()
            if player_input == 'A':
                player_id = None
                break
            elif player_input in ['0', '1', '2', '3']:
                player_id = int(player_input)
                break
            else:
                print("0, 1, 2, 3, または A を入力してください")
                
        return round_index, player_id, player_names
    
    def show_round_overview(self, xml_path, round_index):
        """局の概要を表示"""
        try:
            meta, rounds_data = parse_full_mahjong_log(xml_path)
            round_data = rounds_data[round_index - 1]
            
            print(f"\n=== 第{round_index}局 概要 ===")
            
            # 初期状態の構築
            game_state = GameState()
            game_state.init_round(round_data)
            
            # 局情報
            print(f"局風: {['東', '南', '西', '北'][game_state.round_num_wind // NUM_PLAYERS]}")
            print(f"局数: {(game_state.round_num_wind % NUM_PLAYERS) + 1}")
            print(f"親: プレイヤー{game_state.dealer}")
            print(f"本場: {game_state.honba}")
            print(f"供託: {game_state.kyotaku}")
            
            # 初期点数
            print("\n初期点数:")
            player_names = meta.get('player_names', [f"P{i}" for i in range(NUM_PLAYERS)])
            for i in range(NUM_PLAYERS):
                print(f"  {player_names[i]}: {game_state.current_scores[i]:,}点")
                
            # ドラ表示牌
            print(f"\nドラ表示: {' '.join([tile_id_to_string(t) for t in game_state.dora_indicators])}")
            
            # ツモ数の概算
            events = round_data.get("events", [])
            tsumo_count = sum(1 for event in events 
                            if any(event["tag"].startswith(tag) and event["tag"][1:].isdigit() 
                                  for tag in GameState.TSUMO_TAGS.keys()))
            print(f"総ツモ数: {tsumo_count}")
            
            return True
            
        except Exception as e:
            print(f"[エラー] 局概要の表示に失敗: {e}")
            return False
    
    def run_analysis_with_confirmation(self, xml_path, round_index, player_id, output_dir):
        """確認後に分析を実行"""
        # 推定処理時間の計算
        try:
            meta, rounds_data = parse_full_mahjong_log(xml_path)
            round_data = rounds_data[round_index - 1]
            events = round_data.get("events", [])
            
            if player_id is not None:
                # 特定プレイヤーのツモ数を計算
                tsumo_count = 0
                for event in events:
                    for tag, p_id in GameState.TSUMO_TAGS.items():
                        if event["tag"].startswith(tag) and event["tag"][1:].isdigit() and p_id == player_id:
                            tsumo_count += 1
                            break
            else:
                # 全プレイヤーのツモ数
                tsumo_count = sum(1 for event in events 
                                if any(event["tag"].startswith(tag) and event["tag"][1:].isdigit() 
                                      for tag in GameState.TSUMO_TAGS.keys()))
                                      
            estimated_time = tsumo_count * 5  # 1局面あたり約5秒と仮定
            
            print(f"\n=== 分析実行確認 ===")
            print(f"対象局面数: {tsumo_count}")
            print(f"推定処理時間: {estimated_time // 60}分{estimated_time % 60}秒")
            print(f"出力先: {output_dir}")
            
            confirm = input("\n分析を開始しますか？ (y/n): ").strip().lower()
            if confirm not in ['y', 'yes']:
                print("分析をキャンセルしました")
                return None
                
        except Exception as e:
            print(f"[警告] 推定時間の計算に失敗: {e}")
            
        # 分析実行
        print("\n分析を開始します...")
        system = BatchAnalysisSystem(output_base_dir=output_dir)
        
        try:
            result_dir = system.run_batch_analysis(xml_path, round_index, player_id)
            return result_dir
        except Exception as e:
            print(f"[エラー] 分析中にエラーが発生しました: {e}")
            return None
    
    def visualize_results(self, result_dir):
        """分析結果の可視化"""
        result_path = Path(result_dir)
        summary_file = result_path / "overall_summary.json"
        
        if not summary_file.exists():
            print(f"[エラー] サマリファイルが見つかりません: {summary_file}")
            return
            
        # サマリデータの読み込み
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
            
        overview = summary["overview"]
        moments = summary["moments"]
        
        print(f"\n=== 分析結果サマリ ===")
        print(f"総局面数: {overview['total_moments']}")
        print(f"正解数: {overview['correct_predictions']}")
        print(f"正解率: {overview['accuracy']:.1%}")
        print(f"平均信頼度: {overview['average_confidence']:.1%}")
        
        # 可視化の作成
        self._create_accuracy_plot(moments, result_path)
        self._create_confidence_distribution(moments, result_path)
        self._create_tile_prediction_analysis(result_path)
        
        print(f"\n可視化グラフが {result_path} に保存されました")
    
    def _create_accuracy_plot(self, moments, result_path):
        """正解率の推移をプロット"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 局面ごとの正解/不正解
        tsumo_counts = [m["tsumo_count"] for m in moments]
        correct_flags = [m["correct"] for m in moments]
        colors = ['green' if c else 'red' for c in correct_flags]
        
        ax1.scatter(tsumo_counts, [1 if c else 0 for c in correct_flags], 
                   c=colors, alpha=0.7, s=50)
        ax1.set_xlabel("ツモ巡目")
        ax1.set_ylabel("正解 (1) / 不正解 (0)")
        ax1.set_title("局面別予測結果")
        ax1.grid(True, alpha=0.3)
        
        # 信頼度分布
        confidences = [m["confidence"] for m in moments]
        correct_conf = [c for c, correct in zip(confidences, correct_flags) if correct]
        incorrect_conf = [c for c, correct in zip(confidences, correct_flags) if not correct]
        
        ax2.hist(correct_conf, bins=20, alpha=0.7, label='正解', color='green', density=True)
        ax2.hist(incorrect_conf, bins=20, alpha=0.7, label='不正解', color='red', density=True)
        ax2.set_xlabel("予測信頼度")
        ax2.set_ylabel("密度")
        ax2.set_title("信頼度分布")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(result_path / "accuracy_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_confidence_distribution(self, moments, result_path):
        """信頼度の詳細分析"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        confidences = [m["confidence"] for m in moments]
        tsumo_counts = [m["tsumo_count"] for m in moments]
        correct_flags = [m["correct"] for m in moments]
        
        # 信頼度の時系列変化
        ax1.plot(tsumo_counts, confidences, 'bo-', alpha=0.6, markersize=4)
        ax1.set_xlabel("ツモ巡目")
        ax1.set_ylabel("予測信頼度")
        ax1.set_title("信頼度の時系列変化")
        ax1.grid(True, alpha=0.3)
        
        # 正解率と信頼度の関係
        conf_bins = pd.cut(confidences, bins=10)
        accuracy_by_conf = pd.DataFrame({
            'confidence': confidences,
            'correct': correct_flags,
            'conf_bin': conf_bins
        }).groupby('conf_bin')['correct'].mean()
        
        ax2.bar(range(len(accuracy_by_conf)), accuracy_by_conf.values, alpha=0.7)
        ax2.set_xlabel("信頼度区間")
        ax2.set_ylabel("正解率")
        ax2.set_title("信頼度別正解率")
        ax2.set_xticks(range(len(accuracy_by_conf)))
        ax2.set_xticklabels([f"{interval.left:.2f}-{interval.right:.2f}" 
                           for interval in accuracy_by_conf.index], rotation=45)
        
        # 累積正解率
        cumulative_correct = pd.Series(correct_flags).cumsum()
        cumulative_total = pd.Series(range(1, len(correct_flags) + 1))
        cumulative_accuracy = cumulative_correct / cumulative_total
        
        ax3.plot(tsumo_counts, cumulative_accuracy, 'g-', linewidth=2)
        ax3.set_xlabel("ツモ巡目")
        ax3.set_ylabel("累積正解率")
        ax3.set_title("累積正解率の推移")
        ax3.grid(True, alpha=0.3)
        
        # 予測牌の分布
        predicted_tiles = [m["predicted_tile"] for m in moments]
        tile_counts = Counter(predicted_tiles)
        top_tiles = tile_counts.most_common(10)
        
        ax4.bar([t[0] for t in top_tiles], [t[1] for t in top_tiles])
        ax4.set_xlabel("予測牌")
        ax4.set_ylabel("回数")
        ax4.set_title("予測牌分布 (Top 10)")
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(result_path / "confidence_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_tile_prediction_analysis(self, result_path):
        """牌種別の予測分析"""
        # 各局面の詳細データを読み込み
        moment_dirs = [d for d in result_path.iterdir() if d.is_dir() and d.name.startswith("tsumo_")]
        
        tile_analysis = defaultdict(lambda: {"correct": 0, "total": 0, "confidences": []})
        
        for moment_dir in moment_dirs:
            summary_file = moment_dir / "summary.json"
            if summary_file.exists():
                with open(summary_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                predicted_tile = data["predicted_tile"]
                actual_tile = data["actual_tile"]
                confidence = data["predicted_probability"]
                correct = data["match"]
                
                tile_analysis[predicted_tile]["total"] += 1
                tile_analysis[predicted_tile]["confidences"].append(confidence)
                if correct:
                    tile_analysis[predicted_tile]["correct"] += 1
        
        # 可視化
        if tile_analysis:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 牌種別正解率
            tiles = list(tile_analysis.keys())
            accuracies = [tile_analysis[tile]["correct"] / tile_analysis[tile]["total"] 
                         for tile in tiles]
            totals = [tile_analysis[tile]["total"] for tile in tiles]
            
            # 出現回数でフィルタリング（2回以上）
            filtered_data = [(tile, acc, total) for tile, acc, total in zip(tiles, accuracies, totals) if total >= 2]
            
            if filtered_data:
                filtered_tiles, filtered_acc, filtered_totals = zip(*filtered_data)
                
                bars = ax1.bar(range(len(filtered_tiles)), filtered_acc, 
                             color=['green' if acc > 0.5 else 'red' for acc in filtered_acc])
                ax1.set_xlabel("予測牌")
                ax1.set_ylabel("正解率")
                ax1.set_title("牌種別正解率 (2回以上予測)")
                ax1.set_xticks(range(len(filtered_tiles)))
                ax1.set_xticklabels(filtered_tiles, rotation=45)
                ax1.set_ylim(0, 1)
                
                # 各バーに出現回数を表示
                for i, (bar, total) in enumerate(zip(bars, filtered_totals)):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{total}回', ha='center', va='bottom', fontsize=8)
            
            # 牌種別平均信頼度
            avg_confidences = [sum(tile_analysis[tile]["confidences"]) / len(tile_analysis[tile]["confidences"])
                             for tile in tiles if tile_analysis[tile]["total"] >= 2]
            
            if avg_confidences:
                ax2.bar(range(len(filtered_tiles)), avg_confidences)
                ax2.set_xlabel("予測牌")
                ax2.set_ylabel("平均信頼度")
                ax2.set_title("牌種別平均信頼度")
                ax2.set_xticks(range(len(filtered_tiles)))
                ax2.set_xticklabels(filtered_tiles, rotation=45)
            
            plt.tight_layout()
            plt.savefig(result_path / "tile_analysis.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    def export_for_llm(self, result_dir, output_file=None):
        """LLM用のプロンプトファイルをまとめてエクスポート"""
        result_path = Path(result_dir)
        
        if output_file is None:
            output_file = result_path / "all_prompts.txt"
        
        moment_dirs = sorted([d for d in result_path.iterdir() 
                            if d.is_dir() and d.name.startswith("tsumo_")])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== 麻雀AI分析プロンプト集 ===\n")
            f.write(f"生成日時: {datetime.now().isoformat()}\n")
            f.write(f"ソース: {result_dir}\n")
            f.write("="*50 + "\n\n")
            
            for i, moment_dir in enumerate(moment_dirs):
                prompt_file = moment_dir / "prompt.txt"
                summary_file = moment_dir / "summary.json"
                
                if prompt_file.exists() and summary_file.exists():
                    # サマリ情報の読み込み
                    with open(summary_file, 'r', encoding='utf-8') as sf:
                        summary = json.load(sf)
                    
                    f.write(f"### 局面 {i+1}: ツモ{summary['tsumo_count']} ###\n")
                    f.write(f"予測: {summary['predicted_tile']} (信頼度: {summary['predicted_probability']:.1%})\n")
                    f.write(f"実際: {summary['actual_tile']}\n")
                    f.write(f"正解: {'○' if summary['match'] else '×'}\n")
                    f.write("-" * 30 + "\n\n")
                    
                    # プロンプト内容
                    with open(prompt_file, 'r', encoding='utf-8') as pf:
                        f.write(pf.read())
                    
                    f.write("\n\n" + "="*50 + "\n\n")
        
        print(f"LLM用プロンプトファイルを出力しました: {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(description="インタラクティブな麻雀AI分析システム")
    parser.add_argument("xml_file", help="天鳳形式のXML牌譜ファイルへのパス")
    parser.add_argument("--output_dir", default="analysis_results", 
                       help="出力ディレクトリ")
    parser.add_argument("--auto", action='store_true', 
                       help="非インタラクティブモード（全ての選択をデフォルトで実行）")
    parser.add_argument("--round", type=int, help="局番号（autoモード用）")
    parser.add_argument("--player", type=int, choices=[0,1,2,3], help="プレイヤーID（autoモード用）")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.xml_file):
        print(f"[エラー] ファイルが見つかりません: {args.xml_file}")
        exit(1)
    
    system = InteractiveAnalysisSystem()
    
    try:
        if args.auto:
            # 自動モード
            round_index = args.round or 1
            player_id = args.player
            print(f"自動モード: 局{round_index}, プレイヤー{player_id}")
        else:
            # インタラクティブモード
            round_index, player_id, player_names = system.show_player_selection_menu(args.xml_file)
            if round_index is None:
                exit(1)
            
            # 局概要の表示
            system.show_round_overview(args.xml_file, round_index)
        
        # 分析実行
        if args.auto:
            # 自動実行
            batch_system = BatchAnalysisSystem(output_base_dir=args.output_dir)
            result_dir = batch_system.run_batch_analysis(args.xml_file, round_index, player_id)
        else:
            # 確認付き実行
            result_dir = system.run_analysis_with_confirmation(
                args.xml_file, round_index, player_id, args.output_dir
            )
        
        if result_dir:
            # 結果の可視化
            print("\n結果を可視化しています...")
            system.visualize_results(result_dir)
            
            # LLM用エクスポート
            llm_file = system.export_for_llm(result_dir)
            
            print(f"\n=== 完了 ===")
            print(f"結果ディレクトリ: {result_dir}")
            print(f"可視化グラフ: {result_dir}/*.png")
            print(f"LLM用プロンプト: {llm_file}")
            
        else:
            print("分析が完了しませんでした")
            
    except KeyboardInterrupt:
        print("\n\n処理を中断しました")
    except Exception as e:
        print(f"[エラー] 予期せぬエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 