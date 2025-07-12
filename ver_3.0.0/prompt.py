# prompt.py - 麻雀AI分析結果からLLMプロンプトを生成
import json
import argparse
import sys
import os
from datetime import datetime

# shanten.pyから受け入れ分析に必要な関数をインポート
from shanten import (
    robust_hand_parser, 
    format_tiles_for_display,
    format_shanten,
    get_shanten_after_best_discard
)
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter

def load_analysis_result(json_file):
    """分析結果JSONファイルを読み込み"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {json_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"エラー: JSONファイルの読み込みに失敗しました: {e}")
        sys.exit(1)





def format_hand_composition(hand_tiles):
    """手牌構成を整理して表示"""
    if not hand_tiles or hand_tiles == "hidden":
        return "非公開"
    
    # 牌をソートして一行で表示
    # 萬子、筒子、索子、字牌の順番でソート
    def sort_key(tile):
        if 'm' in tile:
            return (0, int(tile.replace('m', '').replace('0', '5')))  # 0m -> 5m扱い
        elif 'p' in tile:
            return (1, int(tile.replace('p', '').replace('0', '5')))  # 0p -> 5p扱い
        elif 's' in tile:
            return (2, int(tile.replace('s', '').replace('0', '5')))  # 0s -> 5s扱い
        elif tile in ['東', '南', '西', '北']:
            return (3, ['東', '南', '西', '北'].index(tile))
        elif tile in ['白', '發', '中']:
            return (4, ['白', '發', '中'].index(tile))
        else:
            return (5, 0)  # その他
    
    sorted_tiles = sorted(hand_tiles, key=sort_key)
    return ' '.join(sorted_tiles)

def format_all_players_discards(players_state, current_player):
    """全プレイヤーの捨て牌情報を整理して表示"""
    discard_info = []
    for i in range(4):  # NUM_PLAYERS = 4
        player_data = players_state.get(f'player_{i}', {})
        discards = player_data.get('discards', [])
        
        if discards:
            discard_tiles = []
            for discard in discards:
                tile = discard.get('tile', '')
                tsumogiri = discard.get('tsumogiri', False)
                if tsumogiri:
                    tile += '*'  # ツモ切りマーク
                discard_tiles.append(tile)
            discard_str = ' '.join(discard_tiles)
        else:
            discard_str = "なし"
        
        # 自分のプレイヤーには印を付ける
        if i == current_player:
            discard_info.append(f"  P{i}: {discard_str} ← 自分")
        else:
            discard_info.append(f"  P{i}: {discard_str}")
    
    return '\n'.join(discard_info)

def format_top_predictions(top_predictions, actual_tile=None):
    """推奨打牌Top5を整形して表示"""
    if not top_predictions:
        return "推奨打牌データなし"
    
    lines = []
    for pred in top_predictions[:5]:  # Top5のみ
        rank = pred.get('rank', 0)
        tile = pred.get('tile', '')
        prob = pred.get('probability', 0)
        
        # 実際の打牌と一致する場合は印を付ける
        marker = " ★実打牌" if actual_tile and tile == actual_tile.replace('*', '') else ""
        lines.append(f"  {rank}位: {tile} ({prob:.1%}){marker}")
    
    return '\n'.join(lines)

def convert_hand_to_34_array(hand_tiles):
    """
    手牌リストを34種配列に変換
    hand_tiles: ["1m", "2m", "3m", ...] 形式のリスト
    """
    if not hand_tiles:
        return [0] * 34
    
    # 漢字形式の牌名を数字形式に変換
    kanji_to_z = {
        '東': '1z', '南': '2z', '西': '3z', '北': '4z',
        '白': '5z', '發': '6z', '中': '7z'
    }
    
    converted_tiles = []
    for tile in hand_tiles:
        if tile in kanji_to_z:
            converted_tiles.append(kanji_to_z[tile])
        else:
            converted_tiles.append(tile)
    
    # 手牌リストを文字列に結合してrobust_hand_parserで処理
    hand_string = ''.join(converted_tiles)
    return robust_hand_parser(hand_string)

def convert_tile_name_to_index(tile_name):
    """
    牌名（例："1m", "2p", "3s", "1z"）を34種配列のインデックスに変換
    """
    tile_name = tile_name.replace('*', '')  # ツモ切りマーク除去
    
    # 漢字形式の牌名を優先的にチェック
    tile_mapping = {
        '東': 27, '南': 28, '西': 29, '北': 30, 
        '白': 31, '發': 32, '中': 33
    }
    if tile_name in tile_mapping:
        return tile_mapping[tile_name]
    
    if 'm' in tile_name:
        num = int(tile_name.replace('m', '').replace('0', '5'))  # 0m -> 5m
        return num - 1  # 0-8
    elif 'p' in tile_name:
        num = int(tile_name.replace('p', '').replace('0', '5'))  # 0p -> 5p
        return num - 1 + 9  # 9-17
    elif 's' in tile_name:
        num = int(tile_name.replace('s', '').replace('0', '5'))  # 0s -> 5s
        return num - 1 + 18  # 18-26
    elif 'z' in tile_name:
        num = int(tile_name.replace('z', ''))
        return num - 1 + 27  # 27-33
    else:
        return -1

def analyze_ukeire_for_discard(hand_tiles, discard_tile):
    """
    特定の打牌での受け入れ分析を行う
    hand_tiles: 手牌リスト
    discard_tile: 捨てる牌
    """
    # 手牌を34種配列に変換
    tiles_34 = convert_hand_to_34_array(hand_tiles)
    
    # 捨て牌のインデックスを取得
    discard_index = convert_tile_name_to_index(discard_tile)
    if discard_index == -1 or tiles_34[discard_index] == 0:
        return {
            "error": f"無効な捨て牌: {discard_tile}",
            "shanten": 8,
            "ukeire": {},
            "total_count": 0
        }
    
    # 捨て牌を実行して13枚にする
    hand_13_tiles = list(tiles_34)
    hand_13_tiles[discard_index] -= 1
    
    # シャンテン計算
    shanten_calculator = Shanten()
    shanten_13 = shanten_calculator.calculate_shanten(hand_13_tiles)
    
    # 受け入れ計算
    ukeire_for_discard = {}
    
    if shanten_13 == 0:  # 聴牌の場合: 待ち牌を計算
        for draw_index in range(34):
            # 5枚目になる牌は引けない and 自分が捨てた牌はフリテンになるので待ちに含めない
            if tiles_34[draw_index] < 4 and draw_index != discard_index:
                temp_hand_14 = list(hand_13_tiles)
                temp_hand_14[draw_index] += 1
                # アガリ(-1)になる牌を探す
                if shanten_calculator.calculate_shanten(temp_hand_14) == -1:
                    remaining_count = 4 - tiles_34[draw_index]
                    ukeire_for_discard[draw_index] = remaining_count
    else:  # 聴牌していない場合: シャンテン数を進める牌を計算
        for draw_index in range(34):
            if tiles_34[draw_index] < 4:
                hand_14_after_draw = list(hand_13_tiles)
                hand_14_after_draw[draw_index] += 1
                shanten_after_draw_and_discard = get_shanten_after_best_discard(
                    hand_14_after_draw, shanten_calculator, 'calculate_shanten'
                )
                if shanten_after_draw_and_discard < shanten_13:
                    remaining_count = 4 - tiles_34[draw_index]
                    ukeire_for_discard[draw_index] = remaining_count

    total_ukeire_count = sum(ukeire_for_discard.values())
    
    return {
        "shanten": shanten_13,
        "ukeire": ukeire_for_discard,
        "total_count": total_ukeire_count,
        "discard_tile": discard_tile
    }

def compare_discard_options(hand_tiles, ai_recommended_tile, actual_tile):
    """
    AI推奨打牌と実際打牌の受け入れを比較
    """
    # AI推奨打牌の受け入れ分析
    ai_analysis = analyze_ukeire_for_discard(hand_tiles, ai_recommended_tile)
    
    # 実際打牌の受け入れ分析
    actual_analysis = analyze_ukeire_for_discard(hand_tiles, actual_tile.replace('*', ''))
    
    return {
        "ai_recommendation": ai_analysis,
        "actual_discard": actual_analysis
    }

def format_ukeire_analysis(analysis_result):
    """
    受け入れ分析結果を文字列に整形
    """
    if "error" in analysis_result:
        return f"分析エラー: {analysis_result['error']}"
    
    shanten = analysis_result['shanten']
    ukeire = analysis_result['ukeire']
    total_count = analysis_result['total_count']
    discard_tile = analysis_result['discard_tile']
    
    # シャンテン数の表示
    shanten_str = format_shanten(shanten)
    
    # 受け入れ牌の表示
    if ukeire:
        ukeire_tiles = format_tiles_for_display(sorted(ukeire.keys()))
        ukeire_detail = []
        for tile_idx, count in sorted(ukeire.items()):
            tile_str = format_tiles_for_display([tile_idx])
            ukeire_detail.append(f"{tile_str}×{count}")
        ukeire_detail_str = " ".join(ukeire_detail)
    else:
        ukeire_tiles = "なし"
        ukeire_detail_str = "なし"
    
    return f"""打{discard_tile} → {shanten_str}
受け入れ: {ukeire_tiles} (計{total_count}枚)
詳細: {ukeire_detail_str}"""

def format_ukeire_comparison(ukeire_comparison):
    """
    受け入れ比較結果を文字列に整形
    """
    if not ukeire_comparison:
        return "受け入れ分析データが利用できません"
    
    ai_result = ukeire_comparison.get('ai_recommendation', {})
    actual_result = ukeire_comparison.get('actual_discard', {})
    
    if "error" in ai_result and "error" in actual_result:
        return f"分析エラー:\nAI推奨: {ai_result['error']}\n実際打牌: {actual_result['error']}"
    
    lines = []
    
    # AI推奨打牌の分析
    if "error" not in ai_result:
        lines.append("■ AI推奨打牌")
        lines.append(format_ukeire_analysis(ai_result))
    else:
        lines.append("■ AI推奨打牌")
        lines.append(f"分析エラー: {ai_result['error']}")
    
    lines.append("")  # 空行
    
    # 実際打牌の分析
    if "error" not in actual_result:
        lines.append("■ 実際打牌")
        lines.append(format_ukeire_analysis(actual_result))
    else:
        lines.append("■ 実際打牌")
        lines.append(f"分析エラー: {actual_result['error']}")
    
    # 比較結果
    if "error" not in ai_result and "error" not in actual_result:
        lines.append("")  # 空行
        lines.append("■ 比較結果")
        
        ai_count = ai_result.get('total_count', 0)
        actual_count = actual_result.get('total_count', 0)
        ai_shanten = ai_result.get('shanten', 8)
        actual_shanten = actual_result.get('shanten', 8)
        
        # シャンテン数比較
        if ai_shanten < actual_shanten:
            lines.append(f"シャンテン数: AI推奨が有利 ({format_shanten(ai_shanten)} vs {format_shanten(actual_shanten)})")
        elif ai_shanten > actual_shanten:
            lines.append(f"シャンテン数: 実際打牌が有利 ({format_shanten(actual_shanten)} vs {format_shanten(ai_shanten)})")
        else:
            lines.append(f"シャンテン数: 同じ ({format_shanten(ai_shanten)})")
        
        # 受け入れ枚数比較
        if ai_count > actual_count:
            lines.append(f"受け入れ枚数: AI推奨が有利 ({ai_count}枚 vs {actual_count}枚)")
        elif ai_count < actual_count:
            lines.append(f"受け入れ枚数: 実際打牌が有利 ({actual_count}枚 vs {ai_count}枚)")
        else:
            lines.append(f"受け入れ枚数: 同じ ({ai_count}枚)")
    
    return '\n'.join(lines)

def create_comprehensive_prompt(analysis_data):
    """包括的なLLMプロンプトを生成"""
    
    # 基本情報の抽出
    game_situation = analysis_data.get('game_situation', {})
    prediction = analysis_data.get('prediction', {})
    analysis = analysis_data.get('analysis', {})
    players_state = analysis_data.get('players_state', {})
    
    # 現在プレイヤーの手牌情報
    current_player = game_situation.get('current_player', 0)
    current_player_data = players_state.get(f'player_{current_player}', {})
    hand_composition = format_hand_composition(current_player_data.get('hand', []))
    
    # SHAP分析は不要になったので削除
    feature_categories = {}
    
    # アテンション分析結果
    attention_data = analysis.get('attention_weights', {})
    
    # 概念ラベル分析
    concept_data = analysis.get('concept_labels', {})
    
    # 受け入れ分析
    predicted_tile = prediction.get('predicted_tile', '')
    actual_tile_raw = game_situation.get('actual_discard', '')
    actual_tile = actual_tile_raw.replace('*', '')
    hand_tiles = current_player_data.get('hand', [])
    
    # 受け入れ比較分析を実行
    ukeire_comparison = None
    if hand_tiles and predicted_tile and actual_tile:
        try:
            ukeire_comparison = compare_discard_options(hand_tiles, predicted_tile, actual_tile)
        except Exception as e:
            print(f"受け入れ分析エラー: {e}")
            ukeire_comparison = None
    
    # リーチ者の情報を構築
    reach_players = []
    for p in range(4):
        if players_state[f"player_{p}"]["reach_status"] == 2:
            reach_players.append(f"P{p}")
    
    reach_info = "なし" if not reach_players else ", ".join(reach_players)

    # プロンプトの構築
    prompt = f"""あなたは麻雀の専門コーチです。AI分析結果に基づいて、打牌判断の戦術的根拠を分かりやすく説明してください。

【局面状況】
局: {game_situation["round_info"]} ({game_situation["player_wind"]}家)
リーチ者: {reach_info}
残り牌: {game_situation["remaining_tiles"]}枚
ドラ: {" ".join(game_situation["dora_indicators"])}
供託: {game_situation["kyotaku"]}本 / 本場: {game_situation["honba"]}本場

【自分の手牌】
{hand_composition} (P{current_player})

【各プレイヤーの捨て牌】
{format_all_players_discards(players_state, current_player)}

【副露】"""

    # 各プレイヤーの副露を追加
    for p in range(4):
        melds = players_state[f"player_{p}"]["melds"]
        if melds:
            meld_str = ", ".join([f"{meld['type']}{meld['tiles']}" for meld in melds])
            if p == current_player:
                prompt += f"\n  P{p}: {meld_str} ← 自分"
            else:
                prompt += f"\n  P{p}: {meld_str}"
        else:
            if p == current_player:
                prompt += f"\n  P{p}: なし ← 自分"
            else:
                prompt += f"\n  P{p}: なし"

    prompt += f"""

【ツモ牌】
{game_situation["tsumo_tile"]}

【AI判断】
推奨打牌: {predicted_tile} (確信度: {prediction.get('predicted_probability', 0):.1%})
実際打牌: {actual_tile_raw}

【推奨打牌Top5】
{format_top_predictions(prediction.get('top_predictions', []), actual_tile_raw)}

【受け入れ分析比較】
{format_ukeire_comparison(ukeire_comparison)}

【AI思考プロセス】
{create_tactical_analysis(feature_categories, attention_data, concept_data, predicted_tile)}

【解説要求】
この局面での最適な打牌とその理由を以下の観点から分析してください：

■ 手牌分析
- 現在のシャンテン数と手の進行度
- テンパイに向けた最短ルート

■ 受け入れ効率
- AI推奨打牌と実際打牌の受け入れ枚数・質の比較
- より効率的な選択があるかの検討

■ 局面判断
- リーチ者への対応
- ドラの活用方針
- 守備的考慮事項

■ 総合評価
- AI推奨と実際打牌のどちらが良いか
- 改善点やこの局面から学べること

各項目を明確に分けて、実戦で使える知識として説明してください。"""

    return prompt



def create_tactical_analysis(feature_categories, attention_data, concept_data, predicted_tile):
    """戦術的分析セクションを生成"""
    analysis_parts = []
    
    # 手牌評価セクションを復活
    analysis_parts.append("■ 手牌評価")
    if predicted_tile:
        # シンプルな重要度表示（仮の値として0.564を使用）
        analysis_parts.append(f"・{predicted_tile}が最重要要素(重要度0.564) → 不要牌として強く推奨")
    
    # アテンション分析による行動予測（全層表示）
    if attention_data:
        analysis_parts.append("\n■ 注目した相手の動き（層別分析）")
        
        # 各層のアテンションを表示
        for layer_key in sorted(attention_data.keys(), key=lambda x: int(x.split('_')[1])):
            layer_data = attention_data[layer_key]
            layer_num = layer_data.get('layer', 0)
            top_events = layer_data.get('top_attended_events', [])[:8]  # 上位8個に拡張
            
            if top_events:
                analysis_parts.append(f"【Layer {layer_num}】")
                for i, event in enumerate(top_events):
                    event_token = event.get('event_token', '')
                    weight = event.get('attention_weight', 0)
                    interpretation = interpret_event_token(event_token)
                    if weight > 0.01:  # 閾値を下げる
                        analysis_parts.append(f"  {i+1}. {interpretation} (注目度: {weight:.4f})")
                analysis_parts.append("")  # 層間の空行
    
    # 戦略判断
    if concept_data:
        labels = concept_data.get('concept_labels', [])
        if 'Safety' in labels and 'Speed' in labels:
            analysis_parts.append("■ 戦略方針: バランス型")
            analysis_parts.append("・安全性と速度の両方を考慮した判断")
        elif 'Safety' in labels:
            analysis_parts.append("■ 戦略方針: 安全重視")
            analysis_parts.append("・危険牌回避を優先した守備的判断")
        elif 'Speed' in labels:
            analysis_parts.append("■ 戦略方針: 速度重視") 
            analysis_parts.append("・テンパイ速度を優先した攻撃的判断")
    
    return "\n".join(analysis_parts)



def interpret_event_token(event_token):
    """イベントトークンの戦術的解釈"""
    if event_token.startswith('DIS_'):
        parts = event_token.split('_')
        if len(parts) >= 3:
            player = parts[1].replace('P', 'プレイヤー')
            tile = parts[2] if len(parts) > 2 else ""
            return f"{player}が{tile}を捨てた"
    elif event_token.startswith('TSU_'):
        return "自分のツモ牌を考慮"
    elif event_token.startswith('INI_'):
        return "初期配牌の影響"
    elif event_token.startswith('DOR_'):
        return "ドラ表示の影響"
    
    return event_token

def save_prompt(prompt_text, output_file):
    """プロンプトをファイルに保存"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(prompt_text)
    print(f"プロンプトを保存しました: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="麻雀AI分析結果から詳細なLLMプロンプトを生成します。"
    )
    parser.add_argument("json_file", help="predict.pyで生成された分析結果JSONファイル")
    parser.add_argument("--output", help="出力ファイル名 (デフォルト: prompt_[timestamp].txt)")
    parser.add_argument("--preview", action='store_true', help="プロンプトをコンソールに表示")
    
    args = parser.parse_args()
    
    # 分析結果の読み込み
    analysis_data = load_analysis_result(args.json_file)
    
    # プロンプト生成
    prompt_text = create_comprehensive_prompt(analysis_data)
    
    # プレビュー表示
    if args.preview:
        print("="*60)
        print(" 生成されたプロンプト")
        print("="*60)
        print(prompt_text)
        print("="*60)
    
    # ファイル保存
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(args.json_file))[0]
        output_file = f"prompt_{base_name}_{timestamp}.txt"
    
    save_prompt(prompt_text, output_file)
    
    # 統計情報表示
    prediction = analysis_data.get('prediction', {})
    game_situation = analysis_data.get('game_situation', {})
    
    print(f"\n📊 プロンプト生成完了")
    print(f"入力ファイル: {args.json_file}")
    print(f"AI推奨: {prediction.get('predicted_tile', 'N/A')} ({prediction.get('predicted_probability', 0):.1%})")
    print(f"実際打牌: {game_situation.get('actual_discard', 'N/A')}")
    print(f"分析項目: アテンション({len(analysis_data.get('analysis', {}).get('attention_weights', {}))}層), " +
          f"概念分析({analysis_data.get('analysis', {}).get('concept_labels', {}).get('cluster_id', 'N/A')})")

if __name__ == "__main__":
    main() 