# prompt.py - 麻雀AI分析結果からLLMプロンプトを生成
import json
import argparse
import sys
import os
from datetime import datetime

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

def interpret_feature_name(feature_name):
    """特徴量名を人間が理解しやすい形に解釈"""
    interpretations = {
        # 静的特徴量
        "静的_手牌_": "手牌の{}の枚数",
        "静的_ドラ表示_": "ドラ表示牌{}の枚数", 
        "静的_自捨牌_": "自分が捨てた{}の枚数",
        "静的_全見え牌_": "場に見えている{}の枚数",
        "静的_局風": "場風",
        "静的_本場": "本場数",
        "静的_供託": "供託本数",
        "静的_親プレイヤーIdx": "親のプレイヤー番号",
        "静的_壁残枚数": "山に残っている牌数",
        "静的_自身が親か": "自分が親かどうか",
        "静的_巡目(GameState)": "現在の巡目",
        "静的_ドラ表示牌数": "ドラ表示牌の総数",
        "静的_リーチ状態": "リーチ状態",
        "静的_リーチ巡目": "リーチした巡目",
        "静的_自身の捨て牌数": "自分の捨て牌数",
        "静的_自身の副露数": "自分の副露数",
        "静的_自身の手牌数": "自分の手牌数",
        
        # イベント特徴量
        "Event_": "イベント履歴",
        "_タイプ": "のイベント種別",
        "_プレイヤー": "のプレイヤー",
        "_牌Idx": "の牌インデックス", 
        "_巡目": "の巡目",
        "_データA": "の追加データA",
        "_データB": "の追加データB"
    }
    
    # 手牌関連の解釈
    for key, template in interpretations.items():
        if feature_name.startswith(key):
            if "静的_手牌_" in feature_name:
                tile_name = feature_name.replace("静的_手牌_", "")
                return template.format(tile_name)
            elif "静的_ドラ表示_" in feature_name:
                tile_name = feature_name.replace("静的_ドラ表示_", "")
                return template.format(tile_name)
            elif "静的_自捨牌_" in feature_name:
                tile_name = feature_name.replace("静的_自捨牌_", "")
                return template.format(tile_name)
            elif "静的_全見え牌_" in feature_name:
                tile_name = feature_name.replace("静的_全見え牌_", "")
                return template.format(tile_name)
            elif key in feature_name:
                return template
    
    return feature_name

def analyze_shap_features(shap_data):
    """SHAP特徴量を分析してカテゴリ別に整理"""
    categories = {
        "手牌構成": {"features": [], "total_importance": 0},
        "ドラ関連": {"features": [], "total_importance": 0},
        "捨て牌情報": {"features": [], "total_importance": 0},
        "局面状況": {"features": [], "total_importance": 0},
        "イベント履歴": {"features": [], "total_importance": 0},
        "その他": {"features": [], "total_importance": 0}
    }
    
    for feature_name, importance in shap_data.get('feature_importance', []):
        abs_importance = abs(importance)
        feature_info = {
            "name": feature_name,
            "importance": importance,
            "abs_importance": abs_importance,
            "interpretation": interpret_feature_name(feature_name)
        }
        
        # カテゴリ分類
        if "手牌_" in feature_name:
            categories["手牌構成"]["features"].append(feature_info)
            categories["手牌構成"]["total_importance"] += abs_importance
        elif "ドラ" in feature_name:
            categories["ドラ関連"]["features"].append(feature_info)
            categories["ドラ関連"]["total_importance"] += abs_importance
        elif "捨牌" in feature_name:
            categories["捨て牌情報"]["features"].append(feature_info)
            categories["捨て牌情報"]["total_importance"] += abs_importance
        elif any(x in feature_name for x in ["局風", "本場", "供託", "巡目", "残枚数"]):
            categories["局面状況"]["features"].append(feature_info)
            categories["局面状況"]["total_importance"] += abs_importance
        elif "Event_" in feature_name:
            categories["イベント履歴"]["features"].append(feature_info)
            categories["イベント履歴"]["total_importance"] += abs_importance
        else:
            categories["その他"]["features"].append(feature_info)
            categories["その他"]["total_importance"] += abs_importance
    
    return categories

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

def create_comprehensive_prompt(analysis_data, prompt_style="tactical"):
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
    
    # SHAP分析結果の整理
    shap_data = analysis.get('shap_explanation', {})
    feature_categories = analyze_shap_features(shap_data)
    
    # アテンション分析結果
    attention_data = analysis.get('attention_weights', {})
    
    # 概念ラベル分析
    concept_data = analysis.get('concept_labels', {})
    
    # プロンプトのスタイル別テンプレート
    if prompt_style == "quantitative":
        prompt_template = """あなたは麻雀AIの分析専門家です。以下の定量的分析結果に基づいて、打牌選択の数値的根拠を説明してください。

【局面詳細】
{game_context}

【自分の手牌】
{hand_info}

【各プレイヤーの捨て牌】
{discards_info}

【予測結果】
推奨打牌: {predicted_tile} (確率: {predicted_prob:.1%})
実際打牌: {actual_tile}
予測精度: {prediction_accuracy}

【推奨打牌Top5】
{top_predictions}

【定量分析結果】
{quantitative_analysis}

【詳細SHAP分析】
{detailed_shap_analysis}

【要求事項】
1. SHAP値に基づく特徴量重要度の解釈
2. 予測確率の妥当性評価
3. 数値的根拠による打牌判断の検証

回答は客観的な数値分析に基づいて行ってください。"""

    elif prompt_style == "comparative":
        prompt_template = """あなたは麻雀の戦術コーチです。AI分析結果と実際の打牌を比較し、戦術的観点から評価してください。

【局面設定】
{game_context}

【自分の手牌】
{hand_info}

【各プレイヤーの捨て牌】
{discards_info}

【AI vs 実打牌比較】
AI推奨: {predicted_tile} (確率: {predicted_prob:.1%})
実際打牌: {actual_tile}
判定: {ai_vs_actual}

【推奨打牌Top5】
{top_predictions}

【戦術分析】
{tactical_analysis}

【詳細分析データ】
{detailed_shap_analysis}

【比較評価要求】
1. AI推奨打牌の戦術的妥当性
2. 実際打牌の代替戦術としての価値
3. 局面に応じた最適解の考察

複数の戦術的観点から比較評価してください。"""

    else:  # tactical (default)
        prompt_template = """あなたは麻雀の専門コーチです。AI分析結果に基づいて、打牌判断の戦術的根拠を分かりやすく説明してください。

【局面状況】
{game_context}

【自分の手牌】
{hand_info}

【各プレイヤーの捨て牌】
{discards_info}

【AI判断】
推奨打牌: {predicted_tile} (確信度: {predicted_prob:.1%})
実際打牌: {actual_tile}

【推奨打牌Top5】
{top_predictions}

【AI思考プロセス】
{tactical_analysis}

【詳細分析データ】
{detailed_shap_analysis}

【解説要求】
以下の3つの観点から、初心者にも分かりやすく解説してください：

1. **即効性判断** (50文字以内)
   なぜこの牌を切るのが良いのか、端的な理由

2. **戦術的根拠** (150文字以内)  
   手牌構成、局面状況を踏まえた詳細な戦術理由

3. **代替案検討** (100文字以内)
   他の選択肢と比べてなぜこれがベストか

各項目を明確に分けて、実戦で使える知識として説明してください。"""

    # 実際の打牌との一致判定
    predicted_tile = prediction.get('predicted_tile', '')
    actual_tile = game_situation.get('actual_discard', '').replace('*', '')
    prediction_accuracy = "○的中" if predicted_tile == actual_tile else "×不一致"
    ai_vs_actual = "AI的中" if predicted_tile == actual_tile else "AI外れ"
    
    # Top5推奨打牌の整形
    top_predictions_data = prediction.get('top_predictions', [])
    actual_tile = game_situation.get('actual_discard', 'N/A')
    
    # テンプレート用データの準備
    context_data = {
        "game_context": f"""局: {game_situation.get('round_info', '')} ({game_situation.get('player_wind', '')}家)
リーチ者: {sum(1 for p_data in players_state.values() if p_data.get('reach_status') == 2)}人
残り牌: {game_situation.get('remaining_tiles', 0)}枚
ドラ: {' '.join(game_situation.get('dora_indicators', []))}
供託: {game_situation.get('kyotaku', 0)}本 / 本場: {game_situation.get('honba', 0)}本場""",

        "hand_info": f"{hand_composition} (P{current_player})",
        "discards_info": format_all_players_discards(players_state, current_player),
        
        "predicted_tile": predicted_tile,
        "predicted_prob": prediction.get('predicted_probability', 0),
        "actual_tile": actual_tile,
        "prediction_accuracy": prediction_accuracy,
        "ai_vs_actual": ai_vs_actual,
        "top_predictions": format_top_predictions(top_predictions_data, actual_tile),
        
        "quantitative_analysis": create_quantitative_analysis(feature_categories, concept_data),
        "tactical_analysis": create_tactical_analysis(feature_categories, attention_data, concept_data),
        "detailed_shap_analysis": create_detailed_shap_analysis(analysis_data.get('analysis', {}))  # 詳細SHAP分析を追加
    }
    
    return prompt_template.format(**context_data)

def create_quantitative_analysis(feature_categories, concept_data):
    """定量的分析セクションを生成"""
    analysis_parts = []
    
    # 重要度上位カテゴリ（すべて表示）
    sorted_categories = sorted(feature_categories.items(), 
                             key=lambda x: x[1]['total_importance'], reverse=True)
    
    analysis_parts.append("■ 特徴量重要度ランキング")
    for category_name, category_data in sorted_categories:
        if category_data['features'] and category_data['total_importance'] > 0.001:  # 閾値を下げる
            analysis_parts.append(f"【{category_name}】(重要度: {category_data['total_importance']:.3f})")
            # より多くの特徴量を表示（上位10個）
            for feature in category_data['features'][:10]:
                if abs(feature['importance']) > 0.001:  # より小さな値も含める
                    analysis_parts.append(f"  ・{feature['interpretation']}: {feature['importance']:.4f}")
    
    # 概念分析
    if concept_data:
        cluster_id = concept_data.get('cluster_id', -1)
        labels = concept_data.get('concept_labels', [])
        analysis_parts.append(f"\n■ 概念クラスタ分析")
        analysis_parts.append(f"クラスタID: {cluster_id} ({', '.join(labels)})")
    
    return "\n".join(analysis_parts)

def create_tactical_analysis(feature_categories, attention_data, concept_data):
    """戦術的分析セクションを生成"""
    analysis_parts = []
    
    # 手牌構成の戦術的意味
    hand_features = feature_categories.get('手牌構成', {}).get('features', [])
    if hand_features:
        analysis_parts.append("■ 手牌評価")
        top_hand_feature = hand_features[0]
        tile_name = top_hand_feature['name'].replace('静的_手牌_', '')
        importance = top_hand_feature['importance']
        if importance > 0.3:
            analysis_parts.append(f"・{tile_name}が最重要要素(重要度{importance:.3f}) → 不要牌として強く推奨")
        elif importance > 0.1:
            analysis_parts.append(f"・{tile_name}が重要要素(重要度{importance:.3f}) → やや不要牌の傾向")
        else:
            analysis_parts.append(f"・{tile_name}の影響は限定的(重要度{importance:.3f})")
    
    # アテンション分析による行動予測（全層表示）
    if attention_data:
        analysis_parts.append("\n■ attention_weights:注目した相手の動き（層別分析）")
        
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
            analysis_parts.append("\n■ activation_vector戦略方針: バランス型")
            analysis_parts.append("・安全性と速度の両方を考慮した判断")
        elif 'Safety' in labels:
            analysis_parts.append("\n■ activation_vector戦略方針: 安全重視")
            analysis_parts.append("・危険牌回避を優先した守備的判断")
        elif 'Speed' in labels:
            analysis_parts.append("\n■ activation_vector戦略方針: 速度重視") 
            analysis_parts.append("・テンパイ速度を優先した攻撃的判断")
    
    return "\n".join(analysis_parts)

def create_detailed_shap_analysis(analysis_data):
    """詳細なSHAP分析結果を生成"""
    shap_data = analysis_data.get('shap_explanation', {})
    if not shap_data:
        return "SHAP分析データなし"
    
    analysis_parts = []
    analysis_parts.append("■ 詳細SHAP分析")
    
    # 全特徴量の重要度（上位30個）
    feature_importance = shap_data.get('feature_importance', [])
    if feature_importance:
        analysis_parts.append(f"【重要特徴量 Top 30】(対象牌: {shap_data.get('target_class', 'N/A')})")
        
        positive_features = []
        negative_features = []
        
        for name, importance in feature_importance[:30]:
            interpretation = interpret_feature_name(name)
            if importance > 0:
                positive_features.append((interpretation, importance))
            else:
                negative_features.append((interpretation, importance))
        
        if positive_features:
            analysis_parts.append("\n**推奨要因 (正の寄与):**")
            for i, (name, imp) in enumerate(positive_features[:15]):
                analysis_parts.append(f"  +{i+1}. {name}: +{imp:.4f}")
        
        if negative_features:
            analysis_parts.append("\n**反対要因 (負の寄与):**")
            for i, (name, imp) in enumerate(negative_features[:15]):
                analysis_parts.append(f"  -{i+1}. {name}: {imp:.4f}")
    
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
    parser.add_argument("--style", choices=["tactical", "quantitative", "comparative"], 
                       default="tactical", help="プロンプトのスタイル (デフォルト: tactical)")
    parser.add_argument("--output", help="出力ファイル名 (デフォルト: prompt_[style]_[timestamp].txt)")
    parser.add_argument("--preview", action='store_true', help="プロンプトをコンソールに表示")
    
    args = parser.parse_args()
    
    # 分析結果の読み込み
    analysis_data = load_analysis_result(args.json_file)
    
    # プロンプト生成
    prompt_text = create_comprehensive_prompt(analysis_data, args.style)
    
    # プレビュー表示
    if args.preview:
        print("="*60)
        print(f" 生成されたプロンプト ({args.style}スタイル)")
        print("="*60)
        print(prompt_text)
        print("="*60)
    
    # ファイル保存
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(args.json_file))[0]
        output_file = f"prompt_{args.style}_{base_name}_{timestamp}.txt"
    
    save_prompt(prompt_text, output_file)
    
    # 統計情報表示
    prediction = analysis_data.get('prediction', {})
    game_situation = analysis_data.get('game_situation', {})
    
    print(f"\n📊 プロンプト生成完了")
    print(f"入力ファイル: {args.json_file}")
    print(f"スタイル: {args.style}")
    print(f"AI推奨: {prediction.get('predicted_tile', 'N/A')} ({prediction.get('predicted_probability', 0):.1%})")
    print(f"実際打牌: {game_situation.get('actual_discard', 'N/A')}")
    print(f"分析項目: SHAP({len(analysis_data.get('analysis', {}).get('shap_explanation', {}).get('feature_importance', []))}), " +
          f"アテンション({len(analysis_data.get('analysis', {}).get('attention_weights', {}))}層), " +
          f"概念分析({analysis_data.get('analysis', {}).get('concept_labels', {}).get('cluster_id', 'N/A')})")

if __name__ == "__main__":
    main() 