# bertscore_compare.py
# 実行: python bertscore_compare.py
# 事前: pip install bert-score

from bert_score import score
import csv
import re
from typing import List, Tuple, Dict
import argparse
import unicodedata

# 麻雀専門用語の修正辞書
MAHJONG_CORRECTION_DICT = {
    # よくある誤字・誤変換
    "川ぐし": "河底",
    "うわがわ": "上家", 
    "したがわ": "下家",
    "まえがわ": "対面",
    "せんぱい": "先輩",
    "こうはい": "後輩",
    "たんやお": "断ヤオ",
    "ちゅうにん": "中張",
    "やおちゅう": "ヤオチュウ",
    "しゃんぺん": "シャンペン",
    "さんしょく": "三色",
    "いっきつ": "一気通貫",
    "とーいー": "トーイツ",
    "さんあんこ": "三暗刻",
    "さんしょくどうじゅん": "三色同順",
    "ちんいー": "チンイー",
    "ちんろう": "チンロウ",
    "りーち": "リーチ",
    "いーぺーこー": "一発",
    "さんぺーこー": "三倍刻",
    "ほーら": "ホーラ",
    "ぴんふ": "ピンフ",
    "たんやお": "断ヤオ",
    "はんえい": "半荘",
    "ぜんえい": "全荘",
    "ちょうえい": "長荘",
    # 数字の誤変換
    "いち": "1", "に": "2", "さん": "3", "よん": "4", "ご": "5",
    "ろく": "6", "なな": "7", "はち": "8", "きゅう": "9",
    # 牌の誤変換
    "まん": "萬", "ぴん": "ピン", "そう": "索",
    # 動作の誤変換
    "うった": "打った", "きった": "切った", "だした": "出した",
    "つんだ": "積んだ", "つんだ": "積んだ", "ひいた": "引いた",
    # その他
    "たたき": "叩き", "あがり": "和了り", "ロン": "ロン", "ツモ": "ツモ"
}

def clean_transcription_text(text: str) -> str:
    """
    文字起こしテキストの前処理
    - 麻雀専門用語の修正
    - 不要な記号の除去
    - 正規化
    """
    # 全角半角の統一
    text = unicodedata.normalize('NFKC', text)
    
    # 麻雀用語の修正
    for wrong, correct in MAHJONG_CORRECTION_DICT.items():
        text = text.replace(wrong, correct)
    
    # 不要な記号の除去（麻雀に必要な記号は残す）
    text = re.sub(r'[（）\(\)【】\[\]「」『』]', '', text)
    
    # 連続する空白の整理
    text = re.sub(r'\s+', ' ', text)
    
    # 句読点の統一
    text = text.replace('，', '、').replace('．', '。')
    
    return text.strip()

def interactive_text_correction(text: str) -> str:
    """
    ユーザーが手動で文字起こしを修正できるインタラクティブ機能
    """
    print("\n=== 文字起こしテキスト修正モード ===")
    print(f"元のテキスト: {text}")
    print("\n修正したい場合は新しいテキストを入力してください（そのまま使用する場合はEnter）:")
    
    corrected = input("修正後テキスト: ").strip()
    return corrected if corrected else text

def mahjong_sentence_split(text: str) -> List[str]:
    """
    麻雀解説に適した文分割
    - 麻雀の専門用語を考慮した分割
    - 短すぎる文の結合
    - 不完全な文の処理
    """
    # 前処理：文字起こしの修正
    text = clean_transcription_text(text)
    
    # 改行を句点相当に変換
    text = text.replace('\r', '\n')
    
    parts = []
    for line in text.split('\n'):
        if not line.strip():
            continue
            
        # 麻雀解説特有の分割パターン
        # 句点、感嘆符、疑問符で分割
        sents = re.split(r'(?<=[。！？])\s*', line.strip())
        
        for sent in sents:
            sent = sent.strip()
            if not sent:
                continue
                
            # 短すぎる文（麻雀用語以外で5文字以下）は次の文と結合を試みる
            if len(sent) <= 5 and not any(mj_term in sent for mj_term in ['リーチ', 'ツモ', 'ロン', 'ピン', '索', '萬']):
                if parts and len(parts[-1]) < 30:  # 前の文も短い場合は結合
                    parts[-1] += sent
                    continue
                    
            # 不完全な文（末尾が...や、、、など）の処理
            if sent.endswith('...') or sent.endswith('、、、'):
                sent = sent.rstrip('.、')
                
            # 意味のある文のみ追加
            if len(sent) >= 3:  # 最低3文字以上
                parts.append(sent)
    
    return parts

def assess_transcription_quality(text: str) -> Dict[str, float]:
    """
    文字起こし品質の自動評価
    """
    quality_metrics = {
        'completeness': 0.0,  # 完全性（不完全な文の割合）
        'readability': 0.0,   # 可読性（適切な句読点の使用）
        'terminology': 0.0    # 専門用語の正確性
    }
    
    sentences = mahjong_sentence_split(text)
    if not sentences:
        return quality_metrics
    
    # 完全性の評価（不完全な文の割合）
    incomplete_count = sum(1 for s in sentences if s.endswith('...') or s.endswith('、、、') or len(s) < 5)
    quality_metrics['completeness'] = max(0, 1.0 - (incomplete_count / len(sentences)))
    
    # 可読性の評価（適切な句読点の使用）
    proper_punctuation = sum(1 for s in sentences if s.endswith('。') or s.endswith('！') or s.endswith('？'))
    quality_metrics['readability'] = proper_punctuation / len(sentences)
    
    # 専門用語の正確性（修正辞書に含まれる誤変換の数）
    total_words = len(re.findall(r'\S+', text))
    correction_count = sum(1 for wrong in MAHJONG_CORRECTION_DICT.keys() if wrong in text)
    quality_metrics['terminology'] = max(0, 1.0 - (correction_count / max(1, total_words / 10)))
    
    return quality_metrics

def compute_bertscore(cands: List[str], refs: List[str], lang="ja", model_type=None, batch_size=32):
    """
    cands, refs: 同じ長さのリスト（対応するペア）
    model_type: None の場合 lang に応じて自動選択（bert-score のデフォルト）
    戻り値: (P_list, R_list, F1_list) - 各文ごとのスコア (torch tensors or lists)
    """
    P, R, F1 = score(cands, refs, lang=lang, model_type=model_type, batch_size=batch_size, verbose=True)
    # torch tensors -> Python list of floats
    return P.tolist(), R.tolist(), F1.tolist()

def best_match_mode(system_sents: List[str], pro_sents: List[str], lang="ja", model_type=None):
    """
    各 system 文に対して、pro 文の中から BertScore F1 が最大となる pro 文を探して対応づける。
    （O(N*M) の比較）
    """
    matches = []  # list of tuples (sys_idx, pro_idx, P, R, F1)
    # 逐次的に計算して負荷を抑える（小規模データ向け）
    for i, sys in enumerate(system_sents):
        best_f1 = -1.0
        best_p = best_r = 0.0
        best_j = -1
        for j, pro in enumerate(pro_sents):
            P, R, F1 = score([sys], [pro], lang=lang, model_type=model_type)
            f1 = float(F1[0])
            if f1 > best_f1:
                best_f1 = f1
                best_p = float(P[0])
                best_r = float(R[0])
                best_j = j
        matches.append((i, best_j, best_p, best_r, best_f1))
    return matches

def save_to_csv(filename: str, rows: List[Tuple]):
    """
    rows: list of tuples to write as CSV rows (header added by caller)
    """
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

def simple_bertscore_display(cands: List[str], refs: List[str], lang="ja"):
    """
    bert-score.pyスタイルのシンプルなBERTScore表示
    """
    print("=== シンプルBERTScore計算 ===")
    P, R, F1 = score(cands, refs, lang=lang)
    
    print(f"Precision: {P.mean().item():.4f}")
    print(f"Recall: {R.mean().item():.4f}")
    print(f"F1: {F1.mean().item():.4f}")
    
    return P.mean().item(), R.mean().item(), F1.mean().item()

def vocabulary_comparison_test():
    """
    語彙バリエーションの比較テスト
    同じ意味だが異なる語彙の文章を比較
    """
    print("=== 語彙バリエーション比較テスト ===")
    
    # 基準文（プロ解説風）
    reference = "四索打ったのは安全牌って判断したからだね。"
    
    # 語彙バリエーション（より上手な同義表現）
    variations = [
        # 同じ意味の表現
        ("丁寧語", "四索を切ったのは、場況的に安全牌と判断したためです。"),
        ("カジュアル語", "四索切ったのは安全牌だと思ったからさ。"),
        ("専門的表現", "四索の打牌は安全牌との判断による選択である。"),
        ("簡潔表現", "四索は安全牌なので切った。"),
        ("理由重視", "四索を切った理由は安全牌と判断したからです。"),
        ("同義語表現", "四索を捨てたのは、状況的に安全な牌と認識したからです。"),
        ("受動態表現", "四索は安全牌として判断され、切られました。"),
        ("話し言葉", "四索切ったのは安全牌だと思ったから。"),
        
        # 意味が違う表現（対照実験用）
        ("逆の意味", "四索を切ったのは、危険牌と判断したためです。"),
        ("違う牌", "三索を切ったのは、安全牌と判断したためです。"),
        ("違う戦略", "四索を切ったのは、一発を狙うためです。"),
        ("無関係", "今日は天気が良いですね。"),
        ("全く違う", "お腹が空きました。"),
        ("麻雀無関係", "映画を見に行きたいです。")
    ]
    
    print(f"基準文: {reference}\n")
    
    for style, variation in variations:
        P, R, F1 = score([variation], [reference], lang="ja")
        print(f"{style}:")
        print(f"  文章: {variation}")
        print(f"  Precision: {P[0]:.4f}, Recall: {R[0]:.4f}, F1: {F1[0]:.4f}")
        print()
    
    return variations

def advanced_vocabulary_test():
    """
    より高度な語彙バリエーションテスト
    微妙な意味の違いや、より自然な同義表現をテスト
    """
    print("=== 高度な語彙バリエーションテスト ===")
    
    # 基準文
    reference = "四索打ったのは安全牌って判断したからだね。"
    
    # より自然で上手な同義表現
    advanced_variations = [
        # 同じ意味だが表現が異なる（高スコア期待）
        ("自然な話し言葉", "四索切ったのは安全牌だと思ったから。"),
        ("関西弁風", "四索切ったのは安全牌やと思ったからや。"),
        ("若者言葉", "四索切ったのは安全牌って感じで。"),
        ("改まった表現", "四索を打牌したのは安全牌と判断したためです。"),
        ("分析的な表現", "四索の打牌理由は安全牌との判断に基づきます。"),
        
        # 微妙に意味が違う（中程度のスコア期待）
        ("確信度が高い", "四索切ったのは絶対に安全牌だから。"),
        ("確信度が低い", "四索切ったのは多分安全牌だと思ったから。"),
        ("時間的要素", "四索切ったのはその時安全牌と判断したから。"),
        ("比較的表現", "四索切ったのは他の牌より安全牌と判断したから。"),
        
        # 意味がかなり違う（低スコア期待）
        ("逆の判断", "四索切ったのは危険牌と判断したから。"),
        ("違う理由", "四索切ったのは形作り優先だから。"),
        ("違う牌", "三索切ったのは安全牌と判断したから。"),
        ("違う戦略", "四索切ったのは一発狙いだから。"),
        
        # 全く関係ない（最低スコア期待）
        ("完全無関係", "今日はいい天気ですね。"),
        ("麻雀無関係", "お昼ご飯を食べました。"),
        ("数字だけ同じ", "四という数字が好きです。")
    ]
    
    print(f"基準文: {reference}\n")
    
    # スコア別に分類して表示
    high_scores = []
    medium_scores = []
    low_scores = []
    
    for style, variation in advanced_variations:
        P, R, F1 = score([variation], [reference], lang="ja")
        f1_score = F1[0].item()
        
        print(f"{style}:")
        print(f"  文章: {variation}")
        print(f"  Precision: {P[0]:.4f}, Recall: {R[0]:.4f}, F1: {f1_score:.4f}")
        
        # スコア別に分類
        if f1_score >= 0.7:
            high_scores.append((style, variation, f1_score))
        elif f1_score >= 0.4:
            medium_scores.append((style, variation, f1_score))
        else:
            low_scores.append((style, variation, f1_score))
        print()
    
    # 結果の分析
    print("=" * 60)
    print("=== スコア分析結果 ===")
    print(f"高スコア (F1 ≥ 0.7): {len(high_scores)}件")
    for style, _, score in high_scores:
        print(f"  - {style}: {score:.4f}")
    
    print(f"\n中スコア (0.4 ≤ F1 < 0.7): {len(medium_scores)}件")
    for style, _, score in medium_scores:
        print(f"  - {style}: {score:.4f}")
    
    print(f"\n低スコア (F1 < 0.4): {len(low_scores)}件")
    for style, _, score in low_scores:
        print(f"  - {style}: {score:.4f}")
    
    return advanced_variations

def example_run(interactive_mode=True):
    """
    メインの実行関数
    interactive_mode: Trueの場合は手動修正モードを有効にする
    """
    print("=== 麻雀AI解説文とプロ解説のBERTScore比較（語彙バリエーションテスト） ===")
    
    # --- 例: プロ解説（与えられた長いテキストの一部）---
    pro_text = """四索打ったのは安全牌って判断したからだね。河底まで来てるとそんなに危なくないかも。でも上家がリーチかけてるから慎重にならないと。一発消すために先に打っとくって判断もアリだし。ドラ受け優先で打牌選んだのは確実に点取りたいからだろうな。"""
    
    # 文字起こし品質の評価
    print("\n=== 文字起こし品質評価 ===")
    quality = assess_transcription_quality(pro_text)
    print(f"完全性: {quality['completeness']:.3f}")
    print(f"可読性: {quality['readability']:.3f}")
    print(f"専門用語正確性: {quality['terminology']:.3f}")
    
    # 自動修正
    cleaned_pro_text = clean_transcription_text(pro_text)
    print(f"\n自動修正前: {pro_text}")
    print(f"自動修正後: {cleaned_pro_text}")
    
    # インタラクティブ修正モード
    if interactive_mode:
        corrected_pro_text = interactive_text_correction(cleaned_pro_text)
        pro_sents = mahjong_sentence_split(corrected_pro_text)
    else:
        pro_sents = mahjong_sentence_split(cleaned_pro_text)
    
    print(f"\n分割されたプロ解説文: {len(pro_sents)}文")
    for i, sent in enumerate(pro_sents):
        print(f"  {i}: {sent}")

    # --- 語彙バリエーションテスト: 同じ意味だが異なる語彙のシステム出力 ---
    system_sents = [
        # パターン1: 丁寧語（元の形式）
        "四索を切ったのは、場況的に安全牌と判断したためです。",
        "一発を消すために先に打った、という意図があります。",
        "ドラ受けを優先して、打牌を選びました。",
        
        # パターン2: カジュアル語（より自然な話し言葉）
        "四索切ったのは安全牌だと思ったからさ。",
        "一発避けるために先に打ったんだよ。",
        "ドラ受け重視で打牌決めた。",
        
        # パターン3: 専門的表現（麻雀用語を多用）
        "四索の打牌は安全牌との判断による選択である。",
        "一発を回避する目的で先に打牌を行った。",
        "ドラ受けを最優先として打牌を決定した。",
        
        # パターン4: 簡潔表現（短縮形）
        "四索は安全牌なので切った。",
        "一発避けで先に打った。",
        "ドラ受け優先で打牌。",
        
        # パターン5: 理由重視表現
        "四索を切った理由は安全牌と判断したからです。",
        "一発を消すという目的で先に打ちました。",
        "ドラ受けを重視した理由で打牌を選択しました。",
        
        # パターン6: 同義語・類義語を使った表現
        "四索を捨てたのは、状況的に安全な牌と認識したからです。",
        "一発を防ぐために前もって打った、という狙いがあります。",
        "ドラ受けを重視して、打牌を選択しました。",
        
        # パターン7: 受動態・能動態の違い
        "四索は安全牌として判断され、切られました。",
        "一発を避ける目的で、先に打牌が行われました。",
        "ドラ受けが優先され、打牌が選ばれました。",
        
        # パターン8: 意味が全く違う文（対照実験用）
        "三索を切ったのは、危険牌と判断したためです。",
        "一発を狙うために最後に打った、という戦略があります。",
        "ドラを避けて、安全牌を選びました。",
        
        # パターン9: 全く関係ない文（対照実験用）
        "今日は天気が良いですね。",
        "お腹が空きました。",
        "映画を見に行きたいです。"
    ]
    
    print(f"\nシステム出力文: {len(system_sents)}文（語彙バリエーション）")
    patterns = ["丁寧語", "カジュアル語", "専門的表現", "簡潔表現", "理由重視表現", 
                "同義語表現", "受動態表現", "意味違い文", "無関係文"]
    for i, sent in enumerate(system_sents):
        pattern_idx = i // 3
        pattern_name = patterns[pattern_idx] if pattern_idx < len(patterns) else "その他"
        print(f"  {i} ({pattern_name}): {sent}")

    # BERTScoreによる最適マッチング
    print("\n=== BERTScore計算中... ===")
    matches = best_match_mode(system_sents, pro_sents, lang="ja")

    # 結果の詳細表示（語彙パターン別に整理）
    print("\n=== マッチング結果（語彙パターン別） ===")
    total_p = total_r = total_f1 = 0.0
    valid_matches = 0
    pattern_scores = {}
    
    for i, (sys_idx, pro_idx, p, r, f1) in enumerate(matches):
        pro_sent = pro_sents[pro_idx] if (0 <= pro_idx < len(pro_sents)) else ""
        pattern_idx = sys_idx // 3
        pattern_name = patterns[pattern_idx] if pattern_idx < len(patterns) else "その他"
        
        print(f"システム文{sys_idx} ({pattern_name}): {system_sents[sys_idx]}")
        print(f"  → マッチしたプロ解説: {pro_sent}")
        print(f"  → Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
        
        # パターン別スコア集計
        if pattern_name not in pattern_scores:
            pattern_scores[pattern_name] = {'p': [], 'r': [], 'f1': []}
        pattern_scores[pattern_name]['p'].append(p)
        pattern_scores[pattern_name]['r'].append(r)
        pattern_scores[pattern_name]['f1'].append(f1)
        
        # 全体の平均計算用
        total_p += p
        total_r += r
        total_f1 += f1
        valid_matches += 1
        print()
    
    # パターン別平均スコア表示
    print("\n=== 語彙パターン別平均BERTScore ===")
    for pattern_name, scores in pattern_scores.items():
        avg_p = sum(scores['p']) / len(scores['p'])
        avg_r = sum(scores['r']) / len(scores['r'])
        avg_f1 = sum(scores['f1']) / len(scores['f1'])
        print(f"{pattern_name}:")
        print(f"  Precision: {avg_p:.4f}, Recall: {avg_r:.4f}, F1: {avg_f1:.4f}")
        print()
    
    # 全体のBERTScore表示（bert-score.pyスタイル）
    if valid_matches > 0:
        avg_p = total_p / valid_matches
        avg_r = total_r / valid_matches
        avg_f1 = total_f1 / valid_matches
        
        print("=" * 50)
        print("=== 全体のBERTScore ===")
        print(f"Precision: {avg_p:.4f}")
        print(f"Recall: {avg_r:.4f}")
        print(f"F1: {avg_f1:.4f}")
        print("=" * 50)

    # 出力保存
    csv_rows = [("sys_idx","pro_idx","system_sent","pro_sent","P","R","F1")]
    for (i, j, p, r, f1) in matches:
        pro_sent = pro_sents[j] if (0 <= j < len(pro_sents)) else ""
        csv_rows.append((i, j, system_sents[i], pro_sent, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}"))

    save_to_csv("bertscore_matches.csv", csv_rows)
    print("結果を bertscore_matches.csv に保存しました。")

def main():
    """コマンドライン引数処理"""
    parser = argparse.ArgumentParser(description='麻雀AI解説文とプロ解説のBERTScore比較（語彙バリエーションテスト）')
    parser.add_argument('--no-interactive', action='store_true', 
                       help='インタラクティブ修正モードを無効にする')
    parser.add_argument('--pro-text', type=str, 
                       help='プロ解説テキストを指定する（ファイルパスまたは直接入力）')
    parser.add_argument('--system-text', type=str, 
                       help='システム出力テキストを指定する（ファイルパスまたは直接入力）')
    parser.add_argument('--simple', action='store_true',
                       help='シンプルなBERTScore表示のみ（bert-score.pyスタイル）')
    parser.add_argument('--vocab-test', action='store_true',
                       help='語彙バリエーション比較テストのみ実行')
    parser.add_argument('--advanced-vocab', action='store_true',
                       help='高度な語彙バリエーションテストのみ実行')
    
    args = parser.parse_args()
    
    # 高度な語彙バリエーションテストの場合
    if args.advanced_vocab:
        advanced_vocabulary_test()
        return
    
    # 語彙バリエーションテストの場合
    if args.vocab_test:
        vocabulary_comparison_test()
        return
    
    # シンプルモードの場合
    if args.simple:
        # デフォルトの比較文を使用
        cands = ["四索を切ったのは、場況的に安全牌と判断したためです。"]
        refs = ["四索打ったのは安全牌って判断したからだね。"]
        
        simple_bertscore_display(cands, refs, lang="ja")
        return
    
    # テキストファイルから読み込みの場合は実装
    if args.pro_text:
        try:
            with open(args.pro_text, 'r', encoding='utf-8') as f:
                pro_text = f.read()
        except FileNotFoundError:
            pro_text = args.pro_text  # 直接入力として扱う
    else:
        pro_text = None
    
    # メイン実行
    example_run(interactive_mode=not args.no_interactive)

if __name__ == "__main__":
    main()
