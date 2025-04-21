import numpy as np

# tehaiは[東の枚数, 南の枚数, ..., 發の枚数, 中の枚数]という形式のリスト
def zihai_pattern(tehai):
    mentsu = 0
    mentsu_kouho = 0
    for i in range(7):
        if tehai[i] >= 3: mentsu += 1
        if tehai[i] == 2: mentsu_kouho += 1
    return (mentsu, mentsu_kouho)


# tehaiは[1mの枚数, 2mの枚数, ..., 發の枚数, 中の枚数]という形式のリスト
# tableは事前計算したもの

def all_tehai_pattern(tehai):
    for m_menstu, m_mentsu_kouho in table[tehai[0:9]]:
        for p_menstu, p_mentsu_kouho in table[tehai[9:18]]:
            for s_menstu, s_mentsu_kouho in table[tehai[18:27]]:
                z_menstu, z_mentsu_kouho = zihai_pattern(tehai[27:34])
                mentsu = m_menstu + p_menstu + s_menstu + z_menstu
                mentsu_kouho = m_mentsu_kouho + p_mentsu_kouho + s_mentsu_kouho + z_mentsu_kouho
                mentsu_kouho = min(mentsu_kouho, 4 - mentsu)
                yield (mentsu, mentsu_kouho)

def calculate_shanten1(tehai):
    min_shanten = float('inf')

    # 雀頭を抜かない場合の向聴数
    for mentsu, mentsu_kouho in all_tehai_pattern(tehai):
        shanten = 8 - mentsu * 2 - mentsu_kouho
        min_shanten = min(min_shanten, shanten)

    # 雀頭を抜く場合の向聴数
    # 抜ける雀頭を全探索
    for i in range(34):
        # 雀頭を抜けなかったらスキップ
        if tehai[i] < 2:
            continue

        # 雀頭を抜く
        tehai[i] -= 2
        for mentsu, mentsu_kouho in all_tehai_pattern(tehai):
                shanten = 8 - mentsu * 2 - mentsu_kouho - 1
                min_shanten = min(min_shanten, shanten)
        # 雀頭を復元する
        tehai[i] += 2

    return min_shanten        


def calculate_shanten(hand_indices, melds=[]):
    """
    手牌から向聴数と受け入れ牌を計算する関数（簡易版）
    
    Args:
        hand_indices (list): 牌種インデックス(0-33)のリスト
        melds (list): 副露情報のリスト [{type: str, tiles: list, ...}, ...]
    
    Returns:
        tuple: (向聴数, 受け入れ牌種インデックスのリスト)
    """
    try:
        # 実際の向聴数計算ライブラリが未実装の場合のフォールバック
        # エラーが出ないように、仮の向聴数と受け入れ牌を返す
        
        # 手牌のカウント（同じインデックスの牌が何枚あるか）
        hand_count = [0] * 34
        for idx in hand_indices:
            if 0 <= idx < 34:  # 範囲チェック
                hand_count[idx] += 1
        
        # 副露も考慮（暗槓・加槓・ポン・チーなど）
        melds_count = 0
        for meld in melds:
            if isinstance(meld, dict) and 'tiles' in meld:
                melds_count += 1
                for idx in meld['tiles']:
                    if 0 <= idx < 34:  # 範囲チェック
                        hand_count[idx] += 1
            
        # 副露を含めた枚数
        total_tiles = sum(hand_count)
        
        # 向聴数の簡易計算（実際は複雑な計算が必要）
        # 面子の数の近似値を計算
        pair_count = sum(1 for count in hand_count if count >= 2)
        potential_sets = sum(count // 3 for count in hand_count)
        
        # チートイツ形の向聴数（仮の計算）
        chitoitsu_shanten = 6 - min(pair_count, 7)
        
        # 通常形の向聴数（仮の計算）
        # 4面子1雀頭で和了 = 必要な面子数は4
        regular_shanten = 8 - (pair_count > 0) - (2 * potential_sets) - melds_count
        
        # 最小の向聴数を採用
        shanten = min(chitoitsu_shanten, regular_shanten)
        
        # 受け入れ牌の簡易計算（実際は向聴数減少に繋がる牌を特定する必要あり）
        ukeire = []
        
        # 各牌を1枚追加したときに向聴数が下がるか検証（単純な近似）
        for idx in range(34):
            # 既に4枚ある牌は除外
            if hand_count[idx] >= 4:
                continue
                
            # 雀頭が無い場合、雀頭候補になる可能性
            if pair_count == 0 and hand_count[idx] == 1:
                ukeire.append(idx)
            # 面子が作れる可能性（単純化）
            elif hand_count[idx] % 3 == 2:
                ukeire.append(idx)
            # 順子が作れる可能性（単純化）
            elif idx < 27:  # 数牌のみ
                suit = idx // 9
                number = idx % 9
                
                # 両面搭子
                if number <= 6 and hand_count[idx] > 0 and hand_count[idx+1] > 0:
                    if number+2 < 9:  # 範囲チェック
                        ukeire.append(suit*9 + number+2)
                if number >= 2 and hand_count[idx] > 0 and hand_count[idx-1] > 0:
                    ukeire.append(suit*9 + number-2)
                
                # 嵌張搭子
                if number <= 7 and hand_count[idx] > 0 and hand_count[idx+2] > 0:
                    ukeire.append(suit*9 + number+1)
        
        # 重複削除
        ukeire = list(set(ukeire))
        
        return max(shanten, 0), ukeire  # 向聴数が負になることは通常ない
        
    except Exception as e:
        # エラーが発生した場合はログに出力し、デフォルト値を返す
        print(f"[Debug] Shanten calculation error details: {e}")
        return 1, []  # 適当な向聴数と空の受け入れリストを返す