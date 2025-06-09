import re
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter

def robust_hand_parser(hand_string: str) -> list[int]:
    """
    "123m456m"や"0s"(赤5索)のような変則的な文字列も正しく解析するパーサー。
    """
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

def format_tiles_for_display(tile_indices):
    """
    牌のインデックスのリストを表示用の文字列に変換します。
    """
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

def format_shanten(shanten_value: int) -> str:
    """
    シャンテン数を「N向聴」または「聴牌」の文字列に変換します。
    """
    if shanten_value == 0:
        return "聴牌"
    if shanten_value < 0:
        return "和了"
    return f"{shanten_value}向聴"

def get_shanten_after_best_discard(tiles_14, calculator, shanten_func_name):
    """
    14枚の手牌から1枚捨てて13枚にした時の、最小シャンテン数を計算します。
    """
    shanten_func = getattr(calculator, shanten_func_name)
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

def analyze_hand_details(hand_string: str):
    """
    手牌のシャンテン数を形式別に計算し、最適な打牌と受け入れを分析します。
    """
    shanten_calculator = Shanten()
    
    # 新しいパーサーを使って手牌を34種配列に変換
    tiles_34_14 = robust_hand_parser(hand_string)
    unique_tiles_in_hand_14 = sorted([i for i, count in enumerate(tiles_34_14) if count > 0])

    # --- 形式別シャンテン数計算 ---
    shanten_regular = get_shanten_after_best_discard(tiles_34_14, shanten_calculator, 'calculate_shanten_for_regular_hand')
    shanten_chiitoitsu = get_shanten_after_best_discard(tiles_34_14, shanten_calculator, 'calculate_shanten_for_chiitoitsu_hand')
    shanten_kokushi = get_shanten_after_best_discard(tiles_34_14, shanten_calculator, 'calculate_shanten_for_kokushi_hand')

    print(f"現在の手牌: {hand_string}")
    print("--- 形式別シャンテン数 ---")
    print(f"一般形: {format_shanten(shanten_regular)}")
    print(f"七対子: {format_shanten(shanten_chiitoitsu)}")
    print(f"国士無双: {format_shanten(shanten_kokushi)}")
    print("-" * 30)

    # --- 最適な打牌と受け入れの計算 ---
    analysis_results = []
    
    for discard_index in unique_tiles_in_hand_14:
        hand_13_tiles = list(tiles_34_14)
        hand_13_tiles[discard_index] -= 1
        shanten_13 = shanten_calculator.calculate_shanten(hand_13_tiles)
        
        ukeire_for_discard = {}
        
        if shanten_13 == 0:  # 聴牌の場合: 待ち牌を計算
            for draw_index in range(34):
                # 5枚目になる牌は引けない and 自分が捨てた牌はフリテンになるので待ちに含めない
                if tiles_34_14[draw_index] < 4 and draw_index != discard_index:
                    temp_hand_14 = list(hand_13_tiles)
                    temp_hand_14[draw_index] += 1
                    # アガリ(-1)になる牌を探す
                    if shanten_calculator.calculate_shanten(temp_hand_14) == -1:
                        remaining_count = 4 - tiles_34_14[draw_index]
                        ukeire_for_discard[draw_index] = remaining_count
        else:  # 聴牌していない場合: シャンテン数を進める牌を計算
            for draw_index in range(34):
                if tiles_34_14[draw_index] < 4:
                    hand_14_after_draw = list(hand_13_tiles)
                    hand_14_after_draw[draw_index] += 1
                    shanten_after_draw_and_discard = get_shanten_after_best_discard(hand_14_after_draw, shanten_calculator, 'calculate_shanten')
                    if shanten_after_draw_and_discard < shanten_13:
                        remaining_count = 4 - tiles_34_14[draw_index]
                        ukeire_for_discard[draw_index] = remaining_count

        total_ukeire_count = sum(ukeire_for_discard.values())
        analysis_results.append({
            "discard_index": discard_index,
            "ukeire": ukeire_for_discard,
            "total_count": total_ukeire_count,
            "shanten_after_discard": shanten_13
        })
        
    sorted_results = sorted(analysis_results, key=lambda x: (x['shanten_after_discard'], -x['total_count']))
    
    print("--- 打牌候補と受け入れ（シャンテン数・枚数順）---")
    for result in sorted_results:
        discard_str = format_tiles_for_display([result["discard_index"]])
        ukeire_str = format_tiles_for_display(sorted(result["ukeire"].keys()))
        total_枚数 = result["total_count"]
        shanten_display = format_shanten(result["shanten_after_discard"])
        
        print(f"打{discard_str} ({shanten_display}) 摸[{ukeire_str} {total_枚数}枚]")


# 新しい手牌を指定
new_hand_string = "578m1145p14s4z7m"

# 解析を実行
analyze_hand_details(new_hand_string)