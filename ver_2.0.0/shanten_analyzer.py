import re
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
from tile_utils import tile_id_to_index, hand_ids_to_34_array, NUM_TILE_TYPES, NUM_PLAYERS

# Shantenインスタンスは一度だけ生成して再利用する
shanten_calculator = Shanten()

def get_shanten_after_discard(tiles_34_14: list[int]) -> tuple[int, int]:
    """
    14枚の手牌から1枚捨てて13枚にした時の最小シャンテン数を計算します。
    古いバージョンの mahjong ライブラリの書き方に対応。
    """
    min_shanten = 8
    best_discard_tile = -1
    
    unique_tiles_in_hand = [i for i, count in enumerate(tiles_34_14) if count > 0]
    if not unique_tiles_in_hand:
        return min_shanten, best_discard_tile

    for discard_index in unique_tiles_in_hand:
        temp_hand_13 = list(tiles_34_14)
        temp_hand_13[discard_index] -= 1
        
        # ★★★★★★★★★★★★★★★★★★★★★★★ 修正箇所 ★★★★★★★★★★★★★★★★★★★★★★★
        # 一般手、七対子、国士無双のシャンテン数をそれぞれ計算し、最小値を取る
        shanten_regular = shanten_calculator.calculate_shanten_for_regular_hand(temp_hand_13)
        shanten_chiitoitsu = shanten_calculator.calculate_shanten_for_chiitoitsu_hand(temp_hand_13)
        shanten_kokushi = shanten_calculator.calculate_shanten_for_kokushi_hand(temp_hand_13)
        
        shanten = min(shanten_regular, shanten_chiitoitsu, shanten_kokushi)
        # ★★★★★★★★★★★★★★★★★★★★★★★ 修正完了 ★★★★★★★★★★★★★★★★★★★★★★★

        if shanten < min_shanten:
            min_shanten = shanten
            best_discard_tile = discard_index
            
    return min_shanten, best_discard_tile

def calculate_ukeire(tiles_34_13: list[int], all_visible_tiles_34: list[int]) -> tuple[list[int], int]:
    """
    13枚の手牌に対する受け入れ牌と枚数を計算する。
    古いバージョンの mahjong ライブラリの書き方に対応。
    """
    # ★★★★★★★★★★★★★★★★★★★★★★★ 修正箇所 ★★★★★★★★★★★★★★★★★★★★★★★
    shanten_regular = shanten_calculator.calculate_shanten_for_regular_hand(tiles_34_13)
    shanten_chiitoitsu = shanten_calculator.calculate_shanten_for_chiitoitsu_hand(tiles_34_13)
    shanten_kokushi = shanten_calculator.calculate_shanten_for_kokushi_hand(tiles_34_13)
    current_shanten = min(shanten_regular, shanten_chiitoitsu, shanten_kokushi)
    # ★★★★★★★★★★★★★★★★★★★★★★★ 修正完了 ★★★★★★★★★★★★★★★★★★★★★★★
    
    if current_shanten < 0: return [], 0

    ukeire_tiles = {}
    for draw_index in range(NUM_TILE_TYPES):
        if all_visible_tiles_34[draw_index] >= 4:
            continue

        hand_14_after_draw = list(tiles_34_13)
        hand_14_after_draw[draw_index] += 1
        
        shanten_after_draw_and_discard, _ = get_shanten_after_discard(hand_14_after_draw)
        
        if shanten_after_draw_and_discard < current_shanten:
            remaining_count = 4 - all_visible_tiles_34[draw_index]
            ukeire_tiles[draw_index] = remaining_count
            
    ukeire_indices = sorted(ukeire_tiles.keys())
    total_枚数 = sum(ukeire_tiles.values())
    
    return ukeire_indices, total_枚数

def analyze_speed_metrics(hand_14_ids: list[int], discard_id: int, game_state) -> tuple[int, int]:
    """攻撃指標（Speed）を計算する。game_stateを引数に追加"""
    hand_14_34 = hand_ids_to_34_array(hand_14_ids)
    discard_idx = tile_id_to_index(discard_id)
    if discard_idx == -1: return 0, 0

    shanten_before, _ = get_shanten_after_discard(hand_14_34)

    hand_13_34 = list(hand_14_34)
    if hand_13_34[discard_idx] > 0:
        hand_13_34[discard_idx] -= 1
    else: 
        return 0, 0
    
    # ★★★★★★★★★★★★★★★★★★★★★★★ 修正箇所 ★★★★★★★★★★★★★★★★★★★★★★★
    shanten_regular_after = shanten_calculator.calculate_shanten_for_regular_hand(hand_13_34)
    shanten_chiitoitsu_after = shanten_calculator.calculate_shanten_for_chiitoitsu_hand(hand_13_34)
    shanten_kokushi_after = shanten_calculator.calculate_shanten_for_kokushi_hand(hand_13_34)
    shanten_after = min(shanten_regular_after, shanten_chiitoitsu_after, shanten_kokushi_after)
    # ★★★★★★★★★★★★★★★★★★★★★★★ 修正完了 ★★★★★★★★★★★★★★★★★★★★★★★

    shanten_change = shanten_before - shanten_after

    # 全見え牌を計算
    all_visible_tiles_34 = [0] * NUM_TILE_TYPES
    for tile_idx, count in enumerate(hand_13_34):
        all_visible_tiles_34[tile_idx] += count
    for p in range(NUM_PLAYERS):
        for tile, _ in game_state.player_discards[p]:
            idx = tile_id_to_index(tile)
            if idx != -1: all_visible_tiles_34[idx] += 1
        for meld in game_state.player_melds[p]:
            for tile in meld.get('tiles', []):
                idx = tile_id_to_index(tile)
                if idx != -1: all_visible_tiles_34[idx] += 1
    for tile in game_state.dora_indicators:
        idx = tile_id_to_index(tile)
        if idx != -1: all_visible_tiles_34[idx] += 1
    
    _, ukeire_count = calculate_ukeire(hand_13_34, all_visible_tiles_34)

    return shanten_change, ukeire_count