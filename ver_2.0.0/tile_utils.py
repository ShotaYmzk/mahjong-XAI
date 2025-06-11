def tile_id_to_index(tile: int) -> int:
    """
    牌ID（0～135）を牌種インデックス（0～33）に変換する関数
    Converts tile ID (0-135) to tile type index (0-33).
    Returns -1 for invalid IDs.
    """
    if not isinstance(tile, int) or not (0 <= tile <= 135):
        return -1
    if tile < 108: # Man, Pin, Sou
        suit_index = tile // 36  # 0: Man, 1: Pin, 2: Sou
        number_index = (tile % 36) // 4 # 0-8 for 1-9
        return suit_index * 9 + number_index
    else: # Honors
        # 27: East, 28: South, 29: West, 30: North, 31: White, 32: Green, 33: Red
        return 27 + (tile - 108) // 4

def tile_index_to_id(index: int, prefer_red: bool = False) -> int:
    """
    牌種インデックス（0～33）を代表的な牌ID（0～135）に変換する関数
    Converts tile type index (0-33) back to a representative tile ID (0-135).
    Note: This loses information about specific red fives unless requested.
    Returns -1 for invalid indices.
    """
    if not isinstance(index, int) or not (0 <= index <= 33):
        return -1

    if index < 27: # Number tiles
        suit_index = index // 9
        number_index = index % 9
        # Special handling for red fives (IDs 16, 52, 88 are 0m, 0p, 0s)
        if prefer_red:
            if index == 4: return 16 # 5m -> 0m (ID 16)
            if index == 13: return 52 # 5p -> 0p (ID 52)
            if index == 22: return 88 # 5s -> 0s (ID 88)
        # Default non-red ID (e.g., 5m is ID 17)
        base_id = suit_index * 36 + number_index * 4
        # Return the second instance (offset 1) as representative, avoid 0m/0p/0s
        return base_id + 1 if number_index == 4 else base_id
    else: # Honor tiles
        base_id = 108 + (index - 27) * 4
        return base_id

def tile_id_to_string(tile: int) -> str:
    """
    牌ID（0～135）を文字列表現に変換する関数（例："1m", "東", "0s" for red 5s）
    Converts tile ID (0-135) to a string representation (e.g., "1m", "東", "0s" for red 5s).
    Returns "?" for invalid IDs.
    """
    if not isinstance(tile, int) or not (0 <= tile <= 135):
        return "?"

    # Red fives (0m, 0p, 0s) have specific IDs
    if tile == 16: return "0m" # Red 5 Man
    if tile == 52: return "0p" # Red 5 Pin
    if tile == 88: return "0s" # Red 5 Sou

    suits = ["m", "p", "s", "z"]
    numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    # Honors in Tenhou order: E, S, W, N, White, Green, Red
    honors = ["東", "南", "西", "北", "白", "發", "中"]

    if tile < 108: # Man, Pin, Sou
        suit_index = tile // 36
        number_index = (tile % 36) // 4
        # Check if it's a non-red 5
        if number_index == 4 and tile not in [16, 52, 88]:
             num_str = "5" # Display regular 5s as '5'
        else:
             num_str = numbers[number_index]
        return num_str + suits[suit_index]
    else: # Honors
        honor_index = (tile - 108) // 4
        if 0 <= honor_index < len(honors):
            return honors[honor_index]
        else:
            return "?" # Should not happen with valid IDs
        
def is_aka_dora(tile: int) -> bool:
    """
    赤ドラ(aka dora)かどうかを判定する関数
    Returns True if the given tile ID is a red five (aka dora), otherwise False.
    """
    # Tenhou における赤ドラの tile IDs
    AKA_DORA_IDS = {16, 52, 88}
    return tile in AKA_DORA_IDS
