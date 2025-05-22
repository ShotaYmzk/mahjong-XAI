# tile_utils.py

def tile_id_to_index(tile: int) -> int:
    """
    牌ID（0～135）を牌種インデックス（0～33）に変換する関数
    Converts tile ID (0-135) to tile type index (0-33).
    Returns -1 for invalid IDs.
    """
    if not isinstance(tile, int) or not (0 <= tile <= 135):
        return -1
    # 赤ドラも通常牌のインデックスにマッピング
    if tile == 16: tile = 20 # 0m (赤5m) -> 5m (ID 20-23)
    elif tile == 52: tile = 56 # 0p (赤5p) -> 5p (ID 56-59)
    elif tile == 88: tile = 92 # 0s (赤5s) -> 5s (ID 92-95)

    if tile < 108: # Man, Pin, Sou (数牌)
        # 0-8: 1m-9m, 9-17: 1p-9p, 18-26: 1s-9s
        return tile // 4
    else: # Honors (字牌)
        # 27-33: ESWNWhGrR (東南西北白發中)
        return 27 + (tile - 108) // 4

def tile_index_to_id(index: int, prefer_red: bool = False) -> int:
    """
    牌種インデックス（0～33）を代表的な牌ID（0～135）に変換する関数
    Converts tile type index (0-33) back to a representative tile ID (0-135).
    If prefer_red is True and the index corresponds to a 5, it returns the red five ID.
    Otherwise, it returns the ID of the first non-red tile of that kind (e.g., 5m -> ID 20).
    Returns -1 for invalid indices.
    """
    if not isinstance(index, int) or not (0 <= index <= 33):
        return -1

    if index < 27: # Number tiles (数牌)
        suit = index // 9
        num_in_suit = index % 9 # 0-8 for 1-9

        if prefer_red and num_in_suit == 4: # 5の牌の場合
            if suit == 0: return 16 # 赤5m
            if suit == 1: return 52 # 赤5p
            if suit == 2: return 88 # 赤5s
        
        # 通常の牌ID (各牌種の0番目の牌、例: 1m -> 0, 5m -> 20, 1p -> 36)
        return index * 4
    else: # Honor tiles (字牌)
        # 27:E, 28:S, 29:W, 30:N, 31:Wh, 32:Gr, 33:R
        # (index - 27) gives 0-6 for honors.
        # Base ID for East is 108.
        return 108 + (index - 27) * 4

def tile_id_to_string(tile: int) -> str:
    """
    牌ID（0～135）を文字列表現に変換する関数（例："1m", "東", "0s" for red 5s）
    Converts tile ID (0-135) to a string representation (e.g., "1m", "東", "0s" for red 5s).
    Returns "?" for invalid IDs.
    """
    if not isinstance(tile, int) or not (0 <= tile <= 135):
        return "?"

    # 赤ドラのID
    if tile == 16: return "0m" # Red 5 Man
    if tile == 52: return "0p" # Red 5 Pin
    if tile == 88: return "0s" # Red 5 Sou

    # 数牌 (0-107)
    if tile < 108:
        suit_char = ["m", "p", "s"][tile // 36]
        number = (tile % 36) // 4 + 1 # 1-9
        # 通常の5の牌のID (20, 56, 92) かどうかを判定
        # tile_id_to_index で赤ドラを通常牌のインデックスに変換しているので、
        # ここでは number が 5 であれば "5" と表示する
        return str(number) + suit_char
    # 字牌 (108-135)
    else:
        honors = ["東", "南", "西", "北", "白", "發", "中"]
        honor_index = (tile - 108) // 4
        if 0 <= honor_index < len(honors):
            return honors[honor_index]
        else:
            return "?" # Should not happen

def is_aka_dora(tile: int) -> bool:
    """
    赤ドラ(aka dora)かどうかを判定する関数
    Returns True if the given tile ID is a red five (aka dora), otherwise False.
    """
    AKA_DORA_IDS = {16, 52, 88}
    return tile in AKA_DORA_IDS