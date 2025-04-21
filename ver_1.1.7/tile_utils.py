# tile_utils.py
# (Keep the existing code as provided in the prompt)

def tile_id_to_index(tile: int) -> int:
    """
    牌ID（0～135）を牌種インデックス（0～33）に変換する関数
    """
    if tile == -1: # Handle potential invalid tile ID
        return -1
    if tile < 108:
        suit_index = tile // 36  # 0: 萬子, 1: 筒子, 2: 索子
        number_index = (tile % 36) // 4
        return suit_index * 9 + number_index
    else:
        # Ensure tile is within valid range for honors
        if tile < 108 or tile > 135:
             # print(f"[Warning tile_utils] Invalid honor tile ID: {tile}") # Optional warning
             return -1 # Or handle error appropriately
        return 27 + (tile - 108) // 4

def tile_id_to_string(tile: int) -> str:
    """
    牌ID（0～135）を文字列表現に変換する関数（例："1m", "東"）
    """
    if tile < 0 or tile > 135:
        return "不明"
    suits = ["m", "p", "s", "z"]
    numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    honors = ["東", "南", "西", "北", "白", "發", "中"]
    # 赤ドラ処理を追加 (例: 5m, 5p, 5s の ID 16, 52, 88 を赤5として扱うなど。
    # Tenhou logs often use specific IDs like 51 for red 5p. GameState/Parser needs to handle this.
    # If red dora IDs are normalized (e.g., always 16, 52, 88), this function is fine.
    # If specific IDs (like 51) are used, this needs adjustment or upstream handling.
    # Assuming IDs are normalized for now.

    tile_idx = tile_id_to_index(tile)
    if tile_idx == -1: return "不明"

    if tile_idx < 27: # 数牌 (0-26)
        suit_index = tile_idx // 9
        number_index = tile_idx % 9
        # Check for potential red dora based on ID offset (0=normal, 1/2/3=variant including red)
        # A common convention is ID % 4 == 0 for red 5s (e.g. 16, 52, 88)
        # However, Tenhou uses specific IDs (51 for red 5p etc) which vectorize_state needs to handle
        is_red_potential = (numbers[number_index] == '5' and tile % 4 == 0) # Example based on one convention
        # A more robust way depends on how red dora are represented in your specific GameState/parsing
        red_marker = "r" if is_red_potential else "" # Placeholder
        return red_marker + numbers[number_index] + suits[suit_index]
    else: # 字牌 (27-33)
        honor_index = tile_idx - 27
        return honors[honor_index]

# Optional: Add index back to string for convenience
def tile_index_to_string(index: int) -> str:
    """
    牌種インデックス（0-33）を文字列表現に変換
    """
    if index < 0 or index > 33:
        return "不明"
    suits = ["m", "p", "s", "z"]
    numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    honors = ["東", "南", "西", "北", "白", "發", "中"]
    if index < 27: # 0-26
        suit_index = index // 9
        number_index = index % 9
        return numbers[number_index] + suits[suit_index]
    else: # 27-33
        honor_index = index - 27
        return honors[honor_index]