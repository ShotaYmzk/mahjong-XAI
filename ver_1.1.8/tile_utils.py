# naki_utils.py (Revised decode_naki function - Attempts Pon/Kan tile selection)
try:
    from tile_utils import tile_id_to_index, tile_id_to_string
except ImportError:
    print("[Error] Cannot import from tile_utils.py in naki_utils.")
    def tile_id_to_index(tile: int) -> int: return tile // 4 if tile != -1 else -1
    def tile_id_to_string(tile: int) -> str: return f"tile_{tile}" if tile != -1 else "?"

def tile_to_name(tile: int) -> str:
    return tile_id_to_string(tile)

def decode_naki(m: int) -> tuple:
    """
    天鳳の副露面子のビットフィールドをデコードする関数 (詳細解析版)
    Args:
        m (int): <N>タグの m 属性の値
    Returns:
        tuple: (鳴きの種類, 構成牌IDリスト, 鳴かれた牌ID(推定), 誰から鳴いたか(相対, 0:下家, 1:対面, 2:上家, 3:自分))
               鳴きの種類: "チー", "ポン", "大明槓", "加槓", "暗槓", "不明"
               構成牌IDリスト: 面子を構成する牌IDのリスト (3枚 or 4枚) - ポン/カンも確定試行
               鳴かれた牌ID: チーの場合は確定、ポン/大明槓/加槓/暗槓は -1 (パーサーで特定)
               誰から(相対): ビット0,1の値 (パーサーで特定推奨)
    """
    kui = m & 3
    trigger_pai_id = -1
    tiles = []

    try:
        # --- チー ---
        if m & (1 << 2):
            naki_type = "チー"
            t = (m & 0xFC00) >> 10
            r = t % 3 # 鳴いた牌の相対位置(0-2)
            t //= 3
            base_tile_index = (t // 7) * 9 + (t % 7)
            if not (0 <= base_tile_index <= 24): return "不明", [], -1, kui
            indices = [base_tile_index + 0, base_tile_index + 1, base_tile_index + 2]
            offsets = [(m & 0x0018) >> 3, (m & 0x0060) >> 5, (m & 0x0180) >> 7]
            temp_tiles = []
            valid = True
            for idx, offset in zip(indices, offsets):
                tile_id = idx * 4 + offset
                if not (0 <= tile_id <= 135): valid = False; break
                temp_tiles.append(tile_id)
            if not valid: return "不明", [], -1, kui
            if not (0 <= r < 3): return "不明", [], -1, kui
            trigger_pai_id = temp_tiles[r] # Trigger is determined for Chi
            tiles = sorted(temp_tiles)

        # --- ポン ---
        elif m & (1 << 3):
            naki_type = "ポン"
            trigger_pai_id = -1 # Parser determines trigger
            t = (m & 0xFE00) >> 9
            # r = t % 3 # 鳴いた牌の表示位置? Tenhou internal? Not reliable for source.
            t //= 3 # 牌の種類インデックス
            if not (0 <= t <= 33): return "不明", [], -1, kui

            base_id = t * 4
            possible_ids = [base_id + 0, base_id + 1, base_id + 2, base_id + 3]
            # Use unused offset bits (5, 6) to determine the 3 tiles used for the Pon
            unused_offset = (m & 0x0060) >> 5
            tiles = []
            for i in range(4):
                if i != unused_offset:
                    tiles.append(possible_ids[i])
            if len(tiles) != 3:
                 print(f"[Error decode_naki Pon] Failed to select 3 tiles based on unused offset {unused_offset}. m={m}")
                 # Fallback to candidate list if selection fails? Or return error?
                 tiles = [] # Return empty on error
                 naki_type = "不明"

        # --- 加槓 ---
        elif m & (1 << 4):
            naki_type = "加槓"
            trigger_pai_id = -1 # Parser knows added tile from hand
            added_offset = (m & 0x0060) >> 5 # Offset of the tile added from hand
            t = (m & 0xFE00) >> 9
            t //= 3 # 牌の種類インデックス
            if not (0 <= t <= 33): return "不明", [], -1, kui

            base_id = t * 4
            added_pai_id = base_id + added_offset
            # We need the original Pon meld to know the other 3 tiles.
            # This requires GameState. We can only *guess* the 4 tiles.
            # Best effort: return the 4 potential IDs. Parser/GameState must verify.
            tiles = [base_id, base_id + 1, base_id + 2, base_id + 3]
            # Alternatively, try to represent the added tile info?
            # tiles = [added_pai_id] # Signal which tile was added?

        # --- 大明槓 or 暗槓 ---
        else:
            tile_info = (m & 0xFF00) >> 8
            tile_index = tile_info // 4
            if not (0 <= tile_index <= 33): return "不明", [], -1, kui

            base_id = tile_index * 4
            # The 4 tiles are of the same type. Assume standard IDs.
            tiles = [base_id, base_id + 1, base_id + 2, base_id + 3]

            if kui == 0: # 暗槓
                naki_type = "暗槓"
                trigger_pai_id = -1 # No external trigger
                # Tiles need verification against hand in GameState if red fives matter.
            else: # 大明槓
                naki_type = "大明槓"
                trigger_pai_id = -1 # Parser determines trigger
                # Tiles need verification against hand/trigger in GameState if red fives matter.

        # Return determined type, sorted tiles, trigger (only reliable for Chi), and raw kui
        return naki_type, sorted(tiles), trigger_pai_id, kui

    except Exception as e:
        print(f"[Error] Failed to decode naki m={m}: {e}")
        import traceback
        traceback.print_exc()
        return "不明", [], -1, m & 3
    
# tile_utils.py

def tile_id_to_index(tile: int) -> int:
    """
    牌ID（0～135）を牌種インデックス（0～33）に変換する関数
    """
    if tile < 108:
        suit_index = tile // 36  # 0: 萬子, 1: 筒子, 2: 索子
        number_index = (tile % 36) // 4
        return suit_index * 9 + number_index
    else:
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
    if tile < 108:
        suit_index = tile // 36
        number_index = (tile % 36) // 4
        return numbers[number_index] + suits[suit_index]
    else:
        honor_index = (tile - 108) // 4
        return honors[honor_index]