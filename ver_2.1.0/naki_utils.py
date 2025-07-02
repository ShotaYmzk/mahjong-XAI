# /ver_1.1.9/naki_utils.py
import traceback
from typing import Dict, List, Tuple

# --- Modified Import ---
try:
    # tile_utils must be in the same directory or Python path
    from tile_utils import tile_id_to_index, tile_id_to_string, is_aka_dora
    TILE_UTILS_AVAILABLE = True
except ImportError:
    TILE_UTILS_AVAILABLE = False
    # Provide minimal fallbacks ONLY to prevent immediate script crash during import phase.
    # Functionality will be severely limited.
    print("[Critical Error] Cannot import from tile_utils.py. Using Fallbacks in naki_utils.")
    def tile_id_to_index(tile: int) -> int: return tile // 4 if tile >= 0 else -1 # Basic fallback
    def tile_id_to_string(tile: int) -> str: return f"ERR_TILE_{tile}"
    def is_aka_dora(tile_id: int) -> bool: return False
# --- End Modified Import ---


def tile_to_name(tile: int) -> str:
    """Helper function to maintain compatibility"""
    # Use the potentially imported or fallback function
    return tile_id_to_string(tile)

def decode_naki(m: int) -> Dict:
    """
    天鳳の副露面子のビットフィールド(m)をデコードする関数 (修正版)
    Args:
        m (int): <N>タグの m 属性の値
    Returns:
        Dict: {
            "type": 鳴きの種類 ("チー", "ポン", "大明槓", "加槓", "暗槓", "不明"),
            "tiles": 構成牌IDリスト (鳴き元の牌を含む),
            "from_who_relative": 鳴き元プレイヤーの相対位置 (0:下家, 1:対面, 2:上家), -1 for 暗槓/加槓/不明
            "consumed": 手牌から消費された牌IDのリスト (鳴き元の牌を除く),
            "raw_value": m # 元の値を保持
        }
        注意: called_tile (鳴かれた牌そのもの) は m だけでは特定できない場合があるため、
              呼び出し元で直前の捨て牌などから判断する必要がある。
              from_who も相対位置のみデコード。
    """
    result = {
        "type": "不明", "tiles": [], "from_who_relative": -1,
        "consumed": [], "raw_value": m
    }
    try:
        from_who_relative = m & 3 # 0:下家, 1:対面, 2:上家 (槓の場合は意味が異なる可能性あり)
        result["from_who_relative"] = from_who_relative

        # --- チー ---
        if m & (1 << 2):
            result["type"] = "チー"
            # Interpret based on https://gimite.net/pukiwiki/index.php?Tenhou%20Log%20Format
            t = m >> 10
            r = t % 3
            t //= 3
            base_index = -1
            if 0 <= t <= 6: base_index = t # 1m-7m
            elif 7 <= t <= 13: base_index = (t - 7) + 9 # 1p-7p
            elif 14 <= t <= 20: base_index = (t - 14) + 18 # 1s-7s
            else:
                print(f"[Warn decode_naki Chi] Invalid t value {t} after division for base index calculation. m={m}")
                return result # Return with type "不明"

            if not (0 <= base_index <= 24): # Check base index range (sequences start from 1-7)
                print(f"[Warn decode_naki Chi] Invalid base_index {base_index} calculated. m={m}")
                return result

            offsets = [(m >> 3) & 3, (m >> 5) & 3, (m >> 7) & 3]
            tiles_in_sequence = []
            consumed_tiles = []
            for i in range(3):
                tile_kind = base_index + i
                tile_id = tile_kind * 4 + offsets[i]
                if not (0 <= tile_id <= 107): # Must be number tiles
                     print(f"[Warn decode_naki Chi] Invalid tile_id {tile_id} calculated. m={m}")
                     result["tiles"] = []; result["consumed"] = []
                     return result
                tiles_in_sequence.append(tile_id)
                # r番目の牌が鳴かれた牌なので、それ以外が消費された牌
                if i != r:
                    consumed_tiles.append(tile_id)

            result["tiles"] = sorted(tiles_in_sequence)
            result["consumed"] = sorted(consumed_tiles)

        # --- ポン ---
        elif m & (1 << 3):
            result["type"] = "ポン"
            t = m >> 9
            t //= 3
            if not (0 <= t <= 33):
                print(f"[Warn decode_naki Pon] Invalid tile index {t}. m={m}")
                return result

            base_id = t * 4
            possible_ids = [base_id + 0, base_id + 1, base_id + 2, base_id + 3]
            unused_offset = (m >> 5) & 3

            tiles_in_meld = []
            for i in range(4):
                if i != unused_offset:
                    tiles_in_meld.append(possible_ids[i])
            if len(tiles_in_meld) != 3:
                 print(f"[Warn decode_naki Pon] Failed to select 3 tiles (unused={unused_offset}). m={m}")
                 return result

            result["tiles"] = sorted(tiles_in_meld)
            # Consumed tiles cannot be determined from 'm' alone for Pon.
            # GameState needs to identify the called tile and deduce the other two.
            result["consumed"] = [] # Leave empty, GameState handles this

        # --- 加槓 ---
        elif m & (1 << 4):
            result["type"] = "加槓"
            result["from_who_relative"] = -1 # 自分自身からのアクション
            t = m >> 9
            t //= 3
            if not (0 <= t <= 33):
                print(f"[Warn decode_naki Kakan] Invalid tile index {t}. m={m}")
                return result

            base_id = t * 4
            added_offset = (m >> 5) & 3
            added_tile_id = base_id + added_offset

            # Result includes all 4 tiles of the kind for consistency
            result["tiles"] = sorted([base_id, base_id + 1, base_id + 2, base_id + 3])
            # Consumed is the single tile added from hand
            result["consumed"] = [added_tile_id]

        # --- 大明槓 or 暗槓 ---
        else: # ビット2,3,4が0
            tile_id_raw = m >> 8
            tile_index = tile_id_raw // 4
            if not (0 <= tile_index <= 33):
                 print(f"[Warn decode_naki Kan] Invalid tile_index {tile_index} from m={m}")
                 return result

            base_id = tile_index * 4
            result["tiles"] = sorted([base_id, base_id + 1, base_id + 2, base_id + 3])

            if from_who_relative != -1 and from_who_relative != 3 : # 大明槓
                result["type"] = "大明槓"
                # Consumed tiles cannot be determined from 'm' alone for Daiminkan.
                result["consumed"] = [] # GameState handles this
            else: # 暗槓
                result["type"] = "暗槓"
                result["from_who_relative"] = -1 # 自分自身
                # Consumed are the 4 tiles from hand
                result["consumed"] = result["tiles"] # Assume all 4 from hand

        return result

    except Exception as e:
        print(f"[Error] Exception during decode_naki(m={m}): {e}")
        traceback.print_exc()
        result["type"] = "不明" # Ensure type is '不明' on error
        return result