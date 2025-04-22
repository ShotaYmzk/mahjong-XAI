# naki_utils.py (Revised and improved based on user code and Tenhou format)
import traceback
from typing import Dict, List, Tuple

try:
    # tile_utils must be in the same directory or Python path
    from tile_utils import tile_id_to_index, tile_id_to_string, is_aka_dora
except ImportError:
    print("[Critical Error] Cannot import from tile_utils.py. Please ensure it's accessible.")
    # Provide minimal fallbacks ONLY to prevent immediate script crash during import phase.
    # Functionality will be severely limited.
    def tile_id_to_index(tile: int) -> int: return -1
    def tile_id_to_string(tile: int) -> str: return f"ERR_TILE_{tile}"
    def is_aka_dora(tile_id: int) -> bool: return False

def tile_to_name(tile: int) -> str:
    """Helper function to maintain compatibility"""
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
            t = (m >> 10) & 0x3FF # 10ビット分取得
            # 順子の最小牌の / 3 した値 + 7bitオフセット ? (Tenhou docsより)
            # t = (m & 0xFC00) >> 10 # User provided version
            # Let's re-interpret based on https://gimite.net/pukiwiki/index.php?Tenhou%20Log%20Format
            # 「m >> 10: t」 「t % 3: r (鳴いた牌の順子の相対位置 0..2)」
            # 「t /= 3」 「base = (t / 7) * 9 + (t % 7)」: 順子の最小牌インデックス(0..8, 9..17, 18..26)
            # 「(m >> 3) & 3」「(m >> 5) & 3」「(m >> 7) & 3」: 各牌のオフセット(0..3)
            t = m >> 10
            r = t % 3
            t //= 3
            if not (0 <= t <= 20): # (26 / 7) * 9 + (26 % 7) is invalid logic. t must be in 0-20 for sequences.
                 # Check if t is a valid sequence start index calculation input
                 # Max index is 9m(8), 9p(17), 9s(26). Sequence can start up to 7.
                 # Max suit start index = 18 (索子)
                 # Sequence start index = 7m (6), 7p (15), 7s (24) are the highest possible starts.
                 # Let's recalculate base index carefully.
                base_index = -1
                if 0 <= t <= 6: base_index = t # 1m-7m
                elif 7 <= t <= 13: base_index = (t - 7) + 9 # 1p-7p
                elif 14 <= t <= 20: base_index = (t - 14) + 18 # 1s-7s
                else:
                    print(f"[Warn decode_naki Chi] Invalid t value {t} after division for base index calculation. m={m}")
                    return result # Return with type "不明"

            else: # Original interpretation from user code (might be correct)
                if t // 7 > 2: # Suit index check
                    print(f"[Warn decode_naki Chi] Invalid suit derived from t={t}. m={m}")
                    return result
                base_index = (t // 7) * 9 + (t % 7) # 0..6 -> 0..6; 7..13 -> 9..15; 14..20 -> 18..24

            if base_index == -1 or not (0 <= base_index <= 24): # Check base index range
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
                     result["tiles"] = []
                     result["consumed"] = []
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
            # 「m >> 9: t」「t % 3: r (鳴いた牌の相対位置？)」 「t /= 3: 牌の種類インデックス」
            # 「(m >> 5) & 3: unused_offset (0..3 使わなかった牌)」
            t = m >> 9
            # r = t % 3 # ポンにおける r の意味は不明瞭 or 使わない？ from_who_relative で判断
            t //= 3
            if not (0 <= t <= 33):
                print(f"[Warn decode_naki Pon] Invalid tile index {t}. m={m}")
                return result

            base_id = t * 4
            possible_ids = [base_id + 0, base_id + 1, base_id + 2, base_id + 3]
            unused_offset = (m >> 5) & 3

            tiles_in_meld = []
            consumed_tiles = []
            for i in range(4):
                if i != unused_offset:
                    tiles_in_meld.append(possible_ids[i])
            if len(tiles_in_meld) != 3:
                 print(f"[Warn decode_naki Pon] Failed to select 3 tiles (unused={unused_offset}). m={m}")
                 return result # Should not happen

            result["tiles"] = sorted(tiles_in_meld)
            # 消費されたのは鳴き元以外の2枚 -> mからだけでは特定できない。
            # 呼び出し元で called_tile を特定し、それ以外の2枚を consumed とする。
            # result["consumed"] is left empty here.

        # --- 加槓 ---
        elif m & (1 << 4):
            result["type"] = "加槓"
            result["from_who_relative"] = -1 # 自分自身からのアクション
            # 「m >> 9: t」「t /= 3: 牌の種類インデックス」
            # 「(m >> 5) & 3: added_offset (0..3 加えた牌)」
            t = m >> 9
            t //= 3
            if not (0 <= t <= 33):
                print(f"[Warn decode_naki Kakan] Invalid tile index {t}. m={m}")
                return result

            base_id = t * 4
            added_offset = (m >> 5) & 3
            added_tile_id = base_id + added_offset

            # 加槓の場合、元のポン面子+追加牌 = 4枚
            # 元のポン面子は m からは特定できないが、通常は同種4枚
            result["tiles"] = sorted([base_id, base_id + 1, base_id + 2, base_id + 3])
            # 消費されたのは加槓で追加した牌1枚
            result["consumed"] = [added_tile_id]

        # --- 大明槓 or 暗槓 ---
        else: # ビット2,3,4が0
            # 「m >> 8: tile_id」 (Tenhou docsより)
            # 「tile_id / 4: 牌の種類インデックス」
            # 「tile_id % 4: 牌のオフセット？」-> このオフセットの意味は？ 通常の槓なら不要のはず
            tile_id_raw = m >> 8
            tile_index = tile_id_raw // 4
            # offset = tile_id_raw % 4 # What does this offset mean for kan?

            if not (0 <= tile_index <= 33):
                 print(f"[Warn decode_naki Kan] Invalid tile_index {tile_index} from m={m}")
                 return result

            base_id = tile_index * 4
            result["tiles"] = sorted([base_id, base_id + 1, base_id + 2, base_id + 3])

            if from_who_relative != -1 and from_who_relative != 3 : # 大明槓 (下家/対面/上家から)
                result["type"] = "大明槓"
                # 消費されたのは手持ちの3枚 -> mからだけでは特定できない。
                # 呼び出し元で called_tile を特定し、それ以外の3枚を consumed とする。
                # result["consumed"] is left empty here.
            else: # 暗槓 (自分自身)
                result["type"] = "暗槓"
                result["from_who_relative"] = -1 # 自分自身
                # 消費されたのは手牌の4枚
                result["consumed"] = result["tiles"] # Assume all 4 from hand for Ankan

        return result

    except Exception as e:
        print(f"[Error] Exception during decode_naki(m={m}): {e}")
        traceback.print_exc()
        result["type"] = "不明" # Ensure type is '不明' on error
        return result