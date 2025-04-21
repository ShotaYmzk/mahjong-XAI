# naki_utils.py (Revised decode_naki function)
try:
    from tile_utils import tile_id_to_index, tile_id_to_string
except ImportError:
    print("[Error] Cannot import from tile_utils.py in naki_utils.")
    # Define fallback functions if imports fail
    def tile_id_to_index(tile: int) -> int:
        if tile < 0:
            return -1
        if tile < 108:
            suit_index = tile // 36  # 0: 萬子, 1: 筒子, 2: 索子
            number_index = (tile % 36) // 4
            return suit_index * 9 + number_index
        else:
            return 27 + (tile - 108) // 4
            
    def tile_id_to_string(tile: int) -> str:
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

def tile_to_name(tile: int) -> str:
    return tile_id_to_string(tile)

def decode_naki(m: int, last_discard_tile_id: int = -1, from_player: int = -1, naki_player_id: int = -1) -> dict:
    """
    天鳳の副露面子のビットフィールドをデコードする関数 (詳細解析版)
    Args:
        m (int): <N>タグの m 属性の値
        last_discard_tile_id (int): 最後に捨てられた牌のID (オプション)
        from_player (int): 誰が捨てた牌か (オプション)
        naki_player_id (int): 誰が鳴いたか (オプション)
    Returns:
        dict: {
            "type": 鳴きの種類 ("チー", "ポン", "大明槓", "加槓", "暗槓", "不明"),
            "tiles": 構成牌IDリスト,
            "called_tile": 鳴かれた牌ID,
            "from_who": 誰から鳴いたか(相対, 0:下家, 1:対面, 2:上家, 3:自分)
        }
    """
    kui = m & 3 # 0:下家, 1:対面, 2:上家, 3:自分(槓)
    trigger_pai_id = -1
    tiles = []
    naki_type = "不明"

    try:
        # --- チー ---
        if m & (1 << 2):
            naki_type = "チー"
            # If kui != 0 for Chi, it's unexpected based on common understanding, but we keep the value.
            # kui = 0 # Force kui to 0 for Chi if strictly adhering to player perspective interpretation

            t = (m & 0xFC00) >> 10
            r = t % 3 # 鳴いた牌が元々どの位置にあったか (0:左, 1:中, 2:右)
            t //= 3
            base_tile_index = (t // 7) * 9 + (t % 7) # 順子最小牌の種類インデックス (0-24)
            if not (0 <= base_tile_index <= 24): # Check valid range for lowest tile in sequence
                print(f"[Error decode_naki Chi] Invalid base_tile_index {base_tile_index} from m={m}")
                return {"type": "不明", "tiles": [], "called_tile": -1, "from_who": kui}

            # Calculate the three tile IDs based on base index and offsets
            indices = [base_tile_index + 0, base_tile_index + 1, base_tile_index + 2]
            offsets = [(m & 0x0018) >> 3, (m & 0x0060) >> 5, (m & 0x0180) >> 7]
            temp_tiles = [] # Store tiles in original 0,1,2 order first
            valid = True
            for idx, offset in zip(indices, offsets):
                tile_id = idx * 4 + offset
                if not (0 <= tile_id <= 135):
                    print(f"[Error decode_naki Chi] Invalid tile_id {tile_id} calculated for index {idx}, offset {offset}, m={m}")
                    valid = False; break
                temp_tiles.append(tile_id)
            if not valid: return {"type": "不明", "tiles": [], "called_tile": -1, "from_who": kui}

            # The trigger tile is the one at relative position 'r' in the original sequence
            if not (0 <= r < 3):
                 print(f"[Error decode_naki Chi] Invalid relative index r={r} for tiles list, m={m}")
                 return {"type": "不明", "tiles": [], "called_tile": -1, "from_who": kui}
            trigger_pai_id = temp_tiles[r]

            # The final meld shown in game is sorted, with the called tile often placed differently.
            # However, for representing the meld itself, the sorted list of the 3 tiles is useful.
            tiles = sorted(temp_tiles)

        # --- ポン ---
        elif m & (1 << 3):
            naki_type = "ポン"
            unused_bit_idx = (m & 0x0060) >> 5 # ビット5,6: 使わなかった牌のインデックス (0-3) -> これはチーの間違い？ ポンの未使用牌特定へ修正
            # ポンの未使用牌特定: https://gimite.net/pukiwiki/index.php?Tenhou%20Log%20Format
            # ビット5,6 -> どの牌 ID を使わなかったかのマスク (0-3)
            # 0: ID=base+0を使わずbase+1,base+2,base+3を使用
            # 1: ID=base+1を使わずbase+0,base+2,base+3を使用
            # 2: ID=base+2を使わずbase+0,base+1,base+3を使用
            # 3: ID=base+3を使わずbase+0,base+1,base+2を使用
            # ※ mからだけではどの牌が鳴かれた牌かは特定できない

            t = (m & 0xFE00) >> 9
            r = t % 3 # ビット9,10: 鳴いた牌の相対位置 (0:左, 1:中, 2:右) -> これはチーでは？ ポンでは鳴いた牌は常に左に置かれる？ -> 要検証, kuiで誰からかは分かる
            t //= 3 # 牌の種類インデックス (0-33)
            if not (0 <= t <= 33):
                print(f"[Error decode_naki Pon] Invalid tile index {t} from m={m}")
                return {"type": "不明", "tiles": [], "called_tile": -1, "from_who": kui}

            base_id = t * 4
            possible_ids = [base_id + 0, base_id + 1, base_id + 2, base_id + 3]

            # unused_bit_idx を使って3枚の牌を特定
            unused_offset = (m & 0x0060) >> 5 # 再確認: ビット5,6 が未使用牌のインデックス
            tiles = []
            for i in range(4):
                if i != unused_offset:
                    tiles.append(possible_ids[i])
            if len(tiles) != 3:
                 print(f"[Error decode_naki Pon] Failed to select 3 tiles based on unused offset {unused_offset}. m={m}")
                 return {"type": "不明", "tiles": [], "called_tile": -1, "from_who": kui}

            # トリガー牌IDは、mからだけでは完全特定できない。
            # GameState側で、kuiから誰の捨て牌かを見て確定させる必要がある。
            # Use the last_discard_tile_id if provided
            trigger_pai_id = last_discard_tile_id if last_discard_tile_id != -1 else -1

        # --- 加槓 ---
        elif m & (1 << 4):
            naki_type = "加槓"
            # 加槓の場合、kui=3 になるはず？(自分自身からの鳴き)
            # if kui != 3: print(f"[Warn decode_naki Kakan] Expected kui=3 but got {kui}. m={m}") # Tenhou might use 0?

            # どの牌をポンに追加したか？
            added_offset = (m & 0x0060) >> 5 # ビット5,6: 加槓で手牌から追加した牌のオフセット (0-3)

            t = (m & 0xFE00) >> 9
            # r = t % 3 # 関係ないはず
            t //= 3 # 牌の種類インデックス (0-33)
            if not (0 <= t <= 33):
                print(f"[Error decode_naki Kakan] Invalid tile index {t} from m={m}")
                return {"type": "不明", "tiles": [], "called_tile": -1, "from_who": kui}

            base_id = t * 4
            added_pai_id = base_id + added_offset

            # 構成牌はGameStateで特定するのが確実だが、mから推測する
            # ポンしていた3枚を特定する必要がある
            # 面倒なので、ここでは基本形4枚を返す（GameState側での処理推奨）
            tiles = [base_id, base_id + 1, base_id + 2, base_id + 3] # Placeholder, GameState should determine the 4 tiles
            trigger_pai_id = -1 # No trigger from others, the added tile is from hand (GameState knows)

        # --- 大明槓 or 暗槓 ---
        else: # ビット2,3,4が0
            t = (m & 0xFF00) >> 8 # 牌の種類インデックス * 4 + オフセット？ -> 要検証
            # From https://gimite.net/pukiwiki/index.php?Tenhou%20Log%20Format
            # "m >> 8: tile / 4" -> This means t represents the tile index.
            tile_index = t // 4
            if not (0 <= tile_index <= 33):
                print(f"[Error decode_naki Kan] Invalid tile_index {tile_index} derived from t={t}, m={m}")
                return {"type": "不明", "tiles": [], "called_tile": -1, "from_who": kui}

            base_id = tile_index * 4
            # 構成牌4枚は基本的に同じ牌種
            tiles = [base_id, base_id + 1, base_id + 2, base_id + 3]

            if kui == 0: # 暗槓
                naki_type = "暗槓"
                trigger_pai_id = -1
                # 暗槓の場合、どの4枚かは手牌を見ないと確定できないが、通常は全て同じ種類。
            else: # 大明槓
                naki_type = "大明槓"
                # トリガー牌IDはGameState側で特定が必要
                trigger_pai_id = last_discard_tile_id if last_discard_tile_id != -1 else -1

        return {
            "type": naki_type,
            "tiles": sorted(tiles),
            "called_tile": trigger_pai_id,
            "from_who": kui
        }

    except Exception as e:
        print(f"[Error] Failed to decode naki m={m}: {e}")
        import traceback
        traceback.print_exc() # Log the full error traceback
        return {
            "type": "不明",
            "tiles": [],
            "called_tile": -1,
            "from_who": m & 3
        }