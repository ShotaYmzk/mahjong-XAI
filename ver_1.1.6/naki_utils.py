# naki_utils.py (decode_naki修正版)
# (Keep the existing code as provided in the prompt)
try:
    # Ensure tile_utils is accessible
    from tile_utils import tile_id_to_index, tile_id_to_string
except ImportError:
    print("[Error] Cannot import from tile_utils.py in naki_utils.")
    # Define fallbacks ONLY IF ABSOLUTELY necessary for standalone testing
    def tile_id_to_index(tile: int) -> int: return tile // 4 if tile != -1 else -1
    def tile_id_to_string(tile: int) -> str: return f"tile_{tile}" if tile != -1 else "?"

def tile_to_name(tile: int) -> str:
    """牌IDを短い文字列（例: 1m, E）に変換"""
    # Use the robust version from tile_utils
    return tile_id_to_string(tile)

def decode_naki(m: int) -> tuple:
    """
    天鳳の副露面子のビットフィールドをデコードする関数 (参考URLに基づき修正)
    Args:
        m (int): <N>タグの m 属性の値
    Returns:
        tuple: (鳴きの種類, 構成牌IDリスト, 鳴かれた牌ID(推定), 誰から鳴いたか(常に-1))
               鳴きの種類: "チー", "ポン", "大明槓", "加槓", "暗槓", "不明"
               構成牌IDリスト: 面子を構成する牌IDのリスト (赤ドラも考慮)
               鳴かれた牌ID: ポン、チー、大明槓のトリガーとなった牌ID (推定)
               誰から: GameState側で判断するため常に -1
    """
    kui = m & 3 # ビット0, 1: 誰から鳴いたか (チーの場合は常に0?)
    from_player_rel = -1 # GameState側で判断するため、ここでは特定しない
    trigger_pai_id = -1 # 不明で初期化

    try:
        # --- チー ---
        if m & (1 << 2):
            naki_type = "チー"
            t = (m & 0xFC00) >> 10
            r = t % 3 # 鳴いた牌が順子のどこにあるか (0:左, 1:中, 2:右)
            t //= 3
            base_tile_index = (t // 7) * 9 + (t % 7) # 順子最小牌の種類インデックス (0-33)
            if base_tile_index < 0 or base_tile_index > 24: # Check valid range for lowest tile in sequence
                 print(f"[Error decode_naki Chi] Invalid base_tile_index {base_tile_index} from m={m}")
                 return "不明", [], -1, -1
            indices = [base_tile_index + 0, base_tile_index + 1, base_tile_index + 2]

            # 各牌のオフセット (0-3) を取得 -> 牌IDを計算
            offsets = [(m & 0x0018) >> 3, (m & 0x0060) >> 5, (m & 0x0180) >> 7]
            # ★★★ 構成牌IDリストを作成 ★★★
            tiles = []
            valid = True
            for idx, offset in zip(indices, offsets):
                tile_id = idx * 4 + offset
                if not (0 <= tile_id <= 135):
                    print(f"[Error decode_naki Chi] Invalid tile_id {tile_id} calculated for index {idx}, offset {offset}, m={m}")
                    valid = False; break
                tiles.append(tile_id)
            if not valid: return "不明", [], -1, -1

            # ★★★ トリガー牌IDを特定 ★★★
            # tilesリストはデコードされたID牌構成、r は鳴かれた牌がこのリストのどの位置(0,1,2)にあるかを示す
            if r < 0 or r >= len(tiles):
                 print(f"[Error decode_naki Chi] Invalid relative index r={r} for tiles list, m={m}")
                 return "不明", [], -1, -1
            trigger_pai_id = tiles[r] # This assumes the order matches 'r' before sorting

            # tiles リスト自体は、最終的に面子として表示・利用しやすいようにソートしておく
            tiles.sort()

        # --- ポン ---
        elif m & (1 << 3):
            naki_type = "ポン"
            t = (m & 0xFE00) >> 9
            # r = t % 3 # 鳴いた牌の相対位置? -> GameState側で判断するので不要
            t //= 3 # 牌の種類インデックス
            if t < 0 or t > 33:
                print(f"[Error decode_naki Pon] Invalid tile index {t} from m={m}")
                return "不明", [], -1, -1

            # m からだけでは構成牌の具体的なID(赤ドラ区別)を完全特定するのは困難
            # GameState側でトリガー牌と手牌から確定させる必要がある
            # ここでは、考えられるID候補を返すか、GameStateに処理を委ねる前提で種類インデックス基準のIDを返す
            base_id = t * 4
            # GameState側で処理しやすいように、どのIDが使用されたかのビット情報も渡せると良いが、
            # 複雑なので、ここでは基本IDのリストを返すに留める。
            # GameState.process_nakiで手牌と照合して確定させるロジックが必要
            unused_mask_bits = (m >> 3) & 0x1F # ビット 3-7 あたりが関係？ (要検証: https://gimite.net/pukiwiki/index.php?Tenhou%20Log%20Format の解説が複雑)
            # -> GameState側で処理する方針とし、ここでは基本形を返す
            tiles = [base_id, base_id+1, base_id+2, base_id+3] # GameState needs to pick 3

            # トリガー牌IDはGameState側で特定
            trigger_pai_id = -1

        # --- 加槓 ---
        elif m & (1 << 4):
            naki_type = "加槓"
            # 構成牌はGameState側で既存のポンと手牌から特定する
            # mからどの牌が加槓されたかを特定する (加槓される牌=手牌から出す牌)
            # -> ポンと同様に m だけから特定するのは困難な場合あり -> GameState側で処理
            t = (m & 0xFE00) >> 9 # ポン情報に似ている
            t //= 3 # 牌の種類インデックス
            if t < 0 or t > 33:
                print(f"[Error decode_naki Kakan] Invalid tile index {t} from m={m}")
                return "不明", [], -1, -1
            base_id = t * 4
            # GameState needs the tile_index 't' to find the matching pon and the tile from hand
            tiles = [base_id] * 4 # Placeholder, GameState will determine actual 4 tiles
            trigger_pai_id = -1 # 自分自身の牌なのでトリガーは無いが、どの牌かはGameStateで特定

        # --- 大明槓 or 暗槓 ---
        else: # ビット4, 5が0
            kui = m & 3 # 再度取得
            naki_type = "暗槓" if kui == 0 else "大明槓"
            t = (m & 0xFF00) >> 8 # 特殊エンコード (牌ID / 4 ?) -> 牌の種類インデックス
            t //= 4 # 牌の種類インデックス
            if t < 0 or t > 33:
                print(f"[Error decode_naki Kan] Invalid tile index {t} from m={m}")
                return "不明", [], -1, -1
            base_id = t * 4
            # ポンと同様に、mだけでは赤ドラ区別が難しい
            # GameState側で処理する前提で基本ID候補を返す
            tiles = [base_id, base_id+1, base_id+2, base_id+3] # GameState needs to pick 4 (Ankan) or determine trigger (Daiminkan)

            # トリガー牌IDはGameState側で特定 (暗槓はなし)
            trigger_pai_id = -1 # GameState側で特定 (暗槓の場合はそのまま-1)

        # 誰からの情報は常に -1 を返す (GameState側でlast_discard_eventから判断)
        return naki_type, tiles, trigger_pai_id, -1

    except Exception as e:
        print(f"[Error] Failed to decode naki m={m}: {e}")
        import traceback
        traceback.print_exc() # Log the full error traceback
        return "不明", [], -1, -1

# --- 使用例 (m=27031のチー) ---
if __name__ == "__main__":
    # Example values from https://gimite.net/pukiwiki/index.php?Tenhou%20Log%20Format
    test_cases = {
        # チー
        27031: "P3(who=3) が P2(上家, from=2) から 4p(ID=55) をチーして 2p(48)3p(52)4p(55) / m=11010111110011", # 手牌: 48, 52 / 鳴き牌: 55 / r=2(4pの位置) / base=12 / offsets=0,0,3
        8653: "P0(who=0) が P3(上家, from=3) から 6s(ID=91) をチーして 6s(91)7s(92)8s(96) / m=01000011101101", # 手牌: 92, 96 / 鳴き牌: 91 / r=0(6sの位置) / base=21 / offsets=3,0,0
        # ポン
        17449: "P1(who=1) が P0(下家, from=0) から 2m(ID=4) をポン / m=10001000101001", # 手牌: 4, 4 / 鳴き牌: 4 / t=1 / r=0?
        4137: "P1(who=1) が P3(上家, from=3) から 中(ID=132) をポン / m=00100000011001", # 手牌: 132, 132 / 鳴き牌: 132 / t=33 / r=1?
        # 大明槓
        38944: "P0(who=0) が P2(対面, from=2) から 8s(ID=96) を大明槓 / m=10011000000000", # 手牌: 96,96,96 / 鳴き牌: 96 / t=24 / kui=2
        # 加槓
        4124: "P0(who=0) が ポンした發(ID=128) に 加槓 / m=00010000001100", # 手牌: 128 / ポン: 128,128,128 / t=32
        # 暗槓
        49152: "P0(who=0) が 北(ID=120) を暗槓 / m=11000000000000", # 手牌: 120,120,120,120 / t=30 / kui=0
    }

    for m, desc in test_cases.items():
        print(f"\n--- Decoding m={m} ({desc}) ---")
        naki_type, tiles, trigger_pai_id, from_player_rel = decode_naki(m)
        print(f"  Naki Type: {naki_type}")
        print(f"  Tiles (Candidate/Decoded): {[tile_id_to_string(t) for t in tiles]}")
        print(f"  Trigger Pai ID (Decoded/Est): {tile_id_to_string(trigger_pai_id) if trigger_pai_id != -1 else 'N/A (GameState特定)'}")
        print(f"  From Player Rel (Always -1): {from_player_rel}")