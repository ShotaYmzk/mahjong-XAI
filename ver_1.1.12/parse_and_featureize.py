import xml.etree.ElementTree as ET
import urllib.parse
from typing import List, Dict, Any, Tuple
import os # osモジュールを追加

# --- naki_utils.py のインポート ---
# このスクリプトと同じディレクトリに naki_utils.py があると仮定
try:
    # tile_utils.py が naki_utils から使われている場合、
    # tile_utils.py も同じディレクトリにあるか、Pythonパスが通っている必要がある
    print(f"Current working directory: {os.getcwd()}")
    print(f"Attempting to import from: {os.path.join(os.getcwd(), 'naki_utils.py')}")
    from naki_utils import decode_naki, tile_to_name # tile_to_name も使うと便利
    print("Successfully imported decode_naki and tile_to_name from naki_utils.")
    # tile_utils が必要でインポートエラーになる場合は、
    # naki_utils.py 内の import tile_utils 周辺を確認してください。
except ImportError as e:
    print(f"Error importing from naki_utils: {e}")
    print("Please ensure naki_utils.py (and potentially tile_utils.py if needed by naki_utils)")
    print("is in the same directory as this script or in the Python path.")
    # フォールバック関数 (基本的な動作確認用、精度は保証されません)
    def decode_naki(m: int) -> tuple:
        print(f"[Warning] Using fallback decode_naki for m={m}")
        return ("不明", [], -1, -1)
    def tile_to_name(tile: int) -> str:
        # print(f"[Warning] Using fallback tile_to_name for tile={tile}")
        # 簡単な牌表現（例）
        if tile < 0: return "?"
        if 0 <= tile < 36: # 萬子 1-9m (ID 0-35)
            num = tile // 4 + 1
            return f"{num}m"
        elif 36 <= tile < 72: # 筒子 1-9p (ID 36-71)
            num = (tile - 36) // 4 + 1
            return f"{num}p"
        elif 72 <= tile < 108: # 索子 1-9s (ID 72-107)
            num = (tile - 72) // 4 + 1
            return f"{num}s"
        elif 108 <= tile < 136: # 字牌 E S W N Wh Gr R (ID 108-135)
            jihai = ["E", "S", "W", "N", "Wh", "Gr", "R"][(tile - 108) // 4]
            return jihai
        return f"t{tile}" # 不明な牌

# --- 牌IDから牌の種類インデックス(0-33)への変換 ---
# 赤ドラを通常牌として扱うか、別の特徴量にするかは設計によります
# ここでは赤ドラも通常牌のインデックスにマッピングする例
def tile_id_to_kind_index(tile_id: int) -> int:
    """牌ID(0-135)を種類インデックス(0-33)に変換する"""
    if tile_id < 0 or tile_id > 135: return -1
    # 赤ドラ (5m:16, 5p:52, 5s:88) を通常牌のIDに変換してから計算
    if tile_id == 16: tile_id = 20 # 5m (ID 20-23) の先頭へ
    elif tile_id == 52: tile_id = 56 # 5p (ID 56-59) の先頭へ
    elif tile_id == 88: tile_id = 92 # 5s (ID 92-95) の先頭へ
    # tile_id // 4 で種類インデックスが得られる
    # 0-8: 1m-9m, 9-17: 1p-9p, 18-26: 1s-9s, 27-33: ESWNWhGrR
    return tile_id // 4

def parse_tenhou_log(xml_file_path: str) -> List[Dict[str, Any]]:
    """
    天鳳XMLログファイルをパースし、局ごとのアクションと状態を抽出する関数
    """
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file_path}: {e}")
        return []
    except FileNotFoundError:
        print(f"Error: XML file not found at {xml_file_path}")
        return []

    all_kyoku_data = [] # 全ての局データを格納するリスト
    game_info = {'players': {}, 'rule': {}}

    # --- グローバル情報 (対局者、ルール) ---
    player_name_map = {}
    for tag in root:
        if tag.tag == 'UN':
            dans = tag.get('dan', '').split(',')
            rates = tag.get('rate', '').split(',')
            sexes = tag.get('sx', '').split(',')
            for i in range(4):
                player_id = f'p{i}'
                player_name_encoded = tag.get(f'n{i}', '')
                try:
                    player_name = urllib.parse.unquote(player_name_encoded)
                except Exception as e:
                    print(f"Warning: Could not decode player name {player_name_encoded}: {e}")
                    player_name = f"player_{i}_undecoded"
                player_name_map[i] = player_name # マッピングを保存
                game_info['players'][player_id] = {
                    'name': player_name,
                    'dan': dans[i] if i < len(dans) else 'N/A',
                    'rate': rates[i] if i < len(rates) else 'N/A',
                    'sx': sexes[i] if i < len(sexes) else 'N/A'
                }
        elif tag.tag == 'GO':
            game_info['rule']['type'] = tag.get('type')
            game_info['rule']['lobby'] = tag.get('lobby')
        elif tag.tag == 'TAIKYOKU':
            # --- 対局ログ処理 ---
            current_kyoku_data = None
            player_states = [{} for _ in range(4)] # 各プレイヤーの状態

            for element in tag:
                # --- 局開始 ---
                if element.tag == 'INIT':
                    if current_kyoku_data: # 前の局のデータがあればリストに追加
                        all_kyoku_data.append(current_kyoku_data)

                    seed_info = element.get('seed', '0,0,0,0,0,0').split(',') # デフォルト値設定
                    try:
                        kyoku_num = int(seed_info[0])
                        honba = int(seed_info[1])
                        riichi_bou = int(seed_info[2])
                        dice1 = int(seed_info[3]) + 1 # 0-5 -> 1-6
                        dice2 = int(seed_info[4]) + 1 # 0-5 -> 1-6
                        dora_indicator = int(seed_info[5])
                    except (ValueError, IndexError):
                        print(f"Warning: Invalid seed format in INIT tag: {element.get('seed')}")
                        kyoku_num, honba, riichi_bou, dice1, dice2, dora_indicator = -1, -1, -1, -1, -1, -1

                    try:
                        start_scores = [int(s) * 100 for s in element.get('ten', '0,0,0,0').split(',')]
                    except ValueError:
                        print(f"Warning: Invalid ten format in INIT tag: {element.get('ten')}")
                        start_scores = [25000] * 4 # デフォルト値

                    oya = int(element.get('oya', '-1'))

                    current_kyoku_data = {
                        'kyoku': kyoku_num,
                        'honba': honba,
                        'riichi_bou': riichi_bou,
                        'dice': (dice1, dice2),
                        'dora_indicators': [dora_indicator] if dora_indicator != -1 else [], # ドラ表示牌リスト
                        'start_scores': start_scores,
                        'oya': oya,
                        'actions': [], # この局のアクションシーケンス
                        'result': None,
                        'player_names': [player_name_map.get(i, f'p{i}') for i in range(4)] # 名前を追加
                    }

                    # 各プレイヤーの初期状態設定
                    for i in range(4):
                        hai_str = element.get(f'hai{i}', '')
                        try:
                            # カンマ区切りで空文字列を除外し、整数に変換
                            hand = sorted([int(h) for h in hai_str.split(',') if h])
                        except ValueError:
                            print(f"Warning: Invalid hai{i} format in INIT tag: {hai_str}")
                            hand = []
                        player_states[i] = {
                            'id': i,
                            'name': player_name_map.get(i, f'p{i}'),
                            'score': start_scores[i],
                            'hand': hand,
                            'kawa': [], # 捨て牌 (河)
                            'naki': [], # 副露面子リスト
                            'is_riichi': False,
                            'riichi_jun': -1, # リーチ宣言巡目 (打牌基準)
                            'jun': 0 # 現在の巡目 (打牌基準)
                        }
                    # 最初の状態をアクションとして記録（必要なら）
                    # current_kyoku_data['actions'].append({'type': 'INIT', 'player_states': [p.copy() for p in player_states]})

                # --- アクション処理 (INITの後) ---
                elif current_kyoku_data:
                    action_info = {'type': element.tag, 'tag_attributes': element.attrib}
                    player_index = -1

                    # --- ツモ --- (T0, U64, V100, W135 など)
                    if element.tag[0] in ('T', 'U', 'V', 'W') and element.tag[1:].isdigit():
                        player_index = {'T': 0, 'U': 1, 'V': 2, 'W': 3}[element.tag[0]]
                        drawn_tile = int(element.tag[1:])
                        # 手牌に追加する前に、現在の手牌とアクションを記録 (予測のため)
                        action_info['player'] = player_index
                        action_info['tile'] = drawn_tile
                        action_info['player_states_before'] = [p.copy() for p in player_states]
                        current_kyoku_data['actions'].append(action_info)

                        # 状態更新
                        player_states[player_index]['hand'].append(drawn_tile)
                        player_states[player_index]['hand'].sort()


                    # --- 打牌 --- (D0, E64, F100, G135 など)
                    elif element.tag[0] in ('D', 'E', 'F', 'G') and element.tag[1:].isdigit():
                        player_index = {'D': 0, 'E': 1, 'F': 2, 'G': 3}[element.tag[0]]
                        discarded_tile = int(element.tag[1:])

                        # 手牌から削除する前に、現在の手牌とアクションを記録
                        action_info['player'] = player_index
                        action_info['tile'] = discarded_tile
                        action_info['player_states_before'] = [p.copy() for p in player_states]
                        current_kyoku_data['actions'].append(action_info)

                        # 状態更新
                        try:
                            player_states[player_index]['hand'].remove(discarded_tile)
                        except ValueError:
                            # 加槓した牌を切る場合や、何らかのログの不整合？
                            print(f"Warning: Kyoku {current_kyoku_data['kyoku']}-{current_kyoku_data['honba']}, Player {player_index}: Tried to discard tile {tile_to_name(discarded_tile)} ({discarded_tile}) not in hand. Hand: {[tile_to_name(t) for t in player_states[player_index]['hand']]}")
                            # pass # 無視して続行するか、エラー処理するか
                        player_states[player_index]['kawa'].append(discarded_tile)
                        player_states[player_index]['jun'] += 1 # 打牌で巡目を進める


                    # --- 鳴き ---
                    elif element.tag == 'N':
                        player_index = int(element.get('who', '-1'))
                        m_code = int(element.get('m', '0'))
                        if player_index != -1 and m_code != 0:
                            # 鳴きアクションが発生した時点の状態を記録
                            action_info['player'] = player_index
                            action_info['m_code'] = m_code
                            action_info['player_states_before'] = [p.copy() for p in player_states]

                            # decode_naki を呼び出す
                            naki_type, naki_tiles, called_tile, _ = decode_naki(m_code) # from_whoは使わない

                            # 誰から鳴いたか (from_who) を推定 (直前の打牌プレイヤー)
                            from_who = -1
                            if current_kyoku_data['actions']:
                                # actionsリストを逆順に見て、最後の打牌アクションを探す
                                for prev_action in reversed(current_kyoku_data['actions']):
                                     # Check if the tag starts with D, E, F, or G and the rest is a digit
                                     if prev_action['type'][0] in ('D', 'E', 'F', 'G') and prev_action['type'][1:].isdigit():
                                         from_who = prev_action['player']
                                         # Check if the called tile matches the discarded tile
                                         if prev_action.get('tile') == called_tile:
                                             break # Found the correct discard action
                                         else:
                                             # This case might happen (e.g., multiple discards before N), log a warning if necessary
                                             # print(f"Warning: Naki called tile {called_tile} doesn't match last discard {prev_action.get('tile')}")
                                             # Keep the from_who from the last discarder for now
                                             break
                            action_info['from_who'] = from_who
                            action_info['naki_type'] = naki_type
                            action_info['naki_tiles'] = naki_tiles # デコードされた牌リスト
                            action_info['called_tile'] = called_tile # 鳴かれた牌

                            current_kyoku_data['actions'].append(action_info) # アクションを記録

                            # 状態更新 (手牌から鳴きに使用した牌を削除)
                            naki_info_for_state = {
                                'type': naki_type,
                                'tiles': naki_tiles,
                                'from_who': from_who,
                                'called_tile': called_tile,
                                'm': m_code,
                                'jun': player_states[from_who]['jun'] if from_who !=-1 else -1 # 鳴かせた人の巡目
                            }
                            player_states[player_index]['naki'].append(naki_info_for_state)

                            # 手牌から削除する牌を決定
                            tiles_to_remove = []
                            if naki_type in ("チー", "ポン", "大明槓"):
                                # 鳴かれた牌(called_tile)以外を手牌から除く
                                tiles_to_remove = [t for t in naki_tiles if t != called_tile]
                            elif naki_type == "加槓":
                                # ポンしている牌に1枚加えるので、加える牌(called_tile)を手牌から除く
                                tiles_to_remove = [called_tile]
                                # 加槓の場合、既存のポンの面子を探して更新する必要があるかもしれない
                                # ここでは単純に新しいnakiとして追加（要検討）
                            elif naki_type == "暗槓":
                                # 手牌から4枚除く
                                tiles_to_remove = [called_tile] * 4

                            original_hand = player_states[player_index]['hand'].copy()
                            for tile in tiles_to_remove:
                                try:
                                    player_states[player_index]['hand'].remove(tile)
                                except ValueError:
                                    print(f"Warning: Kyoku {current_kyoku_data['kyoku']}-{current_kyoku_data['honba']}, Player {player_index}: Tried to remove tile {tile_to_name(tile)} ({tile}) for naki '{naki_type}' not in hand.")
                                    print(f"  Naki tiles: {[tile_to_name(t) for t in naki_tiles]}, Called: {tile_to_name(called_tile)}")
                                    print(f"  Original Hand: {[tile_to_name(t) for t in original_hand]}")
                                    # pass

                            player_states[player_index]['hand'].sort()


                    # --- リーチ ---
                    elif element.tag == 'REACH':
                        player_index = int(element.get('who', '-1'))
                        step = int(element.get('step', '0'))
                        if player_index != -1:
                            # リーチアクションが発生した時点の状態を記録
                            action_info['player'] = player_index
                            action_info['step'] = step
                            action_info['player_states_before'] = [p.copy() for p in player_states]
                            current_kyoku_data['actions'].append(action_info)

                            # 状態更新
                            if step == 1: # リーチ宣言 (打牌の直前)
                                player_states[player_index]['is_riichi'] = True
                                # リーチ巡目は、このリーチ宣言の *後* の打牌で確定する巡目
                                # player_states[player_index]['riichi_jun'] = player_states[player_index]['jun'] + 1
                            # step 2 はリーチ成立 (供託が増える) -> 点数移動はAGARI/RYUUKYOKUで処理


                    # --- ドラ (槓ドラ) ---
                    elif element.tag == 'DORA':
                        new_dora_indicator = int(element.get('hai', '-1'))
                        # ドラアクションが発生した時点の状態を記録
                        action_info['new_dora_indicator'] = new_dora_indicator
                        action_info['player_states_before'] = [p.copy() for p in player_states]
                        current_kyoku_data['actions'].append(action_info)

                        # 状態更新
                        if new_dora_indicator != -1:
                            current_kyoku_data['dora_indicators'].append(new_dora_indicator)


                    # --- 和了 / 流局 ---
                    elif element.tag in ('AGARI', 'RYUUKYOKU'):
                        # 局の終了アクションとして記録
                        action_info['player_states_before'] = [p.copy() for p in player_states] # 終了直前の状態
                        current_kyoku_data['actions'].append(action_info)

                        # 局の結果をresultに格納
                        current_kyoku_data['result'] = {
                            'type': element.tag,
                            'details': element.attrib
                        }
                        # 点数変動をplayer_statesに反映 (sc属性をパース)
                        sc_info = element.get('sc', '').split(',')
                        if len(sc_info) == 8:
                            try:
                                score_changes = [int(float(sc_info[i*2+1])) * 100 for i in range(4)] # 収支
                                final_scores = [int(float(sc_info[i*2])) * 100 for i in range(4)] # 最終持ち点
                                for i in range(4):
                                    player_states[i]['score'] = final_scores[i]
                                # print(f"Debug: Score changes: {score_changes}, Final scores: {final_scores}") # デバッグ用
                            except ValueError:
                                print(f"Warning: Invalid sc format in {element.tag}: {element.get('sc')}")

                        # owari属性があればゲーム終了情報も格納できる
                        if 'owari' in element.attrib:
                           current_kyoku_data['game_end_info'] = element.get('owari')

                        # 次の局のためにループの先頭に戻る (continueは不要)


                    # --- その他のタグ ---
                    # 例: <BYE who="0"/> など、無視するか処理するかを決定
                    else:
                         # 未知のタグや処理不要なタグの場合
                         action_info['player_states_before'] = [p.copy() for p in player_states]
                         current_kyoku_data['actions'].append(action_info)


            # ループ終了後、最後の局データをリストに追加
            if current_kyoku_data:
                all_kyoku_data.append(current_kyoku_data)

    return all_kyoku_data


def create_feature_vectors(parsed_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    パースされたデータから、各アクション時点での特徴量ベクトルを作成する（簡単な例）
    研究計画に合わせて拡張が必要です。
    """
    feature_sequences = []
    num_tile_types = 34 # 牌の種類数 (萬子1-9, 筒子1-9, 索子1-9, 字牌1-7)

    for kyoku_idx, kyoku_data in enumerate(parsed_data):
        kyoku_features = {
            'kyoku_info': { # アクションリスト以外の局情報をコピー
                k: v for k, v in kyoku_data.items() if k != 'actions'
            },
            'sequence': [] # この局の特徴量シーケンス
        }

        # ドラ表示牌の「種類」インデックスリスト
        current_dora_kind_indices = [tile_id_to_kind_index(d) for d in kyoku_data.get('dora_indicators', [])]

        for action_idx, action in enumerate(kyoku_data['actions']):
            # アクション前の状態を取得
            if 'player_states_before' not in action:
                # print(f"Skipping action without 'player_states_before': {action.get('type')}")
                continue # INITなど、状態が記録されていないアクションはスキップ

            states_before = action['player_states_before']
            player_index = action.get('player', -1) # アクションを行ったプレイヤー

            # --- 特徴量ベクトルの作成 ---
            feature_vector = {}

            # 1. 局の基本情報
            feature_vector['kyoku'] = kyoku_data['kyoku']
            feature_vector['honba'] = kyoku_data['honba']
            feature_vector['riichi_bou'] = kyoku_data['riichi_bou']
            feature_vector['oya'] = kyoku_data['oya']
            # ドラ表示牌（種類インデックス）
            feature_vector['dora_indicators'] = current_dora_kind_indices.copy()
            # feature_vector['num_doras'] = len(current_dora_kind_indices) # ドラの枚数も特徴量に

            # 2. 各プレイヤーの状態 (アクション前の状態)
            for i in range(4):
                p_state = states_before[i]
                p_features = {}

                # 手牌: 34種の牌の枚数カウントベクトル
                hand_counts = [0] * num_tile_types
                for tile in p_state['hand']:
                    kind_idx = tile_id_to_kind_index(tile)
                    if 0 <= kind_idx < num_tile_types:
                        hand_counts[kind_idx] += 1
                p_features['hand_counts'] = hand_counts
                # p_features['hand_len'] = len(p_state['hand']) # 手牌枚数

                # 河 (捨て牌): 捨てられた牌の種類インデックスのリスト
                p_features['kawa'] = [tile_id_to_kind_index(t) for t in p_state['kawa']]
                # p_features['kawa_len'] = len(p_state['kawa'])
                p_features['jun'] = p_state['jun'] # 現在の巡目

                # 鳴き面子: (種類, [構成牌種インデックス], 誰から, 巡目) のリスト
                p_features['naki'] = []
                for naki in p_state['naki']:
                   p_features['naki'].append({
                       'type': naki['type'],
                       'tiles': sorted([tile_id_to_kind_index(t) for t in naki['tiles']]),
                       'from_who': naki['from_who'],
                       'jun': naki['jun']
                   })
                # p_features['num_naki'] = len(p_state['naki'])

                p_features['is_riichi'] = p_state['is_riichi']
                # p_features['riichi_jun'] = p_state['riichi_jun']
                p_features['score'] = p_state['score']

                feature_vector[f'player_{i}'] = p_features

            # 3. 実行されたアクションの情報 (これが直後の状態を引き起こした)
            feature_vector['action_type'] = action['type']
            feature_vector['action_player'] = player_index
            if 'tile' in action: # ツモ or 打牌
                 feature_vector['action_tile'] = tile_id_to_kind_index(action['tile'])
            elif 'naki_type' in action: # 鳴き
                 feature_vector['action_naki_type'] = action['naki_type']
                 feature_vector['action_naki_tiles'] = sorted([tile_id_to_kind_index(t) for t in action.get('naki_tiles', [])])
                 feature_vector['action_naki_from_who'] = action.get('from_who', -1)
            elif 'step' in action: # リーチ
                 feature_vector['action_riichi_step'] = action['step']
            elif 'new_dora_indicator' in action: # ドラ追加
                 feature_vector['action_new_dora'] = tile_id_to_kind_index(action['new_dora_indicator'])

            # --- 目的変数 (例: この状態でプレイヤーが行う打牌予測) ---
            # 次のアクションが打牌の場合、その打牌を target とする例
            target_discard = -1 # デフォルト値
            if action_idx + 1 < len(kyoku_data['actions']):
                next_action = kyoku_data['actions'][action_idx + 1]
                # 次のアクションが、現在のアクションを行ったプレイヤーの打牌の場合
                if next_action.get('player') == player_index and \
                   next_action['type'][0] in ('D','E','F','G') and \
                   next_action['type'][1:].isdigit():
                    target_discard = tile_id_to_kind_index(next_action.get('tile', -1))

            feature_vector['target_discard'] = target_discard # 特徴量ベクトルに追加

            # 作成した特徴量ベクトルをシーケンスに追加
            kyoku_features['sequence'].append(feature_vector)

            # 状態更新（ドラ）: actionループの中でドラが追加された場合、次の特徴量ベクトルで使うために更新
            if action.get('type') == 'DORA' and 'new_dora_indicator' in action:
                new_dora_idx = tile_id_to_kind_index(action['new_dora_indicator'])
                if new_dora_idx != -1:
                    current_dora_kind_indices.append(new_dora_idx)


        # 1局分の特徴量シーケンスが完成したらリストに追加
        if kyoku_features['sequence']: # 空でない場合のみ追加
            feature_sequences.append(kyoku_features)

    return feature_sequences


# --- メイン処理 ---
if __name__ == "__main__":
    xml_log_file = 'test_log.xml' # 処理対象のXMLファイル

    print(f"Parsing log file: {xml_log_file}...")
    parsed_game_data = parse_tenhou_log(xml_log_file)

    if parsed_game_data:
        print(f"Successfully parsed {len(parsed_game_data)} kyoku.")

        # --- パース結果の確認 (任意) ---
        # 例: 最初の局の情報を表示
        # print("\n--- Example: First Kyoku Info ---")
        # print({k: v for k, v in parsed_game_data[0].items() if k != 'actions'})
        # 例: 最初の局の最初のアクション（ツモ）を表示
        # print("\n--- Example: First action (Tsumo) of first kyoku ---")
        # if parsed_game_data[0]['actions']:
        #     print(parsed_game_data[0]['actions'][0]) # player_states_before を含むか確認

        print("\nCreating feature vectors (example)...")
        feature_data = create_feature_vectors(parsed_game_data)
        print(f"Created feature sequences for {len(feature_data)} kyoku.")

        # --- 特徴量生成結果の確認 (任意) ---
        if feature_data and feature_data[0]['sequence']:
             print("\n--- Example: First feature vector of first kyoku sequence ---")
             fv = feature_data[0]['sequence'][0]
             print(f"Kyoku: {fv['kyoku']}-{fv['honba']}, Oya: {fv['oya']}, Riichi-bou: {fv['riichi_bou']}")
             print(f"Dora Indicators (Kind Index): {fv['dora_indicators']}")
             # Player 0 の状態
             print(f"Player 0 Score: {fv['player_0']['score']}")
             print(f"Player 0 Hand Counts: {fv['player_0']['hand_counts']}")
             print(f"Player 0 Kawa (Kind Index): {fv['player_0']['kawa']}")
             print(f"Player 0 Naki: {fv['player_0']['naki']}")
             print(f"Player 0 Riichi: {fv['player_0']['is_riichi']}")
             # アクション情報
             print(f"Action: {fv['action_type']} by Player {fv['action_player']}")
             if 'action_tile' in fv: print(f"  Tile: {fv['action_tile']}")
             # 目的変数（例）
             print(f"Target discard (next action by this player): {fv['target_discard']}")


        # --- ここから先のステップ ---
        print("\n--- Next Steps ---")
        print("1. [Feature Engineering] Enhance feature vectors:")
        print("   - Hand representation: One-hot (34x4=136 features?), embeddings?")
        print("   - Discard info: Include other players' discards, order, counts.")
        print("   - Naki details: Represent naki more precisely (e.g., which tiles exposed).")
        print("   - Game state: Relative scores, remaining tiles, player styles (embeddings).")
        print("   - Shanten calculation: Integrate a library to add shanten count as a feature.")
        print("2. [Tokenization] If using pure Transformer:")
        print("   - Define a vocabulary for actions (Draw, Discard, Chi, Pon, Kan, Riichi) and tiles (0-135 or 0-33).")
        print("   - Convert action sequences into sequences of token IDs.")
        print("3. [Graph Construction] If using Graph Attention:")
        print("   - Define nodes (tiles in hand, discards, naki, dora, player state).")
        print("   - Define edges (tile relations, player-hand, discard order, etc.).")
        print("   - Generate adjacency matrices or edge lists for each state.")
        print("4. [Dataset Preparation] Format data for your model:")
        print("   - Pad sequences to a fixed length.")
        print("   - Create attention masks.")
        print("   - Split into training, validation, and test sets.")
        print("5. [Model Implementation & Training]:")
        print("   - Implement the Hierarchical Graph + Transformer architecture.")
        print("   - Train the model on the prepared dataset.")
        print("6. [Self-Supervised Pre-training]:")
        print("   - Implement tasks like Span Masking, Replaced Tile Detection based on the token sequences or feature vectors.")

    else:
        print("Failed to parse log file or no kyoku data found.")