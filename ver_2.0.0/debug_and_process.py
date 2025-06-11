# /ver_2.0.0/debug_and_process.py
import sys
import argparse
import os
import traceback # 詳細なエラーレポート用
import numpy as np # 特徴量検査用
import xml.etree.ElementTree as ET
import urllib.parse
from typing import List, Dict, Any, Tuple
from collections import defaultdict, deque

# --- 依存モジュールのインポート ---
# このスクリプトは単体で動作するように、必要なクラスと関数を内部に含みます。
# naki_utils.pyとtile_utils.pyは同じディレクトリにあることを期待します。
try:
    from tile_utils import tile_id_to_index, tile_id_to_string, tile_index_to_id
    from naki_utils import decode_naki
    print("tile_utils と naki_utils から正常にインポートしました。")
except ImportError as e:
    print(f"[致命的エラー] tile_utils/naki_utilsからのインポートに失敗しました: {e}")
    print("tile_utils.py と naki_utils.py がこのスクリプトと同じディレクトリにあることを確認してください。")
    sys.exit(1)
# --- 依存モジュールのインポート終了 ---

# --- 定数 (game_state.pyと同期) ---
NUM_PLAYERS = 4
NUM_TILE_TYPES = 34
MAX_EVENT_HISTORY = 60
# ★★★ 新しい特徴量次元数に合わせて更新 ★★★
STATIC_FEATURE_DIM = 543 # 新しい特徴量次元

# イベントタイプと鳴きタイプのエンコーディング
EVENT_TYPES = {
    "INIT": 0, "TSUMO": 1, "DISCARD": 2, "N": 3, "REACH": 4,
    "DORA": 5, "AGARI": 6, "RYUUKYOKU": 7, "PADDING": 8
}
NAKI_TYPES = {"チー": 0, "ポン": 1, "大明槓": 2, "加槓": 3, "暗槓": 4, "不明": -1}
# --- 定数終了 ---

# ==============================================================================
# == XMLパーサー (full_mahjong_parser.py から統合) ==
# ==============================================================================
def parse_full_mahjong_log(xml_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """天鳳XMLログを解析し、メタ情報と局データのリストを返す。"""
    meta, rounds, player_name_map = {}, [], {}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[エラー] XMLファイルの解析に失敗しました: {xml_path} - {e}")
        return {}, []

    current_round_data = None
    for elem in root:
        tag, attrib = elem.tag, elem.attrib
        if tag == "GO":
            meta['go'] = attrib
        elif tag == "UN":
            meta['un'] = attrib
            for i in range(4):
                name_key = f'n{i}'
                if name_key in attrib:
                    player_name_map[i] = urllib.parse.unquote(attrib[name_key])
            meta['player_names'] = [player_name_map.get(i, f'p{i}') for i in range(4)]
        elif tag == "TAIKYOKU":
            meta['taikyoku'] = attrib
        elif tag == "INIT":
            current_round_data = {"round_index": len(rounds) + 1, "init": attrib, "events": [], "result": None}
            rounds.append(current_round_data)
        elif current_round_data is not None:
            event_data = {"tag": tag, "attrib": attrib}
            current_round_data["events"].append(event_data)
            if tag in ["AGARI", "RYUUKYOKU"]:
                current_round_data["result"] = event_data
    return meta, rounds

# ==============================================================================
# == ゲーム状態管理 (game_state.py から統合・改良) ==
# ==============================================================================
class GameState:
    """麻雀の1局の状態を管理し、イベントを処理して特徴量を生成するクラス。"""
    TSUMO_TAGS = {"T": 0, "U": 1, "V": 2, "W": 3}
    DISCARD_TAGS = {"D": 0, "E": 1, "F": 2, "G": 3}

    def __init__(self):
        self.reset_state()

    def reset_state(self):
        """全ての内部状態変数をリセットする。"""
        self.round_num_wind: int = 0
        self.honba: int = 0
        self.kyotaku: int = 0
        self.dealer: int = -1
        self.dora_indicators: List[int] = []
        self.current_scores: List[int] = [25000] * NUM_PLAYERS
        self.player_hands: List[List[int]] = [[] for _ in range(NUM_PLAYERS)]
        self.player_discards: List[List[Tuple[int, bool]]] = [[] for _ in range(NUM_PLAYERS)]
        self.player_melds: List[List[Dict]] = [[] for _ in range(NUM_PLAYERS)]
        self.player_reach_status: List[int] = [0] * NUM_PLAYERS # 0: No, 1: Declared, 2: Accepted
        self.player_reach_junme: List[float] = [-1.0] * NUM_PLAYERS
        self.player_reach_discard_tile: List[int] = [-1] * NUM_PLAYERS # ★リーチ宣言牌を記録
        self.current_player: int = -1
        self.junme: float = 0.0
        self.wall_tile_count: int = 70
        self.event_history: deque = deque(maxlen=MAX_EVENT_HISTORY)
        # 内部追跡用
        self._last_discard_player: int = -1
        self._last_discard_tile: int = -1

    def _add_event(self, event_type: str, player: int, tile: int = -1, data: Dict = None):
        """イベント履歴に構造化されたイベントを追加する。"""
        if data is None: data = {}
        event_info = {
            "type": EVENT_TYPES.get(event_type, -1),
            "player": player,
            "tile_index": tile_id_to_index(tile),
            "junme": int(np.ceil(self.junme)),
            "data": data
        }
        self.event_history.append(event_info)

    def _sort_hand(self, p_id: int):
        self.player_hands[p_id].sort(key=lambda t: (tile_id_to_index(t), t))

    def init_round(self, round_data: Dict):
        """ラウンドデータから状態を初期化する。"""
        self.reset_state()
        init_info = round_data["init"]
        seed = init_info.get("seed", "0,0,0,0,0,0").split(",")
        self.round_num_wind, self.honba, self.kyotaku = int(seed[0]), int(seed[1]), int(seed[2])
        dora_id = int(seed[5])
        if 0 <= dora_id <= 135: self.dora_indicators = [dora_id]
        self.dealer = int(init_info.get("oya", 0))
        self.current_player = self.dealer
        self.current_scores = [int(s) * 100 for s in init_info.get("ten", "250,250,250,250").split(",")]
        for p in range(NUM_PLAYERS):
            hai_str = init_info.get(f"hai{p}", "")
            if hai_str:
                self.player_hands[p] = [int(h) for h in hai_str.split(',') if h]
                self._sort_hand(p)
        self.wall_tile_count = 136 - 14 - sum(len(h) for h in self.player_hands)
        self._add_event("INIT", self.dealer, data={"round": self.round_num_wind, "honba": self.honba})

    def process_tsumo(self, player_id: int, tile_id: int):
        """ツモイベントを処理する。"""
        self.current_player = player_id
        if self.junme == 0.0: self.junme = 0.5 # 最初のツモ
        else: self.junme += 0.5
        self.wall_tile_count -= 1
        self.player_hands[player_id].append(tile_id)
        self._sort_hand(player_id)
        self._add_event("TSUMO", player_id, tile_id)

    def process_discard(self, player_id: int, tile_id: int, tsumogiri: bool):
        """打牌イベントを処理する。"""
        if tile_id in self.player_hands[player_id]:
            self.player_hands[player_id].remove(tile_id)
        else:
            print(f"[警告] P{player_id} が手牌にない牌 {tile_id_to_string(tile_id)} を捨てようとしました。")
        self.player_discards[player_id].append((tile_id, tsumogiri))
        self._last_discard_player = player_id
        self._last_discard_tile = tile_id
        self._add_event("DISCARD", player_id, tile_id, data={"tsumogiri": int(tsumogiri)})
        
        # リーチ成立の処理
        if self.player_reach_status[player_id] == 1:
            self.player_reach_status[player_id] = 2
            self.player_reach_junme[player_id] = self.junme
            self.player_reach_discard_tile[player_id] = tile_id # ★リーチ宣言牌を記録
            self.kyotaku += 1
            self.current_scores[player_id] -= 1000
            self._add_event("REACH", player_id, data={"step": 2, "junme": int(np.ceil(self.junme))})

    def process_naki(self, naki_player_id: int, meld_code: int):
        """鳴きイベントを処理する。"""
        naki_info = decode_naki(meld_code)
        if naki_info['type'] == "不明": return
        
        from_who_rel = naki_info['from_who_relative']
        from_who_abs = (self._last_discard_player + from_who_rel + 1) % NUM_PLAYERS
        
        # decode_nakiから返されるconsumedは手牌から消費される牌
        tiles_to_remove = naki_info['consumed']
        
        # 大明槓、ポンは鳴き元の牌を手動で追加する必要がある
        called_tile = -1
        if naki_info['type'] in ["大明槓", "ポン", "チー"]:
            called_tile = self._last_discard_tile
            naki_info['tiles'].append(called_tile)
            naki_info['tiles'].sort()

        # 加槓の場合は、手牌から1枚消費し、既存のポンを更新
        if naki_info['type'] == "加槓":
            added_tile = tiles_to_remove[0]
            pon_index_to_upgrade = -1
            for i, meld in enumerate(self.player_melds[naki_player_id]):
                if meld['type'] == "ポン" and tile_id_to_index(meld['tiles'][0]) == tile_id_to_index(added_tile):
                    pon_index_to_upgrade = i
                    break
            if pon_index_to_upgrade != -1:
                self.player_melds[naki_player_id][pon_index_to_upgrade]['type'] = "加槓"
                self.player_melds[naki_player_id][pon_index_to_upgrade]['tiles'].append(added_tile)
                self.player_melds[naki_player_id][pon_index_to_upgrade]['tiles'].sort()
        else:
            self.player_melds[naki_player_id].append({
                'type': naki_info['type'],
                'tiles': naki_info['tiles'],
                'from_who': from_who_abs,
            })

        for tile in tiles_to_remove:
            if tile in self.player_hands[naki_player_id]:
                self.player_hands[naki_player_id].remove(tile)
        
        self.current_player = naki_player_id
        self.junme += 0.5 # 鳴きの後、手番が移るので巡目を進める
        self._add_event("N", naki_player_id, called_tile, data={"naki_type": NAKI_TYPES.get(naki_info['type'], -1)})

    def process_reach(self, player_id: int, step: int):
        if step == 1:
            self.player_reach_status[player_id] = 1
            self._add_event("REACH", player_id, data={"step": 1})

    def get_event_sequence_features(self) -> np.ndarray:
        """イベント履歴からシーケンス特徴量を生成する。"""
        sequence = []
        event_total_dim = 6 # 固定
        for event in self.event_history:
            vec = np.zeros(event_total_dim, dtype=np.float32)
            vec[0] = float(event["type"])
            vec[1] = float(event["player"] + 1)
            vec[2] = float(event["tile_index"] + 1)
            vec[3] = float(event["junme"])
            data = event.get("data", {})
            if event["type"] == EVENT_TYPES["DISCARD"]: vec[4] = float(data.get("tsumogiri", 0))
            elif event["type"] == EVENT_TYPES["N"]: vec[4] = float(data.get("naki_type", -1) + 1)
            elif event["type"] == EVENT_TYPES["REACH"]: vec[4] = float(data.get("step", 0))
            sequence.append(vec)
        
        padding_vec = np.zeros(event_total_dim, dtype=np.float32)
        padding_vec[0] = float(EVENT_TYPES["PADDING"])
        padded_sequence = list(self.event_history)[-MAX_EVENT_HISTORY:] + [padding_vec] * (MAX_EVENT_HISTORY - len(self.event_history))
        return np.array([s['vector'] for s in padded_sequence] if isinstance(padded_sequence[0], dict) else padded_sequence, dtype=np.float32)


    def get_static_features(self, player_id: int) -> np.ndarray:
        """指定プレイヤー視点の新しい静的特徴量(543次元)を生成する。"""
        features = np.zeros(STATIC_FEATURE_DIM, dtype=np.float32)
        idx = 0

        # 1. グローバル情報 (5次元)
        features[idx:idx+5] = [self.round_num_wind, self.honba, self.kyotaku, self.wall_tile_count, self.junme]
        idx += 5

        # 2. ドラ情報 (68次元)
        dora_ind_counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        dora_counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        for ind_id in self.dora_indicators:
            ind_idx = tile_id_to_index(ind_id)
            if ind_idx != -1:
                dora_ind_counts[ind_idx] += 1
                dora_idx = -1
                if 0 <= ind_idx <= 26: dora_idx = (ind_idx // 9) * 9 + (ind_idx % 9 + 1) % 9
                elif 27 <= ind_idx <= 30: dora_idx = 27 + (ind_idx - 27 + 1) % 4
                elif 31 <= ind_idx <= 33: dora_idx = 31 + (ind_idx - 31 + 1) % 3
                if dora_idx != -1: dora_counts[dora_idx] += 1
        features[idx:idx+34] = dora_ind_counts; idx += 34
        features[idx:idx+34] = dora_counts; idx += 34

        # 3. 自身の手牌情報 (38次元)
        hand_counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        num_dora = 0
        num_aka = 0
        for tile in self.player_hands[player_id]:
            t_idx = tile_id_to_index(tile)
            if t_idx != -1:
                hand_counts[t_idx] += 1
                if dora_counts[t_idx] > 0: num_dora += dora_counts[t_idx]
                if tile in [16, 52, 88]: num_aka += 1
        features[idx:idx+34] = hand_counts; idx += 34
        features[idx:idx+4] = [float(player_id == self.dealer), self.current_scores[player_id] / 80000.0, num_dora, num_aka]
        idx += 4

        # 4. 全プレイヤーの公開情報 (相対位置順: 自分→下家→対面→上家)
        for i in range(NUM_PLAYERS):
            p_abs = (player_id + i) % NUM_PLAYERS
            
            # リーチ情報 (3次元)
            reach_junme = self.player_reach_junme[p_abs] if self.player_reach_status[p_abs] == 2 else -1.0
            reach_tile_idx = tile_id_to_index(self.player_reach_discard_tile[p_abs])
            features[idx:idx+3] = [float(self.player_reach_status[p_abs] == 2), reach_junme, reach_tile_idx]
            idx += 3

            # 全捨て牌カウント (34次元)
            discard_counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
            for tile, _ in self.player_discards[p_abs]:
                d_idx = tile_id_to_index(tile)
                if d_idx != -1: discard_counts[d_idx] += 1
            features[idx:idx+34] = discard_counts; idx += 34

            # 手出し牌カウント (34次元)
            tedashi_counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
            for tile, tsumogiri in self.player_discards[p_abs]:
                if not tsumogiri:
                    d_idx = tile_id_to_index(tile)
                    if d_idx != -1: tedashi_counts[d_idx] += 1
            features[idx:idx+34] = tedashi_counts; idx += 34
            
            # 直近3巡の捨て牌 (3次元)
            last_3_discards = [-1, -1, -1]
            for j, (tile, _) in enumerate(reversed(self.player_discards[p_abs][-3:])):
                last_3_discards[j] = tile_id_to_index(tile)
            features[idx:idx+3] = last_3_discards; idx += 3

            # 副露牌カウント (34次元)
            meld_counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
            for meld in self.player_melds[p_abs]:
                for tile in meld['tiles']:
                    m_idx = tile_id_to_index(tile)
                    if m_idx != -1: meld_counts[m_idx] += 1
            features[idx:idx+34] = meld_counts; idx += 34

        if idx != STATIC_FEATURE_DIM:
            print(f"[致命的エラー] 特徴量次元が一致しません！期待値: {STATIC_FEATURE_DIM}, 実際: {idx}")
        return features

    def get_valid_discard_options(self, player_id: int) -> list[int]:
        """プレイヤーが捨てられる牌の選択肢（牌種インデックス）を返す。"""
        hand = self.player_hands[player_id]
        if self.player_reach_status[player_id] == 2:
            if len(hand) % 3 == 2 and hand:
                return [tile_id_to_index(hand[-1])]
            return []
        if len(hand) % 3 != 2: return []
        return sorted(list(set(tile_id_to_index(t) for t in hand)))

# ==============================================================================
# == デバッグフレームワーク (表示・ステップ実行) ==
# ==============================================================================
def format_hand(hand_ids: list) -> str:
    return " ".join(sorted([tile_id_to_string(t) for t in hand_ids], key=lambda s: (s[-1], s[0])))

def format_discards(discard_list: list) -> str:
    return " ".join([f"{tile_id_to_string(t)}{'*' if f else ''}" for t, f in discard_list])

def format_melds(meld_list: list) -> str:
    return " | ".join([f"{m['type']}[{format_hand(m['tiles'])}]" for m in meld_list])

def process_event(gs: GameState, tag: str, attrib: dict, event_idx: int, all_events: list, process_only: bool = False):
    """単一イベントを処理し、説明と状態変化を返す。"""
    description, processed, player_id = "", False, -1
    
    # 共通処理関数
    def call_gs_method(method_name, *args):
        nonlocal processed, description
        if not process_only:
            try:
                getattr(gs, method_name)(*args)
            except Exception as e:
                print(f"    [エラー] GameState.{method_name} 失敗: {e}"); traceback.print_exc(limit=1)
        processed = True

    # ツモ
    for t_tag, p_id in GameState.TSUMO_TAGS.items():
        if tag.startswith(t_tag) and tag[1:].isdigit():
            player_id, pai_id = p_id, int(tag[1:])
            description = f"P{player_id} ツモ {tile_id_to_string(pai_id)}"
            call_gs_method('process_tsumo', player_id, pai_id)
            if not process_only: # --- ★特徴量ダンプポイント★ ---
                print("    --- 特徴量生成ポイント ---")
                try:
                    static_features = gs.get_static_features(player_id)
                    print(f"    静的特徴量 Shape: {static_features.shape}")
                    if static_features.shape[0] != STATIC_FEATURE_DIM: print(f"    [警告] 静的特徴量次元不一致！")
                    # 次のイベントから正解ラベルを取得
                    if event_idx + 1 < len(all_events):
                        next_tag = all_events[event_idx + 1]["tag"]
                        for d_tag, d_p_id in GameState.DISCARD_TAGS.items():
                            if next_tag.startswith(d_tag) and next_tag[1:].isdigit() and d_p_id == player_id:
                                label_id = int(next_tag[1:])
                                print(f"    予測すべき打牌 (ラベル): {tile_id_to_string(label_id)} (Index: {tile_id_to_index(label_id)})")
                                break
                except Exception as e: print(f"    [エラー] 特徴量生成/表示で失敗: {e}"); traceback.print_exc(limit=1)
            return description, True

    # 打牌
    for d_tag, p_id in GameState.DISCARD_TAGS.items():
        if tag.startswith(d_tag) and tag[1:].isdigit():
            player_id, pai_id, tsumogiri = p_id, int(tag[1:]), d_tag.islower()
            description = f"P{player_id} 打 {tile_id_to_string(pai_id)}{'*' if tsumogiri else ''}"
            call_gs_method('process_discard', player_id, pai_id, tsumogiri)
            return description, True

    # その他イベント
    if tag == "N":
        p_id, m_code = int(attrib.get("who", -1)), int(attrib.get("m", 0))
        if p_id != -1:
            description = f"P{p_id} 鳴き ({decode_naki(m_code)['type']})"
            call_gs_method('process_naki', p_id, m_code)
    elif tag == "REACH":
        p_id, step = int(attrib.get("who",-1)), int(attrib.get("step",0))
        if p_id != -1 and step == 1:
            description = f"P{p_id} リーチ宣言"
            call_gs_method('process_reach', p_id, step)
    elif tag in ["AGARI", "RYUUKYOKU"]:
        description = f"局終了 ({tag})"
        processed = True # GameStateでの処理は省略
    
    return description if description else f"未処理タグ: {tag}", processed

def print_game_state_summary(gs: GameState):
    """現在のGameStateのサマリーを表示する。"""
    print(f"  局: {gs.round_num_wind} 本場: {gs.honba} 供託: {gs.kyotaku} 巡目: {gs.junme:.1f} 親: P{gs.dealer}")
    print(f"  ドラ表示: {[tile_id_to_string(t) for t in gs.dora_indicators]}")
    print(f"  点数: {gs.current_scores}")
    for p in range(NUM_PLAYERS):
        reach_info = ""
        if gs.player_reach_status[p] == 1: reach_info = "(リーチ宣言)"
        elif gs.player_reach_status[p] == 2: reach_info = f"* (リーチ@{gs.player_reach_junme[p]:.1f}巡)"
        print(f"  --- Player {p} {reach_info} ---")
        print(f"    手牌: {format_hand(gs.player_hands[p])} ({len(gs.player_hands[p])}枚)")
        print(f"    河: {format_discards(gs.player_discards[p])}")
        print(f"    副露: {format_melds(gs.player_melds[p])}")

# ==============================================================================
# == メイン実行ロジック ==
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="天鳳牌譜をステップ実行し、GameStateと特徴量生成をデバッグします。")
    parser.add_argument("xml_file", help="デバッグ対象の天鳳XMLログファイル。")
    parser.add_argument("round_index", type=int, help="デバッグ対象の局インデックス (1-based)。")
    parser.add_argument("--start", type=int, default=1, metavar='EVENT_NUM', help="状態表示を開始するイベント番号 (1-based, default: 1)。")
    parser.add_argument("--count", type=int, default=999, metavar='NUM_EVENTS', help="表示する最大イベント数 (default: all)。")
    args = parser.parse_args()

    if not os.path.exists(args.xml_file):
        print(f"エラー: XMLファイルが見つかりません: {args.xml_file}"); sys.exit(1)

    try:
        print(f"牌譜ファイル {args.xml_file} を解析中...")
        meta, rounds_data = parse_full_mahjong_log(args.xml_file)
        if not (1 <= args.round_index <= len(rounds_data)):
            print(f"エラー: 無効な局インデックス {args.round_index}。範囲: 1-{len(rounds_data)}"); sys.exit(1)

        round_data = rounds_data[args.round_index - 1]
        events = round_data.get("events", [])
        print(f"対象: 第{args.round_index}局 (イベント数: {len(events)})")

        game_state = GameState()
        print("局の初期状態を構築中...")
        game_state.init_round(round_data)
        print("--- 初期状態 (INIT直後) ---")
        print_game_state_summary(game_state); print("-" * 40)

        start_idx = max(0, args.start - 1)
        if start_idx > 0:
            print(f"最初の {start_idx} イベントを早送り中...")
            for i in range(start_idx):
                process_event(game_state, events[i]["tag"], events[i]["attrib"], i, events, process_only=True)
            print("--- 表示開始前の状態 ---")
            print_game_state_summary(game_state); print("-" * 40)

        end_idx = min(start_idx + args.count, len(events))
        print(f"--- イベント {start_idx + 1} から {end_idx} を表示 ---")
        for i in range(start_idx, end_idx):
            event = events[i]
            print(f"\n>>> イベント {i+1}/{len(events)}: <{event['tag']}> {event['attrib']}")
            desc, _ = process_event(game_state, event["tag"], event["attrib"], i, events, process_only=False)
            print(f"    アクション: {desc}")
            print("--- イベント処理後の状態 ---")
            print_game_state_summary(game_state); print("-" * 40)
            if event['tag'] in ["AGARI", "RYUUKYOKU"]:
                print("\n--- 局終了 ---"); break
        
        print("\nデバッグセッション終了。")

    except Exception as e:
        print(f"\n[致命的エラー] 予期せぬエラーが発生しました:"); traceback.print_exc(); sys.exit(1)

if __name__ == "__main__":
    main()