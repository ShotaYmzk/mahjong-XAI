# predict.py (Transformer版 - SHAP説明機能付き・日本語化)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import math # math をインポート
import glob # 背景データ読み込み用
import time # SHAP計算時間計測用

# SHAPとMatplotlibをインポート (なければインストール: pip install shap matplotlib)
try:
    import shap
    import matplotlib.pyplot as plt
    shap_available = True
except ImportError:
    print("[警告] `shap` または `matplotlib` ライブラリが見つかりません。SHAP説明機能はスキップされます。")
    print("インストールするには: pip install shap matplotlib")
    shap_available = False

from full_mahjong_parser import parse_full_mahjong_log
# 修正されたGameStateと関連クラス・定数をインポート
# game_state.py に MAX_EVENT_HISTORY が定義されているか確認
try:
    # ★★★ game_state.py から必要なものをインポート ★★★
    from game_state import GameState, NUM_TILE_TYPES, MAX_EVENT_HISTORY, calculate_shanten
    # ↑↑↑ calculate_shanten もインポート (ダミーでもOKだが結果に影響)
except ImportError as e:
    print(f"[エラー] game_state.pyからのインポートに失敗しました: {e}")
    print("game_state.py が同じディレクトリにあるか、必要な定義が含まれているか確認してください。")
    # 動作継続のためダミー値を設定するが、基本的にはエラー終了させるべき
    from game_state import GameState, NUM_TILE_TYPES # 最低限
    MAX_EVENT_HISTORY = 60 # ★ データ生成時と合わせる必要あり
    def calculate_shanten(hand, melds): return 8, [] # ★ ダミー関数
    print("[警告] game_state から MAX_EVENT_HISTORY または calculate_shanten をインポートできませんでした。デフォルト値/ダミーを使用します。")


from naki_utils import decode_naki
from tile_utils import tile_id_to_string, tile_id_to_index

# --- クラス定義をここにコピー ---
# (train.py から MahjongTransformerModel と PositionalEncoding をコピー)
# (注意: 必ず学習・保存時に使用したバージョンのクラス定義をコピーしてください)

NUM_TILE_TYPES = 34 # train.pyから持ってくる

class PositionalEncoding(nn.Module):
    """Transformer用の位置エンコーディング (batch_first=True 対応)"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        if d_model <= 0: raise ValueError("d_model は正の数である必要があります")
        max_len = max(1, max_len) # max_lenが0以下にならないように

        position = torch.arange(max_len).unsqueeze(1)
        try:
            # div_term の計算
            div_term_base = torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            div_term = torch.exp(div_term_base)
        except ValueError as e:
            print(f"[エラー] PositionalEncoding の div_term 計算に失敗: d_model={d_model}, Error={e}")
            raise e

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        # d_modelが奇数か偶数かで処理を分ける
        if d_model % 2 == 0:
             # d_modelが偶数の場合
             if div_term.shape[0] == pe[:, 1::2].shape[1]: # div_termの長さが一致するか確認
                 pe[:, 1::2] = torch.cos(position * div_term)
             else:
                 print(f"[警告] PositionalEncoding: d_model({d_model}) is even but div_term length ({div_term.shape[0]}) mismatch with target shape ({pe[:, 1::2].shape[1]}). Padding cos term.")
                 len_diff = pe[:, 1::2].shape[1] - div_term.shape[0]
                 if len_diff > 0:
                     pe[:, 1::2][:, :-len_diff] = torch.cos(position * div_term)
                 else: # 通常は発生しないはず
                      pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        else:
             # d_modelが奇数の場合、cosの最後の要素は計算できない
             if div_term.shape[0] > 0: # div_termが空でないことを確認
                pe[:, 1::2] = torch.cos(position * div_term) # 最後の次元は0のままになる

        self.register_buffer('pe', pe.unsqueeze(0)) # バッチ次元を追加 [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] (batch_first=Trueのため)
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
             # シーケンス長がpeの最大長を超えた場合の処理（エラーまたはリサイズ）
             # ここではエラーを発生させる
             raise ValueError(f"入力シーケンス長 ({seq_len}) が PositionalEncoding の最大長 ({self.pe.size(1)}) を超えています。")
        # スライスして加算
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class MahjongTransformerModel(nn.Module):
    """麻雀打牌予測用Transformerモデル"""
    # ★★★ 学習・保存時と同じクラス定義 ★★★
    def __init__(self,
                 event_feature_dim: int,
                 static_feature_dim: int,
                 # ↓↓↓ 学習・保存時と同じデフォルト値 or 固定値を記述 ↓↓↓
                 d_model: int = 128,
                 nhead: int = 4,
                 d_hid: int = 256,  # ← エラーから推測される学習時の値
                 nlayers: int = 2,
                 dropout: float = 0.1,
                 output_dim: int = NUM_TILE_TYPES,
                 max_seq_len: int = 60): # ← 実際のシーケンス長を使う
        super().__init__()
        # 引数チェック
        if d_model <= 0: raise ValueError("d_model は正でなければなりません")
        if nhead <= 0: raise ValueError("nhead は正でなければなりません")
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) は nhead ({nhead}) で割り切れる必要があります")
        if d_hid <= 0: raise ValueError("d_hid は正でなければなりません")
        if nlayers <= 0: raise ValueError("nlayers は正でなければなりません")
        if not (0.0 <= dropout < 1.0): raise ValueError("dropout は 0.0 以上 1.0 未満である必要があります")
        if max_seq_len <= 0: raise ValueError("max_seq_len は正でなければなりません")


        self.d_model = d_model

        # 1. Embedding + Positional Encoding
        self.event_encoder = nn.Linear(event_feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)

        # 2. Transformer Encoder
        # TransformerEncoderLayer のパラメータを確認
        try:
            # PyTorch 1.9 以降の norm_first パラメータ対応を試みる
            from pkg_resources import parse_version
            norm_first_supported = parse_version(torch.__version__) >= parse_version("1.9")
        except ImportError:
            norm_first_supported = False # pkg_resourcesがない場合はFalseと仮定

        encoder_layer_args = {
            'd_model': d_model,
            'nhead': nhead,
            'dim_feedforward': d_hid,
            'dropout': dropout,
            'batch_first': True # 常にTrue
        }
        if norm_first_supported:
            # norm_first のデフォルトは False だが、Trueの方が性能が良いとされる場合もある
            encoder_layer_args['norm_first'] = False # または True

        try:
             encoder_layers = nn.TransformerEncoderLayer(**encoder_layer_args)
        except TypeError as e:
             # 古いバージョンで norm_first などがない場合のエラーハンドリング
             print(f"[警告] TransformerEncoderLayer の初期化に失敗 (古いPyTorchの可能性): {e}")
             # サポートされていない引数を削除して再試行
             unsupported_args = ['norm_first', 'batch_first'] # 古いバージョンで非対応の可能性
             for arg in unsupported_args:
                 encoder_layer_args.pop(arg, None)
             try:
                 print("[情報] 引数を減らして TransformerEncoderLayer を再初期化します:", encoder_layer_args)
                 encoder_layers = nn.TransformerEncoderLayer(**encoder_layer_args)
                 print("[警告] batch_first=True がサポートされていない可能性があります。入力テンソルの調整が必要になるかもしれません。")
             except Exception as fallback_e:
                 print("[エラー] TransformerEncoderLayer の初期化に完全に失敗しました。")
                 raise fallback_e

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        # 3. 静的特徴量用エンコーダー (MLP)
        static_encoder_hid_dim = max(1, d_model // 2)
        self.static_encoder = nn.Sequential(
            nn.Linear(static_feature_dim, static_encoder_hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(static_encoder_hid_dim, static_encoder_hid_dim)
        )

        # 4. 結合層と最終出力層 (MLP)
        decoder_input_dim = d_model + static_encoder_hid_dim
        decoder_hid_dim = max(1, d_model // 2)
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, decoder_hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hid_dim, output_dim)
        )

        self._init_weights() # 重み初期化

    def _init_weights(self) -> None:
        """モデルの重みを初期化する"""
        initrange = 0.1
        # 各Linear層の重みとバイアスを初期化
        layers_to_init = [self.event_encoder] + \
                         list(self.static_encoder.children()) + \
                         list(self.decoder.children())

        for layer in layers_to_init:
            if isinstance(layer, nn.Linear):
                if hasattr(layer, 'weight') and layer.weight is not None:
                    layer.weight.data.uniform_(-initrange, initrange)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, event_seq: torch.Tensor, static_feat: torch.Tensor, src_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """モデルの順伝播"""
        # 入力形状チェック (デバッグ用)
        # print("Input shapes:", event_seq.shape, static_feat.shape, src_padding_mask.shape if src_padding_mask is not None else "None")

        # 1. イベントシーケンス処理
        # embedding_dim が d_model と一致するか確認
        if self.event_encoder.out_features != self.d_model:
             raise ValueError("event_encoder の出力次元が d_model と一致しません")
        embedded_seq = self.event_encoder(event_seq) * math.sqrt(self.d_model)
        pos_encoded_seq = self.pos_encoder(embedded_seq)

        # Transformer Encoder
        transformer_output = self.transformer_encoder(pos_encoded_seq, src_key_padding_mask=src_padding_mask)

        # 平均プーリング (マスク考慮)
        if src_padding_mask is not None:
            active_elements_mask = ~src_padding_mask # Trueが有効
            seq_len_valid = active_elements_mask.sum(dim=1, keepdim=True).float()
            seq_len_valid = torch.max(seq_len_valid, torch.tensor(1.0, device=event_seq.device))
            # マスク部分を0埋めして合計し、有効長で割る
            masked_output = transformer_output.masked_fill(src_padding_mask.unsqueeze(-1), 0.0)
            transformer_pooled = masked_output.sum(dim=1) / seq_len_valid
        else:
            transformer_pooled = transformer_output.mean(dim=1)

        # 2. 静的特徴量処理
        encoded_static = self.static_encoder(static_feat)

        # 3. 結合して最終予測
        # デバッグ: 結合前の形状確認
        # print("Pooled shape:", transformer_pooled.shape, "Static encoded shape:", encoded_static.shape)
        combined_features = torch.cat((transformer_pooled, encoded_static), dim=1)
        output = self.decoder(combined_features)

        return output

# --- ここまでクラス定義コピー ---


# --- 設定 ---
NUM_PLAYERS = 4
DEFAULT_MODEL_PATH = "./trained_model/mahjong_transformer_model.pth" # 相対パスに戻す（環境依存を減らすため）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

print(f"使用デバイス: {DEVICE}")


# --- ヘルパー関数 ---
def format_hand(hand_ids):
    """手牌IDリストを見やすい文字列に変換"""
    if not hand_ids: return "なし"
    # tile_id_to_index と ID 自体でソート
    sorted_ids = sorted(hand_ids, key=lambda t: (tile_id_to_index(t), t))
    return " ".join([tile_id_to_string(t) for t in sorted_ids])

def format_discards(discard_list):
    """捨て牌リスト [(牌ID, ツモ切りフラグ), ...] を文字列に変換"""
    if not discard_list: return "なし"
    return " ".join([f"{tile_id_to_string(t)}{'*' if tsumogiri else ''}" for t, tsumogiri in discard_list])

def format_melds(meld_list):
    """副露リスト [(種類, 牌IDリスト, 鳴かれた牌ID, 誰から), ...] を文字列に変換"""
    if not meld_list: return "なし"
    meld_strs = []
    for m_type, m_tiles, trigger_id, from_who_abs in meld_list:
        # 副露牌はソートして表示
        tiles_str = " ".join([tile_id_to_string(t) for t in sorted(m_tiles, key=lambda x: (tile_id_to_index(x),x))])
        # 誰からの表示（暗槓、加槓以外）
        from_str = f" P{from_who_abs}" if from_who_abs != -1 and m_type not in ["暗槓", "加槓"] else ""
        # 鳴いた牌の表示（トリガー牌がある場合）
        trigger_str = f"({tile_id_to_string(trigger_id)})" if trigger_id != -1 and m_type != "暗槓" else ""
        meld_strs.append(f"{m_type}{trigger_str}[{tiles_str}]{from_str}")
    return " / ".join(meld_strs)


# --- モデルロード関数 ---
def load_trained_model(model_path, event_dim, static_dim, seq_len):
    """訓練済みTransformerモデルをロードする"""
    print(f"モデルファイルをロードしようとしています: {model_path}")
    # ファイル存在確認
    if not os.path.exists(model_path):
        # カレントディレクトリからの相対パスも試す
        current_dir_path = os.path.join(os.getcwd(), os.path.basename(model_path))
        print(f"指定パスにファイルが見つかりません。カレントディレクトリも確認します: {current_dir_path}")
        if os.path.exists(current_dir_path):
            model_path = current_dir_path
            print("カレントディレクトリでモデルファイルを発見しました。")
        else:
            # . または .. から始まる相対パスを試す
            script_dir = os.path.dirname(__file__)
            rel_path_dot = os.path.join(script_dir, model_path)
            rel_path_dotdot = os.path.join(script_dir, "..", model_path)
            if os.path.exists(rel_path_dot):
                 model_path = rel_path_dot
                 print(f"スクリプトディレクトリからの相対パスで発見: {model_path}")
            elif os.path.exists(rel_path_dotdot):
                 model_path = rel_path_dotdot
                 print(f"スクリプトディレクトリの親からの相対パスで発見: {model_path}")
            else:
                 raise FileNotFoundError(f"モデルファイルが見つかりません。指定パス: {model_path}, CWD: {current_dir_path}, 相対パス候補: {rel_path_dot}, {rel_path_dotdot}")


    # モデルインスタンス化 (ファイル内に定義されたクラスを使用)
    model_params = {
        'event_feature_dim': event_dim,
        'static_feature_dim': static_dim,
        'max_seq_len': seq_len,
        'd_model': 128,
        'nhead': 4,
        'd_hid': 256, # 学習時と合わせる
        'nlayers': 2,
        'dropout': 0.1
    }
    print(f"以下のパラメータでモデルを初期化します: {model_params}")
    # ★★★ ファイル内に定義されたクラスを直接呼び出す ★★★
    model = MahjongTransformerModel(**model_params).to(DEVICE)

    try:
        # state_dictをロード
        print(f"state_dict をロードします...")
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval() # 評価モード
        print(f"モデルのロードに成功しました: {model_path}")
        return model
    except RuntimeError as e:
        print(f"[エラー] state_dict のロードに失敗しました (RuntimeError): {e}")
        print(f"  >>> モデル定義（このファイル内）と <<<")
        print(f"  >>> 保存されたモデル ('{model_path}') の構造が一致しているか確認してください。 <<<")
        print(f"  >>> 特に全てのハイパーパラメータ (d_model, nhead, d_hid, nlayers など) を確認してください。 <<<")
        # 保存された state_dict のキーと形状をいくつか表示してみる (デバッグ用)
        print("\n--- Checkpoint State Dict Keys (Sample) ---")
        try:
            count = 0
            loaded_state_dict = torch.load(model_path, map_location='cpu') # CPUにロードして確認
            for key, value in loaded_state_dict.items():
                print(f"  {key}: {value.shape}")
                count += 1
                if count >= 10: break
        except Exception as inspect_e:
             print(f"state_dict の内容を確認できませんでした: {inspect_e}")
        print("------------------------------------------")
        raise e
    except Exception as e:
        print(f"[エラー] state_dict のロード中に予期せぬエラー: {e}")
        raise e

# --- 局面復元関数 ---
def reconstruct_game_state_at_tsumo(xml_path, target_round_index, target_tsumo_event_count_in_round):
    """指定局面までのGameStateを復元"""
    # (この関数は前のコードから変更なし、エラーハンドリングを少し追加)
    print(f"牌譜ファイル {xml_path} を解析中...")
    try:
        meta, rounds_data = parse_full_mahjong_log(xml_path)
    except FileNotFoundError:
        print(f"[エラー] 牌譜ファイルが見つかりません: {xml_path}")
        raise
    except Exception as e:
        print(f"[エラー] 牌譜ファイルの解析中にエラーが発生しました: {e}")
        raise

    if not (1 <= target_round_index <= len(rounds_data)):
        raise ValueError(f"指定された局インデックスが無効です: {target_round_index} (利用可能な範囲: 1-{len(rounds_data)})")

    round_data = rounds_data[target_round_index - 1]
    game_state = GameState()
    print(f"第{target_round_index}局の初期状態を構築中...")
    try:
        game_state.init_round(round_data)
    except Exception as e:
        print(f"[エラー] GameState の初期化中にエラーが発生しました: {e}")
        raise

    current_tsumo_count = 0
    target_tsumo_event_info = None
    actual_discard_event_info = None
    events = round_data.get("events", [])
    print(f"イベントを再生し、{target_tsumo_event_count_in_round} 回目のツモを探します...")

    for i, event_xml in enumerate(events):
        tag = event_xml["tag"]
        attrib = event_xml["attrib"]
        processed = False
        try:
            # --- 自摸イベント ---
            tsumo_player_id = -1; tsumo_pai_id = -1
            for t_tag, p_id in GameState.TSUMO_TAGS.items():
                if tag.startswith(t_tag):
                    try: tsumo_pai_id = int(tag[1:]); tsumo_player_id = p_id; processed = True; break
                    except (ValueError, IndexError): continue # 不正なタグは無視
            if processed:
                current_tsumo_count += 1
                # print(f"  イベント {i}: P{tsumo_player_id} ツモ (カウント: {current_tsumo_count})") # デバッグ用
                if current_tsumo_count == target_tsumo_event_count_in_round:
                    print(f"ターゲットのツモ ({current_tsumo_count}回目) を発見しました。")
                    target_tsumo_event_info = {"player": tsumo_player_id, "pai": tsumo_pai_id, "xml": event_xml}
                    game_state.process_tsumo(tsumo_player_id, tsumo_pai_id) # 状態更新
                    # 次の打牌イベントを探す (なければ None)
                    if i + 1 < len(events):
                        next_event_xml = events[i+1]; next_tag = next_event_xml["tag"]
                        for d_tag, p_id_next in GameState.DISCARD_TAGS.items():
                            if next_tag.startswith(d_tag) and p_id_next == tsumo_player_id:
                                try:
                                    discard_pai_id = int(next_tag[1:]); tsumogiri = next_tag[0].islower()
                                    actual_discard_event_info = {"player": p_id_next, "pai": discard_pai_id, "tsumogiri": tsumogiri, "xml": next_event_xml}
                                    break
                                except (ValueError, IndexError): continue
                    print("指定局面の状態復元が完了しました。")
                    return game_state, target_tsumo_event_info, actual_discard_event_info # 発見したので返す
                else:
                    game_state.process_tsumo(tsumo_player_id, tsumo_pai_id) # 状態更新
                continue

            # --- 打牌イベント ---
            discard_player_id = -1; discard_pai_id = -1; tsumogiri = False
            for d_tag, p_id in GameState.DISCARD_TAGS.items():
                if tag.startswith(d_tag):
                    try: discard_pai_id = int(tag[1:]); discard_player_id = p_id; tsumogiri = tag[0].islower(); processed = True; break
                    except (ValueError, IndexError): continue
            if processed:
                # print(f"  イベント {i}: P{discard_player_id} 打牌") # デバッグ用
                game_state.process_discard(discard_player_id, discard_pai_id, tsumogiri)
                continue

            # --- 鳴きイベント ---
            if not processed and tag == "N":
                try:
                    naki_player_id = int(attrib.get("who", -1)); meld_code = int(attrib.get("m", "0"))
                    if naki_player_id != -1: game_state.process_naki(naki_player_id, meld_code)
                    processed = True;
                except (ValueError, KeyError, Exception) as e: print(f"[警告] 鳴きイベント(N)の処理中にエラー: {e}, Attrib: {attrib}")
                continue

            # --- リーチイベント ---
            if not processed and tag == "REACH":
                 try:
                     reach_player_id = int(attrib.get("who", -1)); step = int(attrib.get("step", 0))
                     if reach_player_id != -1: game_state.process_reach(reach_player_id, step)
                     processed = True;
                 except (ValueError, KeyError, Exception) as e: print(f"[警告] リーチイベント(REACH)の処理中にエラー: {e}, Attrib: {attrib}")
                 continue

            # --- DORA イベント ---
            if not processed and tag == "DORA":
                 try:
                     hai = int(attrib.get("hai", -1))
                     if hai != -1: game_state.process_dora(hai)
                     processed = True;
                 except (ValueError, KeyError, Exception) as e: print(f"[警告] ドラ表示イベント(DORA)の処理中にエラー: {e}, Attrib: {attrib}")
                 continue

            # --- 局終了イベント ---
            if not processed and (tag == "AGARI" or tag == "RYUUKYOKU"):
                 print(f"局終了イベント ({tag}) を検出しました。")
                 try:
                     if tag == "AGARI": game_state.process_agari(attrib)
                     else: game_state.process_ryuukyoku(attrib)
                 except Exception as e: print(f"[警告] 局終了イベントの処理中にエラー: {e}, Attrib: {attrib}")
                 processed = True; break # この局のイベント再生は終了

            # その他のタグは無視 (デバッグ時は表示してもよい)
            # if not processed: print(f"  イベント {i}: 未処理タグ {tag}")

        except Exception as e:
            print(f"[エラー] イベント {i} (タグ: {tag}, 属性: {attrib}) の処理中に予期せぬエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            # エラーが発生した場合、続行するかどうか
            # ここでは続行せずにエラーを投げる
            raise e

    # ループが終了しても指定ツモ回数に達しなかった場合
    raise ValueError(f"指定されたツモ回数 ({target_tsumo_event_count_in_round}) に到達する前に局が終了、またはイベントがありませんでした（局: {target_round_index}）。")


# --- 打牌予測関数 ---
def predict_discard(model, game_state: GameState, player_id: int):
    """モデルで打牌予測を行い、手牌の制約を考慮して最終的な捨て牌を決定する。"""
    # (この関数は前のコードから変更なし)
    # ... (内部実装は変更なし) ...
    try:
        event_sequence = game_state.get_event_sequence_features()
        static_features = game_state.get_static_features(player_id)
    except Exception as e:
        print(f"[エラー] 特徴量生成中にエラーが発生しました: {e}")
        raise

    # 次元数チェック（デバッグ用）
    # print(f"Debug: Seq shape={event_sequence.shape}, Static shape={static_features.shape}")

    seq_tensor = torch.tensor(event_sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    static_tensor = torch.tensor(static_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # パディングコード取得とマスク生成
    try:
        padding_code = GameState.EVENT_TYPES["PADDING"]
    except AttributeError:
        print("[警告] GameState.EVENT_TYPES['PADDING'] が見つかりません。デフォルト値 8 を使用します。")
        padding_code = 8
    mask_tensor = (seq_tensor[:, :, 0] == padding_code).to(DEVICE)

    # モデル予測
    with torch.no_grad():
        try:
            outputs = model(seq_tensor, static_tensor, mask_tensor)
        except Exception as e:
            print(f"[エラー] モデルのforward計算中にエラーが発生しました: {e}")
            # モデルの入力形状などを表示してデバッグ支援
            print(f"  Input shapes: event_seq={seq_tensor.shape}, static_feat={static_tensor.shape}, mask={mask_tensor.shape}")
            raise
        probabilities = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    # 有効牌選択
    valid_discard_indices = game_state.get_valid_discard_options(player_id)
    best_prob = -1.0
    best_index = -1

    if not valid_discard_indices:
        print("[警告] 有効な打牌選択肢がありません！リーチ後のツモ切り牌などを確認してください。")
        # フォールバックとして確率最大の牌を選択
        best_index = np.argmax(probabilities)
        if 0 <= best_index < len(probabilities):
             best_prob = probabilities[best_index]
        else:
             print("[エラー] 確率配列から最大値を取得できませんでした。")
             return 0, 0.0, probabilities # ダミー値を返す
    else:
        for index in valid_discard_indices:
            if 0 <= index < NUM_TILE_TYPES:
                if probabilities[index] > best_prob:
                    best_prob = probabilities[index]
                    best_index = index
            else: print(f"[警告] 無効な牌インデックス {index} が有効選択肢に含まれています。")
        # 万が一、有効牌の中で選択できなかった場合 (確率は通常非負)
        if best_index == -1:
            print(f"[警告] 有効牌の中から最良の打牌を決定できませんでした。最初の有効牌 ({valid_discard_indices[0]}) を選択します。")
            best_index = valid_discard_indices[0]
            best_prob = probabilities[best_index] if 0 <= best_index < len(probabilities) else 0.0

    # 最終チェック
    if not (0 <= best_index < NUM_TILE_TYPES):
        print(f"[エラー] 最終的な打牌インデックス ({best_index}) が不正です。")
        return 0, 0.0, probabilities # ダミー値を返す

    return best_index, best_prob, probabilities

# --- 特徴量名生成関数 ---
def generate_feature_names(event_dim, static_dim, seq_len):
    """SHAP説明のための特徴量名を生成する"""
    feature_names = []
    print(f"特徴量名を生成中... (シーケンス長: {seq_len}, イベント次元: {event_dim}, 静的次元: {static_dim})")

    # 1. イベントシーケンス特徴量名
    event_data_names = ["種別", "プレイヤー", "牌Idx", "巡目", "データ1(ツモ切り/鳴き種別)", "データ2(鳴き元)"] # event_dim=6を想定
    if event_dim != len(event_data_names):
        print(f"[警告] イベント次元({event_dim})が想定({len(event_data_names)})と異なります。データ名が不正確になる可能性があります。")
        # 不足分/超過分を汎用名で埋める
        event_data_names = [f"データ{i}" for i in range(event_dim)]
        event_data_names[0:4] = ["種別", "プレイヤー", "牌Idx", "巡目"] # 基本部分は維持

    for i in range(seq_len):
        for j, name_suffix in enumerate(event_data_names):
            feature_names.append(f"Event_{i}_{name_suffix}")

    # 2. 静的特徴量名 (GameState.get_static_features の実装順に依存！)
    static_start_index = len(feature_names)

    # get_static_features の実装をコメントとして残す（確認用）
    # features.append(my_hand_representation.flatten()) # 34*5 = 170
    # features.append(dora_indicator_vec) # 34
    # features.append(dora_tile_vec) # 34
    # for p_offset in range(NUM_PLAYERS):
    #     target_player = (player_id + p_offset) % NUM_PLAYERS
    #     features.append(discard_counts) # 34
    #     if p_offset != 0: features.append(genbutsu_flag) # 34 * 3 = 102 (自分以外)
    #     features.append(meld_vec.flatten()) # (5*34) = 170
    #     features.append(np.array([reach_stat, reach_jun])) # 2
    # features.append(ba_features) # 6 (round, honba, kyotaku, junme, dealer, wall)
    # features.append(my_wind_vec) # 4 (one-hot)
    # features.append(rotated_scores) # 4
    # features.append(shanten_vec) # 9
    # features.append(ukeire_vec) # 34
    # features.append(np.array([num_ukeire])) # 1

    # 2.1. 自分の手牌 (170)
    tile_names_34 = [tile_id_to_string(i*4) for i in range(NUM_TILE_TYPES)]
    for tile_name in tile_names_34:
        for offset in range(4): feature_names.append(f"静的_手牌_{tile_name}_ID{offset}")
        feature_names.append(f"静的_手牌_{tile_name}_枚数")

    # 2.2. ドラ情報 (34 + 34 = 68)
    for tile_name in tile_names_34: feature_names.append(f"静的_ドラ表示_{tile_name}")
    for tile_name in tile_names_34: feature_names.append(f"静的_ドラ牌_{tile_name}")

    # 2.3. 各プレイヤー公開情報 (自分, 下家, 対面, 上家)
    player_offsets = ["自分", "下家", "対面", "上家"]
    naki_type_names = ["チー", "ポン", "大明槓/加槓", "暗槓"] # GameState.NAKI_TYPES に対応想定
    for p_offset in range(NUM_PLAYERS):
        player_label = player_offsets[p_offset]
        # 捨て牌 (34)
        for tile_name in tile_names_34: feature_names.append(f"静的_{player_label}捨牌_{tile_name}")
        # 現物 (自分以外) (34 * 3 = 102)
        if p_offset != 0:
            for tile_name in tile_names_34: feature_names.append(f"静的_{player_label}現物_{tile_name}")
        # 副露 (len(NAKI_TYPES) * 34 = 5 * 34 = 170)
        # GameStateのNAKI_TYPESと順番を合わせる必要あり。ここでは仮に5種類とする。
        num_naki_types_expected = 5 # チー, ポン, 大明槓, 加槓, 暗槓 を想定
        for naki_idx in range(num_naki_types_expected):
             naki_label = f"鳴き{naki_idx}" # より具体的に "鳴きポン" などにできると良い
             for tile_name in tile_names_34: feature_names.append(f"静的_{player_label}{naki_label}_{tile_name}")
        # リーチ状態 (2)
        feature_names.append(f"静的_{player_label}リーチ状態")
        feature_names.append(f"静的_{player_label}リーチ巡目")

    # 2.4. 場況情報 (6)
    ba_feature_names = ["局進行度", "本場", "供託", "巡目", "親フラグ", "壁残枚数"]
    feature_names.extend([f"静的_{name}" for name in ba_feature_names])

    # 2.5. 自風 (4)
    wind_names = ["東家", "南家", "西家", "北家"]
    feature_names.extend([f"静的_自風_{name}" for name in wind_names])

    # 2.6. 点数 (4)
    score_labels = ["自分", "下家", "対面", "上家"]
    feature_names.extend([f"静的_点数_{label}" for label in score_labels])

    # 2.7. 向聴数と受け入れ (9 + 34 + 1 = 44)
    shanten_labels = ["和了", "聴牌", "1向聴", "2向聴", "3向聴", "4向聴", "5向聴", "6向聴", "7向聴以上"] # 9次元分
    feature_names.extend([f"静的_向聴_{label}" for label in shanten_labels])
    for tile_name in tile_names_34: feature_names.append(f"静的_受入_{tile_name}")
    feature_names.append("静的_受入種類数")

    # --- 次元数チェック ---
    expected_len = seq_len * event_dim + static_dim
    if len(feature_names) != expected_len:
        print(f"[警告] 生成された特徴量名の数 ({len(feature_names)}) が期待値 ({expected_len}) と異なります。")
        print(f"       イベントシーケンス: {seq_len * event_dim}, 静的: {len(feature_names) - static_start_index} (期待: {static_dim})")
        print(f"       GameState.get_static_features の実装と突き合わせてください。")
        # 足りない分/多い分を調整 (デバッグ用)
        if len(feature_names) < expected_len:
            diff = expected_len - len(feature_names)
            feature_names.extend([f"不明な特徴量_{i}" for i in range(diff)])
        else:
            feature_names = feature_names[:expected_len]

    print(f"特徴量名の生成完了 (合計: {len(feature_names)}個)")
    return feature_names


# --- SHAP説明関数 ---
def explain_prediction_with_shap(model, background_data, instance_to_explain, feature_names, target_class_index, n_shap_samples=100):
    """SHAP値を計算し、影響の大きい特徴量を出力・プロットする"""
    if not shap_available:
        print("SHAPライブラリが利用できないため、説明をスキップします。")
        return None

    print("\n--- SHAP 説明生成開始 ---")
    start_time = time.time()
    target_class_name = tile_id_to_string(target_class_index*4) if target_class_index != -1 else "N/A"
    print(f"対象クラス: Index={target_class_index}, 牌種={target_class_name}")

    # インスタンスデータ展開
    event_seq_instance, static_feat_instance, _ = instance_to_explain # マスクはラッパー内で生成

    # 背景データ準備
    bg_sequences, bg_static_features = background_data
    bg_sequences_tensor = torch.tensor(bg_sequences, dtype=torch.float32).to(DEVICE)
    bg_static_features_tensor = torch.tensor(bg_static_features, dtype=torch.float32).to(DEVICE)
    print(f"背景データ数: {len(bg_sequences)}")

    # SHAP Explainer 用の予測関数ラッパー
    seq_len = event_seq_instance.shape[0]
    event_dim = event_seq_instance.shape[1]

    def model_predict_proba_flat(flat_input_tensor_np):
        # SHAPはNumPy配列で渡してくることがあるのでTensorに変換
        if isinstance(flat_input_tensor_np, np.ndarray):
            flat_input_tensor = torch.tensor(flat_input_tensor_np, dtype=torch.float32).to(DEVICE)
        else: # すでにTensorの場合 (Explainerによる)
            flat_input_tensor = flat_input_tensor_np.to(DEVICE)

        batch_size = flat_input_tensor.shape[0]
        # SequenceとStaticに分割
        try:
            event_seq = flat_input_tensor[:, :(seq_len * event_dim)].reshape(batch_size, seq_len, event_dim)
            static_feat = flat_input_tensor[:, (seq_len * event_dim):]
        except Exception as e:
             print(f"[エラー] SHAPラッパー内でのテンソル再構成に失敗: {e}")
             print(f"  Input shape: {flat_input_tensor.shape}, Expected seq_flat: {seq_len * event_dim}, seq_len: {seq_len}, event_dim: {event_dim}")
             # ダミーの確率を返すなどしてエラー回避を試みる (デバッグ用)
             return np.zeros((batch_size,))


        # パディングマスク生成
        padding_code = 8 # GameState.EVENT_TYPES["PADDING"] と合わせる
        mask = (event_seq[:, :, 0] == padding_code) # device は event_seq に依存

        # モデル予測
        with torch.no_grad():
            outputs = model(event_seq, static_feat, mask)
            probabilities = F.softmax(outputs, dim=1)

        # 対象クラスの確率をNumpy配列で返す
        return probabilities[:, target_class_index].cpu().numpy()

    # SHAP Explainer の準備
    # KernelExplainerは時間がかかるため、背景データはサンプリングして使う
    bg_flat = np.concatenate([bg_sequences.reshape(bg_sequences.shape[0], -1), bg_static_features], axis=1)
    # instance も flat にしてバッチ次元追加
    instance_flat = np.concatenate([event_seq_instance.flatten(), static_feat_instance]).reshape(1, -1)

    # shap.sampleで背景データをサンプリング (KernelExplainerの計算量削減)
    n_bg_summary = min(50, len(bg_flat)) # 最大50サンプルで要約
    background_summary = shap.sample(bg_flat, n_bg_summary)
    print(f"SHAP背景データとして {n_bg_summary} サンプルをサンプリングしました。")

    # KernelExplainer 初期化
    try:
        print("SHAP KernelExplainer を初期化中...")
        explainer = shap.KernelExplainer(model_predict_proba_flat, background_summary)
    except Exception as e:
        print(f"[エラー] SHAP Explainer の初期化に失敗: {e}")
        return None

    # SHAP 値の計算 (n_shap_samplesで計算量を調整)
    print(f"SHAP値を計算中 (n_shap_samples={n_shap_samples})... これには時間がかかります...")
    try:
        shap_values = explainer.shap_values(instance_flat, nsamples=n_shap_samples) # nsamples で精度と速度を調整
    except Exception as e:
        print(f"[エラー] SHAP値の計算中にエラーが発生しました: {e}")
        # エラー時の入力形状などを表示
        print(f"  Explainer input shape: {instance_flat.shape}")
        print(f"  Background summary shape: {background_summary.shape}")
        return None

    calculation_time = time.time() - start_time
    print(f"SHAP値の計算完了 ({calculation_time:.2f} 秒)")

    # 特徴量の重要度を出力
    shap_values_flat = shap_values[0] # [1, num_features] -> [num_features]

    if len(feature_names) != len(shap_values_flat):
         print(f"[エラー] 特徴量名の数 ({len(feature_names)}) と SHAP値の数 ({len(shap_values_flat)}) が一致しません。generate_feature_names を確認してください。")
         # 特徴量名が合わない場合は、インデックスで出力
         feature_importance = sorted(enumerate(shap_values_flat), key=lambda x: abs(x[1]), reverse=True)
         print(f"\n影響の大きい特徴量 Top 15 (インデックスとSHAP値):")
         for i, (idx, value) in enumerate(feature_importance[:15]):
             print(f"  {i+1}. Feature_{idx}: {value:.4f}")
    else:
        # 特徴量名とSHAP値を紐付け
        feature_importance_dict = dict(zip(feature_names, shap_values_flat))
        # 絶対値でソート
        feature_importance_sorted = sorted(feature_importance_dict.items(), key=lambda item: abs(item[1]), reverse=True)

        print(f"\n影響の大きい特徴量 Top 15 (SHAP値):")
        for i, (name, value) in enumerate(feature_importance_sorted[:15]):
            print(f"  {i+1}. {name}: {value:.4f}")

    # SHAP Force Plot を保存 (matplotlibが必要)
    if shap_available:
        try:
            print("SHAP Force Plot を生成・保存中...")
            # matplotlib=True を使うと Figure オブジェクトが返る
            force_plot_fig = shap.force_plot(explainer.expected_value, shap_values[0], instance_flat[0], feature_names=feature_names, matplotlib=True, show=False)
            # 保存ファイル名を生成 (引数から取得できないのでグローバル参照または引数で渡す)
            # ここでは仮の名前を使う
            plot_filename = f"shap_force_plot_pred_{target_class_name}.png"
            plt.savefig(plot_filename, bbox_inches='tight')
            print(f"SHAP Force Plot を保存しました: {plot_filename}")
            plt.close(force_plot_fig) # プロットを閉じる
        except Exception as plot_e:
            print(f"[警告] SHAP プロットの生成または保存に失敗しました: {plot_e}")
            # traceback.print_exc() # 詳細デバッグ用

    # LLM入力用に整形したデータを返すことも可能
    # return feature_importance_sorted
    return feature_importance_dict # 辞書で返す例

# --- 局/自風 文字列取得関数 ---
def get_wind_str(round_num_wind, player_id, dealer):
    """局風と自風の文字列を返す"""
    # (変更なし)
    round_winds = ["東", "南", "西", "北"]
    player_winds = ["東", "南", "西", "北"]
    try:
        # 場風: 東=0, 南=1, 西=2, 北=3 (round_num_windから計算)
        round_wind_idx = round_num_wind // NUM_PLAYERS
        # 局数: 1-4 (round_num_windから計算)
        kyoku_num = (round_num_wind % NUM_PLAYERS) + 1
        # 自風: 東=0, 南=1, 西=2, 北=3 (player_idとdealerから計算)
        my_wind_idx = (player_id - dealer + NUM_PLAYERS) % NUM_PLAYERS
        return f"{round_winds[round_wind_idx]}{kyoku_num}局", player_winds[my_wind_idx]
    except IndexError:
        print(f"[警告] get_wind_str でIndexError発生: round_num_wind={round_num_wind}, player_id={player_id}, dealer={dealer}")
        return "不明局", "不明家"
    except TypeError:
         print(f"[警告] get_wind_str でTypeError発生: round_num_wind={round_num_wind}, player_id={player_id}, dealer={dealer}")
         return "不明局", "不明家"


# --- メイン処理 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学習済み麻雀Transformerモデルを使って打牌を予測し、SHAPで説明します。")
    parser.add_argument("xml_file", help="天鳳形式のXML牌譜ファイルへのパス")
    parser.add_argument("round_index", type=int, help="対象局のインデックス (1から開始)")
    parser.add_argument("tsumo_count", type=int, help="対象局内でのツモ回数 (1から開始)")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help=f"学習済みモデル (.pth) ファイルへのパス (デフォルト: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--shap_samples", type=int, default=100, help="SHAP値計算に使用するサンプル数 (KernelExplainer用、多いほど正確だが遅い)")
    parser.add_argument("--background_samples", type=int, default=100, help="SHAPの背景データとして使用するサンプル数")

    args = parser.parse_args()

    try:
        # 1. 指定局面までのGameStateを復元
        game_state, tsumo_info, discard_info = reconstruct_game_state_at_tsumo(
            args.xml_file, args.round_index, args.tsumo_count
        )

        # 予測を行うプレイヤーIDの決定
        player_id = game_state.current_player
        if tsumo_info and player_id != tsumo_info["player"]:
             print(f"[警告] GameStateのcurrent_player ({player_id}) とツモイベントのプレイヤー ({tsumo_info['player']}) が異なります。ツモイベントのプレイヤーを使用します。")
             player_id = tsumo_info["player"]
        elif not tsumo_info:
             print("[エラー] ツモイベント情報が見つかりません。局面復元に失敗した可能性があります。")
             exit()

        # 2. モデル入力次元数を取得 (予測するインスタンスの特徴量から)
        print("予測のための特徴量を生成中...")
        try:
            event_sequence_instance = game_state.get_event_sequence_features()
            static_features_instance = game_state.get_static_features(player_id)
            event_dim = event_sequence_instance.shape[1]
            static_dim = static_features_instance.shape[0]
            seq_len = event_sequence_instance.shape[0]
            print(f"特徴量次元: イベント次元={event_dim}, 静的次元={static_dim}, シーケンス長={seq_len}")
        except Exception as e:
            print(f"[エラー] 特徴量の生成に失敗しました: {e}")
            raise

        # 3. モデルをロード
        model = load_trained_model(args.model_path, event_dim, static_dim, seq_len)

        # 4. 打牌予測を実行
        print("打牌を予測中...")
        predicted_index, predicted_prob, all_probabilities = predict_discard(
            model, game_state, player_id
        )
        predicted_tile_str = tile_id_to_string(predicted_index * 4)
        print("予測完了。")

        # 5. 実際の捨て牌を取得
        actual_discard_str = "N/A (局終了？)"
        if discard_info:
            actual_discard_str = tile_id_to_string(discard_info["pai"])
            if discard_info["tsumogiri"]: actual_discard_str += "*"

        # 6. 結果を表示
        print("\n=== Transformer 予測テスト ===")
        print(f"--- 対象局面 (R:{args.round_index}, TsumoCount:{args.tsumo_count}) ---")
        print(f"牌譜ファイル: {os.path.basename(args.xml_file)}")
        round_str, my_wind_str = get_wind_str(game_state.round_num_wind, player_id, game_state.dealer)
        honba_str = f"{game_state.honba}本場"
        kyotaku_str = f"({game_state.kyotaku}供託)" if game_state.kyotaku > 0 else ""
        print(f"局: {round_str} {honba_str} {kyotaku_str} / プレイヤー: {player_id} ({my_wind_str}家)")
        tsumo_pai_str = tile_id_to_string(tsumo_info['pai']) if tsumo_info else "不明"
        print(f"ツモ牌: {tsumo_pai_str}")
        print(f"巡目: {game_state.junme:.2f}")
        print(f"ドラ表示: {' '.join([tile_id_to_string(t) for t in game_state.dora_indicators])}")
        print(f"点数: {[f'P{i}:{s}' for i, s in enumerate(game_state.current_scores)]}")
        # 向聴数を表示 (ダミーでない場合)
        try:
             hand_indices = game_state.get_hand_indices(player_id)
             melds_info = game_state.get_melds_indices(player_id)
             shanten, _ = calculate_shanten(hand_indices, melds_info)
             shanten_str = f"{shanten}向聴" if shanten >= 0 else "和了"
             print(f"向聴数 (推定): {shanten_str}")
        except Exception as shanten_e:
             print(f"向聴数の計算/表示中にエラー: {shanten_e}")

        print("--- 現在の状態 ---")
        print("手牌 (ツモ後):")
        for p in range(NUM_PLAYERS):
            hand_str = format_hand(game_state.player_hands[p])
            reach_indicator = "*" if game_state.player_reach_status[p] == 2 else ("(宣)" if game_state.player_reach_status[p] == 1 else "")
            print(f"  P{p}{reach_indicator}: {hand_str}")
        print("捨て牌:")
        for p in range(NUM_PLAYERS):
            discard_str = format_discards(game_state.player_discards[p])
            print(f"  P{p}: {discard_str}")
        print("鳴き:")
        for p in range(NUM_PLAYERS):
            meld_str = format_melds(game_state.player_melds[p])
            print(f"  P{p}: {meld_str}")
        print("-" * 20)
        print(f"予測された捨て牌 (牌種): {predicted_tile_str}")
        print(f"予測確率: {predicted_prob:.4f}")
        print(f"実際の捨て牌: {actual_discard_str}")
        print("-" * 20)
        top_n = 5
        indices_sorted = np.argsort(all_probabilities)[::-1]
        print(f"予測確率 Top {top_n}:")
        for i in range(top_n):
            idx = indices_sorted[i]
            prob = all_probabilities[idx]
            tile_str = tile_id_to_string(idx * 4)
            print(f"  {i+1}. {tile_str} ({prob:.4f})")

        # ★★★ SHAP説明生成の呼び出し ★★★
        if shap_available:
            try:
                # 背景データの準備
                print("\nSHAP説明のための背景データをロード中...")
                background_files = sorted(glob.glob(DATA_PATTERN))
                if not background_files:
                    print("[警告] SHAP用の背景データファイルが見つかりません。説明をスキップします。")
                else:
                    background_file = background_files[0] # 最初のファイルを使う
                    with np.load(background_file, allow_pickle=True) as bg_data:
                        n_bg = min(args.background_samples, len(bg_data['sequences'])) # 指定数 or 最大数
                        bg_sequences = bg_data['sequences'][:n_bg]
                        bg_static_features = bg_data['static_features'][:n_bg]
                    background_data = (bg_sequences, bg_static_features)
                    print(f"{len(bg_sequences)} 件の背景データをロードしました。")

                    # 説明対象インスタンス
                    instance_to_explain = (event_sequence_instance, static_features_instance, None)

                    # 特徴量名の生成 (★重要: 実装を確認・修正すること)
                    feature_names = generate_feature_names(event_dim, static_dim, seq_len)

                    # SHAP実行
                    feature_importance_dict = explain_prediction_with_shap(
                        model,
                        background_data,
                        instance_to_explain,
                        feature_names,
                        predicted_index, # 予測されたクラスを説明
                        n_shap_samples=args.shap_samples
                    )
                    # feature_importance_dict を使ってLLMに入力するなどの後続処理...

            except Exception as shap_e:
                print(f"\n[エラー] SHAP説明の生成中にエラーが発生しました: {shap_e}")
                import traceback
                traceback.print_exc()
        else:
            print("\nSHAPライブラリが利用できないため、説明は生成されません。")
        # ★★★ SHAP説明ここまで ★★★


    except FileNotFoundError as e: print(f"エラー: ファイルが見つかりません - {e}")
    except ValueError as e: print(f"エラー: 値が不正です - {e}")
    except ImportError as e: print(f"エラー: インポートに失敗しました - {e}")
    except AttributeError as e: print(f"エラー: 属性エラー（クラス定義やメソッド呼び出しを確認） - {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()