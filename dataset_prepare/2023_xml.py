import os
import shutil
import glob

def copy_2023_xml_files(source_dir, target_dir):
    """
    source_dirから'2023'で始まるファイルを見つけ、target_dirにコピーする。
    target_dirが存在しない場合は作成する。
    """
    if not os.path.exists(source_dir):
        print(f"エラー: ソースディレクトリ '{source_dir}' が見つかりません。")
        return

    if not os.path.exists(target_dir):
        print(f"ターゲットディレクトリ '{target_dir}' を作成します。")
        os.makedirs(target_dir)

    # '2023'で始まり、任意の拡張子を持つファイルを検索
    # XMLファイルに限定する場合は "*.xml" のようにパターンを調整可能
    search_pattern = os.path.join(source_dir, "2023*")
    copied_files_count = 0

    for file_path in glob.glob(search_pattern):
        if os.path.isfile(file_path):
            filename = os.path.basename(file_path)
            destination_path = os.path.join(target_dir, filename)
            
            try:
                print(f"コピー中: '{file_path}' -> '{destination_path}'")
                shutil.copy2(file_path, destination_path) # copy2はメタデータも保持
                copied_files_count += 1
            except Exception as e:
                print(f"エラー: '{filename}' のコピーに失敗しました - {e}")
    
    print(f"処理完了。{copied_files_count} 個のファイルをコピーしました。")

if __name__ == "__main__":
    # 指定されたディレクトリパス
    # Windows環境でパスを扱う場合は raw文字列 (r"...") や os.path.join を推奨
    source_directory = "/home/ubuntu/Documents/XML/xml_logs"
    target_directory = "/home/ubuntu/Documents/xml_logs" # ユーザーの指示通り

    copy_2023_xml_files(source_directory, target_directory)
