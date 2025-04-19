import os
import pandas as pd

# CSVファイルが格納されているフォルダのパスを指定
folder_path = "./your_folder_name"  # 適宜変更してください

# 行数を数えるオプション
include_header = False  # ヘッダーを含めないなら False

# フォルダ内のCSVファイルを走査
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            if not include_header:
                line_count -= 1
            print(f"{filename}: {line_count} rows")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
