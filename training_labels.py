import pandas as pd

# ファイルパス
input_csv = "./training_dataset_pre_image_only.csv"
output_csv = "./training_dataset_with_labels.csv"

# ラベルマッピング定義
label_map = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3
}

# データ読み込み
df = pd.read_csv(input_csv)

# subtype → 数値ラベルへ変換
df["label"] = df["subtype"].map(label_map)

# 未分類ラベルがある場合は除外（または別処理）
df = df[df["label"].notnull()].copy()
df["label"] = df["label"].astype(int)

# 保存
df.to_csv(output_csv, index=False)
print(f"✅ Saved with numeric labels: {output_csv}")
