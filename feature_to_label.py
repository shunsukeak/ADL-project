import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ファイル読み込み
input_csv = "./training_dataset_with_labels.csv"
output_csv = "./training_dataset_with_labels_and_features.csv"

# 属性列を指定
categorical_cols = ["building", "building:material", "building:levels"]

# データ読み込み
df = pd.read_csv(input_csv)

# 属性ごとにエンコード（欠損 or "yes" → "unknown"）
for col in categorical_cols:
    col_clean = col.replace(":", "_") + "_id"  # 例: building:material → building_material_id
    print(f"🔁 Encoding {col} → {col_clean}")

    values = df[col].fillna("unknown").astype(str)

    # 特別処理: "yes" → "unknown" （用途がわからない建物）
    if col == "building":
        values = values.replace("yes", "unknown")

    encoder = LabelEncoder()
    df[col_clean] = encoder.fit_transform(values)

# 保存
df.to_csv(output_csv, index=False)
print(f"✅ Saved encoded dataset: {output_csv}")

# import pandas as pd
# from sklearn.preprocessing import LabelEncoder

# # ファイル読み込み
# input_csv = "./training_dataset_with_labels.csv"
# output_csv = "./training_dataset_with_labels_and_features.csv"

# # 属性列を指定
# categorical_cols = ["building", "building:material", "building:levels"]

# # データ読み込み
# df = pd.read_csv(input_csv)

# # 属性ごとにエンコード（欠損 → "unknown" で対応）
# for col in categorical_cols:
#     col_clean = col.replace(":", "_") + "_id"  # 例: building:material → building_material_id
#     print(f"🔁 Encoding {col} → {col_clean}")

#     values = df[col].fillna("unknown").astype(str)
#     encoder = LabelEncoder()
#     df[col_clean] = encoder.fit_transform(values)

# # 保存
# df.to_csv(output_csv, index=False)
# print(f"✅ Saved encoded dataset: {output_csv}")
