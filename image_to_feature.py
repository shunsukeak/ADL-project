import os
import pandas as pd
from glob import glob

def find_pre_image_path(filename, image_root="/mnt/bigdisk/xbd/geotiffs"):
    pattern = os.path.join(image_root, "tier*", "images", filename)
    matches = glob(pattern)
    return matches[0] if matches else None

def create_training_dataset_from_matched(matched_dir, image_root, output_csv):
    all_records = []

    for fname in os.listdir(matched_dir):
        if not fname.endswith("_matched.csv"):
            continue
        disaster = fname.replace("_matched.csv", "")
        csv_path = os.path.join(matched_dir, fname)
        print(f"📥 Processing: {csv_path}")

        df = pd.read_csv(csv_path)

        # Step 1: post-disaster由来（subtypeがある行）だけ抽出
        df = df[df["subtype"].notnull()].copy()

        # Step 2: pre画像ファイル名の推定
        df["pre_image_filename"] = df["file"].str.replace("_post_disaster.json", "_pre_disaster.tif")

        # Step 3: 実際の画像パスを tier横断で探索
        df["pre_image_path"] = df["pre_image_filename"].apply(lambda x: find_pre_image_path(x, image_root))

        # Step 4: 画像があるものだけ残す
        df = df[df["pre_image_path"].notnull()]

        if df.empty:
            print(f"⚠️ No usable entries for {disaster}")
            continue

        all_records.append(df)

    if all_records:
        full_df = pd.concat(all_records, ignore_index=True)
        full_df.to_csv(output_csv, index=False)
        print(f"✅ Training dataset saved: {output_csv}")
        return full_df
    else:
        print("❌ No data found with valid subtype and pre-disaster image.")
        return None

# === 実行例 ===
if __name__ == "__main__":
    matched_dir = "./matching_full_attributes"
    image_root = "/mnt/bigdisk/xbd/geotiffs"
    output_csv = "./training_dataset_pre_image_only.csv"

    df = create_training_dataset_from_matched(matched_dir, image_root, output_csv)

# import os
# import pandas as pd

# def add_pre_image_path_column(df, image_dir, disaster_name):
#     # file名から pre-disaster 画像パスを生成
#     def get_pre_image_path(file_name):
#         base = file_name.replace("_pre_disaster.json", "_pre_disaster.tif")
#         full_path = os.path.join(image_dir, base)
#         return full_path if os.path.exists(full_path) else None

#     df["pre_image_path"] = df["file"].apply(get_pre_image_path)
#     missing = df["pre_image_path"].isnull().sum()
#     if missing > 0:
#         print(f"⚠️ {disaster_name}: {missing} entries missing pre-disaster image files")
#     return df

# def create_unified_dataframe(matched_dir, image_root_dirs, output_path):
#     all_records = []

#     for fname in os.listdir(matched_dir):
#         if not fname.endswith("_matched.csv"):
#             continue
#         disaster = fname.replace("_matched.csv", "")
#         csv_path = os.path.join(matched_dir, fname)

#         print(f"📥 Loading {csv_path}")
#         df = pd.read_csv(csv_path)

#         # tierを推定して画像ディレクトリを決定
#         found = False
#         for tier in ["tier1", "tier3"]:
#             image_dir = os.path.join(image_root_dirs, tier, "images")
#             sample_file = df["file"].iloc[0].replace("_pre_disaster.json", "_pre_disaster.tif")
#             sample_path = os.path.join(image_dir, sample_file)
#             if os.path.exists(sample_path):
#                 df = add_pre_image_path_column(df, image_dir, disaster)
#                 found = True
#                 break

#         if not found:
#             print(f"❌ Could not find image directory for {disaster}")
#             continue

#         all_records.append(df)

#     if all_records:
#         full_df = pd.concat(all_records, ignore_index=True)
#         full_df.to_csv(output_path, index=False)
#         print(f"✅ Saved unified dataset: {output_path}")
#         return full_df
#     else:
#         print("❌ No records processed.")
#         return None

# # === 実行 ===
# if __name__ == "__main__":
#     matched_dir = "./matching_full_attributes"
#     image_root_dirs = "/mnt/bigdisk/xbd/geotiffs"
#     output_path = "./unified_dataset_with_pre_images.csv"

#     df = create_unified_dataframe(matched_dir, image_root_dirs, output_path)

# import os
# import pandas as pd

# def add_image_path_column(df, image_dir, disaster_name):
#     # file名から画像パスを生成
#     def get_image_path(file_name):
#         base = file_name.replace("_post_disaster.json", "_post_disaster.png")
#         full_path = os.path.join(image_dir, base)
#         return full_path if os.path.exists(full_path) else None

#     df["image_path"] = df["file"].apply(get_image_path)
#     missing = df["image_path"].isnull().sum()
#     if missing > 0:
#         print(f"⚠️ {disaster_name}: {missing} entries missing image files")
#     return df

# def create_unified_dataframe(matched_dir, image_root_dirs, output_path):
#     all_records = []

#     for fname in os.listdir(matched_dir):
#         if not fname.endswith("_matched.csv"):
#             continue
#         disaster = fname.replace("_matched.csv", "")
#         csv_path = os.path.join(matched_dir, fname)

#         print(f"📥 Loading {csv_path}")
#         df = pd.read_csv(csv_path)

#         # tierを推定して画像ディレクトリを決定
#         found = False
#         for tier in ["tier1", "tier3"]:
#             image_dir = os.path.join(image_root_dirs, tier, "images")
#             sample_file = df["file"].iloc[0].replace("_post_disaster.json", "_post_disaster.png")
#             sample_path = os.path.join(image_dir, sample_file)
#             if os.path.exists(sample_path):
#                 df = add_image_path_column(df, image_dir, disaster)
#                 found = True
#                 break

#         if not found:
#             print(f"❌ Could not find image directory for {disaster}")
#             continue

#         all_records.append(df)

#     if all_records:
#         full_df = pd.concat(all_records, ignore_index=True)
#         full_df.to_csv(output_path, index=False)
#         print(f"✅ Saved unified dataset: {output_path}")
#         return full_df
#     else:
#         print("❌ No records processed.")
#         return None

# # === 実行 ===
# if __name__ == "__main__":
#     matched_dir = "./matching_full_attributes"
#     image_root_dirs = "/mnt/bigdisk/xbd/geotiffs"
#     output_path = "./unified_dataset_with_images.csv"

#     df = create_unified_dataframe(matched_dir, image_root_dirs, output_path)
