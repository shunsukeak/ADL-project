import os
import pandas as pd

def add_image_path_column(df, image_dir, disaster_name):
    # file名から画像パスを生成
    def get_image_path(file_name):
        base = file_name.replace("_post_disaster.json", "_post_disaster.png")
        full_path = os.path.join(image_dir, base)
        return full_path if os.path.exists(full_path) else None

    df["image_path"] = df["file"].apply(get_image_path)
    missing = df["image_path"].isnull().sum()
    if missing > 0:
        print(f"⚠️ {disaster_name}: {missing} entries missing image files")
    return df

def create_unified_dataframe(matched_dir, image_root_dirs, output_path):
    all_records = []

    for fname in os.listdir(matched_dir):
        if not fname.endswith("_matched.csv"):
            continue
        disaster = fname.replace("_matched.csv", "")
        csv_path = os.path.join(matched_dir, fname)

        print(f"📥 Loading {csv_path}")
        df = pd.read_csv(csv_path)

        # tierを推定して画像ディレクトリを決定
        found = False
        for tier in ["tier1", "tier3"]:
            image_dir = os.path.join(image_root_dirs, tier, "images")
            sample_file = df["file"].iloc[0].replace("_post_disaster.json", "_post_disaster.png")
            sample_path = os.path.join(image_dir, sample_file)
            if os.path.exists(sample_path):
                df = add_image_path_column(df, image_dir, disaster)
                found = True
                break

        if not found:
            print(f"❌ Could not find image directory for {disaster}")
            continue

        all_records.append(df)

    if all_records:
        full_df = pd.concat(all_records, ignore_index=True)
        full_df.to_csv(output_path, index=False)
        print(f"✅ Saved unified dataset: {output_path}")
        return full_df
    else:
        print("❌ No records processed.")
        return None

# === 実行 ===
if __name__ == "__main__":
    matched_dir = "./matching_full_attributes"
    image_root_dirs = "/mnt/bigdisk/xbd/geotiffs"
    output_path = "./unified_dataset_with_images.csv"

    df = create_unified_dataframe(matched_dir, image_root_dirs, output_path)
