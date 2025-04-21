# generate_training_data.py
import os
import pandas as pd
from glob import glob

def find_pre_image_path(filename, image_root="/mnt/bigdisk/xbd/geotiffs"):
    pattern = os.path.join(image_root, "tier*", "images", filename)
    matches = glob(pattern)
    return matches[0] if matches else None

def create_training_dataset_from_enriched(enriched_dir, image_root, output_csv):
    all_records = []
    for fname in os.listdir(enriched_dir):
        if not fname.endswith("_enriched.csv"):
            continue
        disaster = fname.replace("_enriched.csv", "")
        csv_path = os.path.join(enriched_dir, fname)
        print(f"📥 Processing: {csv_path}")
        df = pd.read_csv(csv_path)
        df = df[df["subtype"].notnull()].copy()
        df["pre_image_filename"] = df["file"].str.replace("_post_disaster.json", "_pre_disaster.tif")
        df["pre_image_path"] = df["pre_image_filename"].apply(lambda x: find_pre_image_path(x, image_root))
        df = df[df["pre_image_path"].notnull()]
        if not df.empty:
            all_records.append(df)

    if all_records:
        full_df = pd.concat(all_records, ignore_index=True)

        label_map = {
            "no-damage": 0,
            "minor-damage": 1,
            "major-damage": 2,
            "destroyed": 3
        }
        full_df["label"] = full_df["subtype"].map(label_map)
        full_df = full_df[full_df["label"].notnull()].copy()
        full_df["label"] = full_df["label"].astype(int)
        full_df.to_csv(output_csv, index=False)
        print(f"✅ Saved: {output_csv}")
        return full_df
    else:
        print("❌ No usable data found.")
        return None

if __name__ == "__main__":
    create_training_dataset_from_enriched(
        enriched_dir="./enriched_all_buildings",
        image_root="/mnt/bigdisk/xbd/geotiffs",
        output_csv="./training_dataset_raw.csv"
    )
