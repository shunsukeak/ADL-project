import os
import json
import pandas as pd
from shapely import wkt
from glob import glob

def find_pre_image_path(filename, image_root="/mnt/bigdisk/xbd/geotiffs"):
    pattern = os.path.join(image_root, "tier*", "images", filename)
    matches = glob(pattern)
    return matches[0] if matches else None

def load_xbd_lnglat_geojson_from_files(json_files):
    records = []
    for fpath in json_files:
        with open(fpath, "r") as f:
            data = json.load(f)
        for feature in data.get("features", {}).get("lng_lat", []):
            props = feature["properties"]
            records.append({
                "geometry": feature["wkt"],
                "subtype": props.get("subtype"),
                "uid": props.get("uid"),
                "file": os.path.basename(fpath)
            })
    return pd.DataFrame(records)

def collect_label_files(tier1_label_dir, tier3_label_dir):
    all_files = []
    for label_dir in [tier1_label_dir, tier3_label_dir]:
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if fname.endswith(".json"):
                all_files.append(os.path.join(label_dir, fname))
    return all_files

def create_training_dataset_from_geojson(label_dirs, image_root, output_csv):
    json_files = collect_label_files(*label_dirs)
    print(f"📥 Found {len(json_files)} label files.")

    df = load_xbd_lnglat_geojson_from_files(json_files)
    df = df[df["subtype"].notnull()].copy()
    df["pre_image_filename"] = df["file"].str.replace("_post_disaster.json", "_pre_disaster.tif")
    df["pre_image_path"] = df["pre_image_filename"].apply(lambda x: find_pre_image_path(x, image_root))
    df = df[df["pre_image_path"].notnull()].copy()

    label_map = {
        "no-damage": 0,
        "minor-damage": 1,
        "major-damage": 2,
        "destroyed": 3
    }
    df["label"] = df["subtype"].map(label_map)
    df = df[df["label"].notnull()].copy()
    df["label"] = df["label"].astype(int)

    df.to_csv(output_csv, index=False)
    print(f"✅ Saved: {output_csv}")
    return df

if __name__ == "__main__":
    label_dirs = (
        "/mnt/bigdisk/xbd/geotiffs/tier1/labels",
        "/mnt/bigdisk/xbd/geotiffs/tier3/labels"
    )
    image_root = "/mnt/bigdisk/xbd/geotiffs"
    output_csv = "training_dataset_image_only.csv"

    create_training_dataset_from_geojson(label_dirs, image_root, output_csv)