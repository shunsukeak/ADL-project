import os
import json
import rasterio
import pandas as pd
import numpy as np
from tqdm import tqdm
from shapely import wkt

# === Step 1: Collect polygon data without cropping ===
def collect_polygons_from_geojson(label_dirs, image_root, csv_path):
    records = []

    for label_dir in label_dirs:
        for fname in tqdm(os.listdir(label_dir), desc=f"Processing {label_dir}"):
            if not fname.endswith(".json"): continue
            fpath = os.path.join(label_dir, fname)
            pre_image = fname.replace("_post_disaster.json", "_pre_disaster.tif")
            image_path = None
            for tier in ["tier1", "tier3"]:
                candidate = os.path.join(image_root, tier, "images", pre_image)
                if os.path.exists(candidate):
                    image_path = candidate
                    break
            if not image_path: continue

            with open(fpath, "r") as f:
                data = json.load(f)

            for i, feature in enumerate(data.get("features", {}).get("lng_lat", [])):
                subtype = feature["properties"].get("subtype")
                if not subtype: continue
                polygon = feature["wkt"]

                records.append({
                    "image_path": image_path,
                    "label": subtype,
                    "geometry": polygon,
                    "file": fname  # For disaster extraction
                })

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved polygon dataset CSV: {csv_path}")

# === Step 2: Add geometry features ===
def add_shape_features(csv_path, output_csv):
    df = pd.read_csv(csv_path)
    areas, peris, ars, exts, convs = [], [], [], [], []
    for w in tqdm(df["geometry"], desc="Computing geometry features"):
        try:
            poly = wkt.loads(w)
            area = poly.area
            peri = poly.length
            bounds = poly.bounds
            w_, h_ = bounds[2]-bounds[0], bounds[3]-bounds[1]
            ar = w_ / h_ if h_ != 0 else 0
            ext = area / (w_ * h_) if w_*h_ != 0 else 0
            conv = area / poly.convex_hull.area if poly.convex_hull.area != 0 else 0
        except:
            area, peri, ar, ext, conv = [0]*5
        areas.append(area); peris.append(peri)
        ars.append(ar); exts.append(ext); convs.append(conv)

    df["area"] = areas
    df["perimeter"] = peris
    df["aspect_ratio"] = ars
    df["extent_ratio"] = exts
    df["convexity"] = convs

    label_map = {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3}
    df = df[df["label"].isin(label_map.keys())].copy()
    df["label"] = df["label"].map(label_map)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved with shape features: {output_csv}")

# === Main execution ===
label_dirs = ["/mnt/bigdisk/xbd/geotiffs/tier1/labels", "/mnt/bigdisk/xbd/geotiffs/tier3/labels"]
image_root = "/mnt/bigdisk/xbd/geotiffs"
raw_csv = "./nocrop_image_dataset_raw.csv"
final_csv = "./nocrop_image_dataset_with_shapes.csv"

collect_polygons_from_geojson(label_dirs, image_root, raw_csv)
add_shape_features(raw_csv, final_csv)
