# === 全体構成 ===
# 1. xBDのGeoJSON + pre-image から建物画像をcropして保存
# 2. geometry featuresを抽出して結合
# 3. CSVとして image_path + label + features を生成
# 4. PyTorch Dataset/Modelを構成して学習開始

# crop_building_images.py
import os
import json
import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd
from shapely import wkt
from shapely.geometry import shape, mapping
import numpy as np
from tqdm import tqdm
import pandas as pd

# === Utility: crop polygon area from tif ===
def crop_polygon_from_image(image_path, polygon, out_path):
    with rasterio.open(image_path) as src:
        try:
            out_image, out_transform = rasterio.mask.mask(src, [mapping(polygon)], crop=True)
            out_meta = src.meta.copy()

            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            with rasterio.open(out_path, "w", **out_meta) as dest:
                dest.write(out_image)
            return True
        except Exception as e:
            print(f"⚠️ Error cropping: {e}")
            return False

# === Step 1: Crop building patches ===
def extract_and_crop_all(label_dirs, image_root, output_dir, csv_path):
    os.makedirs(output_dir, exist_ok=True)
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
                polygon = wkt.loads(feature["wkt"])
                out_file = f"{fname.replace('.json','')}_{i}.tif"
                out_path = os.path.join(output_dir, out_file)

                success = crop_polygon_from_image(image_path, polygon, out_path)
                if success:
                    records.append({
                        "image_path": out_path,
                        "label": subtype,
                        "geometry": feature["wkt"]
                    })

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved dataset CSV: {csv_path}")

# === Step 2: Add geometry features ===
from shapely import wkt

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

# === 実行例 ===
if __name__ == "__main__":
    label_dirs = ["/mnt/bigdisk/xbd/geotiffs/tier1/labels", "/mnt/bigdisk/xbd/geotiffs/tier3/labels"]
    image_root = "/mnt/bigdisk/xbd/geotiffs"
    output_crop_dir = "/mnt/bigdisk/xbd/geotiffs/cropped_images"
    raw_csv = "./cropped_dataset_raw.csv"
    final_csv = "./cropped_dataset_with_shapes.csv"

    extract_and_crop_all(label_dirs, image_root, output_crop_dir, raw_csv)
    add_shape_features(raw_csv, final_csv)
