import os
import json
import pandas as pd
from shapely import wkt
from shapely.geometry import Polygon
from glob import glob
from tqdm import tqdm

def compute_geometry_features(wkt_str):
    try:
        poly = wkt.loads(wkt_str)
        area = poly.area
        perimeter = poly.length
        bounds = poly.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        aspect_ratio = width / height if height != 0 else 0
        extent_ratio = area / (width * height) if width * height != 0 else 0
        convexity = area / poly.convex_hull.area if poly.convex_hull.area != 0 else 0
        return area, perimeter, aspect_ratio, extent_ratio, convexity
    except Exception:
        return [None] * 5

def extract_from_geojson(label_dirs, output_csv):
    records = []
    for label_dir in label_dirs:
        for fname in tqdm(os.listdir(label_dir), desc=f"Processing {label_dir}"):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(label_dir, fname)
            with open(fpath, "r") as f:
                data = json.load(f)
            for feature in data.get("features", {}).get("lng_lat", []):
                props = feature.get("properties", {})
                subtype = props.get("subtype")
                wkt_str = feature.get("wkt")
                if subtype and wkt_str:
                    area, peri, ar, ext, conv = compute_geometry_features(wkt_str)
                    records.append({
                        "file": fname,
                        "subtype": subtype,
                        "geometry": wkt_str,
                        "area": area,
                        "perimeter": peri,
                        "aspect_ratio": ar,
                        "extent_ratio": ext,
                        "convexity": conv
                    })
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved: {output_csv}")
    return df

if __name__ == "__main__":
    tier1_dir = "/mnt/bigdisk/xbd/geotiffs/tier1/labels"
    tier3_dir = "/mnt/bigdisk/xbd/geotiffs/tier3/labels"
    output_csv = "xbd_shape_features.csv"
    extract_from_geojson([tier1_dir, tier3_dir], output_csv)