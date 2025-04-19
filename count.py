import os
import json
from collections import defaultdict

def count_buildings_by_region(folder_path):
    region_counts = defaultdict(int)

    for fname in os.listdir(folder_path):
        if not fname.endswith(".json"):
            continue

        region = fname.split("_")[0]  # 例: 'earthquake-nepal' or 'santa-rosa-wildfire'
        fpath = os.path.join(folder_path, fname)

        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            features = data.get("features", {}).get("lng_lat", [])
            region_counts[region] += len(features)
        except Exception as e:
            print(f"[!] Failed to read {fname}: {e}")

    return dict(region_counts)

# フォルダパスを指定
folder_path = "./new/data/geotiffs/tier3/labels"
region_stats = count_buildings_by_region(folder_path)

# ソートして表示
for region, count in sorted(region_stats.items(), key=lambda x: -x[1]):
    print(f"{region:<25} {count:>6} buildings")
