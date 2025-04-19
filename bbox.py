from collections import defaultdict
from shapely import wkt
import json
import os
import geopandas as gpd

def compute_bboxes_from_xbd_labels(folder_path, margin_ratio=0.05):
    region_geoms = defaultdict(list)

    for fname in os.listdir(folder_path):
        if not fname.endswith(".json"):
            continue
        region = fname.split("_")[0]  # 災害名
        fpath = os.path.join(folder_path, fname)
        with open(fpath, "r") as f:
            data = json.load(f)
        features = data.get("features", {}).get("lng_lat", [])
        for feat in features:
            geom = wkt.loads(feat["wkt"])
            region_geoms[region].append(geom)

    # bbox生成
    region_bboxes = {}
    for region, geoms in region_geoms.items():
        gdf = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")
        minx, miny, maxx, maxy = gdf.total_bounds
        dx = (maxx - minx) * margin_ratio
        dy = (maxy - miny) * margin_ratio

        region_bboxes[region] = {
            "north": maxy + dy,
            "south": miny - dy,
            "east": maxx + dx,
            "west": minx - dx,
        }

    return region_bboxes

folder_path = "./new/data/geotiffs/tier1/labels"  # or tier1
region_bboxes = compute_bboxes_from_xbd_labels(folder_path)
print(region_bboxes)