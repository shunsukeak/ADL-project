import geopandas as gpd
import json
import os
from shapely.geometry import shape

def load_xbd_labels_from_folder(folder_path):
    all_records = []

    for fname in os.listdir(folder_path):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(folder_path, fname)
        with open(fpath, "r") as f:
            data = json.load(f)

        features = data.get("features", {}).get("xy", [])  # 注意：ピクセル座標の場合あり

        for feature in features:
            prop = feature["properties"]
            geom = shape(feature["geometry"])
            record = {
                "geometry": geom,
                "subtype": prop.get("subtype"),
                "uid": prop.get("uid"),
                "file": fname
            }
            all_records.append(record)

    gdf = gpd.GeoDataFrame(all_records)
    gdf.set_crs(epsg=4326, inplace=True, allow_override=True)  # 念のためEPSG:4326設定
    return gdf

# パスを指定
folder_path = "./new/data/geotiffs/tier1/labels"

# 読み込み
xbd_all_gdf = load_xbd_labels_from_folder(folder_path)

# 確認
print(xbd_all_gdf.head())
print(xbd_all_gdf.iloc[0].geometry)
