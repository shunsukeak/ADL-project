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
# import json
# import pandas as pd
# import geopandas as gpd
# from shapely import wkt
# from tqdm import tqdm
# from collections import defaultdict

# # === xBD JSONファイル群から読み込み ===
# def load_xbd_lnglat_geojson_from_files(json_files):
#     all_records = []

#     for fpath in json_files:
#         fname = os.path.basename(fpath)
#         with open(fpath, "r") as f:
#             data = json.load(f)

#         features = data.get("features", {}).get("lng_lat", [])
#         for feature in features:
#             props = feature["properties"]
#             geom = wkt.loads(feature["wkt"])
#             all_records.append({
#                 "geometry": geom,
#                 "subtype": props.get("subtype"),
#                 "uid": props.get("uid"),
#                 "file": fname
#             })

#     gdf = gpd.GeoDataFrame(all_records, crs="EPSG:4326")
#     return gdf

# # === OSM CSV読み込み（building属性があるもの） ===
# def load_osm_csv(csv_path):
#     df = pd.read_csv(csv_path)
#     df = df[df["geometry"].notnull()]
#     df["geometry"] = df["geometry"].apply(wkt.loads)
#     gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
#     gdf = gdf[gdf.geom_type == "Polygon"]
#     return gdf

# # === 重心距離で1対1マッチング & 属性を付加 ===
# def match_buildings_by_centroid(xbd_gdf, osm_gdf, threshold=50):
#     utm_crs = xbd_gdf.estimate_utm_crs()
#     xbd_proj = xbd_gdf.to_crs(utm_crs)
#     osm_proj = osm_gdf.to_crs(utm_crs)

#     xbd_proj["centroid"] = xbd_proj.geometry.centroid
#     osm_proj["centroid"] = osm_proj.geometry.centroid

#     matches = []

#     for i, xbd_row in tqdm(xbd_proj.iterrows(), total=len(xbd_proj), desc="Matching buildings"):
#         xbd_centroid = xbd_row["centroid"]
#         nearby = osm_proj[osm_proj["centroid"].distance(xbd_centroid) < threshold]

#         if not nearby.empty:
#             nearest = nearby.iloc[0].copy()
#             record = {
#                 "xbd_index": i,
#                 "distance": xbd_centroid.distance(nearest["centroid"]),
#             }
#             for col in nearest.index:
#                 if col not in ["geometry", "centroid"]:
#                     record[col] = nearest[col]
#             matches.append(record)

#     matches_df = pd.DataFrame(matches)

#     for col in matches_df.columns:
#         if col not in ["xbd_index", "distance"] and col not in xbd_gdf.columns:
#             xbd_gdf[col] = None

#     for _, row in matches_df.iterrows():
#         for col in row.index:
#             if col not in ["xbd_index", "distance"]:
#                 xbd_gdf.at[row["xbd_index"], col] = row[col]

#     return xbd_gdf

# # === ラベルごとのファイルリスト収集 ===
# def collect_label_files_by_disaster(tier1_label_dir, tier3_label_dir):
#     disaster_files = defaultdict(list)
#     for label_dir in [tier1_label_dir, tier3_label_dir]:
#         if not os.path.isdir(label_dir):
#             continue
#         for fname in os.listdir(label_dir):
#             if not fname.endswith(".json"):
#                 continue
#             disaster = fname.split("_")[0]  # 例: hurricane-florence
#             full_path = os.path.join(label_dir, fname)
#             disaster_files[disaster].append(full_path)
#     return disaster_files

# # === 災害ごとに処理して結果を保存 ===
# def process_disasters_from_mixed_labels(tier1_label_dir, tier3_label_dir, osm_root, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     disaster_files = collect_label_files_by_disaster(tier1_label_dir, tier3_label_dir)

#     for disaster, file_list in disaster_files.items():
#         osm_csv = os.path.join(osm_root, f"{disaster}_osm_buildings.csv")
#         output_csv = os.path.join(output_dir, f"{disaster}_matched.csv")

#         if not os.path.exists(osm_csv):
#             print(f"❌ Skipping {disaster}: OSM CSV not found")
#             continue

#         print(f"\n📦 Processing disaster: {disaster}")
#         try:
#             xbd_gdf = load_xbd_lnglat_geojson_from_files(file_list)
#             osm_gdf = load_osm_csv(osm_csv)

#             print(f"🏗️ xBD buildings: {len(xbd_gdf)}, OSM buildings: {len(osm_gdf)}")

#             xbd_gdf = match_buildings_by_centroid(xbd_gdf, osm_gdf)

#             xbd_gdf.to_csv(output_csv, index=False)
#             print(f"✅ Matched result saved: {output_csv}")
#         except Exception as e:
#             print(f"⚠️ Error processing {disaster}: {e}")

# # === 実行 ===
# if __name__ == "__main__":
#     tier1_label_dir = "/mnt/bigdisk/xbd/geotiffs/tier1/labels"
#     tier3_label_dir = "/mnt/bigdisk/xbd/geotiffs/tier3/labels"
#     osm_root = "./output"
#     output_dir = "./matching_full_attributes"

#     process_disasters_from_mixed_labels(tier1_label_dir, tier3_label_dir, osm_root, output_dir)

# import os
# import json
# import pandas as pd
# import geopandas as gpd
# from shapely import wkt
# from tqdm import tqdm
# from collections import defaultdict

# def load_xbd_lnglat_geojson_from_files(json_files):
#     all_records = []

#     for fpath in json_files:
#         fname = os.path.basename(fpath)
#         with open(fpath, "r") as f:
#             data = json.load(f)

#         features = data.get("features", {}).get("lng_lat", [])
#         for feature in features:
#             props = feature["properties"]
#             geom = wkt.loads(feature["wkt"])
#             all_records.append({
#                 "geometry": geom,
#                 "subtype": props.get("subtype"),
#                 "uid": props.get("uid"),
#                 "file": fname
#             })

#     gdf = gpd.GeoDataFrame(all_records, crs="EPSG:4326")
#     return gdf

# def load_osm_csv(csv_path):
#     df = pd.read_csv(csv_path)
#     df = df[df["geometry"].notnull()]
#     df["geometry"] = df["geometry"].apply(wkt.loads)
#     gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
#     gdf = gdf[gdf.geom_type == "Polygon"]
#     return gdf

# def match_buildings_by_centroid(xbd_gdf, osm_gdf, threshold=50):
#     utm_crs = xbd_gdf.estimate_utm_crs()
#     xbd_proj = xbd_gdf.to_crs(utm_crs)
#     osm_proj = osm_gdf.to_crs(utm_crs)

#     xbd_proj["centroid"] = xbd_proj.geometry.centroid
#     osm_proj["centroid"] = osm_proj.geometry.centroid

#     matches = []
#     for i, xbd_row in tqdm(xbd_proj.iterrows(), total=len(xbd_proj), desc="Matching buildings"):
#         xbd_centroid = xbd_row["centroid"]
#         nearby = osm_proj[osm_proj["centroid"].distance(xbd_centroid) < threshold]
#         if not nearby.empty:
#             nearest = nearby.iloc[0]
#             matches.append({
#                 "xbd_index": i,
#                 "material": nearest["building:material"],
#                 "distance": xbd_centroid.distance(nearest["centroid"])
#             })

#     matches_df = pd.DataFrame(matches)
#     xbd_gdf["material"] = None
#     for _, row in matches_df.iterrows():
#         xbd_gdf.at[row["xbd_index"], "material"] = row["material"]

#     return xbd_gdf

# def summarize_damage_by_material(xbd_gdf):
#     summary = xbd_gdf.groupby(["subtype", "material"]).size().reset_index(name="count")
#     return summary

# def collect_label_files_by_disaster(tier1_label_dir, tier3_label_dir):
#     disaster_files = defaultdict(list)
#     for label_dir in [tier1_label_dir, tier3_label_dir]:
#         if not os.path.isdir(label_dir):
#             continue
#         for fname in os.listdir(label_dir):
#             if not fname.endswith(".json"):
#                 continue
#             disaster = fname.split("_")[0]  # e.g., hurricane-florence
#             full_path = os.path.join(label_dir, fname)
#             disaster_files[disaster].append(full_path)
#     return disaster_files

# def process_disasters_from_mixed_labels(tier1_label_dir, tier3_label_dir, osm_root, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     disaster_files = collect_label_files_by_disaster(tier1_label_dir, tier3_label_dir)

#     for disaster, file_list in disaster_files.items():
#         osm_csv = os.path.join(osm_root, f"{disaster}_osm_buildings.csv")
#         output_csv = os.path.join(output_dir, f"{disaster}_summary.csv")

#         if not os.path.exists(osm_csv):
#             print(f"❌ Skipping {disaster}: OSM CSV not found")
#             continue

#         print(f"\n📦 Processing disaster: {disaster}")
#         try:
#             xbd_gdf = load_xbd_lnglat_geojson_from_files(file_list)
#             osm_gdf = load_osm_csv(osm_csv)

#             print(f"🏗️ xBD buildings: {len(xbd_gdf)}, OSM buildings: {len(osm_gdf)}")

#             xbd_gdf = match_buildings_by_centroid(xbd_gdf, osm_gdf)
#             summary_df = summarize_damage_by_material(xbd_gdf)

#             summary_df.to_csv(output_csv, index=False)
#             print(f"✅ Summary saved: {output_csv}")
#         except Exception as e:
#             print(f"⚠️ Error processing {disaster}: {e}")

# # === 実行 ===
# if __name__ == "__main__":
#     tier1_label_dir = "/mnt/bigdisk/xbd/geotiffs/tier1/labels"
#     tier3_label_dir = "/mnt/bigdisk/xbd/geotiffs/tier3/labels"
#     osm_root = "./output"
#     output_dir = "./matching"

#     process_disasters_from_mixed_labels(tier1_label_dir, tier3_label_dir, osm_root, output_dir)
