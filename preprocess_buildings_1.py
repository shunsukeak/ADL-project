# preprocess_buildings.py
import os
import json
import pandas as pd
import geopandas as gpd
from shapely import wkt
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KDTree

def load_xbd_lnglat_geojson_from_files(json_files):
    records = []
    for fpath in json_files:
        with open(fpath, "r") as f:
            data = json.load(f)
        for feature in data.get("features", {}).get("lng_lat", []):
            props = feature["properties"]
            geom = wkt.loads(feature["wkt"])
            records.append({"geometry": geom, "subtype": props.get("subtype"),
                            "uid": props.get("uid"), "file": os.path.basename(fpath)})
    return gpd.GeoDataFrame(records, crs="EPSG:4326")

def load_osm_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["geometry"].notnull()]
    df["geometry"] = df["geometry"].apply(wkt.loads)
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

def match_buildings_by_centroid(xbd_gdf, osm_gdf, threshold=50):
    utm = xbd_gdf.estimate_utm_crs()
    xbd_proj, osm_proj = xbd_gdf.to_crs(utm), osm_gdf.to_crs(utm)
    xbd_proj["centroid"] = xbd_proj.geometry.centroid
    osm_proj["centroid"] = osm_proj.geometry.centroid
    matches = []
    for i, xbd_row in tqdm(xbd_proj.iterrows(), total=len(xbd_proj)):
        nearby = osm_proj[osm_proj["centroid"].distance(xbd_row["centroid"]) < threshold]
        if not nearby.empty:
            nearest = nearby.iloc[0].copy()
            record = {"xbd_index": i, "distance": xbd_row["centroid"].distance(nearest["centroid"])}
            for col in nearest.index:
                if col not in ["geometry", "centroid"]:
                    record[col] = nearest[col]
            matches.append(record)
    matches_df = pd.DataFrame(matches)
    for col in matches_df.columns:
        if col not in ["xbd_index", "distance"] and col not in xbd_gdf.columns:
            xbd_gdf[col] = None
    for _, row in matches_df.iterrows():
        for col in row.index:
            if col not in ["xbd_index", "distance"]:
                xbd_gdf.at[row["xbd_index"], col] = row[col]
    return xbd_gdf

def impute_missing_attributes(xbd_gdf, attr_cols, radius=50):
    utm = xbd_gdf.estimate_utm_crs()
    gdf_proj = xbd_gdf.to_crs(utm)
    gdf_proj["centroid"] = gdf_proj.geometry.centroid
    coords = np.array([(p.x, p.y) for p in gdf_proj["centroid"]])
    tree = KDTree(coords)
    unknown_summary = {col: 0 for col in attr_cols}
    for idx, row in gdf_proj.iterrows():
        for col in attr_cols:
            if pd.isna(row.get(col)) or row[col] in ["", "nan", None]:
                dists, indices = tree.query_radius([coords[idx]], r=radius, return_distance=True)
                neighbors = gdf_proj.iloc[indices[0]]
                neighbors = neighbors[neighbors[col].notnull() & (neighbors[col] != "")]
                if not neighbors.empty:
                    gdf_proj.at[idx, col] = neighbors[col].mode().iloc[0]
                else:
                    gdf_proj.at[idx, col] = "unknown"
                    unknown_summary[col] += 1
    for col in attr_cols:
        print(f"🧮 {col} unknown: {unknown_summary[col]} / {len(gdf_proj)}")
    return gdf_proj.to_crs("EPSG:4326")

def collect_label_files(tier1_dir, tier3_dir):
    disaster_files = defaultdict(list)
    for label_dir in [tier1_dir, tier3_dir]:
        for fname in os.listdir(label_dir):
            if fname.endswith(".json"):
                disaster = fname.split("_")[0]
                disaster_files[disaster].append(os.path.join(label_dir, fname))
    return disaster_files

def process_all_disasters(tier1_dir, tier3_dir, osm_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    attr_cols = ["building", "building:material", "building:levels"]
    for disaster, file_list in collect_label_files(tier1_dir, tier3_dir).items():
        osm_csv = os.path.join(osm_root, f"{disaster}_osm_buildings.csv")
        out_csv = os.path.join(output_dir, f"{disaster}_enriched.csv")
        if not os.path.exists(osm_csv):
            print(f"❌ No OSM for {disaster}")
            continue
        xbd_gdf = load_xbd_lnglat_geojson_from_files(file_list)
        osm_gdf = load_osm_csv(osm_csv)
        xbd_gdf = match_buildings_by_centroid(xbd_gdf, osm_gdf)
        xbd_gdf = impute_missing_attributes(xbd_gdf, attr_cols)
        xbd_gdf.to_csv(out_csv, index=False)
        print(f"✅ Saved: {out_csv}")

if __name__ == "__main__":
    process_all_disasters(
        tier1_dir="/mnt/bigdisk/xbd/geotiffs/tier1/labels",
        tier3_dir="/mnt/bigdisk/xbd/geotiffs/tier3/labels",
        osm_root="./output",
        output_dir="./enriched_all_buildings"
    )
