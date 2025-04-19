
import geopandas as gpd
import pandas as pd
import osmnx as ox
from tqdm import tqdm

# ✅ このコードの目的：
# 各災害エリア（xBD）に対応する地理範囲を分割（タイル化）し、
# OSMから建物データを取得。
# その中で `building:material` または `building:levels` タグが付いている建物をカウントする。

# 🔧 bbox を tile_size 度ごとに分割（デフォルトは 0.2 度 ≒ 約20km 四方）
def subdivide_bbox(north, south, east, west, tile_size=0.2):
    lat_steps = int((north - south) / tile_size) + 1
    lon_steps = int((east - west) / tile_size) + 1
    tiles = []
    for i in range(lat_steps):
        for j in range(lon_steps):
            n = min(north, south + (i + 1) * tile_size)
            s = south + i * tile_size
            e = min(east, west + (j + 1) * tile_size)
            w = west + j * tile_size
            tiles.append((n, s, e, w))
    return tiles

# 📦 各タイルごとに OSM 建物データを取得 → material/levels タグを持つ建物を抽出
def get_osm_buildings_by_tiles(region_name, bbox, tile_size=0.2):
    tags = {
        "building": True  # OR 条件を避けるため building だけ指定
    }
    tiles = subdivide_bbox(**bbox, tile_size=tile_size)
    gdfs = []
    for n, s, e, w in tqdm(tiles, desc=f"Fetching tiles for {region_name}", leave=False):
        try:
            gdf = ox.geometries_from_bbox(n, s, e, w, tags)
            if gdf.empty:
                continue
            gdf = gdf[gdf.geom_type == "Polygon"]
            gdf = gdf[(gdf["building:material"].notnull()) | (gdf["building:levels"].notnull())]
            gdfs.append(gdf)
        except Exception:
            continue
    if gdfs:
        merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
        if "osmid" in merged.columns:
            merged = merged.drop_duplicates(subset="osmid")
        else:
            merged = merged.drop_duplicates(subset="geometry")
        return merged
    else:
        return gpd.GeoDataFrame(columns=["geometry", "building:material", "building:levels"])

# 🌍 地域ごとの bbox（xBD 災害ごと）
region_bboxes = {
    'santa-rosa-wildfire': {'north': 38.5670, 'south': 38.4006, 'east': -122.6311, 'west': -122.7904},
    'midwest-flooding': {'north': 36.5297, 'south': 34.6265, 'east': -92.1029, 'west': -96.5894},
    'nepal-flooding': {'north': 26.9924, 'south': 26.4630, 'east': 83.5836, 'west': 83.2554},
}

# 🚀 各地域について OSM 建物データ取得＆集計
results = []
for region, bbox in tqdm(region_bboxes.items(), desc="Fetching OSM data (tiled)"):
    try:
        osm_gdf = get_osm_buildings_by_tiles(region, bbox, tile_size=0.2)
        results.append({
            "region": region,
            "osm_buildings_with_material_or_levels": len(osm_gdf),
            "material_count": osm_gdf["building:material"].notnull().sum(),
            "levels_count": osm_gdf["building:levels"].notnull().sum()
        })
    except Exception as e:
        results.append({
            "region": region,
            "osm_buildings_with_material_or_levels": f"Error: {str(e)}",
            "material_count": "Error",
            "levels_count": "Error"
        })

# 📊 結果を表示
summary_df = pd.DataFrame(results)
print(summary_df)