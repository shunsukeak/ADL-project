from pyrosm import get_data, OSM
from shapely.geometry import box
import geopandas as gpd
import time

start_time = time.time()

# bbox定義
bbox = (-90.8853, 14.3564, -90.8003, 14.4486)
bbox_geom = box(*bbox)

# 必要な属性だけを指定して取得
osm = OSM("./new/osm/guatemala-latest.osm.pbf")
buildings = osm.get_buildings(custom_filter={
    'tags_as_columns': ['building:material', 'building:levels']
})

# フィルタ前の数を記録
buildings_count_before = len(buildings)
print(f"フィルタ前の建物数: {buildings_count_before}")

# 空間インデックスを使用したフィルタリング
spatial_index = buildings.sindex
possible_matches_index = list(spatial_index.intersection(bbox_geom.bounds))
possible_matches = buildings.iloc[possible_matches_index]
buildings_filtered = possible_matches[possible_matches.intersects(bbox_geom)]

# フィルタ後の数を記録
buildings_count_after = len(buildings_filtered)
print(f"フィルタ後の建物数: {buildings_count_after}")
print(f"削減率: {(1 - buildings_count_after/buildings_count_before) * 100:.2f}%")

print(f"処理時間: {time.time() - start_time:.2f}秒")
print(buildings_filtered.head())
print(buildings_filtered.columns)