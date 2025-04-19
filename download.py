from pyrosm import get_data, OSM
from shapely.geometry import box
import geopandas as gpd

# ローカルに保存したPBFファイル
osm = OSM("./new/osm/guatemala-latest.osm.pbf")

# 欲しい範囲のbbox（例：nepal-flooding）
# bbox = (138.6891, -34.5462, 138.9288, -34.2226)  # (west, south, east, north)
# bbox_geom = box(*bbox)
# 建物のみ抽出
buildings = osm.get_buildings()

# building:material や building:levels があるものだけに絞る
# buildings = buildings[
#     (buildings["building:material"].notnull()) | 
#     (buildings["building:levels"].notnull())
# ]
# bbox = box(-60.5, -78.2, -60.1, -77.9)
print(buildings)
print(buildings.head())
print(buildings.columns)
bbox = box(-90.8853, 14.3564, -90.8003, 14.4486)
# {'north': 14.448583619172032, 'south': 14.356357842403357, 'east': -90.80026315844066, 'west': -90.88521260205258},
buildings = buildings[buildings.intersects(bbox)]

# bboxでフィルタ
# buildings = buildings[buildings.intersects(bbox_geom)]
print(buildings)
print(buildings.head())
print(buildings.columns)