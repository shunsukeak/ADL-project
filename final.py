from pyrosm import OSM
from shapely.geometry import box
import geopandas as gpd
import os
import time
import pandas as pd

# 災害名とbbox、対応する1つ以上の.pbfファイルリスト（すでに定義済）
disaster_osm_map = {
    # "santa-rosa-wildfire": {"bbox": (-122.7905, 38.4006, -122.6312, 38.5670), "pbf": ["california-latest.osm.pbf"]},
    # "hurricane-michael": {"bbox": (-85.7899, 30.0668, -85.6007, 30.3344), "pbf": ["florida-latest.osm.pbf"]},
    # "hurricane-florence": {"bbox": (-79.1181, 33.5387, -77.7759, 34.9571), "pbf": ["north-carolina-latest.osm.pbf", "south-carolina-latest.osm.pbf"]},
    # "hurricane-harvey": {"bbox": (-95.6952, 29.4126, -95.3211, 30.0894), "pbf": ["texas-latest.osm.pbf"]},
    # "midwest-flooding": {"bbox": (-96.5895, 34.6265, -92.1030, 36.5297), "pbf": ["arkansas-latest.osm.pbf", "missouri-latest.osm.pbf", "kansas-latest.osm.pbf"]},
    # "socal-fire": {"bbox": (-118.9505, 34.0049, -118.7010, 34.1982), "pbf": ["california-latest.osm.pbf"]},
    # "joplin-tornado": {"bbox": (-94.5676, 37.0466, -94.4230, 37.0843), "pbf": ["missouri-latest.osm.pbf"]},
    # "lower-puna-volcano": {"bbox": (-154.9371, 19.4193, -154.8412, 19.5191), "pbf": ["hawaii-latest.osm.pbf"]},
    # "woolsey-fire": {"bbox": (-118.9960, 34.0196, -118.7576, 34.2415), "pbf": ["california-latest.osm.pbf"]},
    # "tuscaloosa-tornado": {"bbox": (-87.5648, 33.1583, -87.3877, 33.2778), "pbf": ["alabama-latest.osm.pbf"]},
    # "moore-tornado": {"bbox": (-97.5500, 35.3009, -97.4137, 35.3560), "pbf": ["oklahoma-latest.osm.pbf"]},
    
    # "mexico-earthquake": {"bbox": (-99.2329, 19.2392, -99.1362, 19.4326), "pbf": ["mexico-latest.osm.pbf"]},
    # "portugal-wildfire": {"bbox": (-8.3133, 39.7677, -8.0698, 40.0370), "pbf": ["portugal-latest.osm.pbf"]},
    # "nepal-flooding": {"bbox": (83.2554, 26.4630, 83.5837, 26.9925), "pbf": ["nepal-latest.osm.pbf"]},
    
    # "guatemala-volcano": {"bbox": (-90.8852, 14.3564, -90.8003, 14.4486), "pbf": ["guatemala-latest.osm.pbf"]},
    # "palu-tsunami": {"bbox": (119.7885, -1.0042, 119.9348, -0.6888), "pbf": ["sulawesi-latest.osm.pbf"]},
    "hurricane-matthew": {"bbox": (-74.1915, 18.1516, -73.4953, 18.6823), "pbf": ["haiti-and-domrep-latest.osm.pbf", "jamaica-latest.osm.pbf"]},
    # "sunda-tsunami": {"bbox": (105.8030, -6.7987, 105.8692, -6.1938), "pbf": ["sumatra-latest.osm.pbf", "java-latest.osm.pbf"]},
    # "pinery-bushfire": {"bbox": (138.6891, -34.5462, 138.9288, -34.2226), "pbf": ["australia-latest.osm.pbf"]},
}
print("Create dir", flush=True)
output_dir = "./new/output/osm_extracted_csv"
os.makedirs(output_dir, exist_ok=True)

# 各災害ごとに建物データを抽出してCSVに保存
for disaster, info in disaster_osm_map.items():
    print(f"\n🚧 Starting processing for: {disaster}", flush=True)
    start_time = time.time()
    try:
        bbox_geom = box(*info["bbox"])
        filtered_all = []

        for pbf_file in info["pbf"]:
            pbf_path = f"./new/osm/{pbf_file}"
            print(f"  📦 Loading {pbf_file}...", flush=True)
            osm = OSM(pbf_path)

            # 建物取得（指定タグ）
            buildings = osm.get_buildings(custom_filter={"tags_as_columns": ["building:material"]})
            print(f"    → Total buildings in file: {len(buildings)}", flush=True)

            # 空間インデックスでフィルタ
            spatial_index = buildings.sindex
            possible_matches_index = list(spatial_index.intersection(bbox_geom.bounds))
            possible_matches = buildings.iloc[possible_matches_index]
            buildings_filtered = possible_matches[possible_matches.intersects(bbox_geom)]

            print(f"    ✅ {len(buildings_filtered)} buildings matched in bbox", flush=True)
            filtered_all.append(buildings_filtered)

        # 災害単位で統合＆保存
        merged = pd.concat(filtered_all, ignore_index=True)
        out_path = os.path.join(output_dir, f"{disaster}_osm_buildings.csv")
        merged.to_csv(out_path, index=False)

        print(f"🎉 {disaster}: Saved {len(merged)} buildings to CSV in {time.time() - start_time:.2f} sec", flush=True)
    except Exception as e:
        print(f"❌ {disaster}: Error - {e}")