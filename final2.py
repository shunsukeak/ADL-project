# Re-import after kernel reset
from pyrosm import OSM
from shapely.geometry import box
import pandas as pd
import os
import time

# 単一災害・単一PBF用の処理関数（小分け保存対応）
def process_disaster_buildings(disaster, bbox_tuple, pbf_files, output_dir="./new/output/osm_extracted_csv_partial"):
    os.makedirs(output_dir, exist_ok=True)
    bbox_geom = box(*bbox_tuple)
    total_matched = 0

    for pbf_file in pbf_files:
        try:
            start = time.time()
            pbf_path = f"./new/osm/{pbf_file}"
            print(f"\n🚧 {disaster}: Loading {pbf_file}...", flush=True)
            osm = OSM(pbf_path)

            buildings = osm.get_buildings(custom_filter={"tags_as_columns": ["building:material"]})
            print(f"  🏗️ Total buildings in file: {len(buildings)}", flush=True)

            spatial_index = buildings.sindex
            possible_matches_index = list(spatial_index.intersection(bbox_geom.bounds))
            possible_matches = buildings.iloc[possible_matches_index]
            buildings_filtered = possible_matches[possible_matches.intersects(bbox_geom)]

            matched_count = len(buildings_filtered)
            total_matched += matched_count
            print(f"  ✅ Matched in bbox: {matched_count}", flush=True)

            # 小分けで保存
            out_file = os.path.join(output_dir, f"{disaster}_{os.path.splitext(pbf_file)[0]}.csv")
            buildings_filtered.to_csv(out_file, index=False)
            print(f"  💾 Saved to {out_file} in {time.time() - start:.2f}s", flush=True)

        except Exception as e:
            print(f"❌ Error processing {pbf_file}: {e}", flush=True)

    print(f"\n🎉 {disaster}: Total matched buildings across all PBFs: {total_matched}", flush=True)


# テスト例（メモリ負荷軽減のため1災害ずつにする）
test_disaster = "sunda-tsunami"
test_bbox = (105.8030, -6.7987, 105.8692, -6.1938)
test_pbf_files = ["sumatra-latest.osm.pbf", "java-latest.osm.pbf"]

process_disaster_buildings(test_disaster, test_bbox, test_pbf_files)

# "sunda-tsunami": {"bbox": (105.8030, -6.7987, 105.8692, -6.1938), "pbf": ["sumatra-latest.osm.pbf", "java-latest.osm.pbf"]},

