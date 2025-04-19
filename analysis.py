import pandas as pd
import geopandas as gpd
from shapely import wkt
import json
import os
from tqdm import tqdm
import osmnx as ox

# Region bboxes provided from xBD data
region_bboxes = {
    'santa-rosa-wildfire': {'north': 38.56701403116966, 'south': 38.40063298738872, 'east': -122.63115829413914, 'west': -122.79045192419426},
    'hurricane-michael': {'north': 30.334383232922356, 'south': 30.066786517802875, 'east': -85.60072244272999, 'west': -85.78988540425088},
    'hurricane-florence': {'north': 34.95707539921712, 'south': 33.53870924773025, 'east': -77.77591029780531, 'west': -79.11810973107005},
    'hurricane-harvey': {'north': 30.089398410595855, 'south': 29.412625123043245, 'east': -95.32114465814341, 'west': -95.69524363434186},
    'hurricane-matthew': {'north': 18.682200318406245, 'south': 18.151563627646805, 'east': -73.49529176841054, 'west': -74.19142669082677},
    'palu-tsunami': {'north': -0.6887863885851513, 'south': -1.0041525868560492, 'east': 119.93480445902304, 'west': 119.78853428418886},
    'midwest-flooding': {'north': 36.529747364228896, 'south': 34.626516250573545, 'east': -92.1029674468532, 'west': -96.58945768426712},
    'socal-fire': {'north': 34.19824968131678, 'south': 34.0048753257693, 'east': -118.70097240944928, 'west': -118.95048666819743},
    'mexico-earthquake': {'north': 19.432641404205015, 'south': 19.239247424130888, 'east': -99.13619069812175, 'west': -99.2329378919422},
    'guatemala-volcano': {'north': 14.448583619172032, 'south': 14.356357842403357, 'east': -90.80026315844066, 'west': -90.88521260205258},
    'portugal-wildfire': {'north': 40.036993475054274, 'south': 39.767650835656966, 'east': -8.069805337841093, 'west': -8.313299565079863},
    'joplin-tornado': {'north': 37.08431859731508, 'south': 37.046609809979955, 'east': -94.42297350270502, 'west': -94.56763974655401},
    'nepal-flooding': {'north': 26.99248911818632, 'south': 26.46301734675262, 'east': 83.58365883037462, 'west': 83.25542663180178},
    'pinery-bushfire': {'north': -34.22251710041457, 'south': -34.546180318087146, 'east': 138.92879454297557, 'west': 138.68908418012361},
    'lower-puna-volcano': {'north': 19.519117287989474, 'south': 19.419281611991504, 'east': -154.84115575286333, 'west': -154.93706758783745},
    'woolsey-fire': {'north': 34.24147833783975, 'south': 34.019606254093624, 'east': -118.75762045377567, 'west': -118.99602168935664},
    'tuscaloosa-tornado': {'north': 33.277777414113, 'south': 33.15834320524514, 'east': -87.38771101232207, 'west': -87.56480076795778},
    'moore-tornado': {'north': 35.35600884206, 'south': 35.30093667083599, 'east': -97.41370525918974, 'west': -97.54999473198279},
    'sunda-tsunami': {'north': -6.193790347034079, 'south': -6.798689670891762, 'east': 105.86924029068803, 'west': 105.80295060885096}
}

def get_osm_buildings_with_tags_bbox(north, south, east, west):
    tags = {
        "building": True,
        "building:material": True,
        "building:levels": True
    }
    bbox = (north, south, east, west)
    gdf = ox.features_from_bbox(bbox, tags)
    gdf = gdf[gdf.geom_type == "Polygon"]
    gdf = gdf[(gdf["building:material"].notnull()) | (gdf["building:levels"].notnull())]
    gdf = gdf.to_crs(epsg=4326)
    return gdf

# Retrieve stats on number of OSM buildings per region
results = []
for region, bbox in tqdm(region_bboxes.items(), desc="Fetching OSM for all regions"):
    try:
        osm_gdf = get_osm_buildings_with_tags_bbox(**bbox)
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

summary_df = pd.DataFrame(results)
print(summary_df)