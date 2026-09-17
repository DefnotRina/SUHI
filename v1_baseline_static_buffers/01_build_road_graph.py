import osmnx as ox
import pandas as pd
import geopandas as gpd
import numpy as np

# 1. Define Davao City Boundary
davao_query = "Davao City, Philippines"

print("[*] Downloading boundaries for Davao City...")
core_districts_gdf = ox.geocode_to_gdf(davao_query)
urban_core_polygon = core_districts_gdf.union_all()

print("[*] Downloading driveable road network (including private residential streets)...")
custom_filter = '["highway"]["area"!~"yes"]["highway"!~"cycleway|footway|path|pedestrian|steps|track|corridor|elevator|escalator|proposed|construction|bridleway|abandoned|platform|raceway"]'
G = ox.graph_from_polygon(urban_core_polygon, custom_filter=custom_filter, simplify=True)

# 2. Convert Graph to GeoDataFrames (Nodes V and Edges E)
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

# 3. Define Hierarchical Lookup Table (Philippine DPWH/LTO Standards)
# Maps OSM 'highway' classification -> default (lanes, speed_kmh, rank_weight)
HIGHWAY_LOOKUP = {
    "trunk": {"lanes": 4, "speed": 60.0, "rank": 4},
    "primary": {"lanes": 4, "speed": 50.0, "rank": 4},
    "primary_link": {"lanes": 2, "speed": 40.0, "rank": 3},
    "secondary": {"lanes": 2, "speed": 40.0, "rank": 3},
    "secondary_link": {"lanes": 2, "speed": 30.0, "rank": 2},
    "tertiary": {"lanes": 2, "speed": 30.0, "rank": 2},
    "tertiary_link": {"lanes": 1, "speed": 20.0, "rank": 1},
    "residential": {"lanes": 1, "speed": 20.0, "rank": 1},
    "unclassified": {"lanes": 1, "speed": 20.0, "rank": 1},
    "default": {"lanes": 1, "speed": 20.0, "rank": 1}
}

def impute_edge_attributes(row):
    """Hierarchically imputes missing maxspeed and lanes based on highway tag."""
    hw = row["highway"]
    # Handle list of tags if segment has multiple classifications
    if isinstance(hw, list):
        hw = hw[0]
    
    defaults = HIGHWAY_LOOKUP.get(hw, HIGHWAY_LOOKUP["default"])
    
    # Impute speed
    speed = row.get("maxspeed", np.nan)
    if isinstance(speed, list):
        speed = speed[0]
    if pd.isna(speed) or str(speed) == "nan":
        speed = defaults["speed"]
    else:
        try:
            speed = float(str(speed).replace(" km/h", "").replace("mph", "").strip())
        except ValueError:
            speed = defaults["speed"]
            
    # Impute lanes
    lanes = row.get("lanes", np.nan)
    if isinstance(lanes, list):
        lanes = lanes[0]
    if pd.isna(lanes) or str(lanes) == "nan":
        lanes = defaults["lanes"]
    else:
        try:
            lanes = float(str(lanes).replace(";", "")[0])
        except (ValueError, IndexError):
            lanes = defaults["lanes"]
            
    return pd.Series([speed, lanes, defaults["rank"]], index=["imputed_speed", "imputed_lanes", "highway_rank"])

print("[*] Applying hierarchical imputation to edge attributes...")
imputed_cols = edges_gdf.apply(impute_edge_attributes, axis=1)
edges_gdf = pd.concat([edges_gdf, imputed_cols], axis=1)

# 4. Calculate Structural Edge Capacity Weight (w_ij)
# w_ij = (highway_rank * imputed_lanes) / (length_in_meters + 1e-5)
edges_gdf["edge_weight"] = (edges_gdf["highway_rank"] * edges_gdf["imputed_lanes"]) / (edges_gdf["length"] + 1e-5)

# Save processed topological artifacts
nodes_gdf.to_crs(epsg=32651).to_file("davao_nodes_utm.gpkg", layer="nodes", driver="GPKG")
edges_gdf.to_crs(epsg=32651).to_file("davao_edges_utm.gpkg", layer="edges", driver="GPKG")
print("[+] Successfully generated: davao_nodes_utm.gpkg & davao_edges_utm.gpkg")