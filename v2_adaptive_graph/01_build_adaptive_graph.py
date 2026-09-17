"""
Step 1: Build Adaptive Graph
This script downloads the driveable road network for Davao City from OpenStreetMap (OSM).
Unlike standard OSM graphs that only have nodes at intersections, this script introduces
an 'Adaptive Gaps' interpolation technique. We physically project the map to UTM (meters)
and interpolate new nodes along road segments every 30 meters. We also apply DPWH standards
to assign buffer radii based on the road hierarchy (e.g., trunk, residential).
This creates a high-resolution spatial graph necessary for fine-grained ST-GCN analysis.
"""
import osmnx as ox
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# 1. Define Davao City Boundary
davao_query = "Davao City, Philippines"
print("[*] Downloading boundaries for Davao City...")
core_districts_gdf = ox.geocode_to_gdf(davao_query)
urban_core_polygon = core_districts_gdf.union_all()

print("[*] Downloading driveable road network...")
custom_filter = '["highway"]["area"!~"yes"]["highway"!~"cycleway|footway|path|pedestrian|steps|track|corridor|elevator|escalator|proposed|construction|bridleway|abandoned|platform|raceway"]'
G = ox.graph_from_polygon(urban_core_polygon, custom_filter=custom_filter, simplify=True)

# Project graph to UTM (meters) before doing length calculations and interpolations!
print("[*] Projecting graph to UTM...")
G_proj = ox.project_graph(G)

# 2. Convert Graph to GeoDataFrames
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_proj)

# 3. Define Hierarchical Lookup Table (Philippine DPWH/LTO Standards)
HIGHWAY_LOOKUP = {
    "trunk": {"buffer_radius": 15.0},
    "primary": {"buffer_radius": 15.0},
    "primary_link": {"buffer_radius": 10.0},
    "secondary": {"buffer_radius": 10.0},
    "secondary_link": {"buffer_radius": 7.5},
    "tertiary": {"buffer_radius": 7.5},
    "tertiary_link": {"buffer_radius": 5.0},
    "residential": {"buffer_radius": 5.0},
    "unclassified": {"buffer_radius": 5.0},
    "default": {"buffer_radius": 5.0}
}

def get_buffer_radius(hw_tag):
    if isinstance(hw_tag, list):
        hw_tag = hw_tag[0]
    return HIGHWAY_LOOKUP.get(hw_tag, HIGHWAY_LOOKUP["default"])["buffer_radius"]

edges_gdf["buffer_radius"] = edges_gdf["highway"].apply(get_buffer_radius)

# 4. Interpolate Nodes (Adaptive Gaps)
# Here we mathematically slice each road segment. Natural OSM nodes are only at intersections.
# By ensuring a node exists at least every 30m, we massively increase our spatial resolution
# for the subsequent raster feature extraction. We track intersection buffer radii to ensure
# nodes at junctions inherit the largest buffer of the intersecting roads.
print("[*] Interpolating nodes with Adaptive Gaps (min gap >= 30m)...")
GLOBAL_MIN_GAP = 30.0

new_nodes = []
# Ensure unique IDs for new nodes
node_id_counter = int(nodes_gdf.index.max()) + 1

# Dictionary to keep track of max buffer radius for intersection nodes
intersection_buffer_radii = {node_id: 0.0 for node_id in nodes_gdf.index}

for idx, edge in edges_gdf.iterrows():
    u, v, key = idx
    geom = edge.geometry
    length = edge["length"]
    radius = edge["buffer_radius"]
    
    # Update the buffer radius for the intersection nodes (u, v)
    intersection_buffer_radii[u] = max(intersection_buffer_radii[u], radius)
    intersection_buffer_radii[v] = max(intersection_buffer_radii[v], radius)
    
    # Interpolation Logic
    N = int(length // GLOBAL_MIN_GAP)
    
    if N >= 2:
        num_points = N - 1
        spacing = length / N
        
        for i in range(1, N):
            point_geom = geom.interpolate(i * spacing)
            new_nodes.append({
                "osmid": node_id_counter,
                "y": point_geom.y,
                "x": point_geom.x,
                "geometry": point_geom,
                "node_type": "interpolated",
                "buffer_radius": radius,
                "highway": edge["highway"]
            })
            node_id_counter += 1

# Create GeoDataFrame for new nodes
if new_nodes:
    interpolated_nodes_gdf = gpd.GeoDataFrame(new_nodes, crs=nodes_gdf.crs)
    interpolated_nodes_gdf.set_index("osmid", inplace=True)
else:
    interpolated_nodes_gdf = gpd.GeoDataFrame(columns=nodes_gdf.columns, crs=nodes_gdf.crs)

# Update intersection nodes with their max buffer radius and node_type
nodes_gdf["node_type"] = "intersection"
nodes_gdf["buffer_radius"] = nodes_gdf.index.map(intersection_buffer_radii)

# 5. Combine all nodes
print(f"[*] Original intersections: {len(nodes_gdf)}")
print(f"[*] Interpolated nodes added: {len(interpolated_nodes_gdf)}")

adaptive_nodes_gdf = pd.concat([nodes_gdf, interpolated_nodes_gdf])
print(f"[*] Total adaptive nodes: {len(adaptive_nodes_gdf)}")

# 6. Save Adaptive Graph
print("[*] Saving davao_adaptive_nodes_utm.gpkg...")
adaptive_nodes_gdf.to_file("davao_adaptive_nodes_utm.gpkg", layer="nodes", driver="GPKG")

print("[+] Successfully generated adaptive graph!")
