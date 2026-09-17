import geopandas as gpd
import pandas as pd

# Load GPKG
gdf = gpd.read_file('v2_adaptive_graph/davao_adaptive_node_features.gpkg', layer='nodes_features')
print(f"Total nodes in GPKG: {len(gdf)}")
print("Node types in GPKG:")
print(gdf['node_type'].value_counts(dropna=False))

# Load original nodes GPKG before extraction
gdf_orig = gpd.read_file('v2_adaptive_graph/davao_adaptive_nodes_utm.gpkg', layer='nodes')
print(f"\nTotal nodes in orig GPKG: {len(gdf_orig)}")
print("Node types in orig GPKG:")
print(gdf_orig['node_type'].value_counts(dropna=False))

