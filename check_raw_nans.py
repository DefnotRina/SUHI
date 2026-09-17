import geopandas as gpd
import pandas as pd
import numpy as np
from rasterstats import zonal_stats
from tqdm import tqdm
import os
import glob
from pathlib import Path
import rasterio

# Load Adaptive Node Coordinates
nodes_gdf = gpd.read_file("v2_adaptive_graph/davao_adaptive_nodes_utm.gpkg", layer="nodes")
raster_dir = Path("data/rasters")
raster_files = glob.glob(str(raster_dir / "*.tif"))
processed_cols = []

# Just process ONE raster to see why NaNs appear
raster_path = raster_files[0]
basename = os.path.basename(raster_path)
feature_year = basename.replace(".tif", "")
feature_col = f"{feature_year}_test"

nodes_buffer_gdf = nodes_gdf.copy()
nodes_buffer_gdf["geometry"] = nodes_buffer_gdf.geometry.buffer(100.0)
nodes_buffer_wgs84 = nodes_buffer_gdf.to_crs(epsg=4326)

with rasterio.open(raster_path) as src:
    affine = src.transform
    array = src.read(1)
    nodata = src.nodata if src.nodata is not None else -9999

chunks = []
chunk_size = 10000
for i in range(0, len(nodes_buffer_wgs84), chunk_size):
    chunks.append(nodes_buffer_wgs84.geometry.iloc[i:i+chunk_size])

stats = []
for chunk_geom in tqdm(chunks, total=len(chunks), desc=f"Processing {feature_col}"):
    geom_list = chunk_geom.to_list()
    chunk_stats = zonal_stats(geom_list, array, affine=affine, stats="median", nodata=nodata)
    stats.extend(chunk_stats)

nodes_gdf[feature_col] = [s["median"] if s["median"] is not None else np.nan for s in stats]

print("Total nodes:", len(nodes_gdf))
print("NaNs for this raster (100m buffer):", nodes_gdf[feature_col].isna().sum())

# Now try with 15m buffer
nodes_buffer_gdf["geometry"] = nodes_gdf.geometry.buffer(15.0)
nodes_buffer_wgs84 = nodes_buffer_gdf.to_crs(epsg=4326)
stats_15 = []
for i in range(0, len(nodes_buffer_wgs84), chunk_size):
    chunk_geom = nodes_buffer_wgs84.geometry.iloc[i:i+chunk_size]
    geom_list = chunk_geom.to_list()
    chunk_stats = zonal_stats(geom_list, array, affine=affine, stats="median", nodata=nodata)
    stats_15.extend(chunk_stats)

nodes_gdf[f"{feature_year}_15m"] = [s["median"] if s["median"] is not None else np.nan for s in stats_15]
print("NaNs for this raster (15m buffer):", nodes_gdf[f"{feature_year}_15m"].isna().sum())
