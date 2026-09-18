#!/opt/homebrew/Caskroom/miniforge/base/bin/python
"""
Step 2: Extract Adaptive Features
This script takes the high-resolution nodes from Step 1 and overlays them onto 
satellite raster imagery (e.g., Land Surface Temperature). 
It applies a Multi-Scale extraction logic:
- Macro-Scale: Fixed 100m buffers for ambient temperature.
- Micro-Scale: Dynamic buffers (5-15m) based on road widths for optical/NDVI data.

Because raster data often has missing pixels (cloud cover, sensor errors), 
we implement a Spatial KD-Tree imputation step that fills missing (NaN) values 
by averaging the 3 nearest valid spatial neighbors.
"""
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterstats import zonal_stats
from tqdm import tqdm
import os
import glob
from pathlib import Path
import rasterio
from scipy.spatial import cKDTree

def main():
    print("==========================================")
    print("   ADAPTIVE MULTI-SCALE EXTRACTION (V2)")
    print("      WITH SPATIAL KD-TREE IMPUTATION     ")
    print("==========================================")
    
    # Make paths relative to the script directory
    script_dir = Path(__file__).parent
    gpkg_path = script_dir / "davao_adaptive_nodes_utm.gpkg"
    
    # 1. Load Adaptive Node Coordinates
    print(f"[*] Loading adaptive nodes from '{gpkg_path.name}'...")
    try:
        nodes_gdf = gpd.read_file(gpkg_path, layer="nodes")
        print(f"[*] Loaded {len(nodes_gdf)} nodes.")
    except Exception as e:
        print(f"[!] Error loading graph: {e}")
        return

    raster_dir = script_dir.parent / "data" / "rasters"
    
    # Find all downloaded TIFs
    raster_files = glob.glob(str(raster_dir / "*.tif"))
    if not raster_files:
        print(f"[!] No raster files found in {raster_dir}. Please ensure they are accessible.")
        return
        
    processed_cols = []

    for raster_path in raster_files:
        basename = os.path.basename(raster_path)
        feature_year = basename.replace(".tif", "") # e.g., "LST_2023"
        
        print("\n==========================================")
        print(f"[*] Processing Raster: {feature_year}")
        print("==========================================")
        
        # 2. Multi-Scale Buffering Logic
        nodes_buffer_gdf = nodes_gdf.copy()
        
        if "LST" in feature_year:
            # Macro-Scale for Thermal (Fixed 100m)
            print("[*] Applying MACRO-SCALE (Fixed 100m) buffer for LST...")
            nodes_buffer_gdf["geometry"] = nodes_buffer_gdf.geometry.buffer(100.0)
            feature_col = f"{feature_year}_100m"
        else:
            # Micro-Scale for Optical (Dynamic Buffer Radius from 5m to 15m)
            print("[*] Applying MICRO-SCALE (Dynamic DPWH Buffer) for Optical...")
            nodes_buffer_gdf["geometry"] = nodes_buffer_gdf.apply(
                lambda row: row.geometry.buffer(float(row["buffer_radius"])), axis=1
            )
            feature_col = f"{feature_year}_adaptive"
            
        processed_cols.append(feature_col)
        
        # 3. Reproject to WGS84 for rasterstats
        print("[*] Reprojecting buffers to WGS84...")
        nodes_buffer_wgs84 = nodes_buffer_gdf.to_crs(epsg=4326)

        print(f"[*] Extracting {feature_col}...")
        try:
            # Load the raster into memory once to prevent GDAL segmentation faults
            with rasterio.open(raster_path) as src:
                affine = src.transform
                array = src.read(1)
                nodata = src.nodata if src.nodata is not None else -9999
            
            # Chunking to handle 310k nodes without blowing up RAM
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
            
        except Exception as e:
            print(f"[!] Error: Raster file '{raster_path}' could not be processed ({e}).")
            print(f"[*] Skipping {feature_col} due to error.")
            nodes_gdf[feature_col] = np.nan

    # 4. Spatial KD-Tree Imputation (Fill NaNs with neighbors)
    # Rationale: Raw satellite rasters are notoriously noisy. When buffering nodes, we may
    # encounter pixels containing NaN. We use a scipy KD-Tree over the projected spatial coordinates
    # to find the 3 closest geographically valid nodes and mathematically impute the average.
    print("\n==========================================")
    print("[*] Performing Spatial KD-Tree Imputation...")
    print("==========================================")
    
    # We use the UTM X/Y coordinates (meters) for accurate spatial distance calculation
    coords = np.vstack((nodes_gdf.geometry.x, nodes_gdf.geometry.y)).T
    
    for col in processed_cols:
        nan_mask = nodes_gdf[col].isna()
        num_nans = nan_mask.sum()
        
        if num_nans > 0:
            print(f"[*] Imputing {num_nans} missing values for {col} using nearest neighbors...")
            
            # Get valid nodes
            valid_mask = ~nan_mask
            valid_coords = coords[valid_mask]
            valid_values = nodes_gdf.loc[valid_mask, col].values
            
            if len(valid_coords) == 0:
                print(f"[!] Critical Error: NO valid data found for {col}. Cannot impute.")
                continue
                
            # Build KDTree using valid coordinates
            tree = cKDTree(valid_coords)
            
            # Query the tree for the 3 nearest valid neighbors for each NaN node
            nan_coords = coords[nan_mask]
            distances, indices = tree.query(nan_coords, k=3)
            
            # Calculate the mean of the 3 nearest neighbors
            # (If k=3, indices is 2D array: [num_nans, 3])
            # Handle edge case where there are less than 3 valid nodes total
            if len(valid_coords) < 3:
                # If only 1 or 2 valid nodes exist, just use mean of all valid
                imputed_values = np.mean(valid_values)
                nodes_gdf.loc[nan_mask, col] = imputed_values
            else:
                imputed_values = np.mean(valid_values[indices], axis=1)
                nodes_gdf.loc[nan_mask, col] = imputed_values
        else:
            print(f"[*] {col} is fully intact (0 missing values).")

    # Verify no NaNs remain
    total_nans_remaining = nodes_gdf[processed_cols].isna().sum().sum()
    print(f"\n[*] Imputation complete. Total NaNs remaining in graph: {total_nans_remaining}")

    # 5. Save Output
    gpkg_out = script_dir / "davao_adaptive_node_features.gpkg"
    csv_out = script_dir / "davao_adaptive_node_features.csv"
    
    print(f"[*] Exporting final attributed node feature tables...")
    nodes_gdf.to_file(gpkg_out, layer="nodes_features", driver="GPKG")
    
    # Drop geometry for CSV output to save space
    csv_gdf = nodes_gdf.drop(columns=['geometry'])
    csv_gdf.to_csv(csv_out, index=True) # Keep osmid as index/column
    
    print(f"[+] Step Complete -> Saved: '{gpkg_out}' & '{csv_out}'")
    print(f"[+] Final Graph Size: {len(nodes_gdf)} nodes.")

if __name__ == "__main__":
    main()
