#!/opt/homebrew/Caskroom/miniforge/base/bin/python
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterstats import zonal_stats
from tqdm import tqdm
import os
import glob
from pathlib import Path

def main():
    print("==========================================")
    print("   MULTI-BUFFER EXTRACTION SCRIPT (YEARLY)")
    print("==========================================")
    
    # 1. Load Metric Node Coordinates from Script 1 (UTM Zone 51N - EPSG:32651)
    print("[*] Loading base intersection nodes from 'davao_nodes_utm.gpkg'...")
    base_nodes_gdf = gpd.read_file("davao_nodes_utm.gpkg", layer="nodes")
    print(f"[*] Loaded {len(base_nodes_gdf)} intersection nodes.")

    raster_dir = Path("data/rasters")
    
    # Find all downloaded TIFs
    # Expected format: LST_2023.tif, NDBI_2024.tif, etc.
    raster_files = glob.glob(str(raster_dir / "*.tif"))
    if not raster_files:
        print("[!] No raster files found in data/rasters/. Please run 01b_download_gee_rasters.py first.")
        return
        
    BUFFER_SIZES = [15, 20, 30, 40, 50, 100]

    for size in BUFFER_SIZES:
        print("\n==========================================")
        print(f"[*] Processing {size}-Meter Buffers")
        print("==========================================")
        
        nodes_gdf = base_nodes_gdf.copy()
        
        # 2. Draw Circular Buffer
        nodes_buffer_gdf = nodes_gdf.copy()
        nodes_buffer_gdf["geometry"] = nodes_buffer_gdf.geometry.buffer(float(size))
        
        # 3. Reproject to WGS84 for rasterstats
        nodes_buffer_wgs84 = nodes_buffer_gdf.to_crs(epsg=4326)

        processed_cols = []

        for raster_path in raster_files:
            # Extract feature name and year from filename (e.g., "LST_2023.tif")
            basename = os.path.basename(raster_path)
            feature_year = basename.replace(".tif", "") # e.g., "LST_2023"
            
            # New column format: LST_2023_100m
            feature_col = f"{feature_year}_{size}m"
            processed_cols.append(feature_col)
            
            print(f"[*] Extracting {feature_col}...")
            try:
                import rasterio
                
                # Load the raster into memory once to prevent GDAL segmentation faults
                with rasterio.open(raster_path) as src:
                    affine = src.transform
                    array = src.read(1)
                    nodata = src.nodata if src.nodata is not None else -9999
                
                chunks = []
                chunk_size = 5000
                for i in range(0, len(nodes_buffer_wgs84), chunk_size):
                    # Keep as GeoSeries here
                    chunks.append(nodes_buffer_wgs84.geometry.iloc[i:i+chunk_size])
                
                stats = []
                # Single-threaded execution
                for chunk_geom in tqdm(chunks, total=len(chunks), desc=f"Processing {feature_col}"):
                    # Explicitly convert GeoSeries to a list of shapely geometries to prevent Fiona segfaults
                    geom_list = chunk_geom.to_list()
                    chunk_stats = zonal_stats(geom_list, array, affine=affine, stats="median", nodata=nodata)
                    stats.extend(chunk_stats)
                
                nodes_gdf[feature_col] = [s["median"] if s["median"] is not None else np.nan for s in stats]
                
            except Exception as e:
                print(f"[!] Error: Raster file '{raster_path}' could not be processed ({e}).")
                print(f"[*] Skipping {feature_col} due to error.")
                nodes_gdf[feature_col] = np.nan

        # 4. Clean Missing Values (Drop nodes falling outside satellite raster bounds)
        initial_count = len(nodes_gdf)
        nodes_gdf = nodes_gdf.dropna(subset=processed_cols)
        print(f"\n[*] Retained {len(nodes_gdf)}/{initial_count} valid intersections after raster overlay for {size}m.")

        # 5. Save Output
        gpkg_out = f"davao_attributed_nodes_{size}m.gpkg"
        csv_out = f"davao_node_features_{size}m.csv"
        print(f"[*] Exporting final attributed node feature tables for {size}m...")
        nodes_gdf.to_file(gpkg_out, layer=f"nodes_{size}m", driver="GPKG")
        nodes_gdf.to_csv(csv_out, index=False)
        print(f"[+] Step Complete -> Saved: '{gpkg_out}' & '{csv_out}'")

    print("\n[+] ALL BUFFER SIZES PROCESSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
