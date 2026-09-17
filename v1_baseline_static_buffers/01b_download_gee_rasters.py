import ee
import osmnx as ox
import os
import requests
import zipfile
import io
from pathlib import Path

# Set up project
PROJECT_ID = "suhi-505907"

print("[*] Initializing Google Earth Engine...")
try:
    ee.Initialize(project=PROJECT_ID)
except Exception as e:
    print(f"[!] Initialization failed. Make sure you ran 'earthengine authenticate' in the terminal. Error: {e}")
    exit(1)

# 1. Define Davao City Region
print("[*] Getting Davao City boundaries...")
davao_gdf = ox.geocode_to_gdf("Davao City, Philippines")
davao_bounds = davao_gdf.total_bounds # [minx, miny, maxx, maxy]

# Create an EE geometry (bbox)
region = ee.Geometry.Rectangle([davao_bounds[0], davao_bounds[1], davao_bounds[2], davao_bounds[3]])

# Setup output directory
OUTPUT_DIR = Path("data/rasters")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Define Helper Functions for Landsat and ECOSTRESS
def apply_scale_factors_landsat(image):
    optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
    return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)

def mask_l8_clouds(image):
    qa = image.select('QA_PIXEL')
    cloud_shadow_bit_mask = 1 << 3
    clouds_bit_mask = 1 << 4
    mask = qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0).And(qa.bitwiseAnd(clouds_bit_mask).eq(0))
    return image.updateMask(mask)

def compute_ndvi(image):
    return image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')

def compute_ndbi(image):
    return image.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')

def process_landsat(year):
    start_date = f"{year}-02-01"
    end_date = f"{year}-04-30"
    
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(region).filterDate(start_date, end_date)
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(region).filterDate(start_date, end_date)
    
    # Merge, mask clouds, scale, and compute median
    merged = l8.merge(l9).map(mask_l8_clouds).map(apply_scale_factors_landsat)
    
    ndvi = merged.map(compute_ndvi).median().clip(region)
    ndbi = merged.map(compute_ndbi).median().clip(region)
    return ndvi, ndbi

def process_ecostress(year):
    start_date = f"{year}-02-01"
    end_date = f"{year}-04-30"
    
    eco = ee.ImageCollection("NASA/ECOSTRESS/L2T_LSTE/V2") \
        .filterBounds(region) \
        .filterDate(start_date, end_date)
        
    # V2 LST is already in Kelvin, just convert to Celsius
    def scale_lst(image):
        return image.select('LST').subtract(273.15).rename('LST')
        
    lst = eco.map(scale_lst).median().clip(region)
    return lst

import time

def download_image(image, filename, scale=30):
    print(f"[*] Requesting download URL for {filename}...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            url = image.getDownloadURL({
                'scale': scale,
                'crs': 'EPSG:4326',
                'region': region,
                'format': 'GEO_TIFF'
            })
            print(f"[*] Downloading to {filename} (Attempt {attempt+1}/{max_retries})...")
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                content_type = r.headers.get('content-type', '')
                if 'zip' in content_type:
                    z = zipfile.ZipFile(io.BytesIO(r.content))
                    z.extractall(OUTPUT_DIR)
                    extracted_file = z.namelist()[0]
                    os.rename(OUTPUT_DIR / extracted_file, OUTPUT_DIR / filename)
                else:
                    with open(OUTPUT_DIR / filename, 'wb') as f:
                        for chunk in r.iter_content(1024):
                            f.write(chunk)
                print(f"[+] Download complete: {filename}")
                return # Success, exit function
            else:
                print(f"[!] Failed to download. Status code: {r.status_code}. Msg: {r.text}")
                if r.status_code == 503:
                    print(f"[*] Transient error (503). Retrying in 10 seconds...")
                    time.sleep(10)
                else:
                    break # Don't retry on other errors (like 404)
        except Exception as e:
            print(f"[!] Error downloading {filename}: {e}")
            print(f"[*] Retrying in 10 seconds...")
            time.sleep(10)
    print(f"[!] Exhausted retries for {filename}.")

# 3. Main Download Loop
years = [2023, 2024, 2025]

for y in years:
    print(f"\n================ Processing Year {y} (Feb-Apr) ================")
    ndvi, ndbi = process_landsat(y)
    lst = process_ecostress(y)
    
    download_image(lst, f"LST_{y}.tif")
    download_image(ndbi, f"NDBI_{y}.tif")
    download_image(ndvi, f"NDVI_{y}.tif")

print("\n[+] All raster downloads complete!")
