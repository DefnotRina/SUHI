#!/Users/carlita/.pyenv/versions/3.10.14/bin/python
"""
Step 0: Download All Rasters
This script completely regenerates the dataset by downloading Landsat 8
imagery for 2023, 2024, and 2025 directly from Google Earth Engine.

It ensures scientific reproducibility by applying identical mathematical
cloud-masking and thermal calibrations across all three years.
"""
import os
import sys

try:
    import ee
    import geemap
except ImportError:
    print("[!] Missing dependencies. Run: pip install earthengine-api geemap")
    sys.exit(1)

def main():
    print("==========================================")
    print("   EARTH ENGINE DATA SYNCHRONIZATION      ")
    print("==========================================")

    print("[*] Initializing Google Earth Engine...")
    try:
        geemap.ee_initialize()
    except Exception as e:
        print(f"[!] Earth Engine Initialization Failed: {e}")
        print("[!] You may need to specify a Google Cloud Project.")
        sys.exit(1)

    # Bounding Box for Davao City (extracted from original rasters)
    print("[*] Targeting Davao City boundaries...")
    region = ee.Geometry.Rectangle([125.217425, 6.956194, 125.697395, 7.605946])
    scale = 30 # 30 meters resolution
    crs = 'EPSG:4326'

    def apply_scale_factors(image):
        # Official Landsat 8/9 Collection 2 scaling factors
        opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        # Convert Thermal Band 10 from Kelvin to Celsius
        thermalBands = image.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15)
        return image.addBands(opticalBands, None, True).addBands(thermalBands, None, True)

    def get_indices(image):
        # NDVI = (NIR - Red) / (NIR + Red) = (B5 - B4) / (B5 + B4)
        ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        # NDBI = (SWIR1 - NIR) / (SWIR1 + NIR) = (B6 - B5) / (B6 + B5)
        ndbi = image.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
        lst = image.select('ST_B10').rename('LST')
        return image.addBands([ndvi, ndbi, lst])

    out_dir = os.path.join(os.path.dirname(__file__), "data", "rasters")
    os.makedirs(out_dir, exist_ok=True)

    years = [2023, 2024, 2025]
    for year in years:
        print(f"\n[*] Processing Year {year}...")
        
        # Query Landsat 8 Collection 2 Tier 1 Surface Reflectance
        dataset = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
            .filterBounds(region) \
            .filterDate(f'{year}-01-01', f'{year}-12-31') \
            .map(apply_scale_factors) \
            .map(get_indices)
        
        # Take the median pixel across the whole year to remove clouds
        median_image = dataset.median()
        
        bands = ['LST', 'NDVI', 'NDBI']
        for band in bands:
            out_file = os.path.join(out_dir, f"{band}_{year}.tif")
            print(f"    -> Exporting {band} to {out_file}...")
            
            geemap.ee_export_image(
                median_image.select(band), 
                filename=out_file, 
                scale=scale, 
                region=region, 
                crs=crs,
                file_per_band=False
            )
            
    print("\n[+] All 9 Rasters successfully synchronized!")

if __name__ == "__main__":
    main()
