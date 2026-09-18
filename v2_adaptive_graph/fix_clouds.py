import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from pathlib import Path

def main():
    script_dir = Path(__file__).parent
    csv_path = script_dir / "davao_adaptive_node_features.csv"
    
    print("[*] Loading CSV...")
    df = pd.read_csv(csv_path, low_memory=False)
    
    years = ['2023', '2024', '2025']
    
    # We need spatial coordinates for KD-Tree
    coords = np.vstack((df['x'].values, df['y'].values)).T
    
    for year in years:
        lst_col = f"LST_{year}_100m"
        ndvi_col = f"NDVI_{year}_adaptive"
        ndbi_col = f"NDBI_{year}_adaptive"
        
        # Identify cloud outliers (LST < 15C in a tropical city is physically impossible)
        cloud_mask = df[lst_col] < 15.0
        num_clouds = cloud_mask.sum()
        
        if num_clouds > 0:
            print(f"[*] Found {num_clouds} cloud artifacts in {year}. Masking as NaN...")
            # Mask LST, NDVI, and NDBI because the cloud obscures everything
            df.loc[cloud_mask, lst_col] = np.nan
            df.loc[cloud_mask, ndvi_col] = np.nan
            df.loc[cloud_mask, ndbi_col] = np.nan
            
            # Impute LST
            valid_mask = ~df[lst_col].isna()
            tree = cKDTree(coords[valid_mask])
            distances, indices = tree.query(coords[cloud_mask], k=3)
            df.loc[cloud_mask, lst_col] = np.mean(df.loc[valid_mask, lst_col].values[indices], axis=1)
            
            # Impute NDVI
            valid_mask = ~df[ndvi_col].isna()
            tree = cKDTree(coords[valid_mask])
            distances, indices = tree.query(coords[cloud_mask], k=3)
            df.loc[cloud_mask, ndvi_col] = np.mean(df.loc[valid_mask, ndvi_col].values[indices], axis=1)
            
            # Impute NDBI
            valid_mask = ~df[ndbi_col].isna()
            tree = cKDTree(coords[valid_mask])
            distances, indices = tree.query(coords[cloud_mask], k=3)
            df.loc[cloud_mask, ndbi_col] = np.mean(df.loc[valid_mask, ndbi_col].values[indices], axis=1)

    print("[*] Saving cleaned data...")
    df.to_csv(csv_path, index=False)
    print("[+] Done!")

if __name__ == "__main__":
    main()
