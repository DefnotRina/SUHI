import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path

def main():
    script_dir = Path(__file__).parent
    csv_path = script_dir / "davao_adaptive_node_features.csv"
    
    print("[*] Loading CSV for Temporal Outlier Detection...")
    df = pd.read_csv(csv_path, low_memory=False)
    
    coords = np.vstack((df['x'].values, df['y'].values)).T
    years = [2023, 2024, 2025]
    
    # We will do this iteratively for each year
    for target_year in years:
        other_years = [y for y in years if y != target_year]
        
        target_lst = f"LST_{target_year}_100m"
        target_ndvi = f"NDVI_{target_year}_adaptive"
        target_ndbi = f"NDBI_{target_year}_adaptive"
        
        other_lst_1 = f"LST_{other_years[0]}_100m"
        other_lst_2 = f"LST_{other_years[1]}_100m"
        
        # Calculate the baseline temperature expected for this node (average of other 2 years)
        expected_lst = (df[other_lst_1] + df[other_lst_2]) / 2.0
        
        # A node is a "thin cloud" if it is more than 5C colder than the expected baseline.
        # It's very rare for urban infrastructure to drop 5+ degrees year-over-year.
        thin_cloud_mask = (expected_lst - df[target_lst]) > 5.0
        
        num_thin_clouds = thin_cloud_mask.sum()
        
        if num_thin_clouds > 0:
            print(f"\n[*] Found {num_thin_clouds} thin cloud artifacts in {target_year} (Temporal Outliers).")
            print(f"    -> Masking these nodes as NaN for {target_year}...")
            
            df.loc[thin_cloud_mask, target_lst] = np.nan
            df.loc[thin_cloud_mask, target_ndvi] = np.nan
            df.loc[thin_cloud_mask, target_ndbi] = np.nan
            
            # Re-impute LST
            valid_mask = ~df[target_lst].isna()
            tree = cKDTree(coords[valid_mask])
            distances, indices = tree.query(coords[thin_cloud_mask], k=3)
            df.loc[thin_cloud_mask, target_lst] = np.mean(df.loc[valid_mask, target_lst].values[indices], axis=1)
            
            # Re-impute NDVI
            valid_mask = ~df[target_ndvi].isna()
            tree = cKDTree(coords[valid_mask])
            distances, indices = tree.query(coords[thin_cloud_mask], k=3)
            df.loc[thin_cloud_mask, target_ndvi] = np.mean(df.loc[valid_mask, target_ndvi].values[indices], axis=1)
            
            # Re-impute NDBI
            valid_mask = ~df[target_ndbi].isna()
            tree = cKDTree(coords[valid_mask])
            distances, indices = tree.query(coords[thin_cloud_mask], k=3)
            df.loc[thin_cloud_mask, target_ndbi] = np.mean(df.loc[valid_mask, target_ndbi].values[indices], axis=1)
            
            print(f"    -> Successfully re-imputed {num_thin_clouds} nodes using KD-Tree spatial neighbors.")

    print("\n[*] Saving temporally cleaned dataset...")
    df.to_csv(csv_path, index=False)
    print("[+] Done!")

if __name__ == "__main__":
    main()
