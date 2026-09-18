import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

print("Loading CSV...")
df = pd.read_csv("davao_adaptive_node_features.csv", low_memory=False)

features = [
    'LST_2023_100m', 'NDVI_2023_adaptive', 'NDBI_2023_adaptive',
    'LST_2024_100m', 'NDVI_2024_adaptive', 'NDBI_2024_adaptive',
    'LST_2025_100m', 'NDVI_2025_adaptive', 'NDBI_2025_adaptive'
]

# Convert exactly 0.0 to NaN for imputation
for col in features:
    zeros_count = (df[col] == 0).sum()
    print(f"{col}: Found {zeros_count} zeros. Converting to NaN.")
    df.loc[df[col] == 0, col] = np.nan

coords = np.vstack((df['x'].values, df['y'].values)).T

print("\nImputing with KD-Tree...")
for col in features:
    nan_mask = df[col].isna()
    num_nans = nan_mask.sum()
    if num_nans > 0:
        valid_mask = ~nan_mask
        valid_coords = coords[valid_mask]
        valid_values = df.loc[valid_mask, col].values
        
        tree = cKDTree(valid_coords)
        nan_coords = coords[nan_mask]
        distances, indices = tree.query(nan_coords, k=3)
        
        imputed_values = np.mean(valid_values[indices], axis=1)
        df.loc[nan_mask, col] = imputed_values
        print(f"Fixed {num_nans} values for {col}.")

print("\nSaving fixed CSV...")
df.to_csv("davao_adaptive_node_features.csv", index=False)
print("Done!")
