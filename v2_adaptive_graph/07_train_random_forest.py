#!/Users/carlita/.pyenv/versions/3.10.14/bin/python
"""
Step 7: Train Random Forest Baseline
This script trains a traditional Machine Learning baseline (Random Forest)
on the exact same historical environmental data as the ST-GCN, but WITHOUT
access to the spatial graph topology (edge_index). 

This serves as a critical benchmark for the thesis to prove whether or not
the spatial graph neural network actually provides a predictive advantage.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import time

def main():
    print("==========================================")
    print("   TRADITIONAL ML BENCHMARK (RANDOM FOREST) ")
    print("==========================================")
    
    script_dir = Path(__file__).parent
    features_csv_path = script_dir / "davao_adaptive_node_features.csv"
    
    # 1. Load Data
    print(f"[*] Loading nodes from '{features_csv_path.name}'...")
    nodes_df = pd.read_csv(features_csv_path, low_memory=False)
    
    # 2. Interpolate Missing Data
    print("[*] Interpolating missing LST_2024 data (Averaging 2023 & 2025)...")
    nodes_df['LST_2024_100m'] = (nodes_df['LST_2023_100m'] + nodes_df['LST_2025_100m']) / 2.0
    
    # 3. Prepare Tabular Features (X) and Target (Y)
    print("[*] Preparing 2D Tabular Feature Matrix (X)...")
    feature_cols = [
        'LST_2023_100m', 'NDVI_2023_adaptive', 'NDBI_2023_adaptive',
        'LST_2024_100m', 'NDVI_2024_adaptive', 'NDBI_2024_adaptive'
    ]
    target_col = 'LST_2025_100m'
    
    # Fill any NaNs
    nodes_df[feature_cols] = nodes_df[feature_cols].fillna(0)
    nodes_df[target_col] = nodes_df[target_col].fillna(nodes_df[target_col].mean())
    
    X = nodes_df[feature_cols].values
    Y = nodes_df[target_col].values
    
    # Normalize features to match GNN preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Train/Val Split (80% / 20%)
    print(f"[*] Splitting 310,000+ nodes into Train/Val sets...")
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
    
    # 5. Train Random Forest
    print(f"[*] Commencing Random Forest Training (Trees: 50, Max Depth: 15)...")
    print(f"    (Utilizing all CPU cores for fast parallel processing)")
    
    rf_model = RandomForestRegressor(
        n_estimators=50, 
        max_depth=15, 
        n_jobs=-1, # Use all available CPU cores
        random_state=42
    )
    
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    elapsed = time.time() - start_time
    print(f"[+] Training complete in {elapsed:.2f} seconds!")
    
    # 6. Evaluation
    print("\n[*] Evaluating Mean Squared Error (MSE)...")
    
    # Train MSE
    train_preds = rf_model.predict(X_train)
    train_mse = mean_squared_error(y_train, train_preds)
    
    # Val MSE
    val_preds = rf_model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_preds)
    
    print(f"    Train MSE Loss: {train_mse:.4f}")
    print(f"    Validation MSE Loss: {val_mse:.4f}")
    print("\n[+] Random Forest benchmark successfully established!")

if __name__ == "__main__":
    main()
