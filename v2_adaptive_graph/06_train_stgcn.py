#!/Users/carlita/.pyenv/versions/3.10.14/bin/python
"""
Step 6: Train Spatio-Temporal GNN (ST-GCN)
This script upgrades the baseline GraphSAGE model into a Spatio-Temporal
architecture. We restructure the flat CSV data into a 3D time-series tensor.
We use an LSTM to capture temporal dynamics (2023 -> 2024) and a GraphSAGE
convolution to capture spatial diffusion across the road network, predicting 
LST for 2025.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from pathlib import Path
import time

def main():
    print("==========================================")
    print("   SPATIO-TEMPORAL GCN (ST-GCN) TRAINING  ")
    print("==========================================")
    
    script_dir = Path(__file__).parent
    features_csv_path = script_dir / "davao_adaptive_node_features.csv"
    edge_csv_path = script_dir / "edge_index.csv"
    
    # 1. Load Data
    print(f"[*] Loading nodes from '{features_csv_path.name}'...")
    nodes_df = pd.read_csv(features_csv_path, low_memory=False)
    
    print(f"[*] Loading edges from '{edge_csv_path.name}'...")
    edges_df = pd.read_csv(edge_csv_path)
    
    # 2. Map OSMIDs to Contiguous Indices
    print("[*] Mapping OSMIDs to contiguous PyG indices...")
    osmid_to_idx = {osmid: i for i, osmid in enumerate(nodes_df['osmid'])}
    
    edges_df['source_idx'] = edges_df['source'].map(osmid_to_idx)
    edges_df['target_idx'] = edges_df['target'].map(osmid_to_idx)
    edges_df = edges_df.dropna(subset=['source_idx', 'target_idx'])
    
    edge_index_np = np.array([
        edges_df['source_idx'].values,
        edges_df['target_idx'].values
    ], dtype=np.int64)
    edge_index = torch.from_numpy(edge_index_np)
    
    # 3. Interpolate Missing Data
    print("[*] Interpolating missing LST_2024 data (Averaging 2023 & 2025)...")
    nodes_df['LST_2024_100m'] = (nodes_df['LST_2023_100m'] + nodes_df['LST_2025_100m']) / 2.0
    
    # 4. Construct 3D Time-Series Tensor
    print("[*] Constructing 3D Time-Series Tensor (Nodes x Time x Features)...")
    
    # T=0 (2023), T=1 (2024)
    # Features: LST, NDVI, NDBI
    t1_cols = ['LST_2023_100m', 'NDVI_2023_adaptive', 'NDBI_2023_adaptive']
    t2_cols = ['LST_2024_100m', 'NDVI_2024_adaptive', 'NDBI_2024_adaptive']
    
    # Fill any NaNs just in case
    nodes_df[t1_cols + t2_cols] = nodes_df[t1_cols + t2_cols].fillna(0)
    
    t1_vals = nodes_df[t1_cols].values
    t2_vals = nodes_df[t2_cols].values
    
    # Stack into shape: (Num_Nodes, 2, 3)
    x_3d = np.stack([t1_vals, t2_vals], axis=1)
    
    # Normalize features across nodes
    # For a 3D tensor, we calculate mean/std over the nodes (axis 0)
    # to keep temporal relationships intact.
    means = np.mean(x_3d, axis=0, keepdims=True)
    stds = np.std(x_3d, axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    
    x_scaled = (x_3d - means) / stds
    x_scaled_f32 = x_scaled.astype(np.float32)
    x = torch.from_numpy(x_scaled_f32) # Shape: (N, 2, 3)
    
    # Target (Y) is LST 2025
    target_col = 'LST_2025_100m'
    nodes_df[target_col] = nodes_df[target_col].fillna(nodes_df[target_col].mean())
    y_f32 = nodes_df[target_col].values.astype(np.float32)
    y = torch.from_numpy(y_f32).view(-1, 1)
    
    # 5. Create Train/Val Masks
    num_nodes = len(nodes_df)
    train_size = int(0.8 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # Basic slicing to bypass Apple Silicon advanced indexing crashes
    train_mask[:train_size] = True
    val_mask[train_size:] = True
    
    # 6. Define ST-GCN Architecture (LSTM -> GraphSAGE)
    class ST_GCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            # LSTM for Temporal Dynamics
            # input_size = 3 (features), batch_first=True means input shape is (N, T, F)
            self.lstm = torch.nn.LSTM(input_size=in_channels, hidden_size=hidden_channels, batch_first=True)
            
            # GraphSAGE for Spatial Diffusion
            self.conv1 = SAGEConv(hidden_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
            
        def forward(self, x, edge_index):
            # 1. Temporal Encoding
            # x shape: (N, T, F)
            # lstm_out shape: (N, T, Hidden)
            lstm_out, _ = self.lstm(x)
            
            # Extract the final hidden state (the last timestep's output)
            # spatial_x shape: (N, Hidden)
            spatial_x = lstm_out[:, -1, :]
            
            # 2. Spatial Convolution
            spatial_x = self.conv1(spatial_x, edge_index)
            spatial_x = F.relu(spatial_x)
            spatial_x = F.dropout(spatial_x, p=0.2, training=self.training)
            spatial_x = self.conv2(spatial_x, edge_index)
            
            return spatial_x

    device = torch.device('cpu')
    print(f"[*] Hardware Acceleration configured to: {device.type.upper()}")
    
    model = ST_GCN(in_channels=3, hidden_channels=32, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    # Data isn't a standard PyG Data object because X is 3D. 
    # We just keep x, y, and edge_index as separate tensors on device.
    x = x.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    
    print(f"[+] ST-GCN constructed. X Shape: {x.shape} | Y Shape: {y.shape}")
    
    # 7. Training Loop
    print("\n[*] Commencing Full-Batch ST-GCN Training (100 Epochs)...")
    for epoch in range(1, 101):
        model.train()
        start_time = time.time()
        
        optimizer.zero_grad()
        
        # Forward pass
        out = model(x, edge_index)
        
        # Compute loss ONLY on training nodes
        loss = criterion(out[train_mask], y[train_mask])
        
        loss.backward()
        optimizer.step()
        
        elapsed = time.time() - start_time
        
        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:03d} | Train MSE Loss: {loss.item():.4f} | Time: {elapsed:.2f}s")
            
    print("[+] ST-GCN benchmark successfully established!")

if __name__ == "__main__":
    main()
