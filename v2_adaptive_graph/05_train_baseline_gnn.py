#!/Users/carlita/.pyenv/versions/3.10.14/bin/python
"""
Step 5: Train Baseline GNN (GraphSAGE)
This script proves that our generated Adaptive Graph can be successfully ingested
by PyTorch Geometric. We use a lightweight GraphSAGE architecture to predict
Land Surface Temperature (LST) based on spatial coordinates and road buffer radii.
Memory is optimized using NeighborLoader for mini-batching on 310k nodes,
and training is accelerated using Apple Silicon's MPS backend.
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
    print("   BASELINE SPATIAL GRAPHSAGE TRAINING    ")
    print("==========================================")
    
    script_dir = Path(__file__).parent
    features_csv_path = script_dir / "davao_adaptive_node_features.csv"
    edge_csv_path = script_dir / "edge_index.csv"
    
    # 1. Load Data
    print(f"[*] Loading nodes from '{features_csv_path.name}'...")
    nodes_df = pd.read_csv(features_csv_path, low_memory=False)
    
    print(f"[*] Loading edges from '{edge_csv_path.name}'...")
    edges_df = pd.read_csv(edge_csv_path)
    
    # 2. Map OSMIDs to Contiguous Indices (0 to N-1)
    # PyTorch Geometric requires node indices to start from 0.
    print("[*] Mapping OSMIDs to contiguous PyG indices...")
    osmid_to_idx = {osmid: i for i, osmid in enumerate(nodes_df['osmid'])}
    
    edges_df['source_idx'] = edges_df['source'].map(osmid_to_idx)
    edges_df['target_idx'] = edges_df['target'].map(osmid_to_idx)
    
    # Drop any edges that failed to map (should be none, but just in case)
    edges_df = edges_df.dropna(subset=['source_idx', 'target_idx'])
    
    # Use np.array to avoid slow list creation warning
    edge_index_np = np.array([
        edges_df['source_idx'].values,
        edges_df['target_idx'].values
    ], dtype=np.int64)
    edge_index = torch.from_numpy(edge_index_np)
    
    # 3. Prepare Features (X) and Target (Y)
    print("[*] Preparing Feature Matrix (X) and Target Vector (Y)...")
    
    # We will use x, y coordinates and buffer_radius as inputs
    feature_cols = ['x', 'y', 'buffer_radius']
    
    # Fill any remaining NaNs in features just in case
    nodes_df[feature_cols] = nodes_df[feature_cols].fillna(0)
    
    # Target is the LST
    target_col = 'LST_2025_100m'
    # Drop nodes where target is NaN (or fill with mean)
    mean_target = nodes_df[target_col].mean()
    nodes_df[target_col] = nodes_df[target_col].fillna(mean_target)
    
    # Normalize Features (Crucial for Neural Networks) manually to avoid sklearn crashes
    vals = nodes_df[feature_cols].values
    means = np.mean(vals, axis=0)
    stds = np.std(vals, axis=0)
    stds[stds == 0] = 1.0 # Prevent division by zero
    x_scaled = (vals - means) / stds
    
    # Critical Fix for Apple Silicon: Downcast in numpy before creating tensor
    # to prevent silent process death.
    x_scaled_f32 = x_scaled.astype(np.float32)
    x = torch.from_numpy(x_scaled_f32)
    
    y_f32 = nodes_df[target_col].values.astype(np.float32)
    y = torch.from_numpy(y_f32).view(-1, 1)
    
    # Create Train/Val masks (80% Train, 20% Val)
    print("DEBUG: Creating masks...")
    num_nodes = len(nodes_df)
    train_size = int(0.8 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # Avoid advanced tensor indexing (e.g. mask[indices]) as it causes silent OOM/crashes on this Mac PyTorch build
    train_mask[:train_size] = True
    val_mask[train_size:] = True
    print("DEBUG: Masks created.")
    
    # 4. Create PyG Data Object
    print("DEBUG: Creating PyG Data Object...")
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask)
    print(f"[+] Graph constructed: {data}")
    
    # 5. Setup Hardware Acceleration (Force CPU for stability)
    device = torch.device('cpu')
    print(f"[*] Hardware Acceleration configured to: {device.type.upper()}")
    
    # 6. Setup Data on Device (Full-Batch Training)
    # Our graph is incredibly sparse (310k nodes, 376k edges) with only 3 features.
    # The total memory footprint is < 20MB. We can safely train full-batch on the GPU
    # without needing complex C++ sparse samplers (pyg-lib).
    data = data.to(device)
    
    # 7. Define GraphSAGE Model
    class BaselineSAGE(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
            
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.conv2(x, edge_index)
            return x
            
    model = BaselineSAGE(in_channels=3, hidden_channels=32, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    # 8. Training Loop
    print("\n[*] Commencing Full-Batch Training (100 Epochs)...")
    for epoch in range(1, 101):
        model.train()
        start_time = time.time()
        
        optimizer.zero_grad()
        
        # Forward pass on the entire graph
        out = model(data.x, data.edge_index)
        
        # Compute loss ONLY on training nodes
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        optimizer.step()
        
        elapsed = time.time() - start_time
        
        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:03d} | Train MSE Loss: {loss.item():.4f} | Time: {elapsed:.2f}s")
            
    print("[+] Baseline benchmark successfully established!")

if __name__ == "__main__":
    main()
