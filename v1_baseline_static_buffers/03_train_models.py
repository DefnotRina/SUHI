import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# -------------------------------------------------------------------------
# 1. LOAD PREPROCESSED GRAPH DATA
# -------------------------------------------------------------------------
print("[*] Loading attributed nodes and edges...")
nodes_gdf = gpd.read_file("davao_attributed_nodes.gpkg", layer="attributed_nodes")
edges_gdf = gpd.read_file("davao_edges_utm.gpkg", layer="edges")

# Create sequential integer IDs for PyTorch index mapping (0 to N-1)
node_id_map = {osmid: idx for idx, osmid in enumerate(nodes_gdf["osmid"])}
nodes_gdf["node_idx"] = nodes_gdf["osmid"].map(node_id_map)

# Filter edges where both source (u) and target (v) exist in clean node set
edges_gdf = edges_gdf[edges_gdf["u"].isin(node_id_map) & edges_gdf["v"].isin(node_id_map)].copy()
edges_gdf["u_idx"] = edges_gdf["u"].map(node_id_map)
edges_gdf["v_idx"] = edges_gdf["v"].map(node_id_map)

# -------------------------------------------------------------------------
# 2. FEATURE PREPARATION & 80/20 TRAIN-TEST SPLIT
# -------------------------------------------------------------------------
feature_cols = ["NDBI", "NDVI"]
target_col = "LST"

X = nodes_gdf[feature_cols].values
y = nodes_gdf[target_col].values

# Standardize predictor features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Generate reproducible 80/20 node indices for fair comparison
indices = np.arange(len(nodes_gdf))
train_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42)

# -------------------------------------------------------------------------
# 3. BASELINE MODEL: RANDOM FOREST REGRESSION
# -------------------------------------------------------------------------
print("\n--- Training Baseline Model: Random Forest Regressor ---")
rf_model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
rf_model.fit(X_scaled[train_idx], y[train_idx])

rf_preds = rf_model.predict(X_scaled[test_idx])

rf_rmse = np.sqrt(mean_squared_error(y[test_idx], rf_preds))
rf_mae = mean_absolute_error(y[test_idx], rf_preds)
rf_r2 = r2_score(y[test_idx], rf_preds)

print(f"[RF Test Metrics] RMSE: {rf_rmse:.4f} °C | MAE: {rf_mae:.4f} °C | R²: {rf_r2:.4f}")

# -------------------------------------------------------------------------
# 4. PROPOSED MODEL: SPATIAL GRAPH CONVOLUTIONAL NETWORK (GCN)
# -------------------------------------------------------------------------
print("\n--- Training Proposed Model: Spatial Graph Convolutional Network ---")

# Construct PyTorch Edge Index (2 x E) and Edge Weight Tensor
edge_index = torch.tensor(
    np.vstack((edges_gdf["u_idx"].values, edges_gdf["v_idx"].values)), 
    dtype=torch.long
)
edge_weights = torch.tensor(edges_gdf["edge_weight"].values, dtype=torch.float)
x_tensor = torch.tensor(X_scaled, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.float).unsqueeze(1)

# PyG Data Object
graph_data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_weights, y=y_tensor)

# Define Spatial GCN Architecture
class SpatialGCN(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super(SpatialGCN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, out_features)

    def forward(self, x, edge_index, edge_weight=None):
        # First graph convolutional aggregation
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.15, training=self.training)
        # Second graph convolutional aggregation
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        # Linear regression output layer
        out = self.regressor(x)
        return out

model = SpatialGCN(in_features=2, hidden_dim=32, out_features=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.MSELoss()

# Training Loop
train_mask = torch.tensor(train_idx, dtype=torch.long)
test_mask = torch.tensor(test_idx, dtype=torch.long)

model.train()
for epoch in range(1, 251):
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
    loss = criterion(out[train_mask], graph_data.y[train_mask])
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:03d} | Train Loss (MSE): {loss.item():.4f}")

# Evaluation Phase
model.eval()
with torch.no_grad():
    gcn_preds = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
    test_preds = gcn_preds[test_mask].squeeze().numpy()
    test_actuals = graph_data.y[test_mask].squeeze().numpy()

gcn_rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
gcn_mae = mean_absolute_error(test_actuals, test_preds)
gcn_r2 = r2_score(test_actuals, test_preds)

print(f"[GCN Test Metrics] RMSE: {gcn_rmse:.4f} °C | MAE: {gcn_mae:.4f} °C | R²: {gcn_r2:.4f}")

# -------------------------------------------------------------------------
# 5. BENCHMARK COMPARISON SUMMARY TABLE
# -------------------------------------------------------------------------
comparison_df = pd.DataFrame({
    "Model": ["Baseline Random Forest (RF)", "Spatial Graph Convolutional Network (GCN)"],
    "RMSE (°C)": [rf_rmse, gcn_rmse],
    "MAE (°C)": [rf_mae, gcn_mae],
    "R²": [rf_r2, gcn_r2]
})

print("\n================ FINAL BENCHMARK SUMMARY ================")
print(comparison_df.to_string(index=False))
print("=========================================================")