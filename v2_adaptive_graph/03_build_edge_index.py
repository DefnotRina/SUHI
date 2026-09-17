#!/opt/homebrew/Caskroom/miniforge/base/bin/python
"""
Step 3: Build Edge Index
Graph Neural Networks (ST-GCN) require an 'Edge Index' (adjacency list) defining node connections.
Since we interpolated new nodes in Step 1, the raw OSM edge list is no longer valid.
This script retraces the interpolation mathematics to generate raw connections, 
and then spatially maps these connections back to the canonical Node IDs (OSMIDs) 
using a highly efficient KD-Tree. Mappings beyond 2.0 meters are discarded to ensure precision.
"""
import osmnx as ox
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from scipy.spatial import cKDTree
from pathlib import Path
from tqdm import tqdm

def main():
    print("==========================================")
    print("   ST-GCN EDGE INDEX RECONSTRUCTION (V2)")
    print("         (SPATIAL KD-TREE MAPPING)        ")
    print("==========================================")
    
    script_dir = Path(__file__).parent
    features_csv_path = script_dir / "davao_adaptive_node_features.csv"
    
    # 1. Load Canonical Nodes (Our extracted dataset)
    print(f"[*] Loading Canonical Nodes from '{features_csv_path.name}'...")
    canonical_df = pd.read_csv(features_csv_path, low_memory=False)
    print(f"[*] Loaded {len(canonical_df)} canonical nodes.")
    
    # Build KDTree for spatial mapping
    print("[*] Building KD-Tree of canonical coordinates...")
    canonical_coords = np.vstack((canonical_df['x'].values, canonical_df['y'].values)).T
    canonical_tree = cKDTree(canonical_coords)
    canonical_osmids = canonical_df['osmid'].values
    
    # 2. Download Current OSM Graph
    print("[*] Downloading driveable road network from OSMnx...")
    ox.settings.use_cache = True
    davao_query = "Davao City, Philippines"
    core_districts_gdf = ox.geocode_to_gdf(davao_query)
    urban_core_polygon = core_districts_gdf.union_all()
    
    custom_filter = '["highway"]["area"!~"yes"]["highway"!~"cycleway|footway|path|pedestrian|steps|track|corridor|elevator|escalator|proposed|construction|bridleway|abandoned|platform|raceway"]'
    G = ox.graph_from_polygon(urban_core_polygon, custom_filter=custom_filter, simplify=True)
    
    print("[*] Projecting graph to UTM...")
    G_proj = ox.project_graph(G)
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_proj)
    
    print(f"[*] Fetched graph with {len(nodes_gdf)} intersections.")
    
    # 3. Generate Edges & Interpolated Coordinates
    # Rationale: Because we generated entirely new node entities via linear interpolation on
    # road segments, PyTorch Geometric has no way to know how these nodes connect. 
    # By retracing the math step-by-step, we generate literal point-to-point connections.
    print("[*] Retracing interpolation logic and extracting mathematical edges...")
    GLOBAL_MIN_GAP = 30.0
    
    # We will build a list of (source_coord, target_coord)
    # where coord is (x, y)
    raw_edges_coords = []
    
    for idx, edge in edges_gdf.iterrows():
        u, v, key = idx
        geom = edge.geometry
        length = edge["length"]
        
        # Original intersection coordinates
        u_pt = nodes_gdf.loc[u].geometry
        v_pt = nodes_gdf.loc[v].geometry
        
        N = int(length // GLOBAL_MIN_GAP)
        
        if N >= 2:
            num_points = N - 1
            spacing = length / N
            
            prev_pt = u_pt
            for i in range(1, N):
                curr_pt = geom.interpolate(i * spacing)
                raw_edges_coords.append( ((prev_pt.x, prev_pt.y), (curr_pt.x, curr_pt.y)) )
                prev_pt = curr_pt
                
            raw_edges_coords.append( ((prev_pt.x, prev_pt.y), (v_pt.x, v_pt.y)) )
        else:
            raw_edges_coords.append( ((u_pt.x, u_pt.y), (v_pt.x, v_pt.y)) )
            
    print(f"[*] Generated {len(raw_edges_coords)} raw edges connecting the nodes.")
    
    # 4. Spatially Map Raw Edges to Canonical OSMIDs
    print("[*] Spatially mapping edges to Canonical OSMIDs...")
    
    # Flatten coordinates for bulk KDTree query
    source_coords = np.array([e[0] for e in raw_edges_coords])
    target_coords = np.array([e[1] for e in raw_edges_coords])
    
    # Query source nodes
    dist_src, idx_src = canonical_tree.query(source_coords, k=1)
    # Query target nodes
    dist_tgt, idx_tgt = canonical_tree.query(target_coords, k=1)
    
    # Filter out edges where the distance is > 2.0 meters (failed to map due to OSM changes)
    valid_mask = (dist_src < 2.0) & (dist_tgt < 2.0)
    
    mapped_sources = canonical_osmids[idx_src[valid_mask]]
    mapped_targets = canonical_osmids[idx_tgt[valid_mask]]
    
    mapped_edges_df = pd.DataFrame({
        "source": mapped_sources,
        "target": mapped_targets
    })
    
    # Remove self-loops if any
    mapped_edges_df = mapped_edges_df[mapped_edges_df["source"] != mapped_edges_df["target"]]
    # Drop duplicates
    mapped_edges_df.drop_duplicates(inplace=True)
    
    print(f"[*] Mapped {len(mapped_edges_df)} valid edges successfully!")
    print(f"[*] Dropped {len(raw_edges_coords) - len(mapped_edges_df)} edges (did not match canonical nodes).")
    
    # 5. Save Edge Index
    out_path = script_dir / "edge_index.csv"
    mapped_edges_df.to_csv(out_path, index=False)
    print(f"[+] Edge index saved to: {out_path.name}")
    
if __name__ == "__main__":
    main()
