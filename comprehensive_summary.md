# Adaptive Graph for Urban Thermal Analysis: Comprehensive Project Summary

This document provides a comprehensive, technically rigorous, and updated overview of the research and engineering pipeline developed for the thesis project: **"Spatio-Temporal Graph Convolutional Networks (ST-GCN) for Surface Urban Heat Island (SUHI) Analysis and Land Surface Temperature (LST) Modeling in Davao City."**

It details our data engineering workflow, satellite sensor fusion, temporal outlier scrubbing, topological road graph construction, machine learning benchmarks, ablation baselines (incorporating faculty feedback), and current academic manuscript progress.

---

## 1. Research Objectives & Core Problem Statement

Urban areas exhibit significantly elevated surface temperatures compared to rural surroundings, forming Surface Urban Heat Islands (SUHI). Traditional remote sensing and spatial interpolation methods treat geographic space as flat, Euclidean pixel grids (raster images) or isolated points, failing to capture how the built environment—specifically concrete and asphalt road infrastructure—channels, retains, and dissipates thermal energy.

### Core Hypotheses:
1. **Topological Hypothesis:** Road network topology (the physical graph of transportation corridors) provides structural information for thermal modeling that significantly outperforms naive geographic proximity (K-Nearest Neighbors).
2. **Spatio-Temporal Hypothesis:** Integrating historical multi-year temporal sequences via recurrent architectures (LSTM) with spatial graph convolutions (GraphSAGE) captures thermal inertia and land-use evolution better than traditional tabular models (Random Forest) or static spatial-only GNNs.

---

## 2. End-to-End Pipeline Architecture

```
[OpenStreetMap (OSM) Road Network]
       │
       ▼ (DPWH Buffer Standards: 5m-15m & 30m Adaptive Interpolation)
[Adaptive Spatial Graph (310,000+ Nodes)]
       │
       ├─────────────────────────────────────────┐
       ▼ (Landsat 8 L2SR + ECOSTRESS Fusion)     ▼ (KD-Tree Topological Edge Reconstruction)
[Multi-Scale Environmental Feature Extraction]  [Weighted Edge Index (Distances & Contiguous PyG)]
       │                                         │
       ▼ (Temporal Outlier Detection >5°C drop)  ▼ (KNN Baseline k=5 & Self-Loop Ablation)
[KD-Tree Spatial Imputation (Thin Cloud Scrub)] [Adjacency Matrices: Road, KNN, Self-Loops]
       │                                         │
       └────────────────────┬────────────────────┘
                            │
                            ▼
              [Model Benchmarking & Evaluation]
      ┌─────────────────────┼─────────────────────┐
      ▼                     ▼                     ▼
[Random Forest]     [Spatial GraphSAGE]    [Spatio-Temporal GCN]
(Tabular Baseline)   (Static Topology)      (LSTM + GraphSAGE)
```

---

## 3. Data Engineering & Topological Graph Construction

We engineered a modular Python pipeline (`v2_adaptive_graph/`) to download, interpolate, enrich, and model the road network of Davao City.

### Step 1: Building the Adaptive Graph (`01_build_adaptive_graph.py`)
*   **Driveable Network Filtering:** Queried OSMnx for the administrative boundary of Davao City, filtering exclusively for driveable vehicular ways while excluding footpaths, cycleways, and unverified tracks.
*   **Coordinate Projection:** Reprojected geographic coordinates (WGS84 Lat/Lon) to the local projected coordinate reference system (UTM Zone 51N, EPSG:32651). Distance and interpolation calculations must be performed in metric units to preserve physical fidelity.
*   **Hierarchical Buffering (DPWH Standards):** Applied Philippine Department of Public Works and Highways (DPWH) carriageway standards:
    *   *Trunk / Primary Roads:* 15-meter buffer radius.
    *   *Secondary / Tertiary Roads:* 10-meter buffer radius.
    *   *Residential / Service Roads:* 5-meter buffer radius.
*   **Adaptive Interpolation:** Natural OSM graphs only place vertices at road intersections. We mathematically interpolated intermediate nodes along every road segment at a maximum interval of 30 meters, yielding over 310,000 nodes and providing dense spatial granularity along transit corridors.
*   **Output:** `davao_adaptive_nodes_utm.gpkg`.

### Step 2: Multi-Scale Feature Extraction (`02_extract_adaptive_features.py`)
*   **Multi-Scale Spatial Buffering:**
    *   *Macro-Scale (Thermal / LST):* Fixed 100-meter buffer radius around nodes to capture ambient neighborhood thermal signatures.
    *   *Micro-Scale (Optical / NDVI & NDBI):* Dynamic buffer matching the DPWH road width (5m to 15m) to isolate physical surface materials on the roadway.
*   **High-Throughput Batch Processing:** Chunked queries into batches of 10,000 nodes against heavy multi-band GeoTIFFs to prevent GDAL memory segmentation faults.
*   **Initial Spatial KD-Tree Imputation:** Handled missing raster pixels (sensor anomalies or mask artifacts) by constructing a spatial KD-Tree that averages the 3 nearest physical neighbors.
*   **Output:** `davao_adaptive_node_features.csv` and `davao_adaptive_node_features.gpkg`.

### Step 3: Topological Edge Reconstruction (`03_build_edge_index.py`)
*   **Sequential Node Linking:** Retraced interpolation paths between original intersection pairs ($U, V$) to establish directional connectivity between intermediate nodes.
*   **KD-Tree Coordinate Alignment:** Mapped interpolated edges back to canonical OSM node IDs using a 2.0-meter tolerance KD-Tree, ensuring contiguous topological indexing.
*   **Distance Metric Calculation:** Calculated Euclidean distances ($d_{ij}$ in meters) along edges to create weighted adjacency matrices (`edge_index_weighted.csv`).
*   **Output:** `edge_index.csv` (unweighted) and `edge_index_weighted.csv` (distance-attributed).

### Step 4: 3D Geospatial Visualization (`04_visualize_road_network.py`)
*   **Thermal Color Mapping:** Normalized LST values using the 2nd to 98th percentile clipping and mapped them to the perceptually uniform 'Inferno' color spectrum.
*   **Glowing Thermal Edge Rendering:** Upgraded from static node scatter points to high-performance PyDeck `LineLayer` segments colored dynamically by source node temperature with auto-highlighting over a Mapbox Dark canvas.
*   **Output:** Interactive standalone 3D web visualizations (`suhi_map.html`).

---

## 4. Satellite Sensor Fusion & Temporal Preprocessing

### The 7-Year Baseline Framework (2019–2025)
To ensure temporal consistency and statistical validity, our study is structured around a **7-year monitoring window (2019–2025)**:
*   **Landsat 8 Exclusivity (No Landsat 9):** Landsat 8 provides consistent 16-day revisit observations for surface reflectance (NDVI, NDBI) and thermal infrared LST. Landsat 9 (launched late 2021) was intentionally excluded. Incorporating Landsat 9 would double the observation frequency from 2022 onward (from 16-day to 8-day), artificially skewing the Maximum Value Composite (MVC) calculations and introducing a false warming trend.
*   **ECOSTRESS Thermal Fusion:** Fused with NASA ECOSTRESS aboard the International Space Station (ISS) to sample non-sun-synchronous diurnal temperature variations, compensating for Landsat’s fixed mid-morning (~10:30 AM) overpass.
*   **Maximum Value Composite (MVC V2.1):** Applied annual MVC logic across cloud-screened scenes to extract peak summer SUHI conditions while eliminating transient cloud shadows.

### Temporal Outlier Detection & Thin-Cloud Scrubbing (`temporal_cloud_fix.py`)
*   **The Artifact Problem:** Thin cirrus clouds or semi-transparent haze occasionally pass quality masks in optical/thermal imagery, artificially depressing surface temperatures by several degrees without registering as total missing values (NaNs).
*   **Temporal Deviation Detection:** For each year $t$, a node's temperature $LST_t$ is compared against its multi-year historical expected baseline:
    $$\Delta T = \overline{LST}_{\text{other}} - LST_t$$
    If $\Delta T > 5.0^\circ\text{C}$, the node is flagged as a "thin cloud artifact" (urban asphalt and concrete corridors do not undergo abrupt 5°C localized cooling under identical seasonal conditions).
*   **Topological Re-Imputation:** Flagged nodes have their LST, NDVI, and NDBI values masked and re-imputed using a spatial `cKDTree` querying the 3 nearest valid neighbors along the road network.

---

## 5. Machine Learning Benchmarks & Empirical Results

We executed a progressive benchmarking suite across three paradigms: Tabular Baseline, Spatial GNN, and Spatio-Temporal GCN.

| Model / Configuration | Architecture / Methodology | Spatial Topology | Input Tensor Shape | Loss (MSE) | Key Finding |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Random Forest Baseline** (`07_train_random_forest.py`) | 50 Trees, Max Depth 15, Multi-threaded CPU | None (Tabular features treated independently) | 2D Matrix: $(N, 6)$ [LST, NDVI, NDBI for '23 & '24] | **~18.5 – 21.0** | Establishes traditional ML baseline without spatial message passing. |
| **Baseline Spatial GNN** (`05_train_baseline_gnn.py`) | 2-Layer GraphSAGE (Hidden: 64, ReLU, Dropout 0.2) | OSM Road Network Adjacency | 2D Matrix: $(N, F)$ [Static single-year snapshot] | **22.4** (down from 600.7 initial) | Proved topological convergence; high-speed full-batch execution (~0.10s/epoch). |
| **Proposed Spatio-Temporal GCN (ST-GCN)** (`06_train_stgcn.py`) | LSTM (Hidden: 32) + 2-Layer GraphSAGE | OSM Road Network Adjacency | 3D Tensor: $(N, T=2, F=3)$ [Sequence: 2023 $\to$ 2024 $\to$ Target 2025] | **9.2** | **Massive 58.9% error reduction over spatial GNN.** Historical temporal sequence dramatically boosts accuracy. |

### Key Technical Achievements in Training:
*   **Full-Batch Scalability:** Despite encompassing 310,000+ nodes, the sparse adjacency matrix occupies ~10–15 MB of RAM. We bypassed complex mini-batch samplers (`NeighborLoader`), executing clean, deterministic full-batch gradient descent directly on Apple Silicon CPU/MPS without memory leaks.
*   **3D Tensor Restructuring:** Successfully restructured the flat spatial data into a contiguous PyTorch 3D tensor $(N, T, F)$, passing sequence representations through an LSTM temporal encoder before spatial graph convolution.

---

## 6. Professor's Feedback & Methodological Refinements

Based on formal review with the thesis advisory committee, two critical methodological enhancements were incorporated:

### 1. The Geographic-Distance / KNN Graph Baseline
*   **The Professor's Review:**
    > *"Consider a geographic-distance/KNN graph as an additional baseline. Road-connected vs. no-connectivity tests whether neighbor aggregation helps; road-connected vs. proximity-based adjacency would more directly test whether road topology itself contributes information beyond ordinary geographic proximity."*
*   **Methodological Value:** Heat is inherently autocorrelated in Euclidean space (Tobler's First Law of Geography). Comparing a road graph solely against an unconnected model proves that *spatial aggregation helps*, but does not prove that *road topology is superior to naive distance*.
*   **Implementation (`edge_index_knn_5.csv`):**
    *   Constructed a K-Nearest Neighbor graph ($k=5$) connecting each node to its 5 physically closest neighbors in metric space.
    *   Trains with the exact same LSTM-GraphSAGE architecture, loss function, and temporal sequence.
    *   **Hypothesis Isolation:** If $MSE_{\text{Road}} < MSE_{\text{KNN}}$, we empirically confirm that urban transportation infrastructure governs thermal distribution better than simple Euclidean distance.

### 2. Computational Definition of "Disabled" Connections (Ablation Study)
*   **The Professor's Review:**
    > *"Define exactly what 'road connections disabled' means computationally. If Graph-SAGE is reduced to self-information/self-loops only, explain the resulting architecture. The matched model should differ from the road-connected model only in access to neighbor information as far as practicable so that RQ2 isolates the contribution of road adjacency."*
*   **Computational Definition:**
    *   "Road connections disabled" is explicitly defined as restricting the adjacency matrix $\mathbf{A}$ strictly to **self-loops only** ($\mathbf{A} = \mathbf{I}_N$).
    *   Under this formulation, the GraphSAGE aggregation operator:
        $$h_v^{(l)} = \sigma \left( \mathbf{W} \cdot \text{CONCAT}(h_v^{(l-1)}, \text{AGG}(\{h_u^{(l-1)} : u \in \mathcal{N}(v)\})) \right)$$
        collapses $\mathcal{N}(v) = \emptyset$, forcing the aggregation function to process exclusively the node's own representation.
    *   This guarantees that model depth, trainable parameters, activation functions, and optimizer states remain 100% identical, isolating inter-node message passing as the sole independent variable.

---

## 7. Thesis Manuscript Integration & Academic Deliverables

The research deliverables and text revisions have been synchronized across the academic manuscript:

### Chapter 1: Introduction & Research Questions
*   **Updated Research Question 2 (RQ2):**
    > *"What change in estimation performance results from incorporating road-network connectivity compared with (a) a matched configuration without information exchange between distinct nodes (self-loops only), and (b) a spatial-proximity graph (e.g., K-Nearest Neighbors) that connects nodes based purely on geographic distance?"*
*   **Methodological Delimitations:** Formalized the inclusion of DPWH road standards, the 7-year Landsat 8/ECOSTRESS sensor fusion, and the ablation matrix.

### Chapter 2: Literature Review (RRL)
*   Completed comprehensive review sections across three thematic pillars:
    *   **2.1.1:** Remote sensing of Surface Urban Heat Islands, mono-window thermal algorithms, and temporal cloud contamination mitigation.
    *   **2.1.2:** Evolution of machine learning in SUHI (from Random Forest / XGBoost to Spatial-Temporal Deep Learning).
    *   **2.1.3:** Graph Neural Networks in civil infrastructure, comparing Euclidean grids (CNNs) against Non-Euclidean graphs (GCN, GraphSAGE).

### Chapter 3: Methodology & Theoretical Framework
*   Synchronized framework outline (`Chapter_3_Framework_and_Methodology_Outline.pdf`) detailing:
    1. Study Area (Davao City administrative boundary and climate classification).
    2. Data Acquisition (GEE Landsat 8 L2SR + ECOSTRESS LST).
    3. Adaptive Graph Engineering (DPWH buffers, 30m interpolation).
    4. Graph Adjacency Formulations (Road Network vs. KNN-$k5$ vs. Identity/Self-Loops).
    5. Spatio-Temporal Model Specification (LSTM temporal encoder + GraphSAGE spatial aggregator).
    6. Evaluation Metrics ($R^2$, RMSE, MAE, Spatial Autocorrelation Moran's $I$).

---

## 8. Summary of Active Codebase Structure

```
THESIS/
├── comprehensive_summary.md               # Complete project overview (this document)
├── professor_comments_v2_notes.md         # Faculty feedback notes & implementation text
├── Chapter_3_Framework_and_Methodology_Outline.pdf # Formal methodology manuscript outline
├── V2 Chapter 1 Revised.pdf               # Revised Chapter 1 submission
├── v2_adaptive_graph/
│   ├── 01_build_adaptive_graph.py         # OSM download, DPWH buffering, 30m interpolation
│   ├── 02_extract_adaptive_features.py    # Macro (100m) & Micro (DPWH) raster extraction
│   ├── 03_build_edge_index.py             # Topological graph reconstruction & edge indexing
│   ├── 04_visualize_road_network.py       # PyDeck 3D glowing thermal edge visualization
│   ├── 05_train_baseline_gnn.py           # Static GraphSAGE spatial convergence baseline
│   ├── 06_train_stgcn.py                  # 3D Tensor LSTM + GraphSAGE spatio-temporal model
│   ├── 07_train_random_forest.py          # Scikit-learn tabular machine learning benchmark
│   ├── temporal_cloud_fix.py              # Multi-year >5°C thin cloud scrubber + KD-Tree patch
│   ├── fix_clouds.py / fix_zeros.py       # Anomaly & NaN sanitation utilities
│   ├── edge_index.csv                     # Topological binary adjacency
│   ├── edge_index_weighted.csv            # Distance-weighted metric road adjacency
│   └── edge_index_knn_5.csv               # Geographic proximity (k=5) baseline adjacency
```

---

*Last Updated: October 2026 | Master of Science in Computer Science Thesis Project*
