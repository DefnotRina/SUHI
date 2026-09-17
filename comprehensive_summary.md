# Adaptive Graph for Urban Thermal Analysis: Project Summary

This document provides a comprehensive, unbiased overview of the data engineering pipeline we've developed to transform OpenStreetMap (OSM) road networks into a highly detailed **Adaptive Graph**. This graph is specifically designed to feed into advanced machine learning models, such as Spatial-Temporal Graph Convolutional Networks (ST-GCN), for analyzing Surface Urban Heat Islands (SUHI) via Land Surface Temperature (LST).

---

## 1. What We Did: The Pipeline in Detail

We designed a robust 4-step Python pipeline to download, mathematically interpolate, feature-enrich, and visualize the road network of Davao City.

### Step 1: Building the Adaptive Graph (`01_build_adaptive_graph.py`)
**Goal:** Create a high-resolution spatial graph where nodes are spaced evenly, not just at natural street intersections.

*   **Network Extraction:** We queried OSMnx for the "Davao City" boundary and downloaded the driveable road network. We specifically filtered out non-driveable paths (footways, cycleways, proposed roads, etc.) using custom OSM tags.
*   **Coordinate Projection:** We projected the geographical coordinates (Lat/Lon) to a local UTM coordinate reference system (meters). This is a critical step because mathematical distance calculations and interpolations are wildly inaccurate in degrees.
*   **Hierarchical Buffering (DPWH Standards):** We applied local Philippine Department of Public Works and Highways (DPWH) standards to assign physical buffer radii to roads. A "trunk" road gets a 15-meter buffer, while a "residential" road gets a 5-meter buffer.
*   **Adaptive Interpolation:** Natural OSM graphs only have nodes at intersections. We mathematically interpolated new nodes along every road segment ensuring a maximum gap of 30 meters. This massively increases our spatial resolution. 
*   **Output:** A localized GeoPackage (`davao_adaptive_nodes_utm.gpkg`) containing both intersection nodes and interpolated nodes.

### Step 2: Extracting Adaptive Features (`02_extract_adaptive_features.py`)
**Goal:** Overlay our adaptive nodes onto satellite raster imagery to extract environmental features (like temperature) for each specific point.

*   **Multi-Scale Buffering:** We implemented a two-tier extraction logic:
    *   **Macro-Scale (Thermal):** For Land Surface Temperature (LST), we buffer nodes by a fixed 100 meters, capturing the ambient neighborhood heat.
    *   **Micro-Scale (Optical):** For optical data (like NDVI/greenness), we dynamically buffer nodes based on their exact road width (5m to 15m) to capture exactly what is physically *on* the road.
*   **Batch Processing:** To prevent RAM crashes (GDAL segmentation faults) when querying 310,000+ nodes against heavy TIF files, we chunked the spatial queries into batches of 10,000.
*   **Spatial KD-Tree Imputation:** Satellite data often has missing pixels (NaNs) due to cloud cover or sensor errors. We built a spatial KD-Tree that automatically identifies nodes with missing data and mathematically imputes the value by averaging the 3 closest spatial neighbors.
*   **Output:** A fully attributed CSV (`davao_adaptive_node_features.csv`) and GeoPackage.

### Step 3: Reconstructing the Edge Index (`03_build_edge_index.py`)
**Goal:** Graph Neural Networks (GCNs) require an adjacency matrix or an "Edge Index" (a list of who is connected to whom). We had to reconstruct this for our newly interpolated nodes.

*   **Retracing Interpolation:** We re-ran the mathematical interpolation logic to map exactly how nodes flow from an intersection `U` to intersection `V`.
*   **KD-Tree Mapping:** Because floating-point math can cause tiny coordinate drifts, we used a KD-Tree to spatially map our newly generated edges back to the exact canonical Node IDs (OSMIDs) from Step 1, allowing a tolerance of 2.0 meters.
*   **Output:** A clean `edge_index.csv` containing `source` and `target` columns, ready to be ingested by PyTorch Geometric.

### Step 4: 3D Visualization (`04_visualize_road_network.py`)
**Goal:** Ensure our data is valid by visualizing it geographically.

*   **Data Projection:** We loaded the 310,000 nodes and reprojected them back from UTM to WGS84 (Lat/Lon) for web mapping compatibility.
*   **Thermal Color Mapping:** We mapped the LST data using the 'Inferno' color palette. We specifically calculated the 2nd and 98th percentiles to clip extreme outliers, ensuring the color gradient represents actual urban heat effectively.
*   **PyDeck Rendering:** We utilized PyDeck to render a high-performance 3D Scatterplot over a Mapbox dark theme, allowing us to pan, tilt, and zoom through the "glowing" thermal road network.

---

## 2. What We Didn't Do (And Why)

*   **We didn't build the actual ST-GCN Model (Yet):** 
    *   *Why:* Preparing high-quality, mathematically sound spatial data is 80% of the work. If we fed messy, un-interpolated OSM data into a GCN, the model would fail to capture continuous spatial patterns. The ST-GCN model will be built in the next phase now that the data architecture is stable.
*   **We haven't extracted Optical/NDVI data:**
    *   *Why:* While the script (`02_extract_adaptive_features.py`) is fully programmed to handle Micro-Scale dynamic buffering (5-15m) for optical data, we only tested it on Macro-Scale LST data. Ultra-high-resolution optical processing is computationally expensive and takes massive storage. We verified the logic first with thermal data.
*   **Temporal Sequences (Time-Series) are currently flat:**
    *   *Why:* ST-GCN implies *Spatio-Temporal* data. Currently, our pipeline extracts a single temporal snapshot (e.g., LST for the year 2025). We needed to perfect the spatial graph before introducing the complexity of the temporal dimension (looping over multiple months/years to create a 3D tensor).

## 3. Things We Will Explore Later

1.  **Temporal Stacking:** Upgrading `02_extract_adaptive_features.py` to iterate over a time-series of TIF files to generate a sequence of features for each node, fulfilling the "Temporal" requirement of the ST-GCN.
2.  **Weighted Edge Indices:** Upgrading `03_build_edge_index.py` to include edge weights. Right now, connections are binary (1 if connected, 0 if not). We will explore weighting edges by physical distance or speed limits to give the Graph Neural Network better context.
3.  **PyTorch Geometric Integration:** Compiling `davao_adaptive_node_features.csv` and `edge_index.csv` into a `torch_geometric.data.Data` object for training.
