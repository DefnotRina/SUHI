import streamlit as st
import pandas as pd
import pydeck as pdk
import geopandas as gpd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
from pathlib import Path

# Configure Streamlit page
st.set_page_config(page_title="Davao Node Features Dashboard", layout="wide")
st.title("Davao Urban Features Dashboard")
st.markdown("Compare LST, NDBI, and NDVI across different buffer sizes.")

# Paths
DATA_DIR = Path(__file__).parent
EDGES_FILE = DATA_DIR / "davao_edges_utm.gpkg"

@st.cache_data
def load_node_data(buffer_size):
    """Load and cache the node features CSV."""
    file_path = DATA_DIR / f"davao_node_features_{buffer_size}m.csv"
    if not file_path.exists():
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    # PyDeck expects lon and lat columns for easier mapping
    df = df.rename(columns={'x': 'lon', 'y': 'lat'})
    return df

@st.cache_data
def load_edges_data():
    """Load and cache the road network edges."""
    if not EDGES_FILE.exists():
        st.error(f"Edges file not found: {EDGES_FILE}")
        return gpd.GeoDataFrame()
    # Read edges and project to EPSG:4326 (lat/lon) for pydeck
    edges_gdf = gpd.read_file(EDGES_FILE)
    if edges_gdf.crs and edges_gdf.crs.to_string() != "EPSG:4326":
        edges_gdf = edges_gdf.to_crs(epsg=4326)
    return edges_gdf

# Sidebar controls
st.sidebar.header("Map Controls")

buffer_size = st.sidebar.selectbox(
    "Select Buffer Size",
    options=[15, 20, 30, 40, 50, 100],
    index=3  # Default to 100m
)

# Load Data
df = load_node_data(buffer_size)

if df.empty:
    st.stop()

# Determine available features for this buffer size (format: LST_2023_100m)
available_features = [col for col in df.columns if any(f in col for f in ['LST', 'NDBI', 'NDVI']) and str(buffer_size) in col]

# Extract feature bases (e.g., LST) and years (e.g., 2023)
feature_bases = set()
years = set()
for col in available_features:
    parts = col.split('_')
    if len(parts) >= 3:
        feature_bases.add(parts[0])
        years.add(parts[1])

selected_feature_base = st.sidebar.selectbox(
    "Select Feature to Visualize",
    options=sorted(list(feature_bases)),
    index=0
)

selected_year = st.sidebar.selectbox(
    "Select Year",
    options=sorted(list(years)),
    index=0
)

# Reconstruct actual column name
feature_col = f"{selected_feature_base}_{selected_year}_{buffer_size}m"

show_edges = st.sidebar.checkbox("Show Road Network Edges", value=True)

radius = st.sidebar.slider("Point Radius (meters)", min_value=10, max_value=200, value=50)

# Process colors
st.sidebar.subheader("Color Scale")
cmap_name = st.sidebar.selectbox(
    "Colormap",
    options=["viridis", "plasma", "inferno", "magma", "cividis", "RdYlGn", "RdYlBu", "coolwarm"],
    index=0
)
reverse_cmap = st.sidebar.checkbox("Reverse Colormap", value=False)

# Normalize and color the nodes
if feature_col in df.columns:
    vmin, vmax = df[feature_col].min(), df[feature_col].max()
    
    # Handle NaNs
    df_clean = df.dropna(subset=[feature_col]).copy()
    
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(f"{cmap_name}_r" if reverse_cmap else cmap_name)
    
    def get_color(val):
        # returns (R, G, B, A) in 0-255 range
        rgba = cmap(norm(val))
        return [int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), 200]
        
    df_clean['color'] = df_clean[feature_col].apply(get_color)
    
    # Map layers
    layers = []
    
    # Edges layer
    if show_edges:
        edges_gdf = load_edges_data()
        if not edges_gdf.empty:
            edges_layer = pdk.Layer(
                "GeoJsonLayer",
                data=edges_gdf,
                get_path="geometry",
                get_line_color=[150, 150, 150, 150],
                get_line_width=2,
                width_min_pixels=1,
            )
            layers.append(edges_layer)
            
    # Nodes layer
    nodes_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_clean,
        get_position=["lon", "lat"],
        get_color="color",
        get_radius=radius,
        pickable=True,
    )
    layers.append(nodes_layer)

    # Initial view state
    view_state = pdk.ViewState(
        longitude=df_clean['lon'].mean(),
        latitude=df_clean['lat'].mean(),
        zoom=11,
        pitch=0
    )
    
    # Render map
    st.subheader(f"Map: {feature_col}")
    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v10",
        tooltip={"text": f"OSMID: {{osmid}}\n{feature_col}: {{{feature_col}}}"}
    )
    st.pydeck_chart(r)
    
    # Display stats
    st.subheader("Data Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Min", f"{vmin:.4f}")
    col2.metric("Max", f"{vmax:.4f}")
    col3.metric("Mean", f"{df_clean[feature_col].mean():.4f}")
    col4.metric("Count", len(df_clean))
    
    # Histogram
    st.bar_chart(np.histogram(df_clean[feature_col], bins=50)[0])

else:
    st.error(f"Column '{feature_col}' not found in the dataset.")
