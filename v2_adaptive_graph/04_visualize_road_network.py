#!/opt/homebrew/Caskroom/miniforge/base/bin/python
"""
Step 4: Visualize Road Network
This script validates our data engineering by rendering a 3D thermal map of the extracted features.
It projects the 310,000+ nodes back to WGS84 for web compatibility and maps the LST feature
to an 'Inferno' thermal color palette. PyDeck is used for high-performance rendering.
"""
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import matplotlib.cm as cm
import matplotlib.colors as colors
from pathlib import Path
import os

def main():
    print("==========================================")
    print("      DAVAO GLOWING ROAD NETWORK MAP      ")
    print("==========================================")
    
    script_dir = Path(__file__).parent
    features_csv_path = script_dir / "davao_adaptive_node_features.csv"
    
    print(f"[*] Loading 310,000 nodes from '{features_csv_path.name}'...")
    df = pd.read_csv(features_csv_path, usecols=["osmid", "x", "y", "LST_2025_100m"])
    
    print("[*] Reprojecting UTM coordinates to WGS84 for PyDeck...")
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:32651")
    gdf = gdf.to_crs(epsg=4326)
    df["lon"] = gdf.geometry.x
    df["lat"] = gdf.geometry.y
    
    print("[*] Generating Thermal Colors (Inferno Palette)...")
    # Use quantiles to ignore extreme outliers
    vmin = df['LST_2025_100m'].quantile(0.02)
    vmax = df['LST_2025_100m'].quantile(0.98)
    
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    # Using 'inferno' - black -> purple -> orange -> yellow/white
    try:
        cmap = cm.get_cmap('inferno')
    except AttributeError:
        # For newer matplotlib versions
        import matplotlib as mpl
        cmap = mpl.colormaps['inferno']
        
    def get_color(val):
        rgba = cmap(norm(val))
        return [int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), 200]
        
    df['color'] = df['LST_2025_100m'].apply(get_color)
    
    print("[*] Loading Edge Connections...")
    edges_csv_path = script_dir / "edge_index_weighted.csv"
    edges_df = pd.read_csv(edges_csv_path)
    
    print("[*] Merging WGS84 coordinates into edges...")
    # Merge source coordinates
    edges_df = edges_df.merge(df[['osmid', 'lon', 'lat']], left_on='source', right_on='osmid', how='inner')
    edges_df.rename(columns={'lon': 'src_lon', 'lat': 'src_lat'}, inplace=True)
    edges_df.drop('osmid', axis=1, inplace=True)
    
    # Merge target coordinates
    edges_df = edges_df.merge(df[['osmid', 'lon', 'lat']], left_on='target', right_on='osmid', how='inner')
    edges_df.rename(columns={'lon': 'tgt_lon', 'lat': 'tgt_lat'}, inplace=True)
    edges_df.drop('osmid', axis=1, inplace=True)
    
    print("[*] Generating Edge Colors (based on distance)...")
    def get_edge_color(dist):
        if dist < 20.0: return [255, 255, 255, 80] # Short edges (white)
        if dist < 60.0: return [255, 165, 0, 80]   # Medium edges (orange)
        return [255, 0, 0, 80]                     # Long edges (red)
        
    edges_df['color'] = edges_df['distance_m'].apply(get_edge_color)
    
    print("[*] Rendering PyDeck Map...")
    
    # Define a layer to display on a map
    layer = pdk.Layer(
        "ScatterplotLayer",
        df,
        pickable=True,
        opacity=0.8,
        stroked=False,
        filled=True,
        radius_scale=1,
        radius_min_pixels=1,
        radius_max_pixels=5,
        line_width_min_pixels=1,
        get_position="[lon, lat]",
        get_radius=15, # 15 meters physical radius
        get_fill_color="color",
    )
    
    # Define LineLayer for the physical edges
    line_layer = pdk.Layer(
        "LineLayer",
        edges_df,
        get_source_position="[src_lon, src_lat]",
        get_target_position="[tgt_lon, tgt_lat]",
        get_color="color",
        get_width=2,
        pickable=True,
        opacity=0.6,
    )
    
    # Set the viewport location
    view_state = pdk.ViewState(
        longitude=df["lon"].median(),
        latitude=df["lat"].median(),
        zoom=12,
        min_zoom=5,
        max_zoom=18,
        pitch=45,
        bearing=0
    )
    
    # Render
    r = pdk.Deck(
        layers=[line_layer, layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip={"text": "LST 2025: {LST_2025_100m}°C"}
    )
    
    out_html = script_dir / "suhi_map.html"
    r.to_html(str(out_html))
    print(f"[+] Map successfully saved to: {out_html.name}")
    print("[+] Open this HTML file in Chrome/Safari to view the glowing road network!")

if __name__ == "__main__":
    main()
