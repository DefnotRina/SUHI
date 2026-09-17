import osmnx as ox
ox.settings.use_cache = True
davao_query = "Davao City, Philippines"
core_districts_gdf = ox.geocode_to_gdf(davao_query)
urban_core_polygon = core_districts_gdf.union_all()
custom_filter = '["highway"]["area"!~"yes"]["highway"!~"cycleway|footway|path|pedestrian|steps|track|corridor|elevator|escalator|proposed|construction|bridleway|abandoned|platform|raceway"]'
G = ox.graph_from_polygon(urban_core_polygon, custom_filter=custom_filter, simplify=True)
nodes, edges = ox.graph_to_gdfs(G)
print(f"Original intersections: {len(nodes)}")
