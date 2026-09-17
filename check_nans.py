import pandas as pd
df = pd.read_csv('v2_adaptive_graph/davao_adaptive_node_features.csv')
print("Total rows:", len(df))
print("Missing values per column:")
print(df.isna().sum())
