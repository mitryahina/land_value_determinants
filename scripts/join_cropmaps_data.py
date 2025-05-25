import pandas as pd

# === Load processed tile-level crop features ===
tile_path = "data/processed/processed_features_crops.csv"
results = pd.read_csv(tile_path)

# === Aggregate to plot level ===
results = results.fillna(0).groupby("plot_id").mean(numeric_only=True).reset_index()

# Normalize to percentages
crop_cols = [x for x in results.columns if x not in ['plot_id', 'tile_index']]
results[crop_cols] = results[crop_cols] / 100

# === Define crop groups ===
crop_columns = [
    'Alfalfa', 'Barley', 'Buckwheat', 'Grape', 'Maize',
    'Other crops', 'Peas', 'Potato', 'Rapeseed', 'Soybeans',
    'Sugar beet', 'Sunflower', 'Wheat'
]
forest_column = 'Forest'
water_columns = ['Water', 'Wetland']
not_cultivated_columns = ['Not cultivated', 'Bare land']
built_up_column = 'Artificial'
grassland_column = 'Grassland'

# === Create derived features ===
results['percent_crops'] = results[crop_columns].sum(axis=1)
results['percent_forest'] = results[forest_column]
results['percent_grassland'] = results[grassland_column]
results['percent_water'] = results[water_columns].sum(axis=1)
results['has_water'] = (results[water_columns].sum(axis=1) > 0).astype(int)
results['percent_built_up'] = results[built_up_column]
results['is_not_cultivated'] = (results[not_cultivated_columns].sum(axis=1) > 0.5).astype(int)

# === Merge with main geo_features ===
geo_features_path = "data/processed/assembled_geo_features.csv"
geo_features = pd.read_csv(geo_features_path)

# Drop existing crop columns to avoid duplication
redundant_cols = [col for col in results.columns if col in geo_features.columns]
geo_features = geo_features.drop(columns=redundant_cols)

# Merge on cadnum = plot_id
results = results.rename(columns={'plot_id': 'cadnum'})
geo_features = geo_features.merge(
    results[['cadnum', 'percent_crops', 'percent_forest', 'percent_grassland',
             'percent_water', 'has_water', 'percent_built_up', 'is_not_cultivated']],
    how='left', on='cadnum'
)

# === Save result ===
geo_features.to_csv(geo_features_path, index=False)
print(f"Cropmap features merged and saved to {geo_features_path}")
