import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from libpysal.weights import KNN, lag_spatial
from esda.moran import Moran, Moran_Local
from splot.esda import lisa_cluster
import matplotlib.pyplot as plt
from spreg import GM_Lag, GM_Error, GM_Combo
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# === Load and prepare data ===
gdf_mod_alt = gpd.read_file("data/processed/gdf_mod_alt.gpkg", layer='data')
gdf_mod_alt[['longitude', 'latitude']] = gdf_mod_alt['centroid_coords'].str.extract(r'\(([^,]+), ([^)]+)\)').astype(float)

# === Spatial weights matrix ===
knn_mod_alt = KNN.from_dataframe(gdf_mod_alt, k=10)
knn_mod_alt.transform = 'r'

# === Moran's I ===
y_full = gdf_mod_alt['PricePerHectar_log'].values
moran = Moran(y_full, knn_mod_alt)
print(f"Moran's I: {moran.I}, p-value: {moran.p_sim}")

# === LISA ===
lisa = Moran_Local(y_full, knn_mod_alt)
gdf_mod_alt['lisa_I'] = lisa.Is
gdf_mod_alt['lisa_p'] = lisa.p_sim
gdf_mod_alt['lisa_cluster'] = lisa.q

sig = lisa.p_sim < 0.05
quad = lisa.q
labels = np.full_like(quad, 'ns', dtype=object)
labels[(quad == 1) & sig] = 'HH'
labels[(quad == 2) & sig] = 'LH'
labels[(quad == 3) & sig] = 'LL'
labels[(quad == 4) & sig] = 'HL'
gdf_mod_alt['lisa_label'] = labels

# === Plot ===
fig, ax = plt.subplots(1, figsize=(12, 8))
lisa_cluster(lisa, gdf_mod_alt, p=0.05, ax=ax, markersize=2)
plt.title("LISA Cluster Map")
plt.show()

# === Spatial lag variable ===
gdf_mod_alt['spatial_lag_price'] = lag_spatial(knn_mod_alt, gdf_mod_alt['PricePerHectar_adjusted'])

# === Feature setup ===
features = [
    'distance_to_kyiv_log', 'distance_to_obl_center_log', 'LandAreaHa_log', 'ValuationPerHectar_log', 'is_war',
    'is_legal_market', 'distance_to_eu_log', 'income_log', 'mountain_hromada', 'near_seas', 'urban_pct',
    'population_2022_log', 'inter_area_mount', 'inter_area_sea', 'percent_crops', 'percent_forest',
    'percent_built_up', 'percent_grassland', 'percent_water', 'is_not_cultivated', 'log_count_water_features',
    'log_dist_to_primary_km', 'log_dist_to_secondary_km', 'edprou_indicator', 'paved_road_density', 'road_density',
    'log_closest_dist_to_occupied_km', 'is_urban_hromada', 'has_water', 'no_primary_road_within_10k',
    'distance_to_hromada_center_log', 'hromada_area_log', 'sale_order'
]
features += [c for c in gdf_mod_alt.columns if c.startswith('purpose_') and c != 'purpose_personal_farm']
features += [c for c in gdf_mod_alt.columns if c.startswith('Ownership') and c != 'OwnershipForm_100.0']
features += [c for c in gdf_mod_alt.columns if c.startswith('Quarter')]

X = gdf_mod_alt[features].dropna().astype(float).values
y = gdf_mod_alt.loc[gdf_mod_alt[features].notna().all(axis=1), 'PricePerHectar_log'].values.reshape(-1, 1)

# === VIF check ===
X_scaled = StandardScaler().fit_transform(gdf_mod_alt[features].dropna())
vif_df = pd.DataFrame({
    "Feature": features,
    "VIF": [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
})
print(vif_df.sort_values('VIF', ascending=False))

# === Fit spatial models ===
slm = GM_Lag(y, X, w=knn_mod_alt, name_y="Log Land Price", name_x=features)
sem = GM_Error(y, X, w=knn_mod_alt, name_y="Log Land Price", name_x=features)
combo = GM_Combo(y, X, w=knn_mod_alt, name_y="Log Land Price", name_x=features)

print(slm.summary)
print(sem.summary)
print(combo.summary)

# === Extract results ===
def extract_spreg_results(model, model_type):
    variables = model.name_x.copy()
    z_stats = [z[0] for z in model.z_stat]
    p_values = [z[1] for z in model.z_stat]
    coefs = list(model.betas.flatten())

    if model.constant:
        variables.insert(0, "CONSTANT")

    if model_type == 'slm':
        variables.append("W_Log Land Price")
    elif model_type == 'combo':
        variables += ["W_Log Land Price", "lambda"]
        z_stats += [None, None]
        p_values += [None, None]
    elif model_type == 'sem':
        variables.append("lambda")
        z_stats.append(None)
        p_values.append(None)

    return pd.DataFrame({
        "Variable": variables,
        "Coefficient": coefs,
        "z-Statistic": z_stats,
        "P-Value": p_values
    })

slm_res = extract_spreg_results(slm, 'slm')
sem_res = extract_spreg_results(sem, 'sem')
combo_res = extract_spreg_results(combo, 'combo')

# === Merge results ===
sem_res = sem_res.rename(columns={col: f"{col}_SEM" for col in sem_res.columns if col != "Variable"})
comparison = (
    combo_res.merge(slm_res, on="Variable", suffixes=("_Combo", "_Lag"))
             .merge(sem_res, on="Variable")
)

def stars(p):
    if p < 0.01: return ' (***)'
    elif p < 0.05: return ' (**)'
    elif p < 0.1: return ' (*)'
    else: return ''

for model in ['Combo', 'Lag', 'SEM']:
    col = f"Coefficient_{model}"
    p_col = f"P-Value_{model}"
    if col in comparison.columns and p_col in comparison.columns:
        comparison[col] = comparison[col].round(4).astype(str) + comparison[p_col].apply(stars)

comparison.drop(columns=[c for c in comparison.columns if "P-Value" in c]).to_csv("results/results.csv", index=False)

# === Cross-validation ===
def cross_validate_spatial(gdf, target_col, feature_cols, coord_cols=('longitude', 'latitude'), k=10, folds=5):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    metrics, coefs = [], []

    for i, (train_idx, test_idx) in enumerate(kf.split(gdf)):
        train, test = gdf.iloc[train_idx], gdf.iloc[test_idx]
        fold = pd.concat([train, test]).reset_index(drop=True)
        geo_fold = gpd.GeoDataFrame(fold, geometry=gpd.points_from_xy(fold[coord_cols[0]], fold[coord_cols[1]]))
        knn = KNN.from_dataframe(geo_fold, k=k)
        knn.transform = 'r'

        y_all = fold[target_col].copy()
        y_all.iloc[len(train):] = 0
        spatial_lag = lag_spatial(knn, y_all)
        train['spatial_lag_price'] = spatial_lag[:len(train)]
        test['spatial_lag_price'] = spatial_lag[len(train):]

        used = feature_cols + ['spatial_lag_price']
        model = LinearRegression().fit(train[used], train[target_col])
        pred = model.predict(test[used])

        metrics.append({
            'Fold': i + 1,
            'R²': r2_score(test[target_col], pred),
            'MAE': mean_absolute_error(test[target_col], pred),
            'RMSE': np.sqrt(mean_squared_error(test[target_col], pred))
        })
        coefs.append(pd.Series(model.coef_, index=used, name=f'Fold_{i+1}'))

    return pd.DataFrame(metrics), pd.concat(coefs, axis=1).T

metrics_df, coefs_df = cross_validate_spatial(
    gdf=gdf_mod_alt,
    target_col='PricePerHectar_log',
    feature_cols=features,
    coord_cols=('longitude', 'latitude'),
    k=10,
    folds=5
)
metrics_df.to_csv("results/cv_metrics.csv", index=False)
coefs_df.to_csv("results/cv_coefs.csv")

print("✅ Modeling and cross-validation complete.")
