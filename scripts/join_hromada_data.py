import pandas as pd
import geopandas as gpd
import ast
import re
from shapely.geometry import Polygon, Point

# === Constants ===
UKRAINE_BOUNDS = {
    "lat_min": 44.0,
    "lat_max": 53.0,
    "lon_min": 22.0,
    "lon_max": 41.0,
}

def is_valid_latitude(lat): return UKRAINE_BOUNDS["lat_min"] <= lat <= UKRAINE_BOUNDS["lat_max"]
def is_valid_longitude(lon): return UKRAINE_BOUNDS["lon_min"] <= lon <= UKRAINE_BOUNDS["lon_max"]

def sort_coordinates(coords):
    """Categorize values into latitude or longitude."""
    latitudes, longitudes = [], []
    for coord in coords:
        if is_valid_latitude(coord):
            latitudes.append(coord)
        elif is_valid_longitude(coord):
            longitudes.append(coord)
    return latitudes, longitudes

def parse_geometry_string(geometry_str):
    """Parse and convert cleaned string to Polygon."""
    geometry_str_cleaned = re.sub(r'[^0-9.,\-\[\] ]', '', geometry_str)
    try:
        geometry_data = ast.literal_eval(geometry_str_cleaned)
        if len(geometry_data) % 2 != 0:
            return None
        latitudes, longitudes = sort_coordinates(geometry_data)
        coordinates = list(zip(longitudes, latitudes))
        return Polygon(coordinates) if len(coordinates) >= 3 else None
    except (SyntaxError, ValueError):
        return None

def main():
    # === File paths ===
    geo_data_path = "data/processed/assembled_geo_features.csv"
    hromada_geom_path = "data/raw/hromada_geography.csv"
    hromada_data_path = "data/raw/hromadas_dataset.csv"
    output_path = "data/processed/geo_features_with_hromadas.csv"

    # === Load data ===
    df = pd.read_csv(geo_data_path)
    hr_geom = pd.read_csv(hromada_geom_path)
    hr_df = pd.read_csv(hromada_data_path)

    # === Parse and clean hromada polygons ===
    hr_geom['geometry'] = hr_geom['geometry'].apply(parse_geometry_string)
    hr_geom_gdf = gpd.GeoDataFrame(hr_geom, geometry='geometry', crs='EPSG:4326')

    # === Convert centroid_coords string to Point geometry ===
    df['centroid_point'] = df['centroid_coords'].apply(lambda s: Point(ast.literal_eval(s)))
    locations_gdf = gpd.GeoDataFrame(df, geometry='centroid_point', crs='EPSG:4326')

    # === Project to metric CRS for accuracy ===
    metric_crs = "EPSG:3857"
    hr_geom_gdf = hr_geom_gdf.to_crs(metric_crs)
    locations_gdf = locations_gdf.to_crs(metric_crs)

    # === Spatial join: point-in-polygon ===
    joined = gpd.sjoin(locations_gdf, hr_geom_gdf, how="left", predicate="within")
    joined = joined[['cadnum', 'hromada_code']]

    # === Merge with binary attributes ===
    binary_cols = [
        'mountain_hromada', 'near_seas', 'bordering_hromadas',
        'hromadas_30km_from_border', 'hromadas_30km_russia_belarus',
        'buffer_nat_15km', 'buffer_int_15km'
    ]
    hr_df[binary_cols] = hr_df[binary_cols].fillna(0)

    hromada_enriched = joined.merge(hr_df, on='hromada_code', how='left')

    # === Merge into main dataset and save ===
    final_df = df.merge(hromada_enriched, on='cadnum', how='left')
    final_df.drop(columns=['centroid_point'], inplace=True)

    final_df.to_csv(output_path, index=False)
    print(f"Hromada features joined and saved to {output_path}")


if __name__ == "__main__":
    main()
