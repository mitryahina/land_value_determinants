import geopandas as gpd
import pandas as pd
import json
from shapely.geometry import Polygon, Point
import numpy as np


def to_polygon(coordinates):
    """Convert nested JSON coordinates into a shapely Polygon."""
    coordinates = json.loads(coordinates)
    return Polygon(coordinates[0][0])


def get_centroid(polygon):
    """Return the centroid coordinates (longitude, latitude)."""
    centroid = polygon.centroid
    return (centroid.x, centroid.y)


def get_area(polygon, crs="EPSG:32635"):
    """Calculate the polygon's area in hectares using projected CRS."""
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    gdf = gdf.to_crs(crs)
    area_ha = gdf.geometry.area.iloc[0] / 10_000  # m² to hectares
    return area_ha


def impute_centroids(df):
    """
    Impute centroids for plots not present in cadastre with Koatuu aprorximate coordinates.
    """
    df = df.copy()

    if "centroid_coords" not in df:
        df["centroid_coords"] = np.nan

    df["is_precise_location"] = df["centroid_coords"].notna()
    df["centroid_coords"] = df.apply(
        lambda row: row["centroid_coords"] if pd.notna(row["centroid_coords"]) else (row["latitude"], row["longitude"]),
        axis=1
    )
    return df


def fix_area(df):
    """Apply rules to correct area values based on polygon-vs-cadaster comparison."""
    area_cad = df["area"]
    area_poly = df["area_by_polygon"]

    use_cadaster = (abs(area_cad - area_poly) <= 0.5)
    poly_more_likely = (area_poly < 0.1) & (area_cad > 2)
    cadaster_more_likely = (area_poly > 500) & (area_cad < 50)
    extreme_discrepancy = (area_cad / area_poly > 100) | (area_poly / area_cad > 100)
    cadaster_sus = area_cad > 370

    corrected_area = area_cad.copy()
    corrected_area[use_cadaster] = area_cad[use_cadaster]
    corrected_area[poly_more_likely] = area_poly[poly_more_likely]
    corrected_area[cadaster_more_likely] = area_cad[cadaster_more_likely]
    corrected_area[extreme_discrepancy] = area_poly[extreme_discrepancy]
    corrected_area[cadaster_sus] = area_poly[cadaster_sus]

    return corrected_area


def process_cadaster_locations(df):
    """Main processing function."""
    df['geometry'] = df['coordinates'].apply(to_polygon)
    df['centroid_coords'] = df['geometry'].apply(get_centroid)
    df['area_by_polygon'] = df['geometry'].apply(get_area)
    df['corrected_area'] = fix_area(df)
    df['is_precise_location'] = 1
    df = impute_centroids(df)
    return df


# Load your data here
# cadaster_locations = pd.read_csv("path_to_input.csv")

# Process and save
# processed = process_cadaster_locations(cadaster_locations)
# processed.to_csv("processed_output.csv", index=False)

