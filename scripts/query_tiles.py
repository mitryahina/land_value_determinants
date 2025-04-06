import geopandas as gpd
import pandas as pd

# Load tiles
tiles = gpd.GeoDataFrame(pd.concat([
    gpd.read_file(r"C:\Users\y.mitriakhina\thesis\data\osm_data\tile_1\tile_1.gpkg"),
]), crs="EPSG:4326")
