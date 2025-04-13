import folium
from shapely.geometry import mapping, Point
import geopandas as gpd
import pandas as pd
import os


# ------------------------
# Surface classification
# ------------------------

ROAD_TYPES = [
    'motorway', 'motorway_link', 'trunk', 'trunk_link',
    'primary', 'primary_link', 'secondary', 'secondary_link',
    'tertiary', 'tertiary_link', 'residential', 'unclassified',
    'living_street', 'service', 'road'
]

PAVED_SURFACES = {
    'asphalt', 'paved', 'concrete', 'concrete:plates', 'concrete:lanes',
    'paving_stones', 'sett', 'paving_stones:trylinka', 'cobblestone',
    'chipseal', 'brick', 'asfalt', 'paved concrete', 'concrete paved',
    'asfalt cu indicatoare', 'asfalt bun', 'asfalt cu denivelari',
    'asfalt fara indicatoare', 'asfalt bun cu indicatoare',
    'paving_stones;asphalt', 'paving_stones,_ground', 'pavement_stones'
}

UNPAVED_SURFACES = {
    'unpaved', 'gravel', 'compacted', 'dirt', 'earth', 'ground', 'sand',
    'fine_gravel', 'mud', 'rock', 'rocks', 'gravel path', 'gravel_path',
    'grass', 'grass_paver', 'dirt;grass', 'gravel; unpaved',
    'gravel;asphalt', 'gravel+dirt', 'ashalpt', 'mixed', 'pebblestone',
    'dirt/sand', 'trail', 'лісовоз', 'грунтове', 'трава'
}


def classify_surface(value):
    if value in PAVED_SURFACES:
        return "paved"
    elif value in UNPAVED_SURFACES:
        return "unpaved"
    else:
        return "unknown"

# ------------------------
# Preprocessing
# ------------------------


def assemble_osm_tiles(tile_folder, tile_indices, file_ext="gpkg", layer=None, deduplicate=True):
    """
    Assembles OSM tiles into a single GeoDataFrame.

    Args:
        tile_folder (str): Folder containing tiles named like 'tile_3/tile_3.gpkg'.
        tile_indices (list): List of tile indices as strings or ints (e.g. ['1', '2.1', 3]).
        file_ext (str): File extension, e.g. 'gpkg' or 'geojson'.
        layer (str or None): Optional layer name if using GPKG with named layers.
        deduplicate (bool): If True, drops duplicates by 'osm_id'.

    Returns:
        GeoDataFrame: Concatenated tiles.
    """
    tile_indices = [str(i) for i in tile_indices]
    tile_gdfs = []

    for idx in tile_indices:
        path = os.path.join(tile_folder, f"tile_{idx}", f"tile_{idx}.{file_ext}")
        try:
            gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
            tile_gdfs.append(gdf)
        except Exception as e:
            print(f"Failed to load {path}: {e}")

    if not tile_gdfs:
        raise ValueError("No valid tiles loaded.")

    all_tiles = gpd.GeoDataFrame(pd.concat(tile_gdfs, ignore_index=True), crs=tile_gdfs[0].crs)

    if deduplicate and "osm_id" in all_tiles.columns:
        all_tiles = all_tiles.drop_duplicates(subset="osm_id")

    return all_tiles


def project_centroids(df, lon_col='longitude', lat_col='latitude'):
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    ).to_crs("EPSG:3857")


def preprocess_osm_water(tiles):
    water_keywords = [
        'river', 'stream', 'canal', 'lake', 'pond', 'basin',
        'reservoir', 'fishpond', 'oxbow', 'Ставок'
    ]
    gdf = tiles.to_crs("EPSG:3857")
    gdf = gdf[
        gdf['water'].isin(water_keywords) |
        gdf['waterway'].isin(water_keywords) |
        gdf['natural'].isin(['water'])
    ]
    return gdf[gdf.geometry.notnull() & gdf.geometry.is_valid]


def preprocess_roads(tiles):
    gdf = tiles[tiles['highway'].isin(ROAD_TYPES)].copy()
    gdf = gdf[gdf.geometry.notnull() & gdf.geometry.is_valid].to_crs("EPSG:3857")
    gdf['surface_class'] = gdf['surface'].apply(classify_surface)
    return gdf

# ------------------------
# Metrics
# ------------------------


def count_features_within_radius(gdf_points, gdf_features, radius_m=10_000):
    sindex = gdf_features.sindex
    counts = []
    for pt in gdf_points.geometry:
        buffer = pt.buffer(radius_m)
        idx = list(sindex.intersection(buffer.bounds))
        match = gdf_features.iloc[idx]
        nearby = match[match.intersects(buffer)].drop_duplicates(subset='geometry')
        counts.append(len(nearby))
    return counts


def distance_to_road_type(gdf_points, gdf_roads, road_types):
    sindex = gdf_roads.sindex
    distances = []
    for pt in gdf_points.geometry:
        idx = list(sindex.intersection(pt.buffer(50_000).bounds))
        match = gdf_roads.iloc[idx]
        subset = match[match['highway'].isin(road_types)]
        distances.append(subset.distance(pt).min() if not subset.empty else None)
    return distances


def road_density(gdf_points, gdf_roads, radius_m=10_000):
    sindex = gdf_roads.sindex
    densities = []
    for pt in gdf_points.geometry:
        buffer = pt.buffer(radius_m)
        idx = list(sindex.intersection(buffer.bounds))
        match = gdf_roads.iloc[idx]
        clipped = gpd.clip(match[match.intersects(buffer)], buffer)
        total_m = clipped.geometry.length.sum()
        area_km2 = 3.1416 * (radius_m / 1000) ** 2
        densities.append(total_m / 1000 / area_km2)
    return densities


def run_all_osm_metrics(cadaster_df, osm_tile_df):
    cadaster_df[['longitude', 'latitude']] = cadaster_df['centroid_coords'].str.strip('()').str.split(', ', expand=True).astype(float)
    gdf_points = project_centroids(cadaster_df)

    # Preprocess features
    gdf_water = preprocess_osm_water(osm_tile_df)
    gdf_roads = preprocess_roads(osm_tile_df)
    gdf_paved = gdf_roads[gdf_roads['surface_class'] == 'paved']

    # Compute metrics
    gdf_points["count_water_features"] = count_features_within_radius(gdf_points, gdf_water)
    gdf_points["dist_to_primary"] = distance_to_road_type(gdf_points, gdf_roads, ['primary', 'primary_link'])
    gdf_points["dist_to_secondary"] = distance_to_road_type(gdf_points, gdf_roads, ['secondary', 'secondary_link'])
    gdf_points["road_density"] = road_density(gdf_points, gdf_roads)
    gdf_points["paved_road_density"] = road_density(gdf_points, gdf_paved)

    return gdf_points


def verify_features_osm(gdf_points, gdf_water, gdf_roads, output_html=None, radius_m=10_000):
    """
    Optional visual check of features around a random cadastre point.

    Args:
        gdf_points (GeoDataFrame): Must have 'longitude' and 'latitude' columns.
        gdf_water (GeoDataFrame): Preprocessed water features (in EPSG:4326 or reprojectable).
        gdf_roads (GeoDataFrame): Preprocessed roads with 'surface_class'.
        output_html (str or None): If set, saves map to this HTML file.
        radius_m (int): Buffer radius in meters.

    Returns:
        folium.Map object (if not saving to file only).
    """
    # Select a random point
    point_row = gdf_points.sample(1).iloc[0]
    lon, lat = point_row.longitude, point_row.latitude
    pt = Point(lon, lat)

    # Reproject point and buffer
    pt_proj = gpd.GeoSeries([pt], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
    buffer_proj = pt_proj.buffer(radius_m)
    buffer = gpd.GeoSeries([buffer_proj], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]
    buffer_gdf_3857 = gpd.GeoDataFrame(geometry=[buffer_proj], crs="EPSG:3857")

    # Reproject features to match buffer
    water_clipped = gpd.clip(gdf_water.to_crs("EPSG:3857"), buffer_gdf_3857).to_crs("EPSG:4326")
    roads_clipped = gpd.clip(gdf_roads.to_crs("EPSG:3857"), buffer_gdf_3857).to_crs("EPSG:4326")

    # Initialize map
    fmap = folium.Map(location=[lat, lon], zoom_start=13)

    # Add point
    folium.Marker(
        [lat, lon],
        tooltip="Cadastre centroid",
        icon=folium.Icon(color="red", icon="star", prefix="fa")
    ).add_to(fmap)

    # Add buffer
    folium.GeoJson(buffer, style_function=lambda x: {"fillOpacity": 0, "color": "gray"}).add_to(fmap)

    # Add water
    for _, row in water_clipped.iterrows():
        folium.GeoJson(mapping(row.geometry), style_function=lambda x: {"color": "blue", "weight": 1}).add_to(fmap)

    # Add roads
    for _, row in roads_clipped.iterrows():
        color = "black" if row.get("surface_class") == "paved" else "orange"
        dash_array = None if color == "black" else "5,5"
        folium.GeoJson(
            mapping(row.geometry),
            style_function=lambda x, c=color, d=dash_array: {"color": c, "weight": 2, "dashArray": d}
        ).add_to(fmap)

    if output_html:
        fmap.save(output_html)
        print(f"Feature verification map saved to {output_html}")

    return fmap


if __name__ == "__main__":
    tile_folder = "/content/drive/MyDrive/ml_project_land_prices/osm_data"
    tile_indices = list(range(3, 15)) + ['2.1', '2.2', '1']

    tiles = assemble_osm_tiles(tile_folder, tile_indices)

    cadaster_df = pd.read_csv('/content/drive/MyDrive/ml_project_land_prices/cadaster_data.csv')
    sample_cadaster = cadaster_df.sample(10)
    gdf_with_metrics = run_all_osm_metrics(sample_cadaster, tiles)
    gdf_water = preprocess_osm_water(tiles)
    gdf_roads = preprocess_roads(tiles)

    run_verification = True
    if run_verification:
        verify_features_osm(
            gdf_points=gdf_with_metrics,
            gdf_water=gdf_water,
            gdf_roads=gdf_roads,
            output_html="verify_features_map.html"
        )
