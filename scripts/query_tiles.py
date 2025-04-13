import geopandas as gpd

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
