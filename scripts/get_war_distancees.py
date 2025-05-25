import requests
import json
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from geopy.distance import geodesic


def fetch_war_polygons(url='https://deepstatemap.live/api/history/1738446489/geojson'):
    """Fetch war zone polygon data from DeepState API."""
    response = requests.get(url)
    return json.loads(response.text)


def is_polygon_in_ukraine(coord):
    """Check if polygon is within plausible bounds of Ukraine."""
    lat, lon = coord
    return 44 <= lat <= 53 and 22 <= lon <= 41


def extract_polygons(geojson):
    """Extract and classify relevant polygons from geojson."""
    occupied, occupied_2014, liberated = [], [], []

    for feature in geojson['features']:
        if feature['geometry']['type'] != 'Polygon':
            continue

        coords = feature['geometry']['coordinates'][0]
        cleaned_coords = [[lon, lat] for lat, lon, *_ in coords]

        if not is_polygon_in_ukraine(cleaned_coords[0]):
            continue

        name = feature['properties'].get('name', '')
        if 'Окуповано' in name or 'Статус невідомий' in name:
            occupied.append(cleaned_coords)
        elif 'Звільнено' in name:
            liberated.append(cleaned_coords)
        elif 'ОРДЛО' in name or 'Окупований Крим' in name:
            occupied_2014.append(cleaned_coords)

    return occupied, liberated, occupied_2014


def distance_to_polygon_km(point, polygon_coords):
    """Compute geodesic distance from point to polygon (in km)."""
    point_geom = Point(point)
    polygon_geom = Polygon(polygon_coords)
    nearest = nearest_points(polygon_geom, point_geom)[0]
    return geodesic((point_geom.y, point_geom.x), (nearest.y, nearest.x)).km


def add_distance_to_polygons(df, polygons, suffix, coord_col='centroid_coords'):
    """Add closest and average distance to polygon set for each point."""
    closest, average = [], []

    for coords in df[coord_col]:
        distances = [distance_to_polygon_km(coords, poly) for poly in polygons]
        closest.append(min(distances))
        average.append(sum(distances) / len(distances) if distances else None)

    df[f'closest_dist_to_{suffix}'] = closest
    df[f'avg_dist_to_{suffix}'] = average
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/locations_with_centroids.csv")
    df['centroid_coords'] = df.apply(lambda row: (row['Longitude'], row['Latitude']), axis=1)

    # Fetch and parse polygons
    geojson = fetch_war_polygons()
    occupied, liberated, occupied_2014 = extract_polygons(geojson)

    # Compute distances
    df = add_distance_to_polygons(df, occupied, 'occupied')
    df = add_distance_to_polygons(df, liberated, 'liberated')
    df = add_distance_to_polygons(df, occupied_2014, 'occupied_2014')

    # Save result
    df.to_csv("data/processed/locations_with_war_distances.csv", index=False)
