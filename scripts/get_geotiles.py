import geopandas as gpd
from shapely.geometry import box, LineString
import folium
from shapely.ops import split


# Step 1: Define bounds for Ukraine (approx)
def define_bounds():
    minx, miny = 22.09, 45.6  # Lower-left corner
    maxx, maxy = 40.11, 52.3  # Upper-right corner
    return minx, miny, maxx, maxy


# Step 2: Grid settings
def create_grid(minx, miny, maxx, maxy, n_cols=4, n_rows=4, overlap_deg=0.15):
    dx = (maxx - minx) / n_cols
    dy = (maxy - miny) / n_rows

    tiles = []
    for i in range(n_cols):
        for j in range(n_rows):
            x0 = minx + i * dx - overlap_deg
            y0 = miny + j * dy - overlap_deg
            x1 = minx + (i + 1) * dx + overlap_deg
            y1 = miny + (j + 1) * dy + overlap_deg
            tile = box(x0, y0, x1, y1)
            tiles.append(tile)

    # Save to GeoDataFrame
    gdf_tiles = gpd.GeoDataFrame(geometry=tiles, crs="EPSG:4326")
    gdf_tiles["tile_id"] = [f"tile_{i}" for i in range(len(tiles))]
    return gdf_tiles


# Step 3: Save GeoDataFrame to GeoJSON for HOT Export Tool
def save_to_geojson(gdf_tiles, filename="ukraine_tiles.geojson"):
    gdf_tiles.to_file(filename, driver="GeoJSON")


# Step 4: Save each tile as a separate GeoJSON file
def save_singular_tile_files(gdf_tiles, output_dir="/content/drive/MyDrive/ml_project_land_prices/tiles"):
    for i, row in gdf_tiles.iterrows():
        tile_id = row['tile_id']
        tile_geom = row.geometry

        gdf_single = gpd.GeoDataFrame(
            {'tile_id': [tile_id]},
            geometry=[tile_geom],
            crs=gdf_tiles.crs
        )

        # Save to GeoJSON
        gdf_single.to_file(f"{output_dir}/{tile_id}.geojson", driver="GeoJSON")


# Step 5: Split polygon into two parts with overlap
def split_polygon_in_half(polygon, direction='vertical', overlap_deg=0.02):
    minx, miny, maxx, maxy = polygon.bounds

    if direction == 'vertical':
        midx = (minx + maxx) / 2
        splitter = LineString([(midx, miny - 1), (midx, maxy + 1)]).buffer(overlap_deg / 2)
    elif direction == 'horizontal':
        midy = (miny + maxy) / 2
        splitter = LineString([(minx - 1, midy), (maxx + 1, midy)]).buffer(overlap_deg / 2)
    else:
        raise ValueError("Direction must be 'vertical' or 'horizontal'")

    # Split the polygon
    split_polys = polygon.difference(splitter).geoms
    overlap_area = polygon.intersection(splitter)

    # Return parts with duplicated overlap added to each side
    parts = []
    for geom in split_polys:
        parts.append(geom.union(overlap_area))

    return parts


# Step 6: Visualize tiles using Folium
def visualize_tiles(gdf_tiles):
    # Get the center of the full area for the map
    center_lat = (gdf_tiles.total_bounds[1] + gdf_tiles.total_bounds[3]) / 2
    center_lon = (gdf_tiles.total_bounds[0] + gdf_tiles.total_bounds[2]) / 2

    # Create a folium map centered on Ukraine
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

    # Add each tile as a polygon layer with a popup
    for _, row in gdf_tiles.iterrows():
        geo_json = folium.GeoJson(row.geometry,
                                  style_function=lambda x: {
                                      'fillColor': 'orange',
                                      'color': 'black',
                                      'weight': 1,
                                      'fillOpacity': 0.1
                                  })
        popup = folium.Popup(f"{row['tile_id']}", parse_html=True)
        geo_json.add_child(popup)
        geo_json.add_to(m)

    return m


# Main function to execute the script
def main():
    minx, miny, maxx, maxy = define_bounds()
    gdf_tiles = create_grid(minx, miny, maxx, maxy)

    # Save grid to GeoJSON for HOT Export Tool
    save_to_geojson(gdf_tiles)

    # Split a polygon in half for testing
    new_polys = split_polygon_in_half(gdf_tiles.geometry.values[2])
    new_polys_df = gpd.GeoDataFrame(geometry=new_polys, crs="EPSG:4326")
    new_polys_df['tile_id'] = ['tile_2.1', 'tile_2.2']
    save_singular_tile_files(new_polys_df)

    # Visualize tiles on a map
    map_object = visualize_tiles(gdf_tiles)
    return map_object


# Run the script
if __name__ == "__main__":
    m = main()
    m.save("ukraine_tiles_map.html")  # Save the map as an HTML file to view in the browser
