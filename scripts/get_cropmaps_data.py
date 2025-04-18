import os
import numpy as np
import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import time
import os

# ------------------------------
# 1. Generate tile bounding boxes
# ------------------------------

def generate_tiles(bbox, step=0.5):
    minx, miny, maxx, maxy = bbox
    lon_steps = np.arange(minx, maxx, step)
    lat_steps = np.arange(miny, maxy, step)
    return [[lon, lat, min(lon + step, maxx), min(lat + step, maxy)]
            for lon in lon_steps for lat in lat_steps]

# ------------------------------
# 2. Download one high-res WMS tile
# ------------------------------

def download_highres_wms(coverage_id, bbox, crs, output_filename, width=2048, height=2048, format="image/png"):
    wms_url = "https://ukraine-cropmaps.com/geoserver/ukraine/wms"
    params = {
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetMap",
        "layers": coverage_id,
        "styles": "",
        "format": format,
        "transparent": "true",
        "width": width,
        "height": height,
        "crs": crs,
        "bbox": ",".join(map(str, bbox))
    }
    headers = {"User-Agent": "CropmapTileDownloader/1.0"}

    response = requests.get(wms_url, params=params, stream=True, headers=headers)
    if response.status_code == 200:
        with open(output_filename, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Downloaded {output_filename}")
    else:
        print(f"Failed to download: {response.status_code} — {response.text}")

# ------------------------------
# 3. Download all tiles
# ------------------------------

def download_tiles(coverage_id, bbox, tile_step=0.5,
                   output_dir="/content/drive/MyDrive/ml_project_land_prices/tiles_crops",
                   crs="CRS:84", width=2048, height=2048, format="image/png"):
    tiles = generate_tiles(bbox, step=tile_step)
    os.makedirs(output_dir, exist_ok=True)

    for i, tile_bbox in enumerate(tiles):
        if i <= 300:
            output_path = os.path.join(output_dir, f"{coverage_id}_tile_{i}.png")
            if os.path.exists(output_path):
                print(f"Skipping tile {i}: already exists.")
                continue

            print(f"Downloading tile {i + 1}/{len(tiles)}: {tile_bbox}")
            try:
                download_highres_wms(coverage_id, tile_bbox, crs, output_path, width, height, format)
            except Exception as e:
                print(f"Error on tile {i}: {e}")
                time.sleep(30)

    return tiles

# ------------------------------
# 4. Download legend image
# ------------------------------

def download_legend(coverage_id, output_path="legend.png"):
    legend_url = (
        f"https://ukraine-cropmaps.com/geoserver/ukraine/ows?"
        f"service=WMS&version=1.3.0&request=GetLegendGraphic&format=image/png&layer={coverage_id}"
    )
    response = requests.get(legend_url)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print("Legend downloaded.")
        return Image.open(output_path)
    else:
        print(f"Failed to download legend: {response.status_code}")
        return None

# ------------------------------
# 5. Extract and visualize legend colors
# ------------------------------

def extract_legend_colors(image_np, margin_left=5):
    left_column = image_np[:, margin_left]
    unique_colors = []

    for color in left_column:
        color = [int(x) for x in color]
        if color != [255, 255, 255] and color not in unique_colors:
            unique_colors.append(color)
    return unique_colors

def visualize_colors(colors):
    plt.figure(figsize=(4, len(colors)))
    for i, color in enumerate(colors):
        normalized = np.array(color) / 255
        plt.fill_between([0, 1], i, i + 1, color=normalized)
    plt.axis('off')
    plt.show()

def match_color_to_crop(colors):
    categories = [
        'Artificial', 'Wheat', 'Rapeseed', 'Buckwheat',
        'Maize', 'Sugar beet', 'Sunflower', 'Soybeans',
        'Other crops', 'Forest', 'Grassland', 'Bare land',
        'Water', 'Wetland', 'Barley', 'Peas', 'Alfalfa',
        'Gardens, parks', 'Grape', 'Potato',
        'Not cultivated', 'Damaged forest'
    ]

    # Match colors to labels
    if len(colors) != len(categories):
        print(f"⚠️ Legend color count ({len(colors)}) does not match category count ({len(categories)})")
    else:
        categories_map = dict(zip(categories, colors))
        print("categories_map =")
        for k, v in categories_map.items():
            print(f"  {k}: {v}")
    return categories_map


coverage_id = "2023-summer"
ukraine_bbox = [21.3, 44.0, 40.7, 52.4]

# 1. Download tiles
# tiles_bounds = download_tiles(coverage_id, bbox=ukraine_bbox)
tiles_bounds = pd.read_csv("/content/drive/MyDrive/ml_project_land_prices/tiles_crops/bounding_boxes.csv")

# 2. Download and show legend
legend_img = download_legend(coverage_id)
if legend_img:
    legend_np = np.array(legend_img)
    legend_colors = extract_legend_colors(legend_np)
    visualize_colors(legend_colors)
    categories_map = match_color_to_crop(legend_colors)
    print("Extracted legend colors:", legend_colors)
