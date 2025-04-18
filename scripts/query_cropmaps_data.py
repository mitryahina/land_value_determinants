import pandas as pd
from collections import Counter
import os
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from PIL import Image
import matplotlib.pyplot as plt
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import mapping
from shapely import wkt

UNMATCHED_TILE_INDEX = 9999


def match_color(color, color_map, tolerance=5):
    for category, ref_color in color_map.items():
        if all(abs(c1 - c2) <= tolerance for c1, c2 in zip(color, ref_color)):
            return category
    return "other"


def get_crop_percentages(image, transform, geometry, categories_map):
    height, width = image.shape[:2]
    mask = rasterize(
        [(geometry, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    masked_pixels = image[mask == 1]
    if masked_pixels.size == 0:
        return {}

    if masked_pixels.shape[-1] == 4:
        masked_pixels = masked_pixels[:, :3]  # drop alpha

    colors = [tuple(int(c) for c in color) for color in masked_pixels]
    categories = [match_color(c, categories_map) for c in colors]
    counts = Counter(categories)
    total = sum(counts.values())

    return {k: v / total * 100 for k, v in counts.items() if total > 0}


def process_tile(tile_index, minx, miny, maxx, maxy, image_path, gdf, categories_map):
    if tile_index == UNMATCHED_TILE_INDEX:
        tile_gdf = gdf.copy()
        return [{
            "plot_id": row.get("id", idx),
            "tile_index": UNMATCHED_TILE_INDEX,
            "unmatched": 100
        } for idx, row in tile_gdf.iterrows()]

    tile_geom = box(minx, miny, maxx, maxy)
    tile_gdf = gdf[gdf.geometry.intersects(tile_geom)].copy()
    if tile_gdf.empty:
        return []

    image = np.array(Image.open(image_path).convert("RGB"))
    height, width = image.shape[:2]
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    results = []
    for idx, row in tile_gdf.iterrows():
        try:
            percentages = get_crop_percentages(image, transform, row.geometry, categories_map)
        except Exception as e:
            print(f"Failed on plot {idx} in tile {tile_index}: {e}")
            percentages = {"unmatched": 100}

        results.append({
            "plot_id": row.get("id", idx),
            "tile_index": tile_index,
            **percentages
        })

    return results


def process_all_tiles(coverage_id, gdf, categories_map,
                      tiles_dir="/content/drive/MyDrive/ml_project_land_prices/tiles_crops/tiles_crops/",
                      tile_bounds_df=None):
    tile_bounds_gdf = tile_bounds_df.copy()
    tile_bounds_gdf["geometry"] = tile_bounds_gdf.apply(
        lambda row: box(row["minx"], row["miny"], row["maxx"], row["maxy"]), axis=1
    )
    tile_bounds_gdf = gpd.GeoDataFrame(tile_bounds_gdf, geometry="geometry", crs="EPSG:4326")

    matched_tiles = set()
    unmatched_indices = []

    for idx, geom in gdf.geometry.items():
        try:
            matches = tile_bounds_gdf[tile_bounds_gdf.intersects(geom)]
            if not matches.empty:
                matched_tiles.update(matches["tile_index"].tolist())
            else:
                unmatched_indices.append(idx)
        except Exception as e:
            print(f"Intersection error on plot {idx}: {e}")
            unmatched_indices.append(idx)

    gdf = gdf.copy()
    gdf["tile_index"] = None
    gdf.loc[unmatched_indices, "tile_index"] = UNMATCHED_TILE_INDEX

    all_results = []

    for _, row in tile_bounds_gdf[tile_bounds_gdf["tile_index"].isin(matched_tiles)].iterrows():
        tile_index = int(row["tile_index"])
        image_path = os.path.join(tiles_dir, f"{coverage_id}_tile_{tile_index}.png")

        if not os.path.exists(image_path):
            print(f"Tile image missing: {image_path}")
            continue

        results = process_tile(tile_index, row["minx"], row["miny"], row["maxx"], row["maxy"], image_path, gdf, categories_map)
        all_results.extend(results)

    # Process unmatched if any
    unmatched_gdf = gdf[gdf.tile_index == UNMATCHED_TILE_INDEX].copy()
    if not unmatched_gdf.empty:
        results = process_tile(UNMATCHED_TILE_INDEX, 0, 0, 0, 0, None, unmatched_gdf, categories_map)
        all_results.extend(results)

    return pd.DataFrame(all_results)


# --------------------------
# Build tile spatial index
# --------------------------

def build_tile_index(tile_bounds_df, coverage_id, tile_dir):
    """
    Converts a DataFrame with bounding box columns into a spatial index GeoDataFrame.

    Args:
        tile_bounds_df (pd.DataFrame): Must have ['tile_index', 'minx', 'miny', 'maxx', 'maxy']
        coverage_id (str): For constructing file names
        tile_dir (str): Folder with downloaded tile images

    Returns:
        GeoDataFrame with geometry and image path
    """
    tile_bounds_df = tile_bounds_df.copy()
    tile_bounds_df["geometry"] = tile_bounds_df.apply(
        lambda row: box(row["minx"], row["miny"], row["maxx"], row["maxy"]), axis=1
    )
    tile_bounds_df["image_path"] = tile_bounds_df["tile_index"].apply(
        lambda i: os.path.join(tile_dir, f"{coverage_id}_tile_{i}.png")
    )
    return gpd.GeoDataFrame(tile_bounds_df, geometry="geometry", crs="EPSG:4326")


# --------------------------
# Verify plots visually + count
# --------------------------


def verify_plots_against_tiles(gdf_small, tile_index_gdf, pad=40):
    """
    For each plot, find the matching tile, count pixels, and show a cropped overlay.

    Args:
        gdf_small (GeoDataFrame): Cadastre subset with geometries.
        tile_index_gdf (GeoDataFrame): Tile bounds + image paths.
        pad (int): Padding (in pixels) around the cropped bounding box.
    """
    for idx, row in gdf_small.iterrows():
        geom = row.geometry

        # Match tile via spatial join
        matched = tile_index_gdf[tile_index_gdf.intersects(geom)]
        if matched.empty:
            print(f"No tile found for plot {idx}")
            continue

        tile = matched.iloc[0]
        image_path = tile.image_path

        if not os.path.exists(image_path):
            print(f"Image missing for tile {tile.tile_index}: {image_path}")
            continue

        image = np.array(Image.open(image_path).convert("RGB"))
        minx, miny, maxx, maxy = tile.minx, tile.miny, tile.maxx, tile.maxy
        height, width = image.shape[:2]
        transform = from_bounds(minx, miny, maxx, maxy, width, height)

        # Rasterize polygon interior (gray fill)
        mask = rasterize(
            [(geom, 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
        overlay = image.copy()
        # Semi-transparent gray overlay
        alpha = 0.5  # 0 = fully transparent, 1 = opaque
        gray = np.array([180, 180, 180], dtype=np.uint8)

        # Blend: new_pixel = alpha * gray + (1 - alpha) * original
        overlay[mask == 1] = (
            alpha * gray + (1 - alpha) * overlay[mask == 1]
        ).astype(np.uint8)
        # overlay[mask == 1] = [180, 180, 180]

        # Rasterize boundary (red border)
        outline = rasterize(
            [(mapping(geom.boundary), 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
        overlay[outline == 1] = [255, 0, 0]

        # Count pixels
        pixel_count = int(np.sum(mask == 1))
        print(f"Plot {idx} → Tile {tile.tile_index} → {pixel_count} pixels")

        # Crop region around the mask
        ys, xs = np.where(mask == 1)
        if len(xs) == 0 or len(ys) == 0:
            print(f"Plot {idx} is too small or not visible")
            continue

        minx_px = max(xs.min() - pad, 0)
        maxx_px = min(xs.max() + pad, overlay.shape[1])
        miny_px = max(ys.min() - pad, 0)
        maxy_px = min(ys.max() + pad, overlay.shape[0])

        cropped = overlay[miny_px:maxy_px, minx_px:maxx_px]

        # Plot cropped overlay
        plt.figure(figsize=(8, 8))
        plt.imshow(cropped)
        plt.title(f"Plot {idx} (Tile {tile.tile_index}) — Pixels: {pixel_count}")
        plt.axis("off")
        plt.show()

# coverage_id = "2023-summer"
# tile_dir = "/content/drive/MyDrive/ml_project_land_prices/tiles_crops/tiles_crops/"
#
# tile_index = build_tile_index(tiles_bounds, coverage_id, tile_dir)

# cadaster_df["geometry"] = cadaster_df["geometry"].apply(wkt.loads)
# gdf = gpd.GeoDataFrame(cadaster_df, geometry="geometry", crs="EPSG:4326")
# results = process_all_tiles(coverage_id, gdf, categories_map, tile_bounds_df=tiles_bounds)
# verify_plots_against_tiles(gdf, tile_index)
