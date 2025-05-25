import pandas as pd
import rasterio
from geopy.distance import geodesic
from rasterio.transform import Affine


class SoilTypeLookup:
    def __init__(self, bil_file, prj_file=None):
        """Initialize and load raster data."""
        self.bil_file = bil_file
        self.prj_file = prj_file
        self._load_raster()

    def _load_raster(self):
        with rasterio.open(self.bil_file) as src:
            self.soil_types = src.read(1)
            self.transform = src.transform
            self.width = src.width
            self.height = src.height
            self.nodata = src.nodata
            self.crs = src.crs

    def latlon_to_raster_coords(self, lat, lon):
        x, y = ~self.transform * (lon, lat)
        return int(round(x)), int(round(y))

    def get_soil_type_at_location(self, lat, lon):
        x, y = self.latlon_to_raster_coords(lat, lon)
        if 0 <= x < self.width and 0 <= y < self.height:
            val = self.soil_types[y, x]
            return val if val != self.nodata else None
        return None

    def get_soil_type_at_multiple_locations(self, coords):
        return [self.get_soil_type_at_location(lat, lon) for lat, lon in coords]


def extract_coords(coord_str):
    """Convert '(lon, lat)' string to (lat, lon) tuple."""
    lon, lat = map(float, coord_str.strip("()").split(", "))
    return lat, lon


if __name__ == "__main__":
    # === File paths ===
    geo_data_path = "data/processed/assembled_geo_features.csv"
    output_path = "data/processed/assembled_geo_features_with_soil.csv"
    soil_raster_path = "data/external/HWSD2.bil"
    soil_prj_path = "data/external/HWSD2.prj"
    smu_excel = "data/external/HWSD2_LAYERS.xlsx"
    wrb_map_excel = "data/external/D_WRB2.xlsx"

    # === Load data ===
    df = pd.read_csv(geo_data_path)
    df['coord_tuple'] = df['centroid_coords'].apply(extract_coords)

    # === Get soil type values ===
    print("Loading raster and assigning soil type IDs...")
    soil_lookup = SoilTypeLookup(soil_raster_path, soil_prj_path)
    df['soil_type'] = soil_lookup.get_soil_type_at_multiple_locations(df['coord_tuple'])

    # === Map SMU ID to WRB2 dominant types ===
    smu = pd.read_excel(smu_excel)
    smu_map = pd.read_excel(wrb_map_excel)

    dominant_soil = (
        smu.groupby("HWSD1_SMU_ID")
        .apply(lambda group: group.sort_values("SHARE", ascending=False).iloc[0])
        .reset_index(drop=True)[["HWSD1_SMU_ID", "WRB2"]]
    )

    smu_map = dominant_soil.merge(
        smu_map.rename(columns={"CODE": "WRB2"}), how="left"
    ).rename(columns={"HWSD1_SMU_ID": "soil_type", "Value": "soil_name"}).drop(columns=["WRB2"])

    # === Join with main data ===
    df = df.merge(smu_map, how="left", on="soil_type")

    # === Save output ===
    df.drop(columns=['coord_tuple']).to_csv(output_path, index=False)
    print(f"Saved with soil features to: {output_path}")
