import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2


def extract_coords(coord_str):
    lon, lat = map(float, coord_str.strip('()').split(', '))
    return lat, lon


def haversine_distance(lat1, lon1, lat2, lon2):
    if None in [lat1, lon1, lat2, lon2]:
        return None
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return 6371.0 * c


def map_land_purpose(purpose):
    purpose_map = {
        'personal_farm': ['01.03', '01.04'],
        'commodity_farm': ['01.01'],
        'commercial_farm': ['01.02'],
        'gardening': ['01.05', '01.06'],
        'other': ['SectionK', '02.01', '02.05', '02.02']
    }
    for category, values in purpose_map.items():
        if purpose in values:
            return category
    return 'other'


def remove_outliers_percentile(df, cols):
    for col in cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df


def add_variables(df):
    df = pd.get_dummies(df, columns=['purpose', 'Quarter', 'OwnershipForm', 'soil_name'], drop_first=False)
    df['is_war'] = (pd.to_datetime(df.RegistrationDate) > pd.to_datetime("2022-02-24")).astype(int)
    df['is_legal_market'] = (pd.to_datetime(df.RegistrationDate) > pd.to_datetime("2024-01-01")).astype(int)

    df['ValuationPerHectar_log'] = np.log1p(df.ValuationPerHectar_adjusted)
    df['PricePerHectar_log'] = np.log1p(df.PricePerHectar_adjusted)
    df['log_closest_dist_to_occupied_km'] = np.log1p(df['is_war'] * df['closest_distance_to_polygon_occupied_km'])

    df['LandAreaHa_log'] = np.log1p(df.LandAreaHa)
    df['distance_to_kyiv_log'] = np.log1p(df.distance_to_kyiv)
    df['distance_to_obl_center_log'] = np.log1p(df.distance_to_obl_center)
    df['distance_to_hromada_center_log'] = np.log1p(df.distance_to_hromada_center)

    df['distance_to_russia_belarus_log'] = np.log1p(df.distance_to_russia_belarus)
    df['distance_to_russia_log'] = np.log1p(df.distance_to_russia)
    df['distance_to_eu_log'] = np.log1p(df.distance_to_eu)
    df['area_log'] = np.log1p(df.area)
    df['population_2022_log'] = np.log1p(df.total_popultaion_2022)
    df['total_declarations'] = np.log1p(df.total_declarations)

    df['inter_area_eu_dist'] = df['LandAreaHa_log'] * df['distance_to_eu_log']
    df['inter_area_mount'] = df['LandAreaHa_log'] * df['mountain_hromada']
    df['inter_area_sea'] = df['LandAreaHa_log'] * df['near_seas']
    df['area_log_sqr'] = df['LandAreaHa_log'] ** 2
    df['log_population'] = np.log1p(df['total_popultaion_2022'])
    df['hromada_area_log'] = np.log1p(df.hromada_area)

    for col in ['income_total_2021', 'income_total_2022']:
        df[f"{col}_log"] = np.log1p(df[col])

    df['income_log'] = df.apply(
        lambda row: row['income_total_2021_log'] if pd.to_datetime(row['RegistrationDate']).year < 2022 else row['income_total_2022_log'],
        axis=1
    )

    ownership_mapping = {
        'OwnershipForm_100.0': 'Ownership_FarmEnterprise',
        'OwnershipForm_200.0': 'Ownership_CorporateEntity',
        'OwnershipForm_300.0': 'Ownership_Cooperative',
        'OwnershipForm_400.0': 'Ownership_StateCommunal'
    }
    df.rename(columns=ownership_mapping, inplace=True)

    return df


def main():
    df = pd.read_csv("data/processed/geo_features_with_all_sources.csv")

    df = df[df.geometry.notna() & df.hromada_code.notna() & df['income_total_2021'].notna()]
    df['is_urban_hromada'] = (df['type'] == 'міська').astype(int)
    df = df.rename(columns={"square": "hromada_area"})

    df['no_primary_road_within_10k'] = df['dist_to_primary'].isna().astype(int)
    df['dist_to_primary'].fillna(10_000, inplace=True)
    df['dist_to_secondary'].fillna(10_000, inplace=True)

    df['log_dist_to_primary_km'] = np.log1p(df['dist_to_primary'] / 1000)
    df['log_dist_to_secondary_km'] = np.log1p(df['dist_to_secondary'] / 1000)
    df['log_count_water_features'] = np.log1p(df['count_water_features'])

    df['distance_to_kyiv'] = df['centroid_coords'].apply(lambda x: haversine_distance(*extract_coords(x), 50.4501, 30.5234))

    oblast_centers = pd.read_csv('data/raw/oblast_centers.csv')
    oblast_centers.rename(columns={'full_address': 'level1_name', 'latitude': 'latitude_obl', 'longitude': 'longitude_obl'}, inplace=True)
    df = df.merge(oblast_centers, how='left')

    df['distance_to_obl_center'] = df.apply(lambda row: haversine_distance(*extract_coords(row['centroid_coords']), row['latitude_obl'], row['longitude_obl']), axis=1)
    df['distance_to_hromada_center'] = df.apply(lambda row: haversine_distance(*extract_coords(row['centroid_coords']), row['lat_center'], row['lon_center']), axis=1)

    df['Quarter'] = pd.to_datetime(df['RegistrationDate']).dt.quarter
    df = df[df['FilteredLandPurpose'].notna() & df['ValuationPerHectar'].notna()]
    df['purpose'] = df['FilteredLandPurpose'].apply(map_land_purpose)

    df = remove_outliers_percentile(df, ['PricePerHectar', 'ValuationPerHectar'])
    df = add_variables(df)

    df.to_csv("data/processed/features_final.csv", index=False)
    print("✅ Feature engineering complete. Output saved to data/processed/features_final.csv")


if __name__ == "__main__":
    main()
