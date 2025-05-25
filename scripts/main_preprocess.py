import pandas as pd
import numpy as np
import json
from utils import log_dataframe_shape
from data_cleaning import apply_land_purpose_categorization
from join_variables import join_all_variables_for_modelling
from join_hromada_data import join_hromada_data
from join_cropmaps_data import merge_crop_features


def load_land_monitor_data(file_path: str):
    """Load dataset from a CSV file."""
    df = pd.read_csv(file_path,
                     sep=';',
                     encoding='windows-1251',
                     on_bad_lines='warn',
                     parse_dates=['Дата оцінки', 'Дата реєстрації права'],
                     dayfirst=True
                     )
    return df


def rename_columns(df):
    """Rename columns to match the required format."""
    new_column_names = {
        '№ з/п': 'SerialNumber',
        'Кадастровий номер земельної ділянки': 'CadastralNumber',
        'КОАТУУ': 'KOATUU',
        'Площа з.д., га': 'LandAreaHa',
        'область': 'Region',
        'район': 'District',
        'населений пункт': 'Settlement',
        'вулиця': 'Street',
        'Назва ТГ': 'TGName',
        'Цільове призначення': 'LandPurpose',
        'Назва угіддя': 'LandType',
        'Тип угоди': 'AgreementType',
        'Ціна земельної ділянки': 'LandPrice',
        'Форма власності': 'OwnershipForm',
        'ЕДРПОУ': 'EDRPOU',
        'Дата реєстрації права': 'RegistrationDate',
        'Реєстраційний номер права': 'RegistrationNumber',
        'Значення НГО з.д., грн': 'NGOLandValue',
        'Дата оцінки': 'ValuationDate'
    }
    return df.rename(columns=new_column_names)


@log_dataframe_shape
def filter_sale_agreements(df):
    """Filter the dataset to only include sale agreements."""
    return df[df.AgreementType == 'договір купівлі-продажу']


@log_dataframe_shape
def clean_duplicates(df):
    """Clean duplicate entries with identical cadastral numbers and prices."""

    def select_row(group):
        if group['LandPrice'].notna().any():
            return group[group['LandPrice'].notna()].iloc[0]
        return group.iloc[-1]

    df_cleaned = df.groupby(['CadastralNumber', 'RegistrationDate'], group_keys=False).apply(select_row)
    df_cleaned = df_cleaned.drop_duplicates(subset=['CadastralNumber', 'LandPrice'], keep='last')
    df_cleaned = df_cleaned.reset_index(drop=True)
    return df_cleaned


@log_dataframe_shape
def clean_missing_values(df):
    """Clean missing values from specific columns."""
    df = df[df.LandPurpose.notna()]
    df = df[df.OwnershipForm.notna()]
    df = df[df.NGOLandValue.notna()]
    df = df[df.LandPrice.notna()]
    return df


def convert_columns(df):
    """Convert columns to the correct data types."""
    numeric_columns = ['LandAreaHa', 'LandPrice', 'OwnershipForm', 'NGOLandValue']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df.KOATUU = df.KOATUU.astype(str)
    df.LandPurpose = df.LandPurpose.astype(str)
    return df


def add_edprou_feature(df):
    df['edprou_indicator'] = np.where(df.EDRPOU.notna(), 1, 0)
    return df


def add_time_features(df):
    """Add time-based features such as year, month, week, and day."""
    df['Year'] = df['RegistrationDate'].dt.year
    df['Month'] = df['RegistrationDate'].dt.month
    df['WeekOfYear'] = df['RegistrationDate'].dt.isocalendar().week
    df['DayOfMonth'] = df['RegistrationDate'].dt.day
    return df


def get_season(date):
    """Return the season based on the month."""
    month = date.month
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'


def add_season_feature(df):
    """Add the season feature based on the registration date."""
    df['season'] = df['RegistrationDate'].apply(get_season)
    return df


def encode_cyclic_features(df, col, max_val):
    """Encode cyclic features like month, week, and day using sine and cosine."""
    df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df


def load_koatuu_data(file_path):
    """Load KOATUU geolocation data."""
    with open(file_path) as f:
        geo_dict = json.load(f)
    geo_df = pd.DataFrame(geo_dict)
    geo_df['level1'] = geo_df['Перший рівень'].apply(lambda x: str(x)[:2])
    geo_df['level2'] = geo_df['Другий рівень'].apply(lambda x: str(x)[2:5])
    geo_df['level3'] = geo_df['Третій рівень'].apply(lambda x: str(x)[5:7])
    geo_df['level4'] = geo_df['Четвертий рівень'].apply(lambda x: str(x)[7:])
    geo_df = geo_df.rename(columns={"Назва об'єкта українською мовою": 'name',
                                    'Категорія': 'category'})
    return geo_df


@log_dataframe_shape
def merge_geo_data(df, geo_df):
    """Merge geo data with the main dataframe."""
    df['cadaster_code'] = df['CadastralNumber'].apply(lambda x: x.split(':')[0])
    df['level1'] = df.cadaster_code.apply(lambda x: str(x)[:2])
    df['level2'] = df.cadaster_code.apply(lambda x: str(x)[2:5])
    df['level3'] = df.cadaster_code.apply(lambda x: str(x)[5:7])
    df['level4'] = df.cadaster_code.apply(lambda x: str(x)[7:])

    level1_df = geo_df[(geo_df.level2 == '') & (geo_df.level3 == '') & (geo_df.level4 == '')]
    df = df.merge(level1_df[['level1', 'name']]).rename(columns={'name': 'level1_name'})

    level2_df = geo_df[(geo_df.level3 == '') & (geo_df.level4 == '') & (geo_df.level2 != '')]

    words_to_remove = ["МІСТА", "РАЙОНИ"]
    pattern = "|".join(words_to_remove)

    level2_df = level2_df[~level2_df.name.str.contains(pattern, case=False, regex=True)]

    df = df.merge(level2_df[['level1', 'level2', 'name']]).rename(columns={'name': 'level2_name'})

    level3_df = geo_df[(geo_df.level4 == '') & (geo_df.level2 != '') & (geo_df.level3 != '')]
    level3_df_unique = level3_df.drop_duplicates(subset=['level1', 'level2', 'level3'])
    words_to_remove = ["СІЛЬРАДИ", "ПІДПОРЯДКОВАНІ", "СЕЛИЩА МІСЬКОГО ТИПУ", "НАСЕЛЕНI ПУНКТИ", "МІСТА", "РАЙОНИ"]
    pattern = "|".join(words_to_remove)
    level3_df_unique = level3_df_unique[~level3_df_unique.name.str.contains(pattern, case=False, regex=True)]
    df = df.merge(level3_df_unique[['level1', 'level2', 'level3', 'name']], how='left').rename(
        columns={'name': 'level3_name'})

    level4_df = geo_df[(geo_df.level4 != '') & (geo_df.level2 != '') & (geo_df.level3 != '')]

    df = df.merge(level4_df[['level1', 'level2', 'level3', 'level4', 'name', 'category']], how='left').rename(
        columns={'name': 'level4_name'})
    df[['level4_name', 'level3_name', 'level2_name']] = df[['level4_name', 'level3_name', 'level2_name']].fillna('')
    return df


def construct_full_address(df):
    """Construct the full address from the geo data."""

    def construct_address(row):
        oblast = row["level1_name"].split("/")[0]
        settlement = ""
        for level in ["level4_name", "level3_name", "level2_name"]:
            if row[level]:
                parts = row[level].split("/")
                settlement = parts[1] if len(parts) > 1 else parts[0]
                break
        rayon = ""
        if settlement.startswith("С.") and row["level2_name"]:
            rayon = row["level2_name"].split("/")[0]
        address_parts = [settlement, rayon, oblast]
        return ", ".join(filter(None, address_parts))

    df["full_address"] = df.apply(construct_address, axis=1)
    return df


def preprocess_land_data(file_path, geo_file_path):
    """Main preprocessing function."""
    df = load_land_monitor_data(file_path)
    df = rename_columns(df)
    df = filter_sale_agreements(df)
    df = clean_duplicates(df)
    df = clean_missing_values(df)
    df = convert_columns(df)
    df = add_edprou_feature(df)
    df = add_time_features(df)
    df = add_season_feature(df)
    df = encode_cyclic_features(df, 'Month', 12)
    df = encode_cyclic_features(df, 'WeekOfYear', 53)
    df = encode_cyclic_features(df, 'DayOfMonth', 31)
    df = apply_land_purpose_categorization(df)
    geo_df = load_koatuu_data(geo_file_path)
    df = merge_geo_data(df, geo_df)
    df = construct_full_address(df)
    return df

def construct_dataset_for_modelling():
    # Step 1: Preprocess main land monitoring data
    df = preprocess_land_data(
        "/content/drive/MyDrive/ml_project_land_prices/trans21_clean.csv",
        "/content/drive/MyDrive/ml_project_land_prices/koatuu.json"
    )

    # Step 2: Load auxiliary files (mocked here)
    cadaster_df = pd.read_csv('/content/drive/MyDrive/ml_project_land_prices/cadaster_data.csv')
    # Intended join: cadastral coordinates and polygons per land plot

    crops_df = pd.read_csv('/content/drive/MyDrive/ml_project_land_prices/tiles_crops/processed_features_crops.csv')
    # Intended join: crop composition and land use shares from remote sensing tiles

    frontline_distances = pd.read_csv('/content/drive/MyDrive/ml_project_land_prices/frontline_dists.csv').drop("centroid_coords", axis=1)
    # Intended join: distance from each land plot to occupied, liberated, and pre-2014 zones

    osm_data = pd.read_csv("/content/drive/MyDrive/ml_project_land_prices/interm_roads.csv")
    # Intended join: road proximity and density indicators from OpenStreetMap

    print("Note: Some source files may require API access, large storage, or preprocessing and are not bundled.")
    # Merges go here

    # Step 3: Join hromada-level administrative and socio-economic data
    join_hromada_data()

    # Step 4: Perform all remaining joins and feature construction for modelling
    join_all_variables_for_modelling()

    print("Final dataset constructed. All spatial, temporal, and institutional features joined.")


if __name__ == "__main__":
    construct_dataset_for_modelling()


