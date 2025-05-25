import pandas as pd


def load_cpi_data(filepath: str) -> pd.DataFrame:
    """
    Load CPI data from Excel file and return the dataframe.
    """
    cpi = pd.read_excel(filepath, sheet_name='data')
    return cpi


def adjust_price_to_base_year(base_value: float, df: pd.DataFrame, cpi_df: pd.DataFrame, price_col: str) -> pd.Series:
    """
    Adjust prices in `price_col` to the given base CPI value.
    """
    df = df.copy()
    df['RegistrationDate'] = pd.to_datetime(df['RegistrationDate'])
    df['Year'] = df['RegistrationDate'].dt.year
    df['Month'] = df['RegistrationDate'].dt.month_name()

    merged_df = pd.merge(df, cpi_df, on=['Year', 'Month'], how='left')
    merged_df['CPI_2010'] = merged_df['CPI_2010'].fillna(method='ffill').fillna(method='bfill')

    return df[price_col] * (base_value / merged_df['CPI_2010'])


def add_inflation_adjusted_prices(df: pd.DataFrame, cpi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds inflation-adjusted price columns to the given DataFrame.
    Expects 'RegistrationDate', 'LandPrice', 'LandAreaHa', 'NGOLandValue' columns to be present.
    """
    df = df.copy()
    df['PricePerHectar'] = df['LandPrice'] / df['LandAreaHa']
    df['ValuationPerHectar'] = df['NGOLandValue'] / df['LandAreaHa']

    # Base CPI from January 2021
    base_cpi = float(cpi_df[(cpi_df.Year == 2021) & (cpi_df.Month == 'January')]['CPI_2010'].values[0])

    for price_col in ['LandPrice', 'PricePerHectar', 'ValuationPerHectar', 'NGOLandValue']:
        df[f"{price_col}_adjusted"] = adjust_price_to_base_year(
            base_value=base_cpi,
            df=df[['RegistrationDate', price_col]].assign(LandAreaHa=df['LandAreaHa']),  # keep shape consistent
            cpi_df=cpi_df,
            price_col=price_col
        )

    return df
