import pandas as pd
import numpy as np


def categorize_land_purpose(value: str) -> str:
    """
    Map raw land purpose string to a standardized category code.
    Returns '' if no match found.
    """
    if pd.isna(value):
        return ''

    value = value.lower()

    if '01.01' in value or 'товарного' in value or '1.1' in value:
        return '01.01'
    elif '01.02' in value or 'фермерського' in value or 'для ведення фермерського господарства' in value or '1.2' in value:
        return '01.02'
    elif '01.03' in value or 'селянського' in value or 'для ведення особистого селянського господарства' in value or '01,03' in value:
        return '01.03'
    elif '01.04' in value or 'підсобного сільського' in value or 'підсобного господарства' in value:
        return '01.04'
    elif '01.05' in value or 'індивідуального садівництва' in value or 'ведення садівництва' in value:
        return '01.05'
    elif '01.06' in value or 'колективного садівництва' in value:
        return '01.06'
    elif '01.07' in value:
        return '01.07'
    elif '01.08' in value or '1.8' in value:
        return '01.08'
    elif '02.01' in value or 'будівництва і обслуговування житлового' in value or '2.1' in value:
        return '02.01'
    elif '02.02' in value or 'колективного житлового будівництва' in value or '2.2' in value:
        return '02.02'
    elif '03.07' in value:
        return '03.07'
    elif '07.03' in value:
        return '07.03'
    elif '02.05' in value or '2.5' in value:
        return '02.05'
    elif (
            'землі запасу резервного фонду' in value or
            '16.0' in value or '17.0' in value or '18.0' in value or '19.0' in value or
            '16 землі запасу' in value or 'k.16' in value
    ):
        return 'SectionK'
    else:
        return ''


def apply_land_purpose_categorization(df: pd.DataFrame, col: str = 'LandPurpose') -> pd.DataFrame:
    """
    Adds a column 'FilteredLandPurpose' to the DataFrame using the categorization logic.
    """
    df = df.copy()
    df['FilteredLandPurpose'] = df[col].apply(categorize_land_purpose)
    return df
