import pandas as pd
import numpy as np

# NA in these columns means "does not have" — not missing data
NAN_MEANS_NONE = [
    "Alley", "PoolQC", "Fence", "MiscFeature", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]

QUAL_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}

ORDINAL_QUAL_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
    "HeatingQC", "KitchenQual", "FireplaceQu",
    "GarageQual", "GarageCond", "PoolQC",
]

BSMT_EXPOSURE_MAP = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0}
BSMT_FIN_TYPE_MAP = {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0}
GARAGE_FINISH_MAP = {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0}


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fill "no feature" NaNs with string "None" before encoding
    for col in NAN_MEANS_NONE:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # Ordinal quality encoding: Ex=5 ... Po=1, None=0
    for col in ORDINAL_QUAL_COLS:
        if col in df.columns:
            df[col] = df[col].map(QUAL_MAP).fillna(0).astype(int)

    if "BsmtExposure" in df.columns:
        df["BsmtExposure"] = df["BsmtExposure"].map(BSMT_EXPOSURE_MAP).fillna(0).astype(int)
    for col in ["BsmtFinType1", "BsmtFinType2"]:
        if col in df.columns:
            df[col] = df[col].map(BSMT_FIN_TYPE_MAP).fillna(0).astype(int)
    if "GarageFinish" in df.columns:
        df["GarageFinish"] = df["GarageFinish"].map(GARAGE_FINISH_MAP).fillna(0).astype(int)

    # Aggregate area feature: total livable square footage
    df["TotalSF"] = (
        df.get("TotalBsmtSF", pd.Series(0, index=df.index)).fillna(0)
        + df.get("1stFlrSF", pd.Series(0, index=df.index)).fillna(0)
        + df.get("2ndFlrSF", pd.Series(0, index=df.index)).fillna(0)
    )

    # Total bathrooms (half baths count as 0.5)
    df["TotalBath"] = (
        df.get("FullBath", pd.Series(0, index=df.index)).fillna(0)
        + 0.5 * df.get("HalfBath", pd.Series(0, index=df.index)).fillna(0)
        + df.get("BsmtFullBath", pd.Series(0, index=df.index)).fillna(0)
        + 0.5 * df.get("BsmtHalfBath", pd.Series(0, index=df.index)).fillna(0)
    )

    # Age at time of sale
    yr_sold = df.get("YrSold", pd.Series(2010, index=df.index)).fillna(2010)
    df["HouseAge"] = yr_sold - df["YearBuilt"].fillna(df["YearBuilt"].median() if "YearBuilt" in df.columns else 1975)
    df["RemodAge"] = yr_sold - df["YearRemodAdd"].fillna(df["YearRemodAdd"].median() if "YearRemodAdd" in df.columns else 1975)

    # Binary presence flags
    df["HasPool"] = (df.get("PoolArea", pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    df["HasGarage"] = (df.get("GarageArea", pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    df["HasFireplace"] = (df.get("Fireplaces", pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    df["HasBsmt"] = (df.get("TotalBsmtSF", pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)

    return df
