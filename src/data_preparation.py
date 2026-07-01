"""Data preparation: cleaning, mapping, and feature engineering."""

import numpy as np
import pandas as pd


def clean_categories(df):
    """Fix inconsistent category codes in escolaridade and estado_civil.

    - escolaridade: maps 0, 5, 6 -> 4 (others)
    - estado_civil: maps 0 -> 3 (others)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with raw integer-coded columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with corrected category codes.
    """
    df = df.copy()
    df["escolaridade"] = df["escolaridade"].replace({0: 4, 5: 4, 6: 4})
    df["estado_civil"] = df["estado_civil"].replace({0: 3})
    return df


def map_categorical_labels(df):
    """Map integer codes to descriptive string labels.

    - sexo: 1 -> 'male', 2 -> 'female'
    - escolaridade: 1 -> 'graduate_school', 2 -> 'university',
                    3 -> 'high_school', 4 -> 'others'
    - estado_civil: 1 -> 'married', 2 -> 'single', 3 -> 'others'

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cleaned integer codes.

    Returns
    -------
    pd.DataFrame
        DataFrame with string labels for categorical columns.
    """
    df = df.copy()
    df["sexo"] = df["sexo"].map({1: "male", 2: "female"})
    df["escolaridade"] = df["escolaridade"].map({
        1: "graduate_school",
        2: "university",
        3: "high_school",
        4: "others",
    })
    df["estado_civil"] = df["estado_civil"].map({
        1: "married",
        2: "single",
        3: "others",
    })
    return df


def create_financial_features(df):
    """Create derived financial ratio features.

    - razao_fatura_limite: valor_fatura_1 / limite_credito
    - razao_pagamento_fatura: valor_pagamento_1 / valor_fatura_2

    Both ratios are clipped to handle inf/nan values (replaced with 0).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the required base columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with two additional feature columns.
    """
    df = df.copy()

    df["razao_fatura_limite"] = np.where(
        df["limite_credito"] != 0,
        df["valor_fatura_1"] / df["limite_credito"],
        0,
    )

    df["razao_pagamento_fatura"] = np.where(
        df["valor_fatura_2"] != 0,
        df["valor_pagamento_1"] / df["valor_fatura_2"],
        0,
    )

    df[["razao_fatura_limite", "razao_pagamento_fatura"]] = (
        df[["razao_fatura_limite", "razao_pagamento_fatura"]]
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    return df


def prepare_features_target(df, target_col="inadimplente_proximo_mes"):
    """Separate features (X) from target (y) and drop ID if present.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared DataFrame.
    target_col : str
        Name of the target column.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y) where X has features and y is the integer target.
    """
    cols_to_drop = [target_col]
    if "ID" in df.columns:
        cols_to_drop.append("ID")

    X = df.drop(columns=cols_to_drop, errors="ignore")
    y = df[target_col].astype(int)
    return X, y


def full_preparation_pipeline(df):
    """Run the complete data preparation: clean, map, engineer features, split.

    Parameters
    ----------
    df : pd.DataFrame
        Raw loaded DataFrame (already renamed).

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y) ready for modeling.
    """
    df = clean_categories(df)
    df = map_categorical_labels(df)
    df = create_financial_features(df)
    X, y = prepare_features_target(df)
    return X, y
