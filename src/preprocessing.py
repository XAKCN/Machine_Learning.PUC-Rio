"""Scikit-learn preprocessing pipeline construction."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PAYMENT_STATUS_COLS = [
    "status_pagamento_1",
    "status_pagamento_2",
    "status_pagamento_3",
    "status_pagamento_4",
    "status_pagamento_5",
    "status_pagamento_6",
]

DEFAULT_CATEGORICAL_FEATURES = [
    "sexo",
    "escolaridade",
    "estado_civil",
    *PAYMENT_STATUS_COLS,
]


def get_feature_groups(X, categorical_features=None):
    """Split feature names into numeric and categorical groups.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    categorical_features : list[str] or None
        Categorical column names. Defaults to DEFAULT_CATEGORICAL_FEATURES.

    Returns
    -------
    tuple[list[str], list[str]]
        (numeric_features, categorical_features)
    """
    if categorical_features is None:
        categorical_features = DEFAULT_CATEGORICAL_FEATURES

    numeric_features = [
        col for col in X.columns if col not in categorical_features
    ]
    return numeric_features, categorical_features


def build_numeric_transformer():
    """Build the numeric feature transformer pipeline.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline with median imputer + standard scaler.
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def build_categorical_transformer():
    """Build the categorical feature transformer pipeline.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline with most-frequent imputer + one-hot encoder.
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])


def build_preprocessor(numeric_features, categorical_features):
    """Build the full ColumnTransformer preprocessing pipeline.

    Parameters
    ----------
    numeric_features : list[str]
        Names of numeric columns.
    categorical_features : list[str]
        Names of categorical columns.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Fitted-ready column transformer.
    """
    numeric_transformer = build_numeric_transformer()
    categorical_transformer = build_categorical_transformer()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def build_model_pipeline(model, numeric_features, categorical_features):
    """Build a full pipeline: preprocessing + model.

    Parameters
    ----------
    model : estimator
        Scikit-learn compatible estimator.
    numeric_features : list[str]
        Numeric column names.
    categorical_features : list[str]
        Categorical column names.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline(preprocessor, model).
    """
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])
