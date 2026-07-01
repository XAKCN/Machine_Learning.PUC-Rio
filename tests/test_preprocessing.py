"""Unit tests for src.preprocessing module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.preprocessing import (
    DEFAULT_CATEGORICAL_FEATURES,
    PAYMENT_STATUS_COLS,
    build_categorical_transformer,
    build_model_pipeline,
    build_numeric_transformer,
    build_preprocessor,
    get_feature_groups,
)


@pytest.fixture
def feature_df():
    """Create a feature DataFrame after preparation (with string categories)."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "limite_credito": np.random.randint(10000, 500000, n),
        "sexo": np.random.choice(["male", "female"], n),
        "escolaridade": np.random.choice(
            ["graduate_school", "university", "high_school", "others"], n
        ),
        "estado_civil": np.random.choice(["married", "single", "others"], n),
        "idade": np.random.randint(20, 70, n),
        "status_pagamento_1": np.random.choice([-2, -1, 0, 1, 2], n),
        "status_pagamento_2": np.random.choice([-2, -1, 0, 1, 2], n),
        "status_pagamento_3": np.random.choice([-2, -1, 0, 1, 2], n),
        "status_pagamento_4": np.random.choice([-2, -1, 0, 1, 2], n),
        "status_pagamento_5": np.random.choice([-2, -1, 0, 1, 2], n),
        "status_pagamento_6": np.random.choice([-2, -1, 0, 1, 2], n),
        "valor_fatura_1": np.random.randint(0, 50000, n),
        "valor_fatura_2": np.random.randint(0, 50000, n),
        "valor_fatura_3": np.random.randint(0, 50000, n),
        "valor_fatura_4": np.random.randint(0, 50000, n),
        "valor_fatura_5": np.random.randint(0, 50000, n),
        "valor_fatura_6": np.random.randint(0, 50000, n),
        "valor_pagamento_1": np.random.randint(0, 20000, n),
        "valor_pagamento_2": np.random.randint(0, 20000, n),
        "valor_pagamento_3": np.random.randint(0, 20000, n),
        "valor_pagamento_4": np.random.randint(0, 20000, n),
        "valor_pagamento_5": np.random.randint(0, 20000, n),
        "valor_pagamento_6": np.random.randint(0, 20000, n),
        "razao_fatura_limite": np.random.uniform(0, 2, n),
        "razao_pagamento_fatura": np.random.uniform(0, 2, n),
    })


@pytest.fixture
def target_series():
    np.random.seed(42)
    return pd.Series(np.random.choice([0, 1], 50, p=[0.78, 0.22]))


class TestGetFeatureGroups:
    def test_default_split(self, feature_df):
        numeric, categorical = get_feature_groups(feature_df)
        assert len(categorical) == len(DEFAULT_CATEGORICAL_FEATURES)
        assert len(numeric) + len(categorical) == len(feature_df.columns)

    def test_custom_categorical(self, feature_df):
        custom_cats = ["sexo", "escolaridade"]
        numeric, categorical = get_feature_groups(feature_df, custom_cats)
        assert categorical == custom_cats
        assert len(numeric) == len(feature_df.columns) - 2

    def test_all_columns_covered(self, feature_df):
        numeric, categorical = get_feature_groups(feature_df)
        all_cols = set(numeric + categorical)
        assert all_cols == set(feature_df.columns)

    def test_no_overlap(self, feature_df):
        numeric, categorical = get_feature_groups(feature_df)
        assert set(numeric).isdisjoint(set(categorical))

    def test_payment_status_in_categorical(self, feature_df):
        _, categorical = get_feature_groups(feature_df)
        for col in PAYMENT_STATUS_COLS:
            assert col in categorical


class TestBuildNumericTransformer:
    def test_creates_pipeline(self):
        transformer = build_numeric_transformer()
        assert hasattr(transformer, "fit")
        assert hasattr(transformer, "transform")

    def test_has_imputer_and_scaler(self):
        transformer = build_numeric_transformer()
        step_names = [name for name, _ in transformer.steps]
        assert "imputer" in step_names
        assert "scaler" in step_names

    def test_fits_and_transforms(self, feature_df):
        transformer = build_numeric_transformer()
        numeric_cols = ["limite_credito", "idade", "valor_fatura_1"]
        X_num = feature_df[numeric_cols].values
        result = transformer.fit_transform(X_num)
        assert result.shape == X_num.shape
        # StandardScaler produces mean ~0 and std ~1
        assert abs(result.mean(axis=0)).max() < 0.1


class TestBuildCategoricalTransformer:
    def test_creates_pipeline(self):
        transformer = build_categorical_transformer()
        assert hasattr(transformer, "fit")
        assert hasattr(transformer, "transform")

    def test_has_imputer_and_onehot(self):
        transformer = build_categorical_transformer()
        step_names = [name for name, _ in transformer.steps]
        assert "imputer" in step_names
        assert "onehot" in step_names

    def test_fits_and_transforms(self, feature_df):
        transformer = build_categorical_transformer()
        cat_cols = ["sexo", "escolaridade", "estado_civil"]
        X_cat = feature_df[cat_cols]
        result = transformer.fit_transform(X_cat)
        # OneHotEncoder expands columns
        assert result.shape[0] == len(feature_df)
        assert result.shape[1] > len(cat_cols)


class TestBuildPreprocessor:
    def test_creates_column_transformer(self, feature_df):
        numeric, categorical = get_feature_groups(feature_df)
        preprocessor = build_preprocessor(numeric, categorical)
        assert hasattr(preprocessor, "fit")
        assert hasattr(preprocessor, "transform")

    def test_fit_transform_output(self, feature_df):
        numeric, categorical = get_feature_groups(feature_df)
        preprocessor = build_preprocessor(numeric, categorical)
        result = preprocessor.fit_transform(feature_df)
        assert result.shape[0] == len(feature_df)
        # Output should have more columns due to one-hot encoding
        assert result.shape[1] > len(feature_df.columns)

    def test_handles_missing_values(self, feature_df):
        feature_df_with_nan = feature_df.copy()
        feature_df_with_nan.loc[0, "limite_credito"] = np.nan
        feature_df_with_nan.loc[1, "sexo"] = np.nan

        numeric, categorical = get_feature_groups(feature_df_with_nan)
        preprocessor = build_preprocessor(numeric, categorical)
        result = preprocessor.fit_transform(feature_df_with_nan)
        assert not np.isnan(result).any()


class TestBuildModelPipeline:
    def test_creates_pipeline(self, feature_df):
        numeric, categorical = get_feature_groups(feature_df)
        model = DummyClassifier(strategy="most_frequent")
        pipeline = build_model_pipeline(model, numeric, categorical)
        assert hasattr(pipeline, "fit")
        assert hasattr(pipeline, "predict")

    def test_fit_and_predict(self, feature_df, target_series):
        numeric, categorical = get_feature_groups(feature_df)
        model = DummyClassifier(strategy="most_frequent", random_state=42)
        pipeline = build_model_pipeline(model, numeric, categorical)
        pipeline.fit(feature_df, target_series)
        predictions = pipeline.predict(feature_df)
        assert len(predictions) == len(feature_df)
        assert set(predictions).issubset({0, 1})

    def test_pipeline_steps(self, feature_df):
        numeric, categorical = get_feature_groups(feature_df)
        model = LogisticRegression(max_iter=100, random_state=42)
        pipeline = build_model_pipeline(model, numeric, categorical)
        step_names = [name for name, _ in pipeline.steps]
        assert "preprocessor" in step_names
        assert "model" in step_names
