"""Unit tests for src.data_preparation module."""

import numpy as np
import pandas as pd
import pytest

from src.data_preparation import (
    clean_categories,
    create_financial_features,
    full_preparation_pipeline,
    map_categorical_labels,
    prepare_features_target,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame mimicking the loaded/renamed dataset."""
    return pd.DataFrame({
        "limite_credito": [20000, 50000, 0, 30000, 10000],
        "sexo": [1, 2, 1, 2, 1],
        "escolaridade": [0, 1, 5, 6, 2],
        "estado_civil": [0, 1, 2, 3, 1],
        "idade": [24, 35, 42, 28, 55],
        "status_pagamento_1": [0, -1, 2, 0, -1],
        "status_pagamento_2": [0, 0, -1, 0, 0],
        "status_pagamento_3": [-1, -1, 0, -1, 0],
        "status_pagamento_4": [0, 0, 0, 0, 0],
        "status_pagamento_5": [-1, 0, 0, -1, 0],
        "status_pagamento_6": [-1, -1, -1, -1, -1],
        "valor_fatura_1": [3913, 2682, 13648, 5000, 0],
        "valor_fatura_2": [3102, 1725, 0, 14138, 5000],
        "valor_fatura_3": [689, 2682, 14331, 14948, 3000],
        "valor_fatura_4": [0, 3272, 14948, 15549, 2000],
        "valor_fatura_5": [0, 3455, 15549, 15549, 1000],
        "valor_fatura_6": [0, 3261, 15549, 15549, 500],
        "valor_pagamento_1": [689, 1000, 1500, 2000, 0],
        "valor_pagamento_2": [0, 1000, 1000, 1000, 500],
        "valor_pagamento_3": [0, 1000, 1000, 1000, 500],
        "valor_pagamento_4": [0, 1000, 1000, 1000, 500],
        "valor_pagamento_5": [0, 0, 1000, 1000, 500],
        "valor_pagamento_6": [0, 2000, 5000, 5000, 500],
        "inadimplente_proximo_mes": [1, 0, 0, 1, 0],
    })


class TestCleanCategories:
    def test_escolaridade_zero_maps_to_4(self, sample_df):
        result = clean_categories(sample_df)
        assert 0 not in result["escolaridade"].values

    def test_escolaridade_5_maps_to_4(self, sample_df):
        result = clean_categories(sample_df)
        assert 5 not in result["escolaridade"].values

    def test_escolaridade_6_maps_to_4(self, sample_df):
        result = clean_categories(sample_df)
        assert 6 not in result["escolaridade"].values

    def test_escolaridade_valid_values_preserved(self, sample_df):
        result = clean_categories(sample_df)
        assert 1 in result["escolaridade"].values
        assert 2 in result["escolaridade"].values

    def test_estado_civil_zero_maps_to_3(self, sample_df):
        result = clean_categories(sample_df)
        assert 0 not in result["estado_civil"].values

    def test_estado_civil_valid_values_preserved(self, sample_df):
        result = clean_categories(sample_df)
        assert 1 in result["estado_civil"].values
        assert 2 in result["estado_civil"].values
        assert 3 in result["estado_civil"].values

    def test_does_not_modify_original(self, sample_df):
        original_vals = sample_df["escolaridade"].copy()
        clean_categories(sample_df)
        pd.testing.assert_series_equal(sample_df["escolaridade"], original_vals)

    def test_row_count_preserved(self, sample_df):
        result = clean_categories(sample_df)
        assert len(result) == len(sample_df)


class TestMapCategoricalLabels:
    def test_sexo_mapping(self, sample_df):
        cleaned = clean_categories(sample_df)
        result = map_categorical_labels(cleaned)
        assert set(result["sexo"].dropna().unique()) == {"male", "female"}

    def test_escolaridade_mapping(self, sample_df):
        cleaned = clean_categories(sample_df)
        result = map_categorical_labels(cleaned)
        valid_labels = {"graduate_school", "university", "high_school", "others"}
        assert set(result["escolaridade"].dropna().unique()).issubset(valid_labels)

    def test_estado_civil_mapping(self, sample_df):
        cleaned = clean_categories(sample_df)
        result = map_categorical_labels(cleaned)
        valid_labels = {"married", "single", "others"}
        assert set(result["estado_civil"].dropna().unique()).issubset(valid_labels)

    def test_does_not_modify_original(self, sample_df):
        cleaned = clean_categories(sample_df)
        original_sexo = cleaned["sexo"].copy()
        map_categorical_labels(cleaned)
        pd.testing.assert_series_equal(cleaned["sexo"], original_sexo)


class TestCreateFinancialFeatures:
    def test_creates_razao_fatura_limite(self, sample_df):
        result = create_financial_features(sample_df)
        assert "razao_fatura_limite" in result.columns

    def test_creates_razao_pagamento_fatura(self, sample_df):
        result = create_financial_features(sample_df)
        assert "razao_pagamento_fatura" in result.columns

    def test_zero_limit_gives_zero_ratio(self, sample_df):
        result = create_financial_features(sample_df)
        zero_limit_mask = sample_df["limite_credito"] == 0
        assert (result.loc[zero_limit_mask, "razao_fatura_limite"] == 0).all()

    def test_zero_fatura2_gives_zero_ratio(self, sample_df):
        result = create_financial_features(sample_df)
        zero_fatura_mask = sample_df["valor_fatura_2"] == 0
        assert (result.loc[zero_fatura_mask, "razao_pagamento_fatura"] == 0).all()

    def test_no_inf_values(self, sample_df):
        result = create_financial_features(sample_df)
        assert not np.isinf(result["razao_fatura_limite"]).any()
        assert not np.isinf(result["razao_pagamento_fatura"]).any()

    def test_no_nan_values(self, sample_df):
        result = create_financial_features(sample_df)
        assert not result["razao_fatura_limite"].isna().any()
        assert not result["razao_pagamento_fatura"].isna().any()

    def test_correct_ratio_computation(self):
        df = pd.DataFrame({
            "limite_credito": [10000],
            "valor_fatura_1": [5000],
            "valor_fatura_2": [2000],
            "valor_pagamento_1": [1000],
            "status_pagamento_1": [0],
            "status_pagamento_2": [0],
            "status_pagamento_3": [0],
            "status_pagamento_4": [0],
            "status_pagamento_5": [0],
            "status_pagamento_6": [0],
        })
        result = create_financial_features(df)
        assert result["razao_fatura_limite"].iloc[0] == pytest.approx(0.5)
        assert result["razao_pagamento_fatura"].iloc[0] == pytest.approx(0.5)

    def test_does_not_modify_original(self, sample_df):
        original_cols = list(sample_df.columns)
        create_financial_features(sample_df)
        assert list(sample_df.columns) == original_cols


class TestPrepareFeaturesTarget:
    def test_target_separated(self, sample_df):
        X, y = prepare_features_target(sample_df)
        assert "inadimplente_proximo_mes" not in X.columns
        assert len(y) == len(sample_df)

    def test_target_is_integer(self, sample_df):
        _, y = prepare_features_target(sample_df)
        assert y.dtype == int or y.dtype == np.int64

    def test_target_values(self, sample_df):
        _, y = prepare_features_target(sample_df)
        assert set(y.unique()).issubset({0, 1})

    def test_feature_count(self, sample_df):
        X, _ = prepare_features_target(sample_df)
        # 24 columns total - 1 target = 23 features (no ID in sample_df)
        assert X.shape[1] == 23

    def test_with_id_column(self, sample_df):
        sample_df["ID"] = range(len(sample_df))
        X, _ = prepare_features_target(sample_df)
        assert "ID" not in X.columns


class TestFullPreparationPipeline:
    def test_returns_tuple(self, sample_df):
        result = full_preparation_pipeline(sample_df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_x_has_financial_features(self, sample_df):
        X, _ = full_preparation_pipeline(sample_df)
        assert "razao_fatura_limite" in X.columns
        assert "razao_pagamento_fatura" in X.columns

    def test_categorical_labels_applied(self, sample_df):
        X, _ = full_preparation_pipeline(sample_df)
        assert pd.api.types.is_string_dtype(X["sexo"])

    def test_target_binary(self, sample_df):
        _, y = full_preparation_pipeline(sample_df)
        assert set(y.unique()).issubset({0, 1})

    def test_no_inf_or_nan_in_ratios(self, sample_df):
        X, _ = full_preparation_pipeline(sample_df)
        assert not np.isinf(X["razao_fatura_limite"]).any()
        assert not X["razao_fatura_limite"].isna().any()
