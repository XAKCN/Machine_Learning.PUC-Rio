"""Unit tests for src.data_loader module."""

import pandas as pd
import pytest

from src.data_loader import (
    EXPECTED_COLUMNS,
    RENAME_MAP,
    load_dataset,
    rename_columns,
    validate_columns,
)


@pytest.fixture
def raw_dataframe():
    """Create a minimal raw DataFrame with original English column names."""
    return pd.DataFrame({
        "ID": [1, 2, 3],
        "LIMIT_BAL": [20000, 50000, 30000],
        "SEX": [1, 2, 1],
        "EDUCATION": [2, 1, 3],
        "MARRIAGE": [1, 2, 1],
        "AGE": [24, 35, 42],
        "PAY_0": [0, -1, 2],
        "PAY_2": [0, 0, -1],
        "PAY_3": [-1, -1, 0],
        "PAY_4": [0, 0, 0],
        "PAY_5": [-1, 0, 0],
        "PAY_6": [-1, -1, -1],
        "BILL_AMT1": [3913, 2682, 13648],
        "BILL_AMT2": [3102, 1725, 14138],
        "BILL_AMT3": [689, 2682, 14331],
        "BILL_AMT4": [0, 3272, 14948],
        "BILL_AMT5": [0, 3455, 15549],
        "BILL_AMT6": [0, 3261, 15549],
        "PAY_AMT1": [689, 1000, 1500],
        "PAY_AMT2": [0, 1000, 1000],
        "PAY_AMT3": [0, 1000, 1000],
        "PAY_AMT4": [0, 1000, 1000],
        "PAY_AMT5": [0, 0, 1000],
        "PAY_AMT6": [0, 2000, 5000],
        "default payment next month": [1, 0, 0],
    })


class TestRenameColumns:
    def test_renames_all_mapped_columns(self, raw_dataframe):
        result = rename_columns(raw_dataframe)
        for original, renamed in RENAME_MAP.items():
            if original in raw_dataframe.columns:
                assert renamed in result.columns

    def test_preserves_unmapped_columns(self, raw_dataframe):
        result = rename_columns(raw_dataframe)
        assert "ID" in result.columns

    def test_does_not_modify_original(self, raw_dataframe):
        original_cols = list(raw_dataframe.columns)
        rename_columns(raw_dataframe)
        assert list(raw_dataframe.columns) == original_cols

    def test_row_count_unchanged(self, raw_dataframe):
        result = rename_columns(raw_dataframe)
        assert len(result) == len(raw_dataframe)


class TestValidateColumns:
    def test_valid_dataframe(self, raw_dataframe):
        renamed = rename_columns(raw_dataframe)
        renamed = renamed.drop(columns="ID")
        is_valid, missing = validate_columns(renamed)
        assert is_valid is True
        assert missing == []

    def test_missing_columns_detected(self):
        df = pd.DataFrame({"limite_credito": [1], "sexo": [1]})
        is_valid, missing = validate_columns(df)
        assert is_valid is False
        assert len(missing) > 0
        assert "escolaridade" in missing

    def test_empty_dataframe_missing_all(self):
        df = pd.DataFrame()
        is_valid, missing = validate_columns(df)
        assert is_valid is False
        assert set(missing) == set(EXPECTED_COLUMNS)


class TestLoadDataset:
    def test_load_csv(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({
            "LIMIT_BAL": [20000],
            "SEX": [1],
            "EDUCATION": [2],
            "MARRIAGE": [1],
            "AGE": [25],
            "PAY_0": [0],
            "PAY_2": [0],
            "PAY_3": [0],
            "PAY_4": [0],
            "PAY_5": [0],
            "PAY_6": [0],
            "BILL_AMT1": [1000],
            "BILL_AMT2": [2000],
            "BILL_AMT3": [3000],
            "BILL_AMT4": [4000],
            "BILL_AMT5": [5000],
            "BILL_AMT6": [6000],
            "PAY_AMT1": [500],
            "PAY_AMT2": [500],
            "PAY_AMT3": [500],
            "PAY_AMT4": [500],
            "PAY_AMT5": [500],
            "PAY_AMT6": [500],
            "default payment next month": [0],
        })
        df.to_csv(csv_path, index=False)
        result = load_dataset(str(csv_path), header=0)
        assert "limite_credito" in result.columns
        assert "ID" not in result.columns

    def test_load_drops_id(self, raw_dataframe, tmp_path):
        csv_path = tmp_path / "with_id.csv"
        raw_dataframe.to_csv(csv_path, index=False)
        result = load_dataset(str(csv_path), header=0)
        assert "ID" not in result.columns

    def test_load_shape(self, raw_dataframe, tmp_path):
        csv_path = tmp_path / "shape_test.csv"
        raw_dataframe.to_csv(csv_path, index=False)
        result = load_dataset(str(csv_path), header=0)
        assert result.shape[0] == 3
        assert result.shape[1] == 24  # 25 original - ID
