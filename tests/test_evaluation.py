"""Unit tests for src.evaluation module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from src.evaluation import (
    compute_threshold_analysis,
    evaluate_binary_classifier,
    get_positive_class_scores,
    select_best_threshold,
)


@pytest.fixture
def binary_dataset():
    """Create a simple binary classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=42,
    )
    return X, y


@pytest.fixture
def fitted_logistic(binary_dataset):
    """Fit a logistic regression model."""
    X, y = binary_dataset
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def fitted_dummy(binary_dataset):
    """Fit a dummy classifier."""
    X, y = binary_dataset
    model = DummyClassifier(strategy="most_frequent", random_state=42)
    model.fit(X, y)
    return model


class TestGetPositiveClassScores:
    def test_with_predict_proba(self, fitted_logistic, binary_dataset):
        X, _ = binary_dataset
        scores = get_positive_class_scores(fitted_logistic, X)
        assert scores is not None
        assert len(scores) == len(X)
        assert all(0 <= s <= 1 for s in scores)

    def test_with_dummy(self, fitted_dummy, binary_dataset):
        X, _ = binary_dataset
        scores = get_positive_class_scores(fitted_dummy, X)
        assert scores is not None
        assert len(scores) == len(X)

    def test_without_predict_proba_or_decision(self):
        class MockModel:
            pass

        model = MockModel()
        scores = get_positive_class_scores(model, np.array([[1, 2]]))
        assert scores is None

    def test_with_decision_function(self, binary_dataset):
        from sklearn.svm import LinearSVC

        X, y = binary_dataset
        model = LinearSVC(random_state=42, max_iter=5000)
        model.fit(X, y)
        scores = get_positive_class_scores(model, X)
        assert scores is not None
        assert len(scores) == len(X)


class TestEvaluateBinaryClassifier:
    def test_returns_dict(self, fitted_logistic, binary_dataset):
        X, y = binary_dataset
        result = evaluate_binary_classifier(
            fitted_logistic, X, y, "LR", "test"
        )
        assert isinstance(result, dict)

    def test_contains_expected_keys(self, fitted_logistic, binary_dataset):
        X, y = binary_dataset
        result = evaluate_binary_classifier(
            fitted_logistic, X, y, "LR", "test"
        )
        expected_keys = {
            "model", "split", "accuracy", "precision",
            "recall", "f1", "roc_auc", "pr_auc"
        }
        assert set(result.keys()) == expected_keys

    def test_model_name_and_split(self, fitted_logistic, binary_dataset):
        X, y = binary_dataset
        result = evaluate_binary_classifier(
            fitted_logistic, X, y, "MyModel", "validation"
        )
        assert result["model"] == "MyModel"
        assert result["split"] == "validation"

    def test_metrics_in_valid_range(self, fitted_logistic, binary_dataset):
        X, y = binary_dataset
        result = evaluate_binary_classifier(
            fitted_logistic, X, y, "LR", "test"
        )
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            assert 0 <= result[metric] <= 1

    def test_dummy_has_low_metrics(self, fitted_dummy, binary_dataset):
        X, y = binary_dataset
        result = evaluate_binary_classifier(
            fitted_dummy, X, y, "Dummy", "test"
        )
        # Dummy should have ~0.5 ROC-AUC
        assert result["roc_auc"] <= 0.55

    def test_model_without_scores(self, binary_dataset):
        X, y = binary_dataset

        class NoScoreModel:
            def predict(self, X):
                return np.ones(len(X), dtype=int)

        model = NoScoreModel()
        result = evaluate_binary_classifier(model, X, y, "NoScore", "test")
        assert np.isnan(result["roc_auc"])
        assert np.isnan(result["pr_auc"])


class TestComputeThresholdAnalysis:
    def test_returns_dataframe(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0.1, 0.3, 0.7, 0.9, 0.4, 0.6])
        thresholds = np.arange(0.1, 0.9, 0.1)
        result = compute_threshold_analysis(y_true, scores, thresholds)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        y_true = np.array([0, 1, 0, 1])
        scores = np.array([0.2, 0.8, 0.3, 0.7])
        thresholds = [0.3, 0.5, 0.7]
        result = compute_threshold_analysis(y_true, scores, thresholds)
        expected_cols = {
            "threshold", "precision", "recall", "f1",
            "false_negatives", "false_positives", "illustrative_cost"
        }
        assert set(result.columns) == expected_cols

    def test_row_count_matches_grid(self):
        y_true = np.array([0, 1, 0, 1])
        scores = np.array([0.2, 0.8, 0.3, 0.7])
        thresholds = [0.2, 0.4, 0.6, 0.8]
        result = compute_threshold_analysis(y_true, scores, thresholds)
        assert len(result) == 4

    def test_threshold_zero_all_positive(self):
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.3, 0.4, 0.6, 0.9])
        result = compute_threshold_analysis(y_true, scores, [0.0])
        row = result.iloc[0]
        assert row["recall"] == 1.0  # All predicted as positive
        assert row["false_negatives"] == 0

    def test_threshold_one_all_negative(self):
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.3, 0.4, 0.6, 0.9])
        result = compute_threshold_analysis(y_true, scores, [1.0])
        row = result.iloc[0]
        assert row["recall"] == 0.0
        assert row["false_positives"] == 0
        assert row["false_negatives"] == 2

    def test_cost_calculation(self):
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.3, 0.4, 0.6, 0.9])
        fn_cost = 10
        fp_cost = 2
        result = compute_threshold_analysis(
            y_true, scores, [0.5], fn_cost=fn_cost, fp_cost=fp_cost
        )
        row = result.iloc[0]
        expected_cost = fn_cost * row["false_negatives"] + fp_cost * row["false_positives"]
        assert row["illustrative_cost"] == expected_cost

    def test_metrics_in_valid_range(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        thresholds = np.arange(0.1, 1.0, 0.1)
        result = compute_threshold_analysis(y_true, scores, thresholds)
        assert (result["precision"] >= 0).all() and (result["precision"] <= 1).all()
        assert (result["recall"] >= 0).all() and (result["recall"] <= 1).all()
        assert (result["f1"] >= 0).all() and (result["f1"] <= 1).all()


class TestSelectBestThreshold:
    def test_selects_minimum_cost(self):
        df = pd.DataFrame({
            "threshold": [0.2, 0.4, 0.6],
            "illustrative_cost": [100, 50, 80],
            "precision": [0.5, 0.7, 0.9],
            "recall": [0.9, 0.7, 0.5],
            "f1": [0.6, 0.7, 0.6],
            "false_negatives": [1, 3, 5],
            "false_positives": [10, 5, 2],
        })
        result = select_best_threshold(df)
        assert result == 0.4

    def test_tiebreaker_lowest_threshold(self):
        df = pd.DataFrame({
            "threshold": [0.3, 0.5, 0.7],
            "illustrative_cost": [50, 50, 80],
            "precision": [0.6, 0.7, 0.9],
            "recall": [0.9, 0.7, 0.5],
            "f1": [0.7, 0.7, 0.6],
            "false_negatives": [1, 3, 5],
            "false_positives": [10, 5, 2],
        })
        result = select_best_threshold(df)
        assert result == 0.3

    def test_returns_float(self):
        df = pd.DataFrame({
            "threshold": [0.5],
            "illustrative_cost": [100],
            "precision": [0.7],
            "recall": [0.7],
            "f1": [0.7],
            "false_negatives": [3],
            "false_positives": [5],
        })
        result = select_best_threshold(df)
        assert isinstance(result, float)
