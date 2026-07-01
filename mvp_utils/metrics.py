"""Reusable metrics and evaluation helpers for the Credit Risk MVP."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_positive_class_scores(
    model: Any,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray | None:
    """Return the score for the positive class used in ranking metrics.

    Tries ``predict_proba`` first, then ``decision_function``.

    Parameters
    ----------
    model : estimator
        A fitted scikit-learn estimator (or pipeline).
    X : array-like
        Feature matrix.

    Returns
    -------
    np.ndarray or None
        Positive-class scores, or None if the model supports neither method.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def evaluate_binary_classifier(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    model_name: str,
    split_name: str,
) -> dict[str, Any]:
    """Compute standard binary classification metrics for a fitted model.

    Parameters
    ----------
    model : estimator
        A fitted scikit-learn estimator (or pipeline).
    X : array-like
        Feature matrix.
    y : array-like
        True binary labels.
    model_name : str
        Name of the model (for identification in results).
    split_name : str
        Name of the data split (e.g., "train", "test").

    Returns
    -------
    dict
        Dictionary with keys: model, split, accuracy, precision, recall,
        f1, roc_auc, pr_auc.
    """
    y_pred = model.predict(X)
    y_score = get_positive_class_scores(model, X)

    results: dict[str, Any] = {
        "model": model_name,
        "split": split_name,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }

    if y_score is not None:
        results["roc_auc"] = roc_auc_score(y, y_score)
        results["pr_auc"] = average_precision_score(y, y_score)
    else:
        results["roc_auc"] = np.nan
        results["pr_auc"] = np.nan

    return results


def extract_cv_metrics(
    scores: dict[str, np.ndarray],
    model_name: str,
    scoring_keys: list[str],
    elapsed_time: float,
) -> dict[str, Any]:
    """Extract mean/std/train/gap metrics from cross_validate output.

    Parameters
    ----------
    scores : dict
        Output of ``sklearn.model_selection.cross_validate`` with
        ``return_train_score=True``.
    model_name : str
        Name of the model for identification.
    scoring_keys : list of str
        The scoring metric names used in cross_validate.
    elapsed_time : float
        Wall-clock time for the cross-validation run.

    Returns
    -------
    dict
        Flat dictionary with keys like ``accuracy_mean``, ``accuracy_std``,
        ``accuracy_train_mean``, ``accuracy_gap``, etc.
    """
    result: dict[str, Any] = {
        "model": model_name,
        "training_time_seconds": elapsed_time,
    }

    for metric in scoring_keys:
        result[f"{metric}_mean"] = scores[f"test_{metric}"].mean()
        result[f"{metric}_std"] = scores[f"test_{metric}"].std()
        result[f"{metric}_train_mean"] = scores[f"train_{metric}"].mean()
        result[f"{metric}_gap"] = result[f"{metric}_train_mean"] - result[f"{metric}_mean"]

    return result


def compute_threshold_analysis(
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
    *,
    threshold_min: float = 0.10,
    threshold_max: float = 0.70,
    threshold_step: float = 0.02,
    fn_cost: int = 5,
    fp_cost: int = 1,
) -> pd.DataFrame:
    """Evaluate precision/recall/F1 and cost across a grid of thresholds.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    scores : np.ndarray
        Predicted positive-class probabilities.
    threshold_min, threshold_max, threshold_step : float
        Range and step for the threshold grid.
    fn_cost : int
        Cost multiplier for false negatives.
    fp_cost : int
        Cost multiplier for false positives.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: threshold, precision, recall, f1,
        false_negatives, false_positives, illustrative_cost.
    """
    y_true = np.asarray(y_true)
    threshold_grid = np.round(
        np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step), 2
    )

    rows = []
    for threshold in threshold_grid:
        pred = (scores >= threshold).astype(int)
        fn = int(((y_true == 1) & (pred == 0)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
                "false_negatives": fn,
                "false_positives": fp,
                "illustrative_cost": fn_cost * fn + fp_cost * fp,
            }
        )

    return pd.DataFrame(rows)
