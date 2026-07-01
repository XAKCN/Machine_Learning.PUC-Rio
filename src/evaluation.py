"""Model evaluation utilities for binary classification."""

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


def get_positive_class_scores(model, X):
    """Return the score for the positive class used in ranking metrics.

    Parameters
    ----------
    model : estimator
        Fitted sklearn-compatible model.
    X : array-like
        Feature matrix.

    Returns
    -------
    np.ndarray or None
        Probability/decision scores for the positive class.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def evaluate_binary_classifier(model, X, y, model_name, split_name):
    """Compute standard binary classification metrics.

    Parameters
    ----------
    model : estimator
        Fitted model.
    X : array-like
        Feature matrix.
    y : array-like
        True labels.
    model_name : str
        Name of the model (for the results dict).
    split_name : str
        Name of the data split (e.g., 'train', 'test').

    Returns
    -------
    dict
        Dictionary with model, split, accuracy, precision, recall, f1,
        roc_auc, and pr_auc.
    """
    y_pred = model.predict(X)
    y_score = get_positive_class_scores(model, X)

    results = {
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


def compute_threshold_analysis(y_true, scores, threshold_grid,
                               fn_cost=5, fp_cost=1):
    """Evaluate classification metrics across a range of thresholds.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    scores : array-like
        Predicted probability scores for the positive class.
    threshold_grid : array-like
        Thresholds to evaluate.
    fn_cost : int
        Cost assigned to each false negative.
    fp_cost : int
        Cost assigned to each false positive.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: threshold, precision, recall, f1,
        false_negatives, false_positives, illustrative_cost.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    rows = []
    for threshold in threshold_grid:
        pred = (scores >= threshold).astype(int)
        fn = int(((y_true == 1) & (pred == 0)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        rows.append({
            "threshold": float(threshold),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "false_negatives": fn,
            "false_positives": fp,
            "illustrative_cost": fn_cost * fn + fp_cost * fp,
        })

    return pd.DataFrame(rows)


def select_best_threshold(threshold_df):
    """Select the threshold with the lowest illustrative cost.

    Parameters
    ----------
    threshold_df : pd.DataFrame
        Output of compute_threshold_analysis.

    Returns
    -------
    float
        Optimal threshold value.
    """
    best_row = threshold_df.sort_values(
        ["illustrative_cost", "threshold"]
    ).iloc[0]
    return float(best_row["threshold"])
