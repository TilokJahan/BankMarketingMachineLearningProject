"""Evaluation helpers for binary classification."""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def evaluate_binary_classifier(y_true, y_pred, y_score=None):
    """
    Calculate common binary-classification metrics.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.
    y_score : array-like, optional
        Positive-class probability or decision score.

    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)

    return metrics
