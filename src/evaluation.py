"""Reusable evaluation utilities for binary classification."""

from typing import Optional

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def evaluate_binary_classifier(
    model_name: str,
    y_true,
    y_pred,
    y_score: Optional[object] = None,
) -> dict:
    """
    Evaluate a binary classifier using the project's standard metrics.

    Parameters
    ----------
    model_name : str
        Human-readable model name.

    y_true : array-like
        True binary labels.

    y_pred : array-like
        Predicted binary labels.

    y_score : array-like, optional
        Probability or decision score for the positive class.
        Required for ROC-AUC.

    Returns
    -------
    dict
        Classification metrics for the model.
    """

    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }

    if y_score is not None:
        results["roc_auc"] = roc_auc_score(
            y_true,
            y_score,
        )
    else:
        results["roc_auc"] = None

    return results


def get_confusion_matrix(y_true, y_pred):
    """
    Return the confusion matrix for a binary classifier.
    """

    return confusion_matrix(
        y_true,
        y_pred,
    )


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """
    Convert model evaluation dictionaries into a comparison table.

    Parameters
    ----------
    results : list of dict
        Results returned by evaluate_binary_classifier().

    Returns
    -------
    pandas.DataFrame
        Model comparison table.
    """

    columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    return pd.DataFrame(
        results,
        columns=columns,
    )