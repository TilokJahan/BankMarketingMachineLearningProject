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
    ConfusionMatrixDisplay, 
    RocCurveDisplay,
    make_scorer,
)

from sklearn.model_selection import (
    cross_validate,
    StratifiedKFold,
)

import matplotlib.pyplot as plt


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

def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str,
):
    """
    Plot a confusion matrix for a binary classifier.
    """

    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["No", "Yes"],
    )

    display.ax_.set_title(
        f"{model_name} - Confusion Matrix"
    )

    plt.tight_layout()

    return display

def plot_roc_curve(
    y_true,
    y_score,
    model_name: str,
):
    """
    Plot the ROC curve for a binary classifier.
    """

    display = RocCurveDisplay.from_predictions(
        y_true,
        y_score,
        name=model_name,
    )

    display.ax_.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    display.ax_.set_title(
        f"{model_name} - ROC Curve"
    )

    display.ax_.legend()

    plt.tight_layout()

    return display

def cross_validate_classifier(
    model_name: str,
    estimator,
    X,
    y,
    cv: int = 5,
) -> dict:
    """
    Evaluate a classifier using cross-validation.

    Returns mean and standard deviation for the project's
    standard classification metrics.
    """

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(
            precision_score,
            zero_division=0,
        ),
        "recall": make_scorer(
            recall_score,
            zero_division=0,
        ),
        "f1": make_scorer(
            f1_score,
            zero_division=0,
        ),
        "roc_auc": "roc_auc",
    }

    cv_strategy = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=42,
    )


    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv_strategy,
        scoring=scoring,
    )

    results = {
        "model": model_name,
    }

    for metric in scoring:
        values = scores[f"test_{metric}"]

        results[f"{metric}_mean"] = values.mean()
        results[f"{metric}_std"] = values.std()

    return results

def cross_validation_results_to_dataframe(
    results: list[dict],
) -> pd.DataFrame:
    """
    Convert cross-validation result dictionaries into a DataFrame.
    """

    return pd.DataFrame(results)