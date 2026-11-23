"""Reusable evaluation utilities for binary classification."""

from typing import Optional
import numpy as np
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

def bootstrap_classification_metrics(
    model_name: str,
    y_true,
    y_pred,
    y_score=None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict:
    """
    Estimate confidence intervals for binary classification metrics
    using bootstrap resampling of held-out predictions.

    Parameters
    ----------
    model_name : str
        Human-readable model name.

    y_true : array-like
        True binary labels from the held-out test set.

    y_pred : array-like
        Predicted binary labels from the held-out test set.

    y_score : array-like, optional
        Positive-class probability or decision score.
        Required for ROC-AUC.

    n_bootstrap : int, default=1000
        Number of bootstrap resamples.

    confidence_level : float, default=0.95
        Confidence level for percentile intervals.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Original metric estimates and bootstrap confidence intervals.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_score is not None:
        y_score = np.asarray(y_score)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    if y_score is not None and len(y_true) != len(y_score):
        raise ValueError(
            "y_true and y_score must have the same length."
        )

    if n_bootstrap <= 0:
        raise ValueError(
            "n_bootstrap must be greater than 0."
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    # Original held-out test metrics
    point_estimates = evaluate_binary_classifier(
        model_name=model_name,
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
    )

    rng = np.random.default_rng(random_state)

    n_samples = len(y_true)

    bootstrap_scores = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
    }

    for _ in range(n_bootstrap):

        # Sample test-set indices WITH replacement
        indices = rng.choice(
            n_samples,
            size=n_samples,
            replace=True,
        )

        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        if y_score is not None:
            y_score_boot = y_score[indices]
        else:
            y_score_boot = None

        bootstrap_scores["accuracy"].append(
            accuracy_score(
                y_true_boot,
                y_pred_boot,
            )
        )

        bootstrap_scores["precision"].append(
            precision_score(
                y_true_boot,
                y_pred_boot,
                zero_division=0,
            )
        )

        bootstrap_scores["recall"].append(
            recall_score(
                y_true_boot,
                y_pred_boot,
                zero_division=0,
            )
        )

        bootstrap_scores["f1"].append(
            f1_score(
                y_true_boot,
                y_pred_boot,
                zero_division=0,
            )
        )

        # ROC-AUC requires both classes to be present
        if (
            y_score_boot is not None
            and len(np.unique(y_true_boot)) == 2
        ):
            bootstrap_scores["roc_auc"].append(
                roc_auc_score(
                    y_true_boot,
                    y_score_boot,
                )
            )

    alpha = 1 - confidence_level

    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    results = {
        "model": model_name,
        "n_bootstrap": n_bootstrap,
        "confidence_level": confidence_level,
    }

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]:
        values = np.asarray(
            bootstrap_scores[metric]
        )

        results[metric] = point_estimates[metric]

        if len(values) > 0:
            results[f"{metric}_ci_lower"] = np.percentile(
                values,
                lower_percentile,
            )

            results[f"{metric}_ci_upper"] = np.percentile(
                values,
                upper_percentile,
            )
        else:
            results[f"{metric}_ci_lower"] = None
            results[f"{metric}_ci_upper"] = None

    return results

def bootstrap_results_to_dataframe(
    results: list[dict],
) -> pd.DataFrame:
    """
    Convert bootstrap classification results into
    a model-comparison DataFrame.
    """

    return pd.DataFrame(results)

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