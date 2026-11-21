import matplotlib
matplotlib.use("Agg")

import pytest

from src.evaluation import (
    bootstrap_classification_metrics,
    bootstrap_results_to_dataframe,    
    cross_validate_classifier,
    cross_validation_results_to_dataframe,
    evaluate_binary_classifier,
    get_confusion_matrix,
    plot_confusion_matrix,
    plot_roc_curve,
    results_to_dataframe,
)


from sklearn.dummy import DummyClassifier

import numpy as np


def test_evaluate_binary_classifier():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 0, 1]
    y_score = [0.1, 0.2, 0.4, 0.9]

    results = evaluate_binary_classifier(
        model_name="Test Model",
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
    )

    assert results["model"] == "Test Model"
    assert results["accuracy"] == 0.75
    assert results["precision"] == 1.0
    assert results["recall"] == 0.5
    assert results["f1"] > 0
    assert results["roc_auc"] == pytest.approx(1.0)


def test_confusion_matrix():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 0, 1]

    cm = get_confusion_matrix(
        y_true,
        y_pred,
    )

    assert cm.shape == (2, 2)


def test_results_dataframe():
    results = [
        {
            "model": "Model A",
            "accuracy": 0.8,
            "precision": 0.7,
            "recall": 0.6,
            "f1": 0.65,
            "roc_auc": 0.75,
        }
    ]

    df = results_to_dataframe(results)

    assert len(df) == 1
    assert df.loc[0, "model"] == "Model A"

def test_plot_confusion_matrix():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 0, 1]

    display = plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        model_name="Test Model",
    )

    expected = np.array([
        [2, 0],
        [1, 1],
    ])

    np.testing.assert_array_equal(
        display.confusion_matrix,
        expected,
    )

def test_plot_roc_curve():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.7, 0.9]

    display = plot_roc_curve(
        y_true=y_true,
        y_score=y_score,
        model_name="Test Model",
    )

    assert display.roc_auc == pytest.approx(1.0)

def test_cross_validate_classifier():
    X = [
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
        [10],
    ]

    y = [
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
    ]

    model = DummyClassifier(
        strategy="most_frequent"
    )

    results = cross_validate_classifier(
        model_name="Test Model",
        estimator=model,
        X=X,
        y=y,
        cv=5,
    )

    assert results["model"] == "Test Model"
    assert "accuracy_mean" in results
    assert "recall_mean" in results
    assert "roc_auc_mean" in results


def test_cross_validation_results_to_dataframe():
    results = [
        {
            "model": "Model A",
            "accuracy_mean": 0.8,
            "accuracy_std": 0.01,
            "precision_mean": 0.7,
            "precision_std": 0.02,
            "recall_mean": 0.6,
            "recall_std": 0.03,
            "f1_mean": 0.65,
            "f1_std": 0.02,
            "roc_auc_mean": 0.75,
            "roc_auc_std": 0.01,
        }
    ]

    df = cross_validation_results_to_dataframe(results)

    assert len(df) == 1
    assert df.loc[0, "model"] == "Model A"

def test_bootstrap_classification_metrics():
    y_true = [
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
    ]

    y_pred = [
        0, 0, 0, 0, 1,
        0, 1, 1, 1, 1,
    ]

    y_score = [
        0.05,
        0.10,
        0.20,
        0.25,
        0.60,
        0.30,
        0.65,
        0.75,
        0.85,
        0.95,
    ]

    results = bootstrap_classification_metrics(
        model_name="Test Model",
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        n_bootstrap=100,
        random_state=42,
    )

    assert results["model"] == "Test Model"

    assert results["n_bootstrap"] == 100

    assert "accuracy_ci_lower" in results
    assert "accuracy_ci_upper" in results

    assert "f1_ci_lower" in results
    assert "f1_ci_upper" in results

    assert "roc_auc_ci_lower" in results
    assert "roc_auc_ci_upper" in results

    assert (
        results["accuracy_ci_lower"]
        <= results["accuracy"]
        <= results["accuracy_ci_upper"]
    )


def test_bootstrap_results_to_dataframe():

    results = [
        {
            "model": "Model A",
            "accuracy": 0.8,
            "accuracy_ci_lower": 0.7,
            "accuracy_ci_upper": 0.9,
        }
    ]

    df = bootstrap_results_to_dataframe(
        results
    )

    assert len(df) == 1
    assert df.loc[0, "model"] == "Model A"