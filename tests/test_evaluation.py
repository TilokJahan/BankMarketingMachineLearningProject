import pytest

from src.evaluation import (
    cross_validate_classifier,
    evaluate_binary_classifier,
    get_confusion_matrix,
    plot_confusion_matrix,
    plot_roc_curve,
    results_to_dataframe,
)

import matplotlib
matplotlib.use("Agg")


from sklearn.dummy import DummyClassifier

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

    assert display.confusion_matrix.shape == (2, 2)

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