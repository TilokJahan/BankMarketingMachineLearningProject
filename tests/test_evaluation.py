import pytest

from src.evaluation import (
    evaluate_binary_classifier,
    get_confusion_matrix,
    results_to_dataframe,
)


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