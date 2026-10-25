from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis,
)
from sklearn.naive_bayes import GaussianNB

from src.models import get_student3_models


def test_get_student3_models():
    models = get_student3_models()

    assert isinstance(
        models[
            "Quadratic Discriminant Analysis"
        ],
        QuadraticDiscriminantAnalysis,
    )

    assert isinstance(
        models[
            "Gaussian Naive Bayes"
        ],
        GaussianNB,
    )

import pandas as pd

from src.preprocessing import build_preprocessor


def test_dense_preprocessor_output():
    X = pd.DataFrame(
        {
            "age": [25, 40, 55],
            "job": [
                "student",
                "admin.",
                "technician",
            ],
        }
    )

    preprocessor = build_preprocessor(
        numeric_features=["age"],
        categorical_features=["job"],
        dense_output=True,
    )

    transformed = preprocessor.fit_transform(X)

    assert not hasattr(
        transformed,
        "toarray",
    )  