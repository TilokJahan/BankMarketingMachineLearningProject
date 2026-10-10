"""Candidate model definitions."""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.naive_bayes import GaussianNB


def get_candidate_models():
    """
    Return initial candidate classifiers.

    Note:
    Some models may require preprocessing adjustments because dense/sparse
    matrix support differs across estimators.
    """
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "knn": KNeighborsClassifier(),
        "lda": LinearDiscriminantAnalysis(),
        "qda": QuadraticDiscriminantAnalysis(),
        "gaussian_nb": GaussianNB(),
    }
