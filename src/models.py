"""Candidate model definitions."""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
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



def get_student3_models():
    """
    Return the classifiers assigned to Student 3.
    """
    return {
        "Quadratic Discriminant Analysis":
            QuadraticDiscriminantAnalysis(
                reg_param=0.1
            ),
        "Gaussian Naive Bayes":
            GaussianNB(),
    }

def get_supplementary_models():
    """
    Return supplementary classifiers evaluated beyond the
    project's five primary candidate models.
    """
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }