"""
Reusable K-Nearest Neighbors (KNN) pipeline and hyperparameter tuning
for the Bank Marketing classification project.
"""

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .preprocessing import build_preprocessor


def build_knn_pipeline(numerical_features, categorical_features):
    """
    Build a KNN classification pipeline with shared preprocessing.

    Parameters
    ----------
    numerical_features : list
        Numerical predictor columns.
    categorical_features : list
        Categorical predictor columns.

    Returns
    -------
    Pipeline
        KNN classification pipeline with preprocessing.
    """

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform",
            metric="minkowski",
            p=2
        ))
    ])

    return pipeline


def tune_knn(pipeline, X_train, y_train, random_state=42):
    """
    Tune KNN using stratified 3-fold cross-validation.

    Parameters
    ----------
    pipeline : Pipeline
        KNN classification pipeline.
    X_train : DataFrame
        Training features.
    y_train : Series
        Training target labels.
    random_state : int
        Random seed for cross-validation.

    Returns
    -------
    GridSearchCV
        Fitted grid search containing the best model and parameters.
    """

    param_grid = {
        "classifier__n_neighbors": [5, 11, 21, 31],
        "classifier__weights": ["uniform", "distance"]
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=random_state
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    return grid_search