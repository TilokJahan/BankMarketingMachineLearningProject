"""
Reusable Logistic Regression pipeline and hyperparameter tuning
for the Bank Marketing classification project.
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.preprocessing import build_preprocessor


def build_logistic_pipeline(
    numerical_features,
    categorical_features,
    random_state=42
):
    """
    Build a Logistic Regression pipeline with shared preprocessing.

    Parameters
    ----------
    numerical_features : list
        Numerical predictor columns.
    categorical_features : list
        Categorical predictor columns.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Pipeline
        Logistic Regression pipeline with preprocessing.
    """

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ))
    ])

    return pipeline


def tune_logistic_regression(
    pipeline,
    X_train,
    y_train,
    random_state=42
):
    """
    Tune Logistic Regression using stratified 5-fold cross-validation.

    Parameters
    ----------
    pipeline : Pipeline
        Logistic Regression pipeline.
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
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__class_weight": [None, "balanced"]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    return grid_search