"""Reusable preprocessing utilities for the Bank Marketing project."""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def build_preprocessor(numeric_features, categorical_features):
    """
    Create a preprocessing pipeline for mixed numeric/categorical data.

    Parameters
    ----------
    numeric_features : list[str]
        Numeric predictor column names.
    categorical_features : list[str]
        Categorical predictor column names.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        A fitted-ready preprocessing object.
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
