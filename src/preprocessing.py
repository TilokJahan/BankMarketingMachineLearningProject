"""Reusable preprocessing utilities for the Bank Marketing project."""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def build_preprocessor(
    numeric_features,
    categorical_features,
    dense_output: bool = False,
    drop_first: bool = False,
):
    """
    Create a preprocessing pipeline for mixed numeric/categorical data.

    Parameters
    ----------
    numeric_features : list[str]
        Numeric predictor column names.
    categorical_features : list[str]
        Categorical predictor column names.
    dense_output : bool, default=False
        If True, configure one-hot encoding to return dense output.
    drop_first : bool, default=False
        If True, drop the first category from each categorical feature
        to reduce dummy-variable collinearity.

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
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=not dense_output,
                    drop="first" if drop_first else None,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
