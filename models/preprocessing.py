"""Dataset loading and leakage-safe preprocessing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import (
    CATEGORICAL_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    NUMERIC_FEATURES,
    AUTO_CATEGORICAL_FEATURES,
    AUTO_ENGINEERED_NUMERIC_FEATURES,
    AUTO_FEATURE_COLUMNS,
    AUTO_NUMERIC_FEATURES,
    AUTO_TARGET_COLUMN,
    detect_dataset_profile,

)


def load_and_validate_dataset(path: Path) -> pd.DataFrame:
    """Load a supported CSV and validate its schema."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    if df.empty:
        raise ValueError(
            "Dataset has no rows."
        )

    # Remove accidental spaces/BOM characters from headers.
    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    profile = detect_dataset_profile(
        df.columns
    )

    if profile == "maritime":
        required = (
            FEATURE_COLUMNS
            + [TARGET_COLUMN]
        )

    elif profile == "auto":
        required = (
            AUTO_FEATURE_COLUMNS
            + [AUTO_TARGET_COLUMN]
        )

    else:
        raise ValueError(
            f"Unsupported dataset profile: {profile}"
        )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{missing}"
        )

    df.attrs["dataset_profile"] = profile

    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return basic dataset information."""

    profile = df.attrs.get(
        "dataset_profile",
        "unknown",
    )

    if profile == "auto":
        numerical = AUTO_NUMERIC_FEATURES
        categorical = AUTO_CATEGORICAL_FEATURES
    else:
        numerical = NUMERIC_FEATURES
        categorical = CATEGORICAL_FEATURES

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": (
            df.isna()
            .sum()
            .to_dict()
        ),
        "profile": profile,
        "numerical": numerical,
        "categorical": categorical,
    }


def build_preprocessor(
    profile: str = "maritime",
) -> ColumnTransformer:
    """Build preprocessing for the selected dataset profile."""

    if profile == "maritime":

        numeric_features = (
            NUMERIC_FEATURES
            + ENGINEERED_NUMERIC_FEATURES
        )

        categorical_features = (
            CATEGORICAL_FEATURES
        )

    elif profile == "auto":

        numeric_features = (
            AUTO_NUMERIC_FEATURES
            + AUTO_ENGINEERED_NUMERIC_FEATURES
        )

        categorical_features = (
            AUTO_CATEGORICAL_FEATURES
        )

    else:

        raise ValueError(
            f"Unsupported preprocessing profile: {profile}"
        )

    numeric_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )