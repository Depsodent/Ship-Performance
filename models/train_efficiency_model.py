"""Generalized ML training engine for supported dataset profiles."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path

import joblib
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from config import (
    # Maritime
    CATEGORICAL_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    FEATURE_COLUMNS,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    TARGET_COLUMN,

    # Automotive
    AUTO_CATEGORICAL_FEATURES,
    AUTO_ENGINEERED_NUMERIC_FEATURES,
    AUTO_FEATURE_COLUMNS,
    AUTO_NUMERIC_FEATURES,
    AUTO_TARGET_COLUMN,

    detect_dataset_profile,
)

from models.features import (
    MaritimeFeatureEngineer,
    AutoFeatureEngineer,
)

from models.preprocessing import (
    build_preprocessor,
    dataset_summary,
    load_and_validate_dataset,
)


# Keep tree models single-threaded for reliable Windows execution.
os.environ.setdefault("OMP_NUM_THREADS", "1")


def get_profile_configuration(profile: str) -> dict:
    """Return target/features/engineer configuration for a dataset profile."""

    if profile == "maritime":
        return {
            "target": TARGET_COLUMN,
            "features": FEATURE_COLUMNS,
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "engineered": ENGINEERED_NUMERIC_FEATURES,
            "engineer": MaritimeFeatureEngineer,
        }

    if profile == "auto":
        return {
            "target": AUTO_TARGET_COLUMN,
            "features": AUTO_FEATURE_COLUMNS,
            "numeric": AUTO_NUMERIC_FEATURES,
            "categorical": AUTO_CATEGORICAL_FEATURES,
            "engineered": AUTO_ENGINEERED_NUMERIC_FEATURES,
            "engineer": AutoFeatureEngineer,
        }

    raise ValueError(
        f"Unsupported dataset profile: {profile}"
    )


def build_pipeline(
    regressor: object,
    profile: str,
) -> Pipeline:
    """Create the complete feature-engineering + preprocessing + model pipeline."""

    configuration = get_profile_configuration(
        profile
    )

    feature_engineer = configuration[
        "engineer"
    ]()

    return Pipeline(
        [
            (
                "feature_engineering",
                feature_engineer,
            ),
            (
                "preprocess",
                build_preprocessor(profile),
            ),
            (
                "regressor",
                regressor,
            ),
        ]
    )


def investigate_dataset(
    df: pd.DataFrame,
    profile: str,
) -> dict:
    """Generate descriptive information for the detected dataset profile."""

    configuration = get_profile_configuration(
        profile
    )

    target = configuration["target"]
    features = configuration["features"]
    numeric = configuration["numeric"]
    categorical = configuration["categorical"]
    engineered_names = configuration["engineered"]
    engineer = configuration["engineer"]

    # Numeric correlations.
    numeric_with_target = (
        df[numeric + [target]]
        .corr(numeric_only=True)[target]
    )

    # Engineered correlations.
    engineered = engineer().transform(
        df[features]
    )

    engineered_with_target = (
        engineered[engineered_names]
        .copy()
    )

    engineered_with_target[target] = (
        df[target].values
    )

    engineered_correlations = (
        engineered_with_target
        .corr(numeric_only=True)[target]
    )

    return {
        "profile": profile,

        "missing_values": {
            key: int(value)
            for key, value in df.isna().sum().items()
        },

        "duplicate_rows": int(
            df.duplicated().sum()
        ),

        "target_distribution": {
            key: float(value)
            for key, value in df[target]
            .describe()
            .items()
        },

        "numeric_feature_target_correlations": {
            key: float(value)
            for key, value in (
                numeric_with_target
                .drop(target)
                .items()
            )
        },

        "engineered_feature_target_correlations": {
            key: float(value)
            for key, value in (
                engineered_correlations
                .drop(target)
                .items()
            )
        },

        "categorical_distributions": {
            column: {
                str(k): int(v)
                for k, v in (
                    df[column]
                    .fillna("<MISSING>")
                    .value_counts()
                    .items()
                )
            }
            for column in categorical
        },
    }


def train_model(
    dataset_path: Path,
    model_path: Path = MODEL_PATH,
    n_estimators: int = 200,
) -> dict:
    """
    Train candidate models using 5-fold cross-validation.

    The dataset profile is detected automatically from the CSV schema.

    The final selected model is evaluated once on an untouched test set.
    """

    # ---------------------------------------------------------
    # 1. Load and detect dataset
    # ---------------------------------------------------------

    df = load_and_validate_dataset(
        dataset_path
    )

    profile = df.attrs.get(
        "dataset_profile"
    )

    if not profile:
        profile = detect_dataset_profile(
            df.columns
        )

    configuration = get_profile_configuration(
        profile
    )

    target = configuration["target"]
    features = configuration["features"]

    # Keep profile metadata after copying.
    df = df.dropna(
        subset=[target]
    ).copy()

    df.attrs["dataset_profile"] = profile

    if len(df) < 10:
        raise ValueError(
            "At least 10 rows with a non-missing target "
            "are required."
        )

    print(
        f"Dataset profile: {profile}"
    )

    print(
        f"Target: {target}"
    )

    # ---------------------------------------------------------
    # 2. Final untouched test set
    # ---------------------------------------------------------

    x_train, x_test, y_train, y_test = (
        train_test_split(
            df[features],
            df[target],
            test_size=0.20,
            random_state=RANDOM_SEED,
        )
    )

    # ---------------------------------------------------------
    # 3. Candidate models
    # ---------------------------------------------------------

    candidates = {
        "DummyRegressor (mean)": (
            DummyRegressor(
                strategy="mean"
            )
        ),

        "LinearRegression": (
            LinearRegression()
        ),

        "Ridge": (
            Ridge(
                alpha=1.0
            )
        ),

        "RandomForestRegressor": (
            RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=RANDOM_SEED,
                n_jobs=1,
            )
        ),

        "ExtraTreesRegressor": (
            ExtraTreesRegressor(
                n_estimators=n_estimators,
                random_state=RANDOM_SEED,
                n_jobs=1,
            )
        ),

        "HistGradientBoostingRegressor": (
            HistGradientBoostingRegressor(
                random_state=RANDOM_SEED
            )
        ),

        "GradientBoostingRegressor": (
            GradientBoostingRegressor(
                random_state=RANDOM_SEED
            )
        ),
    }

    # ---------------------------------------------------------
    # 4. Five-fold cross-validation
    # ---------------------------------------------------------

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }

    results = []
    fitted_models = {}

    for name, estimator in candidates.items():

        print(
            f"Evaluating: {name}"
        )

        model = build_pipeline(
            estimator,
            profile,
        )

        start = time.perf_counter()

        try:

            cv_result = cross_validate(
                model,
                x_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                return_train_score=False,
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            mae_scores = (
                -cv_result["test_MAE"]
            )

            rmse_scores = (
                -cv_result["test_RMSE"]
            )

            r2_scores = (
                cv_result["test_R2"]
            )

            row = {
                "Model": name,
                "status": "completed",

                "CV_MAE_mean": float(
                    mae_scores.mean()
                ),

                "CV_MAE_std": float(
                    mae_scores.std()
                ),

                "CV_RMSE_mean": float(
                    rmse_scores.mean()
                ),

                "CV_RMSE_std": float(
                    rmse_scores.std()
                ),

                "CV_R2_mean": float(
                    r2_scores.mean()
                ),

                "CV_R2_std": float(
                    r2_scores.std()
                ),

                "training_seconds": float(
                    elapsed
                ),
            }

            results.append(row)

            # Fit on the complete training portion.
            model.fit(
                x_train,
                y_train,
            )

            fitted_models[name] = model

        except Exception as error:

            elapsed = (
                time.perf_counter()
                - start
            )

            results.append(
                {
                    "Model": name,
                    "status": "failed",
                    "error": str(error),
                    "training_seconds": float(
                        elapsed
                    ),
                }
            )

    completed = [
        row
        for row in results
        if row["status"] == "completed"
    ]

    if not completed:
        raise RuntimeError(
            "No candidate model could be trained."
        )

    # ---------------------------------------------------------
    # 5. Select model
    # ---------------------------------------------------------

    completed.sort(
        key=lambda row: (
            -row["CV_R2_mean"],
            row["CV_RMSE_mean"],
            row["CV_MAE_mean"],
        )
    )

    selected = completed[0]

    selected_name = selected[
        "Model"
    ]

    selected_model = fitted_models[
        selected_name
    ]

    # ---------------------------------------------------------
    # 6. Final untouched test evaluation
    # ---------------------------------------------------------

    test_prediction = (
        selected_model.predict(
            x_test
        )
    )

    test_mae = mean_absolute_error(
        y_test,
        test_prediction,
    )

    test_rmse = (
        mean_squared_error(
            y_test,
            test_prediction,
        )
        ** 0.5
    )

    test_r2 = r2_score(
        y_test,
        test_prediction,
    )

    # ---------------------------------------------------------
    # 7. Save model
    # ---------------------------------------------------------

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        selected_model,
        model_path,
    )

    # ---------------------------------------------------------
    # 8. Save comparison
    # ---------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    comparison_path = (
        Path("outputs")
        / "model_cv_results.csv"
    )

    comparison_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        comparison_path,
        index=False,
    )

    # ---------------------------------------------------------
    # 9. Dataset investigation
    # ---------------------------------------------------------

    summary = dataset_summary(
        df
    )

    investigation = (
        investigate_dataset(
            df,
            profile,
        )
    )

    # ---------------------------------------------------------
    # 10. Metadata
    # ---------------------------------------------------------

    metadata = {

        "dataset_profile": profile,

        "selected_model": selected_name,

        "dataset_path": str(
            dataset_path
        ),

        "rows": len(df),

        "dataset_columns": (
            summary["columns"]
        ),

        "target_column": target,

        "feature_columns": features,

        "numeric_features": (
            configuration["numeric"]
        ),

        "engineered_numeric_features": (
            configuration["engineered"]
        ),

        "categorical_features": (
            configuration["categorical"]
        ),

        "test_size": 0.20,

        "random_state": RANDOM_SEED,

        "cv_folds": 5,

        # Cross-validation metrics.
        "CV_MAE_mean": selected[
            "CV_MAE_mean"
        ],

        "CV_MAE_std": selected[
            "CV_MAE_std"
        ],

        "CV_RMSE_mean": selected[
            "CV_RMSE_mean"
        ],

        "CV_RMSE_std": selected[
            "CV_RMSE_std"
        ],

        "CV_R2_mean": selected[
            "CV_R2_mean"
        ],

        "CV_R2_std": selected[
            "CV_R2_std"
        ],

        # Final test metrics.
        "test_MAE": float(
            test_mae
        ),

        "test_RMSE": float(
            test_rmse
        ),

        "test_R2": float(
            test_r2
        ),

        # Backward-compatible names.
        "MAE": float(
            test_mae
        ),

        "RMSE": float(
            test_rmse
        ),

        "R2": float(
            test_r2
        ),

        "model_comparison": results,

        "comparison_path": str(
            comparison_path
        ),

        "training_timestamp_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "model_path": str(
            model_path
        ),

        "dataset_investigation": (
            investigation
        ),
    }

    metadata_path = (
        model_path.with_name(
            "model_metadata.json"
        )
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata["metadata_path"] = (
        str(metadata_path)
    )

    return metadata