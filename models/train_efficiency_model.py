"""Train, compare and save efficiency regressors using the supplied real CSV."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
# HistGradientBoosting uses OpenMP/joblib internally. One thread avoids restricted
# Windows worker-pool IPC while keeping the comparison deterministic.
os.environ.setdefault("OMP_NUM_THREADS", "1")

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import CATEGORICAL_FEATURES, ENGINEERED_NUMERIC_FEATURES, FEATURE_COLUMNS, MODEL_PATH, NUMERIC_FEATURES, RANDOM_SEED, TARGET_COLUMN
from models.features import MaritimeFeatureEngineer
from models.preprocessing import build_preprocessor, dataset_summary, load_and_validate_dataset


def _pipeline(regressor: object) -> Pipeline:
    """Make an independent complete preprocessing-plus-estimator pipeline."""
    return Pipeline([("feature_engineering", MaritimeFeatureEngineer()), ("preprocess", build_preprocessor()), ("regressor", regressor)])


def investigate_dataset(df: pd.DataFrame) -> dict:
    """Return descriptive facts; it does not make causal claims about the data."""
    correlations = df[NUMERIC_FEATURES + [TARGET_COLUMN]].corr(numeric_only=True)[TARGET_COLUMN]
    engineered = MaritimeFeatureEngineer().transform(df[FEATURE_COLUMNS])
    engineered_correlations = engineered[ENGINEERED_NUMERIC_FEATURES].assign(**{TARGET_COLUMN: df[TARGET_COLUMN]}).corr(numeric_only=True)[TARGET_COLUMN]
    dates = pd.to_datetime(df["Date"], errors="coerce") if "Date" in df else pd.Series(dtype="datetime64[ns]")
    return {
        "missing_values": {key: int(value) for key, value in df.isna().sum().items()},
        "target_distribution": {key: float(value) for key, value in df[TARGET_COLUMN].describe().items()},
        "numeric_feature_target_correlations": {key: float(value) for key, value in correlations.drop(TARGET_COLUMN).items()},
        "engineered_feature_target_correlations": {key: float(value) for key, value in engineered_correlations.drop(TARGET_COLUMN).items()},
        "categorical_distributions": {column: {str(k): int(v) for k, v in df[column].fillna("<MISSING>").value_counts().items()} for column in CATEGORICAL_FEATURES},
        "date_analysis": {
            "present": "Date" in df, "valid_dates": int(dates.notna().sum()), "invalid_dates": int(dates.isna().sum()),
            "minimum": None if dates.empty else str(dates.min().date()), "maximum": None if dates.empty else str(dates.max().date()),
            "handling": "Excluded: dashboard has no voyage-date input, so raw Date strings or fixed defaults would make inference inconsistent.",
        },
    }


def train_model(dataset_path: Path, model_path: Path = MODEL_PATH, n_estimators: int = 200) -> dict:
    """Compare regressors on one held-out split and save the actual best pipeline."""
    df = load_and_validate_dataset(dataset_path).dropna(subset=[TARGET_COLUMN]).copy()
    if len(df) < 2:
        raise ValueError("At least two rows with a non-missing efficiency target are required.")
    x_train, x_test, y_train, y_test = train_test_split(df[FEATURE_COLUMNS], df[TARGET_COLUMN], test_size=.2, random_state=RANDOM_SEED)
    candidates = {
        "DummyRegressor (mean)": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_SEED, n_jobs=1),
        "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=n_estimators, random_state=RANDOM_SEED, n_jobs=1),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=RANDOM_SEED),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=RANDOM_SEED),
    }
    results, fitted = [], {}
    for name, estimator in candidates.items():
        model = _pipeline(estimator); start = time.perf_counter()
        try:
            model.fit(x_train, y_train)
            prediction = model.predict(x_test)
            results.append({"Model": name, "status": "completed", "MAE": float(mean_absolute_error(y_test, prediction)), "RMSE": float(mean_squared_error(y_test, prediction) ** .5), "R2": float(r2_score(y_test, prediction)), "training_seconds": float(time.perf_counter() - start)})
            fitted[name] = model
        except PermissionError as error:
            # This estimator is still reported, but cannot use its internal worker pool
            # on some restricted Windows environments. Do not substitute a score.
            results.append({"Model": name, "status": "not evaluated", "error": str(error), "training_seconds": float(time.perf_counter() - start)})
    completed = [row for row in results if row["status"] == "completed"]
    if not completed:
        raise RuntimeError("No candidate model could be trained.")
    completed.sort(key=lambda row: (-row["R2"], row["RMSE"], row["MAE"]))
    failed = [row for row in results if row["status"] != "completed"]
    results = completed + failed
    selected = completed[0]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[selected["Model"]], model_path)
    summary = dataset_summary(df)
    metadata = {
        "selected_model": selected["Model"], "dataset_path": str(dataset_path), "rows": len(df), "dataset_columns": summary["columns"],
        "target_column": TARGET_COLUMN, "feature_columns": FEATURE_COLUMNS, "numeric_features": NUMERIC_FEATURES, "engineered_numeric_features": ENGINEERED_NUMERIC_FEATURES, "categorical_features": CATEGORICAL_FEATURES,
        "test_size": .2, "random_state": RANDOM_SEED, "MAE": selected["MAE"], "RMSE": selected["RMSE"], "R2": selected["R2"],
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(), "model_path": str(model_path), "model_comparison": results,
        "dataset_investigation": investigate_dataset(df),
    }
    metadata_path = model_path.with_name("model_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)
    return metadata
