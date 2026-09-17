"""Train and persist an efficiency model using only observed efficiency targets."""
import time
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from config import FEATURE_COLUMNS, MODEL_PATH, RANDOM_SEED, TARGET_COLUMN
from models.preprocessing import build_preprocessor, load_and_validate_dataset

def train_model(dataset_path: Path, model_path: Path = MODEL_PATH) -> dict:
    df = load_and_validate_dataset(dataset_path).dropna(subset=[TARGET_COLUMN])
    x_train, x_test, y_train, y_test = train_test_split(df[FEATURE_COLUMNS], df[TARGET_COLUMN], test_size=.2, random_state=RANDOM_SEED)
    model = Pipeline([("preprocess", build_preprocessor()), ("regressor", RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1))])
    start = time.perf_counter(); model.fit(x_train, y_train); elapsed = time.perf_counter() - start
    pred = model.predict(x_test); model_path.parent.mkdir(exist_ok=True); joblib.dump(model, model_path)
    return {"model": "RandomForestRegressor", "MAE": mean_absolute_error(y_test, pred),
            "RMSE": mean_squared_error(y_test, pred) ** .5, "R2": r2_score(y_test, pred),
            "training_seconds": elapsed, "model_path": str(model_path), "rows": len(df)}
