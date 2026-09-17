"""Reusable, guarded efficiency inference."""
from pathlib import Path
import joblib
import pandas as pd
from config import FEATURE_COLUMNS, MODEL_PATH
from utils.validation import validate_prediction_input

def predict_efficiency(values: dict, model_path: Path = MODEL_PATH) -> float:
    validate_prediction_input(values)
    if not model_path.exists():
        raise FileNotFoundError("Model not trained. Run: python experiments/train_model.py first.")
    value = float(joblib.load(model_path).predict(pd.DataFrame([{k: values[k] for k in FEATURE_COLUMNS}]))[0])
    if value <= 0: raise ValueError("Model returned a non-positive efficiency; check training data.")
    return value
