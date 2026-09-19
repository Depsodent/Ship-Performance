"""Reusable, guarded Auto-MPG fuel-consumption inference."""
from pathlib import Path

import joblib
import pandas as pd

from config import AUTO_FEATURE_COLUMNS, MODEL_PATH


def predict_efficiency(values: dict, model_path: Path = MODEL_PATH) -> float:
    """Predict Auto-MPG fuel consumption and return it for the app."""
    if not model_path.exists():
        raise FileNotFoundError(
            "Model not trained. Run: python experiments/train_model.py first."
        )

    missing = [column for column in AUTO_FEATURE_COLUMNS if column not in values]
    if missing:
        raise ValueError(
            f"Missing Auto-MPG prediction inputs: {', '.join(missing)}"
        )

    input_frame = pd.DataFrame(
        [{column: values[column] for column in AUTO_FEATURE_COLUMNS}]
    )

    model = joblib.load(model_path)
    value = float(model.predict(input_frame)[0])

    if value <= 0:
        raise ValueError(
            "Model returned a non-positive fuel consumption value."
        )

    return value