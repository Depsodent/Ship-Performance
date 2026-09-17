"""Input validation shared by prediction and UI."""
from typing import Mapping
from config import FEATURE_COLUMNS

def validate_prediction_input(values: Mapping[str, object]) -> None:
    missing = [c for c in FEATURE_COLUMNS if c not in values]
    if missing: raise ValueError(f"Missing required fields: {', '.join(missing)}")
    for key in ("Distance_Traveled_nm", "Speed_Over_Ground_knots", "Engine_Power_kW"):
        if float(values[key]) <= 0: raise ValueError(f"{key} must be greater than zero.")
    if float(values["Cargo_Weight_tons"]) < 0: raise ValueError("Cargo_Weight_tons cannot be negative.")
