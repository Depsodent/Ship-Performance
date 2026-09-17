"""Physically motivated, target-free transformations for the efficiency pipeline."""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MaritimeFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create transparent voyage features without using the efficiency target.

    These interactions represent simplified propulsion/load/schedule relationships:
    power-speed demand, power under load, speed-squared resistance proxy, distance
    per turnaround hour, and installed power per cargo ton. They are calculated
    inside the sklearn pipeline after splitting, so they cannot leak the target.
    """

    def fit(self, x: pd.DataFrame, y: object = None) -> "MaritimeFeatureEngineer":
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        frame = x.copy()
        power = pd.to_numeric(frame["Engine_Power_kW"], errors="coerce")
        speed = pd.to_numeric(frame["Speed_Over_Ground_knots"], errors="coerce")
        load = pd.to_numeric(frame["Average_Load_Percentage"], errors="coerce") / 100.0
        distance = pd.to_numeric(frame["Distance_Traveled_nm"], errors="coerce")
        turnaround = pd.to_numeric(frame["Turnaround_Time_hours"], errors="coerce")
        cargo = pd.to_numeric(frame["Cargo_Weight_tons"], errors="coerce")
        frame["Power_Speed_Product"] = power * speed
        frame["Power_Load_Product"] = power * load
        frame["Speed_Squared"] = speed ** 2
        frame["Distance_Per_Turnaround_Hour"] = distance / turnaround.where(turnaround > 0)
        frame["Power_Per_Cargo_Ton"] = power / cargo.where(cargo > 0)
        return frame
