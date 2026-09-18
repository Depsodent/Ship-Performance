"""Leakage-safe feature engineering for maritime and automotive datasets."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MaritimeFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create target-free features for maritime efficiency prediction."""

    def fit(
        self,
        x: pd.DataFrame,
        y: object = None,
    ) -> "MaritimeFeatureEngineer":
        return self

    def transform(
        self,
        x: pd.DataFrame,
    ) -> pd.DataFrame:

        frame = x.copy()

        power = pd.to_numeric(
            frame["Engine_Power_kW"],
            errors="coerce",
        )

        speed = pd.to_numeric(
            frame["Speed_Over_Ground_knots"],
            errors="coerce",
        )

        load = (
            pd.to_numeric(
                frame["Average_Load_Percentage"],
                errors="coerce",
            )
            / 100.0
        )

        distance = pd.to_numeric(
            frame["Distance_Traveled_nm"],
            errors="coerce",
        )

        turnaround = pd.to_numeric(
            frame["Turnaround_Time_hours"],
            errors="coerce",
        )

        cargo = pd.to_numeric(
            frame["Cargo_Weight_tons"],
            errors="coerce",
        )

        frame["Power_Speed_Product"] = (
            power * speed
        )

        frame["Power_Load_Product"] = (
            power * load
        )

        frame["Speed_Squared"] = (
            speed ** 2
        )

        frame["Distance_Per_Turnaround_Hour"] = (
            distance
            / turnaround.where(turnaround > 0)
        )

        frame["Power_Per_Cargo_Ton"] = (
            power
            / cargo.where(cargo > 0)
        )

        return frame


class AutoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create target-free features for automotive fuel prediction."""

    def fit(
        self,
        x: pd.DataFrame,
        y: object = None,
    ) -> "AutoFeatureEngineer":
        return self

    def transform(
        self,
        x: pd.DataFrame,
    ) -> pd.DataFrame:

        frame = x.copy()

        mass = pd.to_numeric(
            frame["m (kg)"],
            errors="coerce",
        )

        test_mass = pd.to_numeric(
            frame["Mt"],
            errors="coerce",
        )

        capacity = pd.to_numeric(
            frame["ec (cm3)"],
            errors="coerce",
        )

        power = pd.to_numeric(
            frame["ep (KW)"],
            errors="coerce",
        )

        frame["Power_per_Mass"] = (
            power
            / mass.where(mass > 0)
        )

        frame["Power_per_Test_Mass"] = (
            power
            / test_mass.where(test_mass > 0)
        )

        frame["Power_per_Engine_Capacity"] = (
            power
            / capacity.where(capacity > 0)
        )

        frame["Mass_per_Power"] = (
            mass
            / power.where(power > 0)
        )

        frame["Engine_Capacity_per_Mass"] = (
            capacity
            / mass.where(mass > 0)
        )

        return frame