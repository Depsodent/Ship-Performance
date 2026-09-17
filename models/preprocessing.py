"""Dataset inspection and leakage-safe preprocessing builders."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from config import CATEGORICAL_FEATURES, ENGINEERED_NUMERIC_FEATURES, NUMERIC_FEATURES, FEATURE_COLUMNS, TARGET_COLUMN

def load_and_validate_dataset(path: Path) -> pd.DataFrame:
    """Load the real CSV and validate the columns used by this prototype.

    The source Date column is deliberately excluded: the dashboard does not collect a
    voyage date, so feeding its raw string to the model would be inconsistent. Date
    feature engineering can be added later only when it is also supplied at inference.
    """
    if not path.exists(): raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Dataset is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Dataset has no rows.")
    return df

def dataset_summary(df: pd.DataFrame) -> dict:
    return {"rows": len(df), "columns": len(df.columns), "missing_values": df.isna().sum().to_dict(),
            "numerical": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES}

def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES),
                              ("categorical", categorical, CATEGORICAL_FEATURES)])
