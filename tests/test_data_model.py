"""Integration tests using the supplied real CSV, never generated training data."""
import pandas as pd
from config import DATASET_PATH, FEATURE_COLUMNS, MODEL_PATH, TARGET_COLUMN
from models.efficiency_predictor import predict_efficiency
from models.features import MaritimeFeatureEngineer
from models.preprocessing import build_preprocessor, load_and_validate_dataset
from models.train_efficiency_model import train_model


def test_real_dataset_loads_and_has_expected_schema():
    frame = load_and_validate_dataset(DATASET_PATH)
    assert frame.shape == (2736, 18)
    assert TARGET_COLUMN in frame
    assert set(FEATURE_COLUMNS).issubset(frame.columns)


def test_preprocessor_handles_real_missing_categories():
    frame = load_and_validate_dataset(DATASET_PATH)
    engineered = MaritimeFeatureEngineer().fit_transform(frame[FEATURE_COLUMNS])
    transformed = build_preprocessor().fit_transform(engineered)
    assert transformed.shape[0] == len(frame)


def test_training_saves_complete_pipeline_and_predicts():
    artifact = MODEL_PATH.parent / "test_efficiency_model.joblib"
    metrics = train_model(DATASET_PATH, artifact, n_estimators=10)
    row = pd.read_csv(DATASET_PATH).iloc[0][FEATURE_COLUMNS].to_dict()
    prediction = predict_efficiency(row, artifact)
    assert artifact.exists()
    assert artifact.with_name("model_metadata.json").exists()
    assert metrics["rows"] == 2736
    assert len(metrics["model_comparison"]) == 6
    assert prediction > 0
