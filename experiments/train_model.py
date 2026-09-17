"""Command-line training entry point."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATASET_PATH
from models.train_efficiency_model import train_model
if __name__ == "__main__":
    metrics = train_model(DATASET_PATH)
    print("=" * 32)
    print("MODEL TRAINING COMPLETE")
    print("=" * 32)
    print(f"Dataset: {DATASET_PATH.relative_to(Path.cwd()) if DATASET_PATH.is_relative_to(Path.cwd()) else DATASET_PATH}")
    print(f"Rows: {metrics['rows']}")
    print(f"Target: {metrics['target_column']}")
    print(f"Selected model: {metrics['selected_model']}")
    print(f"MAE: {metrics['MAE']:.6f}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"R2: {metrics['R2']:.6f}")
    print(f"Model saved: {metrics['model_path']}")
    print(f"Metadata saved: {metrics['metadata_path']}")
    print("=" * 32)
