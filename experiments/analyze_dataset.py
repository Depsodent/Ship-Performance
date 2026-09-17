"""Print an evidence-based diagnostic report for the supplied efficiency dataset."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATASET_PATH, NUMERIC_FEATURES, OUTPUT_DIR, TARGET_COLUMN
from models.preprocessing import load_and_validate_dataset
from models.train_efficiency_model import investigate_dataset


def iqr_outlier_counts(frame):
    """Count values outside 1.5 IQR bounds; this is descriptive, not auto-removal."""
    counts = {}
    for column in NUMERIC_FEATURES + [TARGET_COLUMN]:
        q1, q3 = frame[column].quantile([.25, .75])
        iqr = q3 - q1
        counts[column] = int(((frame[column] < q1 - 1.5 * iqr) | (frame[column] > q3 + 1.5 * iqr)).sum())
    return counts


if __name__ == "__main__":
    data = load_and_validate_dataset(DATASET_PATH)
    report = investigate_dataset(data)
    report["duplicate_rows"] = int(data.duplicated().sum())
    report["numerical_distributions"] = {
        column: {name: float(value) for name, value in data[column].describe().items()}
        for column in NUMERIC_FEATURES
    }
    report["iqr_outlier_counts"] = iqr_outlier_counts(data)
    report["suitability_assessment"] = (
        "The report describes weak or strong associations only; it does not alter values, remove outliers, "
        "or create a target-derived feature. Use it to decide whether additional real vessel telemetry is required."
    )
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "dataset_diagnostic.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("DATASET DIAGNOSTIC REPORT")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")
    print(f"Target: {TARGET_COLUMN}")
    print(f"Duplicate rows: {report['duplicate_rows']}")
    print("Missing values:", report["missing_values"])
    print("Numeric feature correlations with target:", report["numeric_feature_target_correlations"])
    print("Engineered feature correlations with target:", report["engineered_feature_target_correlations"])
    print("Date analysis:", report["date_analysis"])
    print(f"Full JSON report saved to: {output_path}")
