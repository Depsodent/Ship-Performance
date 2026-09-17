"""Command-line training entry point."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATASET_PATH
from models.train_efficiency_model import train_model
if __name__ == "__main__": print(json.dumps(train_model(DATASET_PATH), indent=2))
