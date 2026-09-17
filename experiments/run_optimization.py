"""Run a reproducible demo optimization after supplying an observed/predicted efficiency."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimization.problem import FleetProblem
from optimization.conventional_optimizer import optimize_classical
if __name__ == "__main__":
    print(json.dumps(optimize_classical(FleetProblem(1000, .15, 12000, 200000)), indent=2, default=str))
