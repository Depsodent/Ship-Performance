"""Run and save a benchmark using simulated optimization scenarios."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUTPUT_DIR
from optimization.benchmark import benchmark
from optimization.problem import FleetProblem
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True); results = benchmark(FleetProblem(1000, .15, 12000, 200000))
    (OUTPUT_DIR / "benchmark_results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print("Saved outputs/benchmark_results.json")
