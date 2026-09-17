"""Exhaustive grid-search baseline: transparent and exact for this small demo."""
import time
from optimization.problem import FleetProblem

def optimize_classical(problem: FleetProblem) -> dict:
    start = time.perf_counter(); best = None; history = []; evaluations = 0
    for candidate in problem.candidates():
        result = problem.evaluate(candidate); evaluations += 1
        if best is None or result["objective"] < best["objective"]: best = result
        history.append(best["objective"])
    best.update({"algorithm": "Classical exhaustive grid search", "runtime_seconds": time.perf_counter() - start,
                 "evaluations": evaluations, "convergence_history": history})
    return best
