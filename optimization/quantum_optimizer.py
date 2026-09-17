"""Quantum-inspired evolutionary search running entirely on classical CPU hardware."""
import time
import numpy as np
from optimization.problem import FleetProblem

def optimize_quantum_inspired(problem: FleetProblem, population_size: int = 30, iterations: int = 60,
                               update_strength: float = .12, seed: int = 42) -> dict:
    """Sample categorical Q-bit-like probability amplitudes, then rotate toward elites.

    Each variable is represented by a probability vector (not a physical qubit). Observation
    samples choices from it; after each generation, probability mass rotates toward the best
    observed candidate. This is quantum-inspired terminology, not quantum computation.
    """
    rng = np.random.default_rng(seed); dimensions = [len(problem.capacities_tons), len(problem.speed_options), len(problem.fuel_options), problem.max_fleet_size]
    probabilities = [np.ones(n) / n for n in dimensions]; best = None; history = []; evaluations = 0; start = time.perf_counter()
    for _ in range(iterations):
        generation = []
        for _ in range(population_size):
            candidate = tuple(int(rng.choice(len(p), p=p)) for p in probabilities[:3]) + (int(rng.choice(dimensions[3], p=probabilities[3])) + 1,)
            value = problem.evaluate(candidate); value["_candidate"] = candidate; generation.append(value); evaluations += 1
        elite = min(generation, key=lambda x: x["objective"])
        if best is None or elite["objective"] < best["objective"]: best = elite.copy()
        # Rotation/update: retain exploration while increasing elite-choice probability.
        for dim, choice in enumerate(best["_candidate"]):
            choice = choice - 1 if dim == 3 else choice
            target = np.zeros(dimensions[dim]); target[choice] = 1
            probabilities[dim] = (1 - update_strength) * probabilities[dim] + update_strength * target
            probabilities[dim] /= probabilities[dim].sum()
        history.append(best["objective"])
    best.pop("_candidate", None)
    best.update({"algorithm": "Quantum-Inspired Probability Search (classical CPU)", "runtime_seconds": time.perf_counter() - start,
                 "evaluations": evaluations, "convergence_history": history})
    return best
