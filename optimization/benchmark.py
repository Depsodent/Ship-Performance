"""Fair side-by-side optimizer benchmark."""
from optimization.conventional_optimizer import optimize_classical
from optimization.quantum_optimizer import optimize_quantum_inspired
from optimization.problem import FleetProblem

def benchmark(problem: FleetProblem, population_size: int = 30, iterations: int = 60, seed: int = 42) -> dict:
    classical = optimize_classical(problem)
    quantum = optimize_quantum_inspired(problem, population_size, iterations, seed=seed)
    return {"classical": classical, "quantum_inspired": quantum}
