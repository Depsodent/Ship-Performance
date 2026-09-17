from optimization.conventional_optimizer import optimize_classical
from optimization.problem import FleetProblem
from optimization.quantum_optimizer import optimize_quantum_inspired
def test_optimizers_return_result():
    p = FleetProblem(100, 1, 1000, 1_000_000)
    for result in (optimize_classical(p), optimize_quantum_inspired(p, 8, 5)):
        assert "objective" in result and "feasible" in result and result["evaluations"] > 0
