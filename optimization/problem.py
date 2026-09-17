"""A small, explainable constrained fleet-planning problem."""
from dataclasses import dataclass, field
from typing import Any
from config import DEFAULT_WEIGHTS, FUEL_SCENARIOS, MAX_SPEED_KNOTS, MIN_SPEED_KNOTS
from utils.calculations import energy_for_voyages
from utils.fuel_model import estimate_fuel

@dataclass
class FleetProblem:
    distance_nm: float
    efficiency_nm_per_kwh: float
    cargo_demand_tons: float
    emission_limit_kg: float
    max_fleet_size: int = 3
    capacities_tons: tuple[float, ...] = (5000, 10000, 15000)
    speed_options: tuple[float, ...] = (10, 15, 20)
    fuel_options: tuple[str, ...] = ("Diesel", "HFO", "LNG", "Methanol")
    weights: dict[str, float] = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())

    def decode(self, candidate: tuple[int, int, int, int]) -> dict[str, Any]:
        vessel_idx, speed_idx, fuel_idx, voyages = candidate
        return {"vessel_capacity_tons": self.capacities_tons[vessel_idx], "speed_knots": self.speed_options[speed_idx],
                "fuel": self.fuel_options[fuel_idx], "voyages": voyages}

    def candidates(self):
        for v in range(len(self.capacities_tons)):
            for s in range(len(self.speed_options)):
                for f in range(len(self.fuel_options)):
                    for n in range(1, self.max_fleet_size + 1): yield (v, s, f, n)

    def evaluate(self, candidate: tuple[int, int, int, int]) -> dict[str, Any]:
        plan = self.decode(candidate); violations = []
        if not MIN_SPEED_KNOTS <= plan["speed_knots"] <= MAX_SPEED_KNOTS: violations.append("speed outside configured bounds")
        if plan["vessel_capacity_tons"] * plan["voyages"] < self.cargo_demand_tons: violations.append("cargo demand not satisfied")
        if not FUEL_SCENARIOS[plan["fuel"]]["compatible"]: violations.append("fuel incompatible")
        # Speed multiplier is a transparent simplified proxy for increased resistance.
        energy = energy_for_voyages(self.distance_nm, self.efficiency_nm_per_kwh, plan["voyages"]) * (plan["speed_knots"] / 15) ** 2
        fuel = estimate_fuel(energy, plan["fuel"])
        if fuel.operational_co2_kg > self.emission_limit_kg: violations.append("emission limit exceeded")
        penalty = 1_000_000 * len(violations)
        weighted = (self.weights["cost"] * fuel.cost_usd / 10_000 + self.weights["emissions"] * fuel.operational_co2_kg / 10_000 + self.weights["energy"] * energy / 10_000 + self.weights["schedule"] * (15 / plan["speed_knots"]))
        return {"solution": plan, "objective": weighted + penalty, "cost_usd": fuel.cost_usd, "energy_kwh": energy,
                "fuel_quantity": fuel.quantity, "fuel_unit": fuel.unit, "operational_co2_kg": fuel.operational_co2_kg,
                "lifecycle_co2_kg": fuel.lifecycle_co2_kg, "feasible": not violations, "penalty": penalty, "violations": violations}
