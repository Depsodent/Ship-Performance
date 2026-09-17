"""Fuel scenario conversion layer. Parameters are explicitly demo assumptions."""
from dataclasses import dataclass
from typing import Optional
from config import FUEL_SCENARIOS

@dataclass(frozen=True)
class FuelEstimate:
    fuel: str; unit: str; quantity: float; cost_usd: float
    operational_co2_kg: float; lifecycle_co2_kg: Optional[float]; notes: str

def estimate_fuel(energy_kwh: float, fuel_name: str) -> FuelEstimate:
    """Convert derived energy to fuel quantity, cost and separate CO2 measures."""
    if energy_kwh <= 0:
        raise ValueError("Energy requirement must be greater than zero.")
    if fuel_name not in FUEL_SCENARIOS:
        raise ValueError(f"Unknown fuel '{fuel_name}'. Choose a configured scenario.")
    p = FUEL_SCENARIOS[fuel_name]
    quantity = energy_kwh / p["usable_energy_density_kwh"]
    life = p["lifecycle_co2_kg_per_unit"]
    return FuelEstimate(fuel_name, p["unit"], quantity, quantity * p["price_per_unit"],
                        quantity * p["operational_co2_kg_per_unit"],
                        None if life is None else quantity * life, p["notes"])
