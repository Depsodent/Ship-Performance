"""Transparent energy calculations derived from predicted efficiency."""
def energy_per_voyage(distance_nm: float, efficiency_nm_per_kwh: float) -> float:
    """Return energy (kWh); both inputs must be positive."""
    if distance_nm <= 0 or efficiency_nm_per_kwh <= 0:
        raise ValueError("Distance and efficiency must both be greater than zero.")
    return distance_nm / efficiency_nm_per_kwh

def energy_for_voyages(distance_nm: float, efficiency_nm_per_kwh: float, voyages: int) -> float:
    if voyages <= 0:
        raise ValueError("Voyage count must be positive.")
    return energy_per_voyage(distance_nm, efficiency_nm_per_kwh) * voyages

def energy_per_nautical_mile(efficiency_nm_per_kwh: float) -> float:
    if efficiency_nm_per_kwh <= 0:
        raise ValueError("Efficiency must be greater than zero.")
    return 1 / efficiency_nm_per_kwh
