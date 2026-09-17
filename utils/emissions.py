"""Small explicit emission helpers."""
def emissions_from_quantity(quantity: float, kg_co2_per_unit: float) -> float:
    if quantity < 0 or kg_co2_per_unit < 0:
        raise ValueError("Quantity and factor cannot be negative.")
    return quantity * kg_co2_per_unit
