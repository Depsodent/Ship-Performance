import pytest
from utils.fuel_model import estimate_fuel
def test_fuel_estimate():
    result = estimate_fuel(97, "Diesel"); assert result.quantity == 10 and result.operational_co2_kg > 0
def test_invalid_fuel():
    with pytest.raises(ValueError): estimate_fuel(10, "Unknown")
