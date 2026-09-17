import pytest
from utils.calculations import energy_per_voyage, energy_for_voyages
def test_energy(): assert energy_per_voyage(100, .5) == 200
def test_invalid_energy():
    with pytest.raises(ValueError): energy_per_voyage(0, .5)
def test_multiple_voyages(): assert energy_for_voyages(100, .5, 2) == 400
