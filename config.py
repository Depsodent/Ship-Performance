"""Central configuration for the Green Fleet Optimizer prototype."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "Ship_Performance_Dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "artifacts" / "efficiency_model.joblib"
MODEL_METADATA_PATH = BASE_DIR / "models" / "artifacts" / "model_metadata.json"
OUTPUT_DIR = BASE_DIR / "outputs"
RANDOM_SEED = 42

TARGET_COLUMN = "Efficiency_nm_per_kWh"
FEATURE_COLUMNS = [
    "Ship_Type", "Route_Type", "Engine_Type", "Maintenance_Status", "Weather_Condition",
    "Speed_Over_Ground_knots", "Engine_Power_kW", "Distance_Traveled_nm", "Draft_meters",
    "Cargo_Weight_tons", "Operational_Cost_USD", "Revenue_per_Voyage_USD",
    "Turnaround_Time_hours", "Seasonal_Impact_Score", "Weekly_Voyage_Count",
    "Average_Load_Percentage",
]
NUMERIC_FEATURES = [
    "Speed_Over_Ground_knots", "Engine_Power_kW", "Distance_Traveled_nm", "Draft_meters",
    "Cargo_Weight_tons", "Operational_Cost_USD", "Revenue_per_Voyage_USD",
    "Turnaround_Time_hours", "Seasonal_Impact_Score", "Weekly_Voyage_Count",
    "Average_Load_Percentage",
]
CATEGORICAL_FEATURES = ["Ship_Type", "Route_Type", "Engine_Type", "Maintenance_Status", "Weather_Condition"]
ENGINEERED_NUMERIC_FEATURES = [
    "Power_Speed_Product", "Power_Load_Product", "Speed_Squared",
    "Distance_Per_Turnaround_Hour", "Power_Per_Cargo_Ton",
]

DEFAULT_WEIGHTS = {"cost": 0.35, "emissions": 0.35, "energy": 0.20, "schedule": 0.10}
MIN_SPEED_KNOTS, MAX_SPEED_KNOTS = 5.0, 25.0
DEFAULT_EMISSION_LIMIT_KG = 200_000.0

# All entries are deliberately editable prototype assumptions, not verified fuel data.
FUEL_SCENARIOS = {
    "Diesel": {"unit": "L", "usable_energy_density_kwh": 9.7, "price_per_unit": 1.0,
               "operational_co2_kg_per_unit": 2.68, "lifecycle_co2_kg_per_unit": 3.2,
               "compatible": True, "notes": "DEMO ASSUMPTION — replace with verified marine factors."},
    "HFO": {"unit": "kg", "usable_energy_density_kwh": 11.0, "price_per_unit": 0.55,
            "operational_co2_kg_per_unit": 3.1, "lifecycle_co2_kg_per_unit": None,
            "compatible": True, "notes": "DEMO ASSUMPTION — replace with verified marine factors."},
    "LNG": {"unit": "kg", "usable_energy_density_kwh": 13.0, "price_per_unit": 0.9,
            "operational_co2_kg_per_unit": 2.75, "lifecycle_co2_kg_per_unit": None,
            "compatible": True, "notes": "DEMO ASSUMPTION — methane-slip effects not modelled."},
    "Methanol": {"unit": "kg", "usable_energy_density_kwh": 5.5, "price_per_unit": 0.75,
                 "operational_co2_kg_per_unit": 1.4, "lifecycle_co2_kg_per_unit": None,
                 "compatible": True, "notes": "DEMO ASSUMPTION — pathway dependent."},
    "Hydrogen": {"unit": "kg", "usable_energy_density_kwh": 20.0, "price_per_unit": 6.0,
                 "operational_co2_kg_per_unit": 0.0, "lifecycle_co2_kg_per_unit": None,
                 "compatible": False, "notes": "DEMO ASSUMPTION — only compatible retrofit vessels."},
    "Ammonia": {"unit": "kg", "usable_energy_density_kwh": 4.5, "price_per_unit": 1.2,
                "operational_co2_kg_per_unit": 0.0, "lifecycle_co2_kg_per_unit": None,
                "compatible": False, "notes": "DEMO ASSUMPTION — combustion risks/pathway not modelled."},
    "Shore Power": {"unit": "kWh", "usable_energy_density_kwh": 1.0, "price_per_unit": 0.18,
                    "operational_co2_kg_per_unit": 0.0, "lifecycle_co2_kg_per_unit": None,
                    "compatible": True, "notes": "DEMO ASSUMPTION — appropriate only in port."},
}
