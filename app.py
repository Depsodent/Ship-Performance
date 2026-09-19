"""Streamlit decision-support dashboard for the SIH26138 prototype."""
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from config import DATASET_PATH, DEFAULT_EMISSION_LIMIT_KG, DEFAULT_WEIGHTS, FUEL_SCENARIOS, MODEL_METADATA_PATH, MODEL_PATH
from models.efficiency_predictor import predict_efficiency
from models.preprocessing import load_and_validate_dataset
from models.train_efficiency_model import train_model
from optimization.benchmark import benchmark
from optimization.conventional_optimizer import optimize_classical
from optimization.problem import FleetProblem
from optimization.quantum_optimizer import optimize_quantum_inspired
from utils.calculations import energy_per_voyage
from utils.fuel_model import estimate_fuel

st.set_page_config(page_title="Green Fleet Optimizer", page_icon="🌊", layout="wide")
st.title("🌊 Green Fleet Optimizer")
st.caption("SIH26138 prototype · ML efficiency prediction + transparent fuel scenarios + classical and quantum-inspired optimization")

def select(label, values, index=0): return st.sidebar.selectbox(label, values, index=index)

@st.cache_data(show_spinner=False)
def dataset_categories() -> dict[str, list[str]]:
    """Use the real dataset's categories when it is available for valid UI mapping."""
    fallback = {
        "Ship_Type": ["Bulk Carrier", "Container Ship", "Fish Carrier", "Tanker"],
        "Engine_Type": ["Diesel", "Heavy Fuel Oil (HFO)", "Steam Turbine"],
        "Weather_Condition": ["Calm", "Moderate", "Rough"],
        "Route_Type": ["Coastal", "Long-haul", "Short-haul", "Transoceanic"],
        "Maintenance_Status": ["Critical", "Fair", "Good"],
    }
    try:
        frame = load_and_validate_dataset(DATASET_PATH)
        return {column: sorted(frame[column].dropna().astype(str).unique().tolist()) or choices
                for column, choices in fallback.items()}
    except (FileNotFoundError, ValueError, KeyError):
        return fallback


@st.cache_data(show_spinner=False)
def numeric_input_defaults() -> dict[str, float]:
    """Use visible defaults derived from real-data medians, never random placeholders."""
    fields = ["Operational_Cost_USD", "Revenue_per_Voyage_USD", "Turnaround_Time_hours"]
    try:
        frame = load_and_validate_dataset(DATASET_PATH)
        return {field: float(frame[field].median()) for field in fields}
    except (FileNotFoundError, ValueError, KeyError):
        return {"Operational_Cost_USD": 50_000.0, "Revenue_per_Voyage_USD": 75_000.0, "Turnaround_Time_hours": 48.0}


def model_metadata() -> dict | None:
    """Read the latest saved metadata without attempting any retraining."""
    if not MODEL_METADATA_PATH.exists():
        return None
    try:
        return json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


categories = dataset_categories()
defaults = numeric_input_defaults()
with st.sidebar:
    st.header("Voyage inputs")
    values = {"Ship_Type": select("Ship type", categories["Ship_Type"]), "Engine_Type": select("Engine type", categories["Engine_Type"]),
    "Speed_Over_Ground_knots": st.number_input("Speed (knots)", 1.0, 30.0, 15.0), "Distance_Traveled_nm": st.number_input("Distance (nm)", 1.0, 50000.0, 1000.0),
    "Cargo_Weight_tons": st.number_input("Cargo weight (tons)", 0.0, 100000.0, 12000.0), "Average_Load_Percentage": st.slider("Load (%)", 0, 100, 80),
    "Engine_Power_kW": st.number_input("Engine power (kW)", 1.0, 100000.0, 15000.0), "Draft_meters": st.number_input("Draft (m)", .1, 30.0, 9.0),
    "Operational_Cost_USD": st.number_input("Operational cost (USD)", 0.0, 1e9, defaults["Operational_Cost_USD"], help="Real-dataset median by default; adjust for this voyage."),
    "Revenue_per_Voyage_USD": st.number_input("Revenue per voyage (USD)", 0.0, 1e9, defaults["Revenue_per_Voyage_USD"], help="Real-dataset median by default; adjust for this voyage."),
    "Turnaround_Time_hours": st.number_input("Turnaround time (hours)", .1, 10000.0, defaults["Turnaround_Time_hours"], help="Real-dataset median by default; adjust for this voyage."),
    "Weather_Condition": select("Weather", categories["Weather_Condition"]), "Seasonal_Impact_Score": st.slider("Seasonal impact", 0.0, 10.0, 3.0),
    "Route_Type": select("Route type", categories["Route_Type"]), "Maintenance_Status": select("Maintenance", categories["Maintenance_Status"]),
    "Weekly_Voyage_Count": st.number_input("Weekly voyages", 1, 20, 2)}
    st.header("ML Fuel Model Inputs")
    auto_values = {
        "m (kg)": st.number_input("Vehicle mass (kg)", 500.0, 5000.0, 1500.0),
        "Mt": st.number_input("Test mass (kg)", 500.0, 5000.0, 1600.0),
        "Ft": st.selectbox(
            "Fuel type",
            ["PETROL", "DIESEL", "PETROL/ELECTRIC", "LPG", "E85", "DIESEL/ELECTRIC", "NG", "HYDROGEN"],
        ),
        "Fm": st.selectbox("Fuel model", ["M", "H", "P", "B", "F"]),
        "ec (cm3)": st.number_input("Engine capacity (cm³)", 100.0, 10000.0, 1500.0),
        "ep (KW)": st.number_input("Engine power (kW)", 1.0, 1000.0, 100.0),
    }

    st.header("Optimization")
    fuel = select("Fuel scenario", list(FUEL_SCENARIOS)); demand = st.number_input("Cargo demand (tons)", 1.0, 200000.0, float(values["Cargo_Weight_tons"]))
    limit = st.number_input("Emission limit (kg CO₂)", 1.0, 1e8, DEFAULT_EMISSION_LIMIT_KG); fleet = st.slider("Maximum voyages", 1, 8, 3)
    weights = {key: st.slider(f"{key.title()} weight", 0.0, 1.0, value) for key, value in DEFAULT_WEIGHTS.items()}
    iterations = st.slider("QI iterations", 5, 200, 60); population = st.slider("QI population", 5, 100, 30)

    st.divider()
    st.subheader("ML Model Status")
    metadata = model_metadata()
    st.caption("Dataset: Ship_Performance_Dataset.csv · Target: Efficiency_nm_per_kWh")
    if metadata:
        st.caption(f"Rows: {metadata['rows']} · Model: {metadata['selected_model']}")
        st.caption(f"MAE: {metadata['MAE']:.4f} · RMSE: {metadata['RMSE']:.4f} · R²: {metadata['R2']:.4f}")
        if metadata["R2"] < 0:
            st.warning("Current dataset/model combination has weak predictive signal; R² below 0 means it performs worse than the mean-target baseline on this test split.")
    else:
        st.caption("No trained model metadata found yet.")

st.info("Scientific honesty: efficiency is the observed ML target. Energy and fuel outputs are derived calculations; every fuel parameter is a clearly labelled DEMO ASSUMPTION.")
if not MODEL_PATH.exists():
    st.warning("Model not trained yet. Train it below or run `python experiments/train_model.py` from the project root.")
# Model retraining is kept as a local development operation.

if st.button("Predict Fuel Consumption", type="primary"):
    try:
        fuel_consumption = predict_efficiency(auto_values)
        st.session_state["fuel_consumption"] = fuel_consumption
    except (ValueError, FileNotFoundError) as error:
        st.error(str(error))
fuel_consumption = st.session_state.get("fuel_consumption")

if fuel_consumption:
    st.subheader("Auto-MPG Prediction Result")
    st.metric("Predicted Fuel Consumption", f"{fuel_consumption:.3f}")
    st.caption("Prediction generated using the Auto-MPG dataset and trained Random Forest model.")

problem = (
    FleetProblem(
        values["Distance_Traveled_nm"],
        fuel_consumption,
        demand,
        limit,
        fleet,
        weights=weights,
    )
    if fuel_consumption is not None
    else None
)
buttons = st.columns(3)
result = None
if buttons[0].button("Run Classical Optimization"):
    if problem is None: st.warning("Predict fuel consumption first so optimization uses the trained model output.")
    else: result = {"classical": optimize_classical(problem)}
if buttons[1].button("Run Quantum-Inspired Optimization"):
    if problem is None: st.warning("Predict fuel consumption first so optimization uses the trained model output.")
    else: result = {"quantum-inspired": optimize_quantum_inspired(problem, population, iterations)}
if buttons[2].button("Run Full Benchmark"):
    if problem is None: st.warning("Predict fuel consumption first so optimization uses the trained model output.")
    else: result = benchmark(problem, population, iterations)
if result:
    st.subheader("Optimization result")
    for name, r in result.items():
        st.markdown(f"#### {name.replace('_', ' ').title()}")
        st.json({k: v for k, v in r.items() if k != "convergence_history"})
        st.plotly_chart(px.line(y=r["convergence_history"], labels={"x":"Iteration / evaluation", "y":"Best objective"}, title=f"{name} convergence"), use_container_width=True)
    if len(result) == 2:
        table = pd.DataFrame([{ "Algorithm": k, "Objective": v["objective"], "Cost": v["cost_usd"], "CO₂": v["operational_co2_kg"], "Runtime s": v["runtime_seconds"]} for k,v in result.items()])
        st.dataframe(table, use_container_width=True)
        st.plotly_chart(px.bar(table, x="Algorithm", y="Runtime s", text_auto=".4f", title="Benchmark Runtime Comparison"), use_container_width=True)