"""Streamlit decision-support dashboard for the SIH26138 prototype."""
import pandas as pd
import plotly.express as px
import streamlit as st
from config import DEFAULT_EMISSION_LIMIT_KG, DEFAULT_WEIGHTS, FUEL_SCENARIOS
from models.efficiency_predictor import predict_efficiency
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
with st.sidebar:
    st.header("Voyage inputs")
    values = {"Ship_Type": select("Ship type", ["Container", "Bulk Carrier", "Tanker"]), "Engine_Type": select("Engine type", ["Diesel", "Dual Fuel", "Electric"]),
    "Speed_Over_Ground_knots": st.number_input("Speed (knots)", 1.0, 30.0, 15.0), "Distance_Traveled_nm": st.number_input("Distance (nm)", 1.0, 50000.0, 1000.0),
    "Cargo_Weight_tons": st.number_input("Cargo weight (tons)", 0.0, 100000.0, 12000.0), "Average_Load_Percentage": st.slider("Load (%)", 0, 100, 80),
    "Engine_Power_kW": st.number_input("Engine power (kW)", 1.0, 100000.0, 15000.0), "Draft_meters": st.number_input("Draft (m)", .1, 30.0, 9.0),
    "Weather_Condition": select("Weather", ["Calm", "Moderate", "Rough"]), "Seasonal_Impact_Score": st.slider("Seasonal impact", 0.0, 10.0, 3.0),
    "Route_Type": select("Route type", ["Coastal", "International", "Short Sea"]), "Maintenance_Status": select("Maintenance", ["Good", "Due", "Overdue"]),
    "Weekly_Voyage_Count": st.number_input("Weekly voyages", 1, 20, 2)}
    st.header("Optimization")
    fuel = select("Fuel scenario", list(FUEL_SCENARIOS)); demand = st.number_input("Cargo demand (tons)", 1.0, 200000.0, float(values["Cargo_Weight_tons"]))
    limit = st.number_input("Emission limit (kg CO₂)", 1.0, 1e8, DEFAULT_EMISSION_LIMIT_KG); fleet = st.slider("Maximum voyages", 1, 8, 3)
    weights = {key: st.slider(f"{key.title()} weight", 0.0, 1.0, value) for key, value in DEFAULT_WEIGHTS.items()}
    iterations = st.slider("QI iterations", 5, 200, 60); population = st.slider("QI population", 5, 100, 30)

st.info("Scientific honesty: efficiency is the observed ML target. Energy and fuel outputs are derived calculations; every fuel parameter is a clearly labelled DEMO ASSUMPTION.")
if st.button("Predict Efficiency", type="primary"):
    try:
        efficiency = predict_efficiency(values); st.session_state["efficiency"] = efficiency
    except (ValueError, FileNotFoundError) as error: st.error(str(error))
efficiency = st.session_state.get("efficiency")
if efficiency:
    energy = energy_per_voyage(values["Distance_Traveled_nm"], efficiency); estimate = estimate_fuel(energy, fuel)
    cols = st.columns(5)
    for col, label, value in zip(cols, ["Predicted efficiency", "Energy", "Fuel required", "Estimated cost", "Operational CO₂"], [f"{efficiency:.3f} nm/kWh", f"{energy:,.0f} kWh", f"{estimate.quantity:,.0f} {estimate.unit}", f"${estimate.cost_usd:,.0f}", f"{estimate.operational_co2_kg:,.0f} kg"]): col.metric(label, value)
    st.caption(f"Lifecycle CO₂: {estimate.lifecycle_co2_kg if estimate.lifecycle_co2_kg is not None else 'Not configured'} kg. {estimate.notes}")
    rows = []
    for name in FUEL_SCENARIOS:
        e = estimate_fuel(energy, name); rows.append({"Fuel": name, "Cost USD": e.cost_usd, "Operational CO₂ kg": e.operational_co2_kg, "Fuel quantity": e.quantity})
    st.plotly_chart(px.bar(pd.DataFrame(rows), x="Fuel", y=["Cost USD", "Operational CO₂ kg"], barmode="group", title="Fuel-scenario comparison (DEMO ASSUMPTIONS)"), use_container_width=True)

problem = FleetProblem(values["Distance_Traveled_nm"], efficiency or .15, demand, limit, fleet, weights=weights)
buttons = st.columns(3)
result = None
if buttons[0].button("Run Classical Optimization"): result = {"classical": optimize_classical(problem)}
if buttons[1].button("Run Quantum-Inspired Optimization"): result = {"quantum-inspired": optimize_quantum_inspired(problem, population, iterations)}
if buttons[2].button("Run Full Benchmark"): result = benchmark(problem, population, iterations)
if result:
    st.subheader("Optimization result")
    for name, r in result.items():
        st.markdown(f"#### {name.replace('_', ' ').title()}")
        st.json({k: v for k, v in r.items() if k != "convergence_history"})
        st.plotly_chart(px.line(y=r["convergence_history"], labels={"x":"Iteration / evaluation", "y":"Best objective"}, title=f"{name} convergence"), use_container_width=True)
    if len(result) == 2:
        table = pd.DataFrame([{ "Algorithm": k, "Objective": v["objective"], "Cost": v["cost_usd"], "CO₂": v["operational_co2_kg"], "Runtime s": v["runtime_seconds"]} for k,v in result.items()])
        st.dataframe(table, use_container_width=True); st.plotly_chart(px.scatter(table, x="Cost", y="CO₂", text="Algorithm", title="Cost–emissions trade-off"), use_container_width=True)
