# Green Fleet Optimizer

An SIH26138 prototype for **Quantum-Inspired Fuel Consumption Prediction and Green Fleet Optimization**. It predicts vessel efficiency, derives voyage energy, compares editable fuel scenarios, and benchmarks classical and quantum-inspired optimization.

## Scientific honesty and methodology

The expected Kaggle `Ship_Performance_Dataset.csv` has `Efficiency_nm_per_kWh` as the observed ML target. It does **not** provide direct measured fuel consumption. The project separates observed data, ML prediction, derived energy (`distance / efficiency`), configurable fuel assumptions, and simulated fleet scenarios. Fuel values in `config.py` are visibly labelled `DEMO ASSUMPTION`; no metrics or performance are invented.

## Architecture

```text
Vessel / route inputs → ML efficiency prediction → derived energy
                                             ↓
                              fuel scenario quantity / cost / CO₂
                                             ↓
       classical exhaustive search  ↔  quantum-inspired probability search
                                             ↓
                 benchmark, trade-offs and Streamlit dashboard
```

## Implemented

- Leakage-safe sklearn preprocessing and Random Forest regression (80/20 split; MAE, RMSE, R²).
- Validation, energy/fuel calculations, operational versus lifecycle CO₂, and editable scenarios.
- Weighted multi-objective problem with cargo, speed, emissions, fuel compatibility and voyage constraints.
- Exhaustive-grid classical baseline plus a probability-vector Quantum-Inspired Optimization Algorithm on classical CPU hardware. It is not quantum computing and is not claimed to be superior.
- Reproducible benchmark, convergence/trade-off charts, tests and Streamlit dashboard.

## Setup (Windows / VS Code)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Copy the project team's Kaggle CSV into `data/Ship_Performance_Dataset.csv`. Its URL was not provided, so `data/README.md` deliberately says `ADD_KAGGLE_URL_HERE`.

## Train, run and test

```powershell
python experiments/train_model.py
streamlit run app.py
pytest
python experiments/benchmark_algorithms.py
```

The generated `models/efficiency_model.pkl` is Git-ignored. The dashboard states how to train it if missing.

## Limitations and future work

The dataset, fuel prices/factors and lifecycle methodology must be verified; the fleet simulation is simplified and does not model full engine, weather, port or fuel-pathway effects. Future work: AIS/telemetry, weather/port APIs, verified prices, lifecycle analysis, NSGA-II, and quantum-hardware experiments.

## SIH demo flow (3–5 minutes)

1. Explain the observed target is efficiency—not direct measured fuel use.
2. Enter a voyage and predict efficiency after training the supplied data.
3. Show derived energy and clearly marked fuel scenarios.
4. Run classical and quantum-inspired CPU searches.
5. Discuss convergence and cost–CO₂ trade-offs; choose the feasible plan rather than claiming a universal winner.
