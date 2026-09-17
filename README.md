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

- Leakage-safe sklearn preprocessing and an actual held-out comparison of Dummy, Random Forest, Extra Trees, and Gradient Boosting regressors (80/20 split; MAE, RMSE, R²).
- Validation, energy/fuel calculations, operational versus lifecycle CO₂, and editable scenarios.
- Weighted multi-objective problem with cargo, speed, emissions, fuel compatibility and voyage constraints.
- Exhaustive-grid classical baseline plus a probability-vector Quantum-Inspired Optimization Algorithm on classical CPU hardware. It is not quantum computing and is not claimed to be superior.
- Reproducible benchmark, convergence/trade-off charts, tests and Streamlit dashboard.

## Setup (Windows / VS Code)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy the project team's Kaggle CSV into `data/Ship_Performance_Dataset.csv`. Its URL was not provided, so `data/README.md` deliberately says `ADD_KAGGLE_URL_HERE`.

## Train, run and test

```powershell
python experiments/train_model.py
python -m streamlit run app.py
python -m pytest
python experiments/benchmark_algorithms.py
```

The complete fitted best-performing preprocessing + model pipeline is generated at
`models/artifacts/efficiency_model.joblib` and is Git-ignored. Detailed real-data
investigation, comparison results, selected-model metrics, features, and training
timestamp are saved at `models/artifacts/model_metadata.json`. The `Date` source
column is not passed raw to the model: it is intentionally excluded because the dashboard
does not collect a voyage date. The sidebar shows the saved model status and provides a
**Retrain ML Model** button.

## Current real-data baseline finding

The supplied features currently show weak held-out predictive signal for efficiency.
The application displays the measured R² honestly; an R² below zero means the selected
model performed worse than a mean-target baseline on that particular test split. MAE,
RMSE and R² are regression metrics, not classification “accuracy”. This does not change
the methodology: the ML output is efficiency, energy is derived from it, and fuel/cost/
emissions remain configurable scenario estimates rather than measured fuel consumption.

Latest reproducible comparison on the supplied 2,736-row CSV (80/20 split, `random_state=42`):

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| DummyRegressor (mean) — selected | 0.339630 | 0.396121 | -0.002639 |
| RandomForestRegressor | 0.339944 | 0.397574 | -0.010009 |
| ExtraTreesRegressor | 0.341757 | 0.398641 | -0.015438 |
| GradientBoostingRegressor | 0.342886 | 0.401190 | -0.028464 |

The small correlations between every available numeric feature and the target (all below
an absolute value of 0.02) support the current weak-signal finding. Better operational
telemetry or a revised target definition—not altered target values—is needed before a
complex model can add useful predictive value.

To regenerate the full descriptive diagnostic report (missing values, duplicates,
categorical levels, distributions, outlier counts, target correlations, engineered-feature
correlations, and Date analysis), run:

```powershell
python experiments/analyze_dataset.py
```

The report is saved as `outputs/dataset_diagnostic.json`. The physically motivated
target-free transformations are applied inside every candidate pipeline; they do not use
efficiency or derived energy. They include power × speed, power × load, speed², distance
per turnaround hour, and power per cargo ton.

## Limitations and future work

The dataset, fuel prices/factors and lifecycle methodology must be verified; the fleet simulation is simplified and does not model full engine, weather, port or fuel-pathway effects. Future work: AIS/telemetry, weather/port APIs, verified prices, lifecycle analysis, NSGA-II, and quantum-hardware experiments.

## SIH demo flow (3–5 minutes)

1. Explain the observed target is efficiency—not direct measured fuel use.
2. Enter a voyage and predict efficiency after training the supplied data.
3. Show derived energy and clearly marked fuel scenarios.
4. Run classical and quantum-inspired CPU searches.
5. Discuss convergence and cost–CO₂ trade-offs; choose the feasible plan rather than claiming a universal winner.
