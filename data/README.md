# Dataset notes

Expected file: `Ship_Performance_Dataset.csv` (Kaggle dataset supplied by the project team).

Dataset source URL: `ADD_KAGGLE_URL_HERE`

The expected data has roughly 2,736 rows and includes vessel, route, weather, engine and operational features. This prototype uses the observed `Efficiency_nm_per_kWh` column as its ML target. It does **not** treat the data as containing measured fuel consumption.

Rows missing the target are omitted during training; feature imputation happens inside the training pipeline to prevent test-data leakage. The original CSV is never changed. Place the licensed Kaggle CSV in this directory before training.
