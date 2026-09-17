# Dataset notes

Active file: `Ship_Performance_Dataset.csv` (Kaggle dataset supplied by the project team).

Dataset source URL: `ADD_KAGGLE_URL_HERE`

The supplied CSV has 2,736 rows and 18 columns, with vessel, route, weather, engine and operational features. This prototype uses the observed `Efficiency_nm_per_kWh` column as its ML target. It does **not** treat the data as containing measured fuel consumption.

Rows missing the target are omitted during training; numerical values are median-imputed and categorical values (including missing `Route_Type`) are most-frequent-imputed within the training pipeline to prevent test-data leakage. The original CSV is never changed. The `Date` column is excluded rather than passed as a raw string because the dashboard does not ask users for a voyage date.
