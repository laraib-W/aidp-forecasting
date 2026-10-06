# Store Sales Forecasting on Oracle AIDP

Daily sales forecasting for the Kaggle
[Store Sales: Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
dataset (Corporación Favorita, Ecuador): 1,782 series (54 stores × 33 product families), forecast 16 days ahead.
It compares classic and machine-learning models, and runs them on Oracle AI Data Platform (AIDP) Workbench,
including Spark versions that spread the work across the cluster's workers.

Results so far: **[RESULTS.md](RESULTS.md)**. LightGBM is the most accurate (RMSLE 0.4024 over 3 windows, all series),
with identical scores on a laptop and on AIDP.

## Models

| Model | What it is | Runs on |
|---|---|---|
| Baseline | Average of the same weekday over the last 4 weeks | driver |
| LightGBM | One model for all series, on lag, calendar, holiday, promotion and store features | driver |
| LightGBM (Spark features) | Same model, features built with Spark | features: workers; model: driver |
| Prophet / Prophet (Spark) | One model per series, with holidays and promotions | driver's cores / Spark workers |
| ARIMA / ARIMAX (+ Spark) | One AutoARIMA per series; ARIMAX adds promotions and holidays | driver's cores / Spark workers |

Every model is scored the same way: 3 back-to-back 16-day backtest windows, RMSLE (the Kaggle metric; lower is better).

## Layout

```
store_sales/                 the code
  data.py                    loading, backtest windows, RMSLE, baseline, holidays
  spark_setup.py             Spark session settings (AIDP's session, or a local one)
  lgbm_model.py              features (pandas or Spark) + LightGBM
  prophet_model.py           Prophet, on the driver or on Spark
  arima_model.py             ARIMA / ARIMAX, on the driver, in its own process, or on Spark
compare_models.ipynb         pick models and settings, get one results table, chart and CSV
01_explore_and_baseline.ipynb  data exploration: trend, weekly pattern, decomposition, ACF, baselines
requirements.txt             libraries to install on the AIDP cluster
RESULTS.md                   results and AIDP findings
```

## Data

Not included. Download the CSVs from the Kaggle competition page (you need to accept the competition rules)
and put them in `data/`: `train.csv`, `holidays_events.csv`, `oil.csv`, `stores.csv` (plus `test.csv`,
`transactions.csv`, unused so far). `data/` is in `.gitignore`.

## Run on a laptop

Python 3.11. Spark versions also need Java 17.

```
uv venv -p 3.11 .venv
uv pip install -p .venv -r requirements.txt pandas scikit-learn joblib pyspark==3.5.0 jupyter
.venv/bin/jupyter lab
```

Open `compare_models.ipynb`, keep `PROJECT_DIR = "."`, and run all cells.

## Run on AIDP

1. **Upload** to one workspace folder: `store_sales/`, `compare_models.ipynb`, `01_explore_and_baseline.ipynb`,
   and the CSVs into `data/`.
2. **Install the libraries**: cluster → **Library** → **Install Library** → upload `requirements.txt`
   (the file must be named exactly that) → **Install**, then **Actions → Restart** the cluster.
3. **In `compare_models.ipynb`**, set `PROJECT_DIR` to the workspace folder, choose `SAMPLE`
   (50 for a quick run, 0 for all 1,782 series) and `MODELS`, and run all cells.

The notebook uses AIDP's own Spark session. Each run's table is saved under `results/`.

### AIDP notes (verified 2026-10-06)

- Installing `requirements.txt` also installs second copies of numpy, pyarrow and pandas, which clash with the
  cluster's own inside a notebook session. Effects and workarounds:
  - `import statsforecast` fails in the notebook. ARIMA runs on the Spark workers or in its own process instead,
    where it works.
  - pandas ↔ Spark transfer with Arrow fails, so Arrow is switched off (`store_sales/spark_setup.py`).
- The default driver had 4 cores. Per-series models (Prophet, ARIMA) are much faster on Spark than on that driver.
- Spark workers can read files under `/Workspace/...`.

## Disclaimer

Independent Arbisoft project. Not affiliated with or endorsed by Oracle or Kaggle.
