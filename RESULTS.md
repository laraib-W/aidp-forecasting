# Store Sales Forecasting: Results

Kaggle *Store Sales: Time Series Forecasting* (Corporación Favorita, Ecuador): 1,782 daily series
(54 stores × 33 product families), 2013-01-01 to 2017-08-15.

**How models are scored.** Every model forecasts the same 3 back-to-back 16-day windows
(29 Jun–14 Jul, 15–30 Jul, 31 Jul–15 Aug 2017), seeing only data before each window.
Metric: RMSLE (the Kaggle metric; lower is better). The **mean** of the 3 windows is the headline number.

## Accuracy: all 1,782 series

| Model | 29 Jun–14 Jul | 15–30 Jul | 31 Jul–15 Aug | **Mean** | Where measured |
|---|---|---|---|---|---|
| **LightGBM** | 0.3837 | 0.3992 | 0.4241 | **0.4024** | Laptop + AIDP (identical) |
| ARIMA (AutoARIMA) | 0.4196 | 0.4266 | 0.4932 | 0.4465 | Laptop |
| Baseline (4-week weekday average) | 0.4245 | 0.4427 | 0.5311 | 0.4661 | Laptop + AIDP (identical) |
| Prophet | 0.4680 | 0.4650 | 0.4995 | 0.4775 | AIDP (Spark) |
| ARIMAX (ARIMA + promotions + holidays) | — | — | — | not run | — |

LightGBM is the most accurate in every window: 14% better than the baseline and 10% better than ARIMA
(fixed random seed, so reruns give the same numbers).

## Accuracy: 50-series sample

A fixed random sample of 50 series, used for quick checks. It holds more small, noisy series, so scores are higher.

| Model | Mean | Where measured |
|---|---|---|
| Prophet | 0.4274 (AIDP driver), 0.4275 (AIDP Spark), 0.4278 (laptop) | AIDP + laptop |
| ARIMAX | 0.4831 | Laptop |
| Baseline | 0.5293 | AIDP + laptop |
| ARIMA | 0.5356 | Laptop |

## AIDP: combined run, 50 series (compare_models.ipynb, 2026-10-06)

8 worker cores, 4-core driver. LightGBM trains on all 1,782 series and is scored on the 50; the others fit the 50.

| Model | Mean RMSLE | Laptop (same 50) | Time on AIDP |
|---|---|---|---|
| LightGBM | 0.4139 | — | 67 s (incl. 5 s pandas features) |
| LightGBM (Spark features) | 0.4139 | — | 111 s (incl. 50 s Spark features) |
| Prophet (Spark) | 0.4278 | 0.4278 | 50 s |
| ARIMAX (Spark) | 0.4832 | 0.4831 | 97 s |
| Baseline | 0.5293 | 0.5293 | 0 s |
| ARIMA (Spark) | 0.5363 | 0.5356 | 221 s (ran first; likely includes one-off compile time on the workers, unconfirmed) |

## Same results on AIDP and the laptop

| Check | Laptop | AIDP |
|---|---|---|
| Baseline, 50 series, mean | 0.5293 | 0.5293 (exact) |
| Prophet, 50 series, mean | 0.4278 | 0.4274 (driver), 0.4275 (Spark) |
| Prophet, all series, mean | 0.4776 | 0.4775 (Spark) |
| LightGBM, all series, mean | 0.4024 | 0.4024 (exact, every window) |
| LightGBM on Spark-built features, mean | 0.4024 | 0.4024 (exact, every window) |
| Spark features vs pandas features (734,184 rows) | max difference 7.3e-12, 0 empty cells differ, categories equal | same |

Prophet differs only in the 4th decimal: its solver varies slightly between machines.

## Speed

Prophet, 3 windows (one model per series per window):

| Where | Cores | 50 series | All 1,782 series |
|---|---|---|---|
| Laptop (Apple M2, 16 GB) | 8 | 5 s | 113 s |
| AIDP, driver only | 4 | 164 s | ~1.6 h (projected from 50 series, not measured) |
| AIDP, Spark (`applyInPandas`) | 8 worker cores | 51 s | **287 s** |

LightGBM, all series, 3 windows (one model per window, driver only):

| Where | Cores | Time |
|---|---|---|
| Laptop | 8 | 49 s |
| AIDP driver | 4 | 65 s |
| AIDP driver, on Spark-built features | 4 | 62 s |

Building the feature table: laptop pandas 4 s, laptop Spark (local mode) 18 s, AIDP Spark (8 worker cores) 64 s.

ARIMA, laptop, all series, 3 windows: 21 min (1,782 models per window).

On AIDP, Spark spreads the per-series fits across the workers. As the run was set up, AIDP Spark with 8 worker
cores is slower than the 8-core laptop. Worker count is the lever: see *Scaling* below.

## Scaling (to fill in)

Prophet, all 1,782 series, AIDP Spark, static worker count:

| Worker cores | Time |
|---|---|
| 8 | 287 s |
| | |
| | |

## AIDP findings (verified 2026-10-06)

- The default driver has **4 cores**: `os.cpu_count()` on the driver.
- Installing `requirements.txt` as a cluster library also installs **second copies of numpy, pyarrow and pandas**
  into `/aidp/libraries/python/amd/`, next to the cluster's own in `/opt/dataflow/python/...`. In a notebook session:
  - `import statsforecast` fails: `SchemaError: Cannot interpret 'dtype('S')' as a data type`;
  - `pyarrow.from_numpy_dtype(numpy.dtype(bytes))` fails the same way, but works in a fresh Python process;
  - `spark.createDataFrame(pandas_df)` with Arrow on fails: `AttributeError: 'numpy.ndarray' object has no attribute 'isna'`.
- `statsforecast` imports fine on the Spark **workers** (ARIMA/ARIMAX via `applyInPandas`), though not in the notebook session.
- Workarounds used: ARIMA on Spark workers, or as a separate process on the driver; Arrow off for pandas ↔ Spark.
- Spark workers can read CSV files under `/Workspace/...` (`spark.read.csv` on the workers).
- `groupBy().applyInPandas()` runs on the workers, and libraries installed as cluster libraries are available there.
- With Spark's default of 200 shuffle partitions, a 50-series run creates mostly empty tasks;
  `spark.sql.shuffle.partitions` = 4 × worker cores avoids that.

## Not yet run

- ARIMA and ARIMAX on all 1,782 series on AIDP (Spark); ARIMAX on all 1,782 series anywhere.
- The end-to-end flow: MongoDB → connector → AIDP table → forecast.
- The same notebooks on Databricks, for comparison.
