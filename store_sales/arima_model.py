"""ARIMA and ARIMAX: one AutoARIMA model per store × family series (statsforecast).

On AIDP, `import statsforecast` fails inside a notebook session (duplicate numpy/pyarrow from the
requirements.txt install) but works in a fresh process. `run_in_subprocess` runs this file that way:

    python -m store_sales.arima_model --data <csv folder> [--sample 50] [--windows 3] --out scores.json
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from store_sales.data import HORIZON, KEYS, WINDOWS, backtest, load_train, national_holidays, only, sample_series


def make_forecasters(data_dir, spark=None):
    """Return {"ARIMA": f, "ARIMAX": f}, each f(fit, valid) -> predictions in `valid` row order.
    With a Spark session, also "ARIMA (Spark)" and "ARIMAX (Spark)": the same model, one Spark task per series."""
    national = national_holidays(data_dir).date

    def to_sf(d, exog=False):  # statsforecast expects unique_id, ds (date), y (value), plus any extra inputs
        out = d.assign(unique_id=d.store_nbr.astype(str) + "|" + d.family, ds=d.date, y=np.log1p(d.sales))
        if exog:
            out = out.assign(promo=np.log1p(d.onpromotion), holiday=d.date.isin(national).astype(float))
        return out

    def forecast(fit, valid, exog, history_days=365):
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA

        cols = ["unique_id", "ds", "y"] + (["promo", "holiday"] if exog else [])
        hist = to_sf(fit[fit.date > fit.date.max() - pd.Timedelta(days=history_days)], exog)[cols]
        future = to_sf(valid, exog)[[c for c in cols if c != "y"]] if exog else None
        fc = StatsForecast(models=[AutoARIMA(season_length=7)], freq="D", n_jobs=-1).forecast(
            df=hist, h=HORIZON, X_df=future)
        pred = to_sf(valid).merge(fc, on=["unique_id", "ds"], how="left").AutoARIMA
        assert pred.notna().all(), "ARIMA returned no forecast for some rows"
        return np.expm1(pred.values)

    def forecast_spark(fit, valid, exog, history_days=365):
        # Workers don't have the store_sales folder: send this module's code with each task instead of a reference
        from pyspark import cloudpickle
        cloudpickle.register_pickle_by_value(sys.modules[__name__])

        hist = fit[fit.date > fit.date.max() - pd.Timedelta(days=history_days)]
        both = pd.concat([hist.assign(row=-1), valid.assign(sales=np.nan, row=valid.index)])
        both = pd.DataFrame({  # dates travel as text so Spark's timezone handling can't shift them
            "store_nbr": both.store_nbr, "family": both.family, "date": both.date.dt.strftime("%Y-%m-%d"),
            "y": np.log1p(both.sales), "promo": np.log1p(both.onpromotion).astype(float),
            "holiday": both.date.isin(national).astype(float), "row": both.row.astype("int64")})

        def fit_series(pdf):  # runs on a worker, gets one series as pandas
            from statsforecast.models import AutoARIMA

            pdf = pdf.sort_values("date")
            h, f = pdf[pdf.row < 0], pdf[pdf.row >= 0]
            x = ["promo", "holiday"]
            fc = AutoARIMA(season_length=7).forecast(
                y=h.y.values, h=len(f),
                X=h[x].values if exog else None, X_future=f[x].values if exog else None)
            return pd.DataFrame({"row": f.row.values, "yhat": fc["mean"]})

        out = (spark.createDataFrame(both).groupBy(*KEYS)
               .applyInPandas(fit_series, schema="row long, yhat double").toPandas())
        pred = out.set_index("row").yhat.reindex(valid.index)
        assert pred.notna().all(), "ARIMA returned no forecast for some rows"
        return np.expm1(pred.values)

    forecasters = {"ARIMA": lambda fit, valid: forecast(fit, valid, exog=False),
                   "ARIMAX": lambda fit, valid: forecast(fit, valid, exog=True)}
    if spark is not None:
        forecasters["ARIMA (Spark)"] = lambda fit, valid: forecast_spark(fit, valid, exog=False)
        forecasters["ARIMAX (Spark)"] = lambda fit, valid: forecast_spark(fit, valid, exog=True)
    return forecasters


def run(data_dir, models=("ARIMA", "ARIMAX"), sample=0, windows=WINDOWS):
    """Backtest the chosen models. Returns {model: {"scores": {window: rmsle, "mean": ...}, "seconds": s}}."""
    train = load_train(data_dir)
    if sample:
        train = only(train, sample_series(train, n=sample))
    forecasters = make_forecasters(data_dir)
    out = {}
    for name in models:
        t = time.time()
        scores = backtest(train, forecasters[name], windows)
        out[name] = {"scores": scores.to_dict(), "seconds": round(time.time() - t, 1)}
        print(f"{name}: mean {scores['mean']:.4f}, {out[name]['seconds']:.0f}s", flush=True)
    return out


def run_in_subprocess(data_dir, models=("ARIMA", "ARIMAX"), sample=0, windows=WINDOWS):
    """Same as `run`, in a fresh Python process (the AIDP workaround). Prints its progress as it goes."""
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory() as tmp:
        out_file = os.path.join(tmp, "scores.json")
        cmd = [sys.executable, "-m", "store_sales.arima_model", "--data", os.path.abspath(data_dir),
               "--models", *models, "--sample", str(sample), "--windows", str(windows), "--out", out_file]
        p = subprocess.Popen(cmd, cwd=project_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:  # statsforecast's fit warnings are noise
            if "Warning" not in line and "model.fit()" not in line:
                print(line, end="")
        if p.wait() != 0:
            raise RuntimeError(f"ARIMA process failed (exit code {p.returncode}); see the output above")
        with open(out_file) as f:
            return json.load(f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="folder holding the Kaggle CSVs")
    ap.add_argument("--models", nargs="+", default=["ARIMA", "ARIMAX"], choices=["ARIMA", "ARIMAX"])
    ap.add_argument("--sample", type=int, default=0, help="score only this many fixed random series (0 = all)")
    ap.add_argument("--windows", type=int, default=WINDOWS, help="number of 16-day backtest windows")
    ap.add_argument("--out", help="write the scores as JSON to this file")
    args = ap.parse_args()

    result = run(args.data, args.models, args.sample, args.windows)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f)
    else:
        print(json.dumps(result, indent=2))
