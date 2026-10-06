"""Prophet: one model per store × family series, fitted on the driver's cores or on Spark workers.

Settings: last 365 days, weekly pattern, national holidays, promotions as an extra input, log1p(sales).
3 years with a yearly pattern scored far worse (0.665): runs of zeros in 2014–2015 that look like missing data
were read as a yearly pattern.
"""
import logging

import numpy as np
import pandas as pd

from store_sales.data import KEYS, national_holidays


def fit_one(history, future, holidays):
    """Fit Prophet on one series' history (ds, y, onpromotion) and predict its future rows (log scale)."""
    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    if history.y.max() == 0:  # never sold: nothing to learn, predict 0
        return np.zeros(len(future))
    m = Prophet(holidays=holidays, weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False)
    m.add_regressor("onpromotion")
    m.fit(history)
    return m.predict(future)["yhat"].values


def make_forecasters(data_dir, spark=None, history_days=365):
    """Return {"Prophet": f} (driver, all cores), plus {"Prophet (Spark)": f} when a Spark session is given."""
    nat = national_holidays(data_dir)
    holidays = pd.DataFrame({"holiday": nat.description, "ds": nat.date})

    def recent(fit):
        return fit[fit.date > fit.date.max() - pd.Timedelta(days=history_days)]

    def on_driver(fit, valid):
        from joblib import Parallel, delayed

        hist = recent(fit).assign(ds=lambda d: d.date, y=lambda d: np.log1p(d.sales)).groupby(KEYS)
        jobs = [(rows.index, hist.get_group(key)[["ds", "y", "onpromotion"]],
                 rows.assign(ds=rows.date)[["ds", "onpromotion"]])
                for key, rows in valid.groupby(KEYS)]
        preds = Parallel(n_jobs=-1)(delayed(fit_one)(h, f, holidays) for _, h, f in jobs)
        out = pd.Series(np.nan, index=valid.index)
        for (idx, _, _), p in zip(jobs, preds):
            out[idx] = p
        assert out.notna().all(), "Prophet returned no forecast for some rows"
        return np.expm1(out.values)

    def on_spark(fit, valid):
        # Workers don't have the store_sales folder: send this module's code with each task instead of a reference
        import sys
        from pyspark import cloudpickle
        cloudpickle.register_pickle_by_value(sys.modules[__name__])

        both = pd.concat([
            recent(fit)[[*KEYS, "date", "sales", "onpromotion"]].assign(row=-1),  # history rows
            valid[[*KEYS, "date", "onpromotion"]].assign(sales=np.nan, row=valid.index),  # rows to predict
        ])
        # Dates travel as text so Spark's timezone handling can't shift them
        both = both.assign(date=both.date.dt.strftime("%Y-%m-%d"), onpromotion=both.onpromotion.astype(float),
                           row=both.row.astype("int64"))

        def fit_series(pdf):  # runs on a worker, gets one series as pandas
            pdf = pdf.assign(ds=pd.to_datetime(pdf.date)).sort_values("ds")
            h, f = pdf[pdf.row < 0], pdf[pdf.row >= 0]
            yhat = fit_one(h.assign(y=np.log1p(h.sales))[["ds", "y", "onpromotion"]], f[["ds", "onpromotion"]],
                           holidays)
            return pd.DataFrame({"row": f.row.values, "yhat": yhat})

        out = (spark.createDataFrame(both).groupBy(*KEYS)
               .applyInPandas(fit_series, schema="row long, yhat double").toPandas())
        pred = out.set_index("row").yhat.reindex(valid.index)
        assert pred.notna().all(), "Prophet returned no forecast for some rows"
        return np.expm1(pred.values)

    return {"Prophet": on_driver, **({"Prophet (Spark)": on_spark} if spark is not None else {})}
