"""LightGBM: one model for all series, trained on a feature table built with pandas or with Spark.

Every sales feature looks at least HORIZON (16) days back, so it is known for every day being forecast.
"""
import os

import numpy as np
import pandas as pd

from store_sales.data import HORIZON, KEYS, WINDOWS, national_holidays

LAGS = [16, 21, 28, 35, 42]
MEANS = [7, 14, 28, 56]
CATEGORICAL = ["store_nbr", "family", "city", "type", "cluster"]
HISTORY_DAYS = 365  # each window trains on the 365 days before it


def add_features(train, data_dir):
    """Feature table for every row of `train`, same index as `train`. Built with pandas on the driver."""
    d = train.assign(row=train.index)

    # Add the missing days (Christmas) with 0 sales, so shifting by N rows = N days
    all_days = pd.date_range(d.date.min(), d.date.max())
    missing = all_days.difference(d.date.unique())
    if len(missing):
        extra = d[KEYS].drop_duplicates().merge(pd.DataFrame({"date": missing}), how="cross")
        d = pd.concat([d, extra.assign(sales=0.0, onpromotion=0, row=-1)])
    d = d.sort_values([*KEYS, "date"]).reset_index(drop=True)

    by_series = d.groupby(KEYS).sales
    for k in LAGS:
        d[f"lag_{k}"] = by_series.shift(k)
    d["dow_mean"] = d[["lag_21", "lag_28", "lag_35", "lag_42"]].mean(axis=1)
    past = by_series.shift(HORIZON)  # newest value we may use
    for w in MEANS:
        d[f"mean_{w}"] = past.groupby([d.store_nbr, d.family]).transform(lambda s: s.rolling(w, min_periods=1).mean())

    d["dow"] = d.date.dt.dayofweek
    d["day"] = d.date.dt.day
    d["month"] = d.date.dt.month
    d["payday"] = ((d.day == 15) | d.date.dt.is_month_end).astype(int)
    d["holiday"] = d.date.isin(national_holidays(data_dir).date).astype(int)
    d["oil"] = d.date.map(_daily_oil(data_dir, all_days))

    stores = pd.read_csv(f"{data_dir}/stores.csv")
    d = d.merge(stores[["store_nbr", "city", "type", "cluster"]], on="store_nbr", how="left")
    for c in CATEGORICAL:
        d[c] = d[c].astype("category")

    return d[d.row >= 0].set_index("row").sort_index().drop(columns=["id", "date", "sales"], errors="ignore")


def add_features_spark(spark, train, data_dir, like):
    """The same features as `add_features`, built with Spark from train.csv.

    Returns only the rows the backtest uses (the last WINDOWS × HORIZON + HISTORY_DAYS days), indexed by
    train.csv's `id`, which equals the pandas row number. `like` is a pandas feature table (any rows) giving the
    column order and categories. Spark reads train.csv itself, so the workers must be able to read `data_dir`.
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    tr = spark.read.csv(os.path.abspath(f"{data_dir}/train.csv"), header=True,
                        schema="id long, date date, store_nbr int, family string, sales double, onpromotion long")

    # Every series × every day (adds the missing Christmas days with 0 sales), so lag k = k days back
    days = tr.select(F.explode(F.sequence(F.min("date"), F.max("date"))).alias("date"))
    d = (tr.select(*KEYS).distinct().crossJoin(days)
         .join(tr, [*KEYS, "date"], "left")
         .fillna({"sales": 0.0, "onpromotion": 0}))

    series = Window.partitionBy(*KEYS).orderBy("date")
    for k in LAGS:
        d = d.withColumn(f"lag_{k}", F.lag("sales", k).over(series))
    week_lags = ["lag_21", "lag_28", "lag_35", "lag_42"]  # mean of the ones that exist, like pandas
    d = d.withColumn("dow_mean", sum(F.coalesce(F.col(c), F.lit(0.0)) for c in week_lags)
                     / sum(F.col(c).isNotNull().cast("int") for c in week_lags))
    for w in MEANS:  # average of the w days ending HORIZON days back
        d = d.withColumn(f"mean_{w}", F.avg("sales").over(series.rowsBetween(-(HORIZON + w - 1), -HORIZON)))

    d = (d.withColumn("dow", (F.dayofweek("date") + 5) % 7)  # Monday = 0, like pandas
          .withColumn("day", F.dayofmonth("date"))
          .withColumn("month", F.month("date"))
          .withColumn("payday", ((F.col("day") == 15) | (F.col("date") == F.last_day("date"))).cast("int"))
          .withColumn("holiday", F.col("date").isin([x.date() for x in national_holidays(data_dir).date]).cast("int")))

    # Small lookup tables: read on the driver, handed to Spark as plain rows (no pandas → Spark transfer)
    oil = _daily_oil(data_dir, pd.date_range(train.date.min(), train.date.max()))
    d = d.join(spark.createDataFrame([(x.date(), float(v)) for x, v in oil.items()], "date date, oil double"),
               "date", "left")
    stores = pd.read_csv(f"{data_dir}/stores.csv")[["store_nbr", "city", "type", "cluster"]]
    d = d.join(spark.createDataFrame([tuple(r) for r in stores.itertuples(index=False)],
                                     "store_nbr int, city string, type string, cluster int"), "store_nbr", "left")

    since = train.date.max() - pd.Timedelta(days=HORIZON * WINDOWS + HISTORY_DAYS)
    out = (d.where(F.col("id").isNotNull() & (F.col("date") > F.lit(since.date())))
            .select("id", *like.columns).toPandas().set_index("id").sort_index())
    out.index.name = None
    for c in CATEGORICAL:
        out[c] = out[c].astype(like[c].dtype)  # same categories as the pandas version
    return out


def make_forecaster(features, train):
    """Return f(fit, valid) -> predictions that trains LightGBM on `features` (indexed like `train`).

    It always trains on all series in `train` before the window, even when `valid` is a sample:
    learning across all series is LightGBM's strength, so a 50-series run only changes what is scored."""
    import lightgbm as lgb

    def forecast(fit, valid):
        fit = train[train.date < valid.date.min()]
        rows = fit.index[fit.date > fit.date.max() - pd.Timedelta(days=HISTORY_DAYS)]
        model = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=127, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
        model.fit(features.loc[rows], np.log1p(fit.loc[rows, "sales"]), categorical_feature=CATEGORICAL)
        forecast.model = model  # the last one trained, for the feature-importance chart

        pred = np.expm1(model.predict(features.loc[valid.index]))
        # Series with no sales in the 8 weeks before the window: predict 0
        recent = fit[fit.date > fit.date.max() - pd.Timedelta(weeks=8)].groupby(KEYS).sales.sum()
        dead = valid.join(recent.rename("recent"), on=KEYS).recent.fillna(0).eq(0).values
        pred[dead] = 0
        return np.clip(pred, 0, None)

    return forecast


def _daily_oil(data_dir, days):
    """Oil price for every day in `days`: weekend and holiday gaps filled with the last known price."""
    oil = pd.read_csv(f"{data_dir}/oil.csv", parse_dates=["date"]).set_index("date").dcoilwtico
    return oil.reindex(days).ffill().bfill()
