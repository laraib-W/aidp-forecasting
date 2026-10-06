"""Data, backtest windows, score and baseline: shared by every model so results are comparable."""
import numpy as np
import pandas as pd

HORIZON = 16  # Kaggle test period = 16 days
KEYS = ["store_nbr", "family"]


def load_train(data_dir="data"):
    return pd.read_csv(f"{data_dir}/train.csv", parse_dates=["date"])


def national_holidays(data_dir="data"):
    """Dates of national holidays, skipping moved ones (`transferred`) and extra working days."""
    hol = pd.read_csv(f"{data_dir}/holidays_events.csv", parse_dates=["date"])
    return hol[(hol.locale == "National") & ~hol.transferred & (hol.type != "Work Day")]


WINDOWS = 3  # backtest on the last 3 back-to-back 16-day windows, not just one


def split(train, window=0):
    """Hide HORIZON days: window=0 is the last 16 days, window=1 the 16 before that, and so on.
    Models learn from `fit` (everything before) and are scored on `valid`."""
    end = train.date.max() - pd.Timedelta(days=HORIZON * window)
    cutoff = end - pd.Timedelta(days=HORIZON - 1)
    return train[train.date < cutoff], train[(train.date >= cutoff) & (train.date <= end)]


def backtest(train, predict, windows=WINDOWS):
    """Score `predict(fit, valid) -> predictions` on each window, oldest first, plus the mean."""
    scores = {}
    for w in reversed(range(windows)):
        fit, valid = split(train, w)
        label = f"{valid.date.min():%d %b} to {valid.date.max():%d %b %Y}"
        scores[label] = rmsle(valid.sales.values, predict(fit, valid))
    scores = pd.Series(scores)
    scores["mean"] = scores.mean()
    return scores


def rmsle(actual, pred):
    pred = np.clip(pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(actual)) ** 2)))


def baseline(fit, valid, weeks=4):
    """Predict each weekday as the average of the same weekday over the last `weeks` weeks."""
    window = fit[fit.date > fit.date.max() - pd.Timedelta(weeks=weeks)]
    avg = window.assign(dow=window.date.dt.dayofweek).groupby([*KEYS, "dow"]).sales.mean().rename("pred")
    return valid.assign(dow=valid.date.dt.dayofweek).join(avg, on=[*KEYS, "dow"]).pred.fillna(0).values


def sample_series(train, n=50, seed=0):
    """A fixed random set of store x family series, for models too slow to run on all 1,782."""
    series = train[KEYS].drop_duplicates()
    return series.sample(n, random_state=seed)


def only(df, series):
    """Rows of `df` belonging to `series`, keeping df's index (row numbers)."""
    return df[df.set_index(KEYS).index.isin(series.set_index(KEYS).index)]


if __name__ == "__main__":
    assert rmsle(np.array([5.0]), np.array([5.0])) == 0
    assert rmsle(np.array([1000.0]), np.array([900.0])) < 0.2 < rmsle(np.array([10.0]), np.array([1.0]))
    print("ok")
