"""Rolling-origin backtest of simple, explainable demand forecasts.

The series is held as a day matrix: one row per day, 48 half-hour columns,
where a half-hour belongs to the day it STARTS in (as in the profile). The
snapshot is complete, so row i is always day FIRST_DAY + i.

A forecast is issued at an origin: midnight at the start of day o, knowing
every half-hour before it (rows 0 .. o-1) and nothing after. A lead-L forecast
targets day t = o + L - 1, so lead 1 is "tomorrow, issued at midnight" and
lead 7 is the last day of a week-ahead forecast.

Each method gets only `history = days[:o]`, a slice that ends at the origin,
so it cannot read the day it is forecasting even by mistake: an index past the
origin raises IndexError instead of leaking the answer.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

SLOTS = 48
LEADS = range(1, 8)


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    explain: str


def _take(history: np.ndarray, rows) -> np.ndarray:
    """history[rows], refusing negative indices (numpy would wrap them to the end)
    and anything at or past the origin."""
    rows = np.atleast_1d(rows)
    if rows.min() < 0 or rows.max() >= len(history):
        raise IndexError(f"rows {rows.min()}..{rows.max()} outside history of {len(history)} days")
    return history[rows]


def naive(history: np.ndarray, lead: int) -> np.ndarray:
    """Persistence: every future day looks like the last full day seen."""
    return _take(history, len(history) - 1)[0]


def seasonal_naive(history: np.ndarray, lead: int) -> np.ndarray:
    """The same weekday one week earlier (always seen for leads 1-7)."""
    return _take(history, len(history) + lead - 1 - 7)[0]


def same_weekday_mean(history: np.ndarray, lead: int, weeks: int = 4) -> np.ndarray:
    """Average of the same weekday over the last `weeks` weeks."""
    t = len(history) + lead - 1
    return _take(history, [t - 7 * k for k in range(1, weeks + 1)]).mean(axis=0)


def climatology(history: np.ndarray, lead: int, years: int = 3, window: int = 3) -> np.ndarray:
    """Long-run average: the same weekday within +/- `window` weeks of this
    date in each of the previous `years` years (52-week steps keep the weekday)."""
    t = len(history) + lead - 1
    rows = [t - 364 * y + 7 * w for y in range(1, years + 1) for w in range(-window, window + 1)]
    return _take(history, rows).mean(axis=0)


METHODS = {
    "naive": (Method("naive", "Naive (yesterday)",
                     "the last full day seen, repeated"), naive),
    "seasonal_naive": (Method("seasonal_naive", "Seasonal naive (last week)",
                              "the same weekday one week earlier"), seasonal_naive),
    "mean_4w": (Method("mean_4w", "4-week same-weekday mean",
                       "the average of the same weekday over the last 4 weeks"), same_weekday_mean),
    "climatology": (Method("climatology", "3-year climatology",
                           "the same weekday within ±3 weeks of this date in each of the "
                           "previous 3 years (21 days averaged)"), climatology),
}
REFERENCE = "seasonal_naive"  # the baseline every later method has to beat


def day_matrix(series: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """One region's half-hours (interval_end, demand_mw) -> (days, days x 48 matrix).

    Fails rather than guessing if any half-hour is missing or duplicated.
    """
    start = pd.to_datetime(series["interval_end"]) - pd.Timedelta("30min")
    day = start.dt.normalize()
    slot = (start.dt.hour * 60 + start.dt.minute) // 30
    days = pd.date_range(day.min(), day.max(), freq="D")
    if len(series) != len(days) * SLOTS or start.duplicated().any():
        raise ValueError(f"expected {len(days) * SLOTS} unique half-hours, got {len(series)}")
    m = np.full((len(days), SLOTS), np.nan)
    m[(day - days[0]).dt.days.to_numpy(), slot.to_numpy()] = series["demand_mw"].to_numpy()
    return days, m


def forecasts(days: pd.DatetimeIndex, m: np.ndarray, first: str, last: str,
              methods=tuple(METHODS), leads=LEADS) -> pd.DataFrame:
    """Forecast every target day in [first, last] at every lead with every method.

    Returns one row per (method, lead, target day, slot) with actual and forecast.
    The target days are the same for every lead, so leads compare like with like.
    """
    targets = np.flatnonzero((days >= first) & (days <= last))
    if not len(targets):
        raise ValueError("no target days in range")
    frames = []
    for key in methods:
        fn = METHODS[key][1]
        for lead in leads:
            pred = np.stack([fn(m[: t - lead + 1], lead) for t in targets])
            frames.append(pd.DataFrame({
                "method": key,
                "lead": lead,
                "day": np.repeat(days[targets], SLOTS),
                "slot": np.tile(np.arange(SLOTS), len(targets)),
                "actual": m[targets].ravel(),
                "forecast": pred.ravel(),
            }))
    return pd.concat(frames, ignore_index=True)


def metrics(fc: pd.DataFrame, by=("method", "lead")) -> pd.DataFrame:
    """Accuracy per group.

    mae         mean |forecast - actual|, MW
    wape        sum |error| / sum |actual|: a percentage that stays meaningful
                when demand nears or crosses zero (South Australia at midday)
    mape        mean of |error| / |actual|: reported only to show why it misleads
    bias        mean (forecast - actual), MW; positive = over-forecast
    peak_mae    mean over days of |forecast daily max - actual daily max|, MW
    trough_mae  mean over days of |forecast daily min - actual daily min|, MW
    """
    by = list(by)
    e = fc.assign(err=fc["forecast"] - fc["actual"])
    e["abs_err"] = e["err"].abs()
    e["abs_act"] = e["actual"].abs()
    e["ape"] = e["abs_err"] / e["abs_act"]
    out = e.groupby(by).agg(mae=("abs_err", "mean"), sum_abs_err=("abs_err", "sum"),
                            sum_abs_act=("abs_act", "sum"), mape=("ape", "mean"),
                            bias=("err", "mean"), n=("err", "size"))
    out["wape"] = out.pop("sum_abs_err") / out.pop("sum_abs_act")
    daily = e.groupby(by + ["day"]).agg(f_max=("forecast", "max"), a_max=("actual", "max"),
                                        f_min=("forecast", "min"), a_min=("actual", "min"))
    daily["peak_err"] = (daily["f_max"] - daily["a_max"]).abs()
    daily["trough_err"] = (daily["f_min"] - daily["a_min"]).abs()
    extremes = daily.groupby(by).agg(peak_mae=("peak_err", "mean"), trough_mae=("trough_err", "mean"))
    return out.join(extremes)[["n", "mae", "wape", "mape", "bias", "peak_mae", "trough_mae"]]
