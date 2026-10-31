"""Archived weather forecasts for each region's capital, and how they line up with demand.

Source: Open-Meteo Previous Runs API (https://open-meteo.com/en/docs/previous-runs-api),
free with no key for non-commercial use, data under CC BY 4.0. Its `_previous_dayN`
columns hold what the model predicted at least N x 24 hours before each valid time, which
is what a planner really had. The better-known Historical Forecast API is not used: it
stitches together the first hours of each run, so it is close to what actually happened
and would flatter any "forecast" score.

Model: ECMWF IFS 0.25, picked on dates before the holdout. BOM ACCESS-G has a gap from
Jul 2025, and ECMWF was closer than GFS to ERA5 reanalysis day-ahead at Adelaide, on
temperature and radiation (GFS was closer on radiation at 7 days).

Hourly rows are in NEM time (Etc/GMT-10 is UTC+10). Temperature is the value at the
hour. Radiation is the mean over the hour ending at the stamp.
"""
import numpy as np
import pandas as pd

from demand.source import REGIONS, ROOT

WEATHER = ROOT / "data" / "weather_forecasts.csv.gz"
WEATHER_META = ROOT / "data" / "WEATHER.json"

API = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "ecmwf_ifs025"
TIMEZONE = "Etc/GMT-10"
FIRST_DAY, LAST_DAY = "2024-04-01", "2026-09-30"  # ECMWF radiation is complete from April 2024
FORECAST_LEADS = range(1, 8)
SITES = {
    "NSW1": ("Sydney", -33.87, 151.21),
    "QLD1": ("Brisbane", -27.47, 153.03),
    "VIC1": ("Melbourne", -37.81, 144.96),
    "SA1": ("Adelaide", -34.93, 138.60),
    "TAS1": ("Hobart", -42.88, 147.33),
}
LEADS_STORED = range(0, 8)  # lead 0 is the latest run for that hour, the nearest thing to what happened
COLUMNS = [f"{var}_{lead}" for var in ("temp", "solar") for lead in LEADS_STORED]


def api_variable(short: str, lead: int) -> str:
    name = {"temp": "temperature_2m", "solar": "shortwave_radiation"}[short]
    return name if lead == 0 else f"{name}_previous_day{lead}"


def halfhourly(hourly: pd.DataFrame, days: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """One region's hourly rows -> {"temp_0": days x 48 matrix, ...} on the demand slots.

    A slot is the half-hour starting at slot / 2 hours. Temperature is interpolated
    between the hours, and the last half-hour of the data holds the last value.
    Radiation is an hourly mean, so both half-hours of an hour get that hour's value.
    """
    stamps = pd.date_range(days[0], days[-1] + pd.Timedelta(hours=23), freq="h")
    if len(hourly) != len(stamps) or (pd.DatetimeIndex(hourly["time"]) != stamps).any():
        raise ValueError(f"expected {len(stamps)} consecutive hours from {stamps[0]}, got {len(hourly)}")
    out = {}
    for name in COLUMNS:
        values = hourly[name].to_numpy(float)
        if name.startswith("temp"):
            half = np.empty(2 * len(values))
            half[::2] = values
            half[1::2] = (values + np.append(values[1:], values[-1])) / 2
        else:
            # the stamp HH:00 covers the hour before it; the first hour of the data is unknown
            half = np.repeat(np.append(values[1:], values[-1]), 2)
        out[name] = half.reshape(len(days), 48)
    return out


def load_weather(path=WEATHER) -> tuple[pd.DatetimeIndex, dict]:
    """(days, {region: {"temp_1": days x 48, ...}}) from the committed snapshot."""
    df = pd.read_csv(path, parse_dates=["time"])
    days = pd.date_range(FIRST_DAY, LAST_DAY, freq="D")
    return days, {r: halfhourly(df[df["region"] == r], days) for r in REGIONS}


# ---------------------------------------------------------------- the adjustment

RECENT = 7  # days before the origin that stand for "the weather the blend already knows"
MIN_DAYS = 120  # no fit on fewer days than this: the adjustment stays at zero
HEATS = (14, 16, 18)  # degrees C: heating starts below this
COOLS = (18, 20, 22, 24)  # degrees C: cooling starts above this
LAMBDAS = (30, 100, 300)  # shrinkage, in days of evidence added at zero effect


def strict_lead(lead: int) -> int:
    """Weather column for a forecast `lead` days out, one day older than the name suggests.

    `_previous_dayN` is about N x 24 hours before each hour, so for the evening of the target
    day it can come from a run that was started before the midnight origin but published
    hours after it. One day older is always published by then. Open-Meteo stops at 7.
    """
    return min(lead + 1, max(FORECAST_LEADS))


def shutdown_mask(days: pd.DatetimeIndex) -> np.ndarray:
    """27-31 December: not public holidays, but many workplaces are shut."""
    return (days.month == 12) & (days.day >= 27)


def recent_mean(a: np.ndarray, lead: int, n: int = RECENT) -> np.ndarray:
    """For each target row: the mean of `a` over the `n` days before its origin (NaN if too early)."""
    means = np.lib.stride_tricks.sliding_window_view(a, n, axis=0).mean(axis=-1)  # row i covers days i .. i + n - 1
    out = np.full(a.shape, np.nan)
    origins = np.arange(len(a)) - lead + 1
    ok = origins >= n
    out[ok] = means[origins[ok] - n]
    return out


def features(w: dict, lead: int, heat: float, cool: float, column: int | None = None) -> np.ndarray:
    """days x 48 x 3: how much hotter, colder and sunnier the target looks than the last week.

    `column` is the stored lead the target's weather is read from (0 = what happened). The
    last week always reads column 1, which is older than the origin for every day it covers.
    """
    column = strict_lead(lead) if column is None else column
    cooling = lambda t: np.maximum(t - cool, 0)
    heating = lambda t: np.maximum(heat - t, 0)
    temp, solar, seen_temp, seen_solar = w[f"temp_{column}"], w[f"solar_{column}"], w["temp_1"], w["solar_1"]
    return np.stack([cooling(temp) - recent_mean(cooling(seen_temp), lead),
                     heating(temp) - recent_mean(heating(seen_temp), lead),
                     solar - recent_mean(seen_solar, lead)], axis=2)


def fit_slots(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Ridge regression of y (days x 48) on x (days x 48 x k), one fit per half-hour.

    Returns (slopes 48 x k in MW per unit, intercepts 48). Features are put on a common
    scale first, so `lam` means the same for all of them. A column with no spread (the
    sun at night) gets a slope of zero.
    """
    centre = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1
    z = (x - centre) / scale
    gram = np.einsum("nsk,nsl->skl", z, z) + lam * np.eye(x.shape[2])
    target = y - y.mean(axis=0)
    slopes = np.linalg.solve(gram, np.einsum("nsk,ns->sk", z, target)[..., None])[..., 0] / scale
    return slopes, y.mean(axis=0) - (centre * slopes).sum(axis=1)


def adjustment(residual: np.ndarray, x: np.ndarray, days: pd.DatetimeIndex, lead: int, lam: float,
               skip: np.ndarray, wanted: np.ndarray | None = None) -> np.ndarray:
    """MW to add to a forecast, for each target day at this lead.

    The fit is redone at the start of each month and sees only targets dated before that
    month began. The month is the ORIGIN's, so a lead-7 forecast for the 3rd, issued on
    the 27th, uses the fit from the start of the earlier month. Days in `skip` stay out of the
    fit, because their misses are about the calendar rather than the weather.
    """
    wanted = np.ones(len(days), bool) if wanted is None else wanted
    month = (days - pd.Timedelta(f"{lead - 1}D")).to_period("M")
    usable = ~np.isnan(x).any(axis=(1, 2)) & ~np.isnan(residual).any(axis=1) & ~skip
    out = np.zeros(residual.shape)
    for period in month[wanted].unique():
        train = np.flatnonzero(usable & (days < period.to_timestamp()))
        rows = np.flatnonzero(wanted & (month == period))
        if len(train) < MIN_DAYS:
            continue
        slopes, intercepts = fit_slots(x[train], residual[train], lam)
        out[rows] = np.einsum("nsk,sk->ns", x[rows], slopes) + intercepts
    return out
