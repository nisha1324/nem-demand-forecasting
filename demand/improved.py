"""Part 3: what a calendar and smoothing add to the Part 2 baselines, with no weather.

All four forecasts keep the Part 2 contract: `fn(history, lead)` sees only the
days before the origin. The calendar may cover the target day, because public
holidays are published in advance.

weekday_rule    the rule Part 2 proposed: yesterday midweek, last week after the
                weekend changes shape, a holiday treated as a weekend day
HoltWinters     exponential smoothing of each half-hour with a weekly season
climatology_adj the Part 2 climatology, moved by how far it has been off lately
blend           half Holt-Winters, half climatology_adj
"""
import itertools

import numpy as np
import pandas as pd

from demand.backtest import LEADS, _take, climatology, climatology_steps
from demand.source import ROOT

CALENDAR = ROOT / "data" / "public_holidays.csv"
SATURDAY, SUNDAY = 5, 6
ALPHAS = (.05, .1, .15, .2, .3, .4, .5, .6, .7, .8)
BETAS = (0, .02)
GAMMAS = (.005, .01, .02, .05, .1)
PHI = .8


def holiday_mask(days: pd.DatetimeIndex, region: str, calendar: pd.DataFrame | None = None) -> np.ndarray:
    """True for each day that is a public holiday in the region."""
    calendar = pd.read_csv(CALENDAR) if calendar is None else calendar
    return days.isin(pd.to_datetime(calendar.loc[calendar["region"] == region, "date"]))


def weekday_rule(history: np.ndarray, lead: int, dow: np.ndarray, holiday: np.ndarray) -> np.ndarray:
    """The most recent day of the target's kind.

    Tue-Fri copy the last ordinary weekday, which is yesterday unless it was a holiday.
    Monday and Saturday copy the last ordinary day of that name, because yesterday has
    the wrong shape. Sundays and holidays copy the last weekend day or holiday.
    """
    n = len(history)
    t = n + lead - 1
    seen_dow, seen_holiday = dow[:n], holiday[:n]
    if holiday[t] or dow[t] == SUNDAY:
        same = seen_holiday | (seen_dow >= SATURDAY)
    elif dow[t] in (0, SATURDAY):
        same = (seen_dow == dow[t]) & ~seen_holiday
    else:
        same = (seen_dow < SATURDAY) & ~seen_holiday
    rows = np.flatnonzero(same)
    if not len(rows):
        raise IndexError(f"no similar day in {n} days of history")
    return _take(history, rows[-1])[0]


class HoltWinters:
    """Additive Holt-Winters with a damped trend, run on each half-hour of the day.

    For one half-hour, with y the demand on day t and d its day of the week:

        expected  = level + phi * trend
        level'    = alpha * (y - season[d]) + (1 - alpha) * expected
        trend'    = beta * (level' - level) + (1 - beta) * phi * trend
        season[d] = gamma * (y - expected) + (1 - gamma) * season[d]

    forecast at lead h = level + (phi + ... + phi^h) * trend + season[target weekday]

    A public holiday would pull the level towards a Sunday, so the state skips those
    days, and a target that is one reads the Sunday season instead.

    The state only moves forward through `history`, so an origin never sees later
    days. The object is a cache of one region's state: use one per region.
    """

    def __init__(self, dow: np.ndarray, holiday: np.ndarray, alpha: float, beta: float, gamma: float,
                 phi: float, warmup: int = 28):
        self.dow, self.holiday = dow, holiday
        self.alpha, self.beta, self.gamma, self.phi = alpha, beta, gamma, phi
        self.warmup = warmup
        self.n = 0

    def _start(self, history: np.ndarray) -> None:
        first = history[: self.warmup]
        self.level = first.mean(axis=0)
        self.trend = np.zeros_like(self.level)
        self.season = np.stack([first[self.dow[: self.warmup] == d].mean(axis=0) - self.level
                                for d in range(7)])
        self.n = self.warmup

    def _advance(self, history: np.ndarray) -> None:
        a, b, g, phi = self.alpha, self.beta, self.gamma, self.phi
        level, trend, season = self.level, self.trend, self.season
        for t in range(self.n, len(history)):
            if self.holiday[t]:
                continue
            d = self.dow[t]
            expected = level + phi * trend
            new_level = a * (history[t] - season[d]) + (1 - a) * expected
            trend = b * (new_level - level) + (1 - b) * phi * trend
            season[d] = g * (history[t] - expected) + (1 - g) * season[d]
            level = new_level
        self.level, self.trend = level, trend
        self.n = len(history)

    def __call__(self, history: np.ndarray, lead: int) -> np.ndarray:
        n = len(history)
        if n < self.warmup:
            raise IndexError(f"Holt-Winters needs {self.warmup} days of history, got {n}")
        if n < self.n or self.n == 0:
            self._start(history)
        self._advance(history)
        t = n + lead - 1
        d = SUNDAY if self.holiday[t] else self.dow[t]
        damped = sum(self.phi ** i for i in range(1, lead + 1))
        return self.level + damped * self.trend + self.season[d]


def nearest_sunday(t: int, dow: np.ndarray) -> int:
    """Row of the Sunday closest to day t."""
    back = (dow[t] + 1) % 7
    return t - back if back <= 3 else t + 7 - back


def climatology_adj(history: np.ndarray, lead: int, dow: np.ndarray, holiday: np.ndarray,
                    recent: int = 28) -> np.ndarray:
    """Climatology for the target, plus its mean miss per half-hour over the last `recent` days.

    Part 2 found climatology carries a level shift straight into the forecast (Victoria
    -5.1%, Tasmania +10.2% in the latest year). The miss is measured per half-hour, so
    a midday that keeps getting lower is corrected at midday. Holidays are left out of
    the miss, and a holiday target reads the climatology of the nearest Sunday.
    """
    n = len(history)
    t = n + lead - 1
    target = nearest_sunday(t, dow) if holiday[t] else t
    base = climatology(history, target - n + 1)
    days = np.arange(n - recent, n)
    days = days[~holiday[days]]
    rows = days[:, None] - climatology_steps()
    miss = _take(history, days) - _take(history, rows).mean(axis=1)
    return base + miss.mean(axis=0)


def blend(history: np.ndarray, lead: int, hw: HoltWinters, dow: np.ndarray, holiday: np.ndarray) -> np.ndarray:
    """Half what the last few days suggest, half what this time of year usually looks like."""
    return (hw(history, lead) + climatology_adj(history, lead, dow, holiday)) / 2


def tune(m: np.ndarray, dow: np.ndarray, holiday: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
    """Score every grid setting on `targets` (consecutive day rows): day-ahead and week-ahead WAPE.

    Reads nothing after the last target, so the days after it stay unseen.
    """
    first, last = targets[0], targets[-1]
    demand = np.abs(m[targets]).sum()
    rows = []
    for alpha, beta, gamma in itertools.product(ALPHAS, BETAS, GAMMAS):
        hw = HoltWinters(dow, holiday, alpha, beta, gamma, PHI)
        err = np.zeros(8)
        for origin in range(first - 6, last + 1):
            for lead in LEADS:
                t = origin + lead - 1
                if first <= t <= last:
                    err[lead] += np.abs(hw(m[:origin], lead) - m[t]).sum()
        rows.append((alpha, beta, gamma, err[1] / demand, err[1:].sum() / len(LEADS) / demand))
    return pd.DataFrame(rows, columns=["alpha", "beta", "gamma", "wape_day", "wape_week"])
