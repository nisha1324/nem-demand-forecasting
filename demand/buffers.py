"""Part 4: turning leftover forecast error into MW of buffer, dollars and a battery timetable.

A miss is actual minus forecast, so a positive miss is an under-forecast. A buffer is MW held
on one side of the forecast. A day is covered if no half-hour in the stretch being protected
goes past the buffer, which is a tougher promise than covering most half-hours.

All functions take days x 48 matrices in the backtest's layout. Nothing here reads a file.
"""
import math

import numpy as np
import pandas as pd

UP, DOWN = 1, -1  # UP guards against an under-forecast, DOWN against an over-forecast
WHOLE_DAY = (0, 48)
BLOCKS = {"00:00-06:00": (0, 12), "06:00-10:00": (12, 20), "10:00-16:00": (20, 32),
          "16:00-21:00": (32, 42), "21:00-24:00": (42, 48)}
SEASONS = {"Summer": (12, 1, 2), "Autumn": (3, 4, 5), "Winter": (6, 7, 8), "Spring": (9, 10, 11)}
MONTH_SEASON = {month: name for name, months in SEASONS.items() for month in months}
WINDOW_SLOTS = 8  # a 4-hour battery window


def season_of(days: pd.DatetimeIndex) -> np.ndarray:
    return np.array([MONTH_SEASON[m] for m in days.month])


def rank_quantile(values: np.ndarray, level: float) -> float:
    """The value at rank ceil((n + 1) x level), or the largest if n is too small for that rank.

    One step more careful than a plain percentile, so a small sample doesn't promise more than
    it has seen: if past and future days are alike, a future day falls at or below this value
    with at least `level` chance.
    """
    ordered = np.sort(np.asarray(values, float).ravel())
    rank = math.ceil((len(ordered) + 1) * level)
    return float(ordered[min(rank, len(ordered)) - 1])


def worst_miss(miss: np.ndarray, direction: int, block: tuple = WHOLE_DAY) -> np.ndarray:
    """Each day's largest miss in this direction within the block, in MW."""
    return (direction * miss[:, block[0]: block[1]]).max(axis=1)


def covered(miss: np.ndarray, buffer: np.ndarray, direction: int, block: tuple = WHOLE_DAY) -> np.ndarray:
    """True for each day on which no half-hour in the block goes past its buffer."""
    return ((direction * miss) <= buffer)[:, block[0]: block[1]].all(axis=1)


def flat_buffer(miss: np.ndarray, direction: int, level: float) -> float:
    return rank_quantile(worst_miss(miss, direction), level)


def halfhour_buffer(miss: np.ndarray, direction: int, level: float) -> float:
    """The tempting shortcut: the percentile of all half-hours, not of each day's worst."""
    return rank_quantile(direction * miss, level)


def block_buffers(miss: np.ndarray, groups: np.ndarray, direction: int, level: float) -> dict:
    """{group: 48 MW values}, one buffer per time block, each its own promise for that block."""
    out = {}
    for group in np.unique(groups):
        rows = miss[groups == group]
        out[group] = np.empty(48)
        for block in BLOCKS.values():
            out[group][block[0]: block[1]] = rank_quantile(worst_miss(rows, direction, block), level)
    return out


def stretch(miss: np.ndarray, groups: np.ndarray, direction: int, level: float) -> dict:
    """Block buffers scaled by one factor, so that the whole day meets `level` on past days.

    Each block's buffer alone leaves five chances to miss a day. This keeps their shape and
    finds the smallest stretch that covers the day as a whole.
    """
    shape = block_buffers(miss, groups, direction, level)
    if min(v.min() for v in shape.values()) <= 0:
        raise ValueError("a block's buffer is not positive, so it can't be stretched")
    per_day = np.stack([shape[g] for g in groups])
    factor = rank_quantile((direction * miss / per_day).max(axis=1), level)
    return {g: factor * v for g, v in shape.items()}


def spread_out(buffers: dict, groups: np.ndarray) -> np.ndarray:
    """A {group: 48 values} table as a days x 48 matrix for these days."""
    return np.stack([buffers[g] for g in groups])


def imbalance_cost(miss: np.ndarray, price: np.ndarray, hedge_price: float, share: float = .01) -> np.ndarray:
    """Cost in $ per half-hour of having hedged the forecast rather than the actual load.

    A retailer with `share` of the region's load hedges its forecast F at a fixed price H and
    pays spot on what it actually uses, A. Its bill is A x P - (P - H) x F, and with a perfect
    forecast it would be H x A. The difference is (A - F) x (P - H), for half an hour.
    Under-forecasting into a high price costs money, and so does over-forecasting into a low one.
    """
    return share * miss * (price - hedge_price) * .5


def rolling_mean(signal: np.ndarray, length: int = WINDOW_SLOTS) -> np.ndarray:
    """days x starts: the mean of each run of `length` half-hours within the day."""
    totals = np.cumsum(np.column_stack([np.zeros(len(signal)), signal]), axis=1)
    return (totals[:, length:] - totals[:, :-length]) / length


def pick_windows(signal: np.ndarray, length: int = WINDOW_SLOTS) -> tuple[np.ndarray, np.ndarray]:
    """Start slots of the charge and discharge windows that a signal favours, one pair per day.

    Charging comes first and the windows can't overlap. The pair maximises the signal's
    discharge mean minus its charge mean, and the earliest pair wins a tie.
    """
    means = rolling_mean(signal, length)
    days = np.arange(len(means))
    best_later = np.maximum.accumulate(means[:, ::-1], axis=1)[:, ::-1]
    charge = (best_later[:, length:] - means[:, :-length]).argmax(axis=1)
    later = np.where(np.arange(means.shape[1]) >= (charge + length)[:, None], means, -np.inf)
    return charge, later[days].argmax(axis=1)


def captured_spread(price: np.ndarray, charge: np.ndarray, discharge: np.ndarray, length: int = WINDOW_SLOTS) -> np.ndarray:
    """$/MWh earned each day: mean price while discharging minus mean price while charging."""
    means = rolling_mean(price, length)
    days = np.arange(len(means))
    return means[days, discharge] - means[days, charge]
