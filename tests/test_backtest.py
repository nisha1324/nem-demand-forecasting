"""The backtest harness on hand-made series, where every right answer is known."""
import numpy as np
import pandas as pd
import pytest

from demand import backtest as bt
from demand.backtest import METHODS, SLOTS


def to_series(days: pd.DatetimeIndex, m: np.ndarray) -> pd.DataFrame:
    start = days.repeat(SLOTS) + pd.to_timedelta(np.tile(np.arange(SLOTS) * 30, len(days)), unit="min")
    return pd.DataFrame({"interval_end": start + pd.Timedelta("30min"), "demand_mw": m.ravel()})


DAYS = pd.date_range("2020-01-01", periods=1700, freq="D")


def weekly_pattern() -> np.ndarray:
    """Each weekday has its own fixed shape; every week repeats exactly."""
    shape = 1000 + 100 * np.sin(np.arange(SLOTS) / SLOTS * 2 * np.pi)
    return np.stack([shape + 50 * d.dayofweek for d in DAYS])


def test_day_matrix_round_trips_and_uses_the_start_of_each_half_hour():
    m = np.arange(len(DAYS) * SLOTS, dtype=float).reshape(len(DAYS), SLOTS)
    days, got = bt.day_matrix(to_series(DAYS, m))
    assert (days == DAYS).all() and np.array_equal(got, m)
    # the half-hour ending 00:00 on 2 Jan started at 23:30 on 1 Jan: last slot of day 0
    s = to_series(DAYS, m)
    assert s.loc[s["interval_end"] == "2020-01-02 00:00", "demand_mw"].item() == m[0, -1]


def test_day_matrix_refuses_a_gap():
    s = to_series(DAYS, weekly_pattern()).drop(index=500)
    with pytest.raises(ValueError):
        bt.day_matrix(s)


def test_weekly_repeating_series_is_forecast_exactly_by_weekly_methods():
    fc = bt.forecasts(DAYS, weekly_pattern(), "2024-01-01", "2024-03-31")
    err = (fc["forecast"] - fc["actual"]).abs().groupby(fc["method"]).max()
    assert err["seasonal_naive"] == 0 and err["mean_4w"] == 0
    # 364 days is 52 whole weeks, so the climatology lands on the same weekday too
    assert err["climatology"] == pytest.approx(0, abs=1e-9)  # a mean of 21 equal floats
    # yesterday is a different weekday: the naive method misses by the weekday step
    assert err["naive"] == pytest.approx(300)  # worst on Mondays, forecast from Sunday: (0 - 6) x 50


def test_naive_error_on_a_trend_grows_with_lead():
    m = np.repeat(np.arange(len(DAYS), dtype=float)[:, None] * 10, SLOTS, axis=1)
    fc = bt.forecasts(DAYS, m, "2023-01-01", "2023-01-31", methods=("naive", "seasonal_naive"))
    mt = bt.metrics(fc)
    for lead in bt.LEADS:
        # demand rises 10 MW a day; the last seen day is `lead` days before the target
        assert mt.loc[("naive", lead), "bias"] == pytest.approx(-10 * lead)
        assert mt.loc[("seasonal_naive", lead), "bias"] == pytest.approx(-70)


def test_methods_never_see_the_target_day_or_anything_after_the_origin():
    rng = np.random.default_rng(1)
    m = rng.normal(1000, 50, size=(len(DAYS), SLOTS))
    origin = 1300
    for key, (_, fn) in METHODS.items():
        for lead in bt.LEADS:
            before = fn(m[:origin], lead)
            spoiled = m.copy()
            spoiled[origin:] = 1e9
            assert np.array_equal(fn(spoiled[:origin], lead), before), key


def test_methods_refuse_to_wrap_around_a_short_history():
    m = np.ones((300, SLOTS))
    with pytest.raises(IndexError):
        METHODS["climatology"][1](m, 1)  # needs 3 years of history
    with pytest.raises(IndexError):
        METHODS["seasonal_naive"][1](m, 8)  # lead 8 would need the origin day itself


def test_each_method_picks_the_days_it_says():
    m = np.repeat(np.arange(len(DAYS), dtype=float)[:, None], SLOTS, axis=1)  # value = day index
    o, lead = 1300, 3
    t = o + lead - 1
    pick = {k: fn(m[:o], lead)[0] for k, (_, fn) in METHODS.items()}
    assert pick["naive"] == o - 1
    assert pick["seasonal_naive"] == t - 7
    assert pick["mean_4w"] == np.mean([t - 7, t - 14, t - 21, t - 28])
    assert pick["climatology"] == np.mean([t - 364 * y + 7 * w for y in (1, 2, 3) for w in range(-3, 4)])


def test_forecasts_use_the_same_target_days_at_every_lead():
    fc = bt.forecasts(DAYS, weekly_pattern(), "2024-01-01", "2024-01-10")
    days = fc.groupby(["method", "lead"])["day"].agg(lambda d: tuple(sorted(set(d))))
    assert days.nunique() == 1 and len(days.iloc[0]) == 10


def test_metrics_hand_worked():
    # two days of 48 half-hours; errors chosen so every metric is easy to check by hand
    actual = np.array([[100.0] * 47 + [200.0], [50.0] * 48])
    forecast = np.array([[110.0] * 47 + [180.0], [50.0] * 47 + [10.0]])
    fc = pd.DataFrame({
        "method": "x", "lead": 1,
        "day": np.repeat(pd.to_datetime(["2024-01-01", "2024-01-02"]), SLOTS),
        "slot": np.tile(np.arange(SLOTS), 2),
        "actual": actual.ravel(), "forecast": forecast.ravel(),
    })
    r = bt.metrics(fc).loc[("x", 1)]
    abs_err = 47 * 10 + 20 + 40
    assert r["n"] == 96
    assert r["mae"] == pytest.approx(abs_err / 96)
    assert r["wape"] == pytest.approx(abs_err / (47 * 100 + 200 + 48 * 50))
    assert r["bias"] == pytest.approx((47 * 10 - 20 - 40) / 96)
    assert r["mape"] == pytest.approx((47 * 0.1 + 0.1 + 0.8) / 96)
    # day 1: peak 180 vs 200, trough 110 vs 100; day 2: peak 50 vs 50, trough 10 vs 50
    assert r["peak_mae"] == pytest.approx((20 + 0) / 2)
    assert r["trough_mae"] == pytest.approx((10 + 40) / 2)


def test_wape_survives_zero_crossing_demand_where_mape_explodes():
    actual = np.r_[np.full(47, 100.0), 0.01]
    fc = pd.DataFrame({"method": "x", "lead": 1, "day": pd.Timestamp("2024-01-01"),
                       "slot": np.arange(SLOTS), "actual": actual, "forecast": actual + 5})
    r = bt.metrics(fc).loc[("x", 1)]
    assert r["wape"] == pytest.approx(5 * 48 / (4700.01))
    assert r["mape"] > 10  # one near-zero half-hour dominates the mean of ratios
