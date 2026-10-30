"""The Part 3 methods on hand-made series, where every right answer is known."""
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from demand import backtest as bt
from demand import improved as im
from demand.backtest import SLOTS
from tests.test_backtest import DAYS, weekly_pattern

DOW = DAYS.dayofweek.to_numpy()
NO_HOLIDAYS = np.zeros(len(DAYS), dtype=bool)
INDEX = np.repeat(np.arange(len(DAYS), dtype=float)[:, None], SLOTS, axis=1)  # value = day index


def holidays_at(*rows) -> np.ndarray:
    h = NO_HOLIDAYS.copy()
    h[list(rows)] = True
    return h


def row_of(dow: int, after: int = 1300) -> int:
    return after + (dow - DOW[after]) % 7


def test_holiday_mask_is_per_region():
    days = pd.date_range("2026-03-08", "2026-03-10", freq="D")
    assert list(im.holiday_mask(days, "VIC1")) == [False, True, False]
    assert list(im.holiday_mask(days, "NSW1")) == [False, False, False]


# ------------------------------------------------------------ weekday rule

@pytest.mark.parametrize("dow", [1, 2, 3, 4])
def test_weekday_rule_copies_yesterday_midweek(dow):
    t = row_of(dow)  # includes Tuesday, whose yesterday is the Monday
    assert im.weekday_rule(INDEX[:t], 1, DOW, NO_HOLIDAYS)[0] == t - 1


@pytest.mark.parametrize("dow", [0, 5])
def test_weekday_rule_copies_last_week_for_monday_and_saturday(dow):
    t = row_of(dow)
    assert im.weekday_rule(INDEX[:t], 1, DOW, NO_HOLIDAYS)[0] == t - 7


def test_weekday_rule_lets_sunday_copy_saturday():
    sun = row_of(6)
    assert im.weekday_rule(INDEX[:sun], 1, DOW, NO_HOLIDAYS)[0] == sun - 1


def test_weekday_rule_treats_a_holiday_as_a_weekend_day():
    wed = row_of(2)
    assert im.weekday_rule(INDEX[:wed], 1, DOW, holidays_at(wed))[0] == wed - 3  # Sunday


def test_weekday_rule_never_copies_a_holiday_onto_an_ordinary_day():
    tue = row_of(1)
    monday_off = holidays_at(tue - 1)
    assert im.weekday_rule(INDEX[:tue], 1, DOW, monday_off)[0] == tue - 4  # the Friday before
    next_mon = tue + 6
    assert im.weekday_rule(INDEX[:next_mon], 1, DOW, monday_off)[0] == next_mon - 14


def test_weekday_rule_uses_the_last_day_seen_at_longer_leads():
    mon = row_of(0)
    # issued at midnight on Monday for Thursday: Friday is the last Tue-Fri seen
    assert im.weekday_rule(INDEX[:mon], 4, DOW, NO_HOLIDAYS)[0] == mon - 3


# ------------------------------------------------------------ Holt-Winters

def test_holt_winters_is_exact_on_a_series_that_repeats_every_week():
    m = weekly_pattern()
    hw = im.HoltWinters(DOW, NO_HOLIDAYS, .3, .02, .05, .8)
    for origin, lead in ((400, 1), (401, 3), (900, 7)):
        assert hw(m[:origin], lead) == pytest.approx(m[origin + lead - 1])


@pytest.mark.parametrize("alpha,beta,gamma,phi", [(.3, .02, .05, .8), (.7, 0, .01, .8), (.15, .1, .2, .9)])
def test_holt_winters_matches_statsmodels(alpha, beta, gamma, phi):
    """Same recursions, same fixed settings, same starting state: same state and leads 1-6.

    Lead 7 is left out on purpose. It reads the season of the last day seen, and
    statsmodels 0.15 overwrites that with the value from before the update
    (`s[nobs + m - 1:] = ...` in holtwinters/model.py). The textbook formula reads
    the updated value, as this code does; test_holt_winters_one_step_by_hand covers it.
    """
    y = np.random.default_rng(3).normal(1000, 80, size=(600, 2))
    dow = np.arange(len(DAYS)) % 7
    hw = im.HoltWinters(dow, NO_HOLIDAYS, alpha, beta, gamma, phi)
    hw(y, 1)
    for slot in range(2):
        hw_start = im.HoltWinters(dow, NO_HOLIDAYS, alpha, beta, gamma, phi)
        hw_start._start(y[:28])
        fit = ExponentialSmoothing(
            y[28:, slot], trend="add", damped_trend=True, seasonal="add", seasonal_periods=7,
            initialization_method="known", initial_level=hw_start.level[slot], initial_trend=0.0,
            initial_seasonal=hw_start.season[:, slot],
        ).fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma,
              damping_trend=phi, optimized=False)
        assert hw.level[slot] == pytest.approx(fit.level[-1], rel=1e-9)
        assert hw.trend[slot] == pytest.approx(fit.trend[-1], abs=1e-9)
        got = [hw(y, lead)[slot] for lead in range(1, 7)]
        assert got == pytest.approx(fit.forecast(6), rel=1e-9)


def test_holt_winters_one_step_by_hand():
    y = np.full((29, SLOTS), 100.0)
    dow = np.arange(len(DAYS)) % 7
    y[28, 0] = 140.0  # one surprise on day 28, day of the week 0
    hw = im.HoltWinters(dow, NO_HOLIDAYS, .5, .25, .1, .8)
    hw(y, 1)
    # start: level 100, trend 0, season 0; surprise of 40
    assert hw.level[0] == pytest.approx(120)          # .5 x 140 + .5 x 100
    assert hw.trend[0] == pytest.approx(5)            # .25 x (120 - 100)
    assert hw.season[0, 0] == pytest.approx(4)        # .1 x (140 - 100)
    # lead 1 is day 29, day of the week 1, season 0: 120 + .8 x 5
    assert hw(y, 1)[0] == pytest.approx(124)
    # lead 7 is day 35, day of the week 0, season 4
    assert hw(y, 7)[0] == pytest.approx(120 + sum(.8 ** i for i in range(1, 8)) * 5 + 4)


def test_holt_winters_skips_holidays_and_reads_the_sunday_season_for_one():
    m = weekly_pattern()
    wed = row_of(2, 800)
    spoiled = m.copy()
    spoiled[wed] = 0  # a holiday the state must not learn from
    plain = im.HoltWinters(DOW, holidays_at(wed), .5, 0, .1, .8)
    assert plain(spoiled[: wed + 1], 1) == pytest.approx(m[wed + 1])
    assert im.HoltWinters(DOW, holidays_at(wed), .5, 0, .1, .8)(m[:wed], 1) == pytest.approx(m[wed - 3])


def test_holt_winters_gives_the_same_answer_after_a_restart():
    m = np.random.default_rng(2).normal(1000, 80, size=(900, SLOTS))
    hw = im.HoltWinters(DOW, NO_HOLIDAYS, .4, .02, .05, .8)
    late, early = hw(m[:800], 2).copy(), hw(m[:700], 2).copy()  # going back forces a restart
    assert np.array_equal(early, im.HoltWinters(DOW, NO_HOLIDAYS, .4, .02, .05, .8)(m[:700], 2))
    assert np.array_equal(late, im.HoltWinters(DOW, NO_HOLIDAYS, .4, .02, .05, .8)(m[:800], 2))


def test_holt_winters_refuses_a_short_history():
    with pytest.raises(IndexError):
        im.HoltWinters(DOW, NO_HOLIDAYS, .4, 0, .05, .8)(np.ones((20, SLOTS)), 1)


# ------------------------------------------------------------ corrected climatology

def test_nearest_sunday():
    sun = row_of(6)
    assert [im.nearest_sunday(sun + k, DOW) for k in range(-3, 4)] == [sun] * 7


def test_climatology_adj_removes_a_level_shift_per_half_hour():
    m = weekly_pattern()
    shift = np.linspace(-50, 80, SLOTS)
    origin = 1500
    m[origin - 28:] += shift  # the last 4 weeks and the target sit above history
    got = im.climatology_adj(m[:origin], 2, DOW, NO_HOLIDAYS)
    assert got == pytest.approx(m[origin + 1])
    assert bt.climatology(m[:origin], 2) == pytest.approx(m[origin + 1] - shift)


def test_climatology_adj_leaves_holidays_out_of_the_miss():
    m = weekly_pattern()
    origin = row_of(2, 1500)
    spoiled = m.copy()
    spoiled[origin - 5] = 0
    got = im.climatology_adj(spoiled[:origin], 1, DOW, holidays_at(origin - 5))
    assert got == pytest.approx(m[origin])


def test_climatology_adj_reads_the_nearest_sunday_for_a_holiday():
    m = weekly_pattern()
    wed = row_of(2, 1500)
    assert im.climatology_adj(m[:wed], 1, DOW, holidays_at(wed)) == pytest.approx(m[wed - 3])


def test_blend_is_the_average_of_its_two_parts():
    m = np.random.default_rng(4).normal(1000, 80, size=(1400, SLOTS))
    hw = im.HoltWinters(DOW, NO_HOLIDAYS, .4, 0, .05, .8)
    got = im.blend(m[:1300], 3, hw, DOW, NO_HOLIDAYS)
    want = (hw(m[:1300], 3) + im.climatology_adj(m[:1300], 3, DOW, NO_HOLIDAYS)) / 2
    assert np.array_equal(got, want)


# ------------------------------------------------------------ harness rules

@pytest.mark.parametrize("key", ["rule", "hw", "clim_adj", "blend"])
def test_new_methods_never_see_anything_after_the_origin(key):
    rng = np.random.default_rng(1)
    m = rng.normal(1000, 50, size=(len(DAYS), SLOTS))
    origin = 1300
    holiday = holidays_at(origin + 1, 1290)

    def make():
        hw = im.HoltWinters(DOW, holiday, .4, .02, .05, .8)
        return {"rule": lambda h, L: im.weekday_rule(h, L, DOW, holiday), "hw": hw,
                "clim_adj": lambda h, L: im.climatology_adj(h, L, DOW, holiday),
                "blend": lambda h, L: im.blend(h, L, hw, DOW, holiday)}[key]

    for lead in bt.LEADS:
        before = make()(m[:origin], lead)
        spoiled = m.copy()
        spoiled[origin:] = 1e9
        assert np.array_equal(make()(spoiled[:origin], lead), before)


def test_forecasts_run_extra_methods_on_the_same_days():
    m = weekly_pattern()
    fc = bt.forecasts(DAYS, m, "2024-01-01", "2024-01-14", methods=("naive",),
                      extra={"x": lambda h, L: im.weekday_rule(h, L, DOW, NO_HOLIDAYS)})
    assert set(fc["method"]) == {"naive", "x"} and fc.groupby("method").size().nunique() == 1


def test_tune_scores_a_setting_as_the_backtest_would():
    m = np.random.default_rng(5).normal(1000, 80, size=(120, SLOTS))
    dow = np.arange(len(m)) % 7
    no_hol = np.zeros(len(m) + 7, dtype=bool)
    targets = np.arange(80, 100)
    days = pd.date_range("2020-01-01", periods=len(m), freq="D")
    grid = im.tune(m, np.r_[dow, dow[:7]], no_hol, targets)
    row = grid[(grid["alpha"] == .3) & (grid["beta"] == 0) & (grid["gamma"] == .05)].iloc[0]
    hw = im.HoltWinters(np.r_[dow, dow[:7]], no_hol, .3, 0, .05, im.PHI)
    fc = bt.forecasts(days, m, days[80], days[99], methods=(), extra={"hw": hw})
    assert row["wape_day"] == pytest.approx(bt.metrics(fc).loc[("hw", 1), "wape"])
    assert row["wape_week"] == pytest.approx(bt.metrics(fc, by=("method",)).loc["hw", "wape"])
