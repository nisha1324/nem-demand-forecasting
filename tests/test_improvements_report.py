"""The committed Part 3 report is current, tuning can't see the holdout, and the headline
numbers survive a recount that uses neither the harness nor demand/improved.py."""
import numpy as np
import pandas as pd
import pytest

from demand import backtest as bt
from demand.source import REGIONS, SNAPSHOT
from scripts import backtest as part2
from scripts import improve


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    return out, improve.write_outputs(out)


@pytest.fixture(scope="module")
def snap() -> pd.DataFrame:
    df = pd.read_csv(SNAPSHOT, usecols=["region", "interval_end", "demand_mw"], parse_dates=["interval_end"])
    return df.sort_values(["region", "interval_end"], ignore_index=True)


def days_by_slot(snap: pd.DataFrame, region: str) -> tuple[pd.DatetimeIndex, list]:
    """Plain Python lists: one list of 48 values per day, in order."""
    g = snap[snap["region"] == region]
    start = g["interval_end"] - pd.Timedelta("30min")
    days = pd.DatetimeIndex(start.dt.normalize().drop_duplicates())
    values = g["demand_mw"].tolist()
    return days, [values[i * 48: (i + 1) * 48] for i in range(len(days))]


def holidays_of(region: str) -> set:
    cal = pd.read_csv(improve.im.CALENDAR)
    return set(pd.to_datetime(cal.loc[cal["region"] == region, "date"]))


def recount_holt_winters(days, y, holidays, alpha, gamma) -> dict:
    """Lead-1 forecasts for holdout days, one half-hour at a time, no trend (beta = 0 in the settings)."""
    first, last = (pd.Timestamp(d) for d in improve.WINDOWS[improve.HOLDOUT])
    out = {}
    for slot in range(48):
        level = sum(y[i][slot] for i in range(28)) / 28
        season = [sum(y[i][slot] for i in range(28) if days[i].dayofweek == d) / 4 - level for d in range(7)]
        for i in range(28, len(days)):
            if first <= days[i] <= last:
                d = 6 if days[i] in holidays else days[i].dayofweek
                out[(i, slot)] = level + season[d]
            if days[i] in holidays:
                continue
            d = days[i].dayofweek
            before = level
            level = alpha * (y[i][slot] - season[d]) + (1 - alpha) * before
            season[d] = gamma * (y[i][slot] - before) + (1 - gamma) * season[d]
    return out


def recount_corrected_climatology(days, y, holidays) -> dict:
    first, last = (pd.Timestamp(d) for d in improve.WINDOWS[improve.HOLDOUT])
    steps = [364 * a - 7 * b for a in (1, 2, 3) for b in range(-3, 4)]
    out = {}
    for i, day in enumerate(days):
        if not first <= day <= last:
            continue
        target = i
        if day in holidays:
            back = (day.dayofweek + 1) % 7
            target = i - back if back <= 3 else i + 7 - back
        recent = [k for k in range(i - 28, i) if days[k] not in holidays]
        for slot in range(48):
            base = sum(y[target - s][slot] for s in steps) / 21
            miss = sum(y[k][slot] - sum(y[k - s][slot] for s in steps) / 21 for k in recent) / len(recent)
            out[(i, slot)] = base + miss
    return out


def wape(y, forecast: dict) -> float:
    return (sum(abs(f - y[i][s]) for (i, s), f in forecast.items())
            / sum(abs(y[i][s]) for (i, s) in forecast))


@pytest.mark.parametrize("name", [improve.REPORT.name, improve.METRICS_CSV.name, improve.SETTINGS_CSV.name])
def test_committed_tables_match_a_fresh_run(fresh, name):
    out, _ = fresh
    assert (out / name).read_text() == (improve.RESULTS / name).read_text(), \
        f"results/{name} is stale: run python -m scripts.improve"


@pytest.mark.parametrize("name", improve.CHART_FILES)
def test_committed_charts_match_a_fresh_run(fresh, name):
    out, _ = fresh
    assert (out / "charts" / name).read_bytes() == (improve.CHARTS / name).read_bytes(), \
        f"results/charts/{name} is stale: run python -m scripts.improve"


def test_tuning_never_reads_the_holdout(snap, monkeypatch):
    region = "TAS1"
    monkeypatch.setattr(improve, "REGIONS", (region,))
    days, m = bt.day_matrix(snap[snap["region"] == region])
    spoiled = m.copy()
    spoiled[days >= improve.WINDOWS[improve.HOLDOUT][0]] = 1e9
    got = improve.tune_all({region: (days, spoiled)})
    committed = pd.read_csv(improve.SETTINGS_CSV).set_index(["region", "horizon"]).loc[[region]]
    pd.testing.assert_frame_equal(got, committed, check_exact=False, rtol=1e-5)


def test_blend_recount_without_any_project_code(fresh, snap):
    _, facts = fresh
    region = "VIC1"
    days, y = days_by_slot(snap, region)
    hol = holidays_of(region)
    s = facts["settings"].loc[(region, "day")]
    assert s["beta"] == 0
    hw = recount_holt_winters(days, y, hol, s["alpha"], s["gamma"])
    clim = recount_corrected_climatology(days, y, hol)
    blended = {k: (hw[k] + clim[k]) / 2 for k in hw}
    row = facts["metrics"].xs(("latest", region, 1), level=("window", "region", "lead"))["wape"]
    assert wape(y, hw) == pytest.approx(row["hw_day"])
    assert wape(y, clim) == pytest.approx(row["climatology_adj"])
    assert wape(y, blended) == pytest.approx(row["blend"])


def test_weekday_rule_equals_naive_except_on_monday_saturday_and_next_to_holidays(fresh):
    _, facts = fresh
    d = facts["daily"]
    d = d[(d["window"] == "latest") & (d["lead"] == 1) & (d["region"] == "NSW1")]
    wide = d.pivot(index="day", columns="method", values="abs_err")
    hol = improve.holiday_days("NSW1")
    ordinary = wide.index[wide.index.dayofweek.isin([1, 2, 3, 4, 6])
                          & ~wide.index.isin(hol) & ~(wide.index - pd.Timedelta("1D")).isin(hol)]
    assert (wide.loc[ordinary, "weekday_rule"] == wide.loc[ordinary, "naive"]).all()
    assert (wide["weekday_rule"] != wide["naive"]).sum() > 80  # the Mondays, Saturdays and holidays


def test_baselines_match_the_part_2_metrics_file():
    this = pd.read_csv(improve.METRICS_CSV)
    that = pd.read_csv(part2.METRICS_CSV)
    this = this[this["method"].isin(bt.METHODS)].reset_index(drop=True)
    keys = ["window", "region", "method", "lead"]
    pd.testing.assert_frame_equal(this.sort_values(keys, ignore_index=True), that.sort_values(keys, ignore_index=True))


def test_interval_on_a_fixed_ratio_is_that_ratio():
    days = pd.date_range("2025-10-01", periods=70, freq="D")
    err = np.random.default_rng(1).uniform(10, 100, size=70)
    rows = [("a", err * .9), ("b", err)]
    daily = pd.concat([pd.DataFrame({"method": k, "lead": 1, "day": days, "abs_err": e, "abs_act": 1.0,
                                     "window": "latest", "region": "X"}) for k, e in rows])
    cut, low, high = improve.cut_with_interval({"daily": daily}, "X", "a", "b", (1,))
    assert (cut, low, high) == pytest.approx((.1, .1, .1))


def test_interval_keeps_a_week_together():
    # all the gain sits in one week: resampling weeks must be able to lose it entirely
    days = pd.date_range("2025-10-01", periods=70, freq="D")
    a, b = np.full(70, 100.0), np.full(70, 100.0)
    a[:7] = 0
    daily = pd.concat([pd.DataFrame({"method": k, "lead": 1, "day": days, "abs_err": e, "abs_act": 1.0,
                                     "window": "latest", "region": "X"}) for k, e in (("a", a), ("b", b))])
    cut, low, high = improve.cut_with_interval({"daily": daily}, "X", "a", "b", (1,))
    assert cut == pytest.approx(.1) and low == pytest.approx(0) and high > .2


def test_headline_claims_follow_from_the_metrics(fresh):
    _, facts = fresh
    assert facts["day_wins"] == list(REGIONS)
    assert all(facts["score_day"][r]["cut"][1] > 0 for r in REGIONS), "the report says every interval is clear of zero"
    for r in facts["week_unclear"]:
        low, high = facts["score_week"][r]["cut"][1:]
        assert low <= 0 <= high, "the report says these intervals span zero"
    assert set(facts["rule_lost"]) | set(facts["rule_won"]) == set(REGIONS)
    for r in REGIONS:
        assert abs(facts["bias_after"][r]) <= abs(facts["bias_before"][r])
