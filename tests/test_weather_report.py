"""The committed Part 3b report is current, tuning can't see the holdout, the blend is Part 3's,
and the headline survives a recount that doesn't use demand/weather.py's fit."""
import functools

import numpy as np
import pandas as pd
import pytest

from demand import backtest as bt
from demand import improved as im
from demand import weather as wx
from demand.source import REGIONS, SNAPSHOT
from scripts import improve as part3
from scripts import weather as report


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    return out, report.write_outputs(out)


@pytest.fixture(scope="module")
def snap() -> pd.DataFrame:
    return pd.read_csv(SNAPSHOT, usecols=["region", "interval_end", "demand_mw"])


@pytest.mark.parametrize("name", [report.REPORT.name, report.METRICS_CSV.name, report.WEATHER_SETTINGS_CSV.name])
def test_committed_tables_match_a_fresh_run(fresh, name):
    out, _ = fresh
    assert (out / name).read_text() == (report.RESULTS / name).read_text(), \
        f"results/{name} is stale: run python -m scripts.weather"


@pytest.mark.parametrize("name", report.CHART_FILES)
def test_committed_charts_match_a_fresh_run(fresh, name):
    out, _ = fresh
    assert (out / "charts" / name).read_bytes() == (report.CHARTS / name).read_bytes(), \
        f"results/charts/{name} is stale: run python -m scripts.weather"


def test_the_blend_here_is_the_part_3_blend():
    this = pd.read_csv(report.METRICS_CSV)
    that = pd.read_csv(part3.METRICS_CSV)
    keys = ["window", "region", "method", "lead"]
    this = this[this["method"] == "blend"].sort_values(keys, ignore_index=True)
    that = that[that["method"] == "blend"].sort_values(keys, ignore_index=True)
    pd.testing.assert_frame_equal(this, that)


def test_tuning_never_reads_the_holdout(snap, monkeypatch):
    region = "TAS1"
    monkeypatch.setattr(report, "REGIONS", (region,))
    days, m = bt.day_matrix(snap[snap["region"] == region])
    wx_days, raw = wx.load_weather()
    w = report.align(raw[region], wx_days, days)
    hold = days >= report.WINDOWS[report.HOLDOUT][0]
    m = m.copy()
    m[hold] = 1e9
    for values in w.values():
        values[hold] = 1e9
    hw = pd.read_csv(report.SETTINGS_CSV).set_index(["region", "horizon"])
    _, _, got = report.choose({region: (days, m)}, {region: w}, hw)
    committed = pd.read_csv(report.WEATHER_SETTINGS_CSV).set_index("region").loc[[region]]
    pd.testing.assert_frame_equal(got, committed, check_exact=False, rtol=1e-5, check_dtype=False)


def recount_wape(days, m, base, region_weather, setting, skip) -> float:
    """VIC1's day-ahead holdout WAPE: loops and plain solves, no features() or fit_slots()."""
    temp_target, sun_target = region_weather["temp_2"], region_weather["solar_2"]
    temp_seen, sun_seen = region_weather["temp_1"], region_weather["solar_1"]
    heat, cool, lam = setting["heat"], setting["cool"], setting["lam"]
    start = np.flatnonzero(days == pd.Timestamp(wx.FIRST_DAY))[0]

    def row(t):
        seen = range(t - 7, t)
        out = np.zeros((48, 3))
        for s in range(48):
            cooling = max(temp_target[t, s] - cool, 0) - sum(max(temp_seen[k, s] - cool, 0) for k in seen) / 7
            heating = max(heat - temp_target[t, s], 0) - sum(max(heat - temp_seen[k, s], 0) for k in seen) / 7
            out[s] = cooling, heating, sun_target[t, s] - sum(sun_seen[k, s] for k in seen) / 7
        return out

    rows = {t: row(t) for t in range(start + 7, len(days))}
    err = act = 0.0
    hold = np.flatnonzero(days >= report.WINDOWS[report.HOLDOUT][0])
    for period in days[hold].to_period("M").unique():
        begin = period.to_timestamp()
        train = [t for t in rows if days[t] < begin and not skip[t]]
        X = np.stack([rows[t] for t in train])
        for s in range(48):
            a, y = X[:, s], (m - base)[train, s]
            mean, sd = a.mean(axis=0), a.std(axis=0)
            sd[sd == 0] = 1
            z = (a - mean) / sd
            beta = np.linalg.solve(z.T @ z + lam * np.eye(3), z.T @ (y - y.mean())) / sd
            for t in hold[days[hold].to_period("M") == period]:
                f = base[t, s] + y.mean() + (rows[t][s] - mean) @ beta
                err += abs(f - m[t, s])
                act += abs(m[t, s])
    return err / act


def test_weather_recount_without_the_project_fit(fresh, snap):
    _, facts = fresh
    region = "VIC1"
    days, m = bt.day_matrix(snap[snap["region"] == region])
    wx_days, raw = wx.load_weather()
    w = report.align(raw[region], wx_days, days)
    hw = pd.read_csv(report.SETTINGS_CSV).set_index(["region", "horizon"])
    dow, hol = days.dayofweek.to_numpy(), im.holiday_mask(days, region)
    s = hw.loc[(region, "day")]
    fn = functools.partial(im.blend, hw=im.HoltWinters(dow, hol, s["alpha"], s["beta"], s["gamma"], im.PHI),
                           dow=dow, holiday=hol)
    base = np.full(m.shape, np.nan)
    rows = np.flatnonzero(days >= wx.FIRST_DAY)
    base[rows] = np.stack([fn(m[: t], 1) for t in rows])
    assert not facts["use_shutdown"]
    got = recount_wape(days, m, base, w, facts["settings"].loc[region], hol | wx.shutdown_mask(days))
    assert got == pytest.approx(facts["metrics"].loc[("latest", region, "weather", 1), "wape"], rel=1e-9)


def test_the_adjustment_is_on_from_the_first_day_of_the_tuning_year(fresh):
    d = fresh[1]["daily"]
    d = d[(d["window"] == "previous") & (d["lead"] == 1) & (d["method"].isin(["blend", "weather"]))]
    # the first fit is in October 2024 with 176 days, so the adjustment is on from day one of the window
    wide = d[d["region"] == "SA1"].pivot(index="day", columns="method", values="abs_err")
    assert (wide["blend"] != wide["weather"]).mean() == 1


def test_fresher_and_perfect_weather_rank_as_they_should(fresh):
    w = fresh[1]["wape_by_input"]
    for r in REGIONS:
        assert w[r]["weather_perfect"] <= w[r]["weather_fresh"] <= w[r]["weather"] < w[r]["weather_temp"] < w[r]["blend"]


def test_headline_claims_follow_from_the_metrics(fresh):
    _, facts = fresh
    for h in ("day", "week"):
        assert facts["clear"][h] == list(REGIONS), "the report says every interval is clear of zero"
        assert not facts["worse"][h] and not facts["unclear"][h]
    assert facts["ranked"][-1] == "TAS1", "the report calls Tasmania the exception"
    assert all(f["sun"] < 0 and f["cool"] > 0 and f["heat"] > 0 for f in facts["slopes_read"].values())
    assert 1 / 3 < min(facts["midday_of_saving"].values())


def test_settings_stay_inside_the_grid():
    s = pd.read_csv(report.WEATHER_SETTINGS_CSV)
    assert s["heat"].isin(wx.HEATS).all() and s["cool"].isin(wx.COOLS).all() and s["lam"].isin(wx.LAMBDAS).all()
