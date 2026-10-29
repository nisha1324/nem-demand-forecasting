"""The committed profile is current, and its key numbers survive an independent pandas recount."""
import pandas as pd
import pytest

from demand.source import SNAPSHOT
from scripts import profile


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    facts = profile.write_outputs(out)
    return out, facts


@pytest.fixture(scope="module")
def snap() -> pd.DataFrame:
    df = pd.read_csv(SNAPSHOT, parse_dates=["interval_end"])
    start = df["interval_end"] - pd.to_timedelta(30, unit="min")
    return df.assign(day=start.dt.normalize(), year=start.dt.year,
                     slot=start.dt.hour * 60 + start.dt.minute,
                     weekend=start.dt.dayofweek >= 5)


def test_committed_report_matches_a_fresh_run(fresh):
    out, _ = fresh
    assert (out / profile.REPORT.name).read_text() == profile.REPORT.read_text(), \
        "results/SOURCE_PROFILE.md is stale: run python -m scripts.profile"


@pytest.mark.parametrize("name", profile.CHART_FILES)
def test_committed_charts_match_a_fresh_run(fresh, name):
    out, _ = fresh
    assert (out / "charts" / name).read_bytes() == (profile.CHARTS / name).read_bytes(), \
        f"results/charts/{name} is stale: run python -m scripts.profile"


def test_annual_energy_recount(fresh, snap):
    _, facts = fresh
    for region in ("VIC1", "SA1"):
        mw = snap.loc[(snap["region"] == region) & (snap["year"] == 2025), "demand_mw"]
        assert facts["energy"][(region, 2025)] == pytest.approx(mw.sum() * 0.5 / 1e6, rel=1e-6)


def test_midday_minimum_share_recount(fresh, snap):
    _, facts = fresh
    for region in ("SA1", "NSW1"):
        g = snap[(snap["region"] == region) & snap["year"].isin([2019, 2025])]
        lows = g.loc[g.groupby("day")["demand_mw"].idxmin()]
        share = lows.assign(mid=lows["slot"].between(600, 930)).groupby("year")["mid"].mean()
        assert facts["share"][(region, 2019)] == pytest.approx(share[2019])
        assert facts["share"][(region, 2025)] == pytest.approx(share[2025])


def test_weekend_effect_recount(fresh, snap):
    _, facts = fresh
    g = snap[(snap["region"] == "VIC1") & (snap["year"] == 2025)]
    daily = g.groupby(["day", "weekend"])["demand_mw"].mean().reset_index()
    ratio = (daily.loc[daily["weekend"], "demand_mw"].mean()
             / daily.loc[~daily["weekend"], "demand_mw"].mean())
    assert facts["weekend"]["VIC1"] == pytest.approx(ratio)


def test_sa_minimum_goes_negative_and_lands_at_midday(fresh, snap):
    _, facts = fresh
    g = snap[(snap["region"] == "SA1") & (snap["year"] == 2025)]
    low = g.loc[g["demand_mw"].idxmin()]
    assert facts["mins"][("SA1", 2025)][0] == pytest.approx(low["demand_mw"]) and low["demand_mw"] < 0
    assert 600 <= low["slot"] < 960
    assert "SA1" in facts["midday_min_regions"]
