"""The committed backtest report is current, and its numbers survive a recount
that doesn't use the harness at all (plain pandas shifts on the snapshot)."""
import pandas as pd
import pytest

from demand.source import SNAPSHOT
from scripts import backtest


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    facts = backtest.write_outputs(out)
    return out, facts


@pytest.fixture(scope="module")
def snap() -> pd.DataFrame:
    df = pd.read_csv(SNAPSHOT, usecols=["region", "interval_end", "demand_mw"], parse_dates=["interval_end"])
    return df.sort_values(["region", "interval_end"], ignore_index=True)


def recount(snap, region, lag_halfhours=None, lags=None, first="2025-10-01", last="2026-09-30"):
    """Forecast = demand `lag` half-hours earlier (or the mean over several lags), on target days."""
    g = snap[snap["region"] == region].set_index("interval_end")["demand_mw"]
    lags = lags or [lag_halfhours]
    fc = pd.concat([g.shift(k) for k in lags], axis=1).mean(axis=1)
    day = (g.index - pd.Timedelta("30min")).normalize()
    keep = (day >= first) & (day <= last)
    err = (fc - g)[keep]
    return err.abs().mean(), err.abs().sum() / g[keep].abs().sum(), err.mean()


def test_committed_report_matches_a_fresh_run(fresh):
    out, _ = fresh
    assert (out / backtest.REPORT.name).read_text() == backtest.REPORT.read_text(), \
        "results/BACKTEST.md is stale: run python -m scripts.backtest"


def test_committed_metrics_csv_matches_a_fresh_run(fresh):
    out, _ = fresh
    assert (out / backtest.METRICS_CSV.name).read_text() == backtest.METRICS_CSV.read_text(), \
        "results/backtest_metrics.csv is stale: run python -m scripts.backtest"


@pytest.mark.parametrize("name", backtest.CHART_FILES)
def test_committed_charts_match_a_fresh_run(fresh, name):
    out, _ = fresh
    assert (out / "charts" / name).read_bytes() == (backtest.CHARTS / name).read_bytes(), \
        f"results/charts/{name} is stale: run python -m scripts.backtest"


@pytest.mark.parametrize("region", ["NSW1", "SA1"])
def test_seasonal_naive_recount(fresh, snap, region):
    _, facts = fresh
    mae, wape, bias = recount(snap, region, 7 * 48)
    row = facts["metrics"].loc[("latest", region, "seasonal_naive", 1)]
    assert row["mae"] == pytest.approx(mae) and row["wape"] == pytest.approx(wape)
    assert row["bias"] == pytest.approx(bias, abs=1e-6)


@pytest.mark.parametrize("lead", [1, 4])
def test_naive_recount_by_lead(fresh, snap, lead):
    _, facts = fresh
    # lead L copies the last day seen, which is L days before the target
    mae, wape, _ = recount(snap, "VIC1", lead * 48)
    row = facts["metrics"].loc[("latest", "VIC1", "naive", lead)]
    assert row["mae"] == pytest.approx(mae) and row["wape"] == pytest.approx(wape)


def test_climatology_and_4_week_mean_recount(fresh, snap):
    _, facts = fresh
    clim = [48 * (364 * y - 7 * w) for y in (1, 2, 3) for w in range(-3, 4)]
    _, wape, bias = recount(snap, "TAS1", lags=clim)
    row = facts["metrics"].loc[("latest", "TAS1", "climatology", 1)]
    assert row["wape"] == pytest.approx(wape) and row["bias"] == pytest.approx(bias)
    _, wape, _ = recount(snap, "QLD1", lags=[48 * 7 * k for k in (1, 2, 3, 4)],
                         first="2024-10-01", last="2025-09-30")
    assert facts["metrics"].loc[("previous", "QLD1", "mean_4w", 1), "wape"] == pytest.approx(wape)


def test_week_ahead_pools_all_seven_leads(fresh):
    _, facts = fresh
    mt, wk = facts["metrics"], facts["week"]
    for region in ("NSW1", "TAS1"):
        leads = mt.loc[("latest", region, "naive")]
        # equal n per lead, so pooled WAPE = sum of MAE / sum of mean absolute demand (MAE / WAPE)
        pooled = leads["mae"].sum() / (leads["mae"] / leads["wape"]).sum()
        assert wk.loc[("latest", region, "naive"), "wape"] == pytest.approx(pooled)


def test_headline_claims_follow_from_the_metrics(fresh):
    _, facts = fresh
    mt, wk = facts["metrics"], facts["week"]
    for region, best in facts["day_ahead_best"].items():
        d1 = mt.loc[("latest", region, slice(None), 1), "wape"]
        assert d1.idxmin()[2] == best
    # the weekly methods use the same days at every lead, so their error can't depend on it
    assert facts["flat"]
    for region, best in facts["week_ahead_best"].items():
        assert wk.loc[("latest", region), "wape"].idxmin() == best
    assert 0 < min(facts["midday_share"].values()) and max(facts["midday_share"].values()) < 1
