"""The committed Part 4 report is current, buffers can't see the holdout, the forecasts are Part 3b's,
and the headline figures survive recounts that use none of demand/buffers.py."""
import math

import numpy as np
import pandas as pd
import pytest

from demand import buffers as bf
from demand.source import REGIONS
from scripts import buffers as report
from scripts import weather as part3b


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    return out, report.write_outputs(out)


@pytest.fixture(scope="module")
def data(fresh):
    return fresh[1]["data"]


@pytest.mark.parametrize("name", [report.REPORT.name, report.BUFFERS_CSV.name, report.COVERAGE_CSV.name])
def test_committed_tables_match_a_fresh_run(fresh, name):
    out, _ = fresh
    assert (out / name).read_text() == (report.RESULTS / name).read_text(), \
        f"results/{name} is stale: run python -m scripts.buffers"


@pytest.mark.parametrize("name", report.CHART_FILES)
def test_committed_charts_match_a_fresh_run(fresh, name):
    out, _ = fresh
    assert (out / "charts" / name).read_bytes() == (report.CHARTS / name).read_bytes(), \
        f"results/charts/{name} is stale: run python -m scripts.buffers"


@pytest.mark.parametrize("forecast", ["weather", "blend"])
@pytest.mark.parametrize("lead", [1, 7])
def test_the_forecasts_are_part_3bs(data, forecast, lead):
    scored = pd.read_csv(part3b.METRICS_CSV, dtype={"lead": str})
    scored = scored[(scored["window"] == "latest") & (scored["method"] == forecast) & (scored["lead"] == str(lead))].set_index("region")
    for r in REGIONS:
        d = data[r]
        rows = (d["days"] >= report.TESTED_ON[0]) & (d["days"] <= report.TESTED_ON[1])
        wape = np.abs(d["forecast"][forecast][lead] - d["m"])[rows].sum() / np.abs(d["m"][rows]).sum()
        assert wape == pytest.approx(scored.loc[r, "wape"], rel=1e-5)


def test_the_windows_dont_overlap():
    assert report.SET_ON[1] < report.TESTED_ON[0]
    assert report.SET_ON[0] == report.CARRY_FROM[0] and report.TESTED_ON[1] == report.CARRY_FROM[1]


def spoilt(d: dict) -> dict:
    """A copy of one region with the holdout's demand, prices and forecasts replaced by junk."""
    keep = report.in_window(d["days"], report.TESTED_ON)
    out = dict(d, m=d["m"].copy(), price=d["price"].copy(),
               forecast={k: {lead: a.copy() for lead, a in v.items()} for k, v in d["forecast"].items()})
    out["m"][keep] = 1e9
    out["price"][keep] = -1e9
    for by_lead in out["forecast"].values():
        for a in by_lead.values():
            a[keep] = -1e9
    return out


def test_buffers_set_beforehand_never_read_the_holdout(data):
    d = data["SA1"]
    test, days = report.misses(d, "weather", "day", report.TESTED_ON)
    for horizon in report.HORIZONS:
        clean, clean_days = report.misses(d, "weather", horizon, report.SET_ON)
        junk, junk_days = report.misses(spoilt(d), "weather", horizon, report.SET_ON)
        a = report.recipes(clean, clean_days, days, bf.UP, report.LEVEL)
        b = report.recipes(junk, junk_days, days, bf.UP, report.LEVEL)
        for recipe in report.RECIPES:
            assert (a[recipe] == b[recipe]).all(), recipe


def test_the_timetable_never_reads_the_holdout(data):
    d = data["VIC1"]
    keep = report.in_window(d["days"], report.TESTED_ON)
    clean = report.battery_rules(d, keep)["timetable"]
    junk = report.battery_rules(spoilt(d), keep)["timetable"]
    assert (clean[0] == junk[0]).all() and (clean[1] == junk[1]).all()


def test_only_the_ceilings_read_the_target_day(data):
    d = data["NSW1"]
    keep = report.in_window(d["days"], report.TESTED_ON)
    before = report.battery_rules(d, keep)
    # the last day's actuals are replaced, and its forecasts stay as issued
    changed = dict(d, m=d["m"].copy(), price=d["price"].copy())
    changed["m"][keep[-1]] = d["m"][keep[-1]][::-1]
    changed["price"][keep[-1]] = d["price"][keep[-1]][::-1]
    after = report.battery_rules(changed, keep)
    for rule in ("timetable", "seasonal_naive", "blend", "weather", "yesterday_price"):
        assert before[rule][0][-1] == after[rule][0][-1] and before[rule][1][-1] == after[rule][1][-1], rule
    assert (before["actual_price"][0][-1], before["actual_price"][1][-1]) != (after["actual_price"][0][-1], after["actual_price"][1][-1])


def plain_rank(values: list, level: float) -> float:
    ordered = sorted(values)
    return ordered[min(math.ceil((len(ordered) + 1) * level), len(ordered)) - 1]


def test_flat_buffer_and_its_coverage_recounted_by_hand(fresh, data):
    region, level = "NSW1", .95
    d = data[region]
    long = pd.DataFrame({"day": np.repeat(d["days"], 48), "miss": (d["m"] - d["forecast"]["weather"][1]).ravel()})
    worst = long.groupby("day")["miss"].max()
    buffer = plain_rank(worst[report.SET_ON[0]: report.SET_ON[1]].tolist(), level)
    tested = worst[report.TESTED_ON[0]: report.TESTED_ON[1]]
    c = fresh[1]["coverage"]
    row = c[(c["horizon"] == "day") & (c["region"] == region) & (c["side"] == "up") & (c["level"] == level) & (c["recipe"] == "flat")].iloc[0]
    assert row["mean_buffer_mw"] == pytest.approx(buffer)
    assert row["days_covered"] == pytest.approx((tested <= buffer).mean())


def test_week_ahead_buffer_pools_all_seven_leads(fresh, data):
    region, level = "TAS1", .95
    d = data[region]
    rows = report.in_window(d["days"], report.CARRY_FROM)
    worst = [max((d["m"] - d["forecast"]["weather"][lead])[t].tolist()) for lead in range(1, 8) for t in rows]
    assert len(worst) == 7 * len(rows)
    b = fresh[1]["buffers"]
    cell = b[(b["window"] == "carry_forward") & (b["horizon"] == "week") & (b["region"] == region) & (b["side"] == "up")
             & (b["level"] == level) & (b["season"] == report.ALL) & (b["block"] == report.ALL)].iloc[0]
    assert cell["buffer_mw"] == pytest.approx(plain_rank(worst, level)) and cell["days"] == len(worst)


def test_shaped_buffer_recounted_by_hand(fresh, data):
    region, level = "SA1", .95
    d = data[region]
    rows = report.in_window(d["days"], report.SET_ON)
    miss = -(d["m"] - d["forecast"]["weather"][1])[rows]  # the down side
    spans = [(0, 12), (12, 20), (20, 32), (32, 42), (42, 48)]
    base = [plain_rank([max(day[a:b]) for day in miss.tolist()], level) for a, b in spans]
    ratios = [max(max(day[a:b]) / x for (a, b), x in zip(spans, base)) for day in miss.tolist()]
    factor = plain_rank(ratios, level)
    hours = [(b - a) * x * factor for (a, b), x in zip(spans, base)]
    c = fresh[1]["coverage"]
    row = c[(c["horizon"] == "day") & (c["region"] == region) & (c["side"] == "down") & (c["level"] == level) & (c["recipe"] == "shaped")].iloc[0]
    assert row["mean_buffer_mw"] == pytest.approx(sum(hours) / 48)


def test_cost_recounted_by_hand(fresh, data):
    region = "VIC1"
    d = data[region]
    rows = report.in_window(d["days"], report.TESTED_ON)
    hedge = d["price"][rows].mean()
    total = 0.0
    for t in rows:
        for s in range(48):
            total += .01 * (d["m"][t, s] - d["forecast"]["weather"][1][t, s]) * (d["price"][t, s] - hedge) / 2
    assert fresh[1]["cost"].loc[(region, "weather"), "cost"] == pytest.approx(total)


def test_battery_spread_recounted_by_hand(fresh, data):
    region = "QLD1"
    d = data[region]
    rows = report.in_window(d["days"], report.TESTED_ON)
    win = lambda row, start: sum(row[start: start + 8]) / 8

    def best(row):
        pairs = [(win(row, c), win(row, e), c, e) for c in range(41) for e in range(c + 8, 41)]
        top = max(p[1] - p[0] for p in pairs)
        return next(p for p in pairs if p[1] - p[0] == top)

    earned = []
    for t in rows:
        _, _, c, e = best(d["forecast"]["weather"][1][t].tolist())
        earned.append(win(d["price"][t].tolist(), e) - win(d["price"][t].tolist(), c))
    assert fresh[1]["battery"].loc[(region, "weather"), "spread"] == pytest.approx(sum(earned) / len(earned))


def test_carried_shaped_buffers_meet_their_own_days(fresh, data):
    b = fresh[1]["buffers"]
    b = b[(b["window"] == "carry_forward") & (b["horizon"] == "day") & (b["level"] == report.LEVEL) & (b["promise"] == "whole_day") & (b["block"] != report.ALL)]
    for r in REGIONS:
        miss, _ = report.misses(data[r], "weather", "day", report.CARRY_FROM)
        for name, side in report.SIDES.items():
            cells = b[(b["region"] == r) & (b["side"] == name)].set_index("block")["buffer_mw"]
            buffer = np.empty(48)
            for block, (x, y) in bf.BLOCKS.items():
                buffer[x: y] = cells[block]
            assert bf.covered(miss, np.tile(buffer, (len(miss), 1)), side).mean() >= report.LEVEL


def test_every_buffer_is_positive_and_a_higher_level_never_shrinks_it():
    b = pd.read_csv(report.BUFFERS_CSV)
    assert (b["buffer_mw"] > 0).all()
    keys = ["window", "horizon", "region", "side", "season", "block", "promise"]
    assert set(b["level_pct"]) == {90, 95}
    wide = b.pivot_table(index=keys, columns="level_pct", values="buffer_mw")
    assert not wide.isna().any().any()
    assert (wide[95] >= wide[90]).all()


def test_headline_claims_follow_from_the_numbers(fresh):
    f = fresh[1]
    r = f["recipes"]
    assert r["halfhour"]["mean"] < .75, "the report says the shortcut breaks its promise"
    assert abs(r["flat"]["mean"] - r["shaped"]["mean"]) < .01, "the report calls shaped level with flat"
    assert r["shaped"]["size"].max() < 1, "the report says shaped is smaller in every test"
    assert r["block"]["mean"] < .9 < r["block"]["blocks"], "the report says the block promise holds and the day's doesn't"
    assert r["season_shaped"]["size"].mean() < r["shaped"]["size"].mean(), "the report says seasonal is smaller again"
    s = f["season_cover"]
    assert s.loc["shaped", "Summer"] == s.loc["shaped"].min() and s.loc["shaped", "Winter"] > s.loc["shaped", "Spring"]
    assert s.loc["season_shaped", "Spring"] == s.loc["season_shaped"].min(), "the report calls spring its weak spot"
    assert all(f["up_over_down"][x] > 1.15 for x in ("NSW1", "VIC1", "SA1")), "the report calls under-forecasts larger here"
    assert all(.9 < f["up_over_down"][x] < 1.1 for x in ("QLD1", "TAS1")), "the report calls these nearly equal"
    assert f["hardest_midday"] == 10
    assert f["exceptions_calendar"] / f["exceptions_past"] < .1, "the report says most breaches are on ordinary days"
    assert f["cost_faster"] == list(REGIONS)
    assert min(f["high_share"]["NSW1"], f["high_share"]["SA1"]) > .5, "the report says a cap is aimed at most of the cost there"
    assert max(f["high_share"]["QLD1"], f["high_share"]["VIC1"]) < .1
    assert f["battery_verdict"]["weather"]["better"] == ["SA1"], "the report says only South Australia gains"
    assert f["battery_verdict"]["weather"]["worse"] == ["QLD1"]
    assert f["battery_verdict"]["actual_demand"]["better"] == ["SA1"]
    assert len(f["battery_verdict"]["actual_price"]["better"]) == 5
    assert len(f["battery_verdict"]["yesterday_price"]["worse"]) == 4
