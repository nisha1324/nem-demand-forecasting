"""Part 3: can a calendar and smoothing beat the Part 2 baselines, with no weather?

    python -m scripts.improve

Same harness, target days and leads as scripts/backtest.py. Every design choice is
made on the tuning year (Oct 2024 - Sep 2025); the latest year (Oct 2025 - Sep 2026)
is the holdout and is scored as it comes. Writes results/IMPROVEMENTS.md,
results/improvement_metrics.csv, results/holt_winters_settings.csv and two charts in
results/charts/. Every number in the report's text is computed here, not typed in.
"""
import functools
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from demand import backtest as bt  # noqa: E402
from demand import improved as im  # noqa: E402
from demand.source import REGION_NAMES, REGIONS, ROOT, SNAPSHOT  # noqa: E402
from scripts.backtest import DOW, METHOD_COLOR as BASE_COLOR, MIDDAY, SHORT as BASE_SHORT, pct  # noqa: E402
from scripts.backtest import WINDOWS, write_csv  # noqa: E402
from scripts.profile import GRID, INK, INK_2, SURFACE, md_table, save, style  # noqa: E402

RESULTS = ROOT / "results"
CHARTS = RESULTS / "charts"
REPORT = RESULTS / "IMPROVEMENTS.md"
METRICS_CSV = RESULTS / "improvement_metrics.csv"
SETTINGS_CSV = RESULTS / "holt_winters_settings.csv"
CHART_FILES = ("improvement_vs_bar.png", "blend_error_by_lead.png")

TUNING, HOLDOUT = "previous", "latest"
BASELINES = tuple(bt.METHODS)
NEW = ("weekday_rule", "hw_day", "hw_week", "hw_no_calendar", "climatology_adj", "blend")
BOOTSTRAP = 2000

# dataviz reference palette: the Part 2 methods keep their colours, the two new ones
# take slots 5 and 6 (validated light; yellow and magenta sit below 3:1, so the report
# also gives every number as a table)
METHOD_COLOR = {"naive": BASE_COLOR["naive"], "climatology": BASE_COLOR["climatology"],
                "hw": "#e87ba4", "blend": "#008300"}
SHORT = BASE_SHORT | {"weekday_rule": "Weekday rule", "hw_day": "Holt-Winters", "hw_week": "Holt-Winters",
                      "hw_no_calendar": "Holt-Winters, no calendar", "climatology_adj": "Corrected climatology",
                      "blend": "Blend"}


def settle(grid: pd.DataFrame, objective: str) -> dict:
    """The grid setting with the lowest error (first in grid order on a tie)."""
    best = grid.loc[grid[objective].idxmin()]
    return {"alpha": float(best["alpha"]), "beta": float(best["beta"]), "gamma": float(best["gamma"]),
            "wape": float(best[objective])}


def tune_all(series: dict) -> pd.DataFrame:
    """One day-ahead and one week-ahead Holt-Winters setting per region, from the tuning year only."""
    first, last = WINDOWS[TUNING]
    rows = []
    for region in REGIONS:
        days, m = series[region]
        targets = np.flatnonzero((days >= first) & (days <= last))
        grid = im.tune(m[: targets[-1] + 1], days.dayofweek.to_numpy(), im.holiday_mask(days, region), targets)
        for horizon, objective in (("day", "wape_day"), ("week", "wape_week")):
            rows.append({"region": region, "horizon": horizon} | settle(grid, objective))
    return pd.DataFrame(rows).set_index(["region", "horizon"])


def extra_methods(days: pd.DatetimeIndex, region: str, settings: pd.DataFrame) -> dict:
    dow = days.dayofweek.to_numpy()
    hol = im.holiday_mask(days, region)
    day, week = (settings.loc[(region, h), ["alpha", "beta", "gamma"]].to_dict() for h in ("day", "week"))
    return {
        "weekday_rule": functools.partial(im.weekday_rule, dow=dow, holiday=hol),
        "hw_day": im.HoltWinters(dow, hol, phi=im.PHI, **day),
        "hw_week": im.HoltWinters(dow, hol, phi=im.PHI, **week),
        "hw_no_calendar": im.HoltWinters(dow, np.zeros_like(hol), phi=im.PHI, **day),
        "climatology_adj": functools.partial(im.climatology_adj, dow=dow, holiday=hol),
        "blend": functools.partial(im.blend, hw=im.HoltWinters(dow, hol, phi=im.PHI, **day), dow=dow, holiday=hol),
    }


def run() -> dict:
    """Metrics plus the two roll-ups the report needs (the full forecasts are ~9M rows)."""
    snap = pd.read_csv(SNAPSHOT, usecols=["region", "interval_end", "demand_mw"])
    series = {r: bt.day_matrix(snap[snap["region"] == r]) for r in REGIONS}
    settings = tune_all(series)
    mts, wks, dailies, slots = [], [], [], []
    for region in REGIONS:
        days, m = series[region]
        for window, (first, last) in WINDOWS.items():
            fc = bt.forecasts(days, m, first, last, extra=extra_methods(days, region, settings))
            ids = {"window": window, "region": region}
            mts.append(bt.metrics(fc).assign(**ids))
            wks.append(bt.metrics(fc, by=("method",)).assign(**ids))
            fc["abs_err"] = (fc["forecast"] - fc["actual"]).abs()
            fc["abs_act"] = fc["actual"].abs()
            dailies.append(fc.groupby(["method", "lead", "day"])[["abs_err", "abs_act"]].sum()
                           .reset_index().assign(**ids))
            d1 = fc[fc["lead"] == 1]
            slots.append(d1.groupby(["method", "slot"])["abs_err"].sum().reset_index().assign(**ids))
    mt = pd.concat(mts).set_index(["window", "region"], append=True).reorder_levels(
        ["window", "region", "method", "lead"]).sort_index()
    wk = pd.concat(wks).set_index(["window", "region"], append=True).reorder_levels(
        ["window", "region", "method"]).sort_index()
    return {"settings": settings, "metrics": mt, "week": wk, "daily": pd.concat(dailies, ignore_index=True),
            "slots": pd.concat(slots, ignore_index=True)}


# ---------------------------------------------------------------- helpers

def day_ahead(R: dict, window: str = HOLDOUT) -> pd.DataFrame:
    return R["metrics"].xs((window, 1), level=("window", "lead"))


def week_ahead(R: dict, window: str = HOLDOUT) -> pd.DataFrame:
    return R["week"].xs(window, level="window")


def bar_method(table: pd.DataFrame, region: str) -> str:
    """The best Part 2 baseline in the region. It is picked on the holdout, which flatters it."""
    return table.loc[region].loc[list(BASELINES), "wape"].idxmin()


def error_by_day(R: dict, region: str, method: str, leads, window: str = HOLDOUT) -> pd.DataFrame:
    d = R["daily"]
    d = d[(d["window"] == window) & (d["region"] == region) & (d["method"] == method) & d["lead"].isin(leads)]
    return d.groupby("day")[["abs_err", "abs_act"]].sum().sort_index()


def cut_with_interval(R: dict, region: str, method: str, against: str, leads) -> tuple[float, float, float]:
    """Error cut of `method` against `against`, with a 95% interval from resampling whole weeks.

    Whole weeks, because a bad day tends to sit next to other bad days.
    """
    a = error_by_day(R, region, method, leads)["abs_err"].to_numpy()
    b = error_by_day(R, region, against, leads)["abs_err"].to_numpy()
    block = np.arange(len(a)) // 7
    a_wk, b_wk = np.bincount(block, a), np.bincount(block, b)
    rng = np.random.default_rng(20261030)
    picks = rng.integers(0, len(a_wk), size=(BOOTSTRAP, len(a_wk)))
    cuts = 1 - a_wk[picks].sum(axis=1) / b_wk[picks].sum(axis=1)
    return 1 - a.sum() / b.sum(), float(np.quantile(cuts, .025)), float(np.quantile(cuts, .975))


def signed(x: float, d: int = 1) -> str:
    return f"{100 * x:+.{d}f}%"


def holiday_days(region: str) -> pd.DatetimeIndex:
    first, last = WINDOWS[HOLDOUT]
    days = pd.date_range(first, last, freq="D")
    return days[im.holiday_mask(days, region)]


# ---------------------------------------------------------------- sections

def setup_section(R: dict) -> str:
    settings = R["settings"]
    rows = [[r] + [f"{settings.loc[(r, h), k]:g}" for h in ("day", "week") for k in ("alpha", "gamma")]
            for r in REGIONS]
    first, last = WINDOWS[TUNING]
    hold_first, hold_last = WINDOWS[HOLDOUT]
    trend_used = int((settings["beta"] > 0).sum())
    edge = [f"{r} {h} ({k} {settings.loc[(r, h), k]:g})" for (r, h) in settings.index for k, grid in
            (("alpha", im.ALPHAS), ("gamma", im.GAMMAS)) if settings.loc[(r, h), k] in (grid[0], grid[-1])]
    return f"""## 1. Setup

Same harness, target days, leads and metrics as [`BACKTEST.md`](BACKTEST.md), with one extra rule:
**every choice is made on the tuning year ({first} to {last}), and the holdout year ({hold_first}
to {hold_last}) is only scored.** The recipes below, the Holt-Winters settings and the 50/50 blend were
all fixed before the holdout was first run. One bug was fixed after that run: the weekday rule's Tuesday was
copying the previous Friday instead of Monday, against the rule as Part 2 stated it. Only that method's numbers
changed, and it isn't part of the blend.

| Method | Forecast for a target day |
|---|---:|
| Weekday rule | the rule Part 2 proposed: Tue-Fri copy the last ordinary weekday (yesterday, unless it was a holiday), Monday and Saturday copy last week's, Sundays and holidays copy the last weekend day or holiday |
| Holt-Winters | exponential smoothing of each half-hour with a weekly season; skips holidays, and reads a holiday as a Sunday |
| Corrected climatology | the Part 2 climatology plus its mean miss per half-hour over the last 28 days; a holiday reads the nearest Sunday |
| Blend | half Holt-Winters (day-ahead setting), half corrected climatology, at every lead |

**Holt-Winters settings**, one pair per region for each horizon, chosen from {len(im.ALPHAS) * len(im.BETAS) * len(im.GAMMAS)}
settings on the tuning year only. Alpha is how fast the level reacts to the latest day; gamma is how
fast the weekly shape can change:

{md_table(['Region', 'Day-ahead alpha', 'Day-ahead gamma', 'Week-ahead alpha', 'Week-ahead gamma'], rows)}

The level reacts faster for tomorrow than for next week in every region, and the weekly shape changes
slowly. A damped trend was on offer and was chosen in only {trend_used} of {len(settings)} cases. Settings at the
edge of the grid: {', '.join(edge) or 'none'}.

**The bar** is the best of the four Part 2 baselines in each region, picked on the holdout itself, which
flatters the baselines. **Cut** is the fall in total absolute error against that bar. The interval is a 95%
range from resampling whole weeks of the holdout ({BOOTSTRAP:,} draws): it shows how much the result
depends on which weeks fell in the year, and it says nothing about future years.
"""


def scoreboard(R: dict, table: pd.DataFrame, hw_key: str, leads) -> tuple[list, dict]:
    body, facts = [], {}
    for r in REGIONS:
        bar = bar_method(table, r)
        cut = cut_with_interval(R, r, "blend", bar, leads)
        keys = [hw_key, "weekday_rule", "climatology_adj", "blend"]
        best = table.loc[r, "wape"].idxmin()
        cells = [f"**{pct(table.loc[(r, k), 'wape'])}**" if k == best else pct(table.loc[(r, k), "wape"])
                 for k in keys]
        body.append([r, f"{SHORT[bar]}, {pct(table.loc[(r, bar), 'wape'])}", cells[1], cells[0], cells[2], cells[3],
                     signed(-cut[0]), f"{signed(-cut[2])} to {signed(-cut[1])}"])
        facts[r] = {"bar": bar, "cut": cut, "best": best}
    return body, facts


HEADERS = ["Region", "Bar", "Weekday rule", "Holt-Winters", "Corrected climatology", "Blend",
           "Blend vs bar", "95% interval"]


def results_section(R: dict) -> tuple[str, dict]:
    d1, wk = day_ahead(R), week_ahead(R)
    day_body, day = scoreboard(R, d1, "hw_day", (1,))
    week_body, week = scoreboard(R, wk, "hw_week", bt.LEADS)
    day_wins = [r for r in REGIONS if day[r]["best"] == "blend"]
    week_clear = [r for r in REGIONS if week[r]["cut"][1] > 0]
    week_unclear = [r for r in REGIONS if r not in week_clear]
    day_cuts = [day[r]["cut"][0] for r in REGIONS]
    mw = {r: (d1.loc[(r, day[r]["bar"]), "mae"], d1.loc[(r, "blend"), "mae"]) for r in REGIONS}
    mw_body = [[REGION_NAMES[r], f"{mw[r][0]:,.0f}", f"{mw[r][1]:,.0f}", f"{round(mw[r][0]) - round(mw[r][1]):,}"]
               for r in REGIONS]
    missed = [f"{r} ({signed(-week[r]['cut'][0])}, interval {signed(-week[r]['cut'][2])} to {signed(-week[r]['cut'][1])})"
              for r in week_unclear]
    week_best = {r: SHORT[week[r]["best"]] for r in REGIONS}
    text = f"""## 2. The holdout result

Day-ahead WAPE, {WINDOWS[HOLDOUT][0]} to {WINDOWS[HOLDOUT][1]} (lowest of all methods in bold):

{md_table(HEADERS, day_body)}

Week-ahead WAPE (Holt-Winters with its week-ahead setting):

{md_table(HEADERS, week_body)}

**Day-ahead, the blend is the best method in {len(day_wins)} of {len(REGIONS)} regions** and cuts error by
{pct(min(day_cuts), 0)}-{pct(max(day_cuts), 0)} against the bar, with every interval clear of zero.

**Week-ahead, the win is smaller and not universal.** The blend clears the bar, with an interval clear of zero, in
{len(week_clear)} of {len(REGIONS)} regions ({', '.join(week_clear)}). It doesn't in {' or '.join(missed) or 'any other'}: there the
interval spans zero, so the blend and the bar can't be told apart. The lowest week-ahead error by region:
{', '.join(f'{r} {v.lower()}' for r, v in week_best.items())}.

The day-ahead saving in MW (mean absolute error per half-hour):

{md_table(['Region', 'Bar', 'Blend', 'Saved'], mw_body)}

![Blend against the best baseline](charts/{CHART_FILES[0]})

![Error by lead time](charts/{CHART_FILES[1]})
"""
    return text, {"score_day": day, "score_week": week, "day_wins": day_wins, "week_clear": week_clear,
                  "week_unclear": week_unclear, "mw": mw}


def weekday_section(R: dict) -> tuple[str, dict]:
    d = R["daily"]
    d = d[(d["window"] == HOLDOUT) & (d["lead"] == 1) & d["method"].isin(["naive", "weekday_rule", "blend"])].copy()
    d["dow"] = d["day"].dt.dayofweek
    t = d.groupby(["region", "method", "dow"])[["abs_err", "abs_act"]].sum()
    wape = (t["abs_err"] / t["abs_act"]).unstack("dow")
    body = [[f"{r} {SHORT[k].lower()}"] + [pct(wape.loc[(r, k), x]) for x in range(7)]
            for r in REGIONS for k in ("naive", "weekday_rule", "blend")]
    d1 = day_ahead(R)
    lost = [r for r in REGIONS if d1.loc[(r, "weekday_rule"), "wape"] > d1.loc[(r, "naive"), "wape"]]
    won = [r for r in REGIONS if r not in lost]
    mon_sat = {(r, x): wape.loc[(r, "weekday_rule"), x] < wape.loc[(r, "naive"), x] for r in REGIONS for x in (0, 5)}
    wins = sum(mon_sat.values())
    gain = {(r, x): 1 - wape.loc[(r, "blend"), x] / wape.loc[(r, "naive"), x] for r in REGIONS for x in (0, 5)}
    rule_vs_naive = ", ".join(f"{r} {pct(d1.loc[(r, 'naive'), 'wape'])} to {pct(d1.loc[(r, 'weekday_rule'), 'wape'])}"
                              for r in REGIONS)
    worst_cell = max(((r, x) for r in REGIONS for x in (0, 5)),
                     key=lambda c: wape.loc[(c[0], "weekday_rule"), c[1]] - wape.loc[(c[0], "naive"), c[1]])
    text = f"""## 3. The cheap fix is a coin toss: the weekday rule

Part 2 suggested a free fix: keep yesterday midweek, and switch to last week on the days the week changes
shape. **It beats plain naive day-ahead in only {len(won)} of {len(REGIONS)} regions** ({', '.join(won)}; naive to rule:
{rule_vs_naive}). It only changes Mondays, Saturdays and days on or next to a holiday, and it beats naive in {wins} of
{len(mon_sat)} of those Monday and Saturday comparisons, but its misses are large: {worst_cell[0]} on {DOW[worst_cell[1]]}s goes from
{pct(wape.loc[(worst_cell[0], 'naive'), worst_cell[1]])} to {pct(wape.loc[(worst_cell[0], 'weekday_rule'), worst_cell[1]])}. Copying a week-old day fixes the shape of the day but brings in week-old weather.

The blend gets the benefit without that cost, because it keeps the recent level and adds a weekday offset: its
Monday and Saturday error is {pct(min(gain.values()), 0)}-{pct(max(gain.values()), 0)} below naive's.

Day-ahead WAPE by target weekday:

{md_table(['Region / method'] + list(DOW), body)}
"""
    return text, {"rule_lost": lost, "rule_won": won, "rule_mon_sat_wins": wins, "mon_sat_gain": gain}


def level_section(R: dict) -> tuple[str, dict]:
    d1 = day_ahead(R)
    level = {r: d1.loc[(r, "climatology"), "mae"] / d1.loc[(r, "climatology"), "wape"] for r in REGIONS}
    before = {r: d1.loc[(r, "climatology"), "bias"] for r in REGIONS}
    after = {r: d1.loc[(r, "climatology_adj"), "bias"] for r in REGIONS}
    body = [[r, f"{before[r]:+,.0f} MW ({before[r] / level[r]:+.1%})", f"{after[r]:+,.0f} MW ({after[r] / level[r]:+.1%})",
             pct(d1.loc[(r, "climatology"), "wape"]), pct(d1.loc[(r, "climatology_adj"), "wape"])] for r in REGIONS]
    worst = max(REGIONS, key=lambda r: abs(before[r] / level[r]))
    text = f"""## 4. Fixing the long-run average's level

Part 2 found climatology carries a shift in a region's level straight into the forecast. A 28-day
correction per half-hour takes most of that out (the holdout's day-ahead bias and WAPE):

{md_table(['Region', 'Climatology bias', 'Corrected bias', 'Climatology WAPE', 'Corrected WAPE'], body)}

{worst} shows it best: its bias falls from {before[worst] / level[worst]:+.1%} to {after[worst] / level[worst]:+.1%} of average
demand, and its WAPE from {pct(d1.loc[(worst, 'climatology'), 'wape'])} to {pct(d1.loc[(worst, 'climatology_adj'), 'wape'])}.
"""
    return text, {"bias_before": before, "bias_after": after, "mean_level": level}


def calendar_section(R: dict) -> tuple[str, dict]:
    d1, body, facts = day_ahead(R), [], {}
    for r in REGIONS:
        hol = holiday_days(r)
        bar = bar_method(d1, r)
        wape = {}
        for k in (bar, "hw_no_calendar", "hw_day", "blend"):
            e = error_by_day(R, r, k, (1,)).loc[hol]
            wape[k] = e["abs_err"].sum() / e["abs_act"].sum()
        all_days = d1.loc[(r, "hw_day"), "wape"] - d1.loc[(r, "hw_no_calendar"), "wape"]
        share = (error_by_day(R, r, "hw_day", (1,)).loc[hol, "abs_err"].sum()
                 / error_by_day(R, r, "hw_day", (1,))["abs_err"].sum())
        body.append([r, len(hol), f"{SHORT[bar]}, {pct(wape[bar])}", pct(wape["hw_no_calendar"]), pct(wape["hw_day"]),
                     pct(wape["blend"]), f"{100 * all_days:+.2f} pts"])
        facts[r] = {"n": len(hol), "share": share, "holiday_gain": 1 - wape["hw_day"] / wape["hw_no_calendar"],
                    "all_days": all_days, "bar": wape[bar], "blend": wape["blend"]}
    alld = [facts[r]["all_days"] for r in REGIONS]
    helped = [r for r in REGIONS if facts[r]["all_days"] < 0]
    hurt = [r for r in REGIONS if facts[r]["all_days"] > 0]
    ranked = sorted(REGIONS, key=lambda r: facts[r]["holiday_gain"], reverse=True)
    gain_txt = ", ".join(f"{r} {signed(-facts[r]['holiday_gain'], 0)}" for r in ranked)
    counts = [facts[r]["n"] for r in REGIONS]
    text = f"""## 5. What the holiday calendar is worth

Day-ahead WAPE on public holidays only (the bar is that region's all-year best baseline), and the change the
calendar makes to Holt-Winters across all 365 days:

{md_table(['Region', 'Holidays', 'Bar', 'Holt-Winters, no calendar', 'Holt-Winters', 'Blend', 'All-year effect'], body)}

On the holidays themselves, the calendar changes Holt-Winters' error by {gain_txt}. It matters most in
{ranked[0]} and {ranked[1]}, and much less elsewhere. Over the whole year it moves WAPE by {100 * min(alld):+.2f} to
{100 * max(alld):+.2f} points: better in {len(helped)} of {len(REGIONS)} regions{f', slightly worse in {", ".join(hurt)}' if hurt else ''}. With only
{min(counts)}-{max(counts)} holidays a year, a calendar fixes particular bad days rather than general accuracy.
"""
    return text, {"calendar": facts, "calendar_helped": helped, "calendar_hurt": hurt, "calendar_ranked": ranked}


def where_section(R: dict) -> tuple[str, dict]:
    d1 = day_ahead(R)
    s = R["slots"]
    s = s[s["window"] == HOLDOUT]
    mid = s["slot"].between(MIDDAY[0], MIDDAY[1] - 1)
    body, share_after, mw_saved = [], {}, {}
    for r in REGIONS:
        bar = bar_method(d1, r)
        g = s[s["region"] == r]
        def part(k):
            one = g[g["method"] == k]
            return one.loc[mid[one.index], "abs_err"].sum(), one["abs_err"].sum()
        mb, tb = part(bar)
        ma, ta = part("blend")
        share_after[r] = ma / ta
        mw_saved[r] = (mb - ma) / (tb - ta)
        body.append([r, f"{d1.loc[(r, bar), 'peak_mae']:,.0f}", f"{d1.loc[(r, 'blend'), 'peak_mae']:,.0f}",
                     f"{d1.loc[(r, bar), 'trough_mae']:,.0f}", f"{d1.loc[(r, 'blend'), 'trough_mae']:,.0f}",
                     pct(mb / tb, 0), pct(ma / ta, 0)])
    mainland = [r for r in REGIONS if r != "TAS1"]
    d = R["daily"]
    d = d[(d["window"] == HOLDOUT) & (d["lead"] == 1) & (d["method"] == "blend")]
    worst_body, shutdown = [], 0
    for r in REGIONS:
        g = d[d["region"] == r].assign(w=lambda x: x["abs_err"] / x["abs_act"]).sort_values("w", ascending=False).head(3)
        worst_body.append([r] + [f"{x:%a %d %b %Y} ({pct(v, 0)})" for x, v in zip(g["day"], g["w"])])
        shutdown += int(((g["day"].dt.month == 12) & (g["day"].dt.day >= 27)).sum())
    text = f"""## 6. Where the remaining error sits

Day-ahead, the bar against the blend (peak and trough = mean daily error in MW):

{md_table(['Region', 'Peak, bar', 'Peak, blend', 'Trough, bar', 'Trough, blend', 'Midday share, bar', 'Midday share, blend'], body)}

The middle of the day is still the problem. After the blend, {pct(min(share_after[r] for r in mainland), 0)}-{pct(max(share_after[r] for r in mainland), 0)}
of mainland day-ahead error still falls between 10:00 and 16:00, a quarter of the day, and only
{pct(min(mw_saved[r] for r in mainland), 0)}-{pct(max(mw_saved[r] for r in mainland), 0)} of the MW saved came from those hours. Calendar and smoothing can't know tomorrow's
cloud, which is what the weather step is for.

The three worst day-ahead days for the blend (that day's WAPE):

{md_table(['Region', 'Worst', '2nd', '3rd'], worst_body)}

{shutdown} of these 15 days fall between 27 and 31 December: the shutdown between Christmas and New Year, which
a public-holiday calendar doesn't cover.
"""
    return text, {"midday_share_blend": share_after, "midday_of_saving": mw_saved, "shutdown_worst": shutdown}


def stability_section(R: dict) -> tuple[str, dict]:
    tune_d, hold_d = day_ahead(R, TUNING), day_ahead(R)
    tune_w, hold_w = week_ahead(R, TUNING), week_ahead(R)
    def cut(table, r):
        return 1 - table.loc[(r, "blend"), "wape"] / table.loc[r].loc[list(BASELINES), "wape"].min()
    body = [[r, signed(-cut(tune_d, r)), signed(-cut(hold_d, r)), signed(-cut(tune_w, r)), signed(-cut(hold_w, r))]
            for r in REGIONS]
    held = sum(cut(hold_d, r) > 0 for r in REGIONS)
    held_week = sum(cut(hold_w, r) > 0 for r in REGIONS)
    return f"""## 7. Does it hold outside the year it was tuned on?

Blend against the best baseline of the same year:

{md_table(['Region', 'Day-ahead, tuning year', 'Day-ahead, holdout', 'Week-ahead, tuning year', 'Week-ahead, holdout'], body)}

The tuning-year column is in-sample, so the holdout column is the one to quote. The gain carried over in {held} of
{len(REGIONS)} regions day-ahead and {held_week} of {len(REGIONS)} week-ahead.
""", {"tuning_cut_day": {r: cut(tune_d, r) for r in REGIONS}, "holdout_cut_day": {r: cut(hold_d, r) for r in REGIONS},
      "tuning_cut_week": {r: cut(tune_w, r) for r in REGIONS}, "holdout_cut_week": {r: cut(hold_w, r) for r in REGIONS}}


def bar_section(R: dict) -> str:
    d1, wk = day_ahead(R), week_ahead(R)
    body = [[REGION_NAMES[r], pct(d1.loc[(r, "blend"), "wape"]), pct(wk.loc[(r, "blend"), "wape"]),
             f"{d1.loc[(r, 'blend'), 'mae']:,.0f} MW"] for r in REGIONS]
    return f"""## 8. The bar for the weather step

The blend is what a weather adjustment has to beat. It's the single recipe chosen on the tuning year, so its
holdout scores are fair. Picking the best method per region from the holdout would not be:

{md_table(['Region', 'Day-ahead (WAPE)', 'Week-ahead (WAPE)', 'Day-ahead MAE'], body)}
"""


# ---------------------------------------------------------------- charts

def chart_vs_bar(R: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), facecolor=SURFACE, sharey=True)
    tables = ((axes[0], day_ahead(R), "Day-ahead"), (axes[1], week_ahead(R), "Week-ahead"))
    y = np.arange(len(REGIONS))[::-1]
    for ax, table, title in tables:
        style(ax)
        ax.grid(False)
        ax.xaxis.grid(True, color=GRID, linewidth=.8)
        ax.set_axisbelow(True)
        for yi, r in zip(y, REGIONS):
            bar = 100 * table.loc[(r, bar_method(table, r)), "wape"]
            blend = 100 * table.loc[(r, "blend"), "wape"]
            ax.plot([bar, blend], [yi, yi], color=GRID, linewidth=2.5, solid_capstyle="round", zorder=2)
            ax.scatter(bar, yi, s=70, color=INK_2, edgecolor=SURFACE, linewidth=1.2, zorder=3,
                       label="Best Part 2 baseline" if yi == y[0] else None)
            ax.scatter(blend, yi, s=70, color=METHOD_COLOR["blend"], edgecolor=SURFACE, linewidth=1.2, zorder=3,
                       label="Blend" if yi == y[0] else None)
            left, right = (blend, bar) if blend < bar else (bar, blend)
            ax.text(left - .2, yi, f"{left:.1f}%", ha="right", va="center", color=INK_2, fontsize=8.5)
            ax.text(right + .2, yi, f"{right:.1f}%", ha="left", va="center", color=INK_2, fontsize=8.5)
        ax.set_xlim(2.5, 15.5)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("WAPE, Oct 2025 - Sep 2026 (holdout)", color=INK_2, fontsize=8.5)
    axes[0].set_yticks(y, [REGION_NAMES[r] for r in REGIONS])
    axes[0].set_ylim(-.6, len(REGIONS) - .4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.125, .925), ncol=2, frameon=False,
               fontsize=8.5, labelcolor=INK_2, handletextpad=.2)
    fig.suptitle("The blend beats every baseline day-ahead; a week out the gain is smaller",
                 x=.125, ha="left", color=INK, fontsize=11.5, fontweight="bold")
    fig.subplots_adjust(left=.125, right=.985, top=.77, bottom=.14, wspace=.08)
    save(fig, path)


def chart_by_lead(R: dict, path: Path) -> None:
    lat = R["metrics"].loc[HOLDOUT]
    series = (("naive", "naive"), ("climatology", "climatology"), ("hw_day", "hw"), ("blend", "blend"))
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.7), facecolor=SURFACE)
    for ax, r in zip(axes, REGIONS):
        style(ax)
        for key, slot in series:
            ys = [100 * lat.loc[(r, key, L), "wape"] for L in bt.LEADS]
            ax.plot(list(bt.LEADS), ys, color=METHOD_COLOR[slot], linewidth=2, marker="o", markersize=4,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=SHORT[key])
        ax.set_ylim(0, 100 * lat.loc[r].loc[["naive", "climatology", "hw_day", "blend"], "wape"].max() * 1.12)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
        ax.set_xticks(list(bt.LEADS))
        ax.set_xlim(.7, 7.3)
        ax.set_title(REGION_NAMES[r], color=INK, fontsize=9.5, loc="left")
        ax.set_xlabel("Lead (days ahead)", color=INK_2, fontsize=8)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    axes[0].set_ylabel("WAPE, Oct 2025 - Sep 2026", color=INK_2, fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.045, .9), ncol=4, frameon=False,
               fontsize=8.5, labelcolor=INK_2)
    fig.suptitle("The blend starts lower and stays low: forecast error by lead time",
                 x=.045, ha="left", color=INK, fontsize=11.5, fontweight="bold")
    fig.subplots_adjust(left=.045, right=.99, top=.74, bottom=.15, wspace=.25)
    save(fig, path)


# ---------------------------------------------------------------- main

def build(R: dict) -> tuple[str, dict]:
    facts: dict = {}
    parts = [setup_section(R)]
    for fn in (results_section, weekday_section, level_section, calendar_section, where_section, stability_section):
        text, f = fn(R)
        parts.append(text)
        facts.update(f)
    parts.append(bar_section(R))
    day = facts["score_day"]
    head = f"""# Improving on the baselines: calendar, smoothing and level correction, no weather

*Generated by `scripts/improve.py` from `data/{SNAPSHOT.name}` and `data/public_holidays.csv`. Do not edit by
hand. Every region, method, lead and window is in [`improvement_metrics.csv`](improvement_metrics.csv).*
"""
    tail = f"""## What this means for operations planning

- **Use one recipe, not a rule per horizon.** Half smoothed recent demand, half corrected long-run average cuts
  day-ahead error by {pct(min(day[r]['cut'][0] for r in REGIONS), 0)}-{pct(max(day[r]['cut'][0] for r in REGIONS), 0)} against the best simple baseline. It needs nothing but past
  demand and a calendar, and a planner can see exactly why each number came out as it did.
- **Don't spend effort on weekday rules.** The obvious rule beats doing nothing in only {len(facts['rule_won'])} of {len(REGIONS)} regions.
  Adjust for the weekday; don't swap to an older day.
- **Treat the holiday calendar as a fix for known bad days.** It matters most in {facts['calendar_ranked'][0]} and
  {facts['calendar_ranked'][1]} and barely moves the yearly figure, so judge it on the holidays alone.
- **Keep a recent-level correction on any long-run average.** It's simple arithmetic, and it removes most of the
  bias when a region's level has moved.
- **The midday trough remains the gap.** That's where the next gain has to come from, and it needs weather.

## Limits

- One tuning year and one holdout year. The intervals cover the weeks within the holdout, not other years.
- The holdout wasn't completely unseen. Part 2 reported the baselines on the same year, and its findings (the
  level bias, the holiday misses) suggested what to try here. The settings and the blend were still fixed without it.
- The calendar's 2026 dates were checked against official lists for NSW and Queensland only. The Victorian, South
  Australian and Tasmanian pages block automated requests, so theirs rest on the `holidays` package's rules.
- The 2019-2025 dates reconcile with the official list; every difference is explained in `scripts/build_holidays.py`.
- Part-day and local holidays (Christmas Eve evening, Royal Queensland Show, Royal Hobart Regatta) are not modelled.
- No weather yet: nothing here knows a heatwave or a cloudy day is coming.

*Sources: Australian Energy Market Operator (AEMO), Aggregated price and demand data, used under AEMO's copyright
permissions (any purpose, with attribution to AEMO as author). Public holidays: the `holidays` Python package (MIT);
checked against the Australian Public Holidays Dates Machine Readable Dataset, data.gov.au, CC BY 3.0 AU.*
"""
    return "\n".join([head] + parts + [tail]), facts


def write_outputs(results: Path, R: dict | None = None) -> dict:
    """Write the report, tables and charts under results/ (tests point this at a temp folder)."""
    R = run() if R is None else R
    report, facts = build(R)
    charts = results / "charts"
    chart_vs_bar(R, charts / CHART_FILES[0])
    chart_by_lead(R, charts / CHART_FILES[1])
    write_csv(R["metrics"], R["week"], results / METRICS_CSV.name)
    R["settings"].reset_index().to_csv(results / SETTINGS_CSV.name, index=False, float_format="%.6g",
                                       lineterminator="\n")
    (results / REPORT.name).write_text(report)
    return facts | R


def main() -> int:
    write_outputs(RESULTS)
    print(f"wrote {REPORT.relative_to(ROOT)}, {METRICS_CSV.relative_to(ROOT)}, "
          f"{SETTINGS_CSV.relative_to(ROOT)} and {len(CHART_FILES)} charts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
