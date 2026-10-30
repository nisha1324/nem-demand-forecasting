"""Backtest four explainable baselines: how good is a forecast nobody has to trust blindly?

    python -m scripts.backtest

Rolling-origin backtest on the committed snapshot (see demand/backtest.py for
the mechanics). Every day from 1 Oct 2025 to 30 Sep 2026 is forecast at leads
1-7 by each method, from origins at midnight. Writes results/BACKTEST.md,
results/backtest_metrics.csv and two charts in results/charts/. Every number
in the report's text is computed here, not typed in.

Day-ahead = lead 1. Week-ahead = a forecast issued at midnight for the next 7
days, scored over all 7 (leads 1-7 pooled). A second window, Oct 2024 - Sep
2025, checks whether the ranking holds from one year to the next.
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from demand import backtest as bt  # noqa: E402
from demand.source import REGION_NAMES, REGIONS, ROOT, SNAPSHOT  # noqa: E402
from scripts.profile import GRID, INK, INK_2, SURFACE, md_table, save, style  # noqa: E402

RESULTS = ROOT / "results"
CHARTS = RESULTS / "charts"
REPORT = RESULTS / "BACKTEST.md"
METRICS_CSV = RESULTS / "backtest_metrics.csv"
CHART_FILES = ("backtest_error_by_lead.png", "backtest_error_by_time_of_day.png")

WINDOWS = {"latest": ("2025-10-01", "2026-09-30"), "previous": ("2024-10-01", "2025-09-30")}
MIDDAY = (20, 32)  # slots 20..31 = half-hours starting 10:00 ... 15:30, as in the profile
SUMMER_HOLS = ("12-24", "01-02")  # Christmas to New Year, the stretch weekly methods miss

# dataviz reference palette, categorical slots 1-4 in fixed order (validated
# light mode; slots 3-4 sit below 3:1 contrast, so every chart's numbers are
# also given as a table in the report)
METHOD_COLOR = {"naive": "#2a78d6", "seasonal_naive": "#eb6834",
                "mean_4w": "#1baf7a", "climatology": "#eda100"}
SHORT = {"naive": "Naive", "seasonal_naive": "Seasonal naive",
         "mean_4w": "4-week mean", "climatology": "Climatology"}
DOW = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def run() -> pd.DataFrame:
    """Forecasts for every region, window, method and lead."""
    snap = pd.read_csv(SNAPSHOT, usecols=["region", "interval_end", "demand_mw"])
    frames = []
    for region in REGIONS:
        days, m = bt.day_matrix(snap[snap["region"] == region])
        for window, (first, last) in WINDOWS.items():
            frames.append(bt.forecasts(days, m, first, last).assign(region=region, window=window))
    return pd.concat(frames, ignore_index=True)


def week_ahead(fc: pd.DataFrame) -> pd.DataFrame:
    """Leads 1-7 pooled: one forecast issued at midnight, scored over its whole week."""
    return bt.metrics(fc, by=("window", "region", "method"))


# ---------------------------------------------------------------- sections

def setup_section(mt: pd.DataFrame) -> tuple[str, dict]:
    first, last = WINDOWS["latest"]
    methods = md_table(["Method", "Forecast for a target day"],
                       [[m.label, m.explain] for m, _ in bt.METHODS.values()])
    sa = mt.loc[("latest", "SA1", slice(None), 1)]
    sa_mape = sa["mape"]
    text = f"""## 1. Setup

**Target days:** every day from {first} to {last} (365 days, all four seasons), in all 5 regions.
**Origins:** midnight at the start of a day, knowing every half-hour before it and nothing after.
**Leads:** 1 to 7 days. Lead 1 is the *day-ahead* forecast; leads 1-7 pooled are the *week-ahead*
forecast (one forecast issued at midnight, scored over the whole week it covers). The target days are
the same at every lead, so leads compare like with like.

{methods}

All four use only the same time of day, so the daily shape comes for free. The three weekly methods
also keep the day of the week; only the naive one ignores it. None of them use weather.

**Metrics** (all on half-hourly MW):
- **WAPE** = total absolute error / total absolute demand. The headline, because it stays meaningful
  when demand nears or crosses zero.
- **MAE** in MW, **bias** (mean forecast − actual; positive = over-forecast), and the mean error in
  each day's **peak** and **trough** (the daily max and min).
- **MAPE** is not used. South Australia shows why: its day-ahead MAPE ranges from
  {pct(sa_mape.min(), 0)} to {pct(sa_mape.max(), 0)} across the four methods, because a few half-hours with
  demand near zero dominate a mean of ratios, while its WAPE is {pct(sa['wape'].min())}-{pct(sa['wape'].max())}.
"""
    return text, {"sa_mape": sa_mape.to_dict()}


def day_ahead_section(mt: pd.DataFrame) -> tuple[str, dict]:
    d1 = mt.loc[("latest", slice(None), slice(None), 1)]
    best = {r: d1.loc[r]["wape"].idxmin() for r in REGIONS}
    worst = {r: d1.loc[r]["wape"].idxmax() for r in REGIONS}
    wape_body = [[r] + [pct(d1.loc[(r, k), "wape"]) for k in bt.METHODS] + [SHORT[best[r]]]
                 for r in REGIONS]
    detail = [[r, SHORT[best[r]], f"{d1.loc[(r, best[r]), 'mae']:,.0f}",
               f"{d1.loc[(r, best[r]), 'peak_mae']:,.0f}", f"{d1.loc[(r, best[r]), 'trough_mae']:,.0f}",
               f"{round(d1.loc[(r, best[r]), 'bias']):+,}"] for r in REGIONS]
    naive_wins = [r for r in REGIONS if best[r] == "naive"]
    sn_last = [r for r in REGIONS if worst[r] == "seasonal_naive"]
    gain = {r: 1 - d1.loc[(r, "naive"), "wape"] / d1.loc[(r, "seasonal_naive"), "wape"] for r in REGIONS}
    others = [f"{SHORT[best[r]].lower()} in {r} ({pct(d1.loc[(r, best[r]), 'wape'])})"
              for r in REGIONS if r not in naive_wins]
    headers = ["Region"] + [SHORT[k] for k in bt.METHODS] + ["Best"]
    text = f"""## 2. Day-ahead: yesterday beats last week

Day-ahead WAPE (lower is better):

{md_table(headers, wape_body)}

The best day-ahead method in each region, in MW:

{md_table(['Region', 'Best method', 'MAE', 'Daily peak error', 'Daily trough error', 'Bias'], detail)}

**Naive persistence (tomorrow = today) is the best day-ahead baseline in {len(naive_wins)} of
{len(REGIONS)} regions** ({', '.join(naive_wins)}); the others are {', '.join(others)}.
**Seasonal naive, the textbook baseline, comes last in {len(sn_last)} of {len(REGIONS)}.** Swapping last
week for yesterday cuts day-ahead error by {pct(min(gain.values()), 0)}-{pct(max(gain.values()), 0)}.
The likeliest reason is weather: yesterday's is a far better guide to tomorrow's than last week's,
and over one day that matters more than getting the day of the week right. Section 5 shows the cost of ignoring the
weekday, which is where naive loses.
"""
    return text, {"day_ahead_best": best, "naive_wins": naive_wins, "sn_last": sn_last,
                  "naive_vs_sn_gain": gain}


def week_ahead_section(mt: pd.DataFrame, wk: pd.DataFrame) -> tuple[str, dict]:
    lat = mt.loc["latest"]
    naive_by_lead = {r: [lat.loc[(r, "naive", L), "wape"] for L in bt.LEADS] for r in REGIONS}
    flat = all(lat.loc[(r, k), "wape"].nunique() == 1
               for r in REGIONS for k in bt.METHODS if k != "naive")
    w = wk.loc["latest"]
    best = {r: w.loc[r]["wape"].idxmin() for r in REGIONS}
    body = [[r] + [pct(w.loc[(r, k), "wape"]) for k in bt.METHODS] + [SHORT[best[r]]] for r in REGIONS]
    lead_body = [[r] + [pct(v) for v in naive_by_lead[r]] for r in REGIONS]
    peak_lead = {r: int(np.argmax(naive_by_lead[r])) + 1 for r in REGIONS}
    # first lead at which naive stops being the best method
    overtaken = {}
    for r in REGIONS:
        for L in bt.LEADS:
            if lat.loc[(r, slice(None), L), "wape"].idxmin()[1] != "naive":
                overtaken[r] = L
                break
    clim_wins = [r for r in REGIONS if best[r] == "climatology"]
    headers = ["Region"] + [SHORT[k] for k in bt.METHODS] + ["Best"]
    text = f"""## 3. Week-ahead: averages beat persistence

Week-ahead WAPE (a forecast issued at midnight, scored over all 7 days it covers):

{md_table(headers, body)}

Naive WAPE by lead (days ahead). The other three methods look back whole weeks, so for leads 1-7
they use exactly the same days and their error {'is the same at every lead' if flat else 'varies by lead'}:

{md_table(['Region'] + [f'Lead {L}' for L in bt.LEADS], lead_body)}

Persistence decays fast. From lead {max(overtaken.values())} on, naive is not the best method in any
region; its error peaks at lead
{min(peak_lead.values())}-{max(peak_lead.values())}, and recovers at lead 7, where "the last day seen" is the same weekday
and naive becomes seasonal naive. Over a whole week, **the 3-year climatology is the best baseline in
{len(clim_wins)} of {len(REGIONS)} regions** ({', '.join(clim_wins)}), and the 4-week mean in the rest
({', '.join(r for r in REGIONS if best[r] == 'mean_4w')}). Averaging many similar days washes out
one-off weather, which a single copied day can't do.

![Error by lead time](charts/{CHART_FILES[0]})
"""
    return text, {"week_ahead_best": best, "naive_by_lead": naive_by_lead, "flat": flat,
                  "overtaken": overtaken, "clim_wins": clim_wins}


def time_of_day_section(fc: pd.DataFrame, best: dict) -> tuple[str, dict]:
    d1 = fc[(fc["window"] == "latest") & (fc["lead"] == 1)].copy()
    d1["abs_err"] = (d1["forecast"] - d1["actual"]).abs()
    mid = d1["slot"].between(MIDDAY[0], MIDDAY[1] - 1)
    share = {r: d1.loc[mid & (d1["region"] == r) & (d1["method"] == best[r]), "abs_err"].sum()
             / d1.loc[(d1["region"] == r) & (d1["method"] == best[r]), "abs_err"].sum() for r in REGIONS}
    ratio = {}
    for r in REGIONS:
        g = d1[(d1["region"] == r) & (d1["method"] == best[r])]
        ratio[r] = g.loc[mid[g.index], "abs_err"].mean() / g.loc[~mid[g.index], "abs_err"].mean()
    wd = d1[(d1["method"] == "naive") & (d1["day"].dt.dayofweek < 5)]
    dip = {r: wd.loc[mid[wd.index] & (wd["region"] == r), "actual"].mean()
              / wd.loc[wd["region"] == r, "actual"].mean() for r in REGIONS}
    deepest, shallowest = min(dip, key=dip.get), max(dip, key=dip.get)
    hours = 6 / 24
    body = [[r, SHORT[best[r]], pct(share[r], 0), f"{ratio[r]:.1f}×"] for r in REGIONS]
    mainland = [r for r in REGIONS if r != "TAS1"]
    text = f"""## 4. Where in the day the error sits: midday

Share of each region's day-ahead absolute error (best method) that falls between 10:00 and 16:00,
a quarter of the day, and how much larger the average midday error is than at other times:

{md_table(['Region', 'Best day-ahead method', 'Error share 10:00-16:00', 'Midday vs rest of day'], body)}

On the mainland, {pct(min(share[r] for r in mainland), 0)}-{pct(max(share[r] for r in mainland), 0)} of the
error lands in the {pct(hours, 0)} of the day between 10:00 and 16:00, where midday half-hours miss by
{min(ratio[r] for r in mainland):.1f}-{max(ratio[r] for r in mainland):.1f}× as much as the rest. That fits
rooftop solar: cloud cover decides how much demand the grid sees at noon, and none of these
baselines know tomorrow's cloud. The two ends agree. {deepest}, where weekday midday demand is only
{pct(dip[deepest], 0)} of the all-day average this year (the deepest solar dip), has the most midday-heavy
error ({ratio[deepest]:.1f}×); {shallowest}, at {pct(dip[shallowest], 0)} (the shallowest dip), has the flattest
({ratio[shallowest]:.1f}×).

![Day-ahead error by time of day](charts/{CHART_FILES[1]})
"""
    return text, {"midday_share": share, "midday_ratio": ratio, "midday_dip": dip}


def bad_days_section(fc: pd.DataFrame, best: dict) -> tuple[str, dict]:
    d1 = fc[(fc["window"] == "latest") & (fc["lead"] == 1)].copy()
    d1["abs_err"] = (d1["forecast"] - d1["actual"]).abs()
    d1["abs_act"] = d1["actual"].abs()
    d1["dow"] = d1["day"].dt.dayofweek
    by_dow = d1.groupby(["region", "method", "dow"])[["abs_err", "abs_act"]].sum()
    by_dow = (by_dow["abs_err"] / by_dow["abs_act"]).unstack("dow")
    naive_mon_sat = {r: by_dow.loc[(r, "naive"), [0, 5]].mean() / by_dow.loc[(r, "naive"), [1, 2, 3, 4]].mean()
                     for r in REGIONS}
    dow_body = [[f"{r} {SHORT[k].lower()}"] + [pct(by_dow.loc[(r, k), d]) for d in range(7)]
                for r in REGIONS for k in ("naive", "seasonal_naive")]
    md = d1["day"].dt.strftime("%m-%d")
    hols = (md >= SUMMER_HOLS[0]) | (md <= SUMMER_HOLS[1])
    daily = d1.groupby(["region", "method", "day"])[["abs_err", "abs_act"]].sum()
    daily["wape"] = daily["abs_err"] / daily["abs_act"]
    hol_days = set(d1.loc[hols, "day"])
    hol_share = {(r, k): daily.loc[(r, k)].loc[lambda x: x.index.isin(hol_days), "abs_err"].sum()
                 / daily.loc[(r, k), "abs_err"].sum()
                 for r in REGIONS for k in bt.METHODS}
    n_hol = len(hol_days)
    worst_body = []
    for r in REGIONS:
        top = daily.loc[(r, best[r])].sort_values("wape", ascending=False).head(3)
        worst_body.append([r, SHORT[best[r]]] + [f"{d:%a %d %b %Y} ({pct(v, 0)})" for d, v in top["wape"].items()])
    sn_hol = [hol_share[(r, "seasonal_naive")] for r in REGIONS]
    over = [v / (n_hol / 365) for v in sn_hol]
    xmas = pd.Timestamp("2025-12-25")
    xmas_top3 = [r for r in REGIONS
                 if xmas in daily.loc[(r, best[r])].sort_values("wape", ascending=False).head(3).index]
    text = f"""## 5. Which days break the baselines

Day-ahead WAPE by day of the week (target day), naive vs seasonal naive:

{md_table(['Region / method'] + list(DOW), dow_body)}

**Naive fails on the days the week changes shape.** Its Monday and Saturday error is
{min(naive_mon_sat.values()):.1f}-{max(naive_mon_sat.values()):.1f}× its Tuesday-Friday error, because Monday
is forecast from a Sunday and Saturday from a Friday. Seasonal naive doesn't have that problem, but it
pays for it every other day.

**Holidays are the hardest days.** The {n_hol} days from 24 Dec to 2 Jan are {pct(n_hol / 365, 0)} of the
year but carry {pct(min(sn_hol), 0)}-{pct(max(sn_hol), 0)} of seasonal naive's day-ahead error,
{min(over):.1f}-{max(over):.1f}× their share, because each holiday is forecast from an ordinary day a week
earlier. Christmas Day is among the three worst days for the best method in
**{len(xmas_top3)} of {len(REGIONS)} regions** ({', '.join(xmas_top3)}).

The three worst day-ahead days for each region's best method (that day's WAPE):

{md_table(['Region', 'Method', 'Worst', '2nd', '3rd'], worst_body)}
"""
    return text, {"naive_mon_sat": naive_mon_sat, "hol_share": hol_share, "n_hol": n_hol,
                  "xmas_top3": xmas_top3}


def stability_section(mt: pd.DataFrame, wk: pd.DataFrame) -> tuple[str, dict]:
    keys = list(bt.METHODS)
    rank = {}
    body = []
    for horizon, src in (("Day-ahead", mt.xs(1, level="lead")), ("Week-ahead", wk)):
        for r in REGIONS:
            cur = src.loc[("latest", r)]["wape"]
            prev = src.loc[("previous", r)]["wape"]
            rank[(horizon, r)] = (cur.idxmin(), prev.idxmin())
            body.append([horizon, r, SHORT[cur.idxmin()], pct(cur.min()), SHORT[prev.idxmin()], pct(prev.min())])
    same = sum(a == b for a, b in rank.values())
    clim_bias = {r: mt.loc[("latest", r, "climatology", 1), "bias"] for r in REGIONS}
    clim_bias_pct = {r: clim_bias[r] / (mt.loc[("latest", r, "climatology", 1), "mae"]
                                        / mt.loc[("latest", r, "climatology", 1), "wape"]) for r in REGIONS}
    clim_wape_change = {r: mt.loc[("latest", r, "climatology", 1), "wape"]
                        - mt.loc[("previous", r, "climatology", 1), "wape"] for r in REGIONS}
    first, last = WINDOWS["previous"]
    big = max(REGIONS, key=lambda r: abs(clim_bias_pct[r]))
    worse = [r for r in REGIONS if clim_wape_change[r] > 0.01]
    worse_txt = (", ".join(f"{r} (+{100 * clim_wape_change[r]:.1f} points)" for r in worse)
                 if worse else "none")
    text = f"""## 6. Does the ranking hold? The year before

The same backtest on the previous 12 months ({first} to {last}):

{md_table(['Horizon', 'Region', 'Best, latest year', 'WAPE', 'Best, year before', 'WAPE'], body)}

The best method is the same in **{same} of {len(rank)}** region-horizon pairs. Climatology is the one
to watch: it averages the three previous years, so a shift in a region's level shows up as bias. In
the latest year its bias was {', '.join(f'{r} {clim_bias[r]:+,.0f} MW ({clim_bias_pct[r]:+.1%})' for r in REGIONS)}
of average demand, largest relative to its size in {big}. Regions where climatology's day-ahead WAPE got worse
by more than one point year on year: {worse_txt}. A long-run average is only as good as the
assumption that the level hasn't moved.
"""
    return text, {"rank": rank, "same": same, "clim_bias": clim_bias, "clim_bias_pct": clim_bias_pct,
                  "clim_wape_change": clim_wape_change}


def bar_section(mt: pd.DataFrame, wk: pd.DataFrame, facts: dict) -> str:
    body = []
    for r in REGIONS:
        d, w = facts["day_ahead_best"][r], facts["week_ahead_best"][r]
        body.append([REGION_NAMES[r], f"{SHORT[d]}, {pct(mt.loc[('latest', r, d, 1), 'wape'])}",
                     f"{SHORT[w]}, {pct(wk.loc[('latest', r, w), 'wape'])}",
                     f"{mt.loc[('latest', r, d, 1), 'mae']:,.0f} MW"])
    return f"""## 7. The bar for Part 3

The best baseline per region and horizon (latest year). A method in Part 3 is only worth its extra
complexity if it beats these:

{md_table(['Region', 'Day-ahead (WAPE)', 'Week-ahead (WAPE)', 'Day-ahead MAE'], body)}
"""


# ---------------------------------------------------------------- charts

def chart_by_lead(mt: pd.DataFrame, path: Path) -> None:
    lat = mt.loc["latest"]
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.7), facecolor=SURFACE)
    for ax, r in zip(axes, REGIONS):
        style(ax)
        for k, color in METHOD_COLOR.items():
            ys = [100 * lat.loc[(r, k, L), "wape"] for L in bt.LEADS]
            ax.plot(list(bt.LEADS), ys, color=color, linewidth=2, marker="o", markersize=4,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=SHORT[k])
        ax.set_ylim(0, 100 * lat.loc[r, "wape"].max() * 1.12)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
        ax.set_xticks(list(bt.LEADS))
        ax.set_xlim(0.7, 7.3)
        ax.set_title(REGION_NAMES[r], color=INK, fontsize=9.5, loc="left")
        ax.set_xlabel("Lead (days ahead)", color=INK_2, fontsize=8)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    axes[0].set_ylabel("WAPE, Oct 2025 - Sep 2026", color=INK_2, fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.045, 0.9), ncol=4,
               frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.suptitle("Yesterday is the best guide to tomorrow, but not to next week: forecast error by lead time",
                 x=0.045, ha="left", color=INK, fontsize=11.5, fontweight="bold")
    fig.subplots_adjust(left=0.045, right=0.99, top=0.74, bottom=0.15, wspace=0.25)
    save(fig, path)


def chart_by_time_of_day(fc: pd.DataFrame, path: Path) -> None:
    d1 = fc[(fc["window"] == "latest") & (fc["lead"] == 1)]
    tod = (d1.assign(abs_err=(d1["forecast"] - d1["actual"]).abs())
             .groupby(["region", "method", "slot"])["abs_err"].mean())
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.7), facecolor=SURFACE)
    x = np.arange(bt.SLOTS) / 2  # hours, start of each half-hour
    for ax, r in zip(axes, REGIONS):
        style(ax)
        for k, color in METHOD_COLOR.items():
            ax.plot(x, tod.loc[(r, k)].to_numpy(), color=color, linewidth=2, label=SHORT[k])
        ax.axvspan(MIDDAY[0] / 2, MIDDAY[1] / 2, color=GRID, alpha=0.45, linewidth=0, zorder=0)
        ax.set_ylim(0, tod.loc[r].max() * 1.08)
        ax.set_xticks([0, 6, 12, 18, 24], ["0:00", "6:00", "12:00", "18:00", "24:00"])
        ax.set_xlim(0, 24)
        ax.set_title(REGION_NAMES[r], color=INK, fontsize=9.5, loc="left")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes[0].set_ylabel("Day-ahead mean absolute error, MW", color=INK_2, fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.045, 0.9), ncol=4,
               frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.suptitle("Day-ahead error peaks at midday, the hours rooftop solar shapes (shaded 10:00-16:00)",
                 x=0.045, ha="left", color=INK, fontsize=11.5, fontweight="bold")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.74, bottom=0.1, wspace=0.28)
    save(fig, path)


# ---------------------------------------------------------------- main

def metrics_table(fc: pd.DataFrame) -> pd.DataFrame:
    return bt.metrics(fc, by=("window", "region", "method", "lead"))


def write_csv(mt: pd.DataFrame, wk: pd.DataFrame, path: Path) -> None:
    out = pd.concat([mt.reset_index().assign(lead=lambda d: d["lead"].astype(str)),
                     wk.reset_index().assign(lead="1-7")], ignore_index=True)
    out = out[["window", "region", "method", "lead", "n", "mae", "wape", "mape", "bias",
               "peak_mae", "trough_mae"]]
    out.to_csv(path, index=False, float_format="%.6g", lineterminator="\n")


def build(fc: pd.DataFrame) -> tuple[str, dict, pd.DataFrame, pd.DataFrame]:
    mt = metrics_table(fc)
    wk = week_ahead(fc)
    facts: dict = {}
    parts = []
    for text, f in (setup_section(mt), day_ahead_section(mt)):
        parts.append(text)
        facts.update(f)
    for text, f in (week_ahead_section(mt, wk),
                    time_of_day_section(fc, facts["day_ahead_best"]),
                    bad_days_section(fc, facts["day_ahead_best"]),
                    stability_section(mt, wk)):
        parts.append(text)
        facts.update(f)
    parts.append(bar_section(mt, wk, facts))
    first, last = WINDOWS["latest"]
    head = f"""# Backtest: how good are simple, explainable demand forecasts?

*Generated by `scripts/backtest.py` from `data/{SNAPSHOT.name}`. Do not edit by hand. Per-group
numbers for both windows and every lead are in [`backtest_metrics.csv`](backtest_metrics.csv).*

Rolling-origin backtest of four baselines on every day from {first} to {last}, in all 5 NEM regions,
1 to 7 days ahead. Times are NEM time (AEST); a half-hour belongs to the day it starts in.
"""
    mainland = [r for r in REGIONS if r != "TAS1"]
    tail = f"""## What this means for operations planning

- **Pick the baseline by horizon.** For tomorrow, copy today ({len(facts['naive_wins'])} of {len(REGIONS)}
  regions); for next week, average many similar days. The textbook seasonal naive is the wrong default
  at both horizons here.
- **The hard part is the middle of the day.** {pct(min(facts['midday_share'][r] for r in mainland), 0)}-{pct(max(facts['midday_share'][r] for r in mainland), 0)}
  of mainland day-ahead error sits between 10:00 and 16:00. Buffers and battery plans should be
  sized for the midday trough, not only the evening peak, and the method that most improves on these
  baselines will be the one that knows about tomorrow's cloud and temperature (Part 3).
- **Calendar effects are cheap to fix.** Naive's Monday and Saturday error runs
  {min(facts['naive_mon_sat'].values()):.1f}-{max(facts['naive_mon_sat'].values()):.1f}× its midweek error, and Christmas Day is among the
  three worst days in {len(facts['xmas_top3'])} of {len(REGIONS)} regions. A day-type rule (yesterday on ordinary
  weekdays, last week after the weekend changes shape, a holiday calendar) costs nothing in
  explainability.
- **Watch the level.** A long-run average carries a level shift straight into the forecast as bias,
  so it needs a recent-level correction before it can be trusted on its own.

*Source: Australian Energy Market Operator (AEMO), Aggregated price and demand data, used under
AEMO's copyright permissions (any purpose, with attribution to AEMO as author).*
"""
    return "\n".join([head] + parts + [tail]), facts, mt, wk


def write_outputs(results: Path, fc: pd.DataFrame | None = None) -> dict:
    """Write the report, metrics CSV and charts under results/ (tests point this at a temp folder)."""
    fc = run() if fc is None else fc
    report, facts, mt, wk = build(fc)
    charts = results / "charts"
    chart_by_lead(mt, charts / CHART_FILES[0])
    chart_by_time_of_day(fc, charts / CHART_FILES[1])
    write_csv(mt, wk, results / METRICS_CSV.name)
    (results / REPORT.name).write_text(report)
    facts.update(metrics=mt, week=wk)
    return facts


def main() -> int:
    write_outputs(RESULTS)
    print(f"wrote {REPORT.relative_to(ROOT)}, {METRICS_CSV.relative_to(ROOT)} and {len(CHART_FILES)} charts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
