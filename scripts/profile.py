"""Profile the half-hourly snapshot: is it complete, and what shapes must a forecast capture?

    python -m scripts.profile

Reads data/demand_halfhourly.csv.gz with DuckDB and writes
results/SOURCE_PROFILE.md plus two charts in results/charts/. Every number in
the report's text is computed here, not typed in.

Conventions: a half-hour belongs to the day and time-of-day it STARTS in
(interval_end - 30 min), in NEM time (AEST, no daylight saving). Calendar
years 2019-2025 are complete; 2026 runs to September, so year-on-year tables
stop at 2025.
"""
import json
import sys
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from demand.source import REGION_NAMES, REGIONS, ROOT, SNAPSHOT, SNAPSHOT_META  # noqa: E402

RESULTS = ROOT / "results"
CHARTS = RESULTS / "charts"
REPORT = RESULTS / "SOURCE_PROFILE.md"

FULL_YEARS = list(range(2019, 2026))
FIRST_YEAR, LAST_YEAR = FULL_YEARS[0], FULL_YEARS[-1]
MIDDAY = (10 * 60, 16 * 60)  # half-hours starting 10:00 ... 15:30 NEM time

# dataviz reference palette, light mode. Years are ordered, so they take an
# ordinal blue ramp (steps 250 / 450 / 650, validated with --ordinal).
YEAR_COLOR = {2019: "#86b6ef", 2022: "#2a78d6", 2025: "#104281"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"""
        create table hh as
        select region,
               interval_end::timestamp                           as interval_end,
               demand_mw, rrp, n_intervals, resolution_min,
               interval_end::timestamp - interval 30 minute      as interval_start,
               cast(interval_end::timestamp - interval 30 minute as date) as day,
               extract(hour from interval_end::timestamp - interval 30 minute) * 60
                 + extract(minute from interval_end::timestamp - interval 30 minute) as slot
        from read_csv('{SNAPSHOT}', header = true,
                      columns = {{'region': 'varchar', 'interval_end': 'varchar',
                                 'demand_mw': 'double', 'rrp': 'double',
                                 'n_intervals': 'integer', 'resolution_min': 'integer'}})
    """)
    con.execute("""
        create table days as
        select region, day, year(day) as year,
               isodow(day) >= 6                    as is_weekend,
               avg(demand_mw)                      as mean_mw,
               max(demand_mw)                      as peak_mw,
               min(demand_mw)                      as min_mw,
               arg_min(slot, demand_mw)            as min_slot,
               count(*)                            as n
        from hh group by all
    """)
    return con


def rows(con, sql, *params):
    return con.execute(sql, params).fetchall()


def md_table(headers, body) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---:" if i else "---" for i in range(len(headers))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return "\n".join(out)


def hhmm(slot: int) -> str:
    return f"{int(slot) // 60:02d}:{int(slot) % 60:02d}"


# ---------------------------------------------------------------- sections

def coverage(con, meta) -> tuple[str, dict]:
    expected = rows(con, """
        select count(*) * 48 from range(date '2019-01-01', date '2026-10-01', interval 1 day)
    """)[0][0]
    table = rows(con, """
        select region, count(*), count(*) filter (where resolution_min = 5 and n_intervals < 6),
               round(min(demand_mw), 1), round(max(demand_mw), 1),
               count(*) filter (where demand_mw <= 0),
               count(*) filter (where rrp < 0)
        from hh group by region
    """)
    by_region = {r[0]: r for r in table}
    missing = {r: expected - by_region[r][1] for r in REGIONS}
    short = {r: by_region[r][2] for r in REGIONS}
    checks = meta["raw_checks"]
    body = [[r, f"{by_region[r][1]:,}", f"{missing[r]:,}", f"{short[r]:,}",
             f"{by_region[r][3]:,}", f"{by_region[r][4]:,}"] for r in REGIONS]
    gaps = "no missing half-hours" if not any(missing.values()) else \
        f"{sum(missing.values()):,} missing half-hours"
    shorts = "every 5-minute half-hour has all 6 intervals" if not any(short.values()) else \
        f"{sum(short.values()):,} half-hours have fewer than 6 five-minute intervals"
    ptypes = ", ".join(f"`{k}` ({v:,})" for k, v in checks["period_types"].items())
    text = f"""## 1. Coverage and integrity

{checks['files']} monthly files (5 regions x {checks['files'] // 5} months, Jan 2019 - Sep 2026),
every one matching its pinned SHA-256 in `data/raw_manifest.json`.
They hold {checks.get('raw_rows_30min', 0):,} thirty-minute rows (to Sep 2021) and
{checks.get('raw_rows_5min', 0):,} five-minute rows (from Oct 2021), with
{checks['exact_duplicate_rows']:,} exact duplicate rows. Period types: {ptypes}.

Rolled up to half-hours, each region should have {expected:,} rows. Result: **{gaps}**, and
{shorts}.

{md_table(['Region', 'Half-hours', 'Missing', 'Short (5-min era)', 'Min MW', 'Max MW'], body)}

The 5-minute and 30-minute eras are averaged the same way (time-weighted mean MW), so the
switch to five-minute settlement on 1 Oct 2021 doesn't change what a half-hour means.
"""
    return text, {"expected": expected, "missing": missing, "short": short}


def scale(con) -> tuple[str, dict]:
    energy = {(r, y): e for r, y, e in rows(con, """
        select region, year(day), sum(demand_mw) * 0.5 / 1e6 from hh
        where year(day) between ? and ? group by all""", FIRST_YEAR, LAST_YEAR)}
    peaks = {(r, y): (p, t) for r, y, p, t in rows(con, """
        select region, year(day), max(demand_mw), arg_max(interval_start, demand_mw) from hh
        where year(day) between ? and ? group by all""", FIRST_YEAR, LAST_YEAR)}
    mins = {(r, y): (m, t) for r, y, m, t in rows(con, """
        select region, year(day), min(demand_mw), arg_min(interval_start, demand_mw) from hh
        where year(day) between ? and ? group by all""", FIRST_YEAR, LAST_YEAR)}

    def change(d, r):
        a, b = d[(r, FIRST_YEAR)], d[(r, LAST_YEAR)]
        a, b = (a[0], b[0]) if isinstance(a, tuple) else (a, b)
        return (b - a) / a

    def mw_change(d, r):
        return d[(r, LAST_YEAR)][0] - d[(r, FIRST_YEAR)][0]

    e_body = [[r] + [f"{energy[(r, y)]:.1f}" for y in FULL_YEARS] + [f"{change(energy, r):+.1%}"]
              for r in REGIONS]
    p_body = [[r] + [f"{peaks[(r, y)][0]:,.0f}" for y in FULL_YEARS] + [f"{change(peaks, r):+.1%}"]
              for r in REGIONS]
    # a minimum can cross zero, so its change is shown in MW, not as a percentage
    m_body = [[r] + [f"{mins[(r, y)][0]:,.0f}" for y in FULL_YEARS] + [f"{mw_change(mins, r):+,.0f} MW"]
              for r in REGIONS]
    yrs = [str(y) for y in FULL_YEARS]
    delta = f"{FIRST_YEAR}→{LAST_YEAR}"
    falls = [r for r in REGIONS if mw_change(mins, r) < 0]
    fall_word = "in every region" if len(falls) == len(REGIONS) else f"in {len(falls)} of {len(REGIONS)} regions"
    pct_falls = {r: change(mins, r) for r in REGIONS if mins[(r, LAST_YEAR)][0] > 0}
    biggest = min(pct_falls, key=pct_falls.get)
    below_zero = [(r, y) for r in REGIONS for y in FULL_YEARS if mins[(r, y)][0] < 0]
    if below_zero:
        r0 = below_zero[0][0]
        first_neg = min(y for r, y in below_zero if r == r0)
        zero_note = (f" {r0}'s minimum first went **below zero in {first_neg}** (in this data, which starts in {FIRST_YEAR}) and reached "
                     f"{mins[(r0, LAST_YEAR)][0]:,.0f} MW in {LAST_YEAR}: in those half-hours rooftop solar "
                     "and other small generation inside the region more than covered its own use, "
                     "so the surplus flowed back into the transmission network.")
    else:
        zero_note = ""
    energy_moves = {r: change(energy, r) for r in REGIONS}
    min_slots = {r: mins[(r, LAST_YEAR)][1] for r in REGIONS}
    midday_min = [r for r in REGIONS
                  if MIDDAY[0] <= min_slots[r].hour * 60 + min_slots[r].minute < MIDDAY[1]]
    peak_times = "; ".join(f"{r} {peaks[(r, LAST_YEAR)][1]:%d %b %H:%M}" for r in REGIONS)
    min_times = "; ".join(f"{r} {mins[(r, LAST_YEAR)][1]:%d %b %H:%M}" for r in REGIONS)
    text = f"""## 2. Scale: energy, peaks and minimums by year

Energy (TWh a year, operational demand):

{md_table(['Region'] + yrs + [delta], e_body)}

Highest half-hour (MW):

{md_table(['Region'] + yrs + [delta], p_body)}

Lowest half-hour (MW):

{md_table(['Region'] + yrs + [delta], m_body)}

The minimums move far more than the totals. Annual energy changed by
{min(energy_moves.values()):+.1%} to {max(energy_moves.values()):+.1%} between {FIRST_YEAR} and
{LAST_YEAR}, but the lowest half-hour fell {fall_word}, by as much as
**{abs(pct_falls[biggest]):.0%} in {biggest}** among regions still above zero.{zero_note} In {LAST_YEAR}
the year's lowest half-hour started in the middle of the day (10:00-16:00) in
**{len(midday_min)} of {len(REGIONS)} regions** ({', '.join(midday_min) or 'none'}).
Lowest half-hour, {LAST_YEAR} (start, NEM time): {min_times}.
Highest half-hour, {LAST_YEAR}: {peak_times}.
"""
    return text, {"energy": energy, "peaks": peaks, "mins": mins, "midday_min_regions": midday_min}


def midday(con) -> tuple[str, dict]:
    share = {(r, y): s for r, y, s in rows(con, """
        select region, year, avg((min_slot >= ? and min_slot < ?)::int) from days
        where year between ? and ? group by all""", *MIDDAY, FIRST_YEAR, LAST_YEAR)}
    ratio = {(r, y): v for r, y, v in rows(con, """
        with s as (
            select region, year(day) as year,
                   avg(demand_mw) filter (where slot >= ? and slot < ?) as midday_mw,
                   avg(demand_mw) as all_mw
            from hh where isodow(day) < 6 and year(day) between ? and ? group by all)
        select region, year, midday_mw / all_mw from s""", *MIDDAY, FIRST_YEAR, LAST_YEAR)}
    s_body = [[r] + [f"{share[(r, y)]:.0%}" for y in FULL_YEARS] for r in REGIONS]
    r_body = [[r] + [f"{ratio[(r, y)]:.0%}" for y in FULL_YEARS] for r in REGIONS]
    yrs = [str(y) for y in FULL_YEARS]
    rises = {r: share[(r, LAST_YEAR)] - share[(r, FIRST_YEAR)] for r in REGIONS}
    order = sorted(REGIONS, key=lambda r: -share[(r, LAST_YEAR)])
    lead = order[0]
    text = f"""## 3. The midday trough: rooftop solar is reshaping the day

Operational demand is what the grid supplies, so every rooftop panel that generates at noon
removes demand the grid used to see. Two measures, weekdays and weekends combined unless noted:

Share of days whose lowest half-hour starts between 10:00 and 16:00 (NEM time):

{md_table(['Region'] + yrs, s_body)}

Weekday demand between 10:00 and 16:00 as a share of the weekday all-day average:

{md_table(['Region'] + yrs, r_body)}

In {LAST_YEAR}, **{lead}'s demand bottomed out in the middle of the day on
{share[(lead, LAST_YEAR)]:.0%} of days** (vs {share[(lead, FIRST_YEAR)]:.0%} in {FIRST_YEAR}).
The share rose in {sum(v > 0 for v in rises.values())} of {len(REGIONS)} regions. Ranked by
{LAST_YEAR}: {', '.join(f'{r} {share[(r, LAST_YEAR)]:.0%}' for r in order)}.

![Share of days with a midday minimum, {FIRST_YEAR} vs {LAST_YEAR}](charts/midday_minimum_share.png)

![Average weekday demand by time of day](charts/daily_profile_by_year.png)
"""
    return text, {"share": share, "ratio": ratio, "order": order}


def seasons(con) -> tuple[str, dict]:
    peak_month = {(r, y): m for r, y, m in rows(con, """
        select region, year(day), month(arg_max(day, peak_mw)) from days
        where year(day) between ? and ? group by all""", FIRST_YEAR, LAST_YEAR)}
    season = {(r, s): v for r, s, v in rows(con, """
        select region, case when month(day) in (12, 1, 2) then 'summer'
                            when month(day) in (6, 7, 8) then 'winter' end as season,
               avg(peak_mw) from days
        where year(day) between ? and ? and month(day) in (12, 1, 2, 6, 7, 8)
        group by all""", FIRST_YEAR, LAST_YEAR)}
    weekend = {r: v for r, v in rows(con, """
        select region, avg(mean_mw) filter (where is_weekend) / avg(mean_mw) filter (where not is_weekend)
        from days where year = ? group by region""", LAST_YEAR)}
    hot = (12, 1, 2, 3)
    mainland = [r for r in REGIONS if r != "TAS1"]
    mainland_hot = sum(peak_month[(r, y)] in hot for r in mainland for y in FULL_YEARS)
    mainland_n = len(mainland) * len(FULL_YEARS)
    names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul",
             8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    pm_body = [[r] + [names[peak_month[(r, y)]] for y in FULL_YEARS] for r in REGIONS]
    winter_led = [r for r in REGIONS if season[(r, "winter")] > season[(r, "summer")]]
    s_body = [[r, f"{season[(r, 'summer')]:,.0f}", f"{season[(r, 'winter')]:,.0f}",
               "winter" if r in winter_led else "summer", f"{weekend[r] - 1:+.1%}"]
              for r in REGIONS]
    yrs = [str(y) for y in FULL_YEARS]
    text = f"""## 4. Seasons and weekends

Month of each year's highest half-hour:

{md_table(['Region'] + yrs, pm_body)}

Average daily peak by season ({FIRST_YEAR}-{LAST_YEAR}, summer = Dec-Feb, winter = Jun-Aug) and the
weekend effect ({LAST_YEAR}: weekend average demand vs weekday):

{md_table(['Region', 'Summer daily peak MW', 'Winter daily peak MW', 'Higher season', 'Weekend vs weekday'], s_body)}

Average daily peaks are higher in winter in **{len(winter_led)} of {len(REGIONS)} regions**
({', '.join(winter_led)}). Yet on the mainland the single highest half-hour of the year fell in
Dec-Mar in **{mainland_hot} of {mainland_n} region-years**: winter lifts the typical day, while
summer sets the record. A planner needs both numbers. Weekend demand runs
{1 - max(weekend.values()):.0%}-{1 - min(weekend.values()):.0%} below weekdays, so a forecast that
ignores the day of the week starts out wrong.
"""
    return text, {"peak_month": peak_month, "season": season, "weekend": weekend,
                  "winter_led": winter_led, "mainland_hot": mainland_hot}


def negative_prices(con) -> tuple[str, dict]:
    neg = {(r, y): v for r, y, v in rows(con, """
        select region, year(day), avg((rrp < 0)::int) from hh
        where year(day) between ? and ? group by all""", FIRST_YEAR, LAST_YEAR)}
    at_midday = {r: v for r, v in rows(con, """
        select region, avg((slot >= ? and slot < ?)::int) from hh
        where rrp < 0 and year(day) = ? group by region""", *MIDDAY, LAST_YEAR)}
    body = [[r] + [f"{neg[(r, y)]:.1%}" for y in FULL_YEARS] + [f"{at_midday[r]:.0%}"]
            for r in REGIONS]
    yrs = [str(y) for y in FULL_YEARS]
    top = max(REGIONS, key=lambda r: neg[(r, LAST_YEAR)])
    text = f"""## 5. Context: negative prices

Share of half-hours with a negative average price, and (last column) the share of {LAST_YEAR}'s
negative-price half-hours that start between 10:00 and 16:00, a quarter of the day:

{md_table(['Region'] + yrs + [f'{LAST_YEAR}: at 10:00-16:00'], body)}

In {LAST_YEAR}, {top} had a negative price in {neg[(top, LAST_YEAR)]:.0%} of half-hours, and
{min(at_midday.values()):.0%}-{max(at_midday.values()):.0%} of each region's negative-price
half-hours fell in that midday window. Low midday demand and cheap midday power go together, which
is why the shape of the day, and not only the peak, matters to anyone buying or scheduling around
demand.
"""
    return text, {"neg": neg, "neg_at_midday": at_midday}


# ---------------------------------------------------------------- charts

def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)


def chart_daily_profile(con, path: Path) -> None:
    data = rows(con, """
        select region, year(day), slot, avg(demand_mw) from hh
        where isodow(day) < 6 and year(day) in (2019, 2022, 2025)
        group by all order by all""")
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.6), facecolor=SURFACE)
    for ax, region in zip(axes, REGIONS):
        style(ax)
        for year, color in YEAR_COLOR.items():
            pts = [(s / 60, v) for r, y, s, v in data if r == region and y == year]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=2,
                    label=str(year))
        ax.axvspan(MIDDAY[0] / 60, MIDDAY[1] / 60, color=GRID, alpha=0.45, linewidth=0, zorder=0)
        ax.set_title(REGION_NAMES[region], color=INK, fontsize=9.5, loc="left")
        ax.set_xticks([0, 6, 12, 18, 24], ["0:00", "6:00", "12:00", "18:00", "24:00"])
        ax.set_xlim(0, 24)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes[0].set_ylabel("Average weekday demand, MW", color=INK_2, fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.05, 0.9), ncol=3,
               frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.suptitle("The midday dip deepens: average weekday demand by time of day (NEM time; shaded 10:00-16:00)",
                 x=0.05, ha="left", color=INK, fontsize=11.5, fontweight="bold")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.74, bottom=0.1, wspace=0.28)
    save(fig, path)


def chart_midday_minimum(path: Path, share: dict) -> None:
    order = sorted(REGIONS, key=lambda r: share[(r, LAST_YEAR)])
    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for y, r in enumerate(order):
        a, b = 100 * share[(r, FIRST_YEAR)], 100 * share[(r, LAST_YEAR)]
        ax.plot([a, b], [y, y], color="#c9c8c2", linewidth=2, zorder=1)
        ax.scatter([a], [y], s=52, facecolor=SURFACE, edgecolor=SERIES[0], linewidth=1.8, zorder=2)
        ax.scatter([b], [y], s=64, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(f"{b:.0f}%", (b, y), xytext=(0, 8), textcoords="offset points",
                    ha="center", color=INK, fontsize=8.5)
    ax.set_yticks(range(len(order)), [REGION_NAMES[r] for r in order])
    ax.set_xlim(-3, 103)
    ax.set_ylim(-0.6, len(order) - 0.2)
    ax.set_xlabel("Days whose lowest half-hour starts between 10:00 and 16:00, %",
                  color=INK_2, fontsize=9)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE,
                          markeredgecolor=SERIES[0], markersize=7, label=str(FIRST_YEAR)),
               plt.Line2D([], [], marker="o", linestyle="", color=SERIES[0], markersize=7,
                          label=f"{LAST_YEAR} (labelled)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.2, 0.87), ncol=2,
               frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.suptitle("The daily low has moved from night to midday", x=0.2, ha="left",
                 color=INK, fontsize=11.5, fontweight="bold")
    fig.subplots_adjust(left=0.2, right=0.97, top=0.76, bottom=0.17)
    save(fig, path)


# ---------------------------------------------------------------- main

def build(con) -> tuple[str, dict]:
    meta = json.loads(SNAPSHOT_META.read_text())
    parts, facts = [], {}
    for section in (lambda c: coverage(c, meta), scale, midday, seasons, negative_prices):
        text, f = section(con)
        parts.append(text)
        facts.update(f)
    head = f"""# Source profile: AEMO half-hourly operational demand

*Generated by `scripts/profile.py` from `data/{meta['file']}` (SHA-256 `{meta['sha256'][:12]}…`,
{meta['rows']:,} rows). Do not edit by hand.*

Times are NEM time (AEST, UTC+10, no daylight saving). A half-hour belongs to the day and time it
starts in. Calendar years {FIRST_YEAR}-{LAST_YEAR} are complete; 2026 (Jan-Sep) is left out of
year-on-year tables.
"""
    tail = f"""## What this means for forecasting

- **The data is clean enough to forecast from as is**: no gaps to fill, no conflicting duplicates,
  and one consistent half-hourly grain across the 2021 switch to five-minute settlement.
- **Three cycles to capture**: time of day, day of week (weekends run
  {1 - max(facts['weekend'].values()):.0%}-{1 - min(facts['weekend'].values()):.0%} lower) and season
  (winter-led in {len(facts['winter_led'])} of {len(REGIONS)} regions). A baseline that ignores one of
  them is a straw man.
- **History ages fast**: the midday shape in {LAST_YEAR} isn't the shape of {FIRST_YEAR}, so methods
  that weight recent weeks (seasonal naive, exponential smoothing) should beat long-run averages.
  Part 2 tests that with a rolling-origin backtest rather than assuming it.
- **The minimum is now a planning problem too**: in {len(facts['midday_min_regions'])} of {len(REGIONS)} regions
  the lowest half-hour of {LAST_YEAR} came at midday, so forecast accuracy should be judged at the
  trough as well as the peak.

*{meta['licence']}*
"""
    return "\n".join([head] + parts + [tail]), facts


CHART_FILES = ("daily_profile_by_year.png", "midday_minimum_share.png")


def write_outputs(results: Path) -> dict:
    """Write the report and charts under results/ (tests point this at a temp folder)."""
    con = connect()
    report, facts = build(con)
    charts = results / "charts"
    chart_daily_profile(con, charts / CHART_FILES[0])
    chart_midday_minimum(charts / CHART_FILES[1], facts["share"])
    (results / REPORT.name).write_text(report)
    return facts


def main() -> int:
    write_outputs(RESULTS)
    print(f"wrote {REPORT.relative_to(ROOT)} and {len(CHART_FILES)} charts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
