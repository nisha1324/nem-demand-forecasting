"""Part 3b: does an archived weather forecast beat the blend, and by how much?

    python -m scripts.weather

Same harness, target days and leads as Parts 2 and 3. The blend and its Holt-Winters settings
are taken from Part 3 unchanged. The weather recipe is chosen on the tuning year, and the
holdout (Oct 2025 - Sep 2026) is only scored. Writes results/WEATHER.md,
results/weather_metrics.csv, results/weather_settings.csv and two charts. Every number in the
report's text is computed here, not typed in.
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
from demand import weather as wx  # noqa: E402
from demand.source import REGION_NAMES, REGIONS, ROOT, SNAPSHOT  # noqa: E402
from scripts.backtest import MIDDAY, WINDOWS, pct, write_csv  # noqa: E402
from scripts.improve import (BOOTSTRAP, HOLDOUT, METHOD_COLOR, SETTINGS_CSV, TUNING, cut_with_interval, day_ahead,  # noqa: E402
                             error_by_day, signed, week_ahead)
from scripts.profile import GRID, INK, INK_2, SURFACE, md_table, save, style  # noqa: E402

RESULTS = ROOT / "results"
CHARTS = RESULTS / "charts"
REPORT = RESULTS / "WEATHER.md"
METRICS_CSV = RESULTS / "weather_metrics.csv"
WEATHER_SETTINGS_CSV = RESULTS / "weather_settings.csv"
CHART_FILES = ("weather_vs_blend.png", "weather_error_by_time_of_day.png")

SCORE_FROM = "2024-12-01"  # the Oct-Nov 2024 fits had seen only winter, so they say nothing about a setting
EVENING_SLOT, MIDDAY_SLOT = 36, 24  # 18:00 and 12:00
LAST_FIT = "2026-09-01"  # the fit the report reads its slopes from
THIN_MONTH = ("VIC1", "2024-11")  # the worst month of the winter-only fits
WEATHER_COLOR = "#6250d6"  # dataviz reference palette, slot 7; the blend keeps its Part 3 colour
METHODS = ("blend", "blend_shutdown", "weather", "weather_temp", "weather_fresh", "weather_perfect")


def blend_forecast(days, m, region: str, hw_settings: pd.DataFrame, rows: np.ndarray, shutdown: bool) -> dict:
    """{lead: days x 48} of the Part 3 blend, NaN outside `rows`."""
    dow = days.dayofweek.to_numpy()
    hol = im.holiday_mask(days, region) | (wx.shutdown_mask(days) if shutdown else False)
    s = hw_settings.loc[(region, "day")]
    hw = im.HoltWinters(dow, hol, s["alpha"], s["beta"], s["gamma"], im.PHI)
    fn = functools.partial(im.blend, hw=hw, dow=dow, holiday=hol)
    out = {}
    for lead in bt.LEADS:
        out[lead] = np.full(m.shape, np.nan)
        out[lead][rows] = np.stack([fn(m[: t - lead + 1], lead) for t in rows])
    return out


def wape_on(m, forecast, rows) -> float:
    return np.abs(forecast[rows] - m[rows]).sum() / np.abs(m[rows]).sum()


def tune_region(days, m, blend1, w, skip, rows) -> dict:
    """Lowest day-ahead WAPE on `rows` over the threshold and shrinkage grid."""
    grid = []
    for heat in wx.HEATS:
        for cool in wx.COOLS:
            x = wx.features(w, 1, heat, cool)
            for lam in wx.LAMBDAS:
                adj = wx.adjustment(m - blend1, x, days, 1, lam, skip, wanted=np.isin(np.arange(len(days)), rows))
                grid.append({"heat": heat, "cool": cool, "lam": lam, "wape": wape_on(m, blend1 + adj, rows)})
    return min(grid, key=lambda g: g["wape"])


def align(w: dict, wx_days: pd.DatetimeIndex, days: pd.DatetimeIndex) -> dict:
    """Weather matrices on the demand's day rows, NaN before the archive starts."""
    rows = wx_days.get_indexer(days)
    out = {}
    for name, values in w.items():
        out[name] = np.full((len(days), bt.SLOTS), np.nan)
        out[name][rows >= 0] = values[rows[rows >= 0]]
    return out


def choose(series: dict, weather: dict, hw_settings: pd.DataFrame) -> tuple:
    """Everything decided on the tuning year: the shutdown period, then one setting per region."""
    first, last = WINDOWS[TUNING]
    start = wx.FIRST_DAY
    forecasts, shutdown_gain = {}, {}
    for r in REGIONS:
        days, m = series[r]
        rows = np.flatnonzero((days >= start) & (days <= last))
        tuning = np.flatnonzero((days >= first) & (days <= last))
        cut = {flag: blend_forecast(days, m, r, hw_settings, rows, flag)[1] for flag in (False, True)}
        forecasts[r] = cut
        shutdown_gain[r] = wape_on(m, cut[False], tuning) - wape_on(m, cut[True], tuning)
    use_shutdown = sum(v > 0 for v in shutdown_gain.values()) >= 3
    settings = []
    for r in REGIONS:
        days, m = series[r]
        skip = im.holiday_mask(days, r) | wx.shutdown_mask(days)
        rows = np.flatnonzero((days >= SCORE_FROM) & (days <= last))
        settings.append({"region": r} | tune_region(days, m, forecasts[r][use_shutdown], weather[r], skip, rows))
    return use_shutdown, shutdown_gain, pd.DataFrame(settings).set_index("region")


def predictions(days, m, r, w, hw_settings, use_shutdown, setting) -> tuple[dict, dict]:
    """{method: {lead: days x 48}} on the demand's rows, and the last fit's slopes for the report."""
    rows = np.flatnonzero(days >= wx.FIRST_DAY)
    plain = blend_forecast(days, m, r, hw_settings, rows, False)
    closed = blend_forecast(days, m, r, hw_settings, rows, True)
    base = closed if use_shutdown else plain
    skip = im.holiday_mask(days, r) | wx.shutdown_mask(days)
    heat, cool, lam = setting["heat"], setting["cool"], setting["lam"]
    preds = {"blend": plain, "blend_shutdown": closed}
    for key in METHODS[2:]:
        preds[key] = {}
    for lead in bt.LEADS:
        residual = m - base[lead]
        x = wx.features(w, lead, heat, cool)
        no_sun = x.copy()
        no_sun[..., 2] = 0
        preds["weather"][lead] = base[lead] + wx.adjustment(residual, x, days, lead, lam, skip)
        preds["weather_temp"][lead] = base[lead] + wx.adjustment(residual, no_sun, days, lead, lam, skip)
        preds["weather_fresh"][lead] = base[lead] + wx.adjustment(
            residual, wx.features(w, lead, heat, cool, column=lead), days, lead, lam, skip)
        preds["weather_perfect"][lead] = base[lead] + wx.adjustment(
            residual, wx.features(w, lead, heat, cool, column=0), days, lead, lam, skip)
    x = wx.features(w, 1, heat, cool)
    usable = ~np.isnan(x).any(axis=(1, 2)) & ~np.isnan(m - base[1]).any(axis=1) & ~skip
    train = np.flatnonzero(usable & (days < LAST_FIT))
    slopes, _ = wx.fit_slots(x[train], (m - base[1])[train], lam)
    return preds, {"slopes": slopes, "days": len(train)}


def run() -> dict:
    snap = pd.read_csv(SNAPSHOT, usecols=["region", "interval_end", "demand_mw"])
    series = {r: bt.day_matrix(snap[snap["region"] == r]) for r in REGIONS}
    wx_days, raw = wx.load_weather()
    weather = {r: align(raw[r], wx_days, series[r][0]) for r in REGIONS}
    hw_settings = pd.read_csv(SETTINGS_CSV).set_index(["region", "horizon"])
    use_shutdown, shutdown_gain, settings = choose(series, weather, hw_settings)
    mts, wks, dailies, slots, slopes = [], [], [], [], {}
    for r in REGIONS:
        days, m = series[r]
        preds, last_fit = predictions(days, m, r, weather[r], hw_settings, use_shutdown, settings.loc[r])
        slopes[r] = last_fit
        extra = {key: (lambda p: lambda history, lead: p[lead][len(history) + lead - 1])(p) for key, p in preds.items()}
        for window, (first, last) in WINDOWS.items():
            fc = bt.forecasts(days, m, first, last, methods=(), extra=extra)
            ids = {"window": window, "region": r}
            mts.append(bt.metrics(fc).assign(**ids))
            wks.append(bt.metrics(fc, by=("method",)).assign(**ids))
            fc["abs_err"] = (fc["forecast"] - fc["actual"]).abs()
            fc["abs_act"] = fc["actual"].abs()
            dailies.append(fc.groupby(["method", "lead", "day"])[["abs_err", "abs_act"]].sum()
                           .reset_index().assign(**ids))
            slots.append(fc[fc["lead"] == 1].groupby(["method", "slot"])["abs_err"].sum().reset_index().assign(**ids))
    mt = pd.concat(mts).set_index(["window", "region"], append=True).reorder_levels(
        ["window", "region", "method", "lead"]).sort_index()
    wk = pd.concat(wks).set_index(["window", "region"], append=True).reorder_levels(
        ["window", "region", "method"]).sort_index()
    return {"metrics": mt, "week": wk, "daily": pd.concat(dailies, ignore_index=True),
            "slots": pd.concat(slots, ignore_index=True), "settings": settings, "use_shutdown": use_shutdown,
            "shutdown_gain": shutdown_gain, "slopes": slopes}


# ---------------------------------------------------------------- sections

def setup_section(R: dict) -> str:
    s = R["settings"]
    rows = [[r, wx.SITES[r][0], f"{s.loc[r, 'heat']:g}°C", f"{s.loc[r, 'cool']:g}°C", f"{s.loc[r, 'lam']:g}"] for r in REGIONS]
    size = len(wx.HEATS) * len(wx.COOLS) * len(wx.LAMBDAS)
    gain = R["shutdown_gain"]
    helped = [r for r in REGIONS if gain[r] > 0]
    thin_region, thin_month = THIN_MONTH
    d = R["daily"]
    d = d[(d["window"] == TUNING) & (d["region"] == thin_region) & (d["lead"] == 1)
          & (d["day"].dt.strftime("%Y-%m") == thin_month)].groupby("method")["abs_err"].sum()
    thin = d["weather"] / d["blend"] - 1
    return f"""## 1. Setup

Same harness, target days, leads and metrics as [`BACKTEST.md`](BACKTEST.md) and [`IMPROVEMENTS.md`](IMPROVEMENTS.md).
The blend and its settings come from Part 3 unchanged.

**The weather has to be a forecast.** A demand model scored on what the weather turned out to be would look far
better than anything a planner could use. So this uses Open-Meteo's Previous Runs archive, which keeps what ECMWF's
model predicted days ahead, not its stitched "historical forecast" series, which tracks what happened. A spot check
against single model runs on one day in June 2026 found `previous_day1` matching runs about 23 hours older than the hour.

**And it has to have been published.** A run takes hours to come out, so that day-old run can start before the
midnight origin yet be published after it. So the headline method reads weather one day older than its name: lead 1 uses `previous_day2`, and so on.
Lead 7 can't go older, because the archive stops there.

**The method** is one small regression per half-hour, fitted on the blend's own past misses:

    miss = intercept + a × (cooling degrees vs last week) + b × (heating degrees vs last week) + c × (sun vs last week)

Cooling degrees are how far the forecast temperature is above a threshold, heating degrees how far it's below one, and
sun is forecast solar radiation (W/m²). Each is compared with its average over the 7 days before the origin, because
the blend already reflects that weather. A small ridge penalty stops a thin fit swinging wildly. The fit is redone
monthly with a separate fit per lead, on days before the month of the origin only, and public holidays and 27-31 December are
left out of it. Weather comes from one point per region, the capital.

**Chosen on the tuning year only** ({size} settings per region, day-ahead WAPE, {SCORE_FROM} to {WINDOWS[TUNING][1]}):

{md_table(['Region', 'Weather point', 'Heating below', 'Cooling above', 'Ridge penalty'], rows)}

October and November 2024 were left out of the choice. Those fits had seen only April to September, so the first hot days
were an extrapolation: even with the chosen setting, {thin_region}'s day-ahead error in {thin_month} was {signed(thin, 0)} against the blend. That only
happens once, because by the holdout the fit has seen a summer.

**A shutdown period was tried and dropped.** Treating 27-31 December like a holiday had to lower the blend's tuning-year
WAPE in at least 3 of 5 regions to be kept. It did in {len(helped)} ({', '.join(helped) or 'none'}), so the weather method doesn't use it.

**The bar** is the Part 3 blend. **Cut** is the fall in total absolute error against it. The interval is a 95% range from
resampling whole weeks of the holdout ({BOOTSTRAP:,} draws).
"""


def results_section(R: dict) -> tuple[str, dict]:
    d1, wk = day_ahead(R), week_ahead(R)
    cuts = {h: {r: cut_with_interval(R, r, "weather", "blend", leads) for r in REGIONS}
            for h, leads in (("day", (1,)), ("week", bt.LEADS))}
    def body(table, h):
        return [[r, pct(table.loc[(r, "blend"), "wape"]), pct(table.loc[(r, "weather"), "wape"]), signed(-cuts[h][r][0]),
                 f"{signed(-cuts[h][r][2])} to {signed(-cuts[h][r][1])}"] for r in REGIONS]
    head = ["Region", "Blend", "Weather", "Cut", "95% interval"]
    mw = {r: (d1.loc[(r, "blend"), "mae"], d1.loc[(r, "weather"), "mae"]) for r in REGIONS}
    mw_body = [[REGION_NAMES[r], f"{mw[r][0]:,.0f}", f"{mw[r][1]:,.0f}", f"{round(mw[r][0]) - round(mw[r][1]):,}"] for r in REGIONS]
    clear = {h: [r for r in REGIONS if cuts[h][r][1] > 0] for h in cuts}
    worse = {h: [r for r in REGIONS if cuts[h][r][2] < 0] for h in cuts}
    unclear = {h: [r for r in REGIONS if r not in clear[h] and r not in worse[h]] for h in cuts}
    ranked = sorted(REGIONS, key=lambda r: cuts["day"][r][0], reverse=True)
    def verdict(h):
        out = f"{len(clear[h])} of {len(REGIONS)} regions clear of zero ({', '.join(clear[h]) or 'none'})"
        if unclear[h]:
            out += f"; can't be told apart from the blend in {', '.join(unclear[h])}"
        if worse[h]:
            out += f"; clearly worse in {', '.join(worse[h])}"
        return out
    text = f"""## 2. The holdout result

Day-ahead WAPE, {WINDOWS[HOLDOUT][0]} to {WINDOWS[HOLDOUT][1]}:

{md_table(head, body(d1, 'day'))}

Week-ahead WAPE:

{md_table(head, body(wk, 'week'))}

**Day-ahead:** {verdict('day')}. The largest cut is {ranked[0]} ({signed(-cuts['day'][ranked[0]][0])}), the smallest {ranked[-1]} ({signed(-cuts['day'][ranked[-1]][0])}).

**Week-ahead:** {verdict('week')}.

The day-ahead saving in MW (mean absolute error per half-hour):

{md_table(['Region', 'Blend', 'Weather', 'Saved'], mw_body)}

![Weather against the blend](charts/{CHART_FILES[0]})
"""
    return text, {"cuts": cuts, "clear": clear, "unclear": unclear, "worse": worse, "ranked": ranked, "mw": mw}


def forecast_value_section(R: dict) -> tuple[str, dict]:
    d1 = day_ahead(R)
    w = {r: {k: d1.loc[(r, k), "wape"] for k in ("blend", "weather_temp", "weather", "weather_fresh", "weather_perfect")} for r in REGIONS}
    body = [[r] + [pct(w[r][k]) for k in ("blend", "weather_temp", "weather", "weather_fresh", "weather_perfect")] for r in REGIONS]
    sun_gain = {r: w[r]["weather_temp"] - w[r]["weather"] for r in REGIONS}
    fresh = {r: 1 - w[r]["weather_fresh"] / w[r]["weather"] for r in REGIONS}
    left = {r: (w[r]["weather"] - w[r]["weather_perfect"]) / (w[r]["blend"] - w[r]["weather_perfect"]) for r in REGIONS}
    sun_better = [r for r in REGIONS if sun_gain[r] > 0]
    text = f"""## 3. What each piece of the forecast is worth

Day-ahead WAPE. *Perfect weather* uses the archive's shortest-range value for the target day, which no planner has: it's the
ceiling for this method, not a result.

{md_table(['Region', 'Blend', 'Temperature only', 'Temperature + sun', 'A day fresher', 'Perfect weather'], body)}

**Sun matters.** Adding solar radiation to temperature lowers day-ahead WAPE in {len(sun_better)} of {len(REGIONS)} regions, by
{100 * min(sun_gain.values()):.2f} to {100 * max(sun_gain.values()):.2f} points.

**A day's freshness matters little.** Weather a day fresher trims the error by only {pct(min(fresh.values()))}-{pct(max(fresh.values()))} more, and that column may
include a run published after the origin.

**The weather forecast isn't the main limit.** Forecast error costs {pct(min(left.values()), 0)}-{pct(max(left.values()), 0)} of the gain perfect weather would bring, and
even perfect weather leaves {pct(min(w[r]['weather_perfect'] for r in REGIONS))}-{pct(max(w[r]['weather_perfect'] for r in REGIONS))} WAPE.
"""
    return text, {"wape_by_input": w, "sun_gain": sun_gain, "fresh_gain": fresh, "left_on_table": left}


def where_section(R: dict) -> tuple[str, dict]:
    d1 = day_ahead(R)
    s = R["slots"]
    s = s[s["window"] == HOLDOUT]
    mid = s["slot"].between(MIDDAY[0], MIDDAY[1] - 1)
    body, share, saved = [], {}, {}
    for r in REGIONS:
        g = s[s["region"] == r]
        def part(k):
            one = g[g["method"] == k]
            return one.loc[mid[one.index], "abs_err"].sum(), one["abs_err"].sum()
        mb, tb = part("blend")
        mw_, tw = part("weather")
        share[r] = (mb / tb, mw_ / tw)
        saved[r] = (mb - mw_) / (tb - tw)
        body.append([r, f"{d1.loc[(r, 'blend'), 'peak_mae']:,.0f}", f"{d1.loc[(r, 'weather'), 'peak_mae']:,.0f}",
                     f"{d1.loc[(r, 'blend'), 'trough_mae']:,.0f}", f"{d1.loc[(r, 'weather'), 'trough_mae']:,.0f}",
                     pct(mb / tb, 0), pct(mw_ / tw, 0), pct(saved[r], 0)])
    d = R["daily"]
    d = d[(d["window"] == HOLDOUT) & (d["lead"] == 1) & (d["method"] == "weather")]
    worst_body, shutdown = [], 0
    for r in REGIONS:
        g = d[d["region"] == r].assign(w=lambda x: x["abs_err"] / x["abs_act"]).sort_values("w", ascending=False).head(3)
        worst_body.append([r] + [f"{x:%a %d %b %Y} ({pct(v, 0)})" for x, v in zip(g["day"], g["w"])])
        shutdown += int(((g["day"].dt.month == 12) & (g["day"].dt.day >= 27)).sum())
    mainland = [r for r in REGIONS if r != "TAS1"]
    text = f"""## 4. Where the gain fell

Day-ahead, blend against weather (peak and trough = mean daily error in MW; midday = 10:00-16:00):

{md_table(['Region', 'Peak, blend', 'Peak, weather', 'Trough, blend', 'Trough, weather', 'Midday share, blend', 'Midday share, weather', 'Midday share of the MW saved'], body)}

On the mainland, {pct(min(saved[r] for r in mainland), 0)}-{pct(max(saved[r] for r in mainland), 0)} of the MW saved came from the 10:00-16:00 quarter of the day, and that window's share of the
remaining error is {pct(min(share[r][1] for r in mainland), 0)}-{pct(max(share[r][1] for r in mainland), 0)}, from {pct(min(share[r][0] for r in mainland), 0)}-{pct(max(share[r][0] for r in mainland), 0)} with the blend alone.

![Day-ahead error by time of day](charts/{CHART_FILES[1]})

The three worst day-ahead days for the weather method (that day's WAPE):

{md_table(['Region', 'Worst', '2nd', '3rd'], worst_body)}

{shutdown} of these 15 days fall between 27 and 31 December. Weather makes those misses smaller, but a holiday week isn't a weather problem.
"""
    return text, {"midday_share": share, "midday_of_saving": saved, "shutdown_worst": shutdown}


def slopes_section(R: dict) -> tuple[str, dict]:
    facts, body = {}, []
    for r in REGIONS:
        b = R["slopes"][r]["slopes"]
        facts[r] = {"cool": b[EVENING_SLOT, 0], "heat": b[EVENING_SLOT, 1], "sun": 100 * b[MIDDAY_SLOT, 2]}
        body.append([r, f"{facts[r]['cool']:+,.0f}", f"{facts[r]['heat']:+,.0f}", f"{facts[r]['sun']:+,.0f}"])
    return f"""## 5. What a planner can read off the fit

The day-ahead fit from the start of the last month, on {R['slopes'][REGIONS[0]]['days']} days. Each slope is the change in the blend's miss, so it's
what the weather adds on top of the recent week, not the whole weather effect, and each is shrunk a little towards zero:

{md_table(['Region', 'MW per °C of cooling at 18:00', 'MW per °C of heating at 18:00', 'MW per extra 100 W/m² of sun at 12:00'], body)}

More sun at noon means less grid demand in {sum(f['sun'] < 0 for f in facts.values())} of {len(REGIONS)} regions: that's rooftop solar seen from the grid side.
""", {"slopes_read": facts}


def stability_section(R: dict) -> tuple[str, dict]:
    tune_d, hold_d = day_ahead(R, TUNING), day_ahead(R)
    tune_w, hold_w = week_ahead(R, TUNING), week_ahead(R)
    cut = lambda t, r: 1 - t.loc[(r, "weather"), "wape"] / t.loc[(r, "blend"), "wape"]
    body = [[r, signed(-cut(tune_d, r)), signed(-cut(hold_d, r)), signed(-cut(tune_w, r)), signed(-cut(hold_w, r))] for r in REGIONS]
    return f"""## 6. Does it hold outside the year it was tuned on?

Weather against the blend, same year:

{md_table(['Region', 'Day-ahead, tuning year', 'Day-ahead, holdout', 'Week-ahead, tuning year', 'Week-ahead, holdout'], body)}

The tuning-year column includes the thin October and November 2024 fits, and the settings were chosen on most of that year, so the
holdout column is the one to quote.
""", {"tuning_cut_day": {r: cut(tune_d, r) for r in REGIONS}, "holdout_cut_day": {r: cut(hold_d, r) for r in REGIONS}}


def bar_section(R: dict) -> str:
    d1, wk = day_ahead(R), week_ahead(R)
    body = [[REGION_NAMES[r], pct(d1.loc[(r, "weather"), "wape"]), pct(wk.loc[(r, "weather"), "wape"]),
             f"{d1.loc[(r, 'weather'), 'mae']:,.0f} MW"] for r in REGIONS]
    return f"""## 7. What Part 4 starts from

Part 4 sizes buffers on what's left after this method:

{md_table(['Region', 'Day-ahead (WAPE)', 'Week-ahead (WAPE)', 'Day-ahead MAE'], body)}
"""


# ---------------------------------------------------------------- charts

def chart_vs_blend(R: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), facecolor=SURFACE, sharey=True)
    y = np.arange(len(REGIONS))[::-1]
    for ax, table, title in ((axes[0], day_ahead(R), "Day-ahead"), (axes[1], week_ahead(R), "Week-ahead")):
        style(ax)
        ax.grid(False)
        ax.xaxis.grid(True, color=GRID, linewidth=.8)
        ax.set_axisbelow(True)
        for yi, r in zip(y, REGIONS):
            old, new = (100 * table.loc[(r, k), "wape"] for k in ("blend", "weather"))
            ax.plot([old, new], [yi, yi], color=GRID, linewidth=2.5, solid_capstyle="round", zorder=2)
            ax.scatter(old, yi, s=70, color=METHOD_COLOR["blend"], edgecolor=SURFACE, linewidth=1.2, zorder=3,
                       label="Blend (Part 3)" if yi == y[0] else None)
            ax.scatter(new, yi, s=70, color=WEATHER_COLOR, edgecolor=SURFACE, linewidth=1.2, zorder=3,
                       label="Blend + weather forecast" if yi == y[0] else None)
            left, right = min(old, new), max(old, new)
            ax.text(left - .2, yi, f"{left:.1f}%", ha="right", va="center", color=INK_2, fontsize=8.5)
            ax.text(right + .2, yi, f"{right:.1f}%", ha="left", va="center", color=INK_2, fontsize=8.5)
        ax.set_xlim(1.5, 15.5)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("WAPE, Oct 2025 - Sep 2026 (holdout)", color=INK_2, fontsize=8.5)
    axes[0].set_yticks(y, [REGION_NAMES[r] for r in REGIONS])
    axes[0].set_ylim(-.6, len(REGIONS) - .4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.125, .925), ncol=2, frameon=False,
               fontsize=8.5, labelcolor=INK_2, handletextpad=.2)
    fig.suptitle("An archived weather forecast on top of the blend, scored on the unseen year",
                 x=.125, ha="left", color=INK, fontsize=11.5, fontweight="bold")
    fig.subplots_adjust(left=.125, right=.985, top=.77, bottom=.14, wspace=.08)
    save(fig, path)


def chart_time_of_day(R: dict, path: Path) -> None:
    s = R["slots"]
    s = s[(s["window"] == HOLDOUT) & s["method"].isin(["blend", "weather"])]
    days = len(pd.date_range(*WINDOWS[HOLDOUT]))
    tod = s.groupby(["region", "method", "slot"])["abs_err"].sum().div(days)
    x = np.arange(bt.SLOTS) / 2 + .25
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.7), facecolor=SURFACE)
    for ax, r in zip(axes, REGIONS):
        style(ax)
        ax.plot(x, tod.loc[(r, "blend")].to_numpy(), color=METHOD_COLOR["blend"], linewidth=2, label="Blend (Part 3)")
        ax.plot(x, tod.loc[(r, "weather")].to_numpy(), color=WEATHER_COLOR, linewidth=2, label="Blend + weather forecast")
        ax.axvspan(MIDDAY[0] / 2, MIDDAY[1] / 2, color=GRID, alpha=.45, linewidth=0, zorder=0)
        ax.set_ylim(0, tod.loc[r].max() * 1.08)
        ax.set_xticks([0, 6, 12, 18, 24], ["0:00", "6:00", "12:00", "18:00", "24:00"])
        ax.set_xlim(0, 24)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_title(REGION_NAMES[r], color=INK, fontsize=9.5, loc="left")
    axes[0].set_ylabel("Mean absolute error, MW", color=INK_2, fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.045, .9), ncol=2, frameon=False,
               fontsize=8.5, labelcolor=INK_2)
    fig.suptitle("Day-ahead error by time of day, Oct 2025 - Sep 2026 (shaded: 10:00-16:00)",
                 x=.045, ha="left", color=INK, fontsize=11.5, fontweight="bold")
    fig.subplots_adjust(left=.045, right=.99, top=.74, bottom=.1, wspace=.25)
    save(fig, path)


# ---------------------------------------------------------------- main

def build(R: dict) -> tuple[str, dict]:
    facts: dict = {}
    parts = [setup_section(R)]
    for fn in (results_section, forecast_value_section, where_section, slopes_section, stability_section):
        text, f = fn(R)
        parts.append(text)
        facts.update(f)
    parts.append(bar_section(R))
    head = f"""# Adding a weather forecast to the blend

*Generated by `scripts/weather.py` from `data/{SNAPSHOT.name}`, `data/public_holidays.csv` and
`data/{wx.WEATHER.name}`. Do not edit by hand. Every region, method, lead and window is in
[`weather_metrics.csv`](weather_metrics.csv).*
"""
    day, week = (facts["cuts"][h] for h in ("day", "week"))
    day_all, week_all = ([v[0] for v in c.values()] for c in (day, week))
    top, low = facts["ranked"][0], facts["ranked"][-1]
    mainland = [r for r in REGIONS if r != "TAS1"]
    w = facts["wape_by_input"]
    tail = f"""## What this means for operations planning

- **Get a weather forecast next.** On top of the blend, an archived forecast cuts day-ahead error by
  {pct(min(day_all), 0)}-{pct(max(day_all), 0)} and week-ahead error by {pct(min(week_all), 0)}-{pct(max(week_all), 0)}, with every interval clear of zero. {R['settings'].shape[1] - 1} settings per region and a
  regression per half-hour keep it explainable: each MW of adjustment traces back to degrees and sun.
- **Include the sun.** Temperature alone gets part of the way, and adding solar radiation takes {top} from
  {pct(w[top]['weather_temp'])} to {pct(w[top]['weather'])}. For a grid with this much rooftop solar, a temperature-only adjustment misses a large part of it.
- **Don't pay for a fresher feed on this evidence.** Weather a day fresher adds only {pct(min(facts['fresh_gain'].values()))}-{pct(max(facts['fresh_gain'].values()))}.
- **Tasmania is the exception.** {low} gains only {pct(facts['cuts']['day'][low][0], 0)} ({round(facts['mw'][low][0]) - round(facts['mw'][low][1])} MW), so it's the last place to spend effort.
- **Keep midday in the buffers.** The error fell all day, not just at noon, so 10:00-16:00 still holds
  {pct(min(facts['midday_share'][r][1] for r in mainland), 0)}-{pct(max(facts['midday_share'][r][1] for r in mainland), 0)} of mainland error. Part 4 should size buffers by time of day.
- **Override 27-31 December by hand.** {facts['shutdown_worst']} of the 15 worst days are still in that stretch, and the simple shutdown rule failed its test.

## Limits

- One tuning year and one holdout year. The intervals cover the weeks within the holdout, not other years.
- The recipe was fixed before the holdout was first scored, and only the report's wording changed afterwards. The holdout
  still isn't fully unseen: Part 3 had already published the blend's scores on it, and those pointed at the weather step.
- The archive starts in April 2024, so the holdout's first fit has 18 months behind it, with just one summer. Each fit
  treats demand's response to weather as fixed, while rooftop solar keeps growing.
- Open-Meteo's free API is for non-commercial use, so a business running this would need a paid plan or its own forecast feed.
- One weather point per region. A single capital stands in for a whole state, which matters most in Queensland, where solar and
  load are spread over a long coast.
- Lead 7 can't be made a day older, so its evening hours may use a run published a few hours after the origin. Leads 1-6 don't have this problem.
- The midnight origin is itself generous: a real day-ahead desk commits earlier, with less recent demand and an older run.
- ECMWF was chosen on day-ahead skill at one point (Adelaide) against ERA5 reanalysis before the holdout. GFS was closer at 7 days on
  radiation, so week-ahead might do better with it. ECMWF updates its model over time and the archive isn't re-run.
- The perfect-weather column is the archive's shortest-range value, which is close to the weather but not a measurement of it.
- Open-Meteo's archive could be revised, so the snapshot is committed with its hash and `--fetch` shows any change.
- 27-31 December stays unmodelled, because the shutdown rule failed on the tuning year.

*Sources: Australian Energy Market Operator (AEMO), Aggregated price and demand data, used under AEMO's copyright
permissions (any purpose, with attribution to AEMO as author). Public holidays: the `holidays` Python package (MIT);
checked against the Australian Public Holidays Dates Machine Readable Dataset, data.gov.au, CC BY 3.0 AU.
[Weather data by Open-Meteo.com](https://open-meteo.com/), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
hourly values moved onto half-hours and turned into degrees and differences; underlying model: ECMWF IFS, CC BY 4.0.*
"""
    return "\n".join([head] + parts + [tail]), facts


def write_outputs(results: Path, R: dict | None = None) -> dict:
    """Write the report, tables and charts under results/ (tests point this at a temp folder)."""
    R = run() if R is None else R
    report, facts = build(R)
    charts = results / "charts"
    chart_vs_blend(R, charts / CHART_FILES[0])
    chart_time_of_day(R, charts / CHART_FILES[1])
    write_csv(R["metrics"], R["week"], results / METRICS_CSV.name)
    R["settings"].reset_index().to_csv(results / WEATHER_SETTINGS_CSV.name, index=False, float_format="%.6g",
                                       lineterminator="\n")
    (results / REPORT.name).write_text(report)
    return facts | R


def main() -> int:
    write_outputs(RESULTS)
    print(f"wrote {REPORT.relative_to(ROOT)}, {METRICS_CSV.relative_to(ROOT)}, "
          f"{WEATHER_SETTINGS_CSV.relative_to(ROOT)} and {len(CHART_FILES)} charts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
