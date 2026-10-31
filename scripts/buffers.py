"""Part 4: what the leftover forecast error means for a planner, in MW and dollars.

    python -m scripts.buffers

Takes the Part 3b weather forecast unchanged. Sets buffers on the months before the holdout and
tests them on it, then sizes the buffers to carry forward on everything since the first summer fit.
Prices the same error at spot and asks whether the forecast can time a battery. Writes
results/BUFFERS.md, results/buffers.csv, results/buffer_coverage.csv and three charts. Every number
in the report's text is computed here, not typed in.
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from demand import backtest as bt  # noqa: E402
from demand import buffers as bf  # noqa: E402
from demand import improved as im  # noqa: E402
from demand import weather as wx  # noqa: E402
from demand.source import REGION_NAMES, REGIONS, ROOT, SNAPSHOT  # noqa: E402
from scripts import weather as part3b  # noqa: E402
from scripts.backtest import WINDOWS, pct  # noqa: E402
from scripts.improve import BOOTSTRAP, HOLDOUT, SETTINGS_CSV, signed  # noqa: E402
from scripts.profile import GRID, INK, INK_2, SURFACE, md_table, save, style  # noqa: E402

RESULTS = ROOT / "results"
CHARTS = RESULTS / "charts"
REPORT = RESULTS / "BUFFERS.md"
BUFFERS_CSV = RESULTS / "buffers.csv"
COVERAGE_CSV = RESULTS / "buffer_coverage.csv"
CHART_FILES = ("buffer_promise_kept.png", "buffer_by_season_and_hour.png", "battery_spread.png")

SET_ON = (part3b.SCORE_FROM, "2025-09-30")  # the Oct-Nov 2024 fits had seen only winter, as in Part 3b
TESTED_ON = WINDOWS[HOLDOUT]
CARRY_FROM = (part3b.SCORE_FROM, TESTED_ON[1])
LEVELS = (.90, .95)
LEVEL = .95  # the level the report's text quotes
SIDES = {"up": bf.UP, "down": bf.DOWN}
HORIZONS = {"day": (1,), "week": tuple(bt.LEADS)}
FORECASTS = ("seasonal_naive", "blend", "weather")
RECIPES = ("halfhour", "flat", "block", "season_block", "shaped", "season_shaped")
ALL = "All"
HIGH_PRICE = 300  # $/MWh, the level of ASX's $300 cap futures
SHARE = .01  # the retailer in the cost table serves 1% of its region's load
BATTERY_MWH = 400  # 100 MW for 4 hours
SEED = 20261031


def load() -> dict:
    """{region: days, demand, price and {forecast: {lead: days x 48}}}."""
    snap = pd.read_csv(SNAPSHOT, usecols=["region", "interval_end", "demand_mw", "rrp"])
    wx_days, raw = wx.load_weather()
    hw = pd.read_csv(SETTINGS_CSV).set_index(["region", "horizon"])
    settings = pd.read_csv(part3b.WEATHER_SETTINGS_CSV).set_index("region")
    out = {}
    for r in REGIONS:
        rows = snap[snap["region"] == r]
        days, m = bt.day_matrix(rows)
        _, price = bt.day_matrix(rows[["interval_end", "rrp"]].rename(columns={"rrp": "demand_mw"}))
        preds, _ = part3b.predictions(days, m, r, part3b.align(raw[r], wx_days, days), hw, False, settings.loc[r])
        last_week = np.vstack([np.full((7, bt.SLOTS), np.nan), m[:-7]])
        out[r] = {"days": days, "m": m, "price": price, "holiday": im.holiday_mask(days, r),
                  "forecast": {"weather": preds["weather"], "blend": preds["blend"],
                               "seasonal_naive": {lead: last_week for lead in bt.LEADS}}}
    return out


def in_window(days: pd.DatetimeIndex, window: tuple) -> np.ndarray:
    return np.flatnonzero((days >= window[0]) & (days <= window[1]))


def misses(d: dict, forecast: str, horizon: str, window: tuple) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Actual minus forecast on the window's days, stacked over the horizon's leads."""
    rows = in_window(d["days"], window)
    leads = HORIZONS[horizon]
    return np.vstack([(d["m"] - d["forecast"][forecast][lead])[rows] for lead in leads]), d["days"][np.tile(rows, len(leads))]


def recipes(past: np.ndarray, past_days, future_days, side: int, level: float) -> dict:
    """{recipe: days x 48 buffer} for the future days, using nothing but the past misses."""
    one = np.array([ALL] * len(past))
    one_future = np.array([ALL] * len(future_days))
    season, season_future = bf.season_of(past_days), bf.season_of(future_days)
    size = (len(future_days), bt.SLOTS)
    return {"halfhour": np.full(size, bf.halfhour_buffer(past, side, level)),
            "flat": np.full(size, bf.flat_buffer(past, side, level)),
            "block": bf.spread_out(bf.block_buffers(past, one, side, level), one_future),
            "season_block": bf.spread_out(bf.block_buffers(past, season, side, level), season_future),
            "shaped": bf.spread_out(bf.stretch(past, one, side, level), one_future),
            "season_shaped": bf.spread_out(bf.stretch(past, season, side, level), season_future)}


def coverage_table(data: dict) -> pd.DataFrame:
    rows = []
    for r in REGIONS:
        for horizon in HORIZONS:
            past, past_days = misses(data[r], "weather", horizon, SET_ON)
            test, test_days = misses(data[r], "weather", horizon, TESTED_ON)
            for side_name, side in SIDES.items():
                for level in LEVELS:
                    for recipe, buffer in recipes(past, past_days, test_days, side, level).items():
                        hit = bf.covered(test, buffer, side)
                        low, high = weekly_interval(hit.astype(float)) if horizon == "day" else (np.nan, np.nan)
                        blocks = [bf.covered(test, buffer, side, b).mean() for b in bf.BLOCKS.values()]
                        by_season = {season: hit[bf.season_of(test_days) == season].mean() for season in bf.SEASONS}
                        rows.append({"horizon": horizon, "region": r, "side": side_name, "level": level, "recipe": recipe,
                                     "mean_buffer_mw": buffer.mean(), "days_covered": hit.mean(), "low": low / len(hit),
                                     "high": high / len(hit), "blocks_covered": np.mean(blocks), "worst_block_covered": min(blocks)} | by_season)
    return pd.DataFrame(rows)


def buffer_table(data: dict) -> pd.DataFrame:
    """Every buffer. `promise` says what it covers: the whole day, or only that block of it."""
    rows = []
    for name, window in (("set_before_holdout", SET_ON), ("carry_forward", CARRY_FROM)):
        for r in REGIONS:
            for horizon in HORIZONS:
                miss, days = misses(data[r], "weather", horizon, window)
                season = bf.season_of(days)
                one = np.array([ALL] * len(miss))
                for side_name, side in SIDES.items():
                    for level in LEVELS:
                        ids = {"window": name, "horizon": horizon, "region": r, "side": side_name, "level": level}
                        shaped = bf.stretch(miss, one, side, level)[ALL]
                        for block_name, block in bf.BLOCKS.items():
                            rows.append(ids | {"season": ALL, "block": block_name, "promise": "whole_day",
                                               "buffer_mw": shaped[block[0]], "days": len(miss)})
                        for season_name in (ALL, *bf.SEASONS):
                            pick = miss if season_name == ALL else miss[season == season_name]
                            rows.append(ids | {"season": season_name, "block": ALL, "promise": "whole_day", "days": len(pick),
                                               "buffer_mw": bf.flat_buffer(pick, side, level)})
                            for block_name, block in bf.BLOCKS.items():
                                rows.append(ids | {"season": season_name, "block": block_name, "promise": "this_block", "days": len(pick),
                                                   "buffer_mw": bf.rank_quantile(bf.worst_miss(pick, side, block), level)})
    return pd.DataFrame(rows)[["window", "horizon", "region", "side", "level", "season", "block", "promise", "buffer_mw", "days"]]


def weekly_interval(values: np.ndarray) -> tuple[float, float]:
    """95% range of a yearly total, from resampling whole weeks."""
    weeks = np.bincount(np.arange(len(values)) // 7, values)
    picks = np.random.default_rng(SEED).integers(0, len(weeks), size=(BOOTSTRAP, len(weeks)))
    totals = weeks[picks].sum(axis=1)
    return float(np.quantile(totals, .025)), float(np.quantile(totals, .975))


def cost_table(data: dict) -> tuple[pd.DataFrame, dict]:
    rows, hedge_at = [], {}
    for r in REGIONS:
        d = data[r]
        keep = in_window(d["days"], TESTED_ON)
        price = d["price"][keep]
        hedge = hedge_at[r] = price.mean()
        day_cost = {}
        for key in FORECASTS:
            miss = (d["m"] - d["forecast"][key][1])[keep]
            cost = bf.imbalance_cost(miss, price, hedge, SHARE)
            day_cost[key] = cost.sum(axis=1)
            week = np.mean([bf.imbalance_cost((d["m"] - d["forecast"][key][lead])[keep], price, hedge, SHARE).sum()
                            for lead in bt.LEADS])
            rows.append({"region": r, "forecast": key, "cost": cost.sum(), "week_cost": week,
                         "high_price_cost": cost[price > HIGH_PRICE].sum(), "negative_price_cost": cost[price < 0].sum(),
                         "high_price_slots": int((price > HIGH_PRICE).sum()),
                         "short_mwh_high_price": SHARE * miss[price > HIGH_PRICE].clip(min=0).sum() * .5,
                         "load_mwh": SHARE * d["m"][keep].sum() * .5})
        saved = day_cost["blend"] - day_cost["weather"]
        low, high = weekly_interval(saved)
        rows[-1] |= {"saved_on_blend": saved.sum(), "saved_low": low, "saved_high": high}
    return pd.DataFrame(rows).set_index(["region", "forecast"]), hedge_at


def battery_rules(d: dict, keep: np.ndarray) -> dict:
    """{rule: (charge starts, discharge starts)} for the days in `keep`, each from what it may know."""
    days, m, price = d["days"], d["m"], d["price"]
    earlier = in_window(days, SET_ON)
    charge, discharge = bf.pick_windows(price[earlier].mean(axis=0, keepdims=True))
    timetable = (np.full(len(keep), charge[0]), np.full(len(keep), discharge[0]))
    return {"timetable": timetable,
            "seasonal_naive": bf.pick_windows(d["forecast"]["seasonal_naive"][1][keep]),
            "blend": bf.pick_windows(d["forecast"]["blend"][1][keep]),
            "weather": bf.pick_windows(d["forecast"]["weather"][1][keep]),
            "yesterday_price": bf.pick_windows(price[keep - 1]),
            "actual_demand": bf.pick_windows(m[keep]),
            "actual_price": bf.pick_windows(price[keep])}


def battery_table(data: dict) -> tuple[pd.DataFrame, dict]:
    rows, slots = [], {}
    for r in REGIONS:
        d = data[r]
        keep = in_window(d["days"], TESTED_ON)
        price = d["price"][keep]
        rules = battery_rules(d, keep)
        slots[r] = (int(rules["timetable"][0][0]), int(rules["timetable"][1][0]))
        earned = {k: bf.captured_spread(price, c, e) for k, (c, e) in rules.items()}
        for k, v in earned.items():
            gap = v - earned["timetable"]
            low, high = weekly_interval(gap * BATTERY_MWH)
            rows.append({"region": r, "rule": k, "spread": v.mean(), "yearly_vs_timetable": gap.sum() * BATTERY_MWH,
                         "low": low, "high": high})
    return pd.DataFrame(rows).set_index(["region", "rule"]), slots


def exceptions(data: dict) -> pd.DataFrame:
    """The holdout days past the flat buffer set beforehand, and what they have in common."""
    rows = []
    for r in REGIONS:
        d = data[r]
        past, _ = misses(d, "weather", "day", SET_ON)
        keep = in_window(d["days"], TESTED_ON)
        test, days = misses(d, "weather", "day", TESTED_ON)
        calendar = d["holiday"][keep] | wx.shutdown_mask(days)
        for side_name, side in SIDES.items():
            buffer = bf.flat_buffer(past, side, LEVEL)
            worst = bf.worst_miss(test, side)
            over = worst > buffer
            rows.append({"region": r, "side": side_name, "buffer": buffer, "past": int(over.sum()),
                         "summer": int((over & (bf.season_of(days) == "Summer")).sum()),
                         "calendar": int((over & calendar).sum()), "worst": worst.max(), "worst_day": days[worst.argmax()]})
    return pd.DataFrame(rows).set_index(["region", "side"])


def portfolio(data: dict) -> dict:
    """The buffer for all 5 regions together against the sum of their own buffers."""
    miss = {r: misses(data[r], "weather", "day", CARRY_FROM)[0] for r in REGIONS}
    together = sum(miss.values())
    return {name: {"sum": sum(bf.flat_buffer(miss[r], side, LEVEL) for r in REGIONS), "together": bf.flat_buffer(together, side, LEVEL)}
            for name, side in SIDES.items()}


def forecast_value(data: dict) -> pd.DataFrame:
    """The same promise on the same days, with each forecast's own misses."""
    rows = []
    for r in REGIONS:
        for key in FORECASTS:
            miss, _ = misses(data[r], key, "day", TESTED_ON)
            rows.append({"region": r, "forecast": key, "up": bf.flat_buffer(miss, bf.UP, LEVEL), "down": bf.flat_buffer(miss, bf.DOWN, LEVEL)})
    return pd.DataFrame(rows).set_index(["region", "forecast"])


def run() -> dict:
    data = load()
    cost, hedge = cost_table(data)
    battery, slots = battery_table(data)
    return {"data": data, "coverage": coverage_table(data), "buffers": buffer_table(data), "cost": cost, "hedge": hedge,
            "battery": battery, "timetable": slots, "exceptions": exceptions(data), "portfolio": portfolio(data),
            "forecast_value": forecast_value(data),
            "mean_demand": {r: data[r]["m"][in_window(data[r]["days"], CARRY_FROM)].mean() for r in REGIONS}}


# ---------------------------------------------------------------- report helpers

def mw(x: float) -> str:
    return f"{x:,.0f}"


def money(x: float) -> str:
    """$ with thousands or millions, for tables of yearly costs."""
    sign = "-" if x < 0 else ""
    x = abs(x)
    return f"{sign}${x / 1e6:.2f}m" if x >= 1e6 else f"{sign}${x / 1e3:,.0f}k"


def span(values, fmt=pct) -> str:
    values = list(values)
    return f"{fmt(min(values))}-{fmt(max(values))}"


def names(regions) -> str:
    return ", ".join(REGION_NAMES[r] for r in regions) or "no region"


def whole(x: float) -> str:
    """A share in whole percent that can't print as -0%."""
    return pct(round(x, 2) + 0, 0)


def time_of(slot: int) -> str:
    return f"{slot // 2:02d}:{30 * (slot % 2):02d}"


def recipe_rows(R: dict, horizon: str = "day", level: float = LEVEL) -> pd.DataFrame:
    c = R["coverage"]
    return c[(c["horizon"] == horizon) & (c["level"] == level)]


def carried(R: dict, horizon: str = "day", level: float = LEVEL, promise: str = "whole_day", season: str = ALL) -> pd.DataFrame:
    b = R["buffers"]
    return b[(b["window"] == "carry_forward") & (b["horizon"] == horizon) & (b["level"] == level)
             & (b["promise"] == promise) & (b["season"] == season)].set_index(["region", "side", "block"])["buffer_mw"]


# ---------------------------------------------------------------- sections

RECIPE_LABEL = {"halfhour": ("Half-hour percentile", "one MW figure: the shortcut, the 95th percentile of all half-hours"),
                "flat": ("Flat", "one MW figure: the 95th percentile of each day's worst half-hour"),
                "block": ("By time of day", "one figure per block, each block's own promise"),
                "season_block": ("By season and time of day", "one figure per season and block, each cell's own promise"),
                "shaped": ("Shaped", "by time of day, stretched by one factor until the whole day meets the promise"),
                "season_shaped": ("Shaped by season", "by season and time of day, stretched the same way")}


def setup_section(R: dict) -> str:
    blocks = ", ".join(bf.BLOCKS)
    return f"""## 1. Setup

The forecast is the Part 3b method, unchanged: blend plus archived weather forecast, issued at midnight. A **miss** is actual
minus forecast, so a positive miss is an under-forecast.

A **buffer** is MW held on one side of the forecast. *Up* guards against under-forecasting (short, buying at spot),
*down* against over-forecasting (long, selling at spot). A **day is covered** if no half-hour in it goes past the buffer. That's
the promise a planner needs, because one bad half-hour is enough to breach a limit or a position.

**Set first, tested after.** Buffers are set on {SET_ON[0]} to {SET_ON[1]} and tested on the holdout, {TESTED_ON[0]} to {TESTED_ON[1]}.
October and November 2024 stay out for the same reason as in Part 3b: those fits had seen only winter. So the setting window holds
only one month of spring, and the seasonal recipes are thin there.

**The buffer is a rank, not an average.** It's the value at rank ceil((n + 1) × level) of the past days, which is one step more
careful than a plain percentile, so a small sample doesn't promise more than it has seen.

Time blocks: {blocks}. Seasons: summer is December to February, and so on. *Week-ahead* pools leads 1 to 7, as in Parts 2 to 3b.

**How the holdout was used.** An early look compared half-hour and day-worst percentiles on both windows. After that the recipes
below were fixed, and all of them are shown. The yesterday's-prices rule in section 8 was added after the first battery comparison,
because that comparison raised the question of what to use instead. No recipe was dropped or changed after being scored.

| Recipe | What it is |
|---|---|
""" + "\n".join(f"| {label} | {text} |" for label, text in (RECIPE_LABEL[k] for k in RECIPES)) + "\n"


def promise_section(R: dict) -> tuple[str, dict]:
    t = recipe_rows(R)
    flat = t[t["recipe"] == "flat"].set_index(["region", "side"])["mean_buffer_mw"]
    body, by_recipe = [], {}
    for k in RECIPES:
        g = t[t["recipe"] == k]
        ratio = g.set_index(["region", "side"])["mean_buffer_mw"] / flat
        by_recipe[k] = {"days": g["days_covered"], "mean": g["days_covered"].mean(), "low": g["days_covered"].min(), "high": g["days_covered"].max(),
                        "size": ratio, "blocks": g["blocks_covered"].mean()}
        body.append([RECIPE_LABEL[k][0], pct(ratio.mean(), 0), pct(g["days_covered"].mean()), span(g["days_covered"]), pct(g["blocks_covered"].mean())])
    shaped = t[t["recipe"] == "shaped"].set_index(["region", "side"])
    plain = t[t["recipe"] == "flat"].set_index(["region", "side"])
    rows = [[REGION_NAMES[r], s, mw(plain.loc[(r, s), "mean_buffer_mw"]), pct(plain.loc[(r, s), "days_covered"]),
             mw(shaped.loc[(r, s), "mean_buffer_mw"]), pct(shaped.loc[(r, s), "days_covered"]),
             f"{pct(shaped.loc[(r, s), 'low'])}-{pct(shaped.loc[(r, s), 'high'])}"] for r in REGIONS for s in SIDES]
    inside = int(((shaped["low"] <= LEVEL) & (shaped["high"] >= LEVEL)).sum())
    saved_mwh = (plain["mean_buffer_mw"] - shaped["mean_buffer_mw"]) * 24
    cut = 1 - by_recipe["shaped"]["size"]
    seasons = t.groupby("recipe")[list(bf.SEASONS)].mean()
    season_body = [[RECIPE_LABEL[k][0]] + [pct(seasons.loc[k, x]) for x in bf.SEASONS] for k in ("flat", "shaped", "season_shaped")]
    size_ratio = 1 / by_recipe["halfhour"]["size"].mean()
    facts = {"recipes": by_recipe, "shaped_cut": cut, "shaped_inside": inside, "saved_mwh": saved_mwh, "season_cover": seasons}
    text = f"""## 2. A buffer set in advance, tested on the next year

Day-ahead, promise {pct(LEVEL, 0)}, over 5 regions and both sides (10 tests per recipe):

{md_table(['Recipe', 'Mean size, flat = 100%', 'Whole days covered (mean)', 'Range', 'Blocks covered (mean)'], body)}

**The half-hour shortcut breaks its promise.** A buffer at the 95th percentile of half-hours covers only {pct(by_recipe['halfhour']['mean'], 0)} of days on average
({span(by_recipe['halfhour']['days'], lambda x: pct(x, 0))}), because a day has 48 chances to miss. Sizing on each day's worst half-hour gives {pct(by_recipe['flat']['mean'])}, with a
buffer {size_ratio:.1f}× as large. That's a little short of the promise, so read it as roughly {pct(by_recipe['flat']['mean'], 0)}, not a guarantee.

**Sizing by time of day works, as long as it's stretched.** Giving each block its own {pct(LEVEL, 0)} keeps that promise for the block ({pct(by_recipe['block']['blocks'])} on
average), but with five blocks the whole day falls to {pct(by_recipe['block']['mean'])}. The shaped recipe fixes that with one stretch factor. It covers {pct(by_recipe['shaped']['mean'])} of days,
level with flat, on a buffer {span(cut, lambda x: pct(x, 0))} smaller across the day, or {mw(saved_mwh.min())}-{mw(saved_mwh.max())} MW-hours less cover to hold per day. It holds more than flat at midday and
much less overnight.

**The all-year promise is uneven.** Whole days covered, by season of the holdout (mean of the 10 tests):

{md_table(['Recipe', *bf.SEASONS], season_body)}

An all-year buffer averages close to its promise, but it runs thin in summer ({pct(seasons.loc['shaped', 'Summer'])} for shaped) and thick in autumn and winter. Shaped by season evens that out at
{pct(by_recipe['season_shaped']['size'].mean(), 0)} of flat's size, with {pct(by_recipe['season_shaped']['mean'])} overall. Its weak spot is spring, at {pct(seasons.loc['season_shaped', 'Spring'])}, the one season it learned from only 30 days of. So the seasonal recipe isn't
proven yet, though it's closer than the headline suggests.

![Whole days covered, by recipe](charts/{CHART_FILES[0]})

Flat and shaped by region, set before the holdout and tested on it. The interval is a 95% range from resampling whole weeks:

{md_table(['Region', 'Side', 'Flat, MW', 'Days covered', 'Shaped, mean MW', 'Days covered', '95% interval'], rows)}

The {pct(LEVEL, 0)} promise sits inside the interval in {inside} of 10 tests of the shaped recipe.
"""
    return text, facts


def carry_section(R: dict) -> tuple[str, dict]:
    shaped = carried(R)
    flat = {(h, lv): carried(R, h, lv).xs(ALL, level="block") for h in HORIZONS for lv in LEVELS}
    body = [[REGION_NAMES[r], s] + [mw(shaped[(r, s, b)]) for b in bf.BLOCKS] + [mw(flat[("day", LEVEL)][(r, s)])] for r in REGIONS for s in SIDES]
    summary = [[REGION_NAMES[r], s] + [mw(flat[(h, lv)][(r, s)]) for h in HORIZONS for lv in LEVELS] + [pct(flat[("day", LEVEL)][(r, s)] / R["mean_demand"][r], 0)]
               for r in REGIONS for s in SIDES]
    growth = {(r, s): flat[("week", LEVEL)][(r, s)] / flat[("day", LEVEL)][(r, s)] - 1 for r in REGIONS for s in SIDES}
    lopsided = {r: flat[("day", LEVEL)][(r, "up")] / flat[("day", LEVEL)][(r, "down")] for r in REGIONS}
    midday = {(r, s): shaped[(r, s, "10:00-16:00")] / shaped[(r, s, "00:00-06:00")] for r in REGIONS for s in SIDES}
    mainland = [k for k in midday if k[0] != "TAS1"]
    text = f"""## 3. What to carry forward

Set on all {R['buffers'].query("window == 'carry_forward' and horizon == 'day'")['days'].max()} days from {CARRY_FROM[0]} to {CARRY_FROM[1]}, now that the recipe has passed its test. These have no later year to be tested on.
Day-ahead, shaped, {pct(LEVEL, 0)} of whole days, MW either side of the forecast:

{md_table(['Region', 'Side', *bf.BLOCKS, 'Flat'], body)}

On the mainland the midday buffer is {span((midday[k] for k in mainland), lambda x: f'{x:.1f}×')} the overnight one. In Tasmania it's {span((midday[k] for k in midday if k[0] == 'TAS1'), lambda x: f'{x:.1f}×')}.

Flat buffers at both levels, for both horizons. *Share* is the day-ahead {pct(LEVEL, 0)} buffer as a share of the region's mean demand:

{md_table(['Region', 'Side', 'Day-ahead 90%', 'Day-ahead 95%', 'Week-ahead 90%', 'Week-ahead 95%', 'Share'], summary)}

Looking a week out adds {span(growth.values(), lambda x: pct(x, 0))} to the {pct(LEVEL, 0)} buffer. Under-forecasts are the larger side in New South Wales, Victoria and South Australia
({span((lopsided[r] for r in ('NSW1', 'VIC1', 'SA1')), lambda x: f'{x:.2f}×')} the over-forecast buffer), and the two sides are nearly equal in Queensland and Tasmania ({span((lopsided[r] for r in ('QLD1', 'TAS1')), lambda x: f'{x:.2f}×')}).
Every buffer, including the seasonal ones and the 90% level, is in [`buffers.csv`](buffers.csv).
"""
    return text, {"midday_over_night": midday, "week_growth": growth, "up_over_down": lopsided, "flat_carry": flat[("day", LEVEL)]}


def season_section(R: dict) -> tuple[str, dict]:
    body, hardest = [], {}
    for r in REGIONS:
        for s in SIDES:
            cells = {(season, b): carried(R, promise="this_block", season=season)[(r, s, b)] for season in bf.SEASONS for b in bf.BLOCKS}
            key = max(cells, key=cells.get)
            hardest[(r, s)] = key
            body.append([REGION_NAMES[r], s, f"{key[0]}, {key[1]}", mw(cells[key]), pct(cells[key] / R["mean_demand"][r], 0)])
    middays = sum(k[1] == "10:00-16:00" for k in hardest.values())
    b = R["buffers"]
    season_days = b[(b["window"] == "carry_forward") & (b["horizon"] == "day")].groupby("season")["days"].first().drop(ALL).to_dict()
    return f"""## 4. Which seasons and hours are hardest?

Each cell below is that season and block's own {pct(LEVEL, 0)} day-ahead buffer, on the days from {CARRY_FROM[0]} on, as a share of the region's mean demand so
regions can be compared. It maps where the risk sits. It isn't a whole-day promise, and section 2 shows that a seasonal buffer has yet to prove itself.

![Buffers by season and time of day](charts/{CHART_FILES[1]})

The hardest cell for each region and side:

{md_table(['Region', 'Side', 'Hardest cell', 'Buffer, MW', 'Share of mean demand'], body)}

The hardest cell is the 10:00-16:00 block in {middays} of 10 cases. Spring rests on {season_days['Spring']} days, the other seasons on {min(v for k, v in season_days.items() if k != 'Spring')}-{max(season_days.values())}.
""", {"hardest": hardest, "hardest_midday": middays, "spring_days": season_days["Spring"]}


def value_section(R: dict) -> tuple[str, dict]:
    v = R["forecast_value"]
    body, cuts = [], {}
    for r in REGIONS:
        cuts[r] = {s: 1 - v.loc[(r, "weather"), s] / v.loc[(r, "seasonal_naive"), s] for s in SIDES}
        cuts[r]["blend"] = {s: 1 - v.loc[(r, "weather"), s] / v.loc[(r, "blend"), s] for s in SIDES}
        body.append([REGION_NAMES[r]] + [f"{mw(v.loc[(r, k), 'up'])} / {mw(v.loc[(r, k), 'down'])}" for k in FORECASTS])
    naive_cut = [cuts[r][s] for r in REGIONS for s in SIDES]
    blend_cut = [cuts[r]["blend"][s] for r in REGIONS for s in SIDES]
    return f"""## 5. What the better forecast saved

The same flat {pct(LEVEL, 0)} promise on the holdout's own days, changing only the forecast (up / down, MW):

{md_table(['Region', 'Same day last week', 'Blend (Part 3)', 'With weather (Part 3b)'], body)}

Against the usual "same day last week", the weather method cuts the buffer by {span(naive_cut, lambda x: pct(x, 0))}. Against the blend it cuts {span(blend_cut, lambda x: pct(x, 0))}. These figures are hindsight, for comparing
forecasts: each is the buffer that would have fitted that year.
""", {"cut_on_naive": naive_cut, "cut_on_blend": blend_cut}


def exception_section(R: dict) -> tuple[str, dict]:
    e = R["exceptions"]
    body = [[REGION_NAMES[r], s, mw(e.loc[(r, s), "buffer"]), e.loc[(r, s), "past"], e.loc[(r, s), "summer"], e.loc[(r, s), "calendar"],
             mw(e.loc[(r, s), "worst"]), f"{e.loc[(r, s), 'worst_day']:%a %d %b %Y}"] for r in REGIONS for s in SIDES]
    summer_days = (bf.season_of(pd.date_range(*TESTED_ON)) == "Summer").mean()
    past, summer, calendar = int(e["past"].sum()), int(e["summer"].sum()), int(e["calendar"].sum())
    ratio = e["worst"] / e["buffer"]
    p = R["portfolio"]
    joint = {s: 1 - p[s]["together"] / p[s]["sum"] for s in SIDES}
    # share of the year's days that are holidays or 27-31 December, averaged over the regions
    year = {r: in_window(R["data"][r]["days"], TESTED_ON) for r in REGIONS}
    calendar_share = np.mean([(R["data"][r]["holiday"][year[r]] | wx.shutdown_mask(R["data"][r]["days"][year[r]])).mean() for r in REGIONS])
    return f"""## 6. When a buffer fails, and how far

Every holdout day past the flat {pct(LEVEL, 0)} buffer set beforehand:

{md_table(['Region', 'Side', 'Buffer, MW', 'Days past it', 'In summer', 'Holiday or 27-31 Dec', 'Worst miss, MW', 'On'], body)}

**A buffer is a limit on how often, not how far.** The worst miss of the year was {span(ratio, lambda x: f'{x:.1f}×')} the buffer. A planner still needs a plan for those days.

**Most breaches fall on ordinary days.** Holidays and 27-31 December are {calendar} of {past} breaches ({pct(calendar / past, 0)}), against {pct(calendar_share, 0)} of the year's days.
They're the worst days in percentage terms (Part 3b) but not the main reason a buffer is breached. Summer brings {pct(summer / past, 0)} of breaches, on {pct(summer_days, 0)} of days.

**Regions don't all miss at once.** Across the 5 regions, the sum of their own buffers is {mw(p['up']['sum'])} MW up and {mw(p['down']['sum'])} MW down. A single buffer on their combined miss, for the same
promise, needs {mw(p['up']['together'])} and {mw(p['down']['together'])} MW: {pct(joint['up'], 0)} and {pct(joint['down'], 0)} less. That's an upper limit for a multi-region book, since it ignores interconnector limits.
""", {"exceptions_past": past, "exceptions_calendar": calendar, "exceptions_summer": summer, "calendar_share": calendar_share, "worst_ratio": ratio, "joint_cut": joint}


def cost_section(R: dict) -> tuple[str, dict]:
    c = R["cost"]
    wx_rows = c.xs("weather", level="forecast")
    per_mwh = {k: c.xs(k, level="forecast")["cost"] / wx_rows["load_mwh"] for k in FORECASTS}
    body = [[REGION_NAMES[r], money(c.loc[(r, "seasonal_naive"), "cost"]), money(c.loc[(r, "blend"), "cost"]), money(c.loc[(r, "weather"), "cost"]),
             f"${per_mwh['weather'][r]:.2f}", money(wx_rows.loc[r, "saved_on_blend"]), f"{money(wx_rows.loc[r, 'saved_low'])} to {money(wx_rows.loc[r, 'saved_high'])}"] for r in REGIONS]
    cut = {r: 1 - c.loc[(r, "weather"), "cost"] / c.loc[(r, "blend"), "cost"] for r in REGIONS}
    high = {r: wx_rows.loc[r, "high_price_cost"] / wx_rows.loc[r, "cost"] for r in REGIONS}
    negative = {r: wx_rows.loc[r, "negative_price_cost"] / wx_rows.loc[r, "cost"] for r in REGIONS}
    split = [[REGION_NAMES[r], wx_rows.loc[r, "high_price_slots"], whole(high[r]), whole(negative[r]), whole(1 - high[r] - negative[r])] for r in REGIONS]
    year_slots = len(pd.date_range(*TESTED_ON)) * bt.SLOTS
    wape = pd.read_csv(part3b.METRICS_CSV)
    wape = wape[(wape["window"] == "latest") & (wape["lead"] == "1")].set_index(["region", "method"])["wape"].astype(float)
    wape_cut = {r: 1 - wape[(r, "weather")] / wape[(r, "blend")] for r in REGIONS}
    faster = [r for r in REGIONS if cut[r] > wape_cut[r]]
    clear = [r for r in REGIONS if wx_rows.loc[r, "saved_low"] > 0]
    week_up = {r: wx_rows.loc[r, "week_cost"] / wx_rows.loc[r, "cost"] - 1 for r in REGIONS}
    return f"""## 7. What the error costs at spot

A retailer hedges its forecast at a fixed price and settles the difference at spot. Compared with a perfect forecast, each half-hour costs
*miss × (spot price - hedge price) × half an hour*, so being short into a high price costs money, and so does being long into a low one.
This takes a retailer with {pct(SHARE, 0)} of its region's load, whose misses are {pct(SHARE, 0)} of the region's, hedged at the year's mean spot price. Day-ahead, holdout year:

{md_table(['Region', 'Same day last week', 'Blend', 'With weather', 'With weather, per MWh of load', 'Saved by weather on the blend', '95% interval'], body)}

**The cost falls faster than the error.** Weather cuts this cost by {span(cut.values(), lambda x: pct(x, 0))} on the blend, more than its WAPE cut in {len(faster)} of 5 regions ({span(wape_cut.values(), lambda x: pct(x, 0))}),
so the misses it removes were worth more than the average miss. The saving is clear of zero in {len(clear)} of 5 regions ({names(clear)}). The intervals are wide, because a few half-hours decide the year.

Where the with-weather cost sits:

{md_table(['Region', f'Half-hours above ${HIGH_PRICE}', f'Cost in those half-hours', 'Cost at negative prices', 'Cost at all other prices'], split)}

**The right cover differs by region.** ${HIGH_PRICE}/MWh is the level of ASX's $300 cap futures, so it's used here as the line for a high price. In New South Wales and South Australia the half-hours above it hold {whole(high['NSW1'])} and {whole(high['SA1'])} of the
cost, and in New South Wales that's just {wx_rows.loc['NSW1', 'high_price_slots']} of {year_slots:,} half-hours, so a cap is aimed at the right place. In Queensland and Victoria they hold {whole(high['QLD1'])} and {whole(high['VIC1'])}, while negative-price half-hours hold
{whole(negative['QLD1'])} and {whole(negative['VIC1'])}. A cap pays on high prices, so it doesn't touch those.

Week-ahead, the with-weather cost is {span(week_up.values(), lambda x: pct(x, 0))} higher than day-ahead.
""", {"cost_cut": cut, "wape_cut": wape_cut, "cost_faster": faster, "cost_clear": clear, "high_share": high, "negative_share": negative, "per_mwh": per_mwh["weather"]}


RULE_LABEL = {"timetable": "Fixed timetable", "seasonal_naive": "Same day last week", "blend": "Blend", "weather": "With weather",
              "yesterday_price": "Yesterday's prices", "actual_demand": "Actual demand", "actual_price": "Actual price"}


def battery_section(R: dict) -> tuple[str, dict]:
    b = R["battery"]
    spread = b["spread"].unstack("rule")
    body = [[REGION_NAMES[r]] + [f"${spread.loc[r, k]:,.0f}" for k in RULE_LABEL] for r in REGIONS]
    times = [[REGION_NAMES[r], f"{time_of(c)}-{time_of(c + bf.WINDOW_SLOTS)}", f"{time_of(d)}-{time_of(d + bf.WINDOW_SLOTS)}"] for r, (c, d) in R["timetable"].items()]
    gap = [[REGION_NAMES[r]] + [f"{money(b.loc[(r, k), 'yearly_vs_timetable'])} ({money(b.loc[(r, k), 'low'])} to {money(b.loc[(r, k), 'high'])})"
                                for k in ("weather", "actual_demand", "actual_price")] for r in REGIONS]
    verdict = {}
    for k in ("weather", "blend", "yesterday_price", "actual_demand", "actual_price"):
        g = b.xs(k, level="rule")
        verdict[k] = {"better": [r for r in REGIONS if g.loc[r, "low"] > 0], "worse": [r for r in REGIONS if g.loc[r, "high"] < 0]}
    capture = {r: spread.loc[r, "timetable"] / spread.loc[r, "actual_price"] for r in REGIONS}
    tie = [r for r in REGIONS if r not in verdict["weather"]["better"] + verdict["weather"]["worse"]]
    blend_wins = [r for r in REGIONS if spread.loc[r, "blend"] > spread.loc[r, "weather"]]
    closed = {r: (spread.loc[r, "actual_demand"] - spread.loc[r, "timetable"]) / (spread.loc[r, "actual_price"] - spread.loc[r, "timetable"]) for r in REGIONS}
    text = f"""## 8. Can the forecast time a battery?

A {BATTERY_MWH // 4} MW / {BATTERY_MWH} MWh battery charges for 4 hours and discharges for 4 hours once a day, committed at midnight. Each rule picks the two windows and earns that day's actual prices. The
figures are the mean **spread** in $/MWh, discharge price minus charge price, before losses, for a price-taker. Demand stands in for price: charge when demand looks lowest, discharge when it looks highest.

The fixed timetable is the best pair of windows on average prices before the holdout, used every day:

{md_table(['Region', 'Charge', 'Discharge'], times)}

Holdout spreads. The last two columns know the day's actuals, so they're ceilings, not results:

{md_table(['Region', *RULE_LABEL.values()], body)}

Yearly gain on the fixed timetable for that battery, with a 95% interval from resampling whole weeks:

{md_table(['Region', 'With weather', 'Actual demand (ceiling)', 'Actual price (ceiling)'], gap)}

![Battery spread against a fixed timetable](charts/{CHART_FILES[2]})

**Outside South Australia, a demand forecast doesn't beat a timetable.** With weather, it's clearly ahead only in {names(verdict['weather']['better'])}, clearly behind in {names(verdict['weather']['worse'])}, and can't be told apart in
{names(tie)}. Even knowing the day's demand exactly is clearly better only in {names(verdict['actual_demand']['better'])}. A more accurate forecast didn't reliably help either: the blend out-earned the with-weather forecast in {names(blend_wins)}.

**The gap is in prices, not demand.** The timetable already earns {span(capture.values(), lambda x: pct(x, 0))} of the actual-price ceiling, and knowing the day's demand closes only {span(closed.values(), lambda x: pct(x, 0))} of what's left.
That ceiling shows the prize, not what a real price forecast would earn. The simple price rule tested here falls short: copying yesterday's prices is clearly behind the timetable in {len(verdict['yesterday_price']['worse'])} of 5 regions.
"""
    return text, {"battery_verdict": verdict, "timetable_capture": capture, "battery_tie": tie, "blend_wins": blend_wins, "demand_closes": closed}


def tail(R: dict, f: dict) -> str:
    recipes_ = f["recipes"]
    shaped_cut = f["shaped_cut"]
    return f"""## What this means for operations planning

- **Size a buffer on each day's worst half-hour.** A half-hour percentile called {pct(LEVEL, 0)} covered {pct(recipes_['halfhour']['mean'], 0)} of days.
- **Carry the shaped buffer.** It kept the promise as well as a flat one ({pct(recipes_['shaped']['mean'])} against {pct(recipes_['flat']['mean'])}) and held {span(shaped_cut, lambda x: pct(x, 0))} less. The table is in section 3.
- **Add cover in summer, and retest seasonal buffers.** The all-year buffer covered {pct(f['season_cover'].loc['shaped', 'Summer'], 0)} of summer days and {pct(f['season_cover'].loc['shaped', 'Winter'], 0)} of winter ones. Seasonal buffers even that out, but they aren't proven until spring has more than 30 days behind it. The seasonal figures in `buffers.csv` now have {f['spring_days']}.
- **Plan for the breach.** The worst miss of the year was {span(f['worst_ratio'], lambda x: f'{x:.1f}×')} the buffer, so a buffer needs a fallback.
- **Net buffers across regions where the book allows.** A combined buffer was {pct(f['joint_cut']['down'], 0)}-{pct(f['joint_cut']['up'], 0)} smaller than the sum of the regional ones.
- **Match the cover to the region.** A ${HIGH_PRICE} cap is aimed at most of the cost in New South Wales and South Australia. In Queensland and Victoria {whole(f['negative_share']['VIC1'])}-{whole(f['negative_share']['QLD1'])} sits in negative-price half-hours, which a cap doesn't touch.
- **Keep a battery on a timetable unless a price forecast beats it.** A better demand forecast clearly gained only in {names(f['battery_verdict']['weather']['better'])}. The bigger prize is in prices, so that's the forecast to test next.

## Limits

- One setting period and one test year. The intervals cover the weeks within that year, not other years, and rooftop solar keeps growing.
- The setting period holds only September of spring, so the seasonal recipes are thin there. The carry-forward table has no later year to be tested on.
- The early look described in section 1 means the holdout isn't fully unseen, and Parts 2 to 3b had already reported on the same year.
- These are region-wide misses. A retailer's own error differs from its region's, and a large regional surprise is partly what moves the price, so section 7 overstates how
  closely a small retailer's misses line up with price. Read it as a yardstick for comparing forecasts, not as a bill.
- The hedge is a flat swap at the year's mean spot price, known only in hindsight. Real books are shaped, hedged months ahead, and carry caps, fees and losses. Day-ahead is the last step of that process.
- The battery is a price-taker with one cycle a day, no losses, no degradation, no ancillary-service income and no second cycle for the morning peak. Those would change the dollars,
  and could change the ranking of the rules. Real operators re-bid through the day rather than commit at midnight.
- The timetable is one fixed pair per region, chosen on mean prices, which a few spikes can sway.
- A few half-hours decide each year's dollar figures, so another year could look quite different. That's why the intervals are wide.
- The forecast inherits Part 3b's limits, including its free non-commercial weather source.

*Sources: Australian Energy Market Operator (AEMO), Aggregated price and demand data, used under AEMO's copyright
permissions (any purpose, with attribution to AEMO as author). Public holidays: the `holidays` Python package (MIT);
checked against the Australian Public Holidays Dates Machine Readable Dataset, data.gov.au, CC BY 3.0 AU.
[Weather data by Open-Meteo.com](https://open-meteo.com/), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
hourly values moved onto half-hours and turned into degrees and differences; underlying model: ECMWF IFS, CC BY 4.0.*
"""


# ---------------------------------------------------------------- charts

RECIPE_COLOR = {"halfhour": "#eb6834", "flat": "#2a78d6", "shaped": "#1baf7a"}  # dataviz slots 2, 1, 3
RULE_COLOR = {"weather": "#2a78d6", "actual_demand": "#eb6834", "actual_price": "#1baf7a"}  # slots 1-3, validated all-pairs
HEAT = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#104281"])  # dataviz sequential blue


def chart_promise(R: dict, path: Path) -> None:
    t = recipe_rows(R).set_index(["recipe", "region", "side"])["days_covered"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), facecolor=SURFACE, sharey=True)
    y = np.arange(len(REGIONS))[::-1]
    for ax, side, title in ((axes[0], "up", "Guarding against an under-forecast"), (axes[1], "down", "Guarding against an over-forecast")):
        style(ax)
        ax.grid(False)
        ax.xaxis.grid(True, color=GRID, linewidth=.8)
        ax.set_axisbelow(True)
        ax.axvline(100 * LEVEL, color=INK_2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        for key, label in (("halfhour", "Half-hour percentile"), ("flat", "Flat, on each day's worst half-hour"), ("shaped", "Shaped by time of day")):
            values = [100 * t[(key, r, side)] for r in REGIONS]
            ax.scatter(values, y, s=70, color=RECIPE_COLOR[key], edgecolor=SURFACE, linewidth=1.2, zorder=3, label=label)
            if key == "halfhour":
                for v, yi in zip(values, y):
                    ax.text(v - 1.3, yi, f"{v:.0f}%", ha="right", va="center", color=INK_2, fontsize=8.5)
        ax.set_xlim(45, 100)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Whole days covered, Oct 2025 - Sep 2026 (dashed: the 95% promise)", color=INK_2, fontsize=8.5)
    axes[0].set_yticks(y, [REGION_NAMES[r] for r in REGIONS])
    axes[0].set_ylim(-.6, len(REGIONS) - .4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.125, .925), ncol=3, frameon=False,
               fontsize=8.5, labelcolor=INK_2, handletextpad=.2)
    fig.suptitle("A 95% half-hour buffer covers about two-thirds of days. A day-worst buffer gets close to 95%",
                 x=.125, ha="left", color=INK, fontsize=11.5, fontweight="bold")
    fig.subplots_adjust(left=.125, right=.985, top=.77, bottom=.14, wspace=.08)
    save(fig, path)


def chart_heat(R: dict, path: Path) -> None:
    cells = {(season, r, s, b): carried(R, promise="this_block", season=season)[(r, s, b)] / R["mean_demand"][r]
             for season in bf.SEASONS for r in REGIONS for s in SIDES for b in bf.BLOCKS}
    top = 100 * max(cells.values())
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.2), facecolor=SURFACE, sharex=True, sharey=True)
    for row, (side, name) in enumerate((("up", "Under-forecast"), ("down", "Over-forecast"))):
        for ax, r in zip(axes[row], REGIONS):
            grid = np.array([[100 * cells[(season, r, side, b)] for b in bf.BLOCKS] for season in bf.SEASONS])
            ax.imshow(grid, cmap=HEAT, vmin=0, vmax=top, aspect="auto")
            for (i, j), v in np.ndenumerate(grid):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5, color=SURFACE if v > .6 * top else INK)
            ax.set_xticks(range(len(bf.BLOCKS)), [b[:2] + "-" + b[6:8] for b in bf.BLOCKS], fontsize=7.5, color=INK_2)
            ax.set_yticks(range(len(bf.SEASONS)), list(bf.SEASONS), fontsize=8, color=INK_2)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks(np.arange(-.5, len(bf.BLOCKS)), minor=True)
            ax.set_yticks(np.arange(-.5, len(bf.SEASONS)), minor=True)
            ax.grid(which="minor", color=SURFACE, linewidth=2)
            ax.tick_params(which="minor", length=0)
            ax.set_title(REGION_NAMES[r] if row == 0 else "", color=INK, fontsize=9.5, loc="left")
        axes[row][0].set_ylabel(name, color=INK_2, fontsize=9)
    fig.suptitle("Day-ahead 95% buffer as a percentage of each region's mean demand, by season and hours of the day",
                 x=.06, ha="left", color=INK, fontsize=11.5, fontweight="bold")
    fig.subplots_adjust(left=.06, right=.99, top=.87, bottom=.07, wspace=.07, hspace=.09)
    save(fig, path)


def chart_battery(R: dict, path: Path) -> None:
    spread = R["battery"]["spread"].unstack("rule")
    fig, ax = plt.subplots(figsize=(10.5, 3.9), facecolor=SURFACE)
    style(ax)
    ax.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    y = np.arange(len(REGIONS))[::-1]
    ax.axvline(0, color=INK_2, linewidth=1, zorder=1)
    for key, label in (("weather", "With weather forecast"), ("actual_demand", "Knowing the day's actual demand"), ("actual_price", "Knowing the day's actual prices")):
        gap = [spread.loc[r, key] - spread.loc[r, "timetable"] for r in REGIONS]
        ax.scatter(gap, y, s=70, color=RULE_COLOR[key], edgecolor=SURFACE, linewidth=1.2, zorder=3, label=label)
        if key == "actual_price":
            for g, yi in zip(gap, y):
                ax.text(g + 1, yi, f"+\\${g:.0f}", ha="left", va="center", color=INK_2, fontsize=8.5)
    ax.set_yticks(y, [REGION_NAMES[r] for r in REGIONS])
    ax.set_ylim(-.6, len(REGIONS) - .4)
    ax.set_xlim(-10, 65)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{'+' if v > 0 else '-' if v < 0 else ''}\\${abs(v):.0f}"))
    ax.set_xlabel(r"Daily spread against the fixed timetable (line at \$0), \$/MWh, Oct 2025 - Sep 2026", color=INK_2, fontsize=8.5)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.125, .925), ncol=3, frameon=False,
               fontsize=8.5, labelcolor=INK_2, handletextpad=.2)
    fig.suptitle("A battery on a fixed timetable: a demand forecast adds little, and prices are where the gap is",
                 x=.125, ha="left", color=INK, fontsize=11.5, fontweight="bold")
    fig.subplots_adjust(left=.125, right=.985, top=.77, bottom=.14)
    save(fig, path)


# ---------------------------------------------------------------- main

def build(R: dict) -> tuple[str, dict]:
    facts: dict = {}
    parts = [setup_section(R)]
    for fn in (promise_section, carry_section, season_section, value_section, exception_section, cost_section, battery_section):
        text, f = fn(R)
        parts.append(text)
        facts.update(f)
    head = f"""# From forecast error to MW of buffer, dollars and a battery timetable

*Generated by `scripts/buffers.py` from `data/{SNAPSHOT.name}`, `data/public_holidays.csv`, `data/{wx.WEATHER.name}` and the settings
Parts 3 and 3b chose. Do not edit by hand. Every buffer is in [`buffers.csv`](buffers.csv) and every test in [`buffer_coverage.csv`](buffer_coverage.csv).*
"""
    return "\n".join([head] + parts + [tail(R, facts)]), facts


def write_outputs(results: Path, R: dict | None = None) -> dict:
    """Write the report, tables and charts under results/ (tests point this at a temp folder)."""
    R = run() if R is None else R
    report, facts = build(R)
    charts = results / "charts"
    chart_promise(R, charts / CHART_FILES[0])
    chart_heat(R, charts / CHART_FILES[1])
    chart_battery(R, charts / CHART_FILES[2])
    table = R["buffers"].assign(level=lambda d: (100 * d["level"]).round().astype(int)).rename(columns={"level": "level_pct"})
    table.to_csv(results / BUFFERS_CSV.name, index=False, float_format="%.1f", lineterminator="\n")
    R["coverage"].to_csv(results / COVERAGE_CSV.name, index=False, float_format="%.4f", lineterminator="\n")
    (results / REPORT.name).write_text(report)
    return facts | R


def main() -> int:
    write_outputs(RESULTS)
    print(f"wrote {REPORT.relative_to(ROOT)}, {BUFFERS_CSV.relative_to(ROOT)}, {COVERAGE_CSV.relative_to(ROOT)} "
          f"and {len(CHART_FILES)} charts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
