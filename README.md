# Forecasting electricity demand for operations planning (AEMO, 5 NEM regions)

**Status: in progress, Parts 1-2 of 5 done (data source and profile; baseline backtest).**

How well can **simple, explainable forecasting methods** predict tomorrow's and next week's
half-hourly electricity demand in Australia's National Electricity Market (NEM)? And how much
buffer does a planner need to hold for the error that's left over?

The question matters to anyone who buys, schedules or plans around demand: an energy retailer
hedging next week's load, a network or battery operator choosing when to charge and when to do
maintenance, or a large site deciding when to run flexible equipment. These teams need a forecast
they can **explain and trust**, which is why this project sticks to baselines and classic
smoothing (seasonal naive, exponential smoothing) rather than black-box models, and measures
exactly how good they are.

## Data

| | |
|---|---|
| **Source** | Australian Energy Market Operator (AEMO), [Aggregated price and demand data](https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data) |
| **Licence** | AEMO's [copyright permissions](https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions): "general permission for anyone to use AEMO Material for any purpose", with attribution to AEMO as author |
| **Coverage** | 5 regions (NSW, Queensland, Victoria, South Australia, Tasmania), Jan 2019 - Sep 2026 |
| **Raw files** | 465 monthly CSVs (127 MB), 30-minute rows to Sep 2021, then 5-minute rows; each SHA-256 pinned in [`data/raw_manifest.json`](data/raw_manifest.json) |
| **Committed snapshot** | [`data/demand_halfhourly.csv.gz`](data/demand_halfhourly.csv.gz): 679,200 half-hours (6.7 MB), described in [`data/SNAPSHOT.json`](data/SNAPSHOT.json) |

**Demand** here is AEMO's *operational demand* (`TOTALDEMAND`, MW): what the grid supplies, net
of rooftop solar. **Times** are NEM time (AEST, UTC+10, all year, no daylight saving), stamped
at the end of each interval. The 5-minute rows from Oct 2021 are averaged into half-hours so the
whole series has one grain.

## Part 1: is the data fit to forecast from, and what must a forecast capture?

Full generated report: [`results/SOURCE_PROFILE.md`](results/SOURCE_PROFILE.md).

**The data is complete.** All 465 files match their pinned hashes. Every region has all
135,840 half-hours from Jan 2019 to Sep 2026, with no gaps, no duplicates and no short
half-hours after the switch to five-minute settlement. The forecasts won't need any imputation.

**The shape of the day has changed more than its size.** Annual energy moved by only -5.9% to
+1.8% per region between 2019 and 2025. Over the same years, the year's lowest half-hour fell
in every region (Victoria: 3,058 MW to 1,244 MW, down 59%). South Australia's went **below zero**:
-284 MW in 2025, when rooftop solar and other small generation more than covered the state's own
use. In 2025 the lowest half-hour of the year came at midday in all 5 regions.

![The daily low has moved from night to midday](results/charts/midday_minimum_share.png)

Victoria's daily low now falls at midday on 70% of days, up from 11% in 2019. In South
Australia it's 92%.

![Average weekday demand by time of day, 2019 vs 2022 vs 2025](results/charts/daily_profile_by_year.png)

**Three cycles, two seasons.** Weekend demand runs 3-10% below weekdays. Average daily peaks are
higher in winter in 4 of 5 regions (all but Queensland), but on the mainland the single highest
half-hour of the year fell in Dec-Mar in 26 of 28 region-years.

**So what for the business?**
- A forecast that only targets the evening peak is now half a forecast. The **midday trough** is a
  planning problem too: it's when prices go negative (51-68% of each region's negative-price
  half-hours in 2025 fell between 10:00 and 16:00) and when batteries, flexible loads and
  minimum-demand security limits are being managed.
- Because the shape is drifting year by year, **recent history is worth more than old history**.
  That favours methods that adapt (seasonal naive, exponential smoothing) over long-run averages.
  Part 2 tests that rather than assuming it.
- Any method has to handle time of day, day of week and season at once, so those are the minimum
  bar for every baseline in the backtest.

## Part 2: how good is a forecast anyone can explain?

Full generated report: [`results/BACKTEST.md`](results/BACKTEST.md); every number for every
region, method, lead and window is in [`results/backtest_metrics.csv`](results/backtest_metrics.csv).

**Setup.** A rolling-origin backtest: every day from 1 Oct 2025 to 30 Sep 2026 (365 days, all
seasons) in all 5 regions is forecast 1 to 7 days ahead, from origins at midnight that know nothing
after them. Each method only ever gets the history up to its origin, so it can't peek at the answer.
Four baselines, none using weather:

| Method | Forecast for a target day |
|---|---|
| Naive | the last full day seen ("tomorrow = today") |
| Seasonal naive | the same weekday one week earlier |
| 4-week mean | the same weekday averaged over the last 4 weeks |
| 3-year climatology | the same weekday within ±3 weeks of the date, in each of the previous 3 years |

Accuracy is measured by **WAPE** (total absolute error / total demand), with MAE, bias and daily
peak and trough error in MW alongside. MAPE is left out on purpose: South Australia's demand nears
and crosses zero at midday, which pushes its MAPE to 704-1124% while its WAPE is 13.6-18.7%.

**Day-ahead, yesterday beats last week.** Naive is the best day-ahead baseline in 3 of 5 regions
(NSW 7.1%, Queensland 5.4%, Tasmania 6.4% WAPE); the 4-week mean wins in Victoria (10.0%) and
climatology in South Australia (13.6%). The textbook seasonal naive comes last in 4 of 5, and
swapping last week for yesterday cuts day-ahead error by 13-28%. The likeliest reason is weather:
yesterday's is a far better guide to tomorrow's than last week's.

**Week-ahead, averages win.** Persistence decays fast: from 2 days ahead, naive isn't the best method
in any region. Over a whole week, the 3-year climatology is best in NSW, Queensland and South
Australia, and the 4-week mean in Victoria and Tasmania. Averaging many similar days washes out
one-off weather.

![Forecast error by lead time](results/charts/backtest_error_by_lead.png)

**The error lives at midday.** On the mainland, 40-49% of the best method's day-ahead error falls
between 10:00 and 16:00, a quarter of the day, where half-hours miss by 2.0-2.8× as much as the rest.
South Australia, with the deepest solar dip, is the most midday-heavy (2.8×); Tasmania, with the
shallowest, is the flattest (1.2×). That fits rooftop solar: cloud decides how much demand the grid
sees at noon, and no baseline knows tomorrow's cloud.

![Day-ahead error by time of day](results/charts/backtest_error_by_time_of_day.png)

**Calendar and level effects.** Naive's Monday and Saturday error is 1.2-1.7× its midweek error,
because it forecasts a Monday from a Sunday. Christmas Day is among the three worst days in 4 of 5
regions. Climatology carries any shift in a region's level straight into the forecast: in the latest
year it under-forecast Victoria by 258 MW (5.1% of average demand) and over-forecast Tasmania by
104 MW (10.2%). Re-run on the year before (Oct 2024 - Sep 2025), the best method is the same in 7 of
10 region-horizon pairs, so the ranking is fairly stable but not fixed.

**So what for the business?**
- **Pick the baseline by horizon, not by habit.** A planner scheduling tomorrow should start from
  today's load; one planning next week should start from an average of similar days. The usual
  "same day last week" default is the worst day-ahead choice in 4 of 5 regions and isn't the best at
  either horizon in any region.
- **Size buffers for the midday trough, not only the evening peak.** That's where today's simple
  forecasts miss the most, and where negative prices and minimum-demand limits sit (Part 1).
- **Free wins first.** A day-type rule (yesterday midweek, last week after the weekend changes shape)
  and a holiday calendar target the misses above at no cost to explainability. Part 3 measures how
  much they actually save.
- **The bar for Part 3** is set per region and horizon (table in section 7 of the report): any
  smoothing or weather-adjusted method has to beat, for example, 7.1% day-ahead in NSW and 13.6% in
  South Australia to be worth its extra complexity.

## Plan

| Part | What | Status |
|---|---|---|
| 1 | Pinned AEMO download, half-hourly snapshot, source profile | ✅ done |
| 2 | Rolling-origin backtest harness; naive, seasonal-naive, 4-week-mean and climatology baselines; day-ahead and week-ahead error (WAPE, MAE, bias, peak and trough error) | ✅ done |
| 3 | Exponential smoothing (Holt-Winters), a day-type/holiday rule and a simple, explainable temperature adjustment from free public weather data | next |
| 4 | Business translation: how many MW of buffer cover 90% / 95% of days, which regions and seasons are hardest, recommendations | |
| 5 | CI, TL;DR and final write-up | |

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or requirements-lock.txt for the exact tested versions

python -m scripts.profile              # Part 1: source profile report + charts
python -m scripts.backtest             # Part 2: backtest report, metrics CSV + charts (~20 s)
pytest                                 # 44 tests

# optional: rebuild the snapshot from AEMO's raw files (~127 MB download)
python -m scripts.download_data        # fetches data/raw/, checks every pinned SHA-256
python -m scripts.build_snapshot       # rewrites data/demand_halfhourly.csv.gz byte for byte
```

`download_data --pin` records hashes only for files that have none yet (e.g. after moving
`LAST_MONTH` in `demand/source.py` forward). A file that no longer matches its pinned hash stops
the run instead of silently changing the numbers.

## Tests

- **Parsing and roll-up** (`tests/test_halfhour.py`): hand-worked 5-minute rows average into the
  half-hour they end in; midnight on the 1st belongs to the previous month's file; rows from the
  wrong month or region and wrong headers are rejected; exact duplicates are dropped, conflicting
  ones fail.
- **Snapshot** (`tests/test_snapshot.py`): hash and row count match `SNAPSHOT.json`; every region
  has every half-hour exactly once; the resolution switches on 1 Oct 2021; raw row counts add up;
  a rebuild from the raw files is byte-identical (skipped if they aren't downloaded).
- **Profile** (`tests/test_profile.py`): the committed report and charts match a fresh run, and
  the headline numbers (energy, midday-minimum share, weekend effect, South Australia's negative
  minimum) are recounted independently in pandas.
- **Backtest harness** (`tests/test_backtest.py`): hand-made series with known answers. A series that
  repeats every week is forecast exactly by the weekly methods; on a straight-line trend the naive
  bias is exactly -10 MW per day of lead; each method picks the days it says; replacing everything
  after the origin with garbage doesn't change any forecast (no look-ahead); a short history raises
  instead of wrapping around; every metric is checked on a hand-worked two-day example.
- **Backtest report** (`tests/test_backtest_report.py`): the committed report, CSV and charts match
  a fresh run; seasonal naive, naive at leads 1 and 4, climatology and the 4-week mean are recounted
  with plain pandas shifts on the snapshot (no harness code), and the headline winners follow from the
  metrics. Six deliberate bugs (an off-by-one week, a leaked target day, signed instead of absolute
  demand in WAPE, a slot shifted by one half-hour, a narrower climatology window, peak used in place of
  trough) are each caught.

## Layout

```
demand/source.py          regions, months, file names, paths
demand/halfhour.py        parse a raw file, roll 5-minute rows into half-hours
scripts/download_data.py  fetch + verify the raw files
scripts/build_snapshot.py raw files -> data/demand_halfhourly.csv.gz + SNAPSHOT.json
scripts/profile.py        DuckDB SQL -> results/SOURCE_PROFILE.md + charts
demand/backtest.py        day matrix, the four baselines, rolling-origin forecasts, metrics
scripts/backtest.py       backtest -> results/BACKTEST.md, backtest_metrics.csv + charts
tests/                    pytest
```

*Data: © AEMO, used under AEMO's copyright permissions with attribution.*
