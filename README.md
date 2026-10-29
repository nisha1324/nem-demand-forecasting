# Forecasting electricity demand for operations planning (AEMO, 5 NEM regions)

**Status: in progress, Part 1 of 5 done (data source, snapshot and profile).**

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

## Plan

| Part | What | Status |
|---|---|---|
| 1 | Pinned AEMO download, half-hourly snapshot, source profile | ✅ done |
| 2 | Rolling-origin backtest harness; naive and seasonal-naive baselines; day-ahead and week-ahead error (MAE, MAPE, peak and trough error) | next |
| 3 | Exponential smoothing (Holt-Winters) and a simple, explainable temperature adjustment from free public weather data | |
| 4 | Business translation: how many MW of buffer cover 90% / 95% of days, which regions and seasons are hardest, recommendations | |
| 5 | CI, TL;DR and final write-up | |

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or requirements-lock.txt for the exact tested versions

python -m scripts.profile              # report + charts from the committed snapshot
pytest                                 # 23 tests

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

## Layout

```
demand/source.py          regions, months, file names, paths
demand/halfhour.py        parse a raw file, roll 5-minute rows into half-hours
scripts/download_data.py  fetch + verify the raw files
scripts/build_snapshot.py raw files -> data/demand_halfhourly.csv.gz + SNAPSHOT.json
scripts/profile.py        DuckDB SQL -> results/SOURCE_PROFILE.md + charts
tests/                    pytest
```

*Data: © AEMO, used under AEMO's copyright permissions with attribution.*
