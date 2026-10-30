"""Build the public-holiday calendar the Part 3 forecasts use.

    python -m scripts.build_holidays             # rewrite data/public_holidays.csv
    python -m scripts.build_holidays --official  # also refresh the official check file

The calendar comes from the open-source `holidays` package (MIT licence, version
pinned in requirements-lock.txt), for one reason: one set of rules covers every
year. The government's machine-readable list is marked inactive and stops at
2025, and the backtest runs to Sep 2026.

Holidays are published in advance, so giving a forecast the calendar of the day it
targets isn't looking ahead.

Kept: holidays that apply to a whole state for a whole day. Dropped: the Royal
Queensland Show, which is "Brisbane area only" on the Queensland Government's list.

The official list (data.gov.au, CC BY 3.0 AU) is used as a check, not as an input:
data/official_holidays_2019_2025.csv keeps its rows for the 5 NEM states, and
`reconcile` explains every date the two disagree on, or fails.
"""
import argparse
import csv
import hashlib
import io
import urllib.request

import holidays
import pandas as pd

from demand.source import ROOT

CALENDAR = ROOT / "data" / "public_holidays.csv"
OFFICIAL = ROOT / "data" / "official_holidays_2019_2025.csv"
YEARS = range(2019, 2027)
SUBDIV = {"NSW1": "NSW", "QLD1": "QLD", "VIC1": "VIC", "SA1": "SA", "TAS1": "TAS"}
NOT_STATEWIDE = ("Royal Queensland Show",)

RESOURCE = "https://data.gov.au/data/dataset/b1bc6077-dadd-4f61-9f8c-002ab2cdff10/resource"
OFFICIAL_FILES = {
    f"{RESOURCE}/bda4d4f2-7fde-4bfc-8a23-a6eefc8cef80/download/australian_public_holidays_2019.csv":
        "ef32aaaaa32caaa57a7244c4a48edf0e288c97f672172f3d102aeb9f23d06d06",
    f"{RESOURCE}/c4163dc4-4f5a-4cae-b787-43ef0fcf8d8b/download/australian_public_holidays_2020.csv":
        "72de56e354738d965c9b2ed8e45e266f79080f3b4b76dcc53e879fe62cb3d317",
    f"{RESOURCE}/33673aca-0857-42e5-b8f0-9981b4755686/download/australian-public-holidays-combined-2021-2025.csv":
        "4126fe21c5c96994ea9c2d01227df0df62a62665094e1cded3d677d42bb62356",
}

# why an official row may be missing from the calendar: (state or None, text in the name, reason)
OFFICIAL_ONLY = (
    ("NSW1", "Bank Holiday", "banks only"),
    ("QLD1", "Christmas Eve", "part day (evening)"),
    ("QLD1", "Royal Queensland Show", "Brisbane area only"),
    ("SA1", "Christmas Eve", "part day (evening)"),
    ("SA1", "New Year", "part day (evening)"),
    ("TAS1", "Easter Tuesday", "public service only"),
    ("TAS1", "Royal Hobart Regatta", "southern Tasmania only"),
)
OFFICIAL_BLANK_DATE = "official row has no date"
WEEKEND = "falls on a weekend, already a rest day"


def build_calendar() -> pd.DataFrame:
    rows = []
    for region, subdiv in SUBDIV.items():
        for day, name in holidays.country_holidays("AU", subdiv=subdiv, years=YEARS).items():
            # a date can carry two holidays; the package joins their names with "; "
            names = [n for n in name.split("; ") if not any(x in n for x in NOT_STATEWIDE)]
            if names:
                rows.append((region, day.isoformat(), "; ".join(names)))
    return pd.DataFrame(rows, columns=["region", "date", "name"]).sort_values(["region", "date"], ignore_index=True)


def fetch_official() -> pd.DataFrame:
    rows = []
    state = {v.lower(): k for k, v in SUBDIV.items()}
    for url, sha in OFFICIAL_FILES.items():
        raw = urllib.request.urlopen(url, timeout=60).read()
        got = hashlib.sha256(raw).hexdigest()
        if got != sha:
            raise SystemExit(f"{url}\nexpected {sha}, got {got}")
        for r in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
            region = state.get(r["Jurisdiction"].strip().lower())
            if region:
                d = r["Date"].strip()
                date = f"{d[:4]}-{d[4:6]}-{d[6:]}" if d else ""
                rows.append((region, date, " ".join(r["Holiday Name"].split())))
    return pd.DataFrame(rows, columns=["region", "date", "name"]).sort_values(["region", "date", "name"], ignore_index=True)


def reconcile(calendar: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    """Every date the two lists disagree on, 2019-2025, with its reason. Fails on an unexplained one."""
    cal = calendar[calendar["date"].str[:4] <= "2025"]
    cal_keys = set(zip(cal["region"], cal["date"]))
    off_keys = set(zip(official["region"], official["date"]))
    blank = official[official["date"] == ""]
    out = []
    for r in official.itertuples():
        if not r.date or (r.region, r.date) in cal_keys:
            continue
        why = next((w for reg, text, w in OFFICIAL_ONLY if reg == r.region and text in r.name), None)
        if why is None and pd.Timestamp(r.date).dayofweek >= 5:
            why = WEEKEND
        if why is None:
            raise ValueError(f"unexplained official holiday: {r}")
        out.append((r.region, r.date, r.name, "official only", why))
    for r in cal.itertuples():
        if (r.region, r.date) in off_keys:
            continue
        if r.name in set(blank.loc[blank["region"] == r.region, "name"]):
            why = OFFICIAL_BLANK_DATE
        elif "Day of Mourning" in r.name:
            why = "one-off national holiday, not in the official file"
        elif pd.Timestamp(r.date).dayofweek >= 5:
            why = WEEKEND
        else:
            raise ValueError(f"unexplained calendar holiday: {r}")
        out.append((r.region, r.date, r.name, "calendar only", why))
    return pd.DataFrame(out, columns=["region", "date", "name", "side", "reason"]).sort_values(["region", "date"], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--official", action="store_true", help="re-download the official check file")
    args = ap.parse_args()
    calendar = build_calendar()
    calendar.to_csv(CALENDAR, index=False)
    if args.official:
        fetch_official().to_csv(OFFICIAL, index=False)
    official = pd.read_csv(OFFICIAL, keep_default_na=False)
    diff = reconcile(calendar, official)
    print(f"{len(calendar)} calendar rows; {len(official)} official rows; {len(diff)} explained differences")
    print(diff.groupby(["side", "reason"]).size().to_string())


if __name__ == "__main__":
    main()
