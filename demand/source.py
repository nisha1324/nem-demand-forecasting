"""What is downloaded from AEMO, and where it lives.

Source: AEMO "Aggregated price and demand data", one CSV per NEM region per
month (https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data).
Licence: AEMO's copyright permissions give "general permission for anyone to
use AEMO Material for any purpose, but only with accurate and appropriate
attribution of the relevant AEMO Material and AEMO as its author".

Each row is one market interval in NEM time (AEST, UTC+10, no daylight
saving), stamped with the END of the interval:
- before 1 Oct 2021: 30-minute trading intervals
- from 1 Oct 2021 (five-minute settlement): 5-minute dispatch intervals
TOTALDEMAND is operational demand in MW: what the grid supplies, net of
rooftop solar, which AEMO doesn't meter here.
"""
from pathlib import Path

REGIONS = ("NSW1", "QLD1", "VIC1", "SA1", "TAS1")
REGION_NAMES = {
    "NSW1": "New South Wales",
    "QLD1": "Queensland",
    "VIC1": "Victoria",
    "SA1": "South Australia",
    "TAS1": "Tasmania",
}
FIRST_MONTH = (2019, 1)
LAST_MONTH = (2026, 9)  # last complete month when the snapshot was taken
FIVE_MINUTE_FROM = "2021-10-01"  # first month of 5-minute settlement

BASE_URL = "https://www.aemo.com.au/aemo/data/nem/priceanddemand"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "raw_manifest.json"
SNAPSHOT = ROOT / "data" / "demand_halfhourly.csv.gz"
SNAPSHOT_META = ROOT / "data" / "SNAPSHOT.json"


def months(first=FIRST_MONTH, last=LAST_MONTH):
    """Every (year, month) from first to last inclusive."""
    y, m = first
    while (y, m) <= last:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def file_name(region: str, year: int, month: int) -> str:
    return f"PRICE_AND_DEMAND_{year}{month:02d}_{region}.csv"


def all_files():
    """(region, year, month, file name) for every file in the snapshot."""
    for year, month in months():
        for region in REGIONS:
            yield region, year, month, file_name(region, year, month)


def url(name: str) -> str:
    return f"{BASE_URL}/{name}"
