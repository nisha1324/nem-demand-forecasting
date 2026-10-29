"""Build the committed half-hourly snapshot from the raw AEMO files.

    python -m scripts.download_data   # first, if data/raw/ is empty
    python -m scripts.build_snapshot

Writes data/demand_halfhourly.csv.gz (one row per region per half-hour, NEM
time) and data/SNAPSHOT.json (hash, coverage, licence and the raw-file checks
the profile reports on). The gzip is written with a fixed timestamp, so the
same raw files always give byte-identical output.
"""
import gzip
import hashlib
import json
import sys
from collections import Counter

import pandas as pd

from demand.halfhour import read_raw, resolution_minutes, to_half_hourly
from demand.source import (FIRST_MONTH, FIVE_MINUTE_FROM, LAST_MONTH, MANIFEST, RAW,
                           REGIONS, SNAPSHOT, SNAPSHOT_META, all_files)

ATTRIBUTION = ("Source: Australian Energy Market Operator (AEMO), Aggregated price and "
               "demand data, used under AEMO's copyright permissions (any purpose, with "
               "attribution to AEMO as author).")


def load_raw() -> tuple[pd.DataFrame, dict]:
    pinned = json.loads(MANIFEST.read_text())
    frames, checks = [], Counter()
    period_types = Counter()
    for region, year, month, name in all_files():
        data = (RAW / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != pinned.get(name):
            raise SystemExit(f"{name}: missing or not the pinned file; run scripts.download_data")
        df = read_raw(data, region, year, month)
        res = resolution_minutes(df["interval_end"])
        expected = 5 if f"{year}-{month:02d}-01" >= FIVE_MINUTE_FROM else 30
        if res != expected:
            raise SystemExit(f"{name}: {res}-minute rows, expected {expected}")
        checks[f"raw_rows_{res}min"] += len(df)
        checks["exact_duplicate_rows"] += int(df.duplicated().sum())
        period_types.update(df["period_type"])
        frames.append(df.assign(resolution_min=res))
    raw = pd.concat(frames, ignore_index=True)
    checks["files"] = len(frames)
    return raw, {**dict(checks), "period_types": dict(period_types)}


def write_snapshot(hh: pd.DataFrame) -> str:
    out = hh.assign(interval_end=hh["interval_end"].dt.strftime("%Y-%m-%d %H:%M"),
                    demand_mw=hh["demand_mw"].round(3),
                    rrp=hh["rrp"].round(3))
    text = out.to_csv(index=False, lineterminator="\n").encode()
    with open(SNAPSHOT, "wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh,
                                                   mtime=0, compresslevel=9) as gz:
        gz.write(text)
    return hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()


def main() -> int:
    raw, checks = load_raw()
    hh = to_half_hourly(raw)
    digest = write_snapshot(hh)
    meta = {
        "file": SNAPSHOT.name,
        "sha256": digest,
        "rows": len(hh),
        "regions": list(REGIONS),
        "first_month": "%d-%02d" % FIRST_MONTH,
        "last_month": "%d-%02d" % LAST_MONTH,
        "first_interval_end": str(hh["interval_end"].min()),
        "last_interval_end": str(hh["interval_end"].max()),
        "time_basis": "NEM time (AEST, UTC+10, no daylight saving); interval-ending",
        "columns": {
            "region": "NEM region id (NSW1, QLD1, VIC1, SA1, TAS1)",
            "interval_end": "end of the half-hour, NEM time",
            "demand_mw": "operational demand (TOTALDEMAND), MW, time-weighted mean over the half-hour",
            "rrp": "regional reference price, $/MWh, time-weighted mean over the half-hour",
            "n_intervals": "source intervals averaged (6 for 5-minute data, 1 for 30-minute)",
            "resolution_min": "source resolution in minutes (30 before Oct 2021, then 5)",
        },
        "raw_checks": checks,
        "source": "https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data",
        "licence": ATTRIBUTION,
    }
    SNAPSHOT_META.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{len(hh):,} half-hours -> {SNAPSHOT.name} ({SNAPSHOT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
