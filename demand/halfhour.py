"""Turn AEMO's raw interval files into one half-hourly table.

Since 1 Oct 2021 AEMO publishes 5-minute intervals; before that, 30-minute
ones. Forecasting and planning here work in half-hours (the NEM's
traditional trading interval), so the 5-minute rows are averaged into the
half-hour they belong to. Timestamps mark the END of an interval, so the
5-minute rows ending 00:05 ... 00:30 make up the half-hour ending 00:30.

The averages are time-weighted means of MW, which is what a half-hour of
5-minute readings represents. Energy for a half-hour is demand_mw * 0.5 MWh.
"""
from io import BytesIO

import pandas as pd

HEADER = ["REGION", "SETTLEMENTDATE", "TOTALDEMAND", "RRP", "PERIODTYPE"]


class SourceError(ValueError):
    """A raw file that doesn't look the way the rest of the code assumes."""


def read_raw(data: bytes, region: str, year: int, month: int) -> pd.DataFrame:
    """Parse one monthly file and check it belongs where its name says.

    Returns columns region, interval_end (naive NEM time), demand_mw, rrp,
    period_type. Raises SourceError on a wrong header, region or month window.
    """
    df = pd.read_csv(BytesIO(data), dtype={"REGION": str, "PERIODTYPE": str})
    if list(df.columns) != HEADER:
        raise SourceError(f"{region} {year}-{month:02d}: unexpected header {list(df.columns)}")
    if set(df["REGION"]) != {region}:
        raise SourceError(f"{region} {year}-{month:02d}: regions {sorted(set(df['REGION']))}")
    ts = pd.to_datetime(df["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S")
    start = pd.Timestamp(year, month, 1)
    end = start + pd.offsets.MonthBegin(1)
    # interval-ending stamps: the month runs from just after 00:00 on the 1st
    # up to and including 00:00 on the 1st of the next month
    outside = ~((ts > start) & (ts <= end))
    if outside.any():
        raise SourceError(f"{region} {year}-{month:02d}: {int(outside.sum())} rows outside the month")
    return pd.DataFrame({
        "region": df["REGION"],
        "interval_end": ts,
        "demand_mw": df["TOTALDEMAND"].astype(float),
        "rrp": df["RRP"].astype(float),
        "period_type": df["PERIODTYPE"],
    })


def resolution_minutes(interval_end: pd.Series) -> int:
    """5 or 30, from the spacing of one file's timestamps."""
    steps = interval_end.sort_values().diff().dropna().dt.total_seconds().div(60)
    mode = int(steps.mode().iloc[0])
    if mode not in (5, 30):
        raise SourceError(f"unexpected interval length {mode} min")
    return mode


def to_half_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """Average 5- or 30-minute rows into half-hours ending on :00 / :30.

    raw needs region, interval_end, demand_mw, rrp and resolution_min.
    Exact duplicate rows are dropped first; a timestamp that appears twice with
    different values is an error, because there would be no right answer.
    """
    raw = raw.drop_duplicates()
    clash = raw.duplicated(["region", "interval_end"], keep=False)
    if clash.any():
        raise SourceError(f"{int(clash.sum())} rows share a timestamp with different values")
    hh = raw.assign(interval_end=raw["interval_end"].dt.ceil("30min"))
    out = (hh.groupby(["region", "interval_end"], as_index=False)
             .agg(demand_mw=("demand_mw", "mean"),
                  rrp=("rrp", "mean"),
                  n_intervals=("demand_mw", "size"),
                  resolution_min=("resolution_min", "max")))
    return out.sort_values(["region", "interval_end"], ignore_index=True)
