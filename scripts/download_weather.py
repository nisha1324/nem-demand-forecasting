"""Archived weather forecasts for the 5 capitals: verify the committed snapshot, or fetch it.

    python -m scripts.download_weather           # check data/weather_forecasts.csv.gz against WEATHER.json
    python -m scripts.download_weather --fetch   # download, and fail if it differs from the pinned hash
    python -m scripts.download_weather --pin     # download and write the snapshot + its pinned hash

No key or sign-up. Open-Meteo's free API is for non-commercial use, under 10,000 calls a
day and 600 a minute; a long request counts as several calls, so this asks for half a
year at a time and waits between requests. A run comes to roughly 500 calls. Data: CC BY 4.0.

A missing value stops the run: nothing is imputed.
"""
import argparse
import gzip
import hashlib
import io
import json
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

from demand import weather as wx

USER_AGENT = "nem-demand-forecasting (+https://github.com/nisha1324)"
PAUSE = 5  # seconds between requests


def chunks(first: str, last: str, days: int = 180):
    start, end = pd.Timestamp(first), pd.Timestamp(last)
    while start <= end:
        stop = min(start + pd.Timedelta(f"{days - 1}D"), end)
        yield start.date().isoformat(), stop.date().isoformat()
        start = stop + pd.Timedelta("1D")


def request_url(lat: float, lon: float, first: str, last: str) -> str:
    variables = [wx.api_variable(short, lead) for short in ("temp", "solar") for lead in wx.LEADS_STORED]
    return wx.API + "?" + urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "hourly": ",".join(variables), "models": wx.MODEL,
        "start_date": first, "end_date": last, "timezone": wx.TIMEZONE})


def fetch_json(url: str, attempts: int = 4) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.load(resp)
        except Exception as exc:  # network hiccups or rate limit: back off and retry
            if attempt == attempts:
                raise
            print(f"  retry {attempt}: {exc}", file=sys.stderr)
            time.sleep(30 * attempt)


def frame(payload: dict, region: str) -> pd.DataFrame:
    """API response -> one row per hour, leads in columns. Refuses gaps."""
    hourly = payload["hourly"]
    columns = {name: hourly[wx.api_variable(name.split("_")[0], int(name.split("_")[1]))] for name in wx.COLUMNS}
    df = pd.DataFrame({"region": region, "time": hourly["time"]} | columns)
    missing = df[wx.COLUMNS].isna().sum()
    if missing.any():
        raise ValueError(f"{region}: missing values {missing[missing > 0].to_dict()}")
    return df


def fetch_all() -> pd.DataFrame:
    frames = []
    for region in wx.SITES:
        city, lat, lon = wx.SITES[region]
        for first, last in chunks(wx.FIRST_DAY, wx.LAST_DAY):
            print(f"  {city} {first} to {last}", flush=True)
            frames.append(frame(fetch_json(request_url(lat, lon, first, last)), region))
            time.sleep(PAUSE)
    return pd.concat(frames, ignore_index=True)


def snapshot_bytes(df: pd.DataFrame) -> bytes:
    """Deterministic gzip: same rows, same bytes."""
    text = df.to_csv(index=False, lineterminator="\n", float_format="%.1f").encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=buffer, mtime=0) as gz:
        gz.write(text)
    return buffer.getvalue()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fetch", action="store_true", help="download and compare with the pinned hash")
    parser.add_argument("--pin", action="store_true", help="download and write the snapshot and its hash")
    args = parser.parse_args()
    meta = json.loads(wx.WEATHER_META.read_text()) if wx.WEATHER_META.exists() else None

    if not (args.fetch or args.pin):
        got = sha256(wx.WEATHER.read_bytes())
        if got != meta["sha256"]:
            print(f"SHA-256 MISMATCH: pinned {meta['sha256'][:12]}..., got {got[:12]}...", file=sys.stderr)
            return 1
        print(f"{wx.WEATHER.name} matches {wx.WEATHER_META.name}")
        return 0

    df = fetch_all()
    data = snapshot_bytes(df)
    if args.fetch:
        same = data == wx.WEATHER.read_bytes()
        print("download matches the snapshot" if same else "download differs from the snapshot: nothing changed")
        return 0 if same else 1

    wx.WEATHER.write_bytes(data)
    wx.WEATHER_META.write_text(json.dumps({
        "source": "Open-Meteo Previous Runs API", "url": wx.API, "licence": "CC BY 4.0",
        "attribution": "Weather data by Open-Meteo.com", "model": wx.MODEL, "timezone": wx.TIMEZONE,
        "first_day": wx.FIRST_DAY, "last_day": wx.LAST_DAY,
        "sites": {r: {"city": c, "latitude": a, "longitude": o} for r, (c, a, o) in wx.SITES.items()},
        "rows": len(df), "sha256": sha256(data), "retrieved": pd.Timestamp.now("UTC").date().isoformat(),
    }, indent=1) + "\n")
    print(f"wrote {wx.WEATHER.name}: {len(df):,} rows, {len(data) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
