"""Download the AEMO price-and-demand CSVs into data/raw/ and verify them.

    python -m scripts.download_data          # fetch missing files, check every SHA-256
    python -m scripts.download_data --pin    # also record SHAs for files not yet pinned

5 regions x 93 months (Jan 2019 - Sep 2026) = 465 files, about 125 MB. The raw
files are not committed (data/raw/ is gitignored); the half-hourly snapshot
built from them is (see scripts/build_snapshot.py), so the analysis runs
without this step.

Every file's SHA-256 is pinned in data/raw_manifest.json. A file that differs
from its pinned hash stops the run instead of silently changing the numbers.
--pin only ever ADDS hashes for files that have none (e.g. after moving
LAST_MONTH forward); it never overwrites one.
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from demand.source import MANIFEST, RAW, all_files, url

USER_AGENT = "Mozilla/5.0 (nem-demand-forecasting; +https://github.com/nisha1324)"
WORKERS = 4  # keep it polite: AEMO serves these from a public website


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(name: str, attempts: int = 4) -> bytes:
    req = urllib.request.Request(url(name), headers={"User-Agent": USER_AGENT})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if not data.startswith(b"REGION,SETTLEMENTDATE"):
                raise ValueError(f"{name}: not a price-and-demand CSV (got {data[:40]!r})")
            return data
        except Exception as exc:  # network hiccups: back off and retry
            if attempt == attempts:
                raise
            print(f"  retry {attempt} for {name}: {exc}", file=sys.stderr)
            time.sleep(2 * attempt)


def ensure(name: str) -> tuple[str, str]:
    """Download the file if it isn't on disk; return (name, sha256)."""
    path = RAW / name
    if not path.exists():
        data = fetch(name)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(path)
    return name, sha256(path.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pin", action="store_true",
                        help="record hashes for files that are not pinned yet")
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    pinned = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    names = [name for *_, name in all_files()]

    with ThreadPoolExecutor(WORKERS) as pool:
        hashes = dict(pool.map(ensure, names))

    bad = [n for n in names if n in pinned and pinned[n] != hashes[n]]
    unpinned = [n for n in names if n not in pinned]
    for n in bad:
        print(f"SHA-256 MISMATCH {n}: pinned {pinned[n][:12]}..., got {hashes[n][:12]}...",
              file=sys.stderr)
    if bad:
        print(f"{len(bad)} file(s) differ from the pinned copy; nothing was changed. "
              "Check what AEMO revised before re-pinning.", file=sys.stderr)
        return 1

    if unpinned and args.pin:
        pinned.update({n: hashes[n] for n in unpinned})
        MANIFEST.write_text(json.dumps(dict(sorted(pinned.items())), indent=1) + "\n")
        print(f"pinned {len(unpinned)} new file(s) in {MANIFEST.name}")
    elif unpinned:
        print(f"{len(unpinned)} file(s) have no pinned hash; run with --pin to record them",
              file=sys.stderr)
        return 1

    print(f"{len(names)} files present, all match {MANIFEST.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
