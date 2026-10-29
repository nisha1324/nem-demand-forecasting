"""The committed snapshot is what SNAPSHOT.json says it is, and it is complete."""
import hashlib
import json

import pandas as pd
import pytest

from demand.source import MANIFEST, RAW, REGIONS, SNAPSHOT, SNAPSHOT_META, all_files, months

META = json.loads(SNAPSHOT_META.read_text())


@pytest.fixture(scope="module")
def snap() -> pd.DataFrame:
    return pd.read_csv(SNAPSHOT, parse_dates=["interval_end"])


def test_hash_and_row_count_match_the_metadata(snap):
    assert hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() == META["sha256"]
    assert len(snap) == META["rows"]
    assert list(snap.columns) == list(META["columns"])


def test_every_file_in_the_window_is_pinned():
    pinned = json.loads(MANIFEST.read_text())
    names = [name for *_, name in all_files()]
    assert sorted(names) == sorted(pinned)
    assert META["raw_checks"]["files"] == len(names) == 5 * len(list(months()))


def test_every_region_has_every_half_hour_once(snap):
    start = pd.Timestamp(META["first_interval_end"])
    end = pd.Timestamp(META["last_interval_end"])
    expected = pd.date_range(start, end, freq="30min")
    assert sorted(snap["region"].unique()) == sorted(REGIONS)
    for region, g in snap.groupby("region"):
        assert g["interval_end"].is_monotonic_increasing, region
        assert g["interval_end"].tolist() == expected.tolist(), region


def test_resolution_switches_on_1_october_2021(snap):
    five = snap["interval_end"] > pd.Timestamp("2021-10-01 00:00")
    assert (snap.loc[five, "resolution_min"] == 5).all()
    assert (snap.loc[five, "n_intervals"] == 6).all()
    assert (snap.loc[~five, "resolution_min"] == 30).all()
    assert (snap.loc[~five, "n_intervals"] == 1).all()


def test_raw_row_counts_add_up_to_the_snapshot():
    checks = META["raw_checks"]
    old, new = checks["raw_rows_30min"], checks["raw_rows_5min"]
    assert new % 6 == 0
    assert old + new // 6 == META["rows"]


@pytest.mark.skipif(not RAW.exists() or not any(RAW.iterdir()),
                    reason="raw files not downloaded (python -m scripts.download_data)")
def test_rebuilding_from_raw_gives_the_same_bytes(tmp_path, monkeypatch):
    from scripts import build_snapshot
    monkeypatch.setattr(build_snapshot, "SNAPSHOT", tmp_path / SNAPSHOT.name)
    monkeypatch.setattr(build_snapshot, "SNAPSHOT_META", tmp_path / SNAPSHOT_META.name)
    build_snapshot.main()
    assert (tmp_path / SNAPSHOT.name).read_bytes() == SNAPSHOT.read_bytes()
    assert (tmp_path / SNAPSHOT_META.name).read_text() == SNAPSHOT_META.read_text()
