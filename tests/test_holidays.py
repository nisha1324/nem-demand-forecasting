"""The holiday calendar: current, statewide, and reconciled with the official list."""
import pandas as pd
import pytest

from scripts import build_holidays as bh


@pytest.fixture(scope="module")
def committed() -> pd.DataFrame:
    return pd.read_csv(bh.CALENDAR)


@pytest.fixture(scope="module")
def official() -> pd.DataFrame:
    return pd.read_csv(bh.OFFICIAL, keep_default_na=False)


def test_committed_calendar_matches_a_fresh_build(committed):
    pd.testing.assert_frame_equal(bh.build_calendar(), committed)


@pytest.mark.parametrize("region,date", [
    ("NSW1", "2025-12-25"), ("VIC1", "2025-11-04"),  # Melbourne Cup
    ("VIC1", "2026-09-25"), ("SA1", "2026-03-09"),   # AFL Friday, Adelaide Cup
    ("NSW1", "2026-04-27"), ("TAS1", "2026-03-09"),  # Anzac Day made up, Eight Hours Day
])
def test_known_holidays_are_in(committed, region, date):
    assert date in set(committed.loc[committed["region"] == region, "date"])


@pytest.mark.parametrize("region,date", [
    ("QLD1", "2026-08-12"),  # Royal Queensland Show: Brisbane area only
    ("TAS1", "2026-04-04"),  # Easter Saturday isn't a Tasmanian holiday
    ("VIC1", "2025-10-06"),  # NSW, SA and QLD have this Monday, Victoria doesn't
])
def test_known_non_holidays_are_out(committed, region, date):
    assert date not in set(committed.loc[committed["region"] == region, "date"])


def test_every_difference_from_the_official_list_has_a_reason(committed, official):
    diff = bh.reconcile(committed, official)
    assert diff["reason"].notna().all() and len(diff) > 0
    assert set(diff.loc[diff["reason"] == bh.OFFICIAL_BLANK_DATE, "date"]) == {"2025-09-26"}


def test_an_unexplained_official_holiday_fails(committed, official):
    extra = pd.DataFrame([("NSW1", "2024-03-13", "Some New Holiday")], columns=official.columns)
    with pytest.raises(ValueError, match="unexplained"):
        bh.reconcile(committed, pd.concat([official, extra], ignore_index=True))


def test_an_unexplained_calendar_holiday_fails(committed, official):
    extra = pd.DataFrame([("VIC1", "2024-03-13", "Some Other Holiday")], columns=committed.columns)
    with pytest.raises(ValueError, match="unexplained"):
        bh.reconcile(pd.concat([committed, extra], ignore_index=True), official)
