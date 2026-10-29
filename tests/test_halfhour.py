"""Hand-worked checks for parsing and the 5-minute -> half-hour roll-up."""
import pandas as pd
import pytest

from demand.halfhour import SourceError, read_raw, resolution_minutes, to_half_hourly
from demand.source import file_name, months

HEADER = "REGION,SETTLEMENTDATE,TOTALDEMAND,RRP,PERIODTYPE\n"


def five_minute_file(region="VIC1", day="2025/09/01", start_hour=0, values=None):
    values = values or [100, 110, 120, 130, 140, 150, 200, 200, 200, 200, 200, 200]
    rows = []
    for i, v in enumerate(values, start=1):
        minutes = start_hour * 60 + 5 * i
        rows.append(f"{region},{day} {minutes // 60:02d}:{minutes % 60:02d}:00,{v},{v / 10},TRADE\n")
    return (HEADER + "".join(rows)).encode()


def test_months_and_file_names():
    assert list(months((2019, 11), (2020, 2))) == [(2019, 11), (2019, 12), (2020, 1), (2020, 2)]
    assert file_name("SA1", 2021, 9) == "PRICE_AND_DEMAND_202109_SA1.csv"


def test_five_minute_rows_average_into_the_half_hour_they_end_in():
    raw = read_raw(five_minute_file(), "VIC1", 2025, 9).assign(resolution_min=5)
    assert resolution_minutes(raw["interval_end"]) == 5
    hh = to_half_hourly(raw)
    assert hh["interval_end"].dt.strftime("%H:%M").tolist() == ["00:30", "01:00"]
    # 00:05..00:30 -> mean(100..150) = 125; 00:35..01:00 -> 200
    assert hh["demand_mw"].tolist() == [125.0, 200.0]
    assert hh["rrp"].tolist() == [12.5, 20.0]
    assert hh["n_intervals"].tolist() == [6, 6]


def test_thirty_minute_rows_pass_through_unchanged():
    data = (HEADER + "SA1,2019/01/01 00:30:00,1394.21,75.67,TRADE\n"
                     "SA1,2019/01/01 01:00:00,1338.64,77.81,TRADE\n").encode()
    raw = read_raw(data, "SA1", 2019, 1)
    assert resolution_minutes(raw["interval_end"]) == 30
    hh = to_half_hourly(raw.assign(resolution_min=30))
    assert hh["demand_mw"].tolist() == [1394.21, 1338.64]
    assert hh["n_intervals"].tolist() == [1, 1]


def test_midnight_on_the_first_of_next_month_belongs_to_this_month():
    data = (HEADER + "VIC1,2025/09/30 23:55:00,1,1,TRADE\n"
                     "VIC1,2025/10/01 00:00:00,2,2,TRADE\n").encode()
    raw = read_raw(data, "VIC1", 2025, 9)
    assert raw["interval_end"].max() == pd.Timestamp("2025-10-01 00:00")


@pytest.mark.parametrize("row", ["VIC1,2025/09/01 00:00:00,1,1,TRADE",   # previous month
                                 "VIC1,2025/10/01 00:05:00,1,1,TRADE",   # next month
                                 "NSW1,2025/09/02 00:05:00,1,1,TRADE"])  # wrong region
def test_rows_that_do_not_belong_in_the_file_are_rejected(row):
    with pytest.raises(SourceError):
        read_raw((HEADER + row + "\n").encode(), "VIC1", 2025, 9)


def test_wrong_header_is_rejected():
    with pytest.raises(SourceError):
        read_raw(b"REGION,SETTLEMENTDATE,DEMAND,RRP,PERIODTYPE\n", "VIC1", 2025, 9)


def test_exact_duplicates_are_dropped_but_conflicting_ones_fail():
    raw = read_raw(five_minute_file(), "VIC1", 2025, 9).assign(resolution_min=5)
    doubled = pd.concat([raw, raw.iloc[:1]])
    assert to_half_hourly(doubled)["demand_mw"].tolist() == [125.0, 200.0]
    clash = pd.concat([raw, raw.iloc[:1].assign(demand_mw=999.0)])
    with pytest.raises(SourceError):
        to_half_hourly(clash)


def test_a_short_half_hour_is_kept_and_counted():
    raw = read_raw(five_minute_file(values=[100, 110, 120]), "VIC1", 2025, 9)
    hh = to_half_hourly(raw.assign(resolution_min=5))
    assert hh["n_intervals"].tolist() == [3]
    assert hh["demand_mw"].tolist() == [110.0]
