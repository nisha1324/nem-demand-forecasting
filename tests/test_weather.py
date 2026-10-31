"""The weather step on hand-made data: alignment, features, the fit, and no look-ahead."""
import json

import numpy as np
import pandas as pd
import pytest

from demand import weather as wx
from scripts import download_weather as dl


def hourly_frame(days, temp=None, solar=None) -> pd.DataFrame:
    stamps = pd.date_range(days[0], days[-1] + pd.Timedelta(hours=23), freq="h")
    temp = np.arange(len(stamps), dtype=float) if temp is None else temp
    solar = np.arange(len(stamps), dtype=float) * 10 if solar is None else solar
    data = {"time": stamps}
    for lead in wx.LEADS_STORED:
        data[f"temp_{lead}"] = temp + lead / 10
        data[f"solar_{lead}"] = solar + lead
    return pd.DataFrame(data)


def weather_dict(n, rng=None) -> dict:
    rng = rng or np.random.default_rng(3)
    return {name: rng.uniform(5, 35, (n, 48)) if name.startswith("temp") else rng.uniform(0, 900, (n, 48))
            for name in wx.COLUMNS}


def test_temperature_is_interpolated_between_hours():
    days = pd.date_range("2024-04-01", periods=2, freq="D")
    out = wx.halfhourly(hourly_frame(days), days)["temp_0"]
    assert out[0, :4].tolist() == [0, .5, 1, 1.5]
    assert out[1, 0] == 24
    assert out[1, 47] == 47  # the last half-hour of the data holds the last value


def test_radiation_stamp_covers_the_hour_before_it():
    days = pd.date_range("2024-04-01", periods=1, freq="D")
    out = wx.halfhourly(hourly_frame(days), days)["solar_0"]
    # stamp 01:00 (value 10) is the mean of 00:00-01:00, so it fills slots 0 and 1
    assert out[0, :4].tolist() == [10, 10, 20, 20]


def test_a_missing_hour_is_refused():
    days = pd.date_range("2024-04-01", periods=2, freq="D")
    with pytest.raises(ValueError, match="consecutive hours"):
        wx.halfhourly(hourly_frame(days).drop(index=5).reset_index(drop=True), days)


def test_a_missing_value_from_the_api_stops_the_download():
    hourly = {"time": ["2024-04-01T00:00", "2024-04-01T01:00"]}
    for short in ("temp", "solar"):
        for lead in wx.LEADS_STORED:
            hourly[wx.api_variable(short, lead)] = [1.0, 2.0]
    assert len(dl.frame({"hourly": hourly}, "SA1")) == 2
    hourly["temperature_2m_previous_day3"] = [1.0, None]
    with pytest.raises(ValueError, match="temp_3"):
        dl.frame({"hourly": hourly}, "SA1")


def test_chunks_cover_the_range_without_overlap():
    parts = list(dl.chunks("2024-04-01", "2026-09-30"))
    assert parts[0][0] == "2024-04-01" and parts[-1][1] == "2026-09-30"
    for (_, last), (first, _) in zip(parts, parts[1:]):
        assert pd.Timestamp(first) - pd.Timestamp(last) == pd.Timedelta("1D")


def test_committed_snapshot_matches_its_hash_and_rebuilds_byte_for_byte():
    meta = json.loads(wx.WEATHER_META.read_text())
    data = wx.WEATHER.read_bytes()
    assert dl.sha256(data) == meta["sha256"]
    assert meta["licence"] == "CC BY 4.0" and meta["model"] == wx.MODEL
    df = pd.read_csv(wx.WEATHER)
    assert dl.snapshot_bytes(df) == data


def test_snapshot_is_complete_and_uses_the_nem_clock():
    df = pd.read_csv(wx.WEATHER)
    days = len(pd.date_range(wx.FIRST_DAY, wx.LAST_DAY, freq="D"))
    assert len(df) == 5 * days * 24 == json.loads(wx.WEATHER_META.read_text())["rows"]
    assert not df.isna().any().any()
    assert df.duplicated(["region", "time"]).sum() == 0
    # the sun peaks near noon on the AEST clock, a UTC file would peak ten hours away
    _, w = wx.load_weather()
    for region, peak in (("NSW1", 22), ("QLD1", 22), ("VIC1", 24), ("SA1", 24), ("TAS1", 24)):
        assert int(w[region]["solar_1"].mean(axis=0).argmax()) == peak


def test_forecast_skill_decays_with_lead_as_a_real_forecast_should():
    df = pd.read_csv(wx.WEATHER, parse_dates=["time"])
    df = df[df["time"].dt.hour.between(9, 15) & (df["time"] < "2025-10-01")]
    miss = [(df[f"solar_{lead}"] - df["solar_0"]).abs().mean() for lead in wx.FORECAST_LEADS]
    assert miss == sorted(miss) and miss[0] > 20


@pytest.mark.parametrize("lead, column", [(1, 2), (2, 3), (5, 6), (6, 7), (7, 7)])
def test_strict_lead_reads_a_day_older_until_the_archive_stops(lead, column):
    assert wx.strict_lead(lead) == column


def test_shutdown_mask_is_27_to_31_december():
    days = pd.to_datetime(["2025-12-26", "2025-12-27", "2025-12-31", "2026-01-01"])
    assert wx.shutdown_mask(days).tolist() == [False, True, True, False]


def test_recent_mean_is_the_week_before_the_origin():
    a = np.arange(20, dtype=float).reshape(20, 1)
    out = wx.recent_mean(a, lead=3)
    assert out[15, 0] == np.mean([6, 7, 8, 9, 10, 11, 12])  # origin 13 sees rows 6..12
    assert np.isnan(out[8, 0]) and out[9, 0] == 3  # origin 7 is the first with a full week


def test_recent_mean_survives_nan_rows_before_the_archive_starts():
    # a running sum would turn every later row into NaN: that was a real bug
    a = np.vstack([np.full((10, 2), np.nan), np.ones((20, 2))])
    out = wx.recent_mean(a, lead=1)
    assert np.isnan(out[10:17]).all() and (out[17:] == 1).all()


def test_features_by_hand():
    w = {k: np.zeros((10, 48)) for k in wx.COLUMNS}
    w["temp_1"][:] = 21     # the last week: 1 cooling degree above 20, no heating
    w["solar_1"][:] = 200
    w["temp_2"][9, 0], w["solar_2"][9, 0] = 25, 500     # the target, read a day older
    w["temp_2"][9, 1] = 10  # 4 heating degrees below 14
    x = wx.features(w, 1, heat=14, cool=20)
    assert x[9, 0].tolist() == [5 - 1, 0, 500 - 200]
    assert x[9, 1].tolist() == [0 - 1, 4, 0 - 200]


def test_features_read_only_the_target_column_and_days_before_the_origin():
    n, lead, t = 30, 3, 25
    w = weather_dict(n)
    base = wx.features(w, lead, 14, 20)
    spoiled = {k: v.copy() for k, v in w.items()}
    origin = t - lead + 1
    for name in wx.COLUMNS:
        if name.endswith(f"_{wx.strict_lead(lead)}"):
            spoiled[name][t + 1:] = 1e6  # the target column is read at the target only
        else:
            spoiled[name][origin:] = 1e6
    assert np.array_equal(wx.features(spoiled, lead, 14, 20)[t], base[t])


def test_perfect_weather_reads_column_zero_at_the_target():
    w = weather_dict(30)
    x = wx.features(w, 1, 14, 20, column=0)
    w["temp_0"][25] += 50
    assert not np.array_equal(wx.features(w, 1, 14, 20, column=0)[25], x[25])


def test_fit_recovers_known_slopes_and_intercept():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 48, 3)) * np.array([2, 5, 300])
    y = 40 * x[..., 0] - 10 * x[..., 1] + .5 * x[..., 2] + 7
    slopes, intercepts = wx.fit_slots(x, y, lam=1e-6)
    assert slopes == pytest.approx(np.tile([40, -10, .5], (48, 1)), abs=1e-5)
    assert intercepts == pytest.approx(np.full(48, 7), abs=1e-5)


def test_ridge_shrinks_towards_zero_and_scale_does_not_change_the_shrinkage():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(200, 48, 1))
    y = 5 * x[..., 0]
    light, _ = wx.fit_slots(x, y, 1)
    heavy, _ = wx.fit_slots(x, y, 200)
    assert np.abs(heavy).max() < np.abs(light).min()
    wide, _ = wx.fit_slots(x * 100, y, 200)
    assert wide * 100 == pytest.approx(heavy)


def test_a_column_with_no_spread_gets_a_zero_slope():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(100, 48, 2))
    x[:, :, 1] = 0  # the sun at night
    slopes, _ = wx.fit_slots(x, 3 * x[..., 0], 1)
    assert (slopes[:, 1] == 0).all()


def make_case(n=500, lead=1):
    days = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(5)
    x = rng.normal(size=(n, 48, 3))
    residual = 30 * x[..., 0]
    return days, x, residual, np.zeros(n, bool)


def test_adjustment_recovers_the_effect():
    days, x, residual, skip = make_case()
    adj = wx.adjustment(residual, x, days, 1, 1e-6, skip)
    late = days >= "2025-01-01"
    assert adj[late] == pytest.approx(residual[late], abs=1e-6)


def test_adjustment_is_zero_until_enough_days_exist():
    days, x, residual, skip = make_case()
    adj = wx.adjustment(residual, x, days, 1, 1, skip)
    assert (adj[days < "2024-05-01"] == 0).all()
    assert (adj[days >= "2024-06-01"] != 0).any()


@pytest.mark.parametrize("lead", [1, 7])
def test_adjustment_cannot_see_past_the_start_of_the_origins_month(lead):
    days, x, residual, skip = make_case()
    t = int(np.flatnonzero(days == pd.Timestamp("2025-03-03"))[0])
    stamp = (days[t] - pd.Timedelta(f"{lead - 1}D")).to_period("M").to_timestamp()
    base = wx.adjustment(residual, x, days, lead, 1, skip)
    spoiled = residual.copy()
    spoiled[days >= stamp] = 1e7
    assert np.array_equal(wx.adjustment(spoiled, x, days, lead, 1, skip)[t], base[t])
    # and the day just before that does count
    moved = residual.copy()
    moved[days < stamp] += 1e4 * (days[days < stamp] == stamp - pd.Timedelta("1D"))[:, None]
    assert not np.array_equal(wx.adjustment(moved, x, days, lead, 1, skip)[t], base[t])


def test_skipped_days_stay_out_of_the_fit():
    days, x, residual, skip = make_case()
    skip = (days.month == 6) & (days.year == 2024)
    spoiled = residual.copy()
    spoiled[skip] = 1e7
    a = wx.adjustment(residual, x, days, 1, 1, skip)
    b = wx.adjustment(spoiled, x, days, 1, 1, skip)
    assert np.array_equal(a[days >= "2024-08-01"], b[days >= "2024-08-01"])


def test_nan_rows_are_left_out_of_the_fit():
    days, x, residual, skip = make_case()
    x = x.copy()
    residual = residual.copy()
    x[:50] = np.nan
    residual[:60] = np.nan
    adj = wx.adjustment(residual, x, days, 1, 1, skip)
    assert not np.isnan(adj[days >= "2024-08-01"]).any()
