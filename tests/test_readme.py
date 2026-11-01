"""The README's headline ranges are recounted from the committed result files, not from its own prose."""
import pandas as pd
import pytest

from demand.source import REGIONS, ROOT

README = (ROOT / "README.md").read_text()
RESULTS = ROOT / "results"


def section(start: str, end: str) -> str:
    return README[README.index(start):README.index(end)]


TLDR = section("## TL;DR", "## Data")
RECOMMENDATIONS = section("## Recommendations, ranked", "## Limitations")


def day_ahead(path: str) -> pd.DataFrame:
    d = pd.read_csv(RESULTS / path, dtype={"lead": str})
    d = d[(d["window"] == "latest") & (d["lead"] == "1")]
    return d.pivot(index="region", columns="method", values="wape").loc[list(REGIONS)]


@pytest.fixture(scope="module")
def base() -> pd.DataFrame:
    return day_ahead("backtest_metrics.csv")


@pytest.fixture(scope="module")
def weather() -> pd.Series:
    return day_ahead("weather_metrics.csv")["weather"]


def span(values, digits=0) -> str:
    low, high = min(values), max(values)
    return f"{low:.{digits}f}-{high:.{digits}f}"


def test_day_ahead_error_range(weather):
    assert f"{span(weather * 100, 1)}% of demand" in TLDR


def test_cut_on_the_best_baseline(base, weather):
    cut = span((1 - weather / base.min(axis=1)) * 100)
    assert cut == "22-45"
    assert f"{cut}% on each region's best simple baseline" in TLDR
    assert f"{cut}% below the best simple baseline" in RECOMMENDATIONS


def test_cut_on_same_day_last_week(base, weather):
    cut = span((1 - weather / base["seasonal_naive"]) * 100)
    assert cut == "36-55"
    assert f"cuts its error by {cut}%" in TLDR
    assert f"{cut}% below \"same day last week\"" in RECOMMENDATIONS


def test_same_day_last_week_comes_last_in_4_of_5(base):
    assert (base.idxmax(axis=1) == "seasonal_naive").sum() == 4


def test_yesterday_against_same_day_last_week(base):
    assert f"takes off {span((1 - base['naive'] / base['seasonal_naive']) * 100)}%" in TLDR


def test_south_australia_steps():
    d = day_ahead("weather_metrics.csv").loc["SA1"] * 100
    assert f"from {d['blend']:.1f}% to {d['weather_temp']:.1f}%" in TLDR
    assert f"to {d['weather']:.1f}%" in TLDR


def coverage(recipe: str) -> pd.DataFrame:
    c = pd.read_csv(RESULTS / "buffer_coverage.csv", dtype={"level": str})
    return c[(c["horizon"] == "day") & (c["level"].astype(float) == 0.95) & (c["recipe"] == recipe)]


def test_buffer_coverage():
    assert len(coverage("shaped")) == 10
    assert f"covers only {coverage('halfhour')['days_covered'].mean() * 100:.0f}% of days" in TLDR
    shaped = coverage("shaped")
    assert f"covers {shaped['days_covered'].mean() * 100:.1f}%" in TLDR
    assert f"({shaped['Summer'].mean() * 100:.1f}% of days)" in TLDR
    assert f"({shaped['Winter'].mean() * 100:.1f}%)" in TLDR
    assert f"{shaped['Summer'].mean() * 100:.1f}% of summer days" in RECOMMENDATIONS
    assert f"{shaped['Winter'].mean() * 100:.1f}% of winter ones" in RECOMMENDATIONS


def test_buffer_saving_on_flat():
    keys = ["region", "side"]
    shaped = coverage("shaped").set_index(keys)["mean_buffer_mw"]
    flat = coverage("flat").set_index(keys)["mean_buffer_mw"]
    cut = span((1 - shaped / flat) * 100)
    assert f"on {cut}% less cover" in TLDR
    assert f"on {cut}% less cover than flat" in RECOMMENDATIONS


@pytest.mark.parametrize("readme, report, phrase", [
    ("-284 MW", "SOURCE_PROFILE.md", "-284 MW"),
    ("7 of 10 region-horizon pairs", "BACKTEST.md", "7 of 10"),
    ("42-47%", "WEATHER.md", "42%-47%"),
    ("352 MW", "WEATHER.md", "| NSW1 | +249 | +216 | -352 |"),
    ("5 of the forecast's 15 worst", "WEATHER.md", "5 of the 15 worst"),
    ("8-69%", "BUFFERS.md", "8%-69%"),
    ("30-56% less buffer", "BUFFERS.md", "30%-56%"),
    ("2.5-4.6×", "BUFFERS.md", "2.5×-4.6×"),
    ("38% of breaches on 25% of days", "BUFFERS.md", "38% of breaches, on 25% of days"),
    ("1.4-2.3× the buffer", "BUFFERS.md", "1.4×-2.3× the buffer"),
    ("38-45% smaller", "BUFFERS.md", "45% and 38% less"),
    ("$852k", "BUFFERS.md", "$852k"),
    ("4-23%", "BUFFERS.md", "4%-23%"),
    ("only 6% of buffer breaches", "BUFFERS.md", "14 of 221 breaches (6%"),
])
def test_claims_taken_from_a_report_are_in_that_report(readme, report, phrase):
    assert readme in TLDR + RECOMMENDATIONS
    assert phrase in (RESULTS / report).read_text()
