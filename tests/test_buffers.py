"""Hand-worked checks of the buffer, cost and battery arithmetic."""
import numpy as np
import pandas as pd
import pytest

from demand import buffers as bf


def day(fill=0.0, **slots) -> np.ndarray:
    """A day of 48 half-hours, set by s<slot>=value."""
    row = np.full(48, float(fill))
    for name, value in slots.items():
        row[int(name[1:])] = value
    return row


@pytest.mark.parametrize("n, level, rank", [(19, .95, 19), (39, .95, 38), (9, .9, 9), (304, .95, 290), (30, .95, 30), (5, .95, 5)])
def test_rank_quantile_picks_the_stated_rank(n, level, rank):
    values = np.arange(1.0, n + 1)[::-1]  # unsorted on purpose
    assert bf.rank_quantile(values, level) == rank


def test_seasons_follow_the_southern_calendar():
    days = pd.to_datetime(["2025-12-01", "2026-02-28", "2026-03-01", "2026-06-15", "2026-09-30", "2026-11-30"])
    assert list(bf.season_of(days)) == ["Summer", "Summer", "Autumn", "Winter", "Spring", "Spring"]


def test_blocks_cover_every_half_hour_once():
    slots = [s for a, b in bf.BLOCKS.values() for s in range(a, b)]
    assert slots == list(range(48))


def test_worst_miss_reads_each_side_and_block():
    miss = np.stack([day(s3=50, s25=-80), day(s3=-10, s25=30)])
    assert list(bf.worst_miss(miss, bf.UP)) == [50, 30]
    assert list(bf.worst_miss(miss, bf.DOWN)) == [80, 10]
    assert list(bf.worst_miss(miss, bf.UP, bf.BLOCKS["10:00-16:00"])) == [0, 30]
    assert list(bf.worst_miss(miss, bf.DOWN, bf.BLOCKS["00:00-06:00"])) == [0, 10]


def test_a_day_is_covered_only_if_every_half_hour_is():
    miss = np.stack([day(s3=99), day(s3=100), day(s3=100, s40=101)])
    assert list(bf.covered(miss, np.full(miss.shape, 100.0), bf.UP)) == [True, True, False]


def test_covered_on_a_block_ignores_the_rest_of_the_day():
    miss = np.stack([day(s25=999)])
    buffer = np.full(miss.shape, 100.0)
    assert bf.covered(miss, buffer, bf.UP, bf.BLOCKS["00:00-06:00"])[0]
    assert not bf.covered(miss, buffer, bf.UP, bf.BLOCKS["10:00-16:00"])[0]


def test_a_half_hour_percentile_undercovers_days():
    # 20 days, each with one 100 MW half-hour at a different time: 96% of half-hours are 0, every day is hit
    miss = np.zeros((20, 48))
    miss[np.arange(20), np.arange(20)] = 100
    assert bf.halfhour_buffer(miss, bf.UP, .95) == 0
    assert bf.flat_buffer(miss, bf.UP, .95) == 100


def test_block_buffers_are_per_block_and_per_group():
    miss = np.zeros((4, 48))
    miss[:, 25] = [10, 20, 30, 40]  # midday
    miss[:, 3] = [1, 2, 3, 4]       # overnight
    got = bf.block_buffers(miss, np.array(["a", "a", "b", "b"]), bf.UP, .5)
    # rank ceil(3 x .5) = 2 of 2 in each group
    assert got["a"][25] == 20 and got["b"][25] == 40
    assert got["a"][3] == 2 and got["b"][3] == 4
    assert got["a"][40] == 0


def test_stretch_leaves_buffers_alone_when_they_already_cover_the_day():
    # every block peaks on the same day, so the block values already hold for the whole day
    miss = np.ones((10, 48))
    miss[0] = 4
    miss[0, 25] = 8
    groups = np.array(["all"] * 10)
    plain = bf.block_buffers(miss, groups, bf.UP, .9)["all"]  # rank 10 of 10: each block's maximum
    assert plain[3] == 4 and plain[25] == 8
    assert bf.stretch(miss, groups, bf.UP, .9)["all"] == pytest.approx(plain)


def test_stretch_scales_up_when_blocks_miss_on_different_days():
    # 20 days at 2 MW in every block, except one 3 MW day in each of two blocks, on different days.
    # Rank 19 of 20 leaves that extreme out of each block's buffer, so two days are uncovered
    miss = np.full((20, 48), 2.0)
    miss[0, 3], miss[1, 25] = 3, 3
    groups = np.array(["all"] * 20)
    plain = bf.block_buffers(miss, groups, bf.UP, .9)["all"]
    assert plain[3] == 2 and plain[25] == 2
    got = bf.stretch(miss, groups, bf.UP, .9)["all"]
    # each day's worst ratio to its buffer: 1.5, 1.5, then 1 on 18 days. Rank 19 of 20 is 1.5
    assert got[3] == pytest.approx(3) and got[25] == pytest.approx(3)
    assert bf.covered(miss, bf.spread_out({"all": got}, groups), bf.UP).all()


def test_stretch_refuses_a_buffer_it_cant_scale():
    with pytest.raises(ValueError):
        bf.stretch(np.zeros((5, 48)), np.array(["a"] * 5), bf.UP, .5)


def test_spread_out_follows_the_days():
    table = {"a": np.full(48, 1.0), "b": np.full(48, 2.0)}
    assert list(bf.spread_out(table, np.array(["b", "a", "b"]))[:, 0]) == [2, 1, 2]


def test_imbalance_cost_by_hand():
    # short 100 MW at $300, hedge $100, 1% share: 1 MW x $200 x 0.5 h = $100
    assert bf.imbalance_cost(np.array([100.0]), np.array([300.0]), 100)[0] == pytest.approx(100)
    # long 50 MW at -$20: 0.5 MW x $120 x 0.5 h = $30, a loss too
    assert bf.imbalance_cost(np.array([-50.0]), np.array([-20.0]), 100)[0] == pytest.approx(30)
    # short while spot is below the hedge is a gain
    assert bf.imbalance_cost(np.array([100.0]), np.array([20.0]), 100)[0] == pytest.approx(-40)
    assert bf.imbalance_cost(np.array([0.0]), np.array([9999.0]), 100)[0] == 0


def test_imbalance_cost_is_the_bill_difference():
    actual, forecast, price, hedge = 120.0, 100.0, 250.0, 80.0
    bill = actual * price - (price - hedge) * forecast
    perfect = hedge * actual
    got = bf.imbalance_cost(np.array([actual - forecast]), np.array([price]), hedge, share=1)[0]
    assert got == pytest.approx((bill - perfect) * .5)


def test_rolling_mean_by_hand():
    signal = np.arange(48.0)[None, :]
    got = bf.rolling_mean(signal, 8)
    assert got.shape == (1, 41)
    assert got[0, 0] == pytest.approx(3.5) and got[0, 40] == pytest.approx(43.5)


def test_pick_windows_finds_the_trough_then_the_peak():
    signal = np.full((1, 48), 50.0)
    signal[0, 20:28] = 0      # 10:00-14:00
    signal[0, 34:42] = 100    # 17:00-21:00
    charge, discharge = bf.pick_windows(signal)
    assert (charge[0], discharge[0]) == (20, 34)


def test_pick_windows_never_discharges_before_charging():
    signal = np.full((1, 48), 50.0)
    signal[0, 4:12] = 100     # the peak comes first
    signal[0, 30:38] = 0      # and the trough later
    charge, discharge = bf.pick_windows(signal)
    assert discharge[0] >= charge[0] + 8


def test_pick_windows_never_overlaps():
    rng = np.random.default_rng(5)
    charge, discharge = bf.pick_windows(rng.normal(size=(200, 48)))
    assert (discharge >= charge + 8).all()


def test_pick_windows_matches_a_brute_force_search():
    rng = np.random.default_rng(6)
    signal = rng.normal(size=(50, 48))
    means = bf.rolling_mean(signal)
    charge, discharge = bf.pick_windows(signal)
    for i in range(50):
        best = max(means[i, d] - means[i, c] for c in range(41) for d in range(c + 8, 41))
        assert means[i, discharge[i]] - means[i, charge[i]] == pytest.approx(best)


def test_captured_spread_reads_the_price_not_the_signal():
    price = np.zeros((2, 48))
    price[:, 34:42] = 200
    got = bf.captured_spread(price, np.array([0, 0]), np.array([34, 8]))
    assert list(got) == [200, 0]
