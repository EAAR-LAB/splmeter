"""Exposure-index tests, including the Lden/CNEL distinction the legacy code lost."""

from __future__ import annotations

import numpy as np
import pytest

from splmeter.exposure import (
    CNEL_PERIODS,
    LDEN_PERIODS,
    LDN_PERIODS,
    Period,
    cnel,
    composite_index,
    energy_mean,
    ldn,
    lden,
    period_levels,
)


def hourly_series(level_by_hour):
    """One level per hour of a day, with matching hour labels."""
    hours = np.arange(24)
    levels = np.array([level_by_hour(h) for h in hours], dtype=float)
    return levels, hours


def test_period_definitions_cover_exactly_24_hours():
    for periods in (LDEN_PERIODS, CNEL_PERIODS, LDN_PERIODS):
        assert sum(p.hours for p in periods) == 24


def test_period_hour_spans_match_the_standards():
    assert [p.hours for p in LDEN_PERIODS] == [12, 4, 8]
    assert [p.hours for p in CNEL_PERIODS] == [12, 3, 9]
    assert [p.hours for p in LDN_PERIODS] == [15, 9]


def test_periods_partition_the_day_without_overlap():
    for periods in (LDEN_PERIODS, CNEL_PERIODS, LDN_PERIODS):
        hours = np.arange(24)
        counts = sum(p.contains(hours).astype(int) for p in periods)
        assert np.all(counts == 1)


def test_night_period_wraps_past_midnight():
    night = Period("night", 23, 7)
    assert night.contains(np.array([23, 0, 3, 6])).all()
    assert not night.contains(np.array([7, 12, 22])).any()
    assert night.hours == 8


def test_uniform_level_gives_that_level_before_penalties():
    levels, hours = hourly_series(lambda h: 60.0)
    result = period_levels(levels, hours)
    assert all(v == pytest.approx(60.0) for v in result.values())


def test_penalties_raise_the_index_above_the_flat_level():
    """A flat 60 dB day still yields Lden above 60 because of the night penalty."""
    levels, hours = hourly_series(lambda h: 60.0)
    # 10*log10((12*10^6 + 4*10^6.5 + 8*10^7)/24)
    expected = 10 * np.log10(
        (12 * 10**6.0 + 4 * 10**6.5 + 8 * 10**7.0) / 24
    )
    assert lden(levels, hours) == pytest.approx(expected, abs=1e-9)
    assert lden(levels, hours) > 60.0


def test_lden_and_cnel_differ_on_the_same_data():
    """Regression: the legacy notebook computed the Lden partition and called it CNEL.

    The 22:00 hour is exactly what separates the two definitions -- Lden counts it as
    evening (+5 dB), CNEL as night (+10 dB) -- so a loud event confined to that hour
    drives the indices apart.  A profile that merely differs in shape can leave them
    coincidentally within 0.01 dB of each other, which would make this test pass for
    the wrong reason.
    """
    levels, hours = hourly_series(lambda h: 85.0 if h == 22 else 50.0)
    difference = cnel(levels, hours) - lden(levels, hours)
    assert difference > 1.0, f"Lden and CNEL differ by only {difference:.3f} dB"


def test_ldn_has_no_evening_period():
    assert {p.name for p in LDN_PERIODS} == {"day", "night"}
    levels, hours = hourly_series(lambda h: 60.0)
    expected = 10 * np.log10((15 * 10**6.0 + 9 * 10**7.0) / 24)
    assert ldn(levels, hours) == pytest.approx(expected, abs=1e-9)


def test_loud_night_dominates_the_index():
    quiet_night, hours = hourly_series(lambda h: 70.0 if 7 <= h < 23 else 40.0)
    loud_night, _ = hourly_series(lambda h: 70.0 if 7 <= h < 23 else 70.0)
    assert lden(loud_night, hours) > lden(quiet_night, hours) + 5


def test_energy_mean_ignores_non_finite():
    assert energy_mean(np.array([60.0, -np.inf, np.nan, 60.0])) == pytest.approx(60.0)
    assert energy_mean(np.array([-np.inf])) == -np.inf


def test_missing_period_is_excluded_not_treated_as_silence():
    """A day with no night data must not be scored as though the night were silent."""
    hours = np.arange(7, 19)
    levels = np.full(hours.size, 65.0)
    result = lden(levels, hours)
    # Only the day period contributes, so the index is the day level itself.
    assert result == pytest.approx(65.0, abs=1e-9)


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError):
        period_levels(np.zeros(10), np.zeros(9))


def test_composite_of_empty_input_is_minus_inf():
    assert composite_index(np.array([]), np.array([])) == -np.inf
