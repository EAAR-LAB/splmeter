"""Time-domain metric tests against analytic reference values.

Several of these are regression tests for defects that reached published-adjacent
numbers: the signed-maximum peak, the decimated peak, the ``-inf`` first sample, the
arithmetic mean of decibels, and the inverted exceedance levels.
"""

from __future__ import annotations

import numpy as np
import pytest

from splmeter.core import P_REF
from splmeter.metrics import (
    IMPULSE_DECAY_TAU,
    TIME_CONSTANTS,
    energy_average,
    exceedance_levels,
    frame_mean_square,
    leq,
    lmax,
    lmin,
    lpeak,
    sel,
    time_weight,
)

FS = 48000.0


def sine(amplitude: float, frequency: float, seconds: float, fs: float = FS):
    t = np.arange(int(round(fs * seconds))) / fs
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float64)


def test_leq_of_sine_matches_analytic_rms():
    """For a sine of amplitude A, Leq = 20*log10((A/sqrt2)/p_ref)."""
    amplitude = np.sqrt(2.0)  # 1 Pa RMS -> 93.98 dB
    result = leq(sine(amplitude, 1000.0, 5.0), FS, window_s=1.0)
    expected = 20 * np.log10((amplitude / np.sqrt(2)) / P_REF)
    assert result.shape == (5,)
    assert np.allclose(result, expected, atol=1e-6)
    assert expected == pytest.approx(93.979, abs=0.01)


def test_leq_first_sample_is_real_not_minus_inf():
    """Regression: the old code zero-padded, so the first window was pure silence.

    That produced ``-inf``, which the ETL clipped to ``0.0`` and the notebooks then
    discarded as an outlier -- losing the first second of every packet.
    """
    result = leq(sine(1.0, 1000.0, 3.0), FS)
    assert np.all(np.isfinite(result))
    assert result[0] == pytest.approx(result[1], abs=1e-6)


def test_partial_trailing_frame_is_dropped_not_padded():
    """2.5 s at 1 s windows yields 2 frames, not 3 with a padded final one."""
    assert leq(sine(1.0, 1000.0, 2.5), FS, window_s=1.0).shape == (2,)
    assert frame_mean_square(sine(1.0, 1000.0, 0.5), FS, 1.0).size == 0


def test_leq_silence_is_minus_inf_not_zero():
    """Silence is a legitimate measurement, distinct from a dropout."""
    result = leq(np.zeros(int(FS)), FS)
    assert result[0] == -np.inf


def test_overlapping_windows_match_tumbling():
    """The prefix-sum path and the reshape path must agree where they overlap."""
    pressure = sine(0.5, 997.0, 4.0)
    tumbling = frame_mean_square(pressure, FS, 1.0)
    overlapping = frame_mean_square(pressure, FS, 1.0, hop_s=0.5)
    assert np.allclose(tumbling, overlapping[::2], rtol=1e-9)


# --- peak level -------------------------------------------------------------------

def test_lpeak_of_sine_is_3db_above_leq():
    """A sine's peak exceeds its RMS by exactly 10*log10(2) = 3.01 dB."""
    pressure = sine(np.sqrt(2.0), 1000.0, 2.0)
    assert (lpeak(pressure, FS) - leq(pressure, FS)) == pytest.approx(3.0103, abs=1e-3)


def test_lpeak_uses_absolute_value():
    """Regression: the old code used ``np.max``, not ``np.max(np.abs(...))``.

    A window whose largest excursion is a rarefaction returned the wrong value, and an
    entirely negative window returned ``-inf`` -- precisely the asymmetric transients
    (impacts, gunshots) that peak level exists to capture.
    """
    negative = -np.abs(sine(0.02, 500.0, 2.0))
    result = lpeak(negative, FS)
    assert np.all(np.isfinite(result))
    assert result[0] == pytest.approx(20 * np.log10(0.02 / P_REF), abs=1e-6)


def test_lpeak_is_symmetric_under_sign_flip():
    pressure = sine(0.3, 250.0, 2.0) + 0.1
    assert np.allclose(lpeak(pressure, FS), lpeak(-pressure, FS))


def test_lpeak_catches_a_single_sample_transient():
    """Regression: the old code decimated 48:1 before peak detection.

    Sampling 1 of every 48 samples misses a short transient almost always.  Here the
    spike sits at an index that a 48:1 decimator would skip.
    """
    pressure = np.full(int(FS), 1e-4)
    pressure[17] = 5.0  # not a multiple of 48
    assert lpeak(pressure, FS)[0] == pytest.approx(20 * np.log10(5.0 / P_REF), abs=1e-6)


# --- time weighting ---------------------------------------------------------------

@pytest.mark.parametrize("weighting,tau", [("F", 0.125), ("S", 1.0)])
def test_exponential_step_response_reaches_1_minus_1_over_e(weighting, tau):
    """After one time constant a step reaches (1 - 1/e) of its final value."""
    steady = 0.5
    # Run for 10 time constants so the tail is genuinely settled: at 5 tau a Slow
    # detector has only reached 99.3% of final value.
    pressure = np.full(int(FS * 10 * tau), steady)
    weighted = time_weight(pressure, FS, weighting)
    at_tau = weighted[int(FS * tau)]
    assert at_tau / steady**2 == pytest.approx(1 - 1 / np.e, rel=2e-3)
    assert weighted[-1] / steady**2 == pytest.approx(1.0, rel=1e-3)


def test_fast_settles_quicker_than_slow():
    pressure = np.full(int(FS * 2), 1.0)
    fast = time_weight(pressure, FS, "F")
    slow = time_weight(pressure, FS, "S")
    assert fast[int(FS * 0.25)] > slow[int(FS * 0.25)]


def test_impulse_decays_at_2_9_db_per_second():
    """IEC 61672-1 specifies a 2.9 dB/s decay for the Impulse detector."""
    pressure = np.zeros(int(FS * 4))
    pressure[: int(FS * 0.05)] = 1.0
    weighted = time_weight(pressure, FS, "I")
    one, two = weighted[int(FS * 2)], weighted[int(FS * 3)]
    assert 10 * np.log10(one / two) == pytest.approx(2.9, abs=0.05)


def test_decaying_peak_hold_matches_naive_recursion():
    """The blockwise vectorisation must equal the scalar recursion it replaces."""
    from splmeter.metrics import _decaying_peak_hold

    rng = np.random.default_rng(3)
    values = rng.random(5000) ** 3
    fs, tau = 1000.0, 0.5
    fast = _decaying_peak_hold(values, fs, tau, block=256)
    decay = np.exp(-1.0 / (fs * tau))
    naive, previous = np.empty_like(values), 0.0
    for i, v in enumerate(values):
        previous = max(v, decay * previous)
        naive[i] = previous
    assert np.allclose(fast, naive, rtol=1e-9)


def test_lmax_is_a_real_maximum_over_the_frame():
    """Regression: the old ETL took a max over a single sample -- a no-op.

    Its ``lafmax`` column is really ``laf``: the instantaneous Fast-weighted level.
    """
    pressure = np.full(int(FS * 2), 0.01)
    pressure[int(FS * 0.5) : int(FS * 0.5) + 4800] = 1.0
    assert lmax(pressure, FS, 1.0, "F")[0] > lmax(pressure, FS, 1.0, "F")[1] - 1e-9
    assert lmax(pressure, FS, 1.0, "F")[0] > 80.0


def test_lmin_is_below_lmax():
    rng = np.random.default_rng(5)
    pressure = rng.standard_normal(int(FS * 3)) * 0.05
    assert np.all(lmin(pressure, FS) < lmax(pressure, FS))


# --- exposure and statistics ------------------------------------------------------

def test_sel_equals_leq_plus_10log10_duration():
    duration = 4.0
    pressure = sine(np.sqrt(2.0), 1000.0, duration)
    expected = leq(pressure, FS, window_s=duration)[0] + 10 * np.log10(duration)
    assert sel(pressure, FS) == pytest.approx(expected, abs=1e-6)


def test_energy_average_is_not_the_arithmetic_mean():
    """Regression: the legacy notebooks averaged decibels arithmetically.

    Energy averaging is dominated by the loudest contributions, so it always sits at or
    above the arithmetic mean -- here by more than 6 dB.
    """
    levels = np.array([60.0, 60.0, 60.0, 80.0])
    # 10*log10((3e6 + 1e8)/4) = 74.108 dB, against an arithmetic mean of 65 dB.
    expected = 10 * np.log10((3 * 10**6 + 10**8) / 4)
    assert energy_average(levels) == pytest.approx(expected, abs=1e-9)
    assert energy_average(levels) == pytest.approx(74.108, abs=0.01)
    assert energy_average(levels) > levels.mean() + 9


def test_energy_average_of_equal_levels_is_that_level():
    assert energy_average(np.full(10, 63.5)) == pytest.approx(63.5, abs=1e-9)


def test_energy_average_ignores_minus_inf():
    assert energy_average(np.array([70.0, -np.inf, 70.0])) == pytest.approx(70.0)


def test_exceedance_levels_are_ordered_correctly():
    """Regression: the legacy notebook computed ``L5 = quantile(0.05)`` -- that is L95.

    Ln is the level EXCEEDED n% of the time, so L90 is the quiet background and L10 sits
    near the top.
    """
    levels = np.linspace(40.0, 90.0, 1001)
    result = exceedance_levels(levels, [5, 10, 50, 90])
    assert result[90] < result[50] < result[10] < result[5]
    assert result[50] == pytest.approx(65.0, abs=0.1)
    assert result[90] == pytest.approx(45.0, abs=0.1)  # exceeded 90% of the time
    assert result[10] == pytest.approx(85.0, abs=0.1)


def test_exceedance_levels_survive_silence():
    levels = np.array([50.0, -np.inf, 60.0, 70.0])
    assert np.isfinite(exceedance_levels(levels, [50])[50])


@pytest.mark.parametrize("bad", ["X", "fast", ""])
def test_rejects_unknown_time_weighting(bad):
    with pytest.raises(ValueError):
        time_weight(np.zeros(100), FS, bad)


def test_all_iec_time_constants_present():
    assert TIME_CONSTANTS == {"F": 0.125, "S": 1.0, "I": 0.035}
    assert IMPULSE_DECAY_TAU == pytest.approx(1.4979, abs=1e-3)
