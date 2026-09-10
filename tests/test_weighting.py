"""Frequency-weighting tests, validated against IEC 61672-1:2013.

The regression test that matters most is
:func:`test_c_weighting_corner_frequencies_are_not_scaled` -- the legacy code scaled the
C-weighting poles by ``0.062*pi`` instead of ``2*pi`` and nothing caught it for four
years across three repositories.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from splmeter.weighting import (
    REFERENCE_FREQUENCY,
    WeightingFilter,
    analytic_weighting_db,
    apply_weighting,
    max_valid_frequency,
    weighting_response_db,
    weighting_sos,
)

DATA = Path(__file__).parent / "data" / "iec61672_tolerances.csv"


def _tolerance_table():
    """IEC 61672-1 nominal weightings and class tolerance limits.

    ``-99999`` in the source stands for an unbounded lower limit (the standard prints
    a dash), which is why the deep high-frequency droop of a bilinear-designed filter is
    permitted at 16 kHz and above.
    """
    rows = list(csv.DictReader(DATA.open(encoding="utf-8-sig")))
    nominal = np.array([float(r["Frequency"]) for r in rows])
    # Exact base-ten midband frequencies; the standard tabulates nominal labels but
    # specifies evaluation at the exact frequencies.
    exact = 1000.0 * 10 ** (np.round(10 * np.log10(nominal / 1000.0)) / 10.0)
    return rows, nominal, exact


def test_analytic_matches_iec_table():
    """The pole-zero network must reproduce the standard's tabulated values.

    The table is rounded to 0.1 dB, so agreement to within 0.06 dB is exact agreement.
    """
    rows, _, exact = _tolerance_table()
    for kind in ("A", "C"):
        tabulated = np.array([float(r[kind]) for r in rows])
        computed = analytic_weighting_db(kind, exact)
        assert np.abs(computed - tabulated).max() < 0.06, kind


@pytest.mark.parametrize("fs", [22050.0, 44100.0, 48000.0, 96000.0])
@pytest.mark.parametrize("kind", ["A", "C"])
def test_digital_filter_meets_class_1(kind, fs):
    """Class 1 compliance within the filter's documented valid band.

    Frequencies above ``max_valid_frequency`` are excluded because the bilinear
    transform maps infinite analog frequency onto Nyquist: every filter designed this
    way collapses there.  At 48 kHz nothing in the audible range is excluded that the
    standard bounds; at 16 kHz the 8 kHz band sits at 0.99x Nyquist and is unmeasurable.
    """
    rows, _, exact = _tolerance_table()
    limit = max_valid_frequency(fs)
    tested = 0
    for row, frequency in zip(rows, exact):
        if frequency > limit:
            continue
        tested += 1
        deviation = weighting_response_db(kind, [frequency], fs)[0] - float(row[kind])
        assert float(row["Class 1 Lower"]) <= deviation <= float(row["Class 1 Upper"]), (
            f"{kind}-weighting at {row['Frequency']} Hz, fs={fs}: "
            f"deviation {deviation:+.2f} dB"
        )
    assert tested >= 10, f"only {tested} bands tested at fs={fs}"


@pytest.mark.parametrize("kind", ["A", "C"])
def test_full_range_class_1_at_archive_sample_rate(kind):
    """At 48 kHz the filter must meet Class 1 across the whole tabulated range.

    This is the claim that matters for the project: every recording in the archive is
    48 kHz, so nothing is excluded here for being near Nyquist.
    """
    rows, _, exact = _tolerance_table()
    for row, frequency in zip(rows, exact):
        deviation = weighting_response_db(kind, [frequency], 48000.0)[0] - float(row[kind])
        assert float(row["Class 1 Lower"]) <= deviation <= float(row["Class 1 Upper"]), (
            f"{kind} at {row['Frequency']} Hz: {deviation:+.2f} dB"
        )


# 8 kHz and 16 kHz deliberately sit below MIN_SUPPORTED_FS and warn; that is asserted
# separately in test_warns_below_minimum_supported_sample_rate.
@pytest.mark.filterwarnings("ignore:.*Class 1.*:RuntimeWarning")
@pytest.mark.parametrize("fs", [8000.0, 16000.0, 22050.0, 44100.0, 48000.0, 96000.0])
@pytest.mark.parametrize("kind", ["A", "C", "Z"])
def test_reference_frequency_is_exactly_unity_gain(kind, fs):
    """0 dB at 1 kHz at every sample rate.

    The previous implementation normalised only the analog prototype, leaving a
    sample-rate dependent offset on the 1 kHz reference (+0.157 dB at 8 kHz, +0.039 dB
    at 16 kHz) -- a systematic error on every level it ever reported.
    """
    response = weighting_response_db(kind, [REFERENCE_FREQUENCY], fs)[0]
    assert abs(response) < 1e-6


def test_c_weighting_corner_frequencies_are_not_scaled():
    """Regression: the legacy ``0.062*pi`` pole scaling produced a 32x corner shift.

    C-weighting is defined by corners at 20.6 Hz and 12.2 kHz, giving -3 dB at roughly
    31.5 Hz and 8 kHz.  The broken filter placed them near 0.64 Hz and 378 Hz, so it was
    essentially flat below 380 Hz and rolling off hard above -- the opposite of C.
    """
    assert analytic_weighting_db("C", [31.5])[0] == pytest.approx(-3.0, abs=0.1)
    assert analytic_weighting_db("C", [7943.0])[0] == pytest.approx(-3.0, abs=0.1)
    # And it must still be near-flat across the mid band, which the broken one was not.
    midband = analytic_weighting_db("C", [200.0, 500.0, 1000.0, 2000.0])
    assert np.abs(midband).max() < 0.25


def test_z_weighting_is_flat_and_pass_through():
    frequencies = np.array([10.0, 100.0, 1000.0, 10000.0])
    assert np.allclose(analytic_weighting_db("Z", frequencies), 0.0)
    rng = np.random.default_rng(20260910)
    signal_in = rng.standard_normal(4096).astype(np.float32)
    assert np.allclose(apply_weighting(signal_in, "Z", 48000.0), signal_in)


@pytest.mark.parametrize("kind", ["A", "C", "Z"])
def test_streaming_matches_whole_signal(kind):
    """Chunked filtering with carried state must equal filtering the whole signal.

    Without state carry-over each chunk restarts from rest and begins with a settling
    transient, which is why the old module could not stream a long recording at all.
    """
    rng = np.random.default_rng(7)
    signal_in = rng.standard_normal(48000).astype(np.float32)
    whole = apply_weighting(signal_in, kind, 48000.0)

    streaming = WeightingFilter(kind, 48000.0)
    chunks = [streaming(part) for part in np.array_split(signal_in, 7)]
    assert np.allclose(np.concatenate(chunks), whole, atol=1e-5)


def test_filter_reset_clears_memory():
    rng = np.random.default_rng(11)
    signal_in = rng.standard_normal(8192).astype(np.float32)
    weighting = WeightingFilter("A", 48000.0)
    first = weighting(signal_in)
    weighting.reset()
    assert np.allclose(weighting(signal_in), first, atol=1e-6)


def test_sine_at_reference_frequency_passes_through_unchanged():
    """A 1 kHz tone must survive A-weighting with its amplitude intact."""
    fs = 48000.0
    t = np.arange(int(fs)) / fs
    tone = np.sin(2 * np.pi * REFERENCE_FREQUENCY * t).astype(np.float32)
    weighted = apply_weighting(tone, "A", fs)
    # Skip the settling transient at the start of the filtered signal.
    steady = weighted[4800:]
    assert np.sqrt(np.mean(steady**2)) == pytest.approx(1 / np.sqrt(2), rel=1e-3)


@pytest.mark.parametrize("bad", ["B", "", "AC", "a-weighting", None])
def test_rejects_unknown_weighting(bad):
    """The legacy check was ``if curve not in 'AC'`` -- a substring test that accepted
    both ``''`` and ``'AC'`` and then crashed with an UnboundLocalError."""
    with pytest.raises((ValueError, AttributeError, TypeError)):
        weighting_sos(bad, 48000.0)


def test_rejects_sample_rate_below_reference():
    """1 kHz must lie below Nyquist or the weighting cannot be normalised."""
    with pytest.raises(ValueError):
        weighting_sos("A", 1500.0)


@pytest.mark.parametrize("kind", ["A", "C"])
def test_warns_below_minimum_supported_sample_rate(kind):
    """Below 2*F4 the 12.2 kHz pole pair is above Nyquist and cannot be realised."""
    with pytest.warns(RuntimeWarning, match="Class 1"):
        weighting_sos(kind, 16000.0)


def test_no_warning_at_archive_sample_rate():
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        weighting_sos("A", 48000.0)
