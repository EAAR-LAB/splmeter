"""Band-analysis tests.

The headline test is :func:`test_band_sum_matches_broadband_leq`: summing band energies
must reproduce the broadband Leq of the same signal.  The old implementation could not
satisfy it -- its bands were uniformly 3.01 dB high because it used a peak-amplitude
rather than RMS convention -- and its own test suite asserted the wrong value as
expected, so the discrepancy was never visible.
"""

from __future__ import annotations

import numpy as np
import pytest

from splmeter.bands import (
    G,
    OCTAVE,
    THIRD_OCTAVE,
    TWELFTH_OCTAVE,
    BandSet,
    band_levels,
    band_sum_level,
    spectrogram,
)
from splmeter.core import P_REF
from splmeter.metrics import leq

FS = 48000.0


def sine(amplitude, frequency, seconds=2.0, fs=FS):
    t = np.arange(int(round(fs * seconds))) / fs
    return amplitude * np.sin(2 * np.pi * frequency * t)


def test_standard_band_counts_and_midbands():
    assert len(OCTAVE) == 11
    assert len(THIRD_OCTAVE) == 33
    # Exact base-ten midbands sit close to the familiar nominal labels.
    assert OCTAVE.midbands[np.argmin(abs(OCTAVE.midbands - 1000))] == pytest.approx(1000.0)
    nominal_third = [12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100]
    for nominal, exact in zip(nominal_third, THIRD_OCTAVE.midbands[:10]):
        assert exact == pytest.approx(nominal, rel=0.02)


def test_band_edges_are_geometric():
    for bands in (OCTAVE, THIRD_OCTAVE, TWELFTH_OCTAVE):
        ratio = bands.upper_edges / bands.lower_edges
        assert np.allclose(ratio, G ** (1 / bands.fraction))
        # Adjacent bands meet: one band's upper edge is the next one's lower edge.
        assert np.allclose(bands.upper_edges[:-1], bands.lower_edges[1:], rtol=1e-12)


def test_tone_lands_in_the_expected_band():
    levels = band_levels(sine(1.0, 1000.0), FS, THIRD_OCTAVE)
    loudest = int(np.nanargmax(levels[0]))
    assert THIRD_OCTAVE.midbands[loudest] == pytest.approx(1000.0, rel=0.01)


def test_tone_band_level_uses_rms_not_peak_convention():
    """Regression: the old code reported every band 3.01 dB high.

    A unit-amplitude tone has RMS 1/sqrt(2), so its band level is
    20*log10((1/sqrt2)/p_ref) = 90.97 dB.  The old implementation gave 93.98 dB, and
    encoded that value in its own test as correct.
    """
    levels = band_levels(sine(1.0, 1000.0), FS, THIRD_OCTAVE)
    peak_band = float(np.nanmax(levels[0]))
    expected = 20 * np.log10((1 / np.sqrt(2)) / P_REF)
    assert expected == pytest.approx(90.97, abs=0.01)
    assert peak_band == pytest.approx(expected, abs=0.15)
    # And explicitly NOT the old value.
    assert abs(peak_band - (expected + 3.01)) > 1.0


@pytest.mark.parametrize("bands", [OCTAVE, THIRD_OCTAVE, TWELFTH_OCTAVE])
def test_band_sum_matches_broadband_leq(bands):
    """Summed band energy must reconcile with the broadband Leq of the same signal.

    Band-limited pink-ish noise is used so that essentially all the energy falls inside
    the band set's range; the residual difference is the energy outside it.
    """
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(int(FS * 2)) * 0.02
    # Restrict to well inside the band set so edge losses do not dominate.
    from scipy.signal import butter, sosfilt

    sos = butter(4, [100.0, 8000.0], btype="band", fs=FS, output="sos")
    pressure = sosfilt(sos, noise)

    broadband = leq(pressure, FS, window_s=1.0)
    summed = band_sum_level(band_levels(pressure, FS, bands))
    assert np.allclose(summed, broadband, atol=0.35), (
        f"{bands}: summed {summed} vs broadband {broadband}"
    )


def test_unresolved_low_bands_are_masked():
    """A 1/12-octave band at 12.5 Hz is 0.72 Hz wide, below 1 s FFT resolution."""
    levels = band_levels(sine(0.1, 1000.0), FS, TWELFTH_OCTAVE, window_s=1.0)
    resolvable = TWELFTH_OCTAVE.resolvable(FS, int(FS))
    assert not resolvable[0], "narrowest band should be unresolvable at 1 s"
    assert np.isnan(levels[0, 0])
    assert np.isfinite(levels[0, resolvable][0])


def test_longer_frames_resolve_more_bands():
    short = TWELFTH_OCTAVE.resolvable(FS, int(FS * 1.0)).sum()
    long = TWELFTH_OCTAVE.resolvable(FS, int(FS * 8.0)).sum()
    assert long > short


def test_frame_count_matches_duration():
    levels = band_levels(sine(0.1, 1000.0, seconds=5.0), FS, OCTAVE, window_s=1.0)
    assert levels.shape == (5, len(OCTAVE))


def test_level_scales_with_amplitude():
    quiet = band_levels(sine(0.01, 1000.0), FS, THIRD_OCTAVE)
    loud = band_levels(sine(0.1, 1000.0), FS, THIRD_OCTAVE)
    assert np.nanmax(loud[0]) - np.nanmax(quiet[0]) == pytest.approx(20.0, abs=0.05)


def test_spectrogram_shape_and_peak():
    times, frequencies, levels = spectrogram(sine(1.0, 2000.0, 1.0), FS, window_s=0.125)
    assert levels.shape == (times.size, frequencies.size)
    assert frequencies[int(np.argmax(levels[levels.shape[0] // 2]))] == pytest.approx(
        2000.0, rel=0.02
    )


def test_bandset_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        BandSet(0)


def test_band_exceedance_levels_shape_and_ordering():
    """L90 (background) must sit below L10 (loud events) in every band."""
    from splmeter.bands import band_exceedance_levels

    rng = np.random.default_rng(21)
    levels = rng.normal(55, 6, size=(600, len(THIRD_OCTAVE)))
    result = band_exceedance_levels(levels, [10, 50, 90])
    for n in (10, 50, 90):
        assert result[n].shape == (len(THIRD_OCTAVE),)
    assert np.all(result[90] < result[50])
    assert np.all(result[50] < result[10])


def test_l90_spectrum_is_the_background_not_the_events():
    """A band with rare loud events must show them in L10 but not in L90.

    This is the point of a spectral L90: it recovers the quiet floor's spectral shape
    with transients removed, which a broadband L90 cannot.
    """
    from splmeter.bands import band_exceedance_levels

    levels = np.full((1000, len(THIRD_OCTAVE)), 50.0)
    levels[:20, 5] = 95.0          # loud but rare, in one band only
    result = band_exceedance_levels(levels, [10, 90])
    assert result[90][5] == pytest.approx(50.0, abs=0.5)   # background unaffected
    assert result[10][5] == pytest.approx(50.0, abs=0.5)   # 2% of frames: below L10 too
    result_l1 = band_exceedance_levels(levels, [1])
    assert result_l1[1][5] > 90.0                          # but visible at L1


def test_unresolvable_bands_stay_nan():
    from splmeter.bands import band_exceedance_levels

    levels = np.full((100, len(TWELFTH_OCTAVE)), 50.0)
    levels[:, 0] = np.nan
    result = band_exceedance_levels(levels, [50])
    assert np.isnan(result[50][0])
    assert np.isfinite(result[50][-1])


def test_band_exceedance_rejects_one_dimensional_input():
    from splmeter.bands import band_exceedance_levels

    with pytest.raises(ValueError, match="n_frames, n_bands"):
        band_exceedance_levels(np.zeros(50), [50])


def test_spectral_and_broadband_exceedance_agree_on_a_single_band():
    """With one band, spectral Ln must reduce to the broadband definition."""
    from splmeter.bands import band_exceedance_levels
    from splmeter.metrics import exceedance_levels

    rng = np.random.default_rng(3)
    series = rng.normal(60, 5, 500)
    spectral = band_exceedance_levels(series.reshape(-1, 1), [10, 90])
    broadband = exceedance_levels(series, [10, 90])
    assert spectral[10][0] == pytest.approx(broadband[10], abs=1e-9)
    assert spectral[90][0] == pytest.approx(broadband[90], abs=1e-9)
