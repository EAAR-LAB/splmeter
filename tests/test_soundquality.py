"""Sound-quality metric tests.

These validate the wrapper's contract -- shapes, units, monotonic behaviour -- rather
than re-deriving MOSQITO's own standard-reference validation, which it already carries.
"""

from __future__ import annotations

import numpy as np
import pytest

from splmeter import soundquality as sq

pytestmark = pytest.mark.skipif(
    not sq.available(), reason="MOSQITO not installed (splmeter[soundquality])"
)

FS = 48000.0


def tone(amplitude_pa, frequency, seconds=4.0, fs=FS):
    t = np.arange(int(round(fs * seconds))) / fs
    return amplitude_pa * np.sin(2 * np.pi * frequency * t)


def test_loudness_returns_one_value_per_second():
    result = sq.loudness(tone(0.1, 1000.0, seconds=5.0), FS, window_s=1.0)
    assert result.shape == (5,)
    assert np.all(result > 0)


def test_loudness_increases_with_level():
    quiet = sq.loudness(tone(0.02, 1000.0), FS)
    loud = sq.loudness(tone(0.2, 1000.0), FS)
    assert loud.mean() > quiet.mean() * 2


def test_loudness_of_1khz_tone_is_physically_plausible():
    """A 1 kHz tone at 40 dB SPL is 1 sone by definition; doubling level roughly
    doubles loudness above 40 phon."""
    # 0.002 Pa amplitude -> ~37 dB SPL; a few tenths of a sone up to a few sones.
    soft = sq.loudness(tone(0.002, 1000.0), FS).mean()
    assert 0.05 < soft < 5.0


def test_specific_loudness_spans_the_bark_scale():
    values, specific, bark = sq.specific_loudness(tone(0.1, 1000.0), FS)
    assert specific.shape[0] == bark.size
    assert specific.shape[1] == values.size
    assert bark[0] >= 0 and bark[-1] <= 24.5
    # A 1 kHz tone peaks near 8.5 Bark.
    assert 7.0 < bark[int(np.argmax(specific[:, 0]))] < 10.0


def test_sharpness_is_higher_for_high_frequency_content():
    """Sharpness weights high-frequency loudness, so a 5 kHz tone is sharper than
    a 500 Hz tone of comparable loudness."""
    low = sq.sharpness(tone(0.1, 500.0), FS).mean()
    high = sq.sharpness(tone(0.1, 5000.0), FS).mean()
    assert high > low


def test_sharpness_returns_one_value_per_segment():
    assert sq.sharpness(tone(0.1, 1000.0, seconds=3.0), FS).shape == (3,)


def test_tonality_detects_a_pure_tone_against_noise():
    rng = np.random.default_rng(4)
    signal = tone(0.1, 1000.0, seconds=3.0) + 0.005 * rng.standard_normal(int(FS * 3))
    result = sq.tonality(signal, FS)
    assert result.frequencies.size >= 1
    nearest = result.frequencies[int(np.argmin(np.abs(result.frequencies - 1000.0)))]
    assert nearest == pytest.approx(1000.0, rel=0.05)
    assert result.tnr_total > 8.0


def test_broadband_noise_is_less_tonal_than_a_tone():
    rng = np.random.default_rng(9)
    noise = 0.05 * rng.standard_normal(int(FS * 3))
    tonal = tone(0.1, 2000.0, seconds=3.0)
    assert sq.tonality(tonal, FS).tnr_total > sq.tonality(noise, FS).tnr_total


def test_roughness_peaks_near_70hz_modulation():
    """Roughness is maximal for ~70 Hz amplitude modulation."""
    t = np.arange(int(FS * 1.0)) / FS
    carrier = np.sin(2 * np.pi * 1000 * t)
    slow = 0.1 * (1 + np.sin(2 * np.pi * 5 * t)) * carrier
    rough = 0.1 * (1 + np.sin(2 * np.pi * 70 * t)) * carrier
    assert sq.roughness(rough, FS).mean() > sq.roughness(slow, FS).mean()


def test_available_reports_true_when_installed():
    assert sq.available() is True
