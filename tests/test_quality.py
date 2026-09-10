"""Quality-flag tests.

The old pipeline had no overload detection at all: a clipped recording produced a
plausible number indistinguishable from a valid one.
"""

from __future__ import annotations

import numpy as np
import pytest

from splmeter.core import Calibration
from splmeter.metrics import leq
from splmeter.quality import CLIP_THRESHOLD, assess

FS = 48000.0
CAL = Calibration(mic_sensitivity_v_per_pa=0.050)


def _levels(samples):
    return leq(CAL.to_pascals(samples), FS, 1.0)


def test_clean_signal_is_not_flagged():
    t = np.arange(int(FS * 3)) / FS
    samples = 0.3 * np.sin(2 * np.pi * 500 * t)
    flags = assess(samples, _levels(samples), FS)
    assert not flags.clipped.any()
    assert not flags.underrange.any()


def test_clipped_frame_is_flagged():
    t = np.arange(int(FS * 3)) / FS
    samples = np.clip(2.0 * np.sin(2 * np.pi * 500 * t), -1.0, 1.0)
    flags = assess(samples, _levels(samples), FS)
    assert flags.clipped.all()
    assert (flags.clipped_fraction > 0.3).all()


def test_only_the_clipped_second_is_flagged():
    t = np.arange(int(FS * 3)) / FS
    samples = 0.3 * np.sin(2 * np.pi * 500 * t)
    samples[int(FS) : int(2 * FS)] = np.clip(
        3.0 * np.sin(2 * np.pi * 500 * t[int(FS) : int(2 * FS)]), -1.0, 1.0
    )
    flags = assess(samples, _levels(samples), FS)
    assert list(flags.clipped) == [False, True, False]


def test_isolated_full_scale_sample_is_not_clipping():
    """One sample at full scale is plausible; a flat-topped run is not."""
    samples = 0.2 * np.ones(int(FS * 2))
    samples[1000] = CLIP_THRESHOLD + 1e-6
    flags = assess(samples, _levels(samples), FS)
    assert not flags.clipped.any()
    assert flags.clipped_fraction[0] > 0


def test_silence_is_flagged_underrange():
    samples = np.zeros(int(FS * 2))
    flags = assess(samples, _levels(samples), FS)
    assert flags.underrange.all()
    assert flags.any_suspect.all()


def test_empty_input_returns_empty_flags():
    flags = assess(np.zeros(10), np.array([]), FS)
    assert flags.clipped.size == 0
