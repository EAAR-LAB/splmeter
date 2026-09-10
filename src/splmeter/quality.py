"""Signal quality flags: clipping, overload and underrange.

A professional sound level meter refuses to report a level it cannot stand behind; the
previous pipeline had no such check, so a clipped recording produced a plausible-looking
number with nothing to distinguish it from a valid one.

These flags are stored alongside every measurement so that a later analysis can exclude
suspect seconds rather than discovering the problem in a scatter plot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Fraction of digital full scale above which a sample counts as clipped.  The archive
#: is 16-bit PCM read as normalised floats, so full scale is 1.0.
CLIP_THRESHOLD = 0.999

#: Consecutive clipped samples that constitute a genuine clipping *event*.  A single
#: sample at full scale is plausible in loud but undistorted audio; a run of them is
#: the flat-topped waveform of a converter out of range.
CLIP_RUN_SAMPLES = 3

#: Level below which a frame is almost certainly the electrical noise floor rather than
#: an acoustic measurement -- a disconnected or failed microphone.
UNDERRANGE_DB = 10.0


@dataclass(frozen=True)
class QualityFlags:
    """Per-frame quality assessment."""

    clipped_fraction: np.ndarray
    clipped: np.ndarray
    underrange: np.ndarray

    @property
    def any_suspect(self) -> np.ndarray:
        return self.clipped | self.underrange


def _longest_run_per_frame(mask: np.ndarray) -> np.ndarray:
    """Longest run of True per row, without a Python loop over samples."""
    n_frames, n = mask.shape
    if n == 0:
        return np.zeros(n_frames, dtype=int)
    # Running count that resets at every False, then take the row maximum.
    padded = np.concatenate([np.zeros((n_frames, 1), dtype=bool), mask], axis=1)
    counts = np.zeros((n_frames, n + 1), dtype=np.int32)
    for i in range(1, n + 1):
        counts[:, i] = np.where(padded[:, i], counts[:, i - 1] + 1, 0)
    return counts.max(axis=1)


def assess(
    samples: np.ndarray,
    levels: np.ndarray,
    fs: float,
    window_s: float = 1.0,
) -> QualityFlags:
    """Flag frames whose measurement should not be trusted.

    Parameters
    ----------
    samples:
        Raw normalised samples, **before** calibration -- clipping is a property of the
        converter, so it must be judged against digital full scale, not pascals.
    levels:
        The per-frame levels computed from the same signal, used for the underrange
        test.
    """
    samples = np.asarray(samples)
    frame = int(round(fs * window_s))
    n_frames = min(samples.size // frame, len(levels))
    if n_frames == 0:
        empty_f = np.empty(0)
        empty_b = np.empty(0, dtype=bool)
        return QualityFlags(empty_f, empty_b, empty_b)

    block = samples[: n_frames * frame].reshape(n_frames, frame)
    at_full_scale = np.abs(block) >= CLIP_THRESHOLD
    fraction = at_full_scale.mean(axis=1)
    clipped = _longest_run_per_frame(at_full_scale) >= CLIP_RUN_SAMPLES

    levels = np.asarray(levels, dtype=np.float64)[:n_frames]
    underrange = ~np.isfinite(levels) | (levels < UNDERRANGE_DB)

    return QualityFlags(clipped_fraction=fraction, clipped=clipped, underrange=underrange)
