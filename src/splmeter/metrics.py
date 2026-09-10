"""Time-domain sound level metrics: Leq, time weighting, Lmax/Lmin, Lpeak, SEL, Ln.

All functions take **pressure in pascals** and return **levels in dB re 20 uPa**.
Frequency weighting is applied by the caller beforehand (:mod:`splmeter.weighting`), so
an "LAeq" here is simply :func:`leq` of an A-weighted signal.

Performance
-----------
Every routine is O(N) in the number of samples.  The previous implementation built an
explicit ``(n_frames, window)`` index matrix for each windowed statistic --
``Leq(averaging_window=60)`` over one hour allocated ~3.5 GB, and a 24-hour file ~83 GB.
That cost is what forced the old code to decimate to 1 kHz before measuring, which in
turn destroyed peak levels.  Framing here is done by reshaping a view or by prefix sums,
so a 10-minute packet costs a few tens of MB and full-rate peak detection is affordable.

Correctness notes
-----------------
* :func:`lpeak` uses ``max(|p|)`` at the **full sample rate**.  The old code took the
  signed maximum of a signal decimated 48:1 with no anti-aliasing -- so it missed the
  true peak almost always, and returned ``-inf`` for any window whose largest excursion
  was negative.
* No zero-padding.  The old code prepended zeros so the first output window was entirely
  silence, making the first sample of every ``Leq`` and ``Lpeak`` series ``-inf``; the
  ETL then clipped that to ``0.0`` and the notebooks discarded it as an outlier.  Here a
  partial trailing frame is dropped and reported, never faked.
* Silence yields ``-inf``, not ``0``.  See :func:`splmeter.core.to_db`.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as _signal

from .core import DTYPE, P_REF, to_db

#: Exponential time-weighting constants, seconds (IEC 61672-1 clause 5.8).
TIME_CONSTANTS = {"F": 0.125, "S": 1.0, "I": 0.035}

#: Decay time constant of the Impulse detector, giving the specified 2.9 dB/s decay.
IMPULSE_DECAY_TAU = 1.0 / (0.29 * np.log(10))

#: Reference duration for sound exposure level, seconds (IEC 61672-1 clause 3.6).
SEL_REFERENCE_TIME = 1.0


def _frame_count(n_samples: int, window: int, hop: int) -> int:
    if n_samples < window:
        return 0
    return 1 + (n_samples - window) // hop


def frame_mean_square(
    pressure: np.ndarray, fs: float, window_s: float, hop_s: float | None = None
) -> np.ndarray:
    """Mean-square pressure per analysis frame.

    Uses a reshaped view for the common tumbling-window case (``hop == window``, exact
    division) and prefix sums otherwise.  Both are O(N); neither materialises a
    per-frame copy of the samples.

    A trailing partial frame is **dropped**, not zero-padded: padding would report a
    quiet partial second as though it were a full one.
    """
    pressure = np.asarray(pressure)
    hop_s = window_s if hop_s is None else hop_s
    window = int(round(fs * window_s))
    hop = int(round(fs * hop_s))
    if window <= 0 or hop <= 0:
        raise ValueError(f"window and hop must be positive (got {window_s}, {hop_s})")

    n_frames = _frame_count(pressure.size, window, hop)
    if n_frames == 0:
        return np.empty(0, dtype=np.float64)

    if hop == window and pressure.size >= window:
        usable = n_frames * window
        block = pressure[:usable].reshape(n_frames, window)
        # float64 accumulator: float32 would lose ~7 digits over 48k samples/frame.
        return np.mean(np.square(block, dtype=np.float64), axis=1)

    squared = np.square(pressure, dtype=np.float64)
    prefix = np.concatenate(([0.0], np.cumsum(squared)))
    starts = np.arange(n_frames) * hop
    return (prefix[starts + window] - prefix[starts]) / window


def leq(
    pressure: np.ndarray, fs: float, window_s: float = 1.0, hop_s: float | None = None
) -> np.ndarray:
    """Equivalent continuous sound level per frame, in dB.

    ``Leq = 10*log10( mean(p^2) / p_ref^2 )``.  With an A-weighted input this is LAeq;
    with C-weighted, LCeq; unweighted, LZeq.  Nothing about the weighting is recorded
    here: the caller chooses the weighting and is responsible for labelling the
    result accordingly.
    """
    return to_db(frame_mean_square(pressure, fs, window_s, hop_s))


def time_weight(pressure: np.ndarray, fs: float, weighting: str = "F") -> np.ndarray:
    """Exponentially time-weighted mean-square pressure (not yet in dB).

    Implements the IEC 61672-1 detector ``(1/tau) * integral p^2(x) e^{-(t-x)/tau} dx``
    as a one-pole recursive filter -- O(N) and exact, rather than the truncated explicit
    convolution the old code used, which was O(N*window) and biased low whenever the
    truncation window was shorter than ~5 tau.

    Impulse weighting is the asymmetric detector: a 35 ms exponential average followed
    by a peak-hold that decays at 2.9 dB/s.
    """
    weighting = weighting.upper()
    if weighting not in TIME_CONSTANTS:
        raise ValueError(f"time weighting must be one of {tuple(TIME_CONSTANTS)}")

    squared = np.square(np.asarray(pressure, dtype=np.float64))
    tau = TIME_CONSTANTS[weighting]
    alpha = np.exp(-1.0 / (fs * tau))
    smoothed = _signal.lfilter([1.0 - alpha], [1.0, -alpha], squared)

    if weighting != "I":
        return smoothed
    return _decaying_peak_hold(smoothed, fs, IMPULSE_DECAY_TAU)


def _decaying_peak_hold(
    values: np.ndarray, fs: float, tau: float, block: int = 1 << 16
) -> np.ndarray:
    """``z[n] = max(v[n], d * z[n-1])`` with ``d = exp(-1/(fs*tau))``.

    Vectorised via the identity ``z[n]/d^n = max(v[n]/d^n, z[n-1]/d^(n-1))``, so the
    recursion becomes a cumulative maximum.  ``d^n`` underflows for long signals -- at
    48 kHz a 10-minute packet would need ``d^28.8e6`` -- so it is applied blockwise with
    the running value carried across blocks.
    """
    decay = np.exp(-1.0 / (fs * tau))
    out = np.empty_like(values)
    carried = 0.0
    powers = decay ** np.arange(block, dtype=np.float64)
    for start in range(0, values.size, block):
        chunk = values[start : start + block]
        scale = powers[: chunk.size]
        running = np.maximum.accumulate(chunk / scale)
        held = running * scale
        # Fold in the value carried from the previous block, itself decaying.
        held = np.maximum(held, carried * scale * decay)
        out[start : start + chunk.size] = held
        carried = held[-1]
    return out


def _frame_reduce(values: np.ndarray, fs: float, window_s: float, reducer) -> np.ndarray:
    """Apply ``reducer`` over tumbling frames of ``values``."""
    window = int(round(fs * window_s))
    n_frames = values.size // window
    if n_frames == 0:
        return np.empty(0, dtype=np.float64)
    block = values[: n_frames * window].reshape(n_frames, window)
    return reducer(block, axis=1)


def lmax(
    pressure: np.ndarray, fs: float, window_s: float = 1.0, weighting: str = "F"
) -> np.ndarray:
    """Maximum time-weighted level per frame, e.g. LAFmax with an A-weighted input.

    The old ETL called this with a 1 Hz input and a 1-sample window, so the maximum was
    taken over exactly one value -- a no-op.  Its ``lafmax``/``lasmax`` columns are
    really ``laf``/``las``: the instantaneous weighted level at each tick, not a maximum.
    """
    weighted = time_weight(pressure, fs, weighting)
    return to_db(_frame_reduce(weighted, fs, window_s, np.max))


def lmin(
    pressure: np.ndarray, fs: float, window_s: float = 1.0, weighting: str = "F"
) -> np.ndarray:
    """Minimum time-weighted level per frame. Absent entirely from the old library."""
    weighted = time_weight(pressure, fs, weighting)
    return to_db(_frame_reduce(weighted, fs, window_s, np.min))


def lpeak(pressure: np.ndarray, fs: float, window_s: float = 1.0) -> np.ndarray:
    """Peak sound level per frame: ``20*log10(max|p| / p_ref)``.

    Peak level is **not** time-weighted and, by convention, is measured on a C-weighted
    signal (LCpeak).  It must be computed at the full sample rate: a peak is by
    definition a short-duration excursion, so decimating first -- as the old code did,
    to 1 kHz, without an anti-aliasing filter -- systematically under-reads it.

    Note ``max(|p|)``, not ``max(p)``.  The old code took the signed maximum, so a
    window whose largest excursion was a rarefaction returned the wrong value, and an
    all-negative window returned ``-inf``.
    """
    absolute = np.abs(np.asarray(pressure, dtype=np.float64))
    peak = _frame_reduce(absolute, fs, window_s, np.max)
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(peak / P_REF)


def sel(pressure: np.ndarray, fs: float) -> float:
    """Sound exposure level over the whole signal, in dB.

    ``SEL = 10*log10( integral p^2 dt / (T0 * p_ref^2) )`` with ``T0 = 1 s``.  Equal to
    ``Leq + 10*log10(T)`` for a measurement of duration ``T``.
    """
    pressure = np.asarray(pressure)
    exposure = np.sum(np.square(pressure, dtype=np.float64)) / fs
    with np.errstate(divide="ignore"):
        return float(10.0 * np.log10(exposure / (SEL_REFERENCE_TIME * P_REF**2)))


def energy_average(levels: np.ndarray, axis=None) -> np.ndarray:
    """Energy-average a set of levels: ``10*log10(mean(10^(L/10)))``.

    The only correct way to combine Leq values.  The legacy notebooks used an arithmetic
    mean of decibels for hourly and 24-hour figures, which understates any level series
    that varies -- and they applied it to LCpeak too, where averaging is meaningless.
    """
    levels = np.asarray(levels, dtype=np.float64)
    finite = np.isfinite(levels)
    if not finite.any():
        return np.float64(-np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(
            np.nanmean(np.where(finite, 10.0 ** (levels / 10.0), np.nan), axis=axis)
        )


def exceedance_levels(levels: np.ndarray, percentiles) -> dict[float, float]:
    """Statistical exceedance levels ``Ln`` -- the level exceeded n% of the time.

    ``L90`` is the background level and ``L10`` sits near the top; that is the opposite
    of a plain ``quantile(n/100)``.  The legacy visualisation notebook computed
    ``L5 = quantile(0.05)``, which is L95.

    Absent from the old library entirely, and not reconstructible from summary
    statistics -- which is why the pipeline stores the full 1-second series.
    """
    levels = np.asarray(levels, dtype=np.float64)
    finite = levels[np.isfinite(levels)]
    if finite.size == 0:
        return {float(n): float("-inf") for n in percentiles}
    return {
        float(n): float(np.percentile(finite, 100.0 - float(n)))
        for n in percentiles
    }
