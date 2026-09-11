"""Octave and fractional-octave band analysis (IEC 61260 base-ten system).

Band levels are computed from the FFT of each analysis frame rather than with a bank of
IIR filters.  For this archive that is the right trade: 140 million device-seconds x 3
band resolutions makes a 160-filter bank prohibitive, while one real FFT per second is
essentially free.  The cost is that band *shapes* are rectangular rather than meeting
the IEC 61260 filter-shape masks -- see :ref:`limitations` below.

The +3.01 dB error
------------------
The previous implementation formed ``2/N * |X|`` -- the **peak** amplitude of each
component -- and then reported ``20*log10(rss/p_ref)``.  Every band level it produced
was therefore 3.01 dB high, and band levels could not be reconciled with the broadband
``Leq`` from the same library.  Worse, its test suite asserted the erroneous value as
the expected answer, so the bug was actively protected.

This module scales so that the summed band power equals the mean-square pressure
(Parseval), which makes the reconciliation in
``test_bands.py::test_band_sum_matches_broadband_leq`` an actual check rather than a
tautology.
"""

from __future__ import annotations

import warnings

import numpy as np

from .core import P_REF, to_db

#: Octave ratio of the base-ten system (IEC 61260-1:2014 clause 5.4).  The base-two
#: system (G = 2) is also permitted; base-ten is the one the standard prefers and the
#: one whose nominal midbands match the familiar 31.5 / 63 / 125 ... series.
G = 10 ** (3 / 10)

#: Reference midband frequency, Hz.
F_REF = 1000.0

#: Minimum FFT bins that must fall inside a band for its level to be trustworthy.
#: Narrow low-frequency bands can be thinner than the FFT resolution -- a 1/12-octave
#: band at 12.5 Hz is 0.72 Hz wide, against 1 Hz resolution for a 1-second frame.
MIN_BINS_PER_BAND = 3


class BandSet:
    """Midband frequencies and edges for a fractional-octave set.

    Parameters
    ----------
    fraction:
        1 for octave bands, 3 for one-third octave, 12 for one-twelfth.
    low, high:
        Approximate frequency range to cover, Hz.  Midbands are snapped to the
        standard series, so the realised range may extend slightly beyond.
    """

    def __init__(self, fraction: int, low: float = 16.0, high: float = 20000.0) -> None:
        if fraction < 1:
            raise ValueError("fraction must be >= 1")
        self.fraction = int(fraction)
        lowest = int(np.ceil(fraction * np.log(low / F_REF) / np.log(G)))
        highest = int(np.floor(fraction * np.log(high / F_REF) / np.log(G)))
        self.indices = np.arange(lowest, highest + 1)
        #: Exact midband frequencies (IEC 61260-1 clause 5.4).
        self.midbands = F_REF * G ** (self.indices / self.fraction)
        half = G ** (1 / (2 * self.fraction))
        self.lower_edges = self.midbands / half
        self.upper_edges = self.midbands * half

    def __len__(self) -> int:
        return self.midbands.size

    def __repr__(self) -> str:
        return (
            f"BandSet(1/{self.fraction} octave, {len(self)} bands, "
            f"{self.midbands[0]:.1f}-{self.midbands[-1]:.0f} Hz)"
        )

    def nominal_labels(self) -> list[str]:
        """Human-readable midband labels (e.g. ``'1 kHz'``)."""
        labels = []
        for f in self.midbands:
            if f >= 1000:
                labels.append(f"{f / 1000:.3g} kHz")
            else:
                labels.append(f"{f:.3g} Hz")
        return labels

    def resolvable(self, fs: float, frame_samples: int) -> np.ndarray:
        """Boolean mask of bands wide enough to resolve at this FFT resolution."""
        resolution = fs / frame_samples
        return (self.upper_edges - self.lower_edges) >= MIN_BINS_PER_BAND * resolution


#: The three band sets the pipeline stores per second.
OCTAVE = BandSet(1, 15.0, 17000.0)
THIRD_OCTAVE = BandSet(3, 12.5, 20000.0)
TWELFTH_OCTAVE = BandSet(12, 12.5, 20000.0)


def _frame_power_spectrum(
    pressure: np.ndarray, fs: float, frame_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame power spectrum scaled so that ``sum(power) == mean(p**2)``.

    A Hann window is applied to control spectral leakage -- the old code used a
    rectangular window, which smears a tone across neighbouring bands -- and the result
    is divided by the window's mean square so the calibration is preserved.
    """
    n_frames = pressure.size // frame_samples
    if n_frames == 0:
        return np.empty((0, 0)), np.empty(0)

    block = pressure[: n_frames * frame_samples].reshape(n_frames, frame_samples)
    window = np.hanning(frame_samples)
    spectrum = np.fft.rfft(block * window, axis=1)

    # |X_k|^2 / N^2, doubled for the interior bins that represent both +f and -f.
    power = (np.abs(spectrum) / frame_samples) ** 2
    power[:, 1:-1] *= 2.0
    if frame_samples % 2 == 1:  # no Nyquist bin when N is odd
        power[:, -1] *= 2.0
    power /= np.mean(window**2)

    frequencies = np.fft.rfftfreq(frame_samples, d=1.0 / fs)
    return power, frequencies


def band_levels(
    pressure: np.ndarray,
    fs: float,
    bands: BandSet,
    window_s: float = 1.0,
    mask_unresolved: bool = True,
) -> np.ndarray:
    """Band sound pressure levels, shape ``(n_frames, n_bands)``, in dB.

    Parameters
    ----------
    mask_unresolved:
        When true, bands narrower than :data:`MIN_BINS_PER_BAND` FFT bins are returned
        as ``nan`` rather than as a number computed from one or two bins.  Reporting an
        unresolvable band as though it were measured is how spurious low-frequency
        content gets into a spectrum.
    """
    frame_samples = int(round(fs * window_s))
    power, frequencies = _frame_power_spectrum(
        np.asarray(pressure, dtype=np.float64), fs, frame_samples
    )
    if power.size == 0:
        return np.empty((0, len(bands)))

    # np.searchsorted over sorted edges is O(n_bands * log n_bins); summing with
    # reduceat avoids building a mask per band.
    starts = np.searchsorted(frequencies, bands.lower_edges, side="left")
    stops = np.searchsorted(frequencies, bands.upper_edges, side="right")

    out = np.empty((power.shape[0], len(bands)), dtype=np.float64)
    cumulative = np.concatenate(
        [np.zeros((power.shape[0], 1)), np.cumsum(power, axis=1)], axis=1
    )
    stops = np.clip(stops, 0, power.shape[1])
    starts = np.clip(starts, 0, power.shape[1])
    band_power = cumulative[:, stops] - cumulative[:, starts]

    out = to_db(band_power)
    if mask_unresolved:
        out[:, ~bands.resolvable(fs, frame_samples)] = np.nan
    return out


def band_sum_level(levels: np.ndarray) -> np.ndarray:
    """Combine band levels back into a broadband level, per frame.

    Should reproduce the broadband ``Leq`` of the same signal to within the energy
    excluded by the band set's frequency limits.
    """
    levels = np.asarray(levels, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        power = np.where(np.isfinite(levels), 10.0 ** (levels / 10.0), 0.0)
    return 10.0 * np.log10(np.sum(power, axis=-1))


def band_exceedance_levels(levels: np.ndarray, percentiles) -> dict[float, np.ndarray]:
    """Statistical exceedance levels **per band** -- the 831 calls this spectral Ln.

    Given per-frame band levels of shape ``(n_frames, n_bands)``, returns
    ``{n: array of n_bands}`` where each entry is the level exceeded n% of the time in
    that band.

    The **L90 spectrum is the standard measure of background noise**: it shows the
    spectral shape of the quiet floor with transient events removed, which a broadband
    L90 cannot. L10 per band shows the opposite -- which frequencies the loud events
    occupy. Comparing the two separates a steady tonal source from intermittent
    broadband activity.

    Bands masked as unresolvable (``nan``) stay ``nan`` rather than being dropped, so
    the result stays aligned with the band set.
    """
    levels = np.asarray(levels, dtype=np.float64)
    if levels.ndim != 2:
        raise ValueError(f"expected (n_frames, n_bands), got shape {levels.shape}")

    out: dict[float, np.ndarray] = {}
    for n in percentiles:
        # Ln is the level EXCEEDED n% of the time, so it is the (100-n)th percentile.
        # L90 is the quiet background; a plain quantile(0.90) would give the opposite.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-nan bands
            out[float(n)] = np.nanpercentile(levels, 100.0 - float(n), axis=0)
    return out


def spectrogram(
    pressure: np.ndarray,
    fs: float,
    window_s: float = 0.125,
    overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Narrowband spectrogram: ``(times, frequencies, levels_db)``.

    Computed **on demand** rather than stored.  Retaining 1 Hz-resolution spectra for
    140 million device-seconds would run to petabytes, so the pipeline keeps octave,
    third-octave and twelfth-octave levels and reconstructs narrowband detail from the
    original audio for whichever window the analyst zooms into.
    """
    frame_samples = int(round(fs * window_s))
    hop = max(1, int(round(frame_samples * (1.0 - overlap))))
    pressure = np.asarray(pressure, dtype=np.float64)

    n_frames = 1 + max(0, (pressure.size - frame_samples) // hop)
    window = np.hanning(frame_samples)
    starts = np.arange(n_frames) * hop
    frames = np.lib.stride_tricks.as_strided(
        pressure,
        shape=(n_frames, frame_samples),
        strides=(pressure.strides[0] * hop, pressure.strides[0]),
    )
    spectrum = np.fft.rfft(frames * window, axis=1)
    power = (np.abs(spectrum) / frame_samples) ** 2
    power[:, 1:-1] *= 2.0
    power /= np.mean(window**2)

    times = starts / fs
    frequencies = np.fft.rfftfreq(frame_samples, d=1.0 / fs)
    return times, frequencies, to_db(power)
