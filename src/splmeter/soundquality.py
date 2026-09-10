"""Psychoacoustic sound-quality metrics: loudness, sharpness, tonality, roughness.

These wrap `MOSQITO <https://github.com/Eomys/MoSQITo>`_ (Apache-2.0, compatible with
this project's licence) rather than reimplementing four psychoacoustic standards.
MOSQITO is validated against the reference signals in those standards, which is a far
stronger position than a fresh implementation would start from.

MOSQITO is an **optional** dependency (``pip install "splmeter[soundquality]"``).  It
pulls in matplotlib -- its ``roughness_ecma`` module imports ``matplotlib.pyplot`` at
module scope, so matplotlib is required even for headless computation -- and the core
level metrics have no need of it.

Cost, and why only loudness is computed in bulk
-----------------------------------------------
Measured on one core over a 600 s packet at 48 kHz, against 1.8 s for the entire
broadband + band metric set:

=====================================  ==================  ==========================
Metric                                 Speed               Per 600 s packet
=====================================  ==================  ==========================
Everything else combined               332x realtime       1.8 s
``loudness_zwst_perseg`` (per second)  37x realtime        16 s
``roughness_dw``                       0.4x realtime       ~1,460 s
``loudness_zwtv`` (time-varying)       1.03x realtime      ~580 s
=====================================  ==================  ==========================

ISO 532-1 **time-varying** loudness would take roughly 46 days across 34 workers for the
full archive -- it models temporal masking and emits 500 values per second, neither of
which a 1-second summary needs.  **Stationary loudness evaluated per 1-second segment**
costs about 31 hours across 34 workers, which is less than the archive takes to
download, so it runs essentially for free alongside the transfer.  That is what
:func:`loudness` does, and it is the only sound-quality metric in the bulk pass.

Sharpness, tonality and roughness are computed on demand for selected periods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MISSING = (
    "Sound-quality metrics need MOSQITO. Install with:\n"
    '    pip install "splmeter[soundquality]"'
)


def _mosqito():
    """Import MOSQITO lazily, with a useful message when it is absent."""
    try:
        import matplotlib

        # MOSQITO imports pyplot at module scope; select a non-interactive backend
        # before that happens so importing it cannot try to open a display.
        matplotlib.use("Agg")
        import mosqito.sq_metrics as sq
    except ImportError as exc:  # pragma: no cover - exercised only without mosqito
        raise ImportError(_MISSING) from exc
    return sq


def backend_version() -> str:
    """Version of the MOSQITO backend actually in use.

    Recorded alongside every computed sound-quality value.  A change to MOSQITO's
    loudness implementation would make new numbers incomparable with old ones, and over
    a multi-year archive that is a reproducibility problem, not a footnote -- so the
    provenance is stored rather than assumed.  The dependency is pinned to ``<2`` for
    the same reason.
    """
    try:
        from importlib.metadata import version

        return version("mosqito")
    except Exception:  # pragma: no cover - only when mosqito is absent
        return "unavailable"


def available() -> bool:
    """True when the optional sound-quality dependencies are installed."""
    try:
        _mosqito()
    except ImportError:
        return False
    return True


def loudness(pressure: np.ndarray, fs: float, window_s: float = 1.0) -> np.ndarray:
    """Zwicker loudness in **sones**, one value per ``window_s`` (ISO 532-1).

    Stationary loudness evaluated per segment.  This is the correct choice for a
    1-second environmental-noise summary: the time-varying model exists to capture
    temporal masking within a few hundred milliseconds, which a per-second figure
    averages away regardless -- at roughly 36x the cost.

    Returns loudness only; call :func:`specific_loudness` if the Bark-band detail is
    needed (for example to derive sharpness).
    """
    sq = _mosqito()
    nperseg = int(round(fs * window_s))
    values, _, _, _ = sq.loudness_zwst_perseg(
        np.asarray(pressure, dtype=np.float64), int(fs), nperseg=nperseg, noverlap=0
    )
    return np.atleast_1d(np.asarray(values, dtype=np.float64))


def specific_loudness(
    pressure: np.ndarray, fs: float, window_s: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(loudness, specific_loudness, bark_axis)`` per segment.

    ``specific_loudness`` has shape ``(n_bark, n_segments)`` -- loudness density across
    the critical-band rate scale, which is what sharpness weights.
    """
    sq = _mosqito()
    nperseg = int(round(fs * window_s))
    values, specific, bark, _ = sq.loudness_zwst_perseg(
        np.asarray(pressure, dtype=np.float64), int(fs), nperseg=nperseg, noverlap=0
    )
    return (
        np.atleast_1d(np.asarray(values, dtype=np.float64)),
        np.asarray(specific, dtype=np.float64),
        np.asarray(bark, dtype=np.float64),
    )


def sharpness(pressure: np.ndarray, fs: float, window_s: float = 1.0) -> np.ndarray:
    """Sharpness in **acum** per segment (DIN 45692).

    High-frequency weighted loudness: how shrill a sound is, independent of how loud.
    Not part of the bulk pass -- run it on the periods you care about.
    """
    sq = _mosqito()
    values, specific, _ = specific_loudness(pressure, fs, window_s)
    return np.atleast_1d(
        np.asarray(sq.sharpness_din_from_loudness(values, specific), dtype=np.float64)
    )


@dataclass(frozen=True)
class Tonality:
    """Tonal content of a stationary signal (ECMA-74 / ISO 7779).

    ``tnr`` is the tone-to-noise ratio and ``pr`` the prominence ratio, both in dB, for
    each detected tone.  A tone is generally considered prominent above about 8 dB TNR.
    Useful for construction and HVAC noise, where a discrete tone drives annoyance well
    beyond what its contribution to the A-weighted level suggests.
    """

    frequencies: np.ndarray
    tnr: np.ndarray
    pr: np.ndarray
    tnr_total: float
    pr_total: float

    @property
    def prominent(self) -> np.ndarray:
        """Frequencies of tones exceeding the 8 dB prominence guideline."""
        return self.frequencies[self.tnr >= 8.0]


def tonality(pressure: np.ndarray, fs: float) -> Tonality:
    """Tone-to-noise and prominence ratios for a stationary signal.

    Expects a segment short enough to be stationary (a few seconds).  Not part of the
    bulk pass.
    """
    sq = _mosqito()
    pressure = np.asarray(pressure, dtype=np.float64)
    t_tnr, tnr, prom_tnr, freqs_tnr = sq.tnr_ecma_st(pressure, int(fs))
    t_pr, pr, prom_pr, freqs_pr = sq.pr_ecma_st(pressure, int(fs))
    return Tonality(
        frequencies=np.atleast_1d(np.asarray(freqs_tnr, dtype=np.float64)),
        tnr=np.atleast_1d(np.asarray(prom_tnr, dtype=np.float64)),
        pr=np.atleast_1d(np.asarray(prom_pr, dtype=np.float64)),
        tnr_total=float(np.atleast_1d(t_tnr)[0]),
        pr_total=float(np.atleast_1d(t_pr)[0]),
    )


def roughness(pressure: np.ndarray, fs: float, overlap: float = 0.5) -> np.ndarray:
    """Roughness in **asper** (Daniel & Weber model).

    The most expensive metric here by a wide margin -- roughly 0.4x realtime, i.e. two
    and a half times slower than the audio itself -- and the least standardised of the
    four, since ISO has no agreed roughness method.  Treat results as comparative rather
    than absolute, and run it only on short selected excerpts.
    """
    sq = _mosqito()
    values, _, _, _ = sq.roughness_dw(
        np.asarray(pressure, dtype=np.float64), int(fs), overlap=overlap
    )
    return np.atleast_1d(np.asarray(values, dtype=np.float64))
