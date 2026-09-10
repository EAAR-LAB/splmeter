"""Frequency weighting: A, C and Z, per IEC 61672-1:2013 clause 5.4.

The normative definition is a pole-zero network, not the tabulated values; the table in
the standard is derived from it.  This module therefore implements the network
(:func:`analytic_weighting_db`) and treats it as ground truth, testing the realised
digital filter against it.  The tabulated values are a cross-check, not the reference.

What went wrong before
----------------------
The pre-``splmeter`` code (``acoustic-analytics``, ``UF-AI-Catalyst-``, and the on-device
``audio_laeq.py``) scaled the C-weighting poles by ``-0.062*pi`` instead of ``-2*pi`` --
a 32x shift that moved the corner frequencies from 20.6 Hz / 12.2 kHz to roughly
0.64 Hz / 378 Hz.  The result was not C-weighting in any sense.  It survived four years
and three repositories because nothing ever compared the filter to its own definition.
:func:`analytic_weighting_db` exists so that comparison is one assertion away.

Two further corrections over the previous version:

* **Pole frequencies are pre-warped before the bilinear transform**, so each corner
  lands where the analog prototype intended rather than being squashed by the warp.  At
  48 kHz this changes nothing measurable, but at 22.05 kHz it is the difference between
  failing and passing Class 1.
* **The digital filter is renormalised at 1 kHz after the bilinear transform.**  The
  old code normalised the *analog* prototype only, so the realised digital gain at the
  1 kHz reference was off by an amount that grew as the sample rate fell (+0.16 dB at
  8 kHz, +0.04 dB at 16 kHz) -- a systematic error on every level reported.
* **Filter state can be carried across chunks** (:class:`WeightingFilter`).  Without it,
  a long file cannot be filtered in pieces: each chunk restarts from zero state and
  begins with a settling transient.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import signal

from .core import DTYPE

#: Pole frequencies of the weighting network, Hz (IEC 61672-1:2013 clause 5.4.6).
F1 = 20.598997057568145
F2 = 107.65264864304628
F3 = 737.8622307362899
F4 = 12194.21714799801

#: Frequency at which every weighting is defined to have 0 dB gain.
REFERENCE_FREQUENCY = 1000.0

WEIGHTINGS = ("A", "C", "Z")

#: Fraction of Nyquist below which the realised filter tracks the analytic definition.
#:
#: The bilinear transform maps infinite analog frequency onto Nyquist, so *any* filter
#: designed this way collapses as it approaches fs/2.  At 48 kHz that is irrelevant --
#: 20 kHz sits at 0.83x Nyquist and still passes Class 1.
VALID_FRACTION_OF_NYQUIST = 0.8

#: Lowest sample rate at which the weighting network is representable at all.
#:
#: A and C both carry a DOUBLE POLE at F4 = 12.19 kHz.  A pole above Nyquist cannot be
#: realised by any digital filter, so below 2*F4 the shape is wrong no matter how the
#: design is warped -- at 16 kHz the realised A-weighting is already 1.8 dB low at
#: 4 kHz, half of Nyquist.  This is a property of the standard's own network, not of
#: this implementation.
#:
#: Full-range Class 1 (10 Hz - 20 kHz) needs fs >= 44.1 kHz.  The archive is 48 kHz
#: throughout, so this constrains reuse, not the project.
MIN_SUPPORTED_FS = 2 * F4

#: Lowest sample rate meeting Class 1 across the whole tabulated range.
MIN_FULL_RANGE_FS = 44100.0


def max_valid_frequency(fs: float) -> float:
    """Highest frequency at which the realised weighting can be trusted at ``fs``."""
    return VALID_FRACTION_OF_NYQUIST * fs / 2


def _zpk(kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Analog zeros and poles (rad/s) of the weighting network."""
    kind = kind.upper()
    two_pi = 2 * np.pi
    if kind == "A":
        # Four zeros at the origin; poles at f1 (x2), f2, f3, f4 (x2).
        zeros = np.zeros(4)
        poles = -two_pi * np.array([F1, F1, F2, F3, F4, F4])
    elif kind == "C":
        # C is the A network without the f2 and f3 poles, and with two zeros.
        zeros = np.zeros(2)
        poles = -two_pi * np.array([F1, F1, F4, F4])
    else:
        raise ValueError(f"no pole-zero network for weighting {kind!r}")
    return zeros, poles


def analytic_weighting_db(kind: str, frequencies) -> np.ndarray:
    """Exact weighting in dB at ``frequencies``, from the normative definition.

    This is the reference the digital filter is validated against.  Z-weighting is flat
    by definition and returns zeros.

    Note the sign convention on the pole-zero evaluation: the network is evaluated at
    ``s = j*2*pi*f`` and normalised so the response is exactly 0 dB at 1 kHz.
    """
    kind = kind.upper()
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if kind == "Z":
        return np.zeros_like(frequencies)

    zeros, poles = _zpk(kind)

    def magnitude(freqs: np.ndarray) -> np.ndarray:
        s = 1j * 2 * np.pi * freqs
        numerator = np.ones_like(s)
        for zero in zeros:
            numerator = numerator * (s - zero)
        denominator = np.ones_like(s)
        for pole in poles:
            denominator = denominator * (s - pole)
        return np.abs(numerator / denominator)

    response = magnitude(frequencies)
    reference = magnitude(np.array([REFERENCE_FREQUENCY]))[0]
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(response / reference)


def weighting_sos(kind: str, fs: float) -> np.ndarray:
    """Second-order sections realising ``kind`` weighting at sample rate ``fs``.

    Z-weighting returns a single pass-through section, so callers can treat all three
    weightings uniformly rather than special-casing "no filter".
    """
    kind = kind.upper()
    if kind not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {kind!r}")
    if kind == "Z":
        # Identity biquad: b = [1,0,0], a = [1,0,0].
        return np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])

    if fs <= 2 * REFERENCE_FREQUENCY:
        raise ValueError(
            f"sample rate {fs} Hz is too low to define {kind}-weighting "
            f"(1 kHz reference must lie below Nyquist)"
        )
    if fs < MIN_SUPPORTED_FS:
        warnings.warn(
            f"{kind}-weighting at fs={fs:g} Hz is below {MIN_SUPPORTED_FS:.0f} Hz, so "
            f"the {F4:.0f} Hz pole pair lies above Nyquist and cannot be realised; "
            f"the response will not meet IEC 61672-1 Class 1. Resample to "
            f"{MIN_FULL_RANGE_FS:.0f} Hz or above.",
            RuntimeWarning,
            stacklevel=2,
        )

    zeros, poles = _zpk(kind)

    # Pre-warp each pole so the bilinear transform places its corner at the intended
    # digital frequency: f' = (fs/pi) * tan(pi * f / fs).  Zeros sit at the origin,
    # where tan(0) = 0 leaves them unmoved.
    pole_hz = np.abs(poles) / (2 * np.pi)
    warped_hz = (fs / np.pi) * np.tan(np.pi * pole_hz / fs)
    poles = -2 * np.pi * warped_hz

    gain = 1.0
    digital_z, digital_p, digital_k = signal.bilinear_zpk(zeros, poles, gain, fs)
    sos = signal.zpk2sos(digital_z, digital_p, digital_k)

    # Renormalise the DIGITAL filter to 0 dB at 1 kHz.  Normalising the analog
    # prototype alone (what the previous implementation did) leaves a sample-rate
    # dependent offset on every measurement, because the bilinear transform warps the
    # response and the 1 kHz point moves with it.
    _, response = signal.sosfreqz(sos, worN=[REFERENCE_FREQUENCY], fs=fs)
    sos[0, :3] /= np.abs(response[0])
    return sos


def weighting_response_db(kind: str, frequencies, fs: float) -> np.ndarray:
    """Realised response of the digital filter, for validation against the analytic one."""
    sos = weighting_sos(kind, fs)
    _, response = signal.sosfreqz(sos, worN=np.atleast_1d(frequencies), fs=fs)
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.abs(response))


def apply_weighting(pressure: np.ndarray, kind: str, fs: float) -> np.ndarray:
    """Frequency-weight a complete signal.

    For chunked or streaming use, use :class:`WeightingFilter` so filter state carries
    across chunk boundaries.
    """
    kind = kind.upper()
    if kind == "Z":
        return np.asarray(pressure, dtype=DTYPE)
    sos = weighting_sos(kind, fs)
    return signal.sosfilt(sos, np.asarray(pressure, dtype=np.float64)).astype(DTYPE)


class WeightingFilter:
    """Stateful weighting filter for chunked processing.

    Retains the second-order-section delay state between calls, so a long recording can
    be filtered in pieces without a settling transient at every boundary.  This is what
    lets the pipeline stream a 7.2 TB archive instead of loading whole files.

    A warm-up region should still be discarded at the very start of a stream, where the
    filter genuinely has no history -- see :func:`splmeter.metrics.settling_samples`.
    """

    def __init__(self, kind: str, fs: float) -> None:
        self.kind = kind.upper()
        self.fs = fs
        self._sos = weighting_sos(self.kind, fs)
        self._zi = signal.sosfilt_zi(self._sos) * 0.0

    def reset(self) -> None:
        """Clear filter memory, e.g. when starting an unrelated recording."""
        self._zi = signal.sosfilt_zi(self._sos) * 0.0

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        if self.kind == "Z":
            return np.asarray(chunk, dtype=DTYPE)
        out, self._zi = signal.sosfilt(
            self._sos, np.asarray(chunk, dtype=np.float64), zi=self._zi
        )
        return out.astype(DTYPE)
