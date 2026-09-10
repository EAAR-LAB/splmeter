"""Signal containers, calibration, and shared constants.

Design notes
------------
The engine is built as **plain numpy functions over plain arrays**, with a thin
convenience layer on top.  The previous implementation wrapped every operation in a
Keras-style module that ``deepcopy``-ed the entire signal on each call and materialised
an ``(n_out, window)`` index matrix for windowed statistics; one hour of ``Leq`` needed
~3.5 GB and a 24-hour file ~83 GB.  That is the single reason the old pipeline had to
decimate to 1 kHz before measuring, which in turn corrupted peak levels.  Functions over
arrays keep the hot paths O(N) and let callers stream.

Calibration
-----------
:class:`Calibration` separates the three quantities the old ``VoltToSPL`` conflated into
one ``mic_sensitivity`` number:

* **microphone sensitivity** (V/Pa) -- a property of the capsule,
* **preamp/system gain** (dB) -- a property of the signal chain,
* **ADC full scale** (V) -- what a sample value of 1.0 corresponds to in volts.

``soundfile`` returns normalised floats in [-1, 1], not volts, so a single "sensitivity"
constant was silently absorbing all three.  Keeping them apart is what makes the archive
re-derivable: see :meth:`Calibration.with_offset`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

#: Reference sound pressure, 20 uPa (IEC 61672-1).
P_REF = 2e-5

#: Working dtype.  float32 halves memory and bandwidth against float64 with no
#: meaningful accuracy cost at these levels: 24 bits of mantissa against a signal whose
#: own dynamic range is ~96 dB (16-bit).  Accumulators promote to float64 where it
#: matters (see :func:`splmeter.metrics.leq`).
DTYPE = np.float32


@dataclass(frozen=True)
class Calibration:
    """Converts recorded sample values to sound pressure in pascals.

    Parameters
    ----------
    mic_sensitivity_v_per_pa:
        Open-circuit sensitivity of the capsule, volts per pascal.  Manufacturers often
        quote dB re 1 V/Pa instead -- use :meth:`from_dbv` for that.
    gain_db:
        Any additional gain in the chain (preamp, sound-card input gain), in dB.
        Positive gain means the recorded signal is larger than the mic output.
    fullscale_volts:
        Volts corresponding to a sample value of 1.0.
    offset_db:
        Per-device correction applied on top, from comparison against a reference sound
        level meter.  Held separately from the physical terms so it can be refined
        later without pretending it is a property of the microphone.
    """

    mic_sensitivity_v_per_pa: float
    gain_db: float = 0.0
    fullscale_volts: float = 1.0
    offset_db: float = 0.0

    @classmethod
    def from_dbv(cls, sensitivity_dbv: float, **kwargs) -> "Calibration":
        """Build from a sensitivity quoted in dB re 1 V/Pa (e.g. -26.0)."""
        return cls(mic_sensitivity_v_per_pa=10 ** (sensitivity_dbv / 20), **kwargs)

    @property
    def scale(self) -> float:
        """Multiplier taking a normalised sample value to pascals."""
        gain = 10 ** ((self.gain_db + self.offset_db) / 20)
        return self.fullscale_volts / (self.mic_sensitivity_v_per_pa * gain)

    def to_pascals(self, samples: np.ndarray) -> np.ndarray:
        """Convert normalised samples (or volts, with ``fullscale_volts=1``) to Pa."""
        return np.asarray(samples, dtype=DTYPE) * DTYPE(self.scale)

    def with_offset(self, offset_db: float) -> "Calibration":
        """Return a copy with a different per-device offset.

        Because a sensitivity change is a *constant dB offset* on every level metric,
        levels computed under one calibration can be corrected to another by adding
        ``new.offset_db - old.offset_db``.  That is why the pipeline stores levels
        against a fixed reference calibration and applies the per-device offset at query
        time: recalibration never requires reprocessing the audio.
        """
        return replace(self, offset_db=offset_db)


#: The calibration all archive levels are computed against.  Per-device corrections live
#: in the database, not here, so this value must never change once the bulk run starts.
REFERENCE_CALIBRATION = Calibration(mic_sensitivity_v_per_pa=0.050)


def to_db(power: np.ndarray, ref: float = P_REF) -> np.ndarray:
    """Convert mean-square pressure to a level in dB, mapping silence to ``-inf``.

    ``10*log10(0)`` warns and returns ``-inf`` in numpy.  Silence is a legitimate
    measurement, not an error, so the warning is suppressed deliberately rather than
    papered over by clipping.  The old pipeline clipped levels to ``[0, 1000]``, which
    made a genuinely quiet second indistinguishable from a sensor dropout -- and the
    downstream notebooks then discarded both as outliers.
    """
    power = np.asarray(power, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(power / (ref * ref))
