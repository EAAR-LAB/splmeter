"""splmeter -- a sound level meter calculation engine.

Computes IEC 61672-1 sound level metrics from recorded audio: frequency and time
weighting, equivalent and peak levels, statistical exceedance levels, fractional-octave
band analysis, and long-term exposure indices.

Quick start
-----------
::

    import soundfile as sf
    from splmeter import Calibration, apply_weighting, leq, lpeak

    samples, fs = sf.read("recording.wav")
    pressure = Calibration(mic_sensitivity_v_per_pa=0.050).to_pascals(samples)

    laeq   = leq(apply_weighting(pressure, "A", fs), fs, window_s=1.0)
    lcpeak = lpeak(apply_weighting(pressure, "C", fs), fs, window_s=1.0)

Levels are dB re 20 uPa.  Frequency weighting is explicit rather than implied: an
``leq`` of an A-weighted signal is LAeq, of a C-weighted signal LCeq.
"""

from . import soundquality
from .bands import (
    OCTAVE,
    THIRD_OCTAVE,
    TWELFTH_OCTAVE,
    BandSet,
    band_exceedance_levels,
    band_levels,
    band_sum_level,
    spectrogram,
)
from .core import P_REF, REFERENCE_CALIBRATION, Calibration, to_db
from .exposure import cnel, ldn, lden, period_levels
from .metrics import (
    energy_average,
    exceedance_levels,
    impulsiveness,
    leq,
    leq_time_weighted,
    lmax,
    lmin,
    lpeak,
    sel,
    time_weight,
)
from .quality import QualityFlags, assess
from .weighting import (
    WeightingFilter,
    analytic_weighting_db,
    apply_weighting,
    weighting_response_db,
)

__version__ = "0.1.0"

__all__ = [
    "BandSet", "Calibration", "OCTAVE", "P_REF", "QualityFlags",
    "REFERENCE_CALIBRATION", "THIRD_OCTAVE", "TWELFTH_OCTAVE", "WeightingFilter",
    "analytic_weighting_db", "apply_weighting", "assess", "band_exceedance_levels",
    "band_levels",
    "band_sum_level", "cnel", "energy_average", "exceedance_levels", "ldn", "lden",
    "impulsiveness", "leq", "leq_time_weighted", "lmax", "lmin", "lpeak", "period_levels", "sel", "spectrogram",
    "soundquality", "time_weight", "to_db", "weighting_response_db", "__version__",
]
