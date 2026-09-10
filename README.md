# splmeter

A sound level meter calculation engine in Python, implementing **IEC 61672-1:2013**.

Reads recorded audio and computes the quantities a professional sound level meter
reports: frequency-weighted and time-weighted levels, equivalent continuous and peak
levels, statistical exceedance levels, fractional-octave band spectra, and long-term
exposure indices.

Built for the [EAAR Lab](https://github.com/EAAR-LAB) environmental noise archive —
7.2 TB of continuous outdoor monitoring — so it is designed to run over terabytes of
audio rather than over single clips.

```python
import soundfile as sf
from splmeter import Calibration, apply_weighting, leq, lpeak

samples, fs = sf.read("recording.wav")
pressure = Calibration(mic_sensitivity_v_per_pa=0.050).to_pascals(samples)

laeq   = leq(apply_weighting(pressure, "A", fs), fs, window_s=1.0)   # LAeq,1s
lcpeak = lpeak(apply_weighting(pressure, "C", fs), fs, window_s=1.0) # LCpeak,1s
```

All levels are dB re 20 µPa.

## Install

```bash
pip install splmeter                  # core: numpy, scipy
pip install "splmeter[io]"            # + soundfile, to read WAV files
pip install "splmeter[soundquality]"  # + MOSQITO, for psychoacoustic metrics
```

Requires Python 3.9+.

## What it computes

| Group | Metrics |
|---|---|
| Frequency weighting | A, C, Z |
| Time weighting | Fast (125 ms), Slow (1 s), Impulse (35 ms rise / 2.9 dB·s⁻¹ decay) |
| Levels | Leq, Lmax, Lmin under any weighting; Lpeak |
| Exposure | SEL, Lden, CNEL, Ldn — each with its own period boundaries |
| Statistical | Ln exceedance levels (L10, L50, L90, arbitrary n) |
| Spectral | 1/1-, 1/3- and 1/12-octave bands (IEC 61260 base-ten); narrowband spectrogram |
| Sound quality | Loudness (ISO 532-1), sharpness (DIN 45692), tonality TNR/PR (ECMA-74), roughness |
| Quality | Clipping, overload and underrange flags |

Frequency weighting is **explicit, never implied**: `leq` of an A-weighted signal is
LAeq, of a C-weighted signal LCeq. Nothing is inferred from a metric's name.

## Sound quality

Psychoacoustic metrics wrap [MOSQITO](https://github.com/Eomys/MoSQITo) (Apache-2.0)
rather than reimplementing four standards — it is already validated against their
reference signals. It is optional because it requires matplotlib even headless.

```python
from splmeter import soundquality as sq

sones = sq.loudness(pressure, fs, window_s=1.0)   # ISO 532-1, per second
acum  = sq.sharpness(pressure, fs)                # DIN 45692
tones = sq.tonality(pressure, fs)                 # ECMA-74 TNR / PR
```

These cost far more than the level metrics. Measured on one core over a 600 s packet at
48 kHz:

| Metric | Speed | Per 600 s packet |
|---|---|---|
| All level + band metrics combined | 332× realtime | 1.8 s |
| `loudness` (stationary, per second) | 37× realtime | 16 s |
| Time-varying loudness (`loudness_zwtv`) | 1.0× realtime | ~580 s |
| `roughness` | 0.4× realtime | ~1,460 s |

`loudness` deliberately uses **stationary** loudness evaluated per 1-second segment. The
time-varying model exists to capture temporal masking over a few hundred milliseconds,
which a per-second summary averages away regardless — at roughly 36× the cost. Sharpness,
tonality and roughness are intended for selected excerpts, not bulk processing.

## Design

**Everything is O(N).** A 10-minute packet at 48 kHz (28.8 M samples) runs the full
metric set in **1.8 s — about 332× realtime — with peak RSS around 1.4 GB.** Windowed
statistics use reshaped views and prefix sums rather than materialised index matrices.

**Peak level is measured at the full sample rate.** A peak is by definition a
short-duration excursion; decimating before detection systematically under-reads it.

**Silence is `-inf`, not `0`.** A quiet second and a dead microphone are different
measurements and must remain distinguishable. Nothing is clipped to a floor.

**Filter state carries across chunks.** `WeightingFilter` retains its delay state, so a
long recording can be filtered in pieces without a settling transient at each boundary.

**Calibration is separable.** `Calibration` keeps microphone sensitivity, system gain and
ADC full scale apart instead of folding them into one constant. Since a sensitivity
change is a constant dB offset on every level, `with_offset()` lets a whole archive be
recalibrated arithmetically, without reprocessing any audio.

## Accuracy

Validated against IEC 61672-1:2013 Table 3 and its Class 1 tolerance limits
(`tests/test_weighting.py`):

- The analytic weighting network reproduces the standard's tabulated values to within
  **0.05 dB** — exact agreement, since the table is rounded to 0.1 dB.
- The realised digital filters meet **Class 1** across the full tabulated range at
  44.1 kHz and above, and have **exactly 0 dB gain at the 1 kHz reference at every
  sample rate**.
- Summed band energy reconciles with broadband Leq to within 0.35 dB
  (`tests/test_bands.py`).

**Sample rate matters.** The weighting network has a double pole at 12.19 kHz, so below
2 × 12.19 = 24.4 kHz that pole lies above Nyquist and cannot be realised by any digital
filter. `splmeter` warns below that rate. Full-range Class 1 needs **fs ≥ 44.1 kHz**.

Band analysis is FFT-based. Band *levels* are correct in the RMS sense, but band
*shapes* are rectangular and do not meet the IEC 61260 filter-shape masks; bands
narrower than 3 FFT bins are returned as `nan` rather than reported from one or two
bins. For 140 million device-seconds across three band resolutions, a 160-filter IIR
bank was not affordable — this is a deliberate trade, documented rather than hidden.

## Tests

```bash
pip install "splmeter[dev]" && pytest
```

103 tests, validated against the standard and against analytic reference values
(a sine's peak is exactly 3.01 dB above its RMS; a step reaches 1 − 1/e of final value
after one time constant; SEL equals Leq + 10·log10(T)).

## Changes in 0.1.0

This release is a rewrite. Earlier versions contained defects that materially affected
reported levels, so **results produced with 0.0.x should not be compared with 0.1.0
output**:

| Defect | Effect |
|---|---|
| C-weighting poles scaled by `0.062π` instead of `2π` | 32× corner shift — not C-weighting at all |
| `Lpeak` used the signed maximum | Wrong peak for asymmetric transients; `-inf` for an all-negative window |
| Peak and Leq measured after 48:1 decimation with no anti-aliasing filter | Systematic under-read of peaks; aliased broadband levels |
| 1/3-octave used a peak rather than RMS amplitude convention | Every band level 3.01 dB high, irreconcilable with broadband Leq |
| Zero-padding of the first analysis window | First sample of every series was `-inf` |
| Windowed statistics built an `(n_frames, window)` index matrix | ~3.5 GB for one hour of Leq; 83 GB for a day |
| Digital filter normalised only in the analog domain | Sample-rate-dependent offset on every level (+0.16 dB at 8 kHz) |
| License classifier said MIT while `LICENSE` said Apache-2.0 | Ambiguous terms |

New in 0.1.0: Z-weighting, Impulse time weighting, `Lmin`, exceedance levels, SEL,
Lden/CNEL/Ldn as distinct indices, 1/1- and 1/12-octave bands, on-demand narrowband
spectrograms, quality flags, streaming filter state, and separable calibration.

## License

Apache-2.0.
