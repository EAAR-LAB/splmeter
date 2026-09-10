"""Long-term noise exposure indices: Lday/Levening/Lnight, Lden, Ldn and CNEL.

Each index has its **own** period boundaries and penalties.  They are routinely
conflated, including in this project's own notes (``noiseParams_Azad.pdf`` gives a
12/4/8-hour partition under the heading "CNEL (Lden)", which is the Lden partition, not
CNEL's), and the legacy ``calculation.ipynb`` computed exactly that and labelled the
column ``CNEL``.  The three are kept separate here, each with the periods its defining
document specifies:

======  =====================  ==========================  =========================
Index   Day                    Evening                     Night
======  =====================  ==========================  =========================
Lden    07:00-19:00 (12 h)     19:00-23:00 (4 h), +5 dB    23:00-07:00 (8 h), +10 dB
CNEL    07:00-19:00 (12 h)     19:00-22:00 (3 h), +5 dB    22:00-07:00 (9 h), +10 dB
Ldn     07:00-22:00 (15 h)     --                          22:00-07:00 (9 h), +10 dB
======  =====================  ==========================  =========================

Lden follows Directive 2002/49/EC Annex I; CNEL follows the California airport noise
regulations (CCR Title 21 s.5001); Ldn follows the US EPA / FAA day-night level.

Every combination is energetic -- ``10*log10(mean(10^(L/10)))`` -- never an arithmetic
mean of decibels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Period:
    """A named part of the day, with its penalty.

    ``start_hour`` is inclusive and ``end_hour`` exclusive, both in local clock hours.
    A period wraps past midnight when ``start_hour > end_hour`` (e.g. night 23->07).
    """

    name: str
    start_hour: int
    end_hour: int
    penalty_db: float = 0.0

    @property
    def hours(self) -> int:
        span = self.end_hour - self.start_hour
        return span if span > 0 else span + 24

    def contains(self, hour) -> np.ndarray:
        hour = np.asarray(hour)
        if self.start_hour <= self.end_hour:
            return (hour >= self.start_hour) & (hour < self.end_hour)
        # Wraps past midnight.
        return (hour >= self.start_hour) | (hour < self.end_hour)


#: Lden, Directive 2002/49/EC Annex I.
LDEN_PERIODS = (
    Period("day", 7, 19, 0.0),
    Period("evening", 19, 23, 5.0),
    Period("night", 23, 7, 10.0),
)

#: CNEL, California Code of Regulations Title 21 s.5001.  Note the evening is three
#: hours here, not four, and the night starts an hour earlier than for Lden.
CNEL_PERIODS = (
    Period("day", 7, 19, 0.0),
    Period("evening", 19, 22, 5.0),
    Period("night", 22, 7, 10.0),
)

#: Ldn / DNL, US EPA.  No evening period at all.
LDN_PERIODS = (
    Period("day", 7, 22, 0.0),
    Period("night", 22, 7, 10.0),
)


def energy_mean(levels: np.ndarray) -> float:
    """Energetic mean of a level series, ignoring non-finite entries."""
    levels = np.asarray(levels, dtype=np.float64)
    finite = levels[np.isfinite(levels)]
    if finite.size == 0:
        return float("-inf")
    return float(10.0 * np.log10(np.mean(10.0 ** (finite / 10.0))))


def period_levels(
    levels: np.ndarray, hours: np.ndarray, periods=LDEN_PERIODS
) -> dict[str, float]:
    """Energy-averaged level within each period, **without** penalties applied.

    Parameters
    ----------
    levels:
        Sound levels, typically the 1-second LAeq series for one day.
    hours:
        Local clock hour (0-23) of each level.  Derived from ``timestamptz`` values in
        the site's own timezone, so DST is handled by the caller's conversion.
    """
    levels = np.asarray(levels, dtype=np.float64)
    hours = np.asarray(hours)
    if levels.shape != hours.shape:
        raise ValueError(
            f"levels and hours must align: {levels.shape} vs {hours.shape}"
        )
    return {p.name: energy_mean(levels[p.contains(hours)]) for p in periods}


def composite_index(
    levels: np.ndarray, hours: np.ndarray, periods=LDEN_PERIODS
) -> float:
    """Duration- and penalty-weighted 24-hour index for the given period set.

    ``10*log10( sum(h_i * 10^((L_i + penalty_i)/10)) / 24 )``

    Periods with no data are skipped and their hours excluded from the divisor, so a
    partial day yields the index over what was actually measured rather than silently
    treating missing hours as silence.  The returned value is still only meaningful if
    coverage is reasonable -- callers should check completeness separately.
    """
    per_period = period_levels(levels, hours, periods)
    total = 0.0
    weight = 0.0
    for period in periods:
        level = per_period[period.name]
        if not np.isfinite(level):
            continue
        total += period.hours * 10.0 ** ((level + period.penalty_db) / 10.0)
        weight += period.hours
    if weight == 0:
        return float("-inf")
    return float(10.0 * np.log10(total / weight))


def lden(levels: np.ndarray, hours: np.ndarray) -> float:
    """Day-evening-night level (Directive 2002/49/EC)."""
    return composite_index(levels, hours, LDEN_PERIODS)


def cnel(levels: np.ndarray, hours: np.ndarray) -> float:
    """Community noise equivalent level (California).

    Distinct from :func:`lden`: three evening hours rather than four, and the night
    begins at 22:00 rather than 23:00.
    """
    return composite_index(levels, hours, CNEL_PERIODS)


def ldn(levels: np.ndarray, hours: np.ndarray) -> float:
    """Day-night average sound level (US EPA), with no evening period."""
    return composite_index(levels, hours, LDN_PERIODS)
