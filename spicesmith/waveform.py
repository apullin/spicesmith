"""Waveforms and the accuracy measures of the legacy approximate oracle.

A transient output node is a *timing* node when its mid-rail (VDD/2) crossings are well
defined, and an *analog* node otherwise. Timing nodes are compared by crossing count and
crossing shift, analog nodes by the largest voltage deviation on the reference's grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

# Crossings use hysteresis (a Schmitt trigger): a transition must swing from below to above
# mid-rail +- BAND x VDD, so round-off and small signals around mid-rail do not count. The
# count is fragile when the reference has an excursion that peaks near the band edge (a
# small change in the peak adds or removes a crossing), so a node is a timing node only if
# the reference's count is the same for every band between FRAGILE[0] and FRAGILE[1].
BAND = 0.1  # x VDD, half-width of the counting band
FRAGILE = (0.05, 0.2)  # x VDD
SMOOTH = 0.25  # largest change of the shift between paired crossings, x their interval
STARTUP = 0.02  # x tstop: the power-up window of a transient that starts from uic


@dataclass(frozen=True, eq=False)
class Waveform:
    """A wrdata table written with wr_singlescale and wr_vecnames: one scale (time,
    frequency, sweep) and the observed vectors, complex for AC."""
    scale_name: str
    scale: np.ndarray
    names: tuple[str, ...]
    values: np.ndarray  # (len(scale), len(names))
    components: tuple[int, ...] = ()  # physical columns per logical vector (real or complex)

    @classmethod
    def read(cls, data: Optional[bytes]) -> Reading:
        """The table in an output file, or why it cannot be compared: not written, empty, not
        a rectangular numeric table, non-finite values, or a scale that does not advance
        (time and frequency strictly increasing, a sweep strictly monotonic). Complex vectors
        are two columns of the same name."""
        if data is None:
            return Reading(None, 'not written')
        header, _, body = data.decode(errors='replace').partition('\n')
        names = header.split()
        if not names:
            return Reading(None, 'empty')
        if len(names) < 2:
            return Reading(None, 'no vectors in the header')
        fields = [line.split() for line in body.splitlines() if line.strip()]
        if not fields:
            return Reading(None, 'no rows')
        if any(len(row) != len(names) for row in fields):
            return Reading(None, 'ragged rows')
        try:
            rows = np.array(fields, dtype=float)
        except ValueError:
            return Reading(None, 'not a numeric table')
        if not np.all(np.isfinite(rows)):
            return Reading(None, 'non-finite values')
        columns, kept, components, i = [], [], [], 1
        while i < len(names):
            if names[i] in kept:
                return Reading(None, f'duplicate vector {names[i]}')
            if i + 1 < len(names) and names[i + 1] == names[i]:
                if names[0] != 'frequency':
                    return Reading(None, f'unexpected complex vector {names[i]}')
                columns.append(rows[:, i] + 1j * rows[:, i + 1])
                components.append(2)
                i += 2
            else:
                columns.append(rows[:, i])
                components.append(1)
                i += 1
            kept.append(names[i - 1])
        table = cls(names[0], rows[:, 0], tuple(kept), np.column_stack(columns), tuple(components))
        return Reading(table, table.scale_problem())

    @classmethod
    def parse(cls, data: Optional[bytes]) -> Optional[Waveform]:
        """The table if it is usable (see `read`), else None."""
        reading = cls.read(data)
        return reading.table if reading.usable else None

    def scale_problem(self) -> Optional[str]:
        """Why the scale cannot carry a comparison, None if it can. A single row (an
        operating point, whose first vector stands in for the scale) needs nothing."""
        if len(self.scale) < 2:
            return None
        steps = np.diff(self.scale)
        if self.scale_name in ('time', 'frequency'):
            return None if np.all(steps > 0) else f'{self.scale_name} does not increase'
        return None if np.all(steps > 0) or np.all(steps < 0) else f'{self.scale_name} is not monotonic'

    def mismatch(self, other: Waveform) -> Optional[str]:
        """How `other` differs in shape from this table (scale or vector set), None if alike."""
        if other.scale_name != self.scale_name:
            return f'scale {other.scale_name} where tight has {self.scale_name}'
        missing = [n for n in self.names if n not in other.names]
        extra = [n for n in other.names if n not in self.names]
        if missing or extra:
            parts = ([f'missing {", ".join(missing[:5])}'] if missing else []) + \
                    ([f'extra {", ".join(extra[:5])}'] if extra else [])
            return 'vectors differ from tight (' + '; '.join(parts) + ')'
        mine = dict(zip(self.names, self.components or (1,) * len(self.names), strict=True))
        theirs = dict(zip(other.names, other.components or (1,) * len(other.names), strict=True))
        if mine != theirs:
            return 'real/complex vector representation differs from tight'
        if (len(self.scale) == 1 or len(other.scale) == 1) and len(self.scale) != len(other.scale):
            return f'{len(other.scale)} rows where tight has {len(self.scale)}'
        if self.scale_name == 'frequency' and (len(self.scale) != len(other.scale)
                                               or not np.allclose(self.scale, other.scale, rtol=1e-12, atol=0)):
            return 'frequency points differ from tight'
        if len(self.scale) > 1 and self.scale_name != 'frequency':
            tolerance = max(abs(float(self.scale[-1])), float(np.ptp(self.scale)), 1e-30) * 1e-8
            if abs(float(other.scale[-1] - self.scale[-1])) > tolerance:
                return 'analysis interval differs from tight'
            if not self.is_transient and (len(self.scale) != len(other.scale)
                                          or not np.allclose(self.scale, other.scale, rtol=1e-8, atol=tolerance)):
                return 'sweep points differ from tight'
        return None

    @property
    def is_transient(self) -> bool:
        return self.scale_name == 'time'

    def column(self, name: str) -> np.ndarray:
        return self.values[:, self.names.index(name)]

    def trace(self, name: str) -> Trace:
        return Trace(self.scale, self.column(name).real)


@dataclass(frozen=True, eq=False)
class Reading:
    """An output file read for comparison: its table, and why it cannot be used, if so."""
    table: Optional[Waveform]
    problem: Optional[str] = None

    @property
    def usable(self) -> bool:
        return self.problem is None and self.table is not None


@dataclass(frozen=True, eq=False)
class Crossings:
    """Mid-rail crossings: times and directions (+1 rising, -1 falling). With hysteresis a
    crossing registers only once the signal clears the band; `lag` is the longest such delay,
    so a transition that passes mid-rail just before the end of the window may not count."""
    times: np.ndarray
    directions: np.ndarray
    lag: float = 0.0

    def __len__(self) -> int:
        return len(self.times)

    def match(self, other: Crossings, tend: float, slack: float, startup: float = 0.0) -> Match:
        """Pair crossings in order, as `aligned` does; up to two leading crossings of either
        sequence that fall before `startup` may go unpaired (a uic power-up glitch that one
        time-step pattern resolves and another does not)."""
        first = self.aligned(other, tend, slack)
        if first.agree or startup <= 0:
            return first
        for mine, theirs in ((1, 0), (0, 1), (2, 0), (0, 2), (1, 1)):
            if self.leading(mine, startup) and other.leading(theirs, startup):
                match = self.drop(mine).aligned(other.drop(theirs), tend, slack)
                if match.agree:
                    return match
        return first

    def leading(self, count: int, before: float) -> bool:
        """Whether the first `count` crossings all lie before `before`."""
        return len(self) >= count and bool(np.all(self.times[:count] < before))

    def drop(self, count: int) -> Crossings:
        return Crossings(self.times[count:], self.directions[count:], self.lag)

    def aligned(self, other: Crossings, tend: float, slack: float) -> Match:
        """Pair crossings in order (directions must agree) and find the largest shift.

        Unpaired crossings are forgiven at the end of the window when they lie within the
        matched shift plus `slack` of `tend`: a shifted edge can leave the window, and an
        oscillator whose frequency is slightly off ends a crossing or two ahead, without
        anything qualitative changing. A drifting oscillator's first unpaired crossing can
        come up to one drift increment earlier than that, and a transition can pass mid-rail
        without registering before the end (`lag`), so twice the largest increment and the
        longest lag are added.

        Drift changes the shift little from one pair to the next; pairing out of step (an
        extra or missing transition early on) makes it jump by about an interval, so a jump
        of more than a quarter of the local interval is a disagreement."""
        n = min(len(self), len(other))
        shifts = other.times[:n] - self.times[:n]
        shift = float(np.max(np.abs(shifts))) if n else 0.0
        if (self.directions[:n] != other.directions[:n]).any():
            return Match(False, shift)
        steps = np.abs(np.diff(shifts))
        if n >= 2 and np.any(steps > SMOOTH * np.diff(self.times[:n]) + slack):
            return Match(False, shift)
        extra = np.concatenate([self.times[n:], other.times[n:]])
        increment = 2 * float(steps.max()) if n >= 2 else 0.0
        horizon = tend - (shift + slack + increment + max(self.lag, other.lag))
        return Match(bool(np.all(extra > horizon)), shift)


@dataclass(frozen=True)
class Match:
    agree: bool
    shift: float  # s, largest shift between paired crossings


@dataclass(frozen=True, eq=False)
class Trace:
    """One node's voltage over time."""
    time: np.ndarray
    voltage: np.ndarray

    def crossings(self, level: float, band: float, settle: float = 0.0) -> Crossings:
        """Crossings of `level` with hysteresis: v must move from below level - band to above
        level + band (or back). The time of a crossing is the midpoint between the moment v
        last leaves the old band edge and the moment it clears the new one: a slow edge that
        wobbles across `level` inside the band moves the last (or first) pass through `level`
        by tens of ps, but not that midpoint, and on a clean edge the two agree. Nothing
        counts before v first leaves level +- settle: a node that starts between the rails
        (uic, power-up) settles to one side without a transition."""
        v = np.asarray(self.voltage, dtype=float)
        state = np.where(v > level + band, 1, np.where(v < level - band, -1, 0))
        outside = np.flatnonzero(np.abs(v - level) > max(band, settle))
        state[:outside[0] if outside.size else len(v)] = 0
        defined = np.flatnonzero(state)
        if defined.size < 2:
            return _NO_CROSSINGS
        held = state[defined]
        changes = defined[1:][held[1:] != held[:-1]]  # first sample beyond the band on the new side
        if changes.size == 0:
            return _NO_CROSSINGS
        before = defined[np.searchsorted(defined, changes) - 1]  # last sample beyond the old edge
        directions = state[changes]
        entered = self._at(changes - 1, level + directions * band)
        left = self._at(before, level - directions * band)
        times = (left + entered) / 2
        return Crossings(times, directions, float(np.max(self.time[changes] - times)))

    def _at(self, k: np.ndarray, levels: np.ndarray) -> np.ndarray:
        """Times where the segments k..k+1 reach `levels`, linearly interpolated."""
        t, v = self.time, np.asarray(self.voltage, dtype=float)
        return t[k] + (t[k + 1] - t[k]) * (levels - v[k]) / (v[k + 1] - v[k])

    def deviation_from(self, reference: Trace) -> float:
        """Largest |self - reference| on the reference's grid, self linearly interpolated."""
        if not len(reference.time):
            return 0.0
        return float(np.max(np.abs(np.interp(reference.time, self.time, self.voltage) - reference.voltage)))


_NO_CROSSINGS = Crossings(np.empty(0), np.empty(0, dtype=int))


@dataclass(frozen=True)
class NodeError:
    """One node of one run against the accuracy reference."""
    timing: bool  # crossings are well defined in the reference
    counts: bool  # crossing sequences agree (timing nodes)
    shift: float  # s, largest crossing shift
    dv: float  # V, largest voltage deviation
    crossings: int  # in the reference


@dataclass(frozen=True)
class AccuracyMeasure:
    """Node-by-node errors of transient waveforms against a reference."""
    vdd: float
    tend: float
    slack: float  # s, forgiven for an edge leaving the window
    startup: float = 0.0  # s, power-up window of a uic transient (see Crossings.match)
    band: float = BAND
    fragile: tuple[float, float] = FRAGILE

    def node_error(self, reference: Trace, other: Trace) -> NodeError:
        # Every count establishes a node's first state at the counting band, so the fragility
        # bands differ only in what a later transition must clear.
        level, settle = self.vdd / 2, self.band * self.vdd
        mine = reference.crossings(level, self.band * self.vdd, settle)
        theirs = other.crossings(level, self.band * self.vdd, settle)
        robust = (len(reference.crossings(level, self.fragile[0] * self.vdd, settle))
                  == len(reference.crossings(level, self.fragile[1] * self.vdd, settle)))
        timing = robust and (len(mine) > 0 or len(theirs) > 0)
        match = mine.match(theirs, self.tend, self.slack, self.startup) if timing else Match(True, 0.0)
        return NodeError(timing, match.agree, match.shift, other.deviation_from(reference), len(mine))

    def node_errors(self, reference: Waveform, other: Waveform) -> Mapping[str, NodeError]:
        """Errors of every vector the two waveforms share."""
        return {name: self.node_error(reference.trace(name), other.trace(name))
                for name in reference.names if name in other.names}
