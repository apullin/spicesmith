"""Observation contracts: data quality and completion, independent of accuracy policy."""
from __future__ import annotations

import math
from typing import Optional

from .deck import Deck, spice_number
from .simulation import RunResult
from .waveform import Reading, Waveform


def read_output(deck: Deck, output: str, data: Optional[bytes]) -> Reading:
    reading = Waveform.read(data)
    if not reading.usable or reading.table is None:
        return reading
    table = reading.table
    request = deck.output_requests.get(output)
    if request is None:
        return reading
    analysis, vectors = request
    missing = sorted(set(vectors) - set(table.names))
    extra = sorted(set(table.names) - set(vectors))
    if missing or extra:
        parts = ([f'missing {", ".join(missing)}'] if missing else [])
        parts += [f'extra {", ".join(extra)}'] if extra else []
        return Reading(table, 'requested vectors differ (' + '; '.join(parts) + ')')
    if not analysis:
        return Reading(table, 'no declared analysis for output')
    try:
        problem = interval_problem(table, analysis)
    except (ValueError, IndexError, ZeroDivisionError, OverflowError):
        problem = 'analysis interval cannot be resolved from the deck'
    return Reading(table, problem)


def interval_problem(table: Waveform, analysis: tuple[str, ...]) -> Optional[str]:
    kind = analysis[0].lower()
    if kind == 'op':
        return None if len(table.scale) == 1 else 'operating point must have one row'
    if kind not in ('tran', 'dc', 'ac'):
        return f'unsupported observation contract: {kind}'
    expected_scale = {'tran': 'time', 'dc': 'v-sweep', 'ac': 'frequency'}[kind]
    if kind == 'dc' and analysis[1].lower().startswith('i'):
        expected_scale = 'i-sweep'
    if table.scale_name != expected_scale:
        return f'scale {table.scale_name} where {kind} requires {expected_scale}'
    if kind == 'tran':
        step, stop = map(spice_number, analysis[1:3])
        start = spice_number(analysis[3]) if len(analysis) > 3 and analysis[3].lower() != 'uic' else 0.0
        if not all(math.isfinite(v) for v in (step, stop, start)) or step <= 0 or not 0 <= start < stop:
            return 'invalid transient interval'
        # With uic ngspice starts at a small positive time, at most the requested output step.
        start_slack = abs(step) if 'uic' in (s.lower() for s in analysis) else 0.0
    elif kind == 'dc':
        start, stop, step = map(spice_number, analysis[2:5])
        if not all(math.isfinite(v) for v in (step, stop, start)) or not step or (stop - start) / step < 0:
            return 'invalid DC sweep interval'
        count = math.floor((stop - start) / step + 1e-8)
        stop = start + count * step  # the last point at or before the requested bound
        start_slack = 0.0
        if len(table.scale) != count + 1:
            return f'DC sweep has {len(table.scale)} rows, expected {count + 1}'
        if any(abs(float(point) - (start + i * step)) > max(abs(step), abs(stop), 1e-30) * 1e-8
               for i, point in enumerate(table.scale)):
            return 'DC sweep points differ from request'
    else:
        start, stop = map(spice_number, analysis[3:5])
        per = float(analysis[2])
        if (not all(math.isfinite(v) for v in (per, start, stop)) or per < 1 or per != int(per)
                or start <= 0 or stop < start):
            return 'invalid AC sweep interval'
        start_slack = 0.0
        # Log sweeps need not land exactly on an upper bound that is off their grid.
        if analysis[1].lower() in ('dec', 'oct'):
            base = 10 if analysis[1].lower() == 'dec' else 2
            count = math.floor(math.log(stop / start, base) * per + 1e-8)
            stop = start * base ** (count / per)
            points = (start * base ** (i / per) for i in range(count + 1))
        elif analysis[1].lower() == 'lin':
            count = int(per) - 1
            stop = stop if count else start
            points = (start + (stop - start) * i / max(1, count) for i in range(count + 1))
        else:
            return 'unsupported AC sweep spacing'
        if len(table.scale) != count + 1:
            return f'AC sweep has {len(table.scale)} rows, expected {count + 1}'
        if any(abs(float(actual) - expected) > max(abs(expected), 1e-30) * 1e-8
               for actual, expected in zip(table.scale, points, strict=True)):
            return 'AC sweep points differ from request'
    tolerance = max(abs(start), abs(stop), abs(stop - start), 1e-30) * 1e-8
    first, last = float(table.scale[0]), float(table.scale[-1])
    if abs(first - start) > start_slack + tolerance or abs(last - stop) > tolerance:
        return f'incomplete {kind} interval: {first:g}..{last:g}, requested {start:g}..{stop:g}'
    return None


def output_problems(deck: Deck, run: RunResult) -> dict[str, str]:
    """Every expected artifact is inspected, even if the simulator failed."""
    return {name: reading.problem for name in deck.outputs
            if (reading := read_output(deck, name, run.outputs.get(name))).problem is not None}
