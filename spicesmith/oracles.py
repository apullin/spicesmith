"""The legacy acceptance contracts documented in README.md, as oracles.

An Oracle looks at the Evidence of one case (the deck, every configuration's run, the
front-end dumps, repeated runs) and returns a Judgement: check results with their findings,
notes, metrics, and the reasons a case is inconclusive. The Verdict of a case combines the
judgements of all oracles.

A finding is a sentence that starts with the configuration (or check) it concerns, e.g.
'flags: waveform.txt differs from ref'; the reducer selects findings by prefix. Every check
also records an Outcome under 'configuration:check' for per-check pass rates.
"""
from __future__ import annotations

import functools
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Collection, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np

from .config import CONFIGURATIONS, Claim, Configuration
from .deck import Deck
from .integrity import read_output
from .simulation import BUSY, RunResult
from .waveform import STARTUP, AccuracyMeasure, NodeError, Reading, Waveform


class Outcome(Enum):
    PASS = 'pass'
    FAIL = 'fail'
    SKIP = 'skip'


@dataclass(frozen=True)
class CheckResult:
    key: str  # 'configuration:check', e.g. 'flags:outputs'
    outcome: Outcome
    findings: tuple[str, ...] = ()

    @classmethod
    def of(cls, key: str, findings: Sequence[str]) -> CheckResult:
        """Failed if there are findings, passed otherwise."""
        return cls(key, Outcome.FAIL if findings else Outcome.PASS, tuple(findings))

    @classmethod
    def skipped(cls, key: str) -> CheckResult:
        return cls(key, Outcome.SKIP)


@dataclass(frozen=True)
class Judgement:
    """What one oracle concluded about one case."""
    checks: tuple[CheckResult, ...] = ()
    notes: tuple[str, ...] = ()
    inconclusive: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def findings(self) -> tuple[str, ...]:
        return tuple(f for check in self.checks for f in check.findings)


@dataclass(frozen=True)
class Repeat:
    """One configuration run twice on the same deck."""
    configuration: str
    first: RunResult
    second: RunResult


@dataclass(frozen=True)
class FrontEnd:
    """The parse-only runs of ref and the candidate (default environment) under
    `ngspice -D ngdebug`, and the directories they dumped their decks into."""
    ref_dir: Path
    candidate_dir: Path
    ref_run: Optional[RunResult] = None
    candidate_run: Optional[RunResult] = None


@dataclass(frozen=True)
class Evidence:
    """Everything the oracles look at for one case."""
    deck: Deck
    runs: Mapping[str, RunResult]
    front_end: Optional[FrontEnd] = None
    repeats: tuple[Repeat, ...] = ()
    tight_level: Optional[int] = None  # tolerance level the accuracy reference finished at
    definitions: Mapping[str, Configuration] = field(default_factory=lambda: CONFIGURATIONS)
    samples: Mapping[str, tuple[RunResult, ...]] = field(default_factory=dict)

    @property
    def reference(self) -> RunResult:
        return self.runs['ref']

    @property
    def accuracy_reference(self) -> Optional[RunResult]:
        return self.runs.get('tight')

    def claiming(self, claim: Claim) -> Iterator[tuple[str, RunResult]]:
        """The runs of configurations held to `claim`."""
        for name, run in self.runs.items():
            configuration = self.definitions.get(name)
            if configuration is not None and configuration.claim is claim:
                yield name, run

    def without(self, names: Collection[str]) -> Evidence:
        """This evidence without the runs (and timing samples) of `names`."""
        return replace(self, runs={k: v for k, v in self.runs.items() if k not in names},
                       samples={k: v for k, v in self.samples.items() if k not in names})


class Oracle(ABC):
    """A check of a legacy acceptance claim."""

    @abstractmethod
    def judge(self, evidence: Evidence) -> Judgement:
        """Check one case."""


# --- Bit-exact claim -------------------------------------------------------------------------

@dataclass(frozen=True)
class BitExactOracle(Oracle):
    """Configurations claiming exact equivalence against ref: status, every output byte,
    diagnostics, and ngspice's counters (deck lines, matrix structure, iterations)."""
    counters: tuple[str, ...] = ('deck_lines', 'equations', 'nonzeros', 'fillin', 'total_nonzeros', 'iterations',
                                 'tran_iterations', 'timepoints', 'accepted', 'rejected')

    def judge(self, evidence: Evidence) -> Judgement:
        ref = evidence.reference
        checks: list[CheckResult] = []
        for name, got in evidence.claiming(Claim.EXACT):
            checks += self._compare(name, got, ref)
        return Judgement(tuple(checks))

    def _compare(self, name: str, got: RunResult, ref: RunResult) -> Iterator[CheckResult]:
        if got.outcome != ref.outcome:
            yield CheckResult.of(f'{name}:status', [f'{name}: exit status {got.outcome} vs ref {ref.outcome}'])
            yield CheckResult.skipped(f'{name}:outputs')
        else:
            yield CheckResult.of(f'{name}:status', [])
            yield CheckResult.of(f'{name}:outputs', [f'{name}: {output} differs from ref' for output in ref.outputs
                                                     if got.outputs.get(output) != ref.outputs[output]])
        differ = [] if got.diagnostics == ref.diagnostics else [
            f'{name}: diagnostics differ from ref ({first_difference(ref.diagnostics, got.diagnostics)})']
        yield CheckResult.of(f'{name}:diagnostics', differ)
        required = [k for k in self.counters if k in ref.stats]  # every counter ref printed
        if not required:  # a deck without `rusage all`
            yield CheckResult.skipped(f'{name}:matrix')
            return
        missing = [k for k in required if k not in got.stats]
        changed = [f'{k} {got.stats[k]:g} vs {ref.stats[k]:g}' for k in required
                   if k in got.stats and got.stats[k] != ref.stats[k]]
        findings = ([f'{name}: counters missing ({", ".join(missing)})'] if missing else []) + \
                   ([f'{name}: counters differ from ref ({", ".join(changed)})'] if changed else [])
        yield CheckResult.of(f'{name}:matrix', findings)


def first_difference(expected: Sequence[str], got: Sequence[str]) -> str:
    """The first line only in `got`, or only in `expected`, for a finding's detail."""
    extra = [line for line in got if line not in expected]
    missing = [line for line in expected if line not in got]
    if extra:
        return f"new: '{extra[0][:120]}'"
    if missing:
        return f"missing: '{missing[0][:120]}'"
    return 'order or repetition'


# --- Approximate claim -----------------------------------------------------------------------

class VectorComparison(ABC):
    """How the vectors of one kind of non-transient output compare against tight."""
    check: ClassVar[str]  # the check's name, e.g. 'dc'

    @abstractmethod
    def applies(self, table: Waveform) -> bool:
        """Whether this comparison is for outputs like `table`."""

    @abstractmethod
    def error(self, tight: Waveform, other: Waveform, vector: str) -> float:
        """How far `other`'s vector is from tight's."""

    @abstractmethod
    def allowance(self, reference_error: float, vdd: float) -> float:
        """The largest error a run may have, given ref's."""

    @abstractmethod
    def describe(self, error: float) -> str:
        """An error for a finding's text."""


@dataclass(frozen=True)
class StaticComparison(VectorComparison):
    """Operating points and DC sweeps: the largest |dV| on tight's sweep points. The
    approximate flags change DC results only through round-off and Newton paths, so the
    slack is small: max(0.2% VDD, ref's own error)."""
    check: ClassVar[str] = 'dc'
    slack: float = 0.002  # x VDD

    def applies(self, table: Waveform) -> bool:
        return not table.is_transient and table.scale_name != 'frequency'

    def error(self, tight: Waveform, other: Waveform, vector: str) -> float:
        """Both tables passed Waveform.read and mismatch: single rows match row for row, sweeps
        are monotonic, so other is interpolated onto tight's sweep points."""
        reference, values = tight.column(vector).real, other.column(vector).real
        if len(tight.scale) > 1:
            scale = other.scale
            if scale[0] > scale[-1]:  # a falling sweep
                scale, values = scale[::-1], values[::-1]
            order = np.argsort(tight.scale)
            values = np.interp(tight.scale[order], scale, values)
            reference = reference[order]
        return float(np.max(np.abs(values - reference)))

    def allowance(self, reference_error: float, vdd: float) -> float:
        return reference_error + max(self.slack * vdd, reference_error)

    def describe(self, error: float) -> str:
        return f'{error:.4g} V'


@dataclass(frozen=True)
class SmallSignalComparison(VectorComparison):
    """AC sweeps: the largest complex error relative to tight's magnitude (floored at 1e-3 of
    the vector's peak, so deep notches do not dominate); max(1e-3, ref's own error) slack."""
    check: ClassVar[str] = 'ac'
    floor: float = 1e-3
    slack: float = 1e-3

    def applies(self, table: Waveform) -> bool:
        return table.scale_name == 'frequency'

    def error(self, tight: Waveform, other: Waveform, vector: str) -> float:
        """Both tables have the same frequency points (Waveform.mismatch). A vector that is
        zero in tight is measured against the other's own peak, so it stays finite."""
        reference, values = tight.column(vector), other.column(vector)
        magnitude = np.abs(reference)
        peak = max(float(magnitude.max()), float(np.abs(values).max()), 1e-30)
        scale = np.maximum(magnitude, self.floor * peak)
        return float(np.max(np.abs(values - reference) / scale))

    def allowance(self, reference_error: float, vdd: float) -> float:
        return reference_error + max(self.slack, reference_error)

    def describe(self, error: float) -> str:
        return f'{error:.3g} relative'


@dataclass(frozen=True)
class AccuracyOracle(Oracle):
    """Approximate configurations: succeed wherever ref succeeds, and stay as close to tight
    as ref does. Transient outputs node by node (crossing count and shift on timing nodes,
    |dV| elsewhere); operating points, DC and AC sweeps vector by vector."""
    # The crossing shift may exceed ref's by max(2 ps, 2e-4 x tstop, 5% of ref's shift): the
    # last term matters for oscillators, whose phase error accumulates to hundreds of ps.
    shift_slack: tuple[float, float] = (2e-12, 2e-4)
    shift_relative: float = 0.05
    dv_slack: float = 0.02  # |dV| may exceed ref's by max(2% VDD, ref's)
    comparisons: tuple[VectorComparison, ...] = (StaticComparison(), SmallSignalComparison())
    PRIMARY = 'waveform.txt'  # nodes of other transient outputs are keyed 'v(n)@file'

    def judge(self, evidence: Evidence) -> Judgement:
        ref, tight = evidence.reference, evidence.accuracy_reference
        if tight is None:
            return Judgement()
        inconclusive = [f'{who} {run.outcome}' for who, run in (('ref', ref), ('tight', tight)) if not run.ok]
        context = _Context(evidence.deck, tight, ref, self._measure(evidence.deck))
        if context.measure is None and list(evidence.claiming(Claim.APPROXIMATE)):
            inconclusive.append('legacy accuracy requires a declared supply voltage')
        if ref.ok and tight.ok:
            inconclusive += context.problems  # unusable reference outputs are never a silent skip
        checks: list[CheckResult] = []
        nodes: dict[str, dict[str, Any]] = {}
        metrics: dict[str, Any] = {}
        for name, got in evidence.claiming(Claim.APPROXIMATE):
            checks += self._judge_run(name, got, context, nodes, metrics)
            metrics.update(self._summary(name, nodes))
            metrics.update({f'{name}_{k}': got.stats[k] for k in ('tran_iterations', 'timepoints', 'rejected')
                            if k in got.stats})
        disagree = sorted(n for n, e in nodes.items() if e['timing'] and not e['ref'][0])
        if disagree:
            inconclusive.append(f'ref disagrees with tight on {len(disagree)} node(s): {", ".join(disagree[:5])}')
        if nodes:
            metrics['nodes'] = nodes
        metrics.update({f'ref_{k}': ref.stats[k] for k in ('tran_iterations', 'timepoints', 'rejected')
                        if k in ref.stats})
        return Judgement(tuple(checks), (), tuple(inconclusive), metrics)

    def _measure(self, deck: Deck) -> Optional[AccuracyMeasure]:
        vdd, tstop = deck.vdd, deck.tstop or 0.0
        if vdd is None:
            return None
        uic = any(a.split()[0].lower() == 'tran' and 'uic' in a.lower().split() for a in deck.analyses)
        return AccuracyMeasure(vdd, tstop, max(self.shift_slack[0], self.shift_slack[1] * tstop),
                               STARTUP * tstop if uic else 0.0)

    def _judge_run(self, name: str, got: RunResult, context: _Context, nodes: dict[str, dict[str, Any]],
                   metrics: dict[str, Any]) -> Iterator[CheckResult]:
        ref, tight, measure = context.ref, context.tight, context.measure
        transient_keys = [f'{name}:{check}' for check in ('crossing-count', 'crossing-shift', 'dv')]
        vector_keys = [f'{name}:{c.check}' for c in self.comparisons]
        if not ref.ok:
            yield CheckResult.skipped(f'{name}:status')
        else:
            failed = [] if got.ok else [f'{name}: exit status {got.outcome}, ref succeeded']
            yield CheckResult.of(f'{name}:status', failed)
        output_key = f'{name}:outputs'
        if measure is None or not (ref.ok and tight.ok and got.ok) or not context.tables:
            yield from (CheckResult.skipped(key) for key in [output_key] + transient_keys + vector_keys)
            return
        findings: dict[str, list[str]] = {key: [] for key in [output_key] + transient_keys + vector_keys}
        compared: set[str] = {output_key}
        for output, reference in context.tables.items():
            reading = read_output(context.deck, output, got.outputs.get(output))
            table = reading.table
            problem = reading.problem or (reference.mismatch(table) if table is not None else 'unreadable')
            if problem is not None or table is None:
                findings[output_key].append(f'{name}: {output} unusable ({problem})')
                continue
            if reference.is_transient:
                compared.update(transient_keys)
                self._transient(name, output, reference, table, measure, context, nodes, findings)
                continue
            for comparison in self.comparisons:
                if comparison.applies(reference):
                    key = f'{name}:{comparison.check}'
                    compared.add(key)
                    findings[key] += self._vectors(name, output, comparison, reference, table, measure.vdd, context,
                                                   metrics)
        for key, found in findings.items():
            yield CheckResult.of(key, found) if key in compared else CheckResult.skipped(key)

    def _transient(self, name: str, output: str, reference: Waveform, table: Waveform, measure: AccuracyMeasure,
                   context: _Context, nodes: dict[str, dict[str, Any]], findings: dict[str, list[str]]) -> None:
        reference_errors = context.reference_node_errors(output)
        for node, g in measure.node_errors(reference, table).items():
            r = reference_errors.get(node)
            if r is None:
                continue
            key = node if output == self.PRIMARY else f'{node}@{output}'
            entry = nodes.setdefault(key, {'timing': r.timing, 'crossings': r.crossings,
                                           'ref': [r.counts, r.shift * 1e12, r.dv]})
            entry[name] = [g.counts, g.shift * 1e12, g.dv]
            if r.timing and r.counts and not g.counts:
                findings[f'{name}:crossing-count'].append(
                    f'{name}: mid-rail crossing count differs from tight on {key}, ref agrees')
            elif r.timing and r.counts and g.shift > r.shift + max(measure.slack, self.shift_relative * r.shift):
                findings[f'{name}:crossing-shift'].append(
                    f'{name}: crossing shift {g.shift * 1e12:.2f} ps vs ref {r.shift * 1e12:.2f} ps on {key}')
            elif not (r.timing and r.counts) and g.dv > r.dv + max(self.dv_slack * measure.vdd, r.dv):
                findings[f'{name}:dv'].append(f'{name}: max |dV| {g.dv:.4g} V vs ref {r.dv:.4g} V on {key}')

    @staticmethod
    def _vectors(name: str, output: str, comparison: VectorComparison, reference: Waveform, table: Waveform,
                 vdd: float, context: _Context, metrics: dict[str, Any]) -> list[str]:
        found: list[str] = []
        ref_table = context.reference_table(output)
        if ref_table is None:  # not a comparable output; tables() excludes those
            return found
        worst = {'ref': 0.0, name: 0.0}
        for vector in reference.names:  # ref's and the run's vector sets equal tight's (checked before)
            r, g = comparison.error(reference, ref_table, vector), comparison.error(reference, table, vector)
            worst['ref'], worst[name] = max(worst['ref'], r), max(worst[name], g)
            if g > comparison.allowance(r, vdd):
                found.append(f'{name}: {output} {vector} off by {comparison.describe(g)} vs ref '
                             f'{comparison.describe(r)}')
        stem = output.rsplit('.', 1)[0]
        for who, value in worst.items():
            metrics[f'{who}_{stem}_error'] = max(metrics.get(f'{who}_{stem}_error', 0.0), value)
        return found

    @staticmethod
    def _summary(name: str, nodes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        """Case-level maxima, as in the prototype: crossing shift over the timing nodes where
        the run's count agrees with tight, |dV| over the other nodes."""
        summary: dict[str, Any] = {}
        for who in ('ref', name):
            runs = [(e['timing'], e[who]) for e in nodes.values() if who in e]
            summary[f'{who}_counts'] = all(agree for timing, (agree, _, _) in runs if timing)
            summary[f'{who}_shift_ps'] = max([s for timing, (agree, s, _) in runs if timing and agree], default=0.0)
            summary[f'{who}_dv'] = max([dv for timing, (agree, _, dv) in runs if not (timing and agree)],
                                       default=0.0)
        return summary


@dataclass
class _Context:
    """What every approximate configuration of a case is compared with: tight's tables and
    ref's errors against them, read, checked and measured once."""
    deck: Deck
    tight: RunResult
    ref: RunResult
    measure: Optional[AccuracyMeasure]
    _node_errors: dict[str, Mapping[str, NodeError]] = field(default_factory=dict)

    @functools.cached_property
    def _readings(self) -> dict[str, tuple[Reading, Reading]]:
        return {output: (read_output(self.deck, output, self.tight.outputs.get(output)),
                         read_output(self.deck, output, self.ref.outputs.get(output)))
                for output in self.deck.outputs}

    @functools.cached_property
    def problems(self) -> list[str]:
        """Why outputs cannot be compared: tight's or ref's table unusable, or ref's table
        shaped differently from tight's (the same wrdata command wrote both)."""
        problems = []
        for output, (tight, ref) in self._readings.items():
            if not tight.usable:
                problems.append(f'tight: {output} unusable ({tight.problem})')
            elif not ref.usable:
                problems.append(f'ref: {output} unusable ({ref.problem})')
            elif tight.table is not None and ref.table is not None and tight.table.mismatch(ref.table):
                problems.append(f'ref: {output} {tight.table.mismatch(ref.table)}')
        return problems

    @functools.cached_property
    def tables(self) -> dict[str, Waveform]:
        """tight's tables of the outputs that can be compared."""
        return {output: tight.table for output, (tight, ref) in self._readings.items()
                if tight.usable and ref.usable and tight.table is not None and ref.table is not None
                and tight.table.mismatch(ref.table) is None}

    def reference_table(self, output: str) -> Optional[Waveform]:
        """ref's table of a comparable output."""
        return self._readings[output][1].table if output in self.tables else None

    def reference_node_errors(self, output: str) -> Mapping[str, NodeError]:
        if output not in self._node_errors:
            table = self.reference_table(output)
            self._node_errors[output] = ({} if table is None or self.measure is None
                                         else self.measure.node_errors(self.tables[output], table))
        return self._node_errors[output]


# --- Front-end identity ----------------------------------------------------------------------

DEBUG_FILES = ('debug-out2.txt', 'debug-out3.txt')  # after expansion (before .if), final deck
_MODEL = re.compile(r'^(?P<prefix>.*?\s|)(?P<kind>[.*])model\s+(?P<name>\S+)\s+(?P<type>\S+)', re.I)


@dataclass(frozen=True)
class DumpComparison:
    difference: Optional[str]  # the first difference, None if the dumps agree
    commented: frozenset[str]  # scoped names of the models the candidate commented out


@dataclass(frozen=True)
class FrontEndOracle(Oracle):
    """The candidate's front end (default environment) against ref's, from the decks that
    `ngspice -D ngdebug` dumps after subcircuit expansion and at the end of parsing."""

    def judge(self, evidence: Evidence) -> Judgement:
        front = evidence.front_end
        if front is None:
            return Judgement()
        if front.ref_run is not None and not front.ref_run.ok:
            return Judgement((CheckResult.skipped('front-end:identity'),),
                             inconclusive=(f'front-end: ref parse-only run {front.ref_run.outcome}',))
        ref_dir, candidate_dir = front.ref_dir, front.candidate_dir
        findings: list[str] = []
        if front.candidate_run is not None and not front.candidate_run.ok:
            findings.append(f'front-end: candidate parse-only run {front.candidate_run.outcome}')
        for dump in DEBUG_FILES:
            if not (ref_dir / dump).exists():
                return Judgement((CheckResult.skipped('front-end:identity'),), (f'front-end: ref wrote no {dump}',))
            if not (candidate_dir / dump).exists():
                findings.append(f'front-end: candidate wrote no {dump}')
                continue
            ref_lines = self._read(ref_dir / dump)
            comparison = self.compare(ref_lines, self._read(candidate_dir / dump))
            if comparison.difference:
                findings.append(f'front-end: {dump} differs from ref, {comparison.difference}')
            elif dump == DEBUG_FILES[-1]:
                used = comparison.commented & self.references(ref_lines)
                findings += [f'front-end: referenced model {model} commented out' for model in sorted(used)]
        return Judgement((CheckResult.of('front-end:identity', findings),))

    @staticmethod
    def _read(dump: Path) -> list[str]:
        """The dump's lines, with the run's own directory (where local include files are
        resolved) replaced by a placeholder."""
        return dump.read_text(errors='replace').replace(str(dump.parent), '<run>').splitlines()

    @staticmethod
    def compare(ref_lines: Sequence[str], candidate_lines: Sequence[str]) -> DumpComparison:
        """The candidate may comment out subcircuit-local .model cards ('*model NAME TYPE ...'
        where ref shows '.model SCOPE:NAME TYPE ...', or ref's own '*model SCOPE:NAME ...' in a
        branch .if dropped; the candidate's comment keeps the card's text from before numparam
        substitution), which also drops them from the uncommented listings. Nothing else may
        differ."""
        i = j = 0
        commented: set[str] = set()
        dropped: set[str] = set()
        while i < len(ref_lines) and j < len(candidate_lines):
            r, c = ref_lines[i], candidate_lines[j]
            if r == c:
                i, j = i + 1, j + 1
                continue
            mr, mc = _MODEL.match(r), _MODEL.match(c)
            if (mr and mc and mc['kind'] == '*' and mr['prefix'] == mc['prefix']
                    and mr['type'].lower() == mc['type'].lower()
                    and mr['name'].lower().split(':')[-1] == mc['name'].lower()):
                commented.add(mr['name'].lower())
                i, j = i + 1, j + 1
            elif mr and mr['kind'] == '.':
                dropped.add(mr['name'].lower())
                i += 1
            else:
                return DumpComparison(f'line {j + 1}: ref {r.strip()[:100]!r}, candidate {c.strip()[:100]!r}',
                                      frozenset(commented))
        for r in ref_lines[i:]:
            mr = _MODEL.match(r)
            if not (mr and mr['kind'] == '.'):
                return DumpComparison(f'ref line {i + 1} missing in candidate: {r.strip()[:100]!r}',
                                      frozenset(commented))
            dropped.add(mr['name'].lower())
        if j < len(candidate_lines):
            return DumpComparison(f'extra candidate line {j + 1}: {candidate_lines[j].strip()[:100]!r}',
                                  frozenset(commented))
        unexplained = dropped - commented
        if unexplained:
            return DumpComparison(f'model {sorted(unexplained)[0]} dropped but never shown commented out',
                                  frozenset(commented))
        return DumpComparison(None, frozenset(commented))

    @staticmethod
    def references(final_lines: Iterable[str]) -> set[str]:
        """Lower-case tokens of the non-model cards of a final deck dump (model references)."""
        tokens: set[str] = set()
        for line in final_lines:
            if not _MODEL.match(line):
                tokens.update(re.split(r'[\s=(),]+', line.lower()))
        return tokens


# --- Self-determinism ------------------------------------------------------------------------

@dataclass(frozen=True)
class DeterminismOracle(Oracle):
    """The same configuration run twice gives byte-identical results."""

    def judge(self, evidence: Evidence) -> Judgement:
        checks = []
        for repeat in evidence.repeats:
            first, second = repeat.first, repeat.second
            differ = [what for what, unequal in (('status', first.outcome != second.outcome),
                                                 ('outputs', first.outputs != second.outputs),
                                                 ('diagnostics', first.diagnostics != second.diagnostics))
                      if unequal]
            name = repeat.configuration
            checks.append(CheckResult.of(f'{name}:determinism', [
                f'{name}: not deterministic ({", ".join(differ)} differ between two runs)'] if differ else []))
        return Judgement(tuple(checks))


# --- Verdict ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Verdict:
    """The combined judgement of all oracles on one case."""
    checks: Mapping[str, Outcome]
    findings: tuple[str, ...]
    notes: tuple[str, ...]
    inconclusive: tuple[str, ...]
    metrics: Mapping[str, Any]

    @classmethod
    def combine(cls, judgements: Iterable[Judgement]) -> Verdict:
        judgements = list(judgements)
        metrics: dict[str, Any] = {}
        for judgement in judgements:
            metrics.update(judgement.metrics)
        return cls({c.key: c.outcome for j in judgements for c in j.checks},
                   tuple(f for j in judgements for f in j.findings),
                   tuple(n for j in judgements for n in j.notes),
                   tuple(i for j in judgements for i in j.inconclusive), metrics)

    @property
    def failed(self) -> bool:
        return bool(self.findings)


ORACLES: Sequence[Oracle] = (BitExactOracle(), AccuracyOracle(), FrontEndOracle(),
                             DeterminismOracle())


BUSY_NOTE = 'lock busy, not run'


def judge(evidence: Evidence, oracles: Sequence[Oracle] = ORACLES) -> Verdict:
    """The verdict of all oracles. A run that found a cooperative lock busy never ran: it is a
    note, and no oracle judges it (its configuration's checks are missing, not failed)."""
    busy = tuple(name for name, run in evidence.runs.items() if run.status == BUSY)
    judged = evidence.without(busy) if busy else evidence
    return Verdict.combine([*(oracle.judge(judged) for oracle in oracles),
                            Judgement(notes=tuple(f'{name}: {BUSY_NOTE}' for name in busy))])
