"""Aggregating batches: per-check pass rates, accuracy distributions of the
approximate configurations against their base configuration (ref, or the configuration they add
one approximation to; both measured against tight), overall and per circuit class, the error
budget of the approximations, clusters of findings and notes, and the health of the generator.
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import CONFIGURATIONS
from .harness import CaseSummary
from .oracles import AccuracyOracle, SmallSignalComparison, StaticComparison

TIE = 0.01  # ps or mV: differences below this count as equal


def base_of(configuration: str) -> str:
    """The configuration whose accuracy `configuration`'s own change is measured against."""
    known = CONFIGURATIONS.get(configuration)
    return known.base if known else 'ref'


@dataclass(frozen=True)
class CheckRate:
    key: str
    passed: int = 0
    failed: int = 0
    skipped: int = 0

    @property
    def ran(self) -> int:
        return self.passed + self.failed


@dataclass(frozen=True)
class Distribution:
    """Summary statistics of a sample."""
    values: tuple[float, ...]

    @property
    def median(self) -> float:
        return statistics.median(self.values) if self.values else float('nan')

    @property
    def p90(self) -> float:
        if not self.values:
            return float('nan')
        ordered = sorted(self.values)
        return ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]

    @property
    def maximum(self) -> float:
        return max(self.values, default=float('nan'))

    @property
    def minimum(self) -> float:
        return min(self.values, default=float('nan'))


@dataclass(frozen=True)
class Comparison:
    """An approximate configuration against its base on one measure, one value pair per case (or
    per case and circuit class): which of the two is closer to tight."""
    configuration: str
    measure: str  # 'crossing shift (ps)' or 'max dV (mV)'
    scope: str  # 'all' or a circuit class
    pairs: tuple[tuple[float, float], ...]  # (base, configuration)
    base: str = 'ref'
    seeds: tuple[int, ...] = ()  # the case of each pair

    def worst(self, count: int = 3) -> list[tuple[int, float]]:
        """The seeds with the largest change in error (positive: worse than the base)."""
        ranked = sorted(zip(self.seeds, (c - r for r, c in self.pairs), strict=False), key=lambda p: -p[1])
        return [(seed, d) for seed, d in ranked[:count] if d > TIE]

    @property
    def better(self) -> int:
        return sum(c < r - TIE for r, c in self.pairs)

    @property
    def worse(self) -> int:
        return sum(c > r + TIE for r, c in self.pairs)

    @property
    def reference(self) -> Distribution:
        return Distribution(tuple(r for r, _ in self.pairs))

    @property
    def candidate(self) -> Distribution:
        return Distribution(tuple(c for _, c in self.pairs))

    @property
    def delta(self) -> Distribution:
        """The change in error the configuration's own approximation makes, signed: positive is
        further from tight than the base."""
        return Distribution(tuple(c - r for r, c in self.pairs))


@dataclass(frozen=True)
class Ablation:
    """One approximate configuration against its base: where it finishes, its findings (the
    oracles judge against ref), and the cases in which it is beyond the oracles' allowance when
    judged against its base instead (the same rules, the base in ref's place)."""
    configuration: str
    base: str
    base_finished: int  # cases the base finished
    both_finished: int  # ... and the configuration as well
    findings: tuple[int, ...]  # seeds
    beyond_base: tuple[int, ...]  # seeds


@dataclass(frozen=True)
class Cluster:
    pattern: str
    seeds: tuple[int, ...]


def pattern_of(message: str) -> str:
    """A finding with its specifics (numbers, node names) abstracted, to group alike ones."""
    message = re.sub(r'on v\([^)]*\)', 'on <node>', message)
    message = re.sub(r'node\(s\): .*', 'node(s): ...', message)
    return re.sub(r'[-+]?\d+(\.\d+)?(e[-+]?\d+)?', '#', message)


@dataclass(frozen=True)
class Report:
    summaries: tuple[CaseSummary, ...]
    sources: tuple[str, ...] = ()
    harness_errors: int = 0

    @classmethod
    def load(cls, directories: Iterable[Path]) -> Report:
        summaries, sources, errors = [], [], 0
        for directory in directories:
            sources.append(str(directory))
            errors += len(list(directory.glob('case-*/error.txt')))
            for path in sorted(directory.glob('case-*/summary.json')):
                try:
                    summaries.append(CaseSummary.from_json(json.loads(path.read_text())))
                except (ValueError, KeyError):
                    errors += 1
        return cls(tuple(summaries), tuple(sources), errors)

    # --- Counts ------------------------------------------------------------------------------

    def check_rates(self) -> list[CheckRate]:
        counts: dict[str, Counter] = defaultdict(Counter)
        for summary in self.summaries:
            for key, outcome in summary.checks.items():
                counts[key][outcome] += 1
        return [CheckRate(key, c['pass'], c['fail'], c['skip']) for key, c in sorted(counts.items())]

    def clusters(self, what: str = 'findings') -> list[Cluster]:
        seeds: dict[str, list[int]] = defaultdict(list)
        for summary in self.summaries:
            for pattern in {pattern_of(m) for m in getattr(summary, what)}:
                seeds[pattern].append(summary.seed)
        return sorted((Cluster(p, tuple(s)) for p, s in seeds.items()), key=lambda c: -len(c.seeds))

    def configurations(self) -> list[str]:
        """Approximate configurations with accuracy metrics in these summaries, in the order of
        CONFIGURATIONS."""
        names = {key[:-len('_shift_ps')] for s in self.summaries for key in s.metrics if key.endswith('_shift_ps')}
        order = list(CONFIGURATIONS)
        return sorted(names - {'ref'}, key=lambda n: (order.index(n) if n in order else len(order), n))

    def base_of(self, name: str) -> str:
        bases = {s.provenance.get('bases', {}).get(name, base_of(name)) for s in self.summaries}
        if len(bases) > 1:
            raise ValueError(f'incompatible reporting bases for {name}: {sorted(bases)}')
        return next(iter(bases), base_of(name))

    # --- Accuracy ----------------------------------------------------------------------------

    def comparisons(self, configuration: str) -> list[Comparison]:
        """Crossing shift and |dV| of `configuration` against its base's, per case, overall and
        per circuit class. Shifts count over timing nodes where both agree with tight's count,
        |dV| over the other nodes."""
        base = self.base_of(configuration)
        shifts: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
        deviations: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
        for summary in self.summaries:
            nodes = summary.metrics.get('nodes', {})
            classes = self._classes_of_nodes(summary)
            per_scope: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: {'s': [], 'v': []})
            for node, entry in nodes.items():
                if configuration not in entry or base not in entry:
                    continue
                (r_agree, r_shift, r_dv), (c_agree, c_shift, c_dv) = entry[base], entry[configuration]
                for scope in ('all', *classes.get(node.split('@')[0], ())):  # 'v(n)@file': other outputs
                    # as the oracle: shifts where both agree with tight's count, |dV| wherever
                    # crossings do not apply (analog nodes, or ref's count disagrees)
                    if entry['timing'] and r_agree and c_agree:
                        per_scope[scope]['s'].append((r_shift, c_shift))
                    elif not (entry['timing'] and r_agree):
                        per_scope[scope]['v'].append((r_dv * 1e3, c_dv * 1e3))
            for scope, measures in per_scope.items():
                for kind, sink in (('s', shifts), ('v', deviations)):
                    if measures[kind]:
                        sink[scope].append((summary.seed, max(r for r, _ in measures[kind]),
                                            max(c for _, c in measures[kind])))
        scopes = sorted(set(shifts) | set(deviations), key=lambda s: (s != 'all', s))
        comparisons = []
        for scope in scopes:
            for measure, rows in (('crossing shift (ps)', shifts.get(scope)), ('max dV (mV)', deviations.get(scope))):
                if rows:
                    comparisons.append(Comparison(configuration, measure, scope,
                                                  tuple((r, c) for _, r, c in rows), base,
                                                  tuple(seed for seed, _, _ in rows)))
        return comparisons

    def ablation(self, configuration: str) -> Ablation:
        """`configuration` against its base: finishing, findings, and the oracles' allowance."""
        base, oracle = self.base_of(configuration), AccuracyOracle()
        base_finished = both_finished = 0
        findings, beyond = [], []
        for summary in self.summaries:
            if configuration not in summary.status or base not in summary.status:
                continue
            base_ok, ok = summary.status[base] == 0, summary.status[configuration] == 0
            base_finished += base_ok
            both_finished += base_ok and ok
            if any(f.startswith(f'{configuration}:') for f in summary.findings):
                findings.append(summary.seed)
            if (base_ok and not ok) or (base_ok and self._beyond(summary, configuration, base, oracle)):
                beyond.append(summary.seed)
        return Ablation(configuration, base, base_finished, both_finished, tuple(findings), tuple(beyond))

    @staticmethod
    def _beyond(summary: CaseSummary, configuration: str, base: str, oracle: AccuracyOracle) -> bool:
        """AccuracyOracle's rules with `base` in ref's place, from the stored per-node errors (and
        each output's largest vector error for operating points, DC and AC sweeps)."""
        vdd, tstop = summary.deck.get('vdd'), summary.deck.get('tstop')
        if vdd is None or tstop is None:
            return False
        slack_ps = max(oracle.shift_slack[0], oracle.shift_slack[1] * tstop) * 1e12
        for entry in summary.metrics.get('nodes', {}).values():
            if base not in entry or configuration not in entry:
                continue
            (b_counts, b_shift, b_dv), (c_counts, c_shift, c_dv) = entry[base], entry[configuration]
            timed = entry['timing'] and b_counts
            if timed and (not c_counts or c_shift > b_shift + max(slack_ps, oracle.shift_relative * b_shift)):
                return True
            if not timed and c_dv > b_dv + max(oracle.dv_slack * vdd, b_dv):
                return True
        for comparison, stems in ((StaticComparison(), ('op', 'dc')), (SmallSignalComparison(), ('ac',))):
            for stem in stems:
                b, c = summary.metrics.get(f'{base}_{stem}_error'), summary.metrics.get(f'{configuration}_{stem}_error')
                if b is not None and c is not None and c > comparison.allowance(b, vdd):
                    return True
        return False

    @staticmethod
    def _classes_of_nodes(summary: CaseSummary) -> dict[str, tuple[str, ...]]:
        """'v(net)' -> the circuit classes (and the block kind) of the block owning the net."""
        mapping = {}
        for block in summary.deck.get('blocks', ()):
            labels = tuple(sorted(block.get('classes', ()))) + (f"kind:{block['kind']}",)
            for net in block.get('nets', ()):
                mapping[f'v({net})'] = labels
        return mapping

    # --- Health and cost ---------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        cases = len(self.summaries)
        statuses: dict[str, Counter] = defaultdict(Counter)
        for summary in self.summaries:
            for name, status in summary.status.items():
                statuses[name][str(status)] += 1
        return {
            'cases': cases, 'failing': sum(bool(s.findings) for s in self.summaries),
            'noted': sum(bool(s.notes) for s in self.summaries),
            'inconclusive': sum(bool(s.inconclusive) for s in self.summaries),
            'ref_failed': sum(s.status.get('ref') != 0 for s in self.summaries),
            'tight_levels': dict(sorted(Counter(s.tight_level or 0 for s in self.summaries).items())),
            'statuses': {name: dict(c) for name, c in sorted(statuses.items())},
            'harness_errors': self.harness_errors,
            'generators': sorted({s.deck.get('generator') for s in self.summaries if s.deck.get('generator')}),
            'candidates': sorted({s.provenance.get('candidate', {}).get('sha256', '?')[:12]
                                  for s in self.summaries}),
            'identities': dict(Counter(str(s.provenance.get('identity')) for s in self.summaries)),
        }

    def costs(self) -> dict[str, Any]:
        seconds: dict[str, list[float]] = defaultdict(list)
        for summary in self.summaries:
            for name, value in summary.seconds.items():
                seconds[name].append(value)
        ratios = {}
        for name in self.configurations():
            base = self.base_of(name)
            values = [s.metrics[f'{name}_tran_iterations'] / s.metrics[f'{base}_tran_iterations']
                      for s in self.summaries
                      if s.metrics.get(f'{base}_tran_iterations') and s.metrics.get(f'{name}_tran_iterations')]
            if values:
                ratios[name] = Distribution(tuple(values))
        return {'seconds': {k: Distribution(tuple(v)) for k, v in sorted(seconds.items())},
                'iteration_ratios': ratios}

    # --- Rendering ---------------------------------------------------------------------------

    def markdown(self, clusters_shown: int = 20, seeds_shown: int = 8) -> str:
        health = self.health()
        out = ['# SpiceSmith report', '', f"Batches: {', '.join(self.sources) or '-'}",
               f"Candidates (sha256): {', '.join(health['candidates'])}; generator versions: "
               f"{', '.join(map(str, health['generators']))}", '',
               '## Cases', '',
               f"{health['cases']} cases: {health['failing']} failing, {health['noted']} with notes, "
               f"{health['inconclusive']} inconclusive, ref failed in {health['ref_failed']}, "
               f"{health['harness_errors']} harness errors.",
               f"Accuracy reference level (cases): {health['tight_levels']}.", *self._mixture(health), '',
               '| configuration | outcomes |', '|---|---|']
        out += [f'| {name} | {", ".join(f"{k}: {v}" for k, v in sorted(c.items()))} |'
                for name, c in health['statuses'].items()]
        out += ['', '## Checks', '', '| check | passed | failed | skipped |', '|---|---:|---:|---:|']
        out += [f'| {r.key} | {r.passed} | {r.failed} | {r.skipped} |' for r in self.check_rates()]
        for what, title in (('findings', 'Findings'), ('notes', 'Notes'), ('inconclusive', 'Inconclusive')):
            clusters = self.clusters(what)
            if not clusters:
                continue
            out += ['', f'## {title}', '', '| cases | pattern | seeds |', '|---:|---|---|']
            for cluster in clusters[:clusters_shown]:
                more = ' ...' if len(cluster.seeds) > seeds_shown else ''
                seeds = ', '.join(map(str, cluster.seeds[:seeds_shown])) + more
                out.append(f'| {len(cluster.seeds)} | {cluster.pattern} | {seeds} |')
        out += self._budget()
        for configuration in self.configurations():
            base = self.base_of(configuration)
            out += ['', f'## Accuracy: {configuration} vs {base}, both against tight', '',
                    f'Per case (and circuit class): better/worse = closer to/further from tight than {base}; '
                    f'delta = {configuration} - {base}.', '',
                    f'| scope | measure | cases | better | worse | {base} median | median | {base} p90 | p90 '
                    f'| {base} max | max | delta median | delta p90 | delta max |',
                    '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
            for c in self.comparisons(configuration):
                r, m, d = c.reference, c.candidate, c.delta
                out.append(f'| {c.scope} | {c.measure} | {len(c.pairs)} | {c.better} | {c.worse} | {r.median:.3g} | '
                           f'{m.median:.3g} | {r.p90:.3g} | {m.p90:.3g} | {r.maximum:.3g} | {m.maximum:.3g} | '
                           f'{d.median:.3g} | {d.p90:.3g} | {d.maximum:.3g} |')
        costs = self.costs()
        out += ['', '## Cost', '', '| configuration | median s | p90 s | max s |', '|---|---:|---:|---:|']
        out += [f'| {k} | {d.median:.2f} | {d.p90:.2f} | {d.maximum:.2f} |' for k, d in costs['seconds'].items()]
        for name, d in costs['iteration_ratios'].items():
            out.append(f'\nTransient iterations {name}/{self.base_of(name)}: median {d.median:.3f}, p90 {d.p90:.3f}, '
                       f'max {d.maximum:.3f}.')
        return '\n'.join(out) + '\n'

    @staticmethod
    def _mixture(health: dict[str, Any]) -> list[str]:
        """A warning when the cases come from batches of different identities."""
        identities = health['identities']
        if len(identities) <= 1:
            return []
        counts = ', '.join(f'{identity}: {cases}' for identity, cases in sorted(identities.items()))
        return ['', f'**Mixed identities** (different binaries, inputs, code, configurations or settings; '
                    f'cases per identity): {counts}. Compare like with like.']

    def _budget(self) -> list[str]:
        """The error budget: one row per approximate configuration, against its base."""
        names = self.configurations()
        if not names:
            return []
        kinds = {s.seed: sorted({b['kind'] for b in s.deck.get('blocks', ()) if b.get('kind') != 'supply'})
                 for s in self.summaries}
        out = ['', '## Error budget: each configuration against its base', '',
               'finishes: cases both finish / cases the base finishes. findings: cases with a finding of the '
               'configuration (judged against ref). beyond base: cases outside the oracles\' allowance when '
               'judged against the base (from the stored per-node errors). Per measure: cases, worse/better than '
               'the base, and the signed per-case change in error (configuration - base) median / p90 / max.', '',
               '| configuration | base | finishes | findings | beyond base | shift cases | worse/better '
               '| shift delta ps med / p90 / max | dV cases | worse/better | dV delta mV med / p90 / max |',
               '|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|']
        worst: list[str] = []
        for name in names:
            a = self.ablation(name)
            row = [name, a.base, f'{a.both_finished}/{a.base_finished}', str(len(a.findings)), str(len(a.beyond_base))]
            by_measure = {c.measure: c for c in self.comparisons(name) if c.scope == 'all'}
            for measure in ('crossing shift (ps)', 'max dV (mV)'):
                c = by_measure.get(measure)
                if c is None:
                    row += ['0', '-', '-']
                    continue
                d = c.delta
                row += [str(len(c.pairs)), f'{c.worse}/{c.better}', f'{d.median:.3g} / {d.p90:.3g} / {d.maximum:.3g}']
                for seed, delta in c.worst():
                    worst.append(f'| {name} | {measure} | {seed} | {delta:+.3g} | {", ".join(kinds.get(seed, ()))} |')
            out.append('| ' + ' | '.join(row) + ' |')
        if worst:
            out += ['', 'Worst cases (largest increase in error against the base):', '',
                    '| configuration | measure | seed | delta | block kinds |', '|---|---|---:|---:|---|', *worst]
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            'sources': list(self.sources), 'health': self.health(),
            'checks': [vars(r) for r in self.check_rates()],
            'clusters': {what: [{'pattern': c.pattern, 'seeds': list(c.seeds)} for c in self.clusters(what)]
                         for what in ('findings', 'notes', 'inconclusive')},
            'accuracy': {name: [{'base': c.base, 'measure': c.measure, 'scope': c.scope, 'cases': len(c.pairs),
                                 'better': c.better, 'worse': c.worse, 'ref_median': c.reference.median,
                                 'median': c.candidate.median, 'ref_max': c.reference.maximum,
                                 'max': c.candidate.maximum, 'delta_median': c.delta.median,
                                 'delta_p90': c.delta.p90, 'delta_max': c.delta.maximum,
                                 'worst': c.worst()} for c in self.comparisons(name)]
                         for name in self.configurations()},
            'budget': {name: vars(self.ablation(name)) for name in self.configurations()},
        }
