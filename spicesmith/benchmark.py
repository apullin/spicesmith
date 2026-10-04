"""Repeated timings on a frozen, explicitly identified workload population."""
from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import experiment
from .circuit import fingerprint
from .deck import Deck
from .harness import Testbench
from .plans import ExecutionPlan
from .policies import OBSERVE, Policy
from .population import load_frozen, numerical, write_json
from .provenance import host


def timing(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    times = [s['seconds'] for s in samples]
    complete = bool(samples) and all(s.get('complete') for s in samples)
    usable = complete and all(isinstance(t, (int, float)) and math.isfinite(t) and t > 0 for t in times)
    return {'samples': times, 'outcomes': [s.get('outcome') for s in samples],
            'completed': sum(bool(s.get('complete')) for s in samples), 'usable': usable,
            'median': statistics.median(times) if usable else None,
            'minimum': min(times) if usable else None, 'maximum': max(times) if usable else None,
            'stdev': statistics.stdev(times) if usable and len(times) > 1 else None}


def comparison(observed: Mapping[str, Any], baseline: str) -> dict[str, Any]:
    timings = {name: timing(values.get('samples', ())) for name, values in observed.items()}
    ref = timings.get(baseline, {})
    ratios = {}
    for name, t in timings.items():
        if name == baseline:
            continue
        valid = (t['usable'] and ref.get('usable') and len(t['samples']) == len(ref['samples'])
                 and len(t['samples']) >= 3)
        ratios[name] = {'speedup': ref['median'] / t['median'] if valid else None,
                        'reason': None if valid else 'needs >=3 complete, positive-duration matched samples',
                        'accuracy': 'see independent policy assessment; completion alone is not correctness'}
    return {'baseline': baseline, 'timings': timings, 'comparisons': ratios}


def groups(generation: Mapping[str, Any], observed: Mapping[str, Any]) -> dict[str, str]:
    structure, p = generation.get('structure', {}), generation.get('profile', {})
    if not structure:
        return {'profile': p.get('name', 'unknown'), 'composition': 'unclassified', 'transistors': 'unknown',
                'topology_repetition': 'unknown', 'source_correlation': 'unknown', 'coupling': 'unknown',
                'observed_regime': numerical(observed)['regime'], 'observed_signal_correlation': 'unclassified'}
    count = structure.get('transistors', 0)
    size = '<100' if count < 100 else ('100-999' if count < 1000 else ('1k-9999' if count < 10000 else '10k+'))
    motifs = structure.get('motifs', {})
    total = sum(motifs.values())
    digital = bool(motifs.get('logic') or motifs.get('switching'))
    analog = bool(set(motifs) - {'logic'})
    composition = 'mixed' if digital and analog else ('digital' if digital else 'analog')
    ratio = structure.get('distinct_shapes', total) / max(1, total)
    sources = structure.get('distinct_source_draws', total) / max(1, total)
    behavior = numerical(observed)
    correlation = behavior['mean_absolute_correlation']
    return {'profile': p.get('name', 'unknown'), 'composition': composition, 'transistors': size,
            'topology_repetition': 'high' if ratio < 0.5 else 'low',
            'source_correlation': 'high' if sources < 0.5 else 'low',
            'coupling': 'signal-coupled' if structure.get('roles', {}).get('coupling') else 'supply-only',
            'observed_regime': behavior['regime'],
            'observed_signal_correlation': 'unclassified' if correlation is None else
                ('high' if correlation >= 0.8 else 'low')}


def summarize(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {'cases': 0, 'assessments': Counter(),
                                                            'configurations': {}})
    for case in cases:
        for dimension, value in {'population': 'all', **case['groups']}.items():
            entry = buckets[f'{dimension}={value}']
            entry['cases'] += 1
            entry['assessments'][case['assessment']] += 1
            for name, comparison in case['performance']['comparisons'].items():
                c = entry['configurations'].setdefault(name, {'completed_ratios': [], 'unresolved': 0, 'slowdowns': 0})
                ratio = comparison['speedup']
                if ratio is None:
                    c['unresolved'] += 1
                else:
                    c['completed_ratios'].append(ratio)
                    c['slowdowns'] += ratio < 1
    for entry in buckets.values():
        for c in entry['configurations'].values():
            ratios = c['completed_ratios']
            c['geomean_speedup'] = math.exp(statistics.mean(math.log(r) for r in ratios)) if ratios else None
    return dict(buckets)


def run(population: Path, destination: Path, bench: Testbench, plan: ExecutionPlan,
        policy: Policy = OBSERVE, baseline: str = 'ref') -> dict[str, Any]:
    if plan.repetitions < 3 or baseline not in plan.names or len(plan.names) < 2:
        raise ValueError('benchmark needs >=3 repetitions, a baseline and at least one comparison configuration')
    if any(bench.configurations[n].tight for n in plan.names) and plan.tight_level is None:
        raise ValueError('benchmarking tight requires a fixed --tight-level; fallback work is not a matched timing')
    if destination.exists():
        raise ValueError('benchmark destination must not exist; use a fresh directory')
    manifest = load_frozen(population)
    context = experiment.identity(Deck('benchmark context\n'), bench, plan, {})
    definition = {'version': 1, 'population': manifest['identity'], 'execution': context,
                  'policy': policy.to_json(), 'baseline': baseline, 'host': dict(host())}
    report: dict[str, Any] = {'identity': fingerprint(definition), 'definition': definition,
                              'selection': manifest['selection'], 'excluded_at_freeze': manifest['excluded'],
                              'requested_workloads': len(manifest['cases']), 'all_cases_attempted': False,
                              'timing_scope': 'process wall time including startup; alternating configuration order',
                              'cases': []}
    write_json(destination / 'benchmark.json', report)
    for case in manifest['cases']:
        deck = Deck.load(population / case['path'])
        directory = destination / case['path']
        try:
            result = experiment.run(deck, bench, plan, directory, policy, case['generation'])
            observed = json.loads((directory / 'observations.json').read_text())
            state, findings, reasons = result.state, list(result.findings), list(result.reasons)
        except (OSError, ValueError) as error:
            observed = {name: {'samples': []} for name in plan.names}
            state, findings, reasons = 'inconclusive', [], [str(error)]
        report['cases'].append({'path': case['path'], 'deck_sha256': case['deck_sha256'],
                                'assessment': state, 'policy': policy.to_json(),
                                'findings': findings, 'reasons': reasons,
                                'performance': comparison(observed, baseline),
                                'groups': groups(case['generation'], observed)})
        report['groups'] = summarize(report['cases'])
        write_json(destination / 'benchmark.json', report)
    report['all_cases_attempted'] = len(report['cases']) == len(manifest['cases'])
    report['completed_workloads'] = sum(all(t['completed'] == plan.repetitions
                                           for t in c['performance']['timings'].values()) for c in report['cases'])
    write_json(destination / 'benchmark.json', report)
    return report
