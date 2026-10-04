"""Deterministic stratified discovery, numerical coverage and frozen populations.

Every requested case remains in the denominator, including rejected generation and
interrupted/incomplete execution. Guidance is deliberately a small deficit scheduler.
"""
from __future__ import annotations

import dataclasses
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import experiment, profiles
from .circuit import Circuit, fingerprint
from .deck import Deck
from .harness import Testbench
from .plans import ExecutionPlan
from .policies import OBSERVE, Policy
from .provenance import code_sha256

COUNTERS = ('equations', 'nonzeros', 'fillin', 'iterations', 'tran_iterations', 'accepted', 'rejected', 'timepoints')
REGIMES = ('generated', 'completed', 'active', 'quiescent', 'unclassified')


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def numerical(observations: Mapping[str, Any], amplitude_floor: float = 1e-6) -> dict[str, Any]:
    """Coarse observed descriptors, not an oscillator/amplifier classifier.

    Only saved/observed voltages count; internal device regions remain unclassified.
    All configurations and repetitions must complete for the case to be 'completed'.
    """
    samples = [s for c in observations.values() for s in c.get('samples', ())]
    available = {k: sum(s.get('counters', {}).get(k) is not None for s in samples) for k in COUNTERS}
    signals = [signal for s in samples for artifact in s.get('artifacts', {}).values()
               if artifact.get('scale') == 'time'
               for signal in artifact.get('signals', {}).values()]
    spans = [v for s in signals if math.isfinite(v := s['maximum'] - s['minimum'])]
    correlations = [pair['pearson'] for s in samples for a in s.get('artifacts', {}).values()
                    for pair in a.get('activity_correlation', {}).get('pairs', []) if pair['pearson'] is not None]
    complete = bool(samples) and all(s.get('complete', False) for s in samples)
    regime = 'incomplete' if not complete else ('unclassified' if not spans else
                                              ('active' if max(spans) > amplitude_floor else 'quiescent'))
    steps = [a['steps'] for s in samples for a in s.get('artifacts', {}).values() if 'steps' in a]
    return {'classifier': {'version': 1, 'amplitude_floor_volts': amplitude_floor,
                           'scope': 'observed voltages only; not a functional or accuracy classifier'},
            'regime': regime, 'completed': complete, 'samples': len(samples),
            'counter_availability': available, 'observed_signal_traces': len(signals),
            'maximum_observed_span': max(spans) if spans else None,
            'observed_correlation_pairs': len(correlations),
            'mean_absolute_correlation': (sum(abs(c) for c in correlations) / len(correlations)
                                          if correlations else None),
            'minimum_step': min(s['minimum'] for s in steps) if steps else None,
            'maximum_step': max(s['maximum'] for s in steps) if steps else None,
            'device_regions': 'unclassified', 'oscillation_or_settling': 'unclassified'}


@dataclass(frozen=True)
class Stratum:
    name: str
    profile: profiles.Profile
    quota: int = 1
    target: str = 'generated'

    def __post_init__(self) -> None:
        if not self.name or type(self.quota) is not int or self.quota < 1 or self.target not in REGIMES:
            raise ValueError('a stratum needs a name, positive quota and supported coverage target')

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Stratum:
        p = data['profile']
        recipe = profiles.profile(p) if isinstance(p, str) else profiles.profile(p['name'], p)
        return cls(data['name'], recipe, data.get('quota', 1), data.get('target', 'generated'))


def matches(record: Mapping[str, Any], target: str) -> bool:
    if target == 'generated':
        return 'structure' in record
    observed = record.get('numerical', {})
    return bool(observed.get('completed')) if target == 'completed' else observed.get('regime') == target


@dataclass(frozen=True)
class Campaign:
    directory: Path
    strata: tuple[Stratum, ...]
    seed: int = 1
    budget: int = 20
    feedback: bool = False
    bench: Testbench | None = None
    plan: ExecutionPlan = ExecutionPlan()
    policy: Policy = OBSERVE

    def __post_init__(self) -> None:
        if not self.strata or len({s.name for s in self.strata}) != len(self.strata):
            raise ValueError('strata must have distinct names')
        if self.budget < 1:
            raise ValueError('campaign budget must be positive')
        if self.bench is None and any(s.target != 'generated' for s in self.strata):
            raise ValueError('numerical coverage targets need --execute')

    def definition(self) -> dict[str, Any]:
        execution = experiment.identity(Deck('campaign identity\n'), self.bench, self.plan, {}) if self.bench else None
        return {'version': 1, 'strata': [s.to_json() for s in self.strata], 'seed': self.seed,
                'budget': self.budget, 'feedback': self.feedback, 'execution': execution,
                'policy': self.policy.to_json(), 'code': code_sha256()}

    def select(self, records: Sequence[Mapping[str, Any]]) -> tuple[Stratum, dict[str, Any]] | None:
        counts = {s.name: sum(r['stratum'] == s.name for r in records) for s in self.strata}
        reached = {s.name: sum(r['stratum'] == s.name and matches(r, s.target) for r in records) for s in self.strata}
        # First provide each stratum its requested attempt quota; failures never erase those attempts.
        pending = [s for s in self.strata if counts[s.name] < s.quota]
        if pending:
            selected = min(pending, key=lambda s: (counts[s.name] / s.quota, counts[s.name], s.name))
            reason = 'stratified-quota'
        elif self.feedback:
            pending = [s for s in self.strata if reached[s.name] < s.quota]
            if not pending:
                return None
            selected = min(pending, key=lambda s: (reached[s.name] / s.quota, counts[s.name], s.name))
            reason = 'observed-deficit'
        else:
            return None
        return selected, {'reason': reason, 'issued': counts, 'reached': reached,
                          'target': selected.target, 'scheduler': 'deficit-1'}

    def run(self, resume: bool = False, limit: int | None = None) -> dict[str, Any]:
        path = self.directory / 'campaign.json'
        definition = self.definition()
        if path.exists():
            state = json.loads(path.read_text())
            if not resume:
                raise ValueError('campaign already exists; request resume or use a fresh directory')
            if state['identity'] != fingerprint(definition):
                raise ValueError('incompatible campaign identity (profiles, code, execution or policy changed)')
            # Interrupted cases stay on disk and in the denominator; resumption never overwrites their evidence.
            for unfinished in state['cases']:
                if unfinished['state'] == 'pending':
                    unfinished.update(state='interrupted', reason='interrupted before case finalization')
        else:
            if resume:
                raise ValueError('no campaign to resume')
            state = {'identity': fingerprint(definition), 'definition': definition, 'cases': []}
        records = state['cases']
        count = 0
        while len(records) < self.budget and (limit is None or count < limit):
            choice = self.select(records)
            if choice is None:
                break
            stratum, decision = choice
            index = len(records)
            seed = int(fingerprint([self.seed, index, 'case-seed-1'])[:16], 16)
            directory = self.directory / f'case-{index:06d}'
            if directory.exists():
                raise ValueError(f'{directory}: unrecorded directory; refusing to overwrite')
            record: dict[str, Any] = {'index': index, 'seed': seed, 'stratum': stratum.name,
                                      'path': directory.name, 'decision': decision, 'state': 'pending'}
            records.append(record)
            write_json(path, state)
            try:
                generated = profiles.generate(seed, stratum.profile)
                circuit, meta = generated.circuit, generated.metadata()
                deck = circuit.lower()
                deck.save(directory)
                write_json(directory / 'deck.circuit.json', circuit.to_json())
                write_json(directory / 'deck.generation.json', meta)
                record.update(state='generated', structure=meta['structure'], attempts=list(generated.attempts))
                if self.bench:
                    result = experiment.run(deck, self.bench, self.plan, directory, self.policy, meta)
                    observed = json.loads((directory / 'observations.json').read_text())
                    record.update(state='observed', assessment=result.state,
                                  numerical=numerical(observed), selected=list(result.selected),
                                  findings=list(result.findings))
            except profiles.GenerationError as error:
                record.update(state='rejected', attempts=error.history, reason=str(error))
            except (OSError, ValueError) as error:
                record.update(state='unresolved', reason=str(error))
            except BaseException:
                record.update(state='interrupted', reason='interrupted during case execution')
                raise
            finally:
                write_json(directory / 'case.json', record)
                write_json(path, state)
                write_json(self.directory / 'coverage.json', coverage(self.strata, records))
            count += 1
        write_json(path, state)
        report = coverage(self.strata, records)
        write_json(self.directory / 'coverage.json', report)
        return report


def coverage(strata: Sequence[Stratum], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {'version': 1, 'requested_cases': sum(s.quota for s in strata),
                              'issued_cases': len(records), 'states': dict(Counter(r['state'] for r in records)),
                              'strata': {}}
    for s in strata:
        cases = [r for r in records if r['stratum'] == s.name]
        reached = sum(matches(r, s.target) for r in cases)
        result['strata'][s.name] = {
            'quota': s.quota, 'target': s.target, 'issued': len(cases), 'reached': reached,
            'remaining': max(0, s.quota - reached), 'states': dict(Counter(r['state'] for r in cases)),
            'generation_rejections': sum(a['state'] == 'rejected' for r in cases for a in r.get('attempts', [])),
            'observed_regimes': dict(Counter(r.get('numerical', {}).get('regime', 'unclassified') for r in cases)),
            'completed': sum(r.get('numerical', {}).get('completed', False) for r in cases),
            'assessment_states': dict(Counter(r.get('assessment', 'unassessed') for r in cases)),
            'structures': [r['structure'] for r in cases if 'structure' in r],
            'metric_availability': {k: sum(r.get('numerical', {}).get('counter_availability', {}).get(k, 0)
                                           for r in cases) for k in COUNTERS}}
    return result


def deck_identity(deck: Deck) -> str:
    return fingerprint({'text': deck.text, 'files': dict(deck.files)})


def freeze(campaign: Path, destination: Path) -> dict[str, Any]:
    """Copy all generated cases, even failed simulations, and disclose excluded rejections.

    Frozen discovery populations are still discovery-selected, not representative samples.
    """
    if destination.exists():
        raise ValueError('freeze destination must not exist')
    state = json.loads((campaign / 'campaign.json').read_text())
    records, excluded = [], []
    for case in state['cases']:
        source = campaign / case['path']
        if not (source / 'deck.sp').exists():
            excluded.append({'index': case['index'], 'state': case['state'], 'reason': 'no generated deck'})
            continue
        deck = Deck.load(source)
        if (source / 'evidence.json').exists():
            experiment.verify(source)
        meta_path = source / 'deck.generation.json'
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        graph = source / 'deck.circuit.json'
        if meta.get('circuit_sha256'):
            if not graph.exists():
                raise ValueError(f'{source}: recorded circuit graph is missing')
            circuit = Circuit.from_json(json.loads(graph.read_text()))
            if (fingerprint(circuit.to_json()) != meta['circuit_sha256']
                    or deck_identity(circuit.lower()) != deck_identity(deck)):
                raise ValueError(f'{source}: generated workload or graph changed before freezing')
        path = f'case-{case["index"]:06d}'
        deck.save(destination / path)
        write_json(destination / path / 'deck.generation.json', meta)
        if (source / 'deck.circuit.json').exists():
            write_json(destination / path / 'deck.circuit.json', json.loads((source / 'deck.circuit.json').read_text()))
        records.append({'path': path, 'deck_sha256': deck_identity(deck), 'generation': meta,
                        'source_state': case['state'], 'source_index': case['index']})
    if not records:
        raise ValueError('no generated workloads to freeze')
    payload = {'version': 1, 'origin': str(campaign.resolve()), 'source_identity': state['identity'],
               'selection': 'all generated cases; discovery-selected, not representative of general use',
               'source_definition': state['definition'], 'cases': records, 'excluded': excluded}
    manifest = {**payload, 'identity': fingerprint(payload)}
    write_json(destination / 'population.json', manifest)
    return manifest


def load_frozen(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / 'population.json').read_text())
    if manifest.get('version') != 1 or manifest['identity'] != fingerprint({k: v for k, v in manifest.items()
                                                                         if k != 'identity'}):
        raise ValueError('frozen population manifest changed')
    for case in manifest['cases']:
        path = Path(case['path'])
        if path.is_absolute() or len(path.parts) != 1 or path.name in ('.', '..'):
            raise ValueError('invalid frozen case path')
        if deck_identity(Deck.load(directory / path)) != case['deck_sha256']:
            raise ValueError(f'{path}: frozen workload changed')
        if json.loads((directory / path / 'deck.generation.json').read_text()) != case['generation']:
            raise ValueError(f'{path}: frozen generation metadata changed')
    return manifest
