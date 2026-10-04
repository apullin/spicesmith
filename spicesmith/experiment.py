"""Persist observations separately from policy assessments, and evaluate them later."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .deck import Deck
from .config import Simulators
from .harness import Settings, Testbench, load_evidence
from .host import check_cpus
from .plans import ExecutionPlan, definitions_from_json
from .policies import OBSERVE, Assessment, Policy, assess, observations
from .provenance import BinaryRecord, code_sha256, frozen_inputs, host, sha256, tool


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def identity(deck: Deck, bench: Testbench, plan: ExecutionPlan,
             metadata: Mapping[str, Any]) -> dict[str, Any]:
    paths = {bench.settings.simulators.path(bench.configurations[name].binary).resolve() for name in plan.names}
    return {'deck': digest({'text': deck.text, 'files': dict(deck.files)}),
            'plan': plan.to_json(bench.configurations),
            'binaries': {str(p): BinaryRecord.of(p).sha256 for p in sorted(paths)}, 'inputs': dict(frozen_inputs(deck)),
            'timeout': bench.settings.timeout, 'lock_wait': bench.settings.lock_wait,
            'threads': bench.settings.threads,
            'simulators': {k: str(getattr(bench.settings.simulators, k)) for k in ('reference', 'candidate')},
            'cpus': bench.cpus, 'code': code_sha256(), 'generation': dict(metadata)}


def run(deck: Deck, bench: Testbench, plan: ExecutionPlan, directory: Path,
        policy: Policy = OBSERVE, metadata: Mapping[str, Any] | None = None) -> Assessment:
    if (directory / 'experiment.json').exists():
        raise ValueError(f'{directory}: experiment already exists; use a fresh directory or evaluate saved evidence')
    execution = identity(deck, bench, plan, metadata or {})
    directory.mkdir(parents=True, exist_ok=True)
    record = {'version': 1, 'identity': digest(execution), 'execution': execution,
              'host': dict(host()), 'tool': dict(tool())}
    (directory / 'experiment.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    evidence = bench.run(deck, directory, plan=plan)
    (directory / 'observations.json').write_text(json.dumps(observations(evidence), indent=2, allow_nan=False) + '\n')
    files = {str(p.relative_to(directory)): sha256(p) for p in directory.rglob('*') if p.is_file()}
    (directory / 'evidence.json').write_text(json.dumps({'version': 1, 'files': files}, indent=2) + '\n')
    result = assess(evidence, policy)
    save_assessment(directory, result)
    return result


def save_assessment(directory: Path, result: Assessment) -> Path:
    path = directory / 'assessments' / (digest(result.policy.to_json())[:16] + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_json(), indent=2, allow_nan=False) + '\n')
    return path


def evaluate(directory: Path, policy: Policy) -> Assessment:
    verify(directory)
    result = assess(load_evidence(directory), policy)
    save_assessment(directory, result)
    return result


def verify(directory: Path) -> None:
    """Fail closed when saved experiment artifacts have been changed or pruned."""
    if not (directory / 'evidence.json').exists():
        raise ValueError('no sealed experiment evidence; observe a deck before evaluating it')
    manifest = json.loads((directory / 'evidence.json').read_text())
    if manifest.get('version') != 1:
        raise ValueError('unsupported evidence manifest')
    for name, expected in manifest['files'].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f'{name}: saved evidence changed or is missing')


def replay(source: Path, destination: Path, policy: Policy = OBSERVE) -> Assessment:
    verify(source)
    record = json.loads((source / 'experiment.json').read_text())
    execution = record['execution']
    settings = Settings(Simulators(**{k: Path(v) for k, v in execution['simulators'].items()}),
                        definitions_from_json(execution['plan']), execution['lock_wait'], execution['timeout'],
                        threads=execution.get('threads', 1))
    if execution['cpus']:
        check_cpus(execution['cpus'])
    plan = ExecutionPlan.from_json(execution['plan'])
    bench = Testbench(settings, execution['cpus'])
    deck = Deck.load(source)
    if digest(identity(deck, bench, plan, execution['generation'])) != record['identity']:
        raise ValueError('replay inputs, execution settings or tool code changed')
    return run(deck, bench, plan, destination, policy, execution['generation'])
