"""Execute a reduced bundle through the same runner that discovered its finding."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from . import oracles
from .config import Simulators
from .deck import Deck
from .harness import Settings, Testbench
from .host import check_cpus
from .plans import ExecutionPlan, definitions_from_json
from .provenance import sha256


def replay(directory: Path) -> int:
    directory = directory.resolve()
    record = json.loads((directory / 'repro.json').read_text())
    if record.get('version') != 1:
        raise ValueError('unsupported reproducer version')
    for name, digest in record['files'].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory):
            raise ValueError(f'invalid bundle path: {name}')
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'{name} changed or is missing')
    for path_text, digest in {**record['binaries'], **record['inputs']}.items():
        path = Path(path_text)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'{path} changed or is missing')
    plan = ExecutionPlan.from_json(record['plan'])
    settings = Settings(Simulators(**{k: Path(v) for k, v in record['settings']['simulators'].items()}),
                        definitions_from_json(record['plan']), record['settings']['lock_wait'],
                        record['settings']['timeout'], keep_outputs=True,
                        threads=record['settings'].get('threads', 1))
    cpus = record['cpus']
    if cpus:
        check_cpus(cpus)
    workdir = directory / 'repro'
    if workdir.exists():
        shutil.rmtree(workdir)
    bench = Testbench(settings, cpus)
    evidence = bench.run(Deck.load(directory, 'reduced.sp'), workdir, plan=plan)
    for name, run in evidence.runs.items():
        print(f'{name}: exit {run.outcome}')
    if record.get('predicate'):
        from .predicates import Predicate
        selected = Predicate.from_json(record['predicate']).evaluate(evidence)
        status = {'matched': 'reproduced', 'not-matched': 'not-reproduced', 'unresolved': 'unresolved'}[selected.state]
        (directory / 'replay-result.json').write_text(json.dumps({'status': status, 'reasons': selected.reasons,
                                                                 'predicate': record['predicate']}, indent=2) + '\n')
        print(status)
        return {'matched': 0, 'not-matched': 1, 'unresolved': 2}[selected.state]
    verdict = oracles.judge(evidence)
    for finding in verdict.findings:
        print(f'finding: {finding}')
    matched = [f for f in verdict.findings if f.startswith(record['check'])]
    unavailable = bool(verdict.inconclusive) or any(not run.ok for run in evidence.runs.values())
    status = 'reproduced' if matched else ('unresolved' if unavailable else 'not-reproduced')
    result: dict[str, Any] = {'status': status, 'check': record['check'], 'findings': list(verdict.findings),
                              'inconclusive': list(verdict.inconclusive), 'notes': list(verdict.notes),
                              'checks': {k: v.value for k, v in verdict.checks.items()}}
    (directory / 'replay-result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(status)
    return 0 if matched else (2 if unavailable else 1)


def main() -> None:
    try:
        status = replay(Path(sys.argv[1]))
    except (ValueError, OSError, KeyError) as error:
        print(f'repro.sh: {error}', file=sys.stderr)
        status = 2
    raise SystemExit(status)


if __name__ == '__main__':
    main()
