"""Running decks, cases and batches (see README.md).

A Testbench runs one deck in a set of configurations and gathers the Evidence the oracles
judge. A Batch is a directory of cases, one per seed: case-SEED/ holds deck.sp, a directory
per configuration (the deck as run, log.txt, outputs), the front-end dumps (front-ref,
front-exact) and summary.json. Passing cases drop their outputs and dumps.
"""
from __future__ import annotations

import concurrent.futures
import functools
import hashlib
import itertools
import json
import math
import shutil
import signal
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence, Union

from . import generator, oracles
from .config import CONFIGURATIONS, CPUS, THREADS, TIMEOUT, Configuration, CpuSet, Simulators, binary_json
from .deck import TIGHT_LEVELS, Deck
from .oracles import DEBUG_FILES, Evidence, FrontEnd, Repeat, Verdict
from .plans import ExecutionPlan, definitions_from_json
from .provenance import Provenance
from .simulation import PROCESSES, RunResult, Simulation


@dataclass(frozen=True)
class Settings:
    """How a batch runs its cases."""
    simulators: Simulators = Simulators()
    # The configurations a batch runs: the built-in ones, and any from --extra-configurations.
    configurations: Mapping[str, Configuration] = field(default_factory=lambda: CONFIGURATIONS)
    lock_wait: float = 0.0  # retry a busy cooperative lock this long (seconds) before giving up
    timeout: float = TIMEOUT
    front_end: bool = True  # front-end identity runs
    repeat_every: int = 10  # run ref twice for every N-th seed (self-determinism); 0: never
    keep_outputs: bool = False  # keep the outputs of passing cases
    threads: int = THREADS

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError('timeout must be finite and positive')
        if not math.isfinite(self.lock_wait) or self.lock_wait < 0:
            raise ValueError('lock_wait must be finite and nonnegative')
        if self.repeat_every < 0:
            raise ValueError('repeat_every must be nonnegative')
        if type(self.threads) is not int or self.threads < 1:
            raise ValueError('threads must be a positive integer')

    def check_reference(self) -> None:
        """Validate the caller's reference executable, without a project-specific hash."""
        from .config import Binary
        self.simulators.path(Binary.REFERENCE)


@dataclass(frozen=True)
class Testbench:
    """Runs a deck in a set of configurations and gathers the evidence."""
    __test__ = False  # not a pytest test class
    settings: Settings
    cpus: Optional[str] = None

    @property
    def configurations(self) -> Mapping[str, Configuration]:
        return self.settings.configurations

    def simulation(self, configuration: Configuration, deck: Deck, arguments: tuple[str, ...] = ()) -> Simulation:
        return Simulation(self.settings.simulators.path(configuration.binary),
                          configuration.flags_for(deck.has_small_signal_analysis), self.cpus,
                          self.settings.timeout, configuration.locks, self.settings.lock_wait, arguments,
                          self.settings.threads)

    def run(self, deck: Deck, workdir: Path, names: Optional[Sequence[str]] = None,
            front_end: Optional[bool] = None, repeat: bool = False,
            plan: Optional[ExecutionPlan] = None) -> Evidence:
        if plan is None:
            names = list(names if names is not None else self.configurations)
            if front_end is None:
                front_end = self.settings.front_end and 'ref' in names and 'exact' in names
            plan = ExecutionPlan(tuple(names), front_end, repeat)
        names = list(plan.names)
        unknown = set(names) - self.configurations.keys()
        if unknown:
            raise ValueError(f'unknown configurations: {sorted(unknown)}')
        if sum(self.configurations[name].tight for name in names) > 1:
            raise ValueError('an execution plan supports one tight-tolerance configuration')
        workdir.mkdir(parents=True, exist_ok=True)
        deck.save(workdir)
        (workdir / 'plan.json').write_text(json.dumps(plan.to_json(self.configurations), indent=2))
        runs: dict[str, RunResult] = {}
        tight_level = None
        for name in names:
            configuration = self.configurations[name]
            if configuration.tight:
                if plan.tight_level is None:
                    runs[name], tight_level = self._accuracy_reference(deck, workdir, configuration)
                else:
                    tight_level = plan.tight_level
                    runs[name] = self.simulation(configuration, deck).run(deck.tightened(tight_level),
                                                                         workdir / name, name)
            else:
                runs[name] = self.simulation(configuration, deck).run(deck, workdir / name, name)
        front = self._front_end(deck, workdir) if plan.front_end else None
        repeats: tuple[Repeat, ...] = ()
        if plan.repeat_reference:
            again = self.simulation(self.configurations['ref'], deck).run(deck, workdir / 'ref-repeat', 'ref')
            repeats = (Repeat('ref', runs['ref'], again),)
        samples = {name: [run] for name, run in runs.items()}
        for index in range(1, plan.repetitions):
            for name in reversed(names) if index % 2 else names:
                configuration = self.configurations[name]
                variant = deck.tightened(tight_level or 0) if configuration.tight else deck
                samples[name].append(self.simulation(configuration, deck).run(
                    variant, workdir / f'{name}-sample-{index}', name))
        return Evidence(deck, runs, front, repeats, tight_level, self.configurations,
                        {name: tuple(values) for name, values in samples.items()})

    def _accuracy_reference(self, deck: Deck, workdir: Path, configuration: Configuration) -> tuple[RunResult, int]:
        """tight at the first tolerance level that finishes. Fallback directories of an
        earlier run in this work directory go first, so none can pass for this run's."""
        name = configuration.name
        directories = [workdir / (name if level == 0 else f'{name}-{level}') for level in range(len(TIGHT_LEVELS))]
        for directory in directories[1:]:
            shutil.rmtree(directory, ignore_errors=True)
        simulation = self.simulation(configuration, deck)
        for level, directory in enumerate(directories):
            run = simulation.run(deck.tightened(level), directory, name)
            if run.ok:
                break
        return run, level

    def _front_end(self, deck: Deck, workdir: Path) -> FrontEnd:
        """ngdebug dumps of the parse-only deck from ref and the candidate's default
        environment, each in a fresh directory so no earlier dump can stand in for them."""
        parse_only = deck.front_end()
        dirs = (workdir / 'front-ref', workdir / 'front-exact')
        runs = []
        for directory, name in zip(dirs, ('ref', 'exact'), strict=True):
            shutil.rmtree(directory, ignore_errors=True)
            simulation = self.simulation(self.configurations[name], deck, ('-D', 'ngdebug'))
            runs.append(simulation.run(parse_only, directory, directory.name))
        return FrontEnd(dirs[0], dirs[1], runs[0], runs[1])


def load_evidence(directory: Path) -> Evidence:
    """The evidence stored in a case or reproducer directory, to judge it again: one run per
    configuration directory (tight from its last fallback level), the front-end dumps and a
    repeated ref run when present."""
    runs: dict[str, RunResult] = {}
    tight_level = None
    record = json.loads((directory / 'plan.json').read_text()) if (directory / 'plan.json').exists() else None
    plan = ExecutionPlan.from_json(record) if record is not None else None
    definitions = definitions_from_json(record) if record is not None else CONFIGURATIONS
    for name in plan.names if plan else CONFIGURATIONS:
        levels = [directory / name]
        if definitions[name].tight and (plan is None or plan.tight_level is None):
            levels += [directory / f'{name}-{i}' for i in range(1, len(TIGHT_LEVELS))]
        stored = [d for d in levels if (d / 'log.txt').exists()]
        if stored:
            runs[name] = RunResult.load(stored[-1])
            if definitions[name].tight:
                tight_level = levels.index(stored[-1])
    if not runs:
        raise SystemExit(f'{directory}: no runs to load')
    if plan and plan.tight_level is not None:
        tight_level = plan.tight_level
    deck_directory = directory if (directory / 'deck.sp').exists() else directory / next(iter(runs))
    dumps = (directory / 'front-ref', directory / 'front-exact')
    front = None
    if (plan is None or plan.front_end) and all((d / 'log.txt').exists() for d in dumps):
        front = FrontEnd(dumps[0], dumps[1], RunResult.load(dumps[0]), RunResult.load(dumps[1]))
    repeats: tuple[Repeat, ...] = ()
    if (plan is None or plan.repeat_reference) and 'ref' in runs and (directory / 'ref-repeat/log.txt').exists():
        repeats = (Repeat('ref', runs['ref'], RunResult.load(directory / 'ref-repeat')),)
    samples = {name: (run,) + tuple(RunResult.load(directory / f'{name}-sample-{i}')
                                   for i in range(1, plan.repetitions if plan else 1))
               for name, run in runs.items()}
    return Evidence(Deck.load(deck_directory), runs, front, repeats, tight_level, definitions, samples)


@dataclass(frozen=True)
class BatchIdentity:
    """What makes cases comparable: the binaries and frozen inputs, the generator, the code
    that generates, runs and judges, the configurations, and the settings that change which
    checks run. Cases of different identities do not belong in one batch's statistics."""
    reference: str
    candidate: str
    extra_binaries: Mapping[str, str]  # path -> sha256 of the binaries extra configurations name
    inputs: str  # sha256 over the frozen inputs' hashes
    generator: int
    scale: int
    code: str
    configurations: str  # sha256 over the enabled configurations' definitions
    front_end: bool
    repeat_every: int
    timeout: Optional[float] = None
    lock_wait: Optional[float] = None
    threads: int = THREADS

    @classmethod
    def of(cls, provenance: Provenance, settings: Settings) -> BatchIdentity:
        enabled = {name: {'binary': binary_json(c.binary), 'claim': c.claim.value,
                          'flags': dict(sorted(c.flags.items())), 'tight': c.tight, 'base': c.base,
                          'locks': list(c.locks)}
                   for name, c in settings.configurations.items()}
        return cls(provenance.reference.sha256, provenance.candidate.sha256,
                   {record.path: record.sha256 for record in provenance.extra_binaries},
                   _digest(dict(sorted(provenance.inputs.items()))), provenance.generator, provenance.scale,
                   str(provenance.tool.get('code_sha256', '')), _digest(enabled), settings.front_end,
                   settings.repeat_every, settings.timeout, settings.lock_wait, settings.threads)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> BatchIdentity:
        return cls(str(data.get('reference')), str(data.get('candidate')), dict(data.get('extra_binaries') or {}),
                   str(data.get('inputs')), int(data.get('generator') or 0), int(data.get('scale') or 1),
                   str(data.get('code')), str(data.get('configurations')),
                   bool(data.get('front_end')), int(data.get('repeat_every') or 0),
                   data.get('timeout'), data.get('lock_wait'), data.get('threads', THREADS))

    def to_json(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def digest(self) -> str:
        return _digest(self.to_json())[:16]

    def differences(self, other: BatchIdentity) -> list[str]:
        """'field: this -> other' for every field that differs."""
        return [f'{name}: {getattr(self, name)} -> {getattr(other, name)}' for name in self.__dataclass_fields__
                if getattr(self, name) != getattr(other, name)]


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class CaseSummary:
    """summary.json of one case."""
    seed: int
    findings: tuple[str, ...]
    notes: tuple[str, ...]
    inconclusive: tuple[str, ...]
    status: Mapping[str, Any]
    seconds: Mapping[str, float]
    tight_level: Optional[int]
    ref_errors: tuple[str, ...]
    deck: Mapping[str, Any]
    checks: Mapping[str, str]
    metrics: Mapping[str, Any]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def of(cls, seed: int, evidence: Evidence, verdict: Verdict, provenance: Provenance,
           identity: Optional[str] = None) -> CaseSummary:
        deck = evidence.deck
        description = deck.info.to_json() if deck.info else {'vdd': deck.vdd, 'tstop': deck.tstop}
        runs = evidence.runs
        return cls(seed, verdict.findings, verdict.notes, verdict.inconclusive,
                   {k: r.outcome for k, r in runs.items()}, {k: round(r.seconds, 2) for k, r in runs.items()},
                   evidence.tight_level, runs['ref'].diagnostics[:5], description,
                   {k: v.value for k, v in verdict.checks.items()}, verdict.metrics,
                   {**provenance.to_json(), 'identity': identity,
                    'bases': {name: c.base for name, c in evidence.definitions.items()},
                    'runs': {k: {'command': list(r.command), 'env': dict(r.environment)} for k, r in runs.items()}})

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> CaseSummary:
        return cls(data['seed'], tuple(data.get('findings', ())), tuple(data.get('notes', ())),
                   tuple(data.get('inconclusive', ())), data.get('status', {}), data.get('seconds', {}),
                   data.get('tight_level'), tuple(data.get('ref_errors', ())), data.get('deck', {}),
                   data.get('checks', {}), data.get('metrics') or {}, data.get('provenance', {}))

    def to_json(self) -> dict[str, Any]:
        return {'seed': self.seed, 'findings': list(self.findings), 'notes': list(self.notes),
                'inconclusive': list(self.inconclusive), 'status': dict(self.status), 'seconds': dict(self.seconds),
                'tight_level': self.tight_level, 'ref_errors': list(self.ref_errors), 'deck': dict(self.deck),
                'checks': dict(self.checks), 'metrics': dict(self.metrics), 'provenance': dict(self.provenance)}

    @property
    def clean(self) -> bool:
        """A plain pass: no findings, notes or inconclusive reasons (only then may the case
        drop its outputs and dumps)."""
        return not (self.findings or self.notes or self.inconclusive)

    def line(self) -> str:
        """One console line."""
        verdict = 'FAIL ' + '; '.join(self.findings) if self.findings else 'ok'
        if self.notes:
            verdict += ' (note: ' + '; '.join(self.notes) + ')'
        if self.inconclusive:
            verdict += ' (inconclusive: ' + '; '.join(self.inconclusive) + ')'
        unusual = {k: v for k, v in self.status.items() if v != 0}
        status = 'all exit 0' if not unusual else ' '.join(f'{k} {v}' for k, v in unusual.items())
        if self.tight_level:
            status += f', tight level {self.tight_level}'
        extra = sorted({key.removesuffix('_shift_ps') for key in self.metrics if key.endswith('_shift_ps')}
                       - CONFIGURATIONS.keys())
        shifts = ' / '.join(f"{name} {self.metrics[name + '_shift_ps']:.2f} ps" for name in ('ref', 'approx', *extra)
                            if name + '_shift_ps' in self.metrics)
        return f'seed {self.seed:6d} {verdict} | {status}' + (f' | {shifts}' if shifts else '')


@dataclass(frozen=True)
class CaseError:
    """A case the harness itself failed on (traceback in error.txt)."""
    seed: int
    message: str

    def line(self) -> str:
        return f'seed {self.seed:6d} HARNESS ERROR {self.message} (see error.txt)'


@dataclass(frozen=True)
class Tally:
    """Counts over the finished cases of a batch."""
    cases: int = 0
    failing: int = 0
    noted: int = 0
    inconclusive: int = 0
    ref_failed: int = 0
    tight_fallback: int = 0
    harness_errors: int = 0
    failed_checks: Mapping[str, int] = field(default_factory=dict)

    @classmethod
    def of(cls, summaries: Iterable[CaseSummary], harness_errors: int = 0) -> Tally:
        summaries = list(summaries)
        failed: dict[str, int] = {}
        for s in summaries:
            for check, result in s.checks.items():
                if result == 'fail':
                    failed[check] = failed.get(check, 0) + 1
        return cls(len(summaries), sum(bool(s.findings) for s in summaries), sum(bool(s.notes) for s in summaries),
                   sum(bool(s.inconclusive) for s in summaries), sum(s.status.get('ref') != 0 for s in summaries),
                   sum(bool(s.tight_level) for s in summaries), harness_errors, dict(sorted(failed.items())))

    def to_json(self) -> dict[str, Any]:
        return {'cases': self.cases, 'failing': self.failing, 'noted': self.noted, 'inconclusive': self.inconclusive,
                'ref_failed': self.ref_failed, 'tight_fallback': self.tight_fallback,
                'harness_errors': self.harness_errors, 'failed_checks': dict(self.failed_checks)}


class Batch:
    """A batch directory of cases."""

    def __init__(self, directory: Path, settings: Settings, provenance: Provenance,
                 generate: Callable[[int], Deck] = generator.generate):
        self.directory = directory.resolve()
        self.settings = settings
        self.provenance = provenance
        self.generate = generate

    # --- Contents ----------------------------------------------------------------------------

    def case_directory(self, seed: int) -> Path:
        return self.directory / f'case-{seed:06d}'

    def finished(self) -> set[int]:
        """Seeds of the cases that have a summary."""
        return {int(p.parent.name.split('-')[1]) for p in self.directory.glob('case-*/summary.json')}

    def summaries(self) -> Iterator[CaseSummary]:
        for path in sorted(self.directory.glob('case-*/summary.json')):
            try:
                yield CaseSummary.from_json(json.loads(path.read_text()))
            except (ValueError, KeyError):
                continue

    def tally(self) -> Tally:
        return Tally.of(self.summaries(), len(list(self.directory.glob('case-*/error.txt'))))

    def seeds(self, start: int, cases: int, resume: bool = False, fresh: bool = False) -> Iterator[int]:
        """`cases` seeds from `start` (0: endless), skipping finished cases unless `fresh`;
        with `resume`, from just after the last finished seed."""
        finished = set() if fresh else self.finished()
        if resume and finished:
            start = max(finished) + 1
        numbers: Iterable[int] = itertools.count(start) if cases == 0 else range(start, start + cases)
        return (seed for seed in numbers if seed not in finished)

    def record(self, arguments: Mapping[str, Any], force: bool = False) -> None:
        """Add this session to batch.json. A batch holds cases of one BatchIdentity: a session
        with another identity is refused, unless forced, and then recorded beside the earlier
        ones (each summary names the identity it was made under, for the report)."""
        path = self.directory / 'batch.json'
        identity = self.identity
        record: dict[str, Any] = {'identity': identity.to_json(), 'sessions': []}
        if path.exists():
            record = json.loads(path.read_text())
            if 'identity' not in record and not force:
                raise SystemExit(f'{path} predates batch identities; use another --out or --force')
            differences = BatchIdentity.from_json(record['identity']).differences(identity) \
                if 'identity' in record else ['identity']
            if differences and not force:
                raise SystemExit(f'{path}: this session differs in ' + '; '.join(differences)
                                 + '\nuse another --out, or --force to mix them')
            record.setdefault('identity', identity.to_json())
            record.setdefault('sessions', [])
        record['sessions'].append({'started': time.strftime('%Y-%m-%dT%H:%M:%S'), 'identity': identity.digest(),
                                   'args': dict(arguments), 'provenance': self.provenance.to_json()})
        self.directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1))

    @functools.cached_property
    def identity(self) -> BatchIdentity:
        return BatchIdentity.of(self.provenance, self.settings)

    # --- Running -----------------------------------------------------------------------------

    def run_case(self, seed: int, cpus: Optional[str] = None) -> Union[CaseSummary, CaseError]:
        """Generate, run and judge one case. A harness error becomes error.txt in the case
        and a console line instead of stopping an unattended batch."""
        case = self.case_directory(seed)
        try:
            if case.exists():
                shutil.rmtree(case)
            case.mkdir(parents=True)
            deck = self.generate(seed)
            deck.save(case)
            repeat = bool(self.settings.repeat_every) and seed % self.settings.repeat_every == 0
            evidence = Testbench(self.settings, cpus).run(deck, case, repeat=repeat)
            summary = CaseSummary.of(seed, evidence, oracles.judge(evidence), self.provenance, self.identity.digest())
            (case / 'summary.json').write_text(json.dumps(summary.to_json(), indent=1, allow_nan=False))
            if summary.clean and not self.settings.keep_outputs:
                self._tidy(case, deck.outputs)
            return summary
        except Exception:
            case.mkdir(parents=True, exist_ok=True)
            (case / 'error.txt').write_text(traceback.format_exc())
            return CaseError(seed, traceback.format_exc().strip().splitlines()[-1])

    def run(self, seeds: Iterable[int], jobs: int, cpus: Optional[str] = CPUS,
            echo: Callable[[str], None] = lambda line: print(line, flush=True)) -> Tally:
        """Run the cases of `seeds` (possibly endless) on up to `jobs` disjoint CPU sets.
        Ctrl-C or SIGTERM stops submitting and lets running cases finish; a second Ctrl-C
        kills them. Without --cpus jobs are unpinned."""
        if jobs < 1:
            raise ValueError('jobs must be positive')
        sets = [str(s) for s in CpuSet.split(cpus, self.settings.threads)[:jobs]] if cpus else [None] * jobs
        if not sets:
            raise SystemExit(f'no {self.settings.threads}-CPU set in {cpus}')
        previous = signal.signal(signal.SIGTERM, _interrupt)
        started, count = time.monotonic(), 0
        seeds = iter(seeds)
        with concurrent.futures.ThreadPoolExecutor(len(sets)) as pool:
            running: dict[concurrent.futures.Future, Optional[str]] = {}
            free, exhausted = list(sets), False
            try:
                while True:
                    while free and not exhausted:
                        seed = next(seeds, None)
                        if seed is None:
                            exhausted = True
                            break
                        cpu = free.pop()
                        running[pool.submit(self.run_case, seed, cpu)] = cpu
                    if not running:
                        break
                    finished, _ = concurrent.futures.wait(running, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in finished:
                        free.append(running.pop(future))
                        echo(future.result().line())
                        count += 1
            except KeyboardInterrupt:
                echo(f'interrupted: finishing {len(running)} running case(s); Ctrl-C again to kill them')
                try:
                    for future in concurrent.futures.as_completed(running):
                        echo(future.result().line())
                        count += 1
                except KeyboardInterrupt:
                    PROCESSES.kill_all()
                    pool.shutdown(cancel_futures=True)
                    raise
            finally:
                signal.signal(signal.SIGTERM, previous)
        tally = self.tally()
        (self.directory / 'tally.json').write_text(json.dumps(tally.to_json(), indent=1))
        echo(f'{count} case(s) in {time.monotonic() - started:.0f} s; batch: {json.dumps(tally.to_json())}')
        return tally

    @staticmethod
    def _tidy(case: Path, outputs: Sequence[str]) -> None:
        """A passing case keeps its decks, logs and summary, not its outputs or dumps."""
        for directory in case.iterdir():
            if directory.is_dir():
                for name in [*outputs, *DEBUG_FILES, 'debug-out.txt']:
                    (directory / name).unlink(missing_ok=True)


def _interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt
