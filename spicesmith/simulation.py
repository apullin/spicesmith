"""Running a simulator on a deck, and what the run reported.

A Simulation knows how to run one binary (flags, CPU pinning, cooperative locks, time
limit); running it on a Deck in a directory gives a RunResult: exit status, outputs,
diagnostics and ngspice's rusage counters. The simulator sees a clean environment, all of
which the result records.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Optional, Union

from .config import THREADS, TIMEOUT, executable
from .deck import Deck

Status = Union[int, str]  # exit status, or 'timeout' / BUSY
BUSY = 'busy'  # a cooperative lock was held elsewhere: the simulator did not run
LOCK_BUSY = 75  # flock's exit status (-E) for a lock held elsewhere (EX_TEMPFAIL)
NGDEBUG_DUMPS = ('debug-out.txt', 'debug-out2.txt', 'debug-out3.txt', 'debug-out-mc.txt')  # -D ngdebug writes these

# Variables the simulator inherits from the caller; everything else is set here, so no
# credentials from the caller's environment end up in a summary.
INHERITED = ('HOME', 'USER', 'TMPDIR')


@dataclass(frozen=True)
class SimulatorLog:
    """What ngspice printed (stdout, then stderr)."""
    text: str

    DIAGNOSTIC = re.compile(r'error|warning|singular|too small|abort|fatal|fail|stepping', re.I)
    # Lines that match DIAGNOSTIC but are environment, banners or structured log records.
    BENIGN = re.compile(r"^Warning: can't find the initialization file|^\s*\{\"|Copyright|Creation date",
                        re.I)
    ABORTED = re.compile(r'simulation\(s\) aborted|Timestep too small', re.I)
    STATS = {
        'Number of lines in the deck': 'deck_lines', 'Total iterations': 'iterations',
        'Transient iterations': 'tran_iterations', 'Circuit Equations': 'equations',
        'Circuit original non-zeroes': 'nonzeros', 'Circuit fill-in non-zeroes': 'fillin',
        'Circuit total non-zeroes': 'total_nonzeros', 'Transient timepoints': 'timepoints',
        'Accepted timepoints': 'accepted', 'Rejected timepoints': 'rejected',
    }
    STAT_LINE = re.compile(r'^(%s) = (\S+)\s*$' % '|'.join(map(re.escape, STATS)), re.M)

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """Errors, warnings, aborts and convergence aids, in order."""
        return tuple(line.rstrip() for line in self.text.splitlines()
                     if self.DIAGNOSTIC.search(line) and not self.BENIGN.search(line))

    @property
    def stats(self) -> Mapping[str, float]:
        """The `rusage all` counters (the last value of each)."""
        return {self.STATS[name]: float(value) for name, value in self.STAT_LINE.findall(self.text)}

    @property
    def aborted(self) -> bool:
        """An analysis aborted; ngspice still exits 0 and writes the partial outputs."""
        return bool(self.ABORTED.search(self.text))


@dataclass(frozen=True)
class RunResult:
    """One simulator run of one deck."""
    configuration: str
    status: Status
    seconds: float = 0.0
    outputs: Mapping[str, Optional[bytes]] = field(default_factory=dict)  # None: not written
    diagnostics: tuple[str, ...] = ()
    stats: Mapping[str, float] = field(default_factory=dict)
    aborted: bool = False
    command: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Finished, no analysis aborted, every output written."""
        return (self.status == 0 and not self.aborted
                and all(data is not None for data in self.outputs.values()))

    @property
    def outcome(self) -> Status:
        """The exit status, or 'aborted' / 'no-output' for runs that exit 0 without finishing."""
        if self.status == 0 and self.aborted:
            return 'aborted'
        if self.status == 0 and not self.ok:
            return 'no-output'
        return self.status

    def record(self) -> dict[str, object]:
        """run.json: what the log and the outputs do not say."""
        return {'configuration': self.configuration, 'status': self.status, 'seconds': self.seconds,
                'command': list(self.command), 'environment': dict(self.environment)}

    @classmethod
    def load(cls, directory: Path) -> RunResult:
        """A run stored in `directory`: run.json, log.txt, deck.sp (as run) and its outputs."""
        record = json.loads((directory / 'run.json').read_text()) if (directory / 'run.json').exists() else {}
        log = SimulatorLog((directory / 'log.txt').read_text(errors='replace')
                           if (directory / 'log.txt').exists() else '')
        deck = (directory / 'deck.sp').read_text(errors='replace') if (directory / 'deck.sp').exists() else ''
        names = [Path(path).name for path in re.findall(r'(?m)^wrdata (\S+)', deck)]
        outputs = {name: (directory / name).read_bytes() if (directory / name).exists() else None for name in names}
        return cls(record.get('configuration', directory.name), record.get('status', 'unknown'),
                   record.get('seconds', 0.0), outputs, log.diagnostics, log.stats, log.aborted,
                   tuple(record.get('command', ())), record.get('environment', {}))


class ProcessGroups:
    """The process groups of running simulations, so that an interrupted batch can kill them."""

    def __init__(self) -> None:
        self._pids: set[int] = set()
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def tracking(self, pid: int) -> Iterator[None]:
        with self._lock:
            self._pids.add(pid)
        try:
            yield
        finally:
            with self._lock:
                self._pids.discard(pid)

    def kill_all(self) -> None:
        with self._lock:
            for pid in self._pids:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(pid, signal.SIGKILL)


PROCESSES = ProcessGroups()
_LOCKED = threading.Lock()  # one locking run at a time in this process; the flocks cover others


@dataclass(frozen=True)
class Simulation:
    """How to run one simulator binary."""
    binary: Path
    flags: Mapping[str, str] = field(default_factory=dict)
    cpus: Optional[str] = None  # taskset CPU list; None: not pinned
    timeout: float = TIMEOUT
    locks: tuple[str, ...] = ()  # cooperative lock files to hold, the first taken first
    lock_wait: float = 0.0  # retry a busy lock this long before giving up
    arguments: tuple[str, ...] = ()  # extra command-line arguments, e.g. ('-D', 'ngdebug')
    threads: int = THREADS

    def environment(self) -> dict[str, str]:
        environment = {k: os.environ[k] for k in INHERITED if k in os.environ}
        environment.update(PATH='/usr/bin:/bin', LC_ALL='C', OMP_NUM_THREADS=str(self.threads),
                           RAYON_NUM_THREADS=str(self.threads), OMP_WAIT_POLICY='PASSIVE')
        environment.update(self.flags)
        return environment

    def command(self) -> list[str]:
        command = [str(executable(self.binary)), '-n', '-b', *self.arguments, 'deck.sp']
        if self.cpus:
            command = ['taskset', '-c', self.cpus, *command]
        for lock in reversed(self.locks):
            command = ['flock', '-n', '-E', str(LOCK_BUSY), lock, *command]
        return command

    def run(self, deck: Deck, workdir: Path, name: str = 'run') -> RunResult:
        """Run the deck in `workdir` (deck.sp as run, log.txt, the outputs)."""
        from .ngspice import prepare
        deck = prepare(deck, self.threads)
        workdir = workdir.resolve()  # the deck names its outputs by absolute path
        self._prepare(deck, workdir)
        environment = self.environment()
        command = self.command()
        status, log, seconds = self._execute_with_lock_retry(command, workdir, environment)
        (workdir / 'log.txt').write_text(log.text)
        outputs = {f: (workdir / f).read_bytes() if (workdir / f).exists() else None for f in deck.outputs}
        result = RunResult(name, status, seconds, outputs, log.diagnostics, log.stats, log.aborted,
                           tuple(command), environment)
        (workdir / 'run.json').write_text(json.dumps(result.record(), indent=1))
        return result

    # --- Internals -------------------------------------------------------------------------

    @staticmethod
    def _prepare(deck: Deck, workdir: Path) -> None:
        workdir.mkdir(parents=True, exist_ok=True)
        for output in [*deck.outputs, *NGDEBUG_DUMPS]:  # nothing stale may pass for this run's output
            (workdir / output).unlink(missing_ok=True)
        (workdir / 'deck.sp').write_text(deck.bound_to(workdir))
        deck.write_files(workdir)

    def _execute_with_lock_retry(self, command: list[str], workdir: Path,
                                 environment: Mapping[str, str]) -> tuple[Status, SimulatorLog, float]:
        deadline = time.monotonic() + self.lock_wait
        while True:
            with _LOCKED if self.locks else contextlib.nullcontext():
                started = time.monotonic()
                status, log = self._execute(command, workdir, environment)
                seconds = time.monotonic() - started
            busy = self.locks and status == LOCK_BUSY and not log.text
            if not busy:
                return status, log, seconds
            if time.monotonic() >= deadline:
                return BUSY, log, seconds
            time.sleep(min(10.0, max(0.0, deadline - time.monotonic())))

    def _execute(self, command: list[str], workdir: Path,
                 environment: Mapping[str, str]) -> tuple[Status, SimulatorLog]:
        """Run in a new process group, killed as a whole on timeout or interrupt: flock forks
        the simulator, which would otherwise outlive it and keep the locks."""
        process = subprocess.Popen(command, cwd=workdir, env=dict(environment), stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, start_new_session=True)
        status: Status
        with PROCESSES.tracking(process.pid):
            try:
                out, err = process.communicate(timeout=self.timeout)
                status = process.returncode
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                out, err = process.communicate()
                status = 'timeout'
            except BaseException:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
        return status, SimulatorLog(out.decode(errors='replace') + err.decode(errors='replace'))
