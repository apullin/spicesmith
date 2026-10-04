"""Reducing a failing deck to a small reproducer (see README.md).

A Reduction is one way of making a deck smaller: removing top-level lines, whole subcircuit
definitions, lines inside subcircuits or observed vectors, shortening the transient, or
simplifying numbers. Removals run ddmin over their units; simplifications try candidate
decks one at a time. The Reducer applies reductions until none makes progress, yielding each
smaller deck; a FindingTest decides whether a deck still shows the finding, on a deck the
reference still accepts (otherwise reduction drifts into invalid decks).
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Collection, Generic, Iterator, Mapping, Optional, Sequence, TypeVar

from . import oracles
from .config import Claim
from .deck import Deck, eng, spice_number
from .harness import Testbench
from .plans import ExecutionPlan
from .provenance import REPO, sha256

DeckTest = Callable[[Deck], bool]


class BudgetExhausted(Exception):
    """No more predicate evaluations are permitted in this reduction search."""


@dataclass
class TrialBudget:
    test: DeckTest
    limit: int
    used: int = 0

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError('trial budget must be nonnegative')

    def __call__(self, deck: Deck) -> bool:
        if self.used >= self.limit:
            raise BudgetExhausted
        self.used += 1
        return self.test(deck)


# --- Deck structure --------------------------------------------------------------------------

@dataclass(frozen=True)
class Layout:
    """Where things are in a deck's lines: subcircuit blocks, the .control block, and the
    lines that are neither (title and .end excluded)."""
    subcircuits: tuple[tuple[int, int], ...]  # (.subckt line, .ends line)
    control: tuple[int, int]  # (.control line, .endc line); (n, n) without one
    top_level: tuple[int, ...]

    @classmethod
    def of(cls, deck: Deck) -> Layout:
        lines = deck.lines
        subcircuits, start = [], None
        control = (len(lines), len(lines))
        for i, line in enumerate(lines):
            word = line.split()[0].lower() if line.split() else ''
            if word == '.subckt':
                start = i
            elif word == '.ends' and start is not None:
                subcircuits.append((start, i))
                start = None
            elif word == '.control':
                control = (i, next((j for j in range(i, len(lines)) if lines[j].strip().lower() == '.endc'),
                                   len(lines) - 1))
        inside = {i for a, b in subcircuits for i in range(a, b + 1)} | set(range(control[0], control[1] + 1))
        top = tuple(i for i, line in enumerate(lines) if i > 0 and i not in inside and line.strip()
                    and line.strip().lower() != '.end' and not line.startswith('*'))
        return cls(tuple(subcircuits), control, top)

    def subcircuit_body(self) -> tuple[int, ...]:
        return tuple(i for a, b in self.subcircuits for i in range(a + 1, b))


def without_lines(deck: Deck, removed: Collection[int]) -> Deck:
    return deck.with_text('\n'.join(line for i, line in enumerate(deck.lines) if i not in removed))


# --- Reductions ------------------------------------------------------------------------------

class Reduction(ABC):
    """One way of making a deck smaller."""
    name: ClassVar[str]

    @abstractmethod
    def steps(self, deck: Deck, test: DeckTest) -> Iterator[Deck]:
        """Successively smaller decks that pass `test`, each from the last."""


U = TypeVar('U')  # a removable unit: a line index, a subcircuit's span, an observed vector


class Removal(Reduction, Generic[U]):
    """Removes units of a deck, ddmin style."""

    @abstractmethod
    def units(self, deck: Deck) -> Sequence[U]:
        """The removable units of a deck."""

    @abstractmethod
    def remove(self, deck: Deck, units: Collection[U]) -> Deck:
        """The deck without those units."""

    def steps(self, deck: Deck, test: DeckTest) -> Iterator[Deck]:
        units = list(self.units(deck))
        removed: set[int] = set()  # indices into units
        current = deck
        chunk = max(1, len(units) // 2)
        while chunk >= 1 and len(removed) < len(units):
            progress = False
            live = [i for i in range(len(units)) if i not in removed]
            for start in range(0, len(live), chunk):
                trial = removed | set(live[start:start + chunk])
                candidate = self.remove(deck, [units[i] for i in trial])
                if candidate.text != current.text and test(candidate):
                    removed, current, progress = trial, candidate, True
                    yield candidate
            if not progress:
                chunk //= 2


class Simplification(Reduction):
    """Tries candidate decks one at a time, keeping each that passes."""

    @abstractmethod
    def candidates(self, deck: Deck) -> Iterator[Deck]:
        """Simpler variants of `deck`, best first."""

    def steps(self, deck: Deck, test: DeckTest) -> Iterator[Deck]:
        improved = True
        while improved:
            improved = False
            for candidate in self.candidates(deck):
                if candidate.text != deck.text and test(candidate):
                    deck, improved = candidate, True
                    yield deck
                    break


@dataclass(frozen=True)
class RemoveSubcircuits(Removal[tuple[int, int]]):
    """Whole .subckt ... .ends definitions (an unused one, or one whose instances went)."""
    name: ClassVar[str] = 'subcircuits'

    def units(self, deck: Deck) -> Sequence[tuple[int, int]]:
        return Layout.of(deck).subcircuits

    def remove(self, deck: Deck, units: Collection[tuple[int, int]]) -> Deck:
        return without_lines(deck, {i for start, end in units for i in range(start, end + 1)})


@dataclass(frozen=True)
class RemoveLines(Removal[int]):
    """Top-level lines: elements, instances, .param, .ic, .model, .temp, options ..."""
    name: ClassVar[str] = 'lines'

    def units(self, deck: Deck) -> Sequence[int]:
        return Layout.of(deck).top_level

    def remove(self, deck: Deck, units: Collection[int]) -> Deck:
        return without_lines(deck, set(units))


@dataclass(frozen=True)
class RemoveSubcircuitLines(Removal[int]):
    """Lines inside subcircuit definitions."""
    name: ClassVar[str] = 'subcircuit-lines'

    def units(self, deck: Deck) -> Sequence[int]:
        return Layout.of(deck).subcircuit_body()

    def remove(self, deck: Deck, units: Collection[int]) -> Deck:
        return without_lines(deck, set(units))


@dataclass(frozen=True)
class DropOutputs(Removal[tuple[int, str]]):
    """Observed vectors that do not matter (at least one stays per output)."""
    name: ClassVar[str] = 'outputs'

    def units(self, deck: Deck) -> Sequence[tuple[int, str]]:
        return [(i, word) for i, line in enumerate(deck.lines) if line.startswith('wrdata ')
                for word in line.split()[2:]]

    def remove(self, deck: Deck, units: Collection[tuple[int, str]]) -> Deck:
        lines = deck.lines
        for i, line in enumerate(lines):
            if line.startswith('wrdata '):
                words = line.split()
                kept = [w for w in words[2:] if (i, w) not in units] or words[2:3]
                lines[i] = ' '.join(words[:2] + kept)
        return deck.with_text('\n'.join(lines))


@dataclass(frozen=True)
class ShortenTransient(Simplification):
    """Halve the transient's stop time (keeping its step), down to 1% of the original."""
    name: ClassVar[str] = 'tstop'

    def candidates(self, deck: Deck) -> Iterator[Deck]:
        lines = deck.lines
        for i, line in enumerate(lines):
            words = line.split()
            if words and words[0].lower() == 'tran' and len(words) >= 3:
                step, stop = spice_number(words[1]), spice_number(words[2])
                if stop / 2 >= 10 * step:
                    lines[i] = ' '.join([words[0], words[1], eng(stop / 2), *words[3:]])
                    yield deck.with_text('\n'.join(lines))
                return


@dataclass(frozen=True)
class SimplifyNumbers(Simplification):
    """Round parameter values (w=0.47u) and R/C/L values to one significant digit."""
    name: ClassVar[str] = 'numbers'

    def candidates(self, deck: Deck) -> Iterator[Deck]:
        layout = Layout.of(deck)
        lines = deck.lines
        for i in sorted(set(layout.top_level) | set(layout.subcircuit_body())):
            words = lines[i].split()
            for j, word in enumerate(words):
                key, eq, value = word.rpartition('=')
                is_value = bool(eq) or (j == 3 and words[0][:1].lower() in 'rcl' and len(words) == 4)
                simpler = self._simpler(value if eq else word) if is_value else None
                if simpler is not None:
                    changed = list(lines)
                    changed[i] = ' '.join(words[:j] + [f'{key}={simpler}' if eq else simpler] + words[j + 1:])
                    yield deck.with_text('\n'.join(changed))

    @staticmethod
    def _simpler(token: str) -> Optional[str]:
        try:
            value = spice_number(token)
        except ValueError:
            return None  # an expression or a parameter name
        simpler = eng(float(f'{value:.1g}')) if value else '0'
        return simpler if simpler != token and len(simpler) <= len(token) else None


# Shorter transients first: every later trial runs faster.
REDUCTIONS: Sequence[Reduction] = (ShortenTransient(), RemoveLines(), RemoveSubcircuits(), RemoveSubcircuitLines(),
                                   DropOutputs(), SimplifyNumbers())


# --- The test --------------------------------------------------------------------------------

@dataclass
class FindingTest:
    """Whether a deck still shows a finding (one starting with `check`), judged as in a
    case, on a deck the reference accepts: ref must succeed without diagnostics the
    original deck did not have, and tight must succeed for approximate checks."""
    testbench: Testbench
    check: str  # finding prefix, e.g. 'exact:' or 'approx: crossing'
    workdir: Path
    baseline: tuple[str, ...] = ()  # ref's diagnostics on the original deck
    trials: int = 0
    _memo: dict[str, bool] = field(default_factory=dict, repr=False)

    @property
    def subject(self) -> str:
        return self.check.split(':')[0].strip()

    def configurations(self) -> list[str]:
        """What the check needs: ref, the configuration concerned, tight for accuracy."""
        names = ['ref']
        configuration = self.testbench.configurations.get(self.subject)
        if configuration is not None and configuration.name != 'ref':
            names.append(configuration.name)
            if configuration.claim is Claim.APPROXIMATE:
                names.append('tight')
        if self.subject == 'front-end':
            names.append('exact')
        return names

    def evidence(self, deck: Deck) -> oracles.Evidence:
        self.trials += 1
        return self.testbench.run(deck, self.workdir, self.configurations(), front_end=self.subject == 'front-end',
                                  repeat=self.subject == 'ref')

    def __call__(self, deck: Deck) -> bool:
        key = hashlib.sha256(deck.text.encode()).hexdigest()
        if key not in self._memo:
            self._memo[key] = self._reproduces(deck)
        return self._memo[key]

    def _reproduces(self, deck: Deck) -> bool:
        evidence = self.evidence(deck)
        ref, tight = evidence.reference, evidence.accuracy_reference
        if not ref.ok or any(line not in self.baseline for line in ref.diagnostics):
            return False
        if tight is not None and not tight.ok:
            return False
        return any(finding.startswith(self.check) for finding in oracles.judge(evidence).findings)


# --- The reducer -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Step:
    reduction: str
    deck: Deck

    @property
    def size(self) -> int:
        return len([line for line in self.deck.lines if line.strip()])


@dataclass(frozen=True)
class Reducer:
    reductions: Sequence[Reduction] = REDUCTIONS

    def steps(self, deck: Deck, test: DeckTest) -> Iterator[Step]:
        """Smaller and smaller decks that still pass `test`, until no reduction helps."""
        progress = True
        try:
            while progress:
                progress = False
                for reduction in self.reductions:
                    for smaller in reduction.steps(deck, test):
                        deck, progress = smaller, True
                        yield Step(reduction.name, deck)
        except BudgetExhausted:
            return

    def reduce(self, deck: Deck, test: DeckTest, on_step: Optional[Callable[[Step], None]] = None) -> Deck:
        for step in self.steps(deck, test):
            deck = step.deck
            if on_step is not None:
                on_step(step)
        return deck


# --- Reproducer script -----------------------------------------------------------------------

@dataclass(frozen=True)
class Reproducer:
    """A self-contained copy of the runner and a replayable execution manifest.

    NumPy and Python remain runtime dependencies. Simulator binaries and external model
    inputs stay at their recorded paths and are verified before running.
    """
    test: FindingTest
    finding: str
    tight_level: int = 0

    def files(self, deck: Deck) -> dict[str, str]:
        names = self.test.configurations()
        plan = ExecutionPlan(tuple(names), self.test.subject == 'front-end', self.test.subject == 'ref',
                             tight_level=self.tight_level if 'tight' in names else None)
        return bundle_files(deck, self.test.testbench, plan, self.test.check, self.finding)

    def script(self, deck: Deck) -> str:
        return REPLAY_SCRIPT


REPLAY_SCRIPT = ('#!/bin/sh\nset -eu\n'
                 'here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
                 'exec env PYTHONPATH="$here/runtime" python3 -P -m spicesmith.replay "$here"\n')


def bundle_files(deck: Deck, bench: Testbench, plan: ExecutionPlan, check: str = '', finding: str = '',
                 predicate: Optional[Mapping[str, Any]] = None) -> dict[str, str]:
    from .provenance import frozen_inputs

    settings = bench.settings
    files = {'reduced.sp': deck.text, **deck.files}
    for source in sorted((REPO / 'spicesmith').glob('*.py')):
        files['runtime/spicesmith/' + source.name] = source.read_text()
    binaries = {str(settings.simulators.path(bench.configurations[name].binary).resolve()) for name in plan.names}
    manifest = {
        'version': 1, 'check': check, 'finding': finding, 'predicate': predicate,
        'plan': plan.to_json(bench.configurations), 'cpus': bench.cpus,
        'settings': {'simulators': {key: str(getattr(settings.simulators, key))
                                   for key in ('reference', 'candidate')},
                     'timeout': settings.timeout, 'lock_wait': settings.lock_wait, 'threads': settings.threads},
        'binaries': {path: sha256(Path(path)) for path in sorted(binaries)},
        'inputs': dict(frozen_inputs(deck)),
        'files': {name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()},
    }
    files['repro.json'] = json.dumps(manifest, indent=2, allow_nan=False) + '\n'
    files['repro.sh'] = REPLAY_SCRIPT
    return files
