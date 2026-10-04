"""The regression corpus: minimized decks from past findings, run on
every new candidate before it is promoted.

Each entry is a directory with deck.sp and entry.json. An entry that records a fixed bug must
pass every check; an accepted weakness (a known finding of an approximate change) may show its
own finding again but nothing else. Acceptance rule: bit-exact claims need zero
diffs, approximate changes need no accuracy regression.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional

from . import oracles
from .config import Claim
from .deck import Deck
from .harness import Testbench
from .oracles import Outcome, Verdict

CORPUS = Path(__file__).resolve().parent.parent / 'corpus'
_MEASURED = re.compile(r'(?<![\w.(-])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?(?![\w.])')


def normalized(finding: str) -> str:
    """A finding with its measured numbers abstracted ('6.43 ps' -> '# ps') and everything
    that identifies it kept: configuration, check, output file, node."""
    return _MEASURED.sub('#', finding)


class Expectation(Enum):
    PASS = 'pass'  # a fixed bug: every check passes
    KNOWN = 'known'  # an accepted weakness: its own finding may reproduce, nothing else may fail


class Status(Enum):
    PASS = 'pass'
    KNOWN = 'known'  # the accepted finding reproduced, as expected
    FIXED = 'fixed'  # an accepted finding no longer reproduces: update the entry
    REGRESSION = 'REGRESSION'
    INCONCLUSIVE = 'inconclusive'
    SKIPPED = 'skipped'


@dataclass(frozen=True)
class Entry:
    name: str
    deck: Deck
    check: str  # prefix of the entry's finding, e.g. 'approx: mid-rail'
    expect: Expectation
    finding: str  # the finding as first reported
    origin: str  # where it came from: batch, seed, candidate
    notes: str = ''
    requires: tuple[str, ...] = ()  # configurations it needs beyond the built-in ones
    expected: tuple[str, ...] = ()  # an accepted weakness's findings, normalized; default: its finding

    @classmethod
    def load(cls, directory: Path) -> Entry:
        meta = json.loads((directory / 'entry.json').read_text())
        return cls(directory.name, Deck.load(directory), meta['check'],
                   Expectation(meta['expect']), meta['finding'], meta['origin'], meta.get('notes', ''),
                   tuple(meta.get('requires', ())), tuple(meta.get('expected', ())))

    @property
    def accepted(self) -> frozenset[str]:
        """The normalized findings an accepted weakness may show."""
        return frozenset(self.expected or (normalized(self.finding),))

    def save(self, corpus: Path) -> Path:
        directory = corpus / self.name
        self.deck.save(directory)
        record: dict[str, object] = {'check': self.check, 'expect': self.expect.value, 'finding': self.finding,
                                     'origin': self.origin, 'notes': self.notes}
        if self.requires:
            record['requires'] = list(self.requires)
        if self.expected:
            record['expected'] = list(self.expected)
        (directory / 'entry.json').write_text(json.dumps(record, indent=1) + '\n')
        return directory


@dataclass(frozen=True)
class EntryResult:
    entry: Entry
    verdict: Optional[Verdict]  # None when skipped
    required: tuple[str, ...] = ()

    @property
    def status(self) -> Status:
        if self.verdict is None:
            return Status.SKIPPED
        if self.unexpected:
            return Status.REGRESSION
        if self.unresolved:
            return Status.INCONCLUSIVE
        if self.entry.expect is Expectation.PASS:
            return Status.PASS
        return Status.KNOWN if self.verdict.findings else Status.FIXED

    @property
    def unresolved(self) -> tuple[str, ...]:
        if self.verdict is None:
            return ()
        reasons = list(self.verdict.inconclusive)
        if not self.verdict.checks:
            reasons.append('no policy checks were evaluated')
        reasons += [f'required check {key} unavailable' for key in self.required
                    if self.verdict.checks.get(key) in (None, Outcome.SKIP)]
        reasons += [note for note in self.verdict.notes if oracles.BUSY_NOTE in note]
        return tuple(dict.fromkeys(reasons))

    @property
    def unexpected(self) -> tuple[str, ...]:
        """The findings that make this a regression: any, for a fixed bug; for an accepted
        weakness, every finding that is not one of its own (another node, output or check
        counts, even with the same prefix)."""
        if self.verdict is None:
            return ()
        if self.entry.expect is Expectation.PASS:
            return self.verdict.findings
        return tuple(f for f in self.verdict.findings if normalized(f) not in self.entry.accepted)


@dataclass(frozen=True)
class Corpus:
    directory: Path = CORPUS

    def entries(self) -> list[Entry]:
        return [Entry.load(d) for d in sorted(self.directory.iterdir()) if (d / 'entry.json').exists()]

    def run(self, testbench: Testbench, workdir: Path) -> Iterator[EntryResult]:
        """Every entry in every configuration of the testbench's settings, judged by all oracles;
        an entry that requires a configuration the settings lack is skipped."""
        for entry in self.entries():
            if not set(entry.requires) <= testbench.configurations.keys():
                yield EntryResult(entry, None)
                continue
            shutil.rmtree(workdir / entry.name, ignore_errors=True)  # nothing from an earlier run counts
            evidence = testbench.run(entry.deck, workdir / entry.name, repeat=True)
            required = ['ref:determinism']
            if testbench.settings.front_end:
                required.append('front-end:identity')
            for name in evidence.runs:
                if evidence.definitions[name].claim in (Claim.EXACT, Claim.APPROXIMATE):
                    required.extend((f'{name}:status', f'{name}:outputs'))
            yield EntryResult(entry, oracles.judge(evidence), tuple(required))
