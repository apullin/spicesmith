"""Which flags a finding needs: single-flag sufficiency and omission-based necessity.

A configuration's flags are its base configuration's (claimed bit-exact) plus a few more.
For each extra flag, the finding is checked with that flag alone on top of the base, and with
every extra flag but that one: a flag that reproduces the finding alone is sufficient, one
without which the finding goes away is necessary.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from . import oracles
from .config import Configuration
from .deck import Deck
from .harness import Testbench


@dataclass(frozen=True)
class Variant:
    label: str  # 'only FLAG' or 'without FLAG'
    flag: str
    flags: Mapping[str, str]


@dataclass(frozen=True)
class Trial:
    variant: Variant
    reproduces: bool
    findings: tuple[str, ...]


@dataclass(frozen=True)
class FlagSplit:
    """Splits the flags of the configuration a finding concerns."""
    testbench: Testbench
    check: str  # finding prefix, e.g. 'approx: crossing'
    base: str = 'exact'  # configuration whose flags the split starts from

    @property
    def subject(self) -> Configuration:
        return self.testbench.configurations[self.check.split(':')[0].strip()]

    def extra_flags(self) -> dict[str, str]:
        base = self.testbench.configurations[self.base].flags
        return {k: v for k, v in self.subject.flags.items() if base.get(k) != v}

    def variants(self) -> Iterator[Variant]:
        base, extra = self.testbench.configurations[self.base].flags, self.extra_flags()
        for flag, value in extra.items():
            yield Variant(f'only {flag}', flag, {**base, flag: value})
            yield Variant(f'without {flag}', flag, {**base, **{k: v for k, v in extra.items() if k != flag}})

    def trials(self, deck: Deck, workdir: Path) -> Iterator[Trial]:
        """Run every variant, as the subject configuration with other flags."""
        subject = self.subject
        names = ['ref', subject.name] + (['tight'] if subject.claim.name == 'APPROXIMATE' else [])
        for i, variant in enumerate(self.variants()):
            settings = dataclasses.replace(self.testbench.settings, configurations={
                **self.testbench.configurations, subject.name: dataclasses.replace(subject, flags=variant.flags)})
            evidence = dataclasses.replace(self.testbench, settings=settings).run(
                deck, workdir / f'variant-{i:02d}', names, front_end=False)
            findings = tuple(f for f in oracles.judge(evidence).findings if f.startswith(self.check))
            yield Trial(variant, bool(findings), findings)

    @staticmethod
    def conclusion(trials: Sequence[Trial]) -> str:
        sufficient = [t.variant.flag for t in trials if t.variant.label.startswith('only') and t.reproduces]
        necessary = [t.variant.flag for t in trials if t.variant.label.startswith('without') and not t.reproduces]
        parts = []
        if sufficient:
            parts.append('sufficient alone: ' + ', '.join(sufficient))
        if necessary:
            parts.append('necessary: ' + ', '.join(necessary))
        return '; '.join(parts) or 'no single flag decides it (an interaction, or not reproducible)'
