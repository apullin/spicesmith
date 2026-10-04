"""Caller-selected simulators and execution configurations."""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

REFERENCE = Path('ngspice')
CANDIDATE = Path('ngspice')
THREADS = 1
CPUS = None  # no affinity unless requested by the caller
TIMEOUT = 300.0


def executable(path: Path) -> Path:
    """Resolve a bare command using the caller's PATH, or validate an explicit path."""
    found = shutil.which(str(path)) if path.parent == Path('.') and not path.is_file() else None
    resolved = Path(found or path).absolute()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f'simulator executable not found: {path}; pass --reference/--candidate explicitly')
    return resolved


class Claim(Enum):
    """What the oracles hold a configuration to."""
    REFERENCE = 'reference'
    EXACT = 'exact'  # byte-identical to the reference
    APPROXIMATE = 'approximate'  # no less accurate than the reference, against the accuracy reference
    ACCURACY = 'accuracy'  # the tight-tolerance accuracy reference


class Binary(Enum):
    """The simulator roles a configuration can run (an extra configuration may name a binary)."""
    REFERENCE = 'reference'
    CANDIDATE = 'candidate'


@dataclass(frozen=True)
class Simulators:
    """The binaries of the reference and candidate roles."""
    reference: Path = REFERENCE
    candidate: Path = CANDIDATE

    def path(self, binary: Binary | Path) -> Path:
        """The binary a configuration runs: its role's, or the fixed one it names."""
        if isinstance(binary, Path):
            return executable(binary)
        return executable({Binary.REFERENCE: self.reference, Binary.CANDIDATE: self.candidate}[binary])


@dataclass(frozen=True)
class Configuration:
    """One way of running a deck: a binary, its environment flags, and the claim it is held to."""
    name: str
    binary: Binary | Path  # a role, resolved through Simulators, or a fixed binary
    claim: Claim
    flags: Mapping[str, str] = field(default_factory=dict)
    tight: bool = False  # run the deck at the accuracy reference's tolerances
    # The configuration this one adds its own approximation to: the report measures the
    # change in accuracy against it (the oracles hold every approximate one to ref).
    base: str = 'ref'
    # Cooperative lock files every run holds (flock -n), the first taken first. Runs that hold
    # locks go one at a time in this process; a lock held elsewhere makes a run 'busy'.
    locks: tuple[str, ...] = ()

    def flags_for(self, has_small_signal_analysis: bool) -> Mapping[str, str]:
        """Caller flags are passed unchanged; no knowledge of private simulator switches."""
        return self.flags


CONFIGURATIONS: Mapping[str, Configuration] = {c.name: c for c in (
    Configuration('ref', Binary.REFERENCE, Claim.REFERENCE),
    Configuration('exact', Binary.CANDIDATE, Claim.EXACT),
    Configuration('approx', Binary.CANDIDATE, Claim.APPROXIMATE),
    Configuration('tight', Binary.REFERENCE, Claim.ACCURACY, tight=True),
)}


EXTRA_KEYS = frozenset({'name', 'binary', 'claim', 'extends', 'flags', 'locks'})


def simple_name(name: str) -> bool:
    """Usable as a directory name of its own, as configuration names are."""
    return name not in ('', '.', '..') and '/' not in name and '\\' not in name


def binary_json(binary: Binary | Path) -> str:
    """A configuration's binary as recorded: the role's name, or the fixed binary's path."""
    return binary.value if isinstance(binary, Binary) else str(binary)


def binary_from_json(text: str) -> Binary | Path:
    return Binary(text) if text in {b.value for b in Binary} else Path(text)


def load_extra_configurations(path: Path) -> dict[str, Configuration]:
    """Configurations defined outside SpiceSmith (--extra-configurations, see docs/USAGE.md):
    a JSON list of {"name", "binary", "claim", "extends", "flags", "locks"}. A binary is a
    role ("reference", "candidate") or a simulator path, relative to the file; the claim is
    "exact" or "approximate"; the flags add to those of the configuration it extends."""
    definitions = json.loads(Path(path).read_text())
    if not isinstance(definitions, list):
        raise ValueError(f'{path}: expected a JSON list of configurations')
    extra: dict[str, Configuration] = {}
    for item in definitions:
        name = item.get('name') if isinstance(item, dict) else None
        where = f'{path}: configuration {name!r}'
        if not isinstance(name, str) or not simple_name(name) or name in CONFIGURATIONS or name in extra:
            raise ValueError(f'{where} needs a new name that is usable as a directory name')
        unknown = sorted(set(item) - EXTRA_KEYS)
        if unknown:
            raise ValueError(f'{where}: unknown keys {unknown}')
        if item.get('claim') not in (Claim.EXACT.value, Claim.APPROXIMATE.value):
            raise ValueError(f'{where}: the claim must be "exact" or "approximate"')
        known = {**CONFIGURATIONS, **extra}
        parent = item.get('extends')
        if parent is not None and parent not in known:
            raise ValueError(f'{where} extends an unknown configuration {parent!r}')
        flags, locks = item.get('flags', {}), item.get('locks', [])
        if not (isinstance(item.get('binary'), str) and isinstance(flags, dict) and isinstance(locks, list)
                and all(isinstance(value, str) for value in [*flags.values(), *locks])):
            raise ValueError(f'{where}: the binary, flag values and locks must be strings')
        binary = binary_from_json(item['binary'])
        if isinstance(binary, Path):
            binary = Path(path).parent.absolute() / binary
        extra[name] = Configuration(name, binary, Claim(item['claim']),
                                    {**(known[parent].flags if parent else {}), **flags}, locks=tuple(locks))
    return extra


@dataclass(frozen=True)
class CpuSet:
    """CPUs one job is pinned to (taskset -c)."""
    cpus: tuple[int, ...]

    def __str__(self) -> str:
        first, last = self.cpus[0], self.cpus[-1]
        if len(self.cpus) > 1 and self.cpus == tuple(range(first, last + 1)):
            return f'{first}-{last}'
        return ','.join(map(str, self.cpus))

    @staticmethod
    def parse(text: str) -> tuple[int, ...]:
        """CPU numbers of a list such as '8-31' or '8-15,24-31'."""
        cpus: list[int] = []
        for part in text.split(','):
            low, _, high = part.partition('-')
            cpus += range(int(low), int(high or low) + 1)
        return tuple(cpus)

    @classmethod
    def split(cls, text: str, threads: int = THREADS) -> Sequence[CpuSet]:
        """Disjoint sets of `threads` CPUs each; leftover CPUs stay unused."""
        cpus = cls.parse(text)
        return [cls(cpus[i:i + threads]) for i in range(0, len(cpus) - threads + 1, threads)]
