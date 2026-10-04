"""Serializable execution plans shared by experiments, reduction and reproducers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .config import CONFIGURATIONS, Claim, Configuration, binary_from_json, binary_json, simple_name
from .deck import TIGHT_LEVELS


def configuration_json(c: Configuration) -> dict[str, Any]:
    return {'name': c.name, 'binary': binary_json(c.binary), 'claim': c.claim.value, 'flags': dict(c.flags),
            'tight': c.tight, 'base': c.base, 'locks': list(c.locks)}


def configuration_from_json(data: Mapping[str, Any]) -> Configuration:
    return Configuration(data['name'], binary_from_json(data['binary']), Claim(data['claim']), data.get('flags', {}),
                         bool(data.get('tight')), data.get('base', 'ref'), tuple(data.get('locks', ())))


@dataclass(frozen=True)
class ExecutionPlan:
    names: tuple[str, ...] = ('ref', 'exact')
    front_end: bool = False
    repeat_reference: bool = False
    repetitions: int = 1
    tight_level: Optional[int] = None  # None: normal fallback; integer: replay the chosen tolerance

    def __post_init__(self) -> None:
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError('a plan needs distinct configuration names')
        if not all(simple_name(name) for name in self.names):
            raise ValueError('configuration names must be simple directory names')
        if self.repetitions < 1:
            raise ValueError('repetitions must be positive')
        if self.repeat_reference and 'ref' not in self.names:
            raise ValueError('a reference repeat requires ref in the plan')
        if self.front_end and not {'ref', 'exact'} <= set(self.names):
            raise ValueError('front-end comparison requires ref and exact in the plan')
        if self.tight_level is not None and self.tight_level not in range(len(TIGHT_LEVELS)):
            raise ValueError('invalid tight tolerance level')

    def to_json(self, definitions: Mapping[str, Configuration] = CONFIGURATIONS) -> dict[str, Any]:
        return {'version': 1, 'names': list(self.names), 'front_end': self.front_end,
                'repeat_reference': self.repeat_reference, 'repetitions': self.repetitions,
                'tight_level': self.tight_level,
                'configurations': {name: configuration_json(definitions[name]) for name in self.names}}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ExecutionPlan:
        if data.get('version') != 1:
            raise ValueError('unsupported execution plan version')
        return cls(tuple(data['names']), bool(data.get('front_end')), bool(data.get('repeat_reference')),
                   int(data.get('repetitions', 1)), data.get('tight_level'))


def definitions_from_json(data: Mapping[str, Any]) -> dict[str, Configuration]:
    return {name: configuration_from_json(value) for name, value in data['configurations'].items()}
