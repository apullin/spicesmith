"""Serializable reduction predicates: selected observations, not universal accuracy."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .benchmark import comparison
from .circuit import ARITY
from .deck import Deck
from .harness import Testbench
from .oracles import Evidence
from .plans import ExecutionPlan
from .policies import Policy, assess, observations
from .population import numerical
from .reduce import Layout


@dataclass(frozen=True)
class PredicateResult:
    state: str  # matched, not-matched, unresolved
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class Predicate:
    kind: str
    configuration: str = 'exact'
    baseline: str = 'ref'
    outcome: str | int = 'timeout'
    threshold: float = 0.0
    relative: float = 0.0
    finding: str = ''
    minimum_devices: int = 0
    minimum_transistors: int = 0
    connections: tuple[tuple[str, str], ...] = ()  # direct primitive terminals, not reachability via ground
    regime: str | None = None
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1 or self.kind not in ('outcome', 'exact', 'difference', 'structure', 'slowdown'):
            raise ValueError('unsupported predicate')
        if any(not math.isfinite(v) or v < 0 for v in (self.threshold, self.relative)):
            raise ValueError('predicate thresholds must be finite and nonnegative')
        if self.minimum_devices < 0 or self.minimum_transistors < 0:
            raise ValueError('structural bounds must be nonnegative')
        if self.regime not in (None, 'active', 'quiescent', 'unclassified'):
            raise ValueError('unsupported observed-regime guard')
        if any(len(c) != 2 for c in self.connections):
            raise ValueError('connection guards need pairs of net names')

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Predicate:
        return cls(**{**data, 'connections': tuple(tuple(c) for c in data.get('connections', ()))})

    def structure_matches(self, deck: Deck) -> bool:
        devices = [deck.lines[i].split() for i in Layout.of(deck).top_level]
        devices = [d for d in devices if d and d[0][0].upper() in (*ARITY, 'X')]
        transistors = sum(d[0][0].upper() == 'M' or d[0][0].upper() == 'X' and
                          any(t in ('sg13_lv_nmos', 'sg13_lv_pmos') for t in d) for d in devices)
        if len(devices) < self.minimum_devices or transistors < self.minimum_transistors:
            return False
        terminals = [set(d[1:1 + ARITY[d[0][0].upper()]]) for d in devices if d[0][0].upper() in ARITY]
        return all(any(set(pair) <= nodes for nodes in terminals) for pair in self.connections)

    def evaluate(self, evidence: Evidence) -> PredicateResult:
        if not self.structure_matches(evidence.deck):
            return PredicateResult('not-matched', ('structural guard not retained',))
        observed = observations(evidence)
        if self.regime and numerical(observed)['regime'] != self.regime:
            return PredicateResult('not-matched', ('observed-regime guard not retained',))
        run = evidence.runs.get(self.configuration)
        if self.kind == 'outcome':
            if run is None:
                return PredicateResult('unresolved', ('selected configuration unavailable',))
            return PredicateResult('matched' if run.outcome == self.outcome else 'not-matched')
        if self.kind == 'structure':
            complete = numerical(observed)['completed']
            return PredicateResult('matched' if complete else 'unresolved',
                                   () if complete else ('structural candidate did not complete its execution plan',))
        if self.kind == 'slowdown':
            result = comparison(observed, self.baseline)['comparisons'].get(self.configuration, {})
            ratio = result.get('speedup')
            if ratio is None:
                return PredicateResult('unresolved', ('slowdown requires >=3 matched completed timings',))
            return PredicateResult('matched' if 1 / ratio >= self.threshold else 'not-matched')
        policy = Policy(self.kind, self.baseline,
                        {'absolute': self.threshold, 'relative': self.relative} if self.kind == 'difference' else {})
        result = assess(evidence, policy)
        if result.state == 'inconclusive' or result.reasons:
            return PredicateResult('unresolved', result.reasons)
        findings = result.selected if self.kind == 'difference' else result.findings
        matching = [f for f in findings if (f.startswith(self.finding) if self.finding else
                                           (f.startswith(f'difference: {self.configuration} ')
                                            if self.kind == 'difference' else
                                            f.startswith(self.configuration + ':')))]
        return PredicateResult('matched' if matching else 'not-matched', tuple(matching))


@dataclass
class PredicateTest:
    testbench: Testbench
    plan: ExecutionPlan
    predicate: Predicate
    workdir: Path
    trials: int = 0
    results: list[Mapping[str, Any]] = field(default_factory=list)

    def evidence(self, deck: Deck) -> Evidence:
        self.trials += 1
        return self.testbench.run(deck, self.workdir, plan=self.plan)

    def __call__(self, deck: Deck) -> bool:
        if not self.predicate.structure_matches(deck):
            self.results.append({'state': 'not-matched', 'reasons': ['structural guard not retained']})
            return False
        result = self.predicate.evaluate(self.evidence(deck))
        self.results.append(dataclasses.asdict(result))
        return result.state == 'matched'
