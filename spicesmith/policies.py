"""Optional interpretation of observations. Data quality is independent of accuracy."""
from __future__ import annotations

import dataclasses
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import numpy as np

from . import oracles
from .integrity import output_problems, read_output
from .oracles import Evidence, Verdict
from .simulation import RunResult

POLICIES = ('observe', 'exact', 'difference', 'legacy-accuracy')


@dataclass(frozen=True)
class Policy:
    name: str = 'observe'
    baseline: str = 'ref'
    parameters: Mapping[str, float] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError('unsupported policy version')
        allowed = {'absolute', 'relative', 'floor'} if self.name == 'difference' else set()
        if self.name in POLICIES and set(self.parameters) - allowed:
            raise ValueError(f'unsupported parameters for {self.name}')
        if any(not math.isfinite(v) or v < 0 for v in self.parameters.values()):
            raise ValueError('policy parameters must be finite and nonnegative')

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Policy:
        return cls(data['name'], data.get('baseline', 'ref'), data.get('parameters', {}), data.get('version', 1))


@dataclass(frozen=True)
class Assessment:
    policy: Policy
    state: str  # unassessed, inconclusive, pass, fail
    findings: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    checks: Mapping[str, str] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    selected: tuple[str, ...] = ()  # discrepancy triage, not accuracy findings

    def to_json(self) -> dict[str, Any]:
        requirements = ['ref', 'tight'] if self.policy.name == 'legacy-accuracy' else (
            [self.policy.baseline, 'one or more comparison configurations']
            if self.policy.name in ('exact', 'difference') else [])
        return {**dataclasses.asdict(self), 'prerequisites': {'configurations': requirements,
                'artifact_integrity': 'required for successful comparisons; recorded even without a verdict'}}


OBSERVE = Policy()


def rms(values: np.ndarray) -> float | None:
    with np.errstate(over='ignore', invalid='ignore'):
        magnitudes = np.abs(values)
        scale = float(magnitudes.max())
        if not math.isfinite(scale):
            return None  # metric unavailable; do not serialize Infinity
        return scale * float(np.sqrt(np.mean((magnitudes / scale) ** 2))) if scale else 0.0


def observe_run(evidence: Evidence, run: RunResult) -> dict[str, Any]:
    artifacts = {}
    problems = output_problems(evidence.deck, run)
    for output in evidence.deck.outputs:
        reading = read_output(evidence.deck, output, run.outputs.get(output))
        data, table = run.outputs.get(output), reading.table
        detail: dict[str, Any] = {'problem': reading.problem,
                                  'sha256': hashlib.sha256(data).hexdigest() if data is not None else None}
        if reading.usable and table is not None:
            detail.update(rows=len(table.scale), scale=table.scale_name,
                          interval=[float(table.scale[0]), float(table.scale[-1])], vectors=list(table.names))
            detail['signals'] = {name: {'minimum': float(np.min(table.column(name).real)),
                                         'maximum': float(np.max(table.column(name).real)),
                                         'rms': rms(table.column(name))}
                                 for name in table.names}
            pairs = []
            if table.is_transient and len(table.scale) > 2:
                for left, right in zip(table.names[:8], table.names[1:9], strict=False):
                    a, b = table.column(left).real, table.column(right).real
                    a = a / max(float(np.max(np.abs(a))), 1e-300)
                    b = b / max(float(np.max(np.abs(b))), 1e-300)
                    a, b = a - a.mean(), b - b.mean()
                    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
                    pairs.append({'vectors': [left, right], 'pearson': float(np.dot(a, b) / norm) if norm else None})
            detail['activity_correlation'] = {'scope': 'first eight adjacent observed-vector pairs, native samples',
                                               'pairs': pairs}
            if table.is_transient and len(table.scale) > 1:
                steps = np.diff(table.scale)
                detail['steps'] = {'minimum': float(steps.min()), 'median': float(np.median(steps)),
                                   'maximum': float(steps.max())}
        artifacts[output] = detail
    return {'outcome': run.outcome, 'complete': run.ok and not problems, 'seconds': run.seconds,
            'diagnostics': list(run.diagnostics),
            'counters': {k: v if math.isfinite(v) else None for k, v in run.stats.items()},
            'artifacts': artifacts}


def observations(evidence: Evidence) -> dict[str, Any]:
    return {name: {'samples': [observe_run(evidence, sample)
                               for sample in evidence.samples.get(name, (run,))]}
            for name, run in evidence.runs.items()}


def assess(evidence: Evidence, policy: Policy = OBSERVE,
           custom: Optional[Callable[[Evidence], Verdict]] = None) -> Assessment:
    """Assess saved evidence; observe/difference never issue an analog-accuracy verdict.

    A custom policy requires an explicit, versioned Policy and a caller-supplied evaluator.
    It is not loaded implicitly from a saved file.
    """
    reasons = []
    for name, run in evidence.runs.items():
        for index, sample in enumerate(evidence.samples.get(name, (run,))):
            label = name if index == 0 else f'{name} sample {index + 1}'
            if not sample.ok:
                reasons.append(f'{label}: {sample.outcome}')
            if any(not math.isfinite(v) for v in sample.stats.values()):
                reasons.append(f'{label}: non-finite solver counters')
            reasons += [f'{label}: {output}: {problem}'
                        for output, problem in output_problems(evidence.deck, sample).items()]
    if policy.name == 'observe':
        return Assessment(policy, 'unassessed', reasons=tuple(reasons))
    if policy.name in ('exact', 'difference', 'legacy-accuracy'):
        required = {'ref', 'tight'} if policy.name == 'legacy-accuracy' else {policy.baseline}
        reasons += [f'required configuration {name} unavailable' for name in sorted(required - evidence.runs.keys())]
        if required - evidence.runs.keys():
            return Assessment(policy, 'inconclusive', reasons=tuple(reasons))
    if policy.name == 'difference':
        return differences(evidence, policy, reasons)
    if policy.name == 'exact':
        if len(evidence.runs) < 2:
            reasons.append('exact comparison needs at least two configurations')
        baseline = evidence.runs[policy.baseline]
        checks = tuple(check for name, run in evidence.runs.items()
                       for index, sample in enumerate(evidence.samples.get(name, (run,)))
                       if name != policy.baseline or index > 0
                       for check in oracles.BitExactOracle()._compare(
                           name if index == 0 else f'{name}[{index + 1}]', sample, baseline))
        verdict = Verdict.combine((oracles.Judgement(checks), oracles.FrontEndOracle().judge(evidence),
                                   oracles.DeterminismOracle().judge(evidence)))
    elif policy.name == 'legacy-accuracy':
        if evidence.deck.vdd is None or (any(a.lower().startswith('tran ') for a in evidence.deck.analyses)
                                         and evidence.deck.tstop is None):
            reasons.append('legacy accuracy needs a declared supply and transient horizon')
        verdict = oracles.judge(evidence)
    elif custom is not None:
        verdict = custom(evidence)
    else:
        raise ValueError(f'unknown policy {policy.name!r}; custom policies need an explicit evaluator')
    reasons += list(verdict.inconclusive)
    checked = any(outcome is oracles.Outcome.PASS for outcome in verdict.checks.values())
    state = 'fail' if verdict.findings else ('inconclusive' if reasons or not checked else 'pass')
    return Assessment(policy, state, verdict.findings, tuple(dict.fromkeys(reasons)),
                      {k: v.value for k, v in verdict.checks.items()}, verdict.metrics)


def differences(evidence: Evidence, policy: Policy, reasons: list[str]) -> Assessment:
    metrics: dict[str, Any] = {}
    selected = []
    baseline = evidence.runs[policy.baseline]
    if len(evidence.runs) < 2:
        reasons.append('difference comparison needs at least two configurations')
    for name, run in evidence.runs.items():
        if name == policy.baseline:
            continue
        metrics[name] = {'outcome_changed': run.outcome != baseline.outcome,
                         'diagnostics_changed': run.diagnostics != baseline.diagnostics,
                         'counters': {k: {'baseline': finite(baseline.stats.get(k)),
                                          'candidate': finite(run.stats.get(k))}
                                      for k in sorted(set(run.stats) | set(baseline.stats))}, 'outputs': {}}
        for output in evidence.deck.outputs:
            ref = read_output(evidence.deck, output, baseline.outputs.get(output))
            got = read_output(evidence.deck, output, run.outputs.get(output))
            if not (ref.usable and got.usable and ref.table is not None and got.table is not None):
                continue
            mismatch = ref.table.mismatch(got.table)
            if mismatch:
                reasons.append(f'{name}: {output}: {mismatch}')
                continue
            reference, candidate = ref.table, got.table
            vectors = {}
            for vector in reference.names:
                expected, values = reference.column(vector), candidate.column(vector)
                # Compare only the common covered interval: never extrapolate missing samples.
                lo, hi = max(reference.scale.min(), candidate.scale.min()), min(reference.scale.max(),
                                                                               candidate.scale.max())
                mask = (reference.scale >= lo) & (reference.scale <= hi)
                grid = reference.scale[mask]
                order = np.argsort(candidate.scale)
                interpolated = np.interp(grid, candidate.scale[order], values.real[order])
                if np.iscomplexobj(values):
                    interpolated = interpolated + 1j * np.interp(grid, candidate.scale[order], values.imag[order])
                with np.errstate(over='ignore', invalid='ignore'):
                    delta = np.abs(interpolated - expected[mask])
                if not len(delta):
                    reasons.append(f'{name}: {output}: no common samples')
                    continue
                if not np.all(np.isfinite(delta)):
                    reasons.append(f'{name}: {output}: {vector}: comparison arithmetic overflow')
                    continue
                absolute = float(delta.max())
                normalizer = max(float(np.max(np.abs(expected[mask]))), policy.parameters.get('floor', 1e-12), 1e-30)
                relative = absolute / normalizer
                if not math.isfinite(relative):
                    reasons.append(f'{name}: {output}: {vector}: relative metric overflow')
                    continue
                vectors[vector] = {'max_absolute': absolute, 'rms': rms(delta),
                                   'relative_to_baseline_peak': relative, 'samples': int(len(delta)),
                                   'interval': [float(grid.min()), float(grid.max())]}
                if (absolute > policy.parameters.get('absolute', 0.0)
                        and relative > policy.parameters.get('relative', 0.0)):
                    selected.append(f'difference: {name} {output} {vector}')
            metrics[name]['outputs'][output] = vectors
    metrics['method'] = {'grid': 'baseline samples in common interval', 'interpolation': 'linear, no extrapolation',
                         'repetitions': 'primary run per configuration; all timing samples integrity-checked',
                         'normalization': 'baseline vector peak with declared floor',
                         'scope': evidence.deck.output_requests}
    return Assessment(policy, 'inconclusive' if reasons else 'unassessed', reasons=tuple(dict.fromkeys(reasons)),
                      metrics=metrics, selected=tuple(selected))


def finite(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None
