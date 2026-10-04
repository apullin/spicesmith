"""Versioned circuit graph, independent of simulation and interpretation.

The first IR deliberately admits a small, typed SPICE subset. Stable device/motif
identities and explicit conditioning/coupling survive lowering and serialization.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

from .deck import Deck

VERSION = 1
NAME = re.compile(r'[a-zA-Z][a-zA-Z0-9_]*\Z')
ARITY = {'R': 2, 'C': 2, 'L': 2, 'V': 2, 'I': 2, 'D': 2, 'M': 4, 'E': 4}
MODELS = {'n': '.model n nmos level=54 version=4.8',
          'p': '.model p pmos level=54 version=4.8',
          'diode': '.model diode d is=1e-14 n=1.05 rs=2 cjo=5f'}


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Device:
    id: str
    kind: str
    nodes: tuple[str, ...]
    value: float = 0.0
    model: str = ''
    parameters: Mapping[str, float] = field(default_factory=dict)
    stimulus: Mapping[str, Any] = field(default_factory=dict)
    owner: str = 'shared'
    role: str = 'component'

    def lower(self, model_family: str) -> str:
        kind, model = self.kind, self.model
        head = f'{kind}{self.id} ' + ' '.join(self.nodes)
        if self.kind in ('M', 'D'):
            return head + ' ' + model + ''.join(f' {k}={v:.12g}' for k, v in sorted(self.parameters.items()))
        if self.stimulus:
            s = self.stimulus
            if s['kind'] == 'sin':
                tail = 'SIN(' + ' '.join(f'{s[k]:.12g}' for k in
                                         ('offset', 'amplitude', 'frequency', 'delay', 'damping', 'phase')) + ')'
            else:
                tail = 'PULSE(' + ' '.join(f'{s[k]:.12g}' for k in
                                           ('low', 'high', 'delay', 'rise', 'fall', 'width', 'period')) + ')'
            return head + ' ' + tail
        return head + f' {self.value:.12g}'


@dataclass(frozen=True)
class Motif:
    id: str
    kind: str
    ports: Mapping[str, str]
    shape_seed: int
    parameter_seed: int
    source_seed: int
    intended_regime: str
    implementation: str = 'transistor/passive'
    region: int = 0


@dataclass(frozen=True)
class Analysis:
    command: str
    output: str
    vectors: tuple[str, ...]


@dataclass(frozen=True)
class Circuit:
    seed: int
    nets: tuple[str, ...]
    devices: tuple[Device, ...]
    motifs: tuple[Motif, ...]
    analyses: tuple[Analysis, ...]
    profile: Mapping[str, Any]
    model_family: str = 'bsim4'
    models: Mapping[str, str] = field(default_factory=lambda: MODELS.copy())
    initial_conditions: Mapping[str, float] = field(default_factory=dict)
    history: tuple[Mapping[str, Any], ...] = ()
    version: int = VERSION

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Circuit:
        result = cls(data['seed'], tuple(data['nets']),
                     tuple(Device(**{**d, 'nodes': tuple(d['nodes'])}) for d in data['devices']),
                     tuple(Motif(**m) for m in data['motifs']),
                     tuple(Analysis(**{**a, 'vectors': tuple(a['vectors'])}) for a in data['analyses']),
                     data['profile'], data['model_family'], data['models'], data.get('initial_conditions', {}),
                     tuple(data.get('history', ())), data['version'])
        result.validate()
        return result

    def validate(self) -> None:
        if self.version != VERSION or self.model_family != 'bsim4':
            raise ValueError('unsupported circuit version or model family')
        nets = set(self.nets)
        if ('0' not in nets or len({n.lower() for n in nets}) != len(self.nets)
                or any(n != '0' and not NAME.fullmatch(n) for n in nets)):
            raise ValueError('nets must be distinct valid identifiers including ground')
        if len(nets) > self.profile.get('max_nets', 200000):
            raise ValueError('net budget exceeded')
        if len(self.devices) > self.profile.get('max_devices', 200000):
            raise ValueError('device budget exceeded')
        owners = {'shared', *(m.id for m in self.motifs)}
        if len(owners) != len(self.motifs) + 1 or any(not NAME.fullmatch(m.id) for m in self.motifs):
            raise ValueError('motif identifiers must be distinct')
        ids: set[str] = set()
        ideal_pairs: set[tuple[str, ...]] = set()
        for d in self.devices:
            if not NAME.fullmatch(d.id) or d.id.lower() in ids:
                raise ValueError('device identifiers must be distinct (case-insensitive)')
            ids.add(d.id.lower())
            if d.kind not in ARITY or len(d.nodes) != ARITY[d.kind] or not set(d.nodes) <= nets:
                raise ValueError(f'{d.id}: invalid device terminals')
            if d.owner not in owners or not math.isfinite(d.value):
                raise ValueError(f'{d.id}: invalid owner or value')
            if any(not NAME.fullmatch(k) or not math.isfinite(v) for k, v in d.parameters.items()):
                raise ValueError(f'{d.id}: invalid parameters')
            if d.kind in ('R', 'C', 'L') and d.value <= 0:
                raise ValueError(f'{d.id}: passive values must be positive')
            if d.kind in ('M', 'D') and d.model not in self.models:
                raise ValueError(f'{d.id}: unavailable model')
            if d.kind == 'M' and (d.model not in ('n', 'p') or set(d.parameters) != {'w', 'l'}
                                 or any(v <= 0 for v in d.parameters.values())):
                raise ValueError(f'{d.id}: invalid transistor geometry')
            if d.kind == 'V':
                pair = tuple(sorted(d.nodes))
                if pair in ideal_pairs or len(set(pair)) != 2:
                    raise ValueError(f'{d.id}: parallel/shorted ideal voltage sources')
                ideal_pairs.add(pair)
            if d.stimulus:
                keys = ({'offset', 'amplitude', 'frequency', 'delay', 'damping', 'phase'}
                        if d.stimulus.get('kind') == 'sin' else
                        {'low', 'high', 'delay', 'rise', 'fall', 'width', 'period'})
                if (d.kind not in ('V', 'I') or d.stimulus.get('kind') not in ('sin', 'pulse')
                        or set(d.stimulus) != keys | {'kind'}
                        or any(not math.isfinite(d.stimulus[k]) for k in keys)):
                    raise ValueError(f'{d.id}: invalid excitation')
        for motif in self.motifs:
            if not set(motif.ports.values()) <= nets:
                raise ValueError(f'{motif.id}: unresolved port')
        if not set(self.initial_conditions) <= nets or any(not math.isfinite(v)
                                                          for v in self.initial_conditions.values()):
            raise ValueError('invalid initial conditions')
        outputs: set[str] = set()
        observed: set[str] = set()
        for analysis in self.analyses:
            # The generated subset has one transient contract; other Deck inputs remain supported by the harness.
            words = analysis.command.split()
            if len(words) != 5 or words[0] != 'tran' or words[3] != '0':
                raise ValueError('unsupported IR analysis')
            numbers = [float(words[i]) for i in (1, 2, 4)]
            if any(not math.isfinite(v) or v <= 0 for v in numbers) or numbers[2] > numbers[1]:
                raise ValueError('invalid transient interval')
            if (not re.fullmatch(r'[a-zA-Z0-9_-]+\.txt', analysis.output) or analysis.output in outputs
                    or not analysis.vectors or len(set(analysis.vectors)) != len(analysis.vectors)):
                raise ValueError('invalid observation contract')
            outputs.add(analysis.output)
            for vector in analysis.vectors:
                if not vector.startswith('v(') or not vector.endswith(')') or vector[2:-1] not in nets:
                    raise ValueError('unresolved observation terminal')
                observed.add(vector[2:-1])
        if not outputs or len(observed) > self.profile.get('max_observed', 64):
            raise ValueError('observation budget exceeded or no observations')

    def lower(self) -> Deck:
        self.validate()
        lines = [f'spicesmith structural generator {VERSION} seed {self.seed}',
                 f'.param vsupply={self.profile["vdd"]:.12g}',
                 '.option reltol=1e-3 abstol=1e-13 vntol=1e-6 method=trap']
        lines += list(self.models.values())
        lines += [d.lower(self.model_family) for d in self.devices]
        lines += [f'.ic v({net})={value:.12g}' for net, value in self.initial_conditions.items()]
        for a in self.analyses:
            lines += ['.' + a.command, '* spicesmith-output ' + a.output + ' ' + ' '.join(a.vectors)]
        lines += ['.save ' + ' '.join(dict.fromkeys(v for a in self.analyses for v in a.vectors))]
        return Deck('\n'.join(lines + ['.end', '']))

    def structural(self) -> dict[str, Any]:
        """Linear-time accounting; all graph connectivity is structural, not a regime claim."""
        degree: Counter[str] = Counter()
        parent = {n: n for n in self.nets}

        def root(net: str) -> str:
            while parent[net] != net:
                parent[net] = parent[parent[net]]
                net = parent[net]
            return net

        for d in self.devices:
            degree.update(set(d.nodes))
            for n in d.nodes[1:]:
                parent[root(n)] = root(d.nodes[0])
        observed = {v[2:-1] for a in self.analyses for v in a.vectors}
        active_nets = set(self.nets) - {'0'}
        owned: dict[str, list[Device]] = {m.id: [] for m in self.motifs}
        ranges: dict[str, list[float]] = {}
        for d in self.devices:
            if d.owner in owned:
                owned[d.owner].append(d)
            if d.kind in ('R', 'C', 'L', 'I'):
                ranges.setdefault(d.kind, []).append(d.value)
            for k, v in d.parameters.items():
                ranges.setdefault(f'{d.kind}.{k}', []).append(v)
        shapes, parameters = set(), set()
        for m in self.motifs:
            local = {v: k for k, v in sorted(m.ports.items())}
            shape, values = [], []
            for d in owned[m.id]:
                if d.role == 'coupling':
                    continue  # cross-motif connectivity is counted separately from the scaffold
                for n in d.nodes:
                    if n not in local:
                        local[n] = f'local{len(local)}'
                shape.append((d.kind, tuple(local[n] for n in d.nodes), d.model, d.role))
                values.append((d.kind, d.value, dict(d.parameters)))
            shapes.add(fingerprint(shape))
            parameters.add(fingerprint(values))
        return {'devices': len(self.devices), 'transistors': sum(d.kind == 'M' for d in self.devices),
                'nets': len(self.nets), 'families': dict(Counter(d.kind for d in self.devices)),
                'motifs': dict(Counter(m.kind for m in self.motifs)),
                'roles': dict(Counter(d.role for d in self.devices)),
                'connected_components': len({root(n) for n in self.nets}),
                'degree_histogram': {str(k): v for k, v in sorted(Counter(degree[n] for n in self.nets).items())},
                'parameter_ranges': {k: {'minimum': min(v), 'maximum': max(v)} for k, v in ranges.items()},
                'distinct_shapes': len(shapes), 'distinct_parameter_sets': len(parameters),
                'distinct_topology_draws': len({(m.kind, m.shape_seed) for m in self.motifs}),
                'distinct_parameter_draws': len({(m.kind, m.parameter_seed) for m in self.motifs}),
                'distinct_source_draws': len({m.source_seed for m in self.motifs}),
                'hierarchy': {'representation': 'flat devices with motif/region ownership',
                              'regions': len({m.region for m in self.motifs})},
                'observation_scope': {'observed_nets': len(observed), 'total_non_ground_nets': len(active_nets),
                                      'unobserved_nets': len(active_nets - observed)},
                'intended_regimes': dict(Counter(m.intended_regime for m in self.motifs)),
                'model_family': self.model_family,
                'controlled_source_scaffolds': sum(m.implementation == 'controlled-source' for m in self.motifs)}
