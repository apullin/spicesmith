"""Controlled structural generation, including analog hints and macro-scale populations.

Named recipes are distributions, not functional specifications.
No simulator or accuracy oracle participates in admission.
"""
from __future__ import annotations

import dataclasses
import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping

from .circuit import Analysis, Circuit, Device, Motif, fingerprint

ANALOG = ('bias', 'gain', 'feedback', 'rc', 'rlc', 'nonlinear', 'primitive')
HINTS = (*ANALOG, 'switching', 'logic')
WEIGHTS = {name: 1.0 for name in HINTS}


@dataclass(frozen=True)
class Profile:
    name: str = 'heterogeneous-continuous'
    weights: Mapping[str, float] = field(default_factory=lambda: {k: 1.0 for k in ANALOG})
    disposition: str = 'analog'  # analog, digital, mixed; filters family weights explicitly
    motifs: int = 6
    target_transistors: int = 0  # 0: use motif count; otherwise stop at/just above this target
    depth: int = 3
    topology_reuse: float = 0.1
    parameter_reuse: float = 0.0
    diversity: float = 1.0  # log10 range about nominal parameters; geometry is separately bounded
    coupling: float = 0.8
    interconnect: str = 'mesh'
    fanout: int = 2
    region_size: int = 32
    source_correlation: float = 0.0  # probability of reusing the common excitation draw
    vdd: float = 1.2
    bias_current: float = 20e-6
    common_mode: float = 0.6
    amplitude: float = 0.05
    excursion: str = 'small'
    loading: float = 10000.0
    feedback: float = 0.5
    feedback_polarity: int = -1
    compensation: float = 1e-12
    nominal_q: float = 3.0
    time_scale_spread: float = 2.0  # log10 spread in characteristic frequencies
    tstop: float = 1e-6
    maxstep: float = 2e-9
    supply_impedance: float = 0.5
    return_impedance: float = 0.1
    conditioning: bool = True
    shunt: float = 1e12
    model_family: str = 'bsim4'
    max_devices: int = 200000
    max_nets: int = 150000
    max_motifs: int = 12000
    max_observed: int = 48
    max_requested_points: int = 100000  # tstop/maxstep; adaptive steps can be smaller
    max_attempts: int = 4
    version: int = 1

    def __post_init__(self) -> None:
        # JSON spells the same control as 1 or 1.0. Normalize before hashing or sampling.
        for f in dataclasses.fields(self):
            if f.type in ('float', float):
                value = getattr(self, f.name)
                if type(value) not in (int, float):
                    raise ValueError(f'{f.name} must be numeric')
                object.__setattr__(self, f.name, float(value))
        if any(type(v) not in (int, float) for v in self.weights.values()):
            raise ValueError('hint weights must be numeric')
        object.__setattr__(self, 'weights', {k: float(v) for k, v in self.weights.items()})
        if self.version != 1 or self.model_family != 'bsim4':
            raise ValueError('unsupported profile version or model family')
        if self.disposition not in ('analog', 'digital', 'mixed') or self.interconnect not in ('chain', 'tree', 'mesh'):
            raise ValueError('invalid disposition or interconnect')
        if self.excursion not in ('small', 'large', 'recovery') or self.feedback_polarity not in (-1, 1):
            raise ValueError('invalid excursion or feedback polarity')
        if not isinstance(self.conditioning, bool):
            raise ValueError('conditioning must be boolean')
        integer_fields = ('motifs', 'depth', 'fanout', 'region_size', 'max_devices', 'max_nets',
                          'max_motifs', 'max_observed', 'max_attempts', 'max_requested_points')
        if any(type(getattr(self, k)) is not int or getattr(self, k) < 1 for k in integer_fields):
            raise ValueError('counts and budgets must be positive integers')
        if type(self.target_transistors) is not int or self.target_transistors < 0:
            raise ValueError('target_transistors must be a nonnegative integer')
        for k in ('topology_reuse', 'parameter_reuse', 'source_correlation', 'coupling', 'feedback'):
            if not math.isfinite(getattr(self, k)) or not 0 <= getattr(self, k) <= 1:
                raise ValueError(f'{k} must be in [0, 1]')
        for k in ('vdd', 'bias_current', 'loading', 'compensation', 'nominal_q', 'tstop', 'maxstep',
                  'supply_impedance', 'return_impedance', 'shunt'):
            if not math.isfinite(getattr(self, k)) or getattr(self, k) <= 0:
                raise ValueError(f'{k} must be finite and positive')
        for k in ('diversity', 'time_scale_spread', 'amplitude', 'common_mode'):
            if not math.isfinite(getattr(self, k)) or getattr(self, k) < 0:
                raise ValueError(f'{k} must be finite and nonnegative')
        if self.diversity > 6 or self.time_scale_spread > 12 or self.maxstep > self.tstop:
            raise ValueError('parameter/time-scale range exceeds admitted bounds')
        if self.tstop / self.maxstep > self.max_requested_points:
            raise ValueError('requested transient sampling exceeds point budget')
        if not self.weights or set(self.weights) - set(HINTS):
            raise ValueError('unknown or empty hint weights')
        if any(not math.isfinite(v) or v < 0 for v in self.weights.values()) or not self.effective_weights:
            raise ValueError('need positive finite weights in the selected disposition')
        if self.target_transistors and not any(k in self.effective_weights for k in
                                               ('bias', 'gain', 'nonlinear', 'switching', 'logic')):
            raise ValueError('a transistor target needs transistor-bearing hint weights')
        if self.depth > 32 or self.fanout > 16 or self.max_attempts > 100:
            raise ValueError('depth, fanout or attempt budget exceeds hard generation limit')
        if self.max_devices > 2000000 or self.max_nets > 2000000 or self.max_motifs > 100000:
            raise ValueError('resource budget exceeds hard generation limit')

    @property
    def effective_weights(self) -> dict[str, float]:
        admitted = set(ANALOG) if self.disposition == 'analog' else (
            {'logic'} if self.disposition == 'digital' else set(HINTS))
        return {k: v for k, v in sorted(self.weights.items()) if k in admitted and v > 0}

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


RECIPES: Mapping[str, Mapping[str, Any]] = {
    'heterogeneous-continuous': {},
    'coupled-feedback': {'weights': {'gain': 1, 'feedback': 2, 'bias': 1}, 'coupling': 1.0},
    'irregular-passive': {'weights': {'rc': 1, 'rlc': 1, 'primitive': 1}, 'diversity': 2.0},
    'mixed-time-scales': {'disposition': 'mixed', 'weights': WEIGHTS, 'time_scale_spread': 4.0},
    'repetitive-switching': {'disposition': 'digital', 'weights': {'logic': 1}, 'topology_reuse': 1.0,
                             'parameter_reuse': 1.0, 'source_correlation': 1.0, 'tstop': 20e-9, 'maxstep': 50e-12},
    'asynchronous-repeated': {'disposition': 'digital', 'weights': {'logic': 1}, 'topology_reuse': 1.0,
                              'parameter_reuse': 1.0, 'source_correlation': 0.0, 'tstop': 20e-9, 'maxstep': 50e-12},
    'repeated-analog': {'weights': {'gain': 1}, 'topology_reuse': 1.0, 'parameter_reuse': 1.0,
                        'source_correlation': 1.0},
    'nonlinear-small': {'weights': {'nonlinear': 1}, 'amplitude': 0.02},
    'nonlinear-large': {'weights': {'nonlinear': 1}, 'amplitude': 1.2, 'excursion': 'large'},
    'nonlinear-recovery': {'weights': {'nonlinear': 1}, 'amplitude': 1.2, 'excursion': 'recovery'},
    'switching-disturbance': {'disposition': 'mixed', 'weights': {'switching': 1}, 'coupling': 1.0},
    'large-digital': {'disposition': 'digital', 'weights': {'logic': 1}, 'target_transistors': 23000,
                       'tstop': 20e-9, 'maxstep': 100e-12},
    'large-analog': {'weights': {'gain': 3, 'bias': 1, 'nonlinear': 1}, 'target_transistors': 23000},
    'large-mixed': {'disposition': 'mixed', 'weights': {'logic': 3, 'gain': 2, 'switching': 2, 'bias': 1},
                     'target_transistors': 23000},
}


def profile(name: str, overrides: Mapping[str, Any] | None = None) -> Profile:
    if name not in RECIPES:
        raise ValueError(f'unknown structural profile {name!r}')
    values = {'name': name, **RECIPES[name], **(overrides or {})}
    unknown = values.keys() - Profile.__dataclass_fields__.keys()
    if unknown:
        raise ValueError(f'unknown profile controls: {sorted(unknown)}')
    return Profile(**values)


class GenerationError(ValueError):
    def __init__(self, history: list[dict[str, Any]]):
        self.history = history
        super().__init__('generation exhausted: ' + '; '.join(h['reason'] for h in history))


@dataclass(frozen=True)
class Generated:
    circuit: Circuit
    attempts: tuple[Mapping[str, Any], ...]

    def metadata(self) -> dict[str, Any]:
        return {'generator': 'structural-1', 'seed': self.circuit.seed, 'profile': self.circuit.profile,
                'circuit_sha256': fingerprint(self.circuit.to_json()), 'attempts': list(self.attempts),
                'structure': self.circuit.structural(), 'parents': list(self.circuit.history)}


class Builder:
    def __init__(self, seed: int, p: Profile, attempt: int):
        self.seed, self.p = seed, p
        self.attempt = attempt
        self.r = random.Random(f'{seed}:structural-1:{attempt}')
        self.nets = ['0']
        self.devices: list[Device] = []
        self.motifs: list[Motif] = []
        self.transistors = 0
        self.owner = 'shared'
        self.supply, self.ground = self.net(), self.net()
        rail = self.net()
        self.add('V', (rail, '0'), p.vdd, role='supply')
        self.add('R', (rail, self.supply), p.supply_impedance, role='shared-supply')
        self.add('R', (self.ground, '0'), p.return_impedance, role='shared-return')
        self.add('C', (self.supply, self.ground), p.compensation * 100, role='decoupling')
        self.prior: dict[str, list[tuple[int, int]]] = {}
        self.common_source = self.r.getrandbits(64)
        self.shape = random.Random(0)
        self.params = random.Random(0)

    def net(self) -> str:
        if len(self.nets) >= self.p.max_nets:
            raise ValueError('net budget exceeded')
        name = f'n{len(self.nets)}'
        self.nets.append(name)
        return name

    def add(self, kind: str, nodes: tuple[str, ...], value: float = 0.0, *, role: str = 'component',
            model: str = '', parameters: Mapping[str, float] | None = None,
            stimulus: Mapping[str, Any] | None = None) -> None:
        if len(self.devices) >= self.p.max_devices:
            raise ValueError('device budget exceeded')
        self.devices.append(Device(f'd{len(self.devices)}', kind, nodes, value, model,
                                   parameters or {}, stimulus or {}, self.owner, role))
        self.transistors += kind == 'M'

    def value(self, nominal: float) -> float:
        return nominal * 10 ** self.params.uniform(-self.p.diversity / 2, self.p.diversity / 2)

    def mos(self, drain: str, gate: str, source: str, model: str = 'n', role: str = 'active') -> None:
        bulk = self.supply if model == 'p' else self.ground
        width = min(15e-6, max(0.15e-6, self.value(1e-6 if model == 'p' else 0.5e-6)))
        length = min(1e-6, max(0.13e-6, self.value(0.18e-6)))
        self.add('M', (drain, gate, source, bulk), model=model, parameters={'w': width, 'l': length}, role=role)

    def inverter(self, source: str, output: str) -> None:
        self.mos(output, source, self.ground)
        self.mos(output, source, self.supply, 'p')

    def excitation(self, node: str, source_seed: int, digital: bool = False) -> None:
        p = self.p
        rng = random.Random(source_seed)
        frequency = 3 / p.tstop * 10 ** rng.uniform(-p.time_scale_spread / 2, p.time_scale_spread / 2)
        if digital or p.excursion == 'recovery':
            period = 1 / frequency
            values = {'kind': 'pulse', 'low': 0.0 if digital else p.common_mode,
                      'high': p.vdd if digital else p.common_mode + p.amplitude,
                      'delay': rng.random() * period, 'rise': period / 50, 'fall': period / 50,
                      'width': period * (0.1 if p.excursion == 'recovery' else 0.5), 'period': period}
        else:
            values = {'kind': 'sin', 'offset': p.common_mode, 'amplitude': p.amplitude,
                      'frequency': frequency, 'delay': 0.0, 'damping': 0.0, 'phase': rng.random() * 360}
        self.add('V', (node, self.ground), stimulus=values, role='excitation')

    def primitive(self, source: str, depth: int) -> str:
        """An irregular connected graph, not just an RC ladder."""
        nodes = [source]
        for _ in range(depth + 2):
            out = self.net()
            self.add('R', (self.shape.choice(nodes), out), self.value(self.p.loading), role='graph-edge')
            self.add('C', (out, self.ground), self.value(self.p.compensation))
            if len(nodes) > 1:
                self.add('C', (out, self.shape.choice(nodes)), self.value(self.p.compensation), role='bridge')
            nodes.append(out)
        return nodes[-1]

    def bias(self, source: str, depth: int) -> tuple[str, str]:
        bias = self.net()
        self.add('I', (self.supply, bias), self.value(self.p.bias_current), role='bias-reference')
        self.mos(bias, bias, self.ground, role='bias-reference')
        previous, out = bias, bias
        for _ in range(depth):
            out = self.net()
            self.mos(out, previous, self.ground, role='bias-branch')
            self.add('R', (self.supply, out), self.value(self.p.loading), role='load')
            self.add('C', (source, out), self.value(self.p.compensation), role='bias-excitation')
            # Mirror tree: a separately loaded diode-connected child biases the next branch.
            child = self.net()
            self.add('R', (out, child), self.value(self.p.loading), role='bias-distribution')
            self.mos(child, child, self.ground, role='bias-child')
            previous = child
        return out, bias

    def gain(self, source: str, depth: int, external_bias: str | None) -> tuple[str, str]:
        tail, bias, common = self.net(), external_bias or self.net(), self.net()
        if external_bias is None:
            self.add('I', (self.supply, bias), self.value(self.p.bias_current), role='bias-reference')
            self.mos(bias, bias, self.ground, role='bias-reference')
        self.mos(tail, bias, self.ground, role='tail-bias')
        self.add('V', (common, self.ground), self.p.common_mode, role='common-mode')
        out, other = self.net(), self.net()
        self.mos(out, source, tail, role='differential-pair')
        self.mos(other, common, tail, role='differential-pair')
        for node in (out, other):
            self.add('R', (self.supply, node), self.value(self.p.loading), role='load')
        for _ in range(depth):
            gate, following = self.net(), self.net()
            self.add('R', (out, gate), self.value(self.p.loading / 10), role='interstage')
            self.mos(following, gate, self.ground)
            self.add('R', (self.supply, following), self.value(self.p.loading), role='load')
            self.add('C', (following, self.ground), self.value(self.p.compensation), role='load')
            out = following
        return out, bias

    def feedback(self, source: str, depth: int) -> str:
        negative, out = self.net(), source
        for i in range(depth):
            ideal, following = self.net(), self.net()
            controls = (source, negative) if i == 0 else (out, self.ground)
            self.add('E', (ideal, self.ground, *controls), self.value(3.0), role='gain-scaffold')
            self.add('R', (ideal, following), self.value(self.p.loading / 10), role='interstage')
            self.add('C', (following, self.ground), self.value(self.p.compensation), role='compensation')
            out = following
        self.add('R', (negative, self.ground), self.p.loading, role='feedback-divider')
        if self.p.feedback:
            node = out
            if self.p.feedback_polarity > 0:
                node = self.net()
                self.add('E', (node, self.ground, out, self.ground), -1, role='feedback-polarity')
            self.add('R', (node, negative), self.p.loading / self.p.feedback, role='feedback')
            self.add('C', (out, source), self.value(self.p.compensation), role='compensation')
        return out

    def resonator(self, source: str, depth: int) -> str:
        out = source
        for _ in range(depth + 1):
            frequency = 3 / self.p.tstop * 10 ** self.params.uniform(-self.p.time_scale_spread / 2,
                                                                   self.p.time_scale_spread / 2)
            capacitance = self.value(self.p.compensation)
            inductance = 1 / ((2 * math.pi * frequency) ** 2 * capacitance)
            resistance = math.sqrt(inductance / capacitance) / self.value(self.p.nominal_q)
            mid, following = self.net(), self.net()
            self.add('R', (out, mid), resistance, role='resonant-coupling')
            self.add('L', (mid, following), inductance, role='resonator')
            self.add('C', (following, self.ground), capacitance, role='resonator')
            out = following
        return out

    def nonlinear(self, source: str, depth: int) -> str:
        out = source
        for _ in range(depth):
            following = self.net()
            self.add('R', (out, following), self.value(self.p.loading), role='interstage')
            self.add('D', (following, self.supply), model='diode', role='clamp')
            self.add('D', (self.ground, following), model='diode', role='clamp')
            active = self.net()
            self.mos(active, following, self.ground)
            self.add('R', (self.supply, active), self.value(self.p.loading), role='load')
            self.add('C', (active, self.ground), self.value(self.p.compensation), role='memory')
            out = active
        return out

    def logic(self, source: str, depth: int) -> str:
        nodes = [source]
        for _ in range(depth + 1):
            a = self.shape.choice(nodes)
            out = self.net()
            if len(nodes) < 2 or self.shape.random() < 0.5:
                self.inverter(a, out)
            else:
                b, series = self.shape.choice(nodes), self.net()
                # CMOS NAND scaffold with fanin from earlier stages; not independent inverter copies.
                self.mos(out, a, self.supply, 'p')
                self.mos(out, b, self.supply, 'p')
                self.mos(out, a, series)
                self.mos(series, b, self.ground)
            self.add('C', (out, self.ground), self.value(self.p.compensation / 10), role='load')
            nodes.append(out)
        return nodes[-1]

    def motif(self, kind: str, index: int) -> None:
        p = self.p
        def axis(name: str) -> random.Random:
            return random.Random(f'{self.seed}:structural-1:{self.attempt}:{index}:{name}')
        shape_rng, parameter_rng, source_rng, rng = (axis(name) for name in
                                                    ('shape', 'parameter', 'source', 'coupling'))
        prior = self.prior.setdefault(kind, [])
        shape_seed = (shape_rng.choice(prior)[0] if prior and shape_rng.random() < p.topology_reuse
                      else shape_rng.getrandbits(64))
        parameter_seed = (parameter_rng.choice(prior)[1] if prior and parameter_rng.random() < p.parameter_reuse
                          else parameter_rng.getrandbits(64))
        source_seed = self.common_source if source_rng.random() < p.source_correlation else source_rng.getrandbits(64)
        prior.append((shape_seed, parameter_seed))
        self.shape, self.params = random.Random(shape_seed), random.Random(parameter_seed)
        self.owner = f'm{index}'
        depth = self.shape.randint(1, p.depth)
        source, incoming = self.net(), self.net()
        self.excitation(source, source_seed, digital=kind == 'logic')
        self.add('R', (source, incoming), self.value(p.loading / 10), role='source-impedance')
        candidates = []
        if self.motifs and rng.random() < p.coupling:
            if p.interconnect == 'chain':
                candidates = [self.motifs[-1]]
            elif p.interconnect == 'tree':
                candidates = [self.motifs[(index - 1) // p.fanout]]
            else:
                candidates = rng.sample(self.motifs, min(p.fanout, len(self.motifs)))
        for previous in candidates:
            self.add('R', (previous.ports['out'], incoming), p.loading / max(p.coupling, 1e-6), role='coupling')
        ports = {'in': incoming, 'supply': self.supply, 'return': self.ground}
        external_bias = next((m.ports['bias'] for m in reversed(candidates) if 'bias' in m.ports), None)
        implementation = 'transistor/passive'
        if kind == 'bias':
            out, ports['bias'] = self.bias(incoming, depth)
        elif kind == 'gain':
            out, ports['bias'] = self.gain(incoming, depth, external_bias)
        elif kind == 'feedback':
            out = self.feedback(incoming, depth)
            implementation = 'controlled-source'
        elif kind == 'rlc':
            out = self.resonator(incoming, depth)
        elif kind == 'nonlinear':
            out = self.nonlinear(incoming, depth)
        elif kind == 'logic':
            out = self.logic(incoming, depth)
        elif kind == 'switching':
            clock, switched = self.net(), self.net()
            self.excitation(clock, source_seed ^ 0xABCD, digital=True)
            self.inverter(clock, switched)
            out, ports['bias'] = self.gain(incoming, depth, external_bias)
            self.add('C', (switched, incoming), self.value(p.compensation), role='switching-injection')
            self.add('C', (switched, self.ground), self.value(p.compensation * 10), role='switching-load')
            ports['switching'] = switched
        else:
            out = self.primitive(incoming, depth)
        self.add('R', (out, self.ground), self.value(p.loading * 10), role='terminal-load')
        ports['out'] = out
        self.motifs.append(Motif(self.owner, kind, ports, shape_seed, parameter_seed, source_seed,
                                 'switching' if kind == 'logic' else p.excursion, implementation,
                                 index // p.region_size))

    def build(self) -> Circuit:
        p, rng = self.p, self.r
        weights = p.effective_weights
        global_supply, global_return = self.supply, self.ground
        for index in range(p.max_motifs):
            if (p.target_transistors and self.transistors >= p.target_transistors
                    or not p.target_transistors and index >= p.motifs):
                break
            if index % p.region_size == 0:
                self.owner = 'shared'
                self.supply, self.ground = self.net(), self.net()
                self.add('R', (global_supply, self.supply), p.supply_impedance, role='regional-supply')
                self.add('R', (self.ground, global_return), p.return_impedance, role='regional-return')
                self.add('C', (self.supply, self.ground), p.compensation * 100, role='decoupling')
            self.motif(rng.choices(list(weights), list(weights.values()))[0], index)
        if (p.target_transistors and self.transistors < p.target_transistors
                or not p.target_transistors and len(self.motifs) < p.motifs):
            raise ValueError('motif budget exhausted before requested size')
        if p.conditioning:
            self.owner = 'shared'
            for net in self.nets[1:]:
                self.add('R', (net, '0'), p.shunt, role='conditioning')
        # Spread a bounded observation budget across the whole circuit, not just its first region.
        candidates = list(dict.fromkeys([global_supply, global_return] + [m.ports['out'] for m in self.motifs]))
        count = min(p.max_observed, len(candidates))
        selected = [candidates[i * (len(candidates) - 1) // max(1, count - 1)] for i in range(count)]
        analysis = Analysis(f'tran {p.maxstep:.12g} {p.tstop:.12g} 0 {p.maxstep:.12g}',
                            'waveform.txt', tuple(f'v({net})' for net in selected))
        return Circuit(self.seed, tuple(self.nets), tuple(self.devices), tuple(self.motifs), (analysis,),
                       p.to_json(), p.model_family)


def generate(seed: int, p: Profile) -> Generated:
    history = []
    for attempt in range(p.max_attempts):
        try:
            circuit = Builder(seed, p, attempt).build()
            circuit.validate()
        except ValueError as error:
            history.append({'attempt': attempt, 'state': 'rejected', 'reason': str(error)})
            continue
        history.append({'attempt': attempt, 'state': 'accepted'})
        return Generated(circuit, tuple(history))
    raise GenerationError(history)


def transform(parent: Circuit, operation: str, seed: int) -> Generated:
    """Transform the actual graph, never substitute a newly sampled parent."""
    rng = random.Random(seed)
    p = Profile(**parent.profile)
    circuit = parent
    if operation == 'duplicate':
        prefix = f'copy{len(parent.history) + 1}_'
        mapping = {n: prefix + n if n != '0' else n for n in parent.nets}
        copies = tuple(dataclasses.replace(d, id=prefix + d.id, nodes=tuple(mapping[n] for n in d.nodes),
                                           owner=prefix + d.owner if d.owner != 'shared' else 'shared')
                       for d in parent.devices)
        region_offset = max(x.region for x in parent.motifs) + 1
        motifs = tuple(dataclasses.replace(m, id=prefix + m.id, ports={k: mapping[v] for k, v in m.ports.items()},
                                           region=m.region + region_offset)
                       for m in parent.motifs)
        link = Device(prefix + 'link', 'R', (parent.motifs[-1].ports['out'], motifs[0].ports['in']),
                      p.loading, role='coupling')
        p = dataclasses.replace(p, motifs=len(parent.motifs) * 2, target_transistors=0)
        vectors = [v for a in parent.analyses for v in a.vectors]
        vectors += [f'v({mapping[v[2:-1]]})' for v in vectors]
        count = min(len(vectors), p.max_observed)
        selected = tuple(vectors[i * (len(vectors) - 1) // max(1, count - 1)] for i in range(count))
        circuit = dataclasses.replace(parent, nets=(*parent.nets, *(mapping[n] for n in parent.nets if n != '0')),
                                      devices=(*parent.devices, *copies, link), motifs=(*parent.motifs, *motifs),
                                      profile=p.to_json(), analyses=(dataclasses.replace(parent.analyses[0],
                                                                                       vectors=selected),),
                                      initial_conditions={**parent.initial_conditions,
                                                          **{mapping[k]: v for k, v in
                                                             parent.initial_conditions.items()}})
    elif operation not in ('diversify', 'excitation', 'reconnect'):
        raise ValueError('transform must be duplicate, diversify, reconnect or excitation')
    devices = []
    owner_index = {m.id: i for i, m in enumerate(circuit.motifs)}
    for d in circuit.devices:
        if operation == 'reconnect' and d.role == 'coupling':
            index = rng.randrange(len(circuit.motifs) - 1)
            if index >= owner_index.get(d.owner, len(circuit.motifs)):
                index += 1
            d = dataclasses.replace(d, nodes=(circuit.motifs[index].ports['out'], d.nodes[1]))
        if operation == 'diversify' and d.role not in ('conditioning', 'supply'):
            factor = 10 ** rng.uniform(-p.diversity / 2, p.diversity / 2)
            if d.kind in ('R', 'C', 'L', 'I'):
                d = dataclasses.replace(d, value=d.value * factor)
            elif d.kind == 'M':
                d = dataclasses.replace(d, parameters={**d.parameters,
                                                       'w': min(15e-6, max(0.15e-6, d.parameters['w'] * factor))})
        if operation == 'excitation' and d.stimulus:
            values = dict(d.stimulus)
            if values['kind'] == 'sin':
                values['phase'] = rng.random() * 360
            else:
                values['delay'] = rng.random() * values['period']
            d = dataclasses.replace(d, stimulus=values)
        devices.append(d)
    history = (*parent.history, {'parent': fingerprint(parent.to_json()), 'operation': operation, 'seed': seed})
    motifs = tuple(dataclasses.replace(m, parameter_seed=rng.getrandbits(64)) if operation == 'diversify' else
                   dataclasses.replace(m, source_seed=rng.getrandbits(64)) if operation == 'excitation' else m
                   for m in circuit.motifs)
    result = dataclasses.replace(circuit, devices=tuple(devices), motifs=motifs, history=history)
    result.validate()
    return Generated(result, ({'attempt': 0, 'state': 'accepted', 'operation': operation},))
