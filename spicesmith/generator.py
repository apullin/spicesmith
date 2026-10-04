"""Seeded random block-based circuits with self-contained transistor models.

The same seed always gives the same deck. A deck is a handful of *blocks*, circuit fragments
such as inverter chains, latches or RC ladders, around built-in BSIM4 transistors,
parameterized subcircuits with .if, subcircuit-local models, R/C/L and
controlled and behavioural sources. New kinds of circuit are new Block classes.
"""
from __future__ import annotations

import dataclasses
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Mapping, Optional, Sequence

from .deck import BlockInfo, CircuitInfo, Deck, eng

VERSION = 7  # portable models and standard analysis directives replace the previous bindings

SUPPLIES = (0.9, 1.0, 1.2, 1.2, 1.32)  # V; repeats weight the draw
STOP_TIMES = (2e-9, 5e-9, 10e-9, 20e-9)  # s
WIDTHS = (0.15e-6, 0.3e-6, 0.5e-6, 0.8e-6, 1e-6, 2e-6, 4e-6)
LENGTHS = (0.13e-6, 0.13e-6, 0.18e-6, 0.3e-6, 0.5e-6, 1e-6)
EDGES = (10e-12, 20e-12, 50e-12, 100e-12)
UNDER_RESOLVED = 10  # a resonance with fewer time steps per period than this is labelled so
HIGH_Q = 10


@dataclass(frozen=True)
class Subcircuit:
    """A parameterized subcircuit definition the blocks share; defined once per deck."""
    name: str
    lines: tuple[str, ...]
    uses: tuple[str, ...] = ()  # subcircuits it instantiates


# --- Subcircuit library ---------------------------------------------------------------------

SUBCIRCUITS: Mapping[str, Subcircuit] = {s.name: s for s in (
    Subcircuit('finv', (
        '.subckt finv in out vdd vss wn=0.5u wp=1u l=0.13u',
        'Xn out in vss vss smith_nmos w={wn} l={l} ng=1 m=1',
        'Xp out in vdd vdd smith_pmos w={wp} l={l} ng=1 m=1',
        '.ends')),
    Subcircuit('fnand', (
        '.subckt fnand a b out vdd vss wn=0.5u wp=0.5u l=0.13u',
        'Xn1 out a mid vss smith_nmos w={2*wn} l={l} ng=1 m=1',
        'Xn2 mid b vss vss smith_nmos w={2*wn} l={l} ng=1 m=1',
        'Xp1 out a vdd vdd smith_pmos w={wp} l={l} ng=1 m=1',
        'Xp2 out b vdd vdd smith_pmos w={wp} l={l} ng=1 m=1',
        '.ends')),
    # sizing chosen by .if on a parameter; both branches reference devices
    Subcircuit('fsel', (
        '.subckt fsel in out vdd vss wn=0.5u l=0.13u strong=0',
        '.if (strong == 1)',
        'Xn out in vss vss smith_nmos w={2*wn} l={l} ng=2 m=1',
        'Xp out in vdd vdd smith_pmos w={4*wn} l={l} ng=2 m=1',
        '.else',
        'Xn out in vss vss smith_nmos w={wn} l={max(l, 0.13u)} ng=1 m=1',
        'Xp out in vdd vdd smith_pmos w={2*wn} l={max(l, 0.13u)} ng=1 m=1',
        '.endif',
        '.ends')),
    # subcircuit-local models, one of them unused
    Subcircuit('fclamp', (
        '.subckt fclamp a vss isat=1e-14 rs=10',
        '.model dclamp d is={isat} n=1.05 rs={rs} cjo=5f',
        '.model dunused d is=1e-15',
        'D1 vss a dclamp',
        'Rp a vss {1meg*(1+isat*1e12)}',
        '.ends')),
    # nested parameterized subcircuit with a derived parameter
    Subcircuit('fbuf', (
        '.subckt fbuf in out vdd vss wn=0.5u ratio=2',
        '.param wmid={wn*ratio}',
        'X1 in mid vdd vss finv wn={wn} wp={2*wn}',
        'X2 mid out vdd vss finv wn={wmid} wp={2*wmid}',
        'Cm mid vss 0.2f',
        '.ends'), uses=('finv',)),
)}


# A local library with sections that pick the cell sizing, and a nested include that defines
# the cell; written next to the deck (Deck.files).
LIBRARY_FILES: Mapping[str, str] = {
    'fuzzlib.lib': '\n'.join([
        '* SpiceSmith local library: the section picks the cell sizing',
        '.lib typ', '.param lib_wn=0.4u lib_wp=0.8u', '.include "fuzzcell.inc"', '.endl typ',
        '.lib wide', '.param lib_wn=0.8u lib_wp=1.6u', '.include "fuzzcell.inc"', '.endl wide', '']),
    'fuzzcell.inc': '\n'.join([
        '* SpiceSmith library cell, included from a section of fuzzlib.lib',
        '.subckt lcell in out vdd vss wn={lib_wn} wp={lib_wp}',
        'Xn out in vss vss smith_nmos w={wn} l=0.13u ng=1 m=1',
        'Xp out in vdd vdd smith_pmos w={wp} l=0.13u ng=1 m=1',
        '.ends', '']),
}
LIBRARY_SECTIONS = ('typ', 'wide')

# Built-in BSIM4 models binned by length (instances name nbin/pbin, ngspice picks nbin.1 or
# nbin.2), and a binned model nothing uses, for the unused-model removal.
BINNED_MODELS = (
    '.model nbin.1 nmos level=54 version=4.8 lmin=0.1u lmax=0.4u wmin=0.1u wmax=20u',
    '.model nbin.2 nmos level=54 version=4.8 lmin=0.4u lmax=4u wmin=0.1u wmax=20u',
    '.model pbin.1 pmos level=54 version=4.8 lmin=0.1u lmax=0.4u wmin=0.1u wmax=20u',
    '.model pbin.2 pmos level=54 version=4.8 lmin=0.4u lmax=4u wmin=0.1u wmax=20u',
    '.model nunused.1 nmos level=54 version=4.8 lmin=0.1u lmax=4u wmin=0.1u wmax=20u',
)

# .func definitions blocks may use
FUNCTIONS: Mapping[str, str] = {
    'rpar': '.func rpar(a,b) {a*b/(a+b)}',
    'rser': '.func rser(a,b,c) {a+b+c}',
}


class CircuitBuilder:
    """One random circuit while it is being built.

    Blocks draw every random choice from `rng`, in a fixed order, so that a seed keeps
    giving the same deck; mutable by design, it produces an immutable Deck.
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)  # what blocks draw from; a copy of an array swaps in its own
        self._deck_rng = self.rng  # deck-wide choices, the same for every copy
        self.vdd: float = self.rng.choice(SUPPLIES)
        self.tstop: float = self.rng.choice(STOP_TIMES)
        self.elements: list[str] = []
        self.params: list[str] = []
        self.ics: list[str] = []
        self.subcircuits: dict[str, Subcircuit] = {}
        self.functions: dict[str, str] = {}
        self.nodesets: list[str] = []
        self.files: dict[str, str] = {}
        self.library_section: Optional[str] = None
        self.models: list[str] = []
        self.nets: list[str] = []
        self.blocks: list[BlockInfo] = []
        self._count = 0
        self._attributes: dict[str, float] = {}

    # --- Names ---------------------------------------------------------------------------------

    def name(self, prefix: str) -> str:
        """A fresh element or parameter name."""
        self._count += 1
        return f'{prefix}{self._count}'

    def net(self, prefix: str = 'n') -> str:
        """A fresh top-level net; top-level nets are the observed outputs."""
        net = self.name(prefix)
        self.nets.append(net)
        return net

    def define(self, name: str) -> str:
        """Use a library subcircuit (and what it instantiates); returns its name."""
        subcircuit = SUBCIRCUITS[name]
        if name not in self.subcircuits:
            for dependency in subcircuit.uses:
                self.define(dependency)
            self.subcircuits[name] = subcircuit
        return name

    # --- Random parts --------------------------------------------------------------------------

    def width(self) -> float:
        return self.rng.choice(WIDTHS)

    def length(self) -> float:
        return self.rng.choice(LENGTHS)

    def time(self, fraction: float) -> str:
        return eng(self.tstop * fraction)

    def mos(self, kind: str, drain: object, gate: object, source: object, bulk: object,
            width: Optional[float] = None, length: Optional[float] = None) -> str:
        """A transistor through a self-contained wrapper; ng scales the finger width."""
        ng = self.rng.choice([1, 1, 1, 2, 3])
        m = self.rng.choice([1, 1, 2])
        return (f'X{self.name("m")} {drain} {gate} {source} {bulk} smith_{kind} '
                f'w={eng(width or self.width())} l={eng(length or self.length())} ng={ng} m={m}')

    def pulse(self, low: float = 0.0, high: Optional[float] = None) -> str:
        high = self.vdd if high is None else high
        delay, period = self.rng.uniform(0.05, 0.4), self.rng.uniform(0.15, 0.6)
        edge = self.rng.choice(EDGES)
        return (f'PULSE({low:.4g} {high:.4g} {self.time(delay)} {eng(edge)} {eng(edge)} '
                f'{self.time(period / 2)} {self.time(period)})')

    def source(self, node: str, kind: Optional[str] = None) -> None:
        """A voltage source driving `node`: pulse, piecewise linear, sine, exponential or
        single-frequency FM (SFFM), all within the rails."""
        r = self.rng
        kind = kind or r.choice(['pulse', 'pulse', 'pwl', 'sin', 'exp', 'sffm'])
        if kind == 'pulse':
            wave = self.pulse()
        elif kind == 'pwl':
            points, t = [], 0.0
            for _ in range(r.randint(2, 6)):
                t += r.uniform(0.05, 0.3) * self.tstop
                points.append(f'{eng(t)} {r.choice([0.0, self.vdd, self.vdd / 2]):.4g}')
            wave = 'PWL(0 0 ' + ' '.join(points) + ')'
        elif kind == 'sin':
            wave = (f'SIN({self.vdd / 2:.4g} {r.uniform(0.05, 0.5) * self.vdd:.4g} '
                    f'{eng(r.uniform(1, 6) / self.tstop)})')
        elif kind == 'exp':
            rise_delay = r.uniform(0.05, 0.3) * self.tstop
            fall_delay = rise_delay + r.uniform(0.2, 0.5) * self.tstop
            wave = (f'EXP(0 {r.choice([self.vdd, self.vdd / 2]):.4g} {eng(rise_delay)} '
                    f'{eng(r.uniform(0.01, 0.1) * self.tstop)} {eng(fall_delay)} '
                    f'{eng(r.uniform(0.01, 0.1) * self.tstop)})')
        else:
            carrier = r.uniform(2, 8) / self.tstop
            signal = carrier / r.uniform(3, 10)
            # a modulation index above carrier/signal frequency makes ngspice warn and clip it
            wave = (f'SFFM({self.vdd / 2:.4g} {r.uniform(0.05, 0.45) * self.vdd:.4g} {eng(carrier)} '
                    f'{r.uniform(0.1, 0.9) * carrier / signal:.4g} {eng(signal)})')
        self.elements.append(f'V{self.name("s")} {node} 0 {wave}')

    def multiplicity(self) -> str:
        """' m=N' on some subcircuit instances (ngspice scales the devices inside)."""
        m = self.rng.choice([1, 1, 1, 2, 3])
        return f' m={m}' if m > 1 else ''

    def library_cell(self) -> str:
        """The cell of the local library; returns its name. The section is a deck-wide choice,
        drawn once from the deck's generator: drawn from a copy's, the first copy of an array
        would consume a draw the others skip and come out different."""
        if self.library_section is None:
            self.library_section = self._deck_rng.choice(LIBRARY_SECTIONS)
            self.files.update(LIBRARY_FILES)
        return 'lcell'

    def binned_models(self) -> tuple[str, str]:
        """Define the binned BSIM4 models (once); returns the n and p model names."""
        if not self.models:
            self.models += BINNED_MODELS
        return 'nbin', 'pbin'

    def function(self, name: str) -> str:
        """Use a .func; returns its name."""
        self.functions.setdefault(name, FUNCTIONS[name])
        return name

    # --- Blocks --------------------------------------------------------------------------------

    def supply(self) -> str:
        """The supply net every block is powered from, driven at {vsupply}."""
        vdd = self.net('vdd')
        self.elements.append(f'Vdd {vdd} 0 {{vsupply}}')
        self.blocks.append(BlockInfo('supply', frozenset(), (vdd,)))
        return vdd

    def add(self, block: Block, vdd: str, rng: Optional[random.Random] = None) -> None:
        """Build one block, recording the nets it creates and what it said about itself. With
        `rng` the block draws from it instead (identical copies replay the same draws)."""
        before = len(self.nets)
        self._attributes = {}
        main, self.rng = self.rng, rng or self.rng
        try:
            block.build(self, vdd)
        finally:
            self.rng = main
        self.blocks.append(BlockInfo(block.kind, block.classes, tuple(self.nets[before:]), self._attributes))

    def add_array(self, block: Block, vdd: str, copies: int) -> None:
        """`copies` identical instances of a block (same sizes and sources, their own nets)."""
        seed = self.rng.getrandbits(32)
        for _ in range(copies):
            self.add(block, vdd, random.Random(seed))

    def observed(self, limit: int) -> list[str]:
        """The nets written to the waveform: all of them up to `limit`; beyond, the nets of the
        first copy of every block, then one net of each other copy in turn."""
        if len(self.nets) <= limit:
            return list(self.nets)
        chosen: list[str] = []
        seen_kinds: set[tuple[str, int]] = set()
        rest: list[str] = []
        for block in self.blocks:
            shape = (block.kind, len(block.nets))
            if shape in seen_kinds:
                rest.append(block.nets[-1])
            else:
                chosen.extend(block.nets)
                seen_kinds.add(shape)
        picks = [net for net in rest if net not in chosen]
        return [net for net in self.nets if net in set((chosen + picks)[:limit])]

    def describe(self, **attributes: float) -> None:
        """Attributes of the block being built (e.g. a resonance's f0 and Q)."""
        self._attributes.update(attributes)

    def classify_resolution(self, step: float) -> None:
        """Label resonances by how well the time step resolves them."""
        for i, block in enumerate(self.blocks):
            if 'f0' not in block.attributes:
                continue
            steps = 1 / (block.attributes['f0'] * step)
            labels = {'under-resolved'} if steps < UNDER_RESOLVED else {'resolved'}
            if block.attributes.get('q', 0.0) > HIGH_Q:
                labels.add('high-q')
            self.blocks[i] = dataclasses.replace(block, classes=block.classes | labels,
                                                 attributes={**block.attributes, 'steps_per_period': steps})

    def info(self, scale: int = 1) -> CircuitInfo:
        return CircuitInfo(self.seed, VERSION, self.vdd, self.tstop, tuple(self.blocks), scale)


class Block(ABC):
    """A kind of circuit fragment the generator places in decks."""
    kind: ClassVar[str]
    classes: ClassVar[frozenset[str]] = frozenset()  # for per-class reporting

    @abstractmethod
    def build(self, circuit: CircuitBuilder, vdd: str) -> None:
        """Add this block's elements to `circuit`, powered from the net `vdd`."""


@dataclass(frozen=True)
class InverterChain(Block):
    kind: ClassVar[str] = 'inverter-chain'
    classes: ClassVar[frozenset[str]] = frozenset({'digital'})

    def build(self, circuit: CircuitBuilder, vdd: str) -> None:
        r = circuit.rng
        a = circuit.net('in')
        circuit.source(a, 'pulse')
        for _ in range(r.randint(1, 6)):
            b = circuit.net('c')
            if r.random() < 0.5:
                wn = r.choice([0.3e-6, 0.5e-6, 0.5e-6, 1e-6])  # repeated parameter sets
                circuit.elements.append(f'X{circuit.name("i")} {a} {b} {vdd} 0 {circuit.define("finv")} '
                                        f'wn={eng(wn)} wp={eng(2 * wn)}{circuit.multiplicity()}')
            else:
                circuit.elements += [circuit.mos('nmos', b, a, 0, 0), circuit.mos('pmos', b, a, vdd, vdd)]
            circuit.elements.append(f'C{circuit.name("l")} {b} 0 {eng(r.choice([1e-15, 2e-15, 5e-15, 20e-15]))}')
            a = b


@dataclass(frozen=True)
class RingOscillator(Block):
    """A NAND-gated ring: the NAND and an even number of inverters, an odd number of
    inversions in all, so it oscillates once enabled."""
    kind: ClassVar[str] = 'ring'
    classes: ClassVar[frozenset[str]] = frozenset({'digital', 'oscillator'})

    def build(self, circuit: CircuitBuilder, vdd: str) -> None:
        r = circuit.rng
        en = circuit.net('en')
        circuit.elements.append(f'V{circuit.name("s")} {en} 0 PWL(0 0 {circuit.time(0.1)} 0 '
                                f'{circuit.time(0.12)} {circuit.vdd:.4g})')
        stages = [circuit.net('r') for _ in range(r.choice([3, 5, 7]))]
        circuit.elements.append(f'X{circuit.name("g")} {en} {stages[-1]} {stages[0]} {vdd} 0 '
                                f'{circuit.define("fnand")} wn=0.3u wp=0.6u')
        for a, b in zip(stages, stages[1:], strict=False):
            if r.random() < 0.5:
                circuit.elements.append(f'X{circuit.name("i")} {a} {b} {vdd} 0 {circuit.define("fsel")} '
                                        f'wn=0.4u strong={r.randint(0, 1)}')
            else:
                circuit.elements.append(f'X{circuit.name("i")} {a} {b} {vdd} 0 {circuit.define("finv")} '
                                        f'wn=0.3u wp=0.6u')
            circuit.elements.append(f'C{circuit.name("l")} {b} 0 {eng(r.choice([1e-15, 3e-15, 10e-15]))}')


@dataclass(frozen=True)
class Latch(Block):
    """A cross-coupled inverter pair with access transistors, like an SRAM cell."""
    kind: ClassVar[str] = 'latch'
    classes: ClassVar[frozenset[str]] = frozenset({'digital', 'memory'})

    def build(self, circuit: CircuitBuilder, vdd: str) -> None:
        r = circuit.rng
        q, qb, bl, blb, wl = (circuit.net(p) for p in ('q', 'qb', 'bl', 'blb', 'wl'))
        finv = circuit.define('finv')
        circuit.elements += [f'X{circuit.name("i")} {q} {qb} {vdd} 0 {finv} wn=0.3u wp=0.3u',
                             f'X{circuit.name("i")} {qb} {q} {vdd} 0 {finv} wn=0.3u wp=0.3u',
                             circuit.mos('nmos', bl, wl, q, 0, width=0.5e-6),
                             circuit.mos('nmos', blb, wl, qb, 0, width=0.5e-6)]
        # The word line is high only in the middle half of each period and the bit lines
        # switch at period boundaries, so a write never releases the cell while the bit lines
        # pass mid-rail (round-off would decide its state, as for a latch without .ic).
        period = r.uniform(0.15, 0.4) * circuit.tstop
        delay = r.uniform(0.05, 0.2) * circuit.tstop
        edge = min(r.choice(EDGES), period / 8)
        high = f'{circuit.vdd:.4g}'
        circuit.elements += [
            f'V{circuit.name("s")} {wl} 0 PULSE(0 {high} {eng(delay + period / 4)} {eng(edge)} {eng(edge)} '
            f'{eng(period / 2 - edge)} {eng(period)})',
            f'V{circuit.name("s")} {bl} 0 PULSE(0 {high} {eng(delay)} {eng(edge)} {eng(edge)} '
            f'{eng(period - edge)} {eng(2 * period)})',
            f'B{circuit.name("b")} {blb} 0 V={high}-v({bl})']
        # A latch released from its unstable equilibrium falls whichever way round-off
        # tips it; .ic holds a state through the operating point, with or without uic.
        circuit.ics += [f'v({q})={circuit.vdd:.4g}', f'v({qb})=0']


@dataclass(frozen=True)
class CurrentMirror(Block):
    kind: ClassVar[str] = 'mirror'
    classes: ClassVar[frozenset[str]] = frozenset({'analog'})

    def build(self, circuit: CircuitBuilder, vdd: str) -> None:
        r = circuit.rng
        ref, out = circuit.net('mref'), circuit.net('mout')
        current = r.choice([5e-6, 20e-6, 100e-6])
        circuit.elements.append(f'I{circuit.name("i")} {vdd} {ref} PULSE(0 {eng(current)} {circuit.time(0.1)} '
                                f'{circuit.time(0.05)} {circuit.time(0.05)} {circuit.time(0.3)} '
                                f'{circuit.time(0.7)})')
        width, length = circuit.width(), circuit.length()
        circuit.elements += [circuit.mos('nmos', ref, ref, 0, 0, width, length),
                             circuit.mos('nmos', out, ref, 0, 0, 2 * width, length),
                             f'R{circuit.name("r")} {vdd} {out} {eng(r.choice([2e3, 10e3, 50e3]))}']


@dataclass(frozen=True)
class DiffPair(Block):
    kind: ClassVar[str] = 'diff-pair'
    classes: ClassVar[frozenset[str]] = frozenset({'analog'})

    def build(self, circuit: CircuitBuilder, vdd: str) -> None:
        r = circuit.rng
        inp, inn, tail, op, on = (circuit.net(p) for p in ('dip', 'din', 'dt', 'dop', 'don'))
        circuit.source(inp, 'sin')
        circuit.elements.append(f'V{circuit.name("s")} {inn} 0 {circuit.vdd / 2:.4g}')
        width, length = circuit.width(), circuit.length()
        load = eng(r.choice([5e3, 20e3]))
        circuit.elements += [circuit.mos('nmos', op, inp, tail, 0, width, length),
                             circuit.mos('nmos', on, inn, tail, 0, width, length),
                             f'R{circuit.name("r")} {vdd} {op} {load}', f'R{circuit.name("r")} {vdd} {on} {load}',
                             f'I{circuit.name("i")} {tail} 0 {eng(r.choice([10e-6, 50e-6]))}']


@dataclass(frozen=True)
class RcLadder(Block):
    """R/C sections spanning 5 decades of R and 3 of C; some resistors are bare .param names."""
    kind: ClassVar[str] = 'rc-ladder'
    classes: ClassVar[frozenset[str]] = frozenset({'linear'})

    def build(self, circuit: CircuitBuilder, vdd: str) -> None:
        r = circuit.rng
        a = circuit.net('rc')
        circuit.source(a)
        for _ in range(r.randint(2, 8)):
            b = circuit.net('rc')
            resistance = 10 ** r.uniform(1, 6)
            capacitance = 10 ** r.uniform(-15, -12)
            style = r.random()
            if style < 0.25:
                parameter = circuit.name('p')
                circuit.params.append(f'.param {parameter}={eng(resistance)}')
                circuit.elements.append(f'R{circuit.name("r")} {a} {b} {parameter}')  # unquoted parameter
            elif style < 0.4:
                # a .param dependency chain, combined through a .func
                first, second, third = circuit.name('pa'), circuit.name('pb'), circuit.name('pc')
                k = r.choice([2, 3, 5])
                circuit.params.append(f'.param {first}={eng(resistance * (k + 1) / k)} {second}={{{k}*{first}}} '
                                      f'{third}={{{second}/{k}+{first}/2}}')
                value = f'{{{circuit.function("rpar")}({first}, {second})}}'
                circuit.elements.append(f'R{circuit.name("r")} {a} {b} {value}')
            else:
                circuit.elements.append(f'R{circuit.name("r")} {a} {b} {eng(resistance)}')
            circuit.elements.append(f'C{circuit.name("c")} {b} 0 {eng(capacitance)}')
            a = b


@dataclass(frozen=True)
class SeriesRlc(Block):
    kind: ClassVar[str] = 'rlc'
    classes: ClassVar[frozenset[str]] = frozenset({'linear', 'rlc'})

    def build(self, circuit: CircuitBuilder, vdd: str) -> None:
        r = circuit.rng
        a, b, c = circuit.net('l'), circuit.net('l'), circuit.net('l')
        circuit.source(a, 'pulse')
        resistance, inductance = r.choice([5, 50, 500]), r.choice([0.1e-9, 1e-9, 10e-9])
        capacitance = r.choice([10e-15, 100e-15, 1e-12])
        circuit.elements += [f'R{circuit.name("r")} {a} {b} {eng(resistance)}',
                             f'L{circuit.name("l")} {b} {c} {eng(inductance)}',
                             f'C{circuit.name("c")} {c} 0 {eng(capacitance)}']
        circuit.describe(f0=1 / (2 * math.pi * math.sqrt(inductance * capacitance)),
                         q=math.sqrt(inductance / capacitance) / resistance)


@dataclass(frozen=True)
class ControlledSources(Block):
    """An E/G controlled-source loop."""
    kind: ClassVar[str] = 'controlled'
    classes: ClassVar[frozenset[str]] = frozenset({'linear'})

    def build(self, circuit: CircuitBuilder, vdd: str) -> None:
        r = circuit.rng
        a, b, c = circuit.net('k'), circuit.net('k'), circuit.net('k')
        circuit.source(a, r.choice(['sin', 'sffm', 'exp']))
        circuit.elements += [f'E{circuit.name("e")} {b} 0 {a} 0 {r.choice([0.5, -1, 2]):.4g}',
                             f'R{circuit.name("r")} {b} {c} 1k',
                             f'C{circuit.name("c")} {c} 0 {eng(r.choice([10e-15, 1e-12]))}',
                             f'G{circuit.name("g")} {c} 0 {b} 0 {eng(r.choice([1e-6, 1e-4]))}']


@dataclass(frozen=True)
class DiodeClamp(Block):
    """A subcircuit diode clamp with a used and an unused local .model."""
    kind: ClassVar[str] = 'clamp'
    classes: ClassVar[frozenset[str]] = frozenset({'analog'})

    def build(self, circuit: CircuitBuilder, vdd: str) -> None:
        a = circuit.net('cl')
        circuit.elements.append(f'R{circuit.name("r")} {vdd} {a} 10k')
        isat = circuit.rng.choice(['1e-14', '1e-14', '5e-15'])
        circuit.elements.append(f'X{circuit.name("d")} {a} 0 {circuit.define("fclamp")} isat={isat}')


@dataclass(frozen=True)
class Behavioural(Block):
    """A B source with agauss (numparam random draws; .options seed makes them reproducible)."""
    kind: ClassVar[str] = 'behavioural'
    classes: ClassVar[frozenset[str]] = frozenset({'behavioural'})

    def build(self, circuit: CircuitBuilder, vdd: str) -> None:
        a, b = circuit.net('bh'), circuit.net('bh')
        circuit.source(a, circuit.rng.choice(['sin', 'sffm']))
        circuit.elements += [f'B{circuit.name("b")} {b} 0 V=0.5*v({a})+agauss(0, 0.01, 1)',
                             f'R{circuit.name("r")} {b} 0 100k']


@dataclass(frozen=True)
class BufferBank(Block):
    """A chain of nested fbuf subcircuits with a parameter derived inside."""
    kind: ClassVar[str] = 'buffer-bank'
    classes: ClassVar[frozenset[str]] = frozenset({'digital'})

    def build(self, circuit: CircuitBuilder, vdd: str) -> None:
        r = circuit.rng
        a = circuit.net('bb')
        circuit.source(a, 'pulse')
        for _ in range(r.randint(1, 4)):
            b = circuit.net('bb')
            circuit.elements.append(f'X{circuit.name("u")} {a} {b} {vdd} 0 {circuit.define("fbuf")} '
                                    f'wn={r.choice(["0.3u", "0.5u"])} ratio={r.choice([1, 2, 3])}'
                                    f'{circuit.multiplicity()}')
            a = b


@dataclass(frozen=True)
class LibraryChain(Block):
    """Inverters from a local .lib section whose cell comes from a nested .include."""
    kind: ClassVar[str] = 'library-chain'
    classes: ClassVar[frozenset[str]] = frozenset({'digital'})

    def build(self, circuit: CircuitBuilder, vdd: str) -> None:
        r = circuit.rng
        cell = circuit.library_cell()
        a = circuit.net('lb')
        circuit.source(a, 'pulse')
        for _ in range(r.randint(1, 4)):
            b = circuit.net('lb')
            sizing = r.choice(['', '', ' wn=0.3u', ' wp=1.2u'])  # section defaults or an override
            circuit.elements.append(f'X{circuit.name("y")} {a} {b} {vdd} 0 {cell}{sizing}')
            circuit.elements.append(f'C{circuit.name("l")} {b} 0 {eng(r.choice([1e-15, 5e-15]))}')
            a = b


@dataclass(frozen=True)
class BinnedChain(Block):
    """Inverters of built-in BSIM4 transistors whose lengths fall in either model bin."""
    kind: ClassVar[str] = 'binned-chain'
    classes: ClassVar[frozenset[str]] = frozenset({'digital', 'bsim4'})

    def build(self, circuit: CircuitBuilder, vdd: str) -> None:
        r = circuit.rng
        n, p = circuit.binned_models()
        a = circuit.net('bn')
        circuit.source(a, 'pulse')
        for _ in range(r.randint(1, 4)):
            b = circuit.net('bn')
            width, length = r.choice([0.3e-6, 0.5e-6, 1e-6]), r.choice([0.13e-6, 0.18e-6, 0.5e-6, 1e-6])
            circuit.elements += [f'M{circuit.name("m")} {b} {a} 0 0 {n} w={eng(width)} l={eng(length)}',
                                 f'M{circuit.name("m")} {b} {a} {vdd} {vdd} {p} w={eng(2 * width)} l={eng(length)}',
                                 f'C{circuit.name("l")} {b} 0 {eng(r.choice([1e-15, 2e-15, 5e-15]))}']
            a = b


# The order is part of the seed -> deck mapping.
BLOCKS: Sequence[Block] = (InverterChain(), RingOscillator(), Latch(), CurrentMirror(), DiffPair(),
                           RcLadder(), SeriesRlc(), ControlledSources(), DiodeClamp(), Behavioural(),
                           BufferBank(), LibraryChain(), BinnedChain())


@dataclass(frozen=True)
class Generator:
    """Random decks: the same seed (and scale) always gives the same deck.

    `scale` is the legacy size knob: above 1, every block is an array of `scale`
    identical copies, from tens of devices up to thousands. Large decks observe at most
    `max_observed` nets."""
    blocks: Sequence[Block] = BLOCKS
    scale: int = 1
    max_observed: int = 200

    def generate(self, seed: int) -> Deck:
        circuit = CircuitBuilder(seed)
        r = circuit.rng
        vdd = circuit.supply()
        for block in r.choices(self.blocks, k=r.randint(2, 6)):
            if self.scale == 1:
                circuit.add(block, vdd)
            else:
                circuit.add_array(block, vdd, self.scale)
        for node in circuit.nets:  # every net keeps a DC path
            circuit.elements.append(f'R{circuit.name("leak")} {node} 0 {eng(r.choice([1e9, 1e10]))}')
        method = r.choice(['gear', 'trap'])
        reltol = r.choice(['1e-3', '1e-4'])
        options = f'reltol={reltol} abstol=1e-13 vntol=1e-6 method={method}'
        if method == 'gear' and r.random() < 0.5:
            options += f' maxord={r.randint(2, 6)}'
        if r.random() < 0.3:
            options += f' trtol={r.choice([1, 3, 5, 10])}'
        held = {ic.split('(')[1].split(')')[0] for ic in circuit.ics}
        candidates = [n for n in circuit.nets if n not in held]
        if candidates and r.random() < 0.3:  # operating-point hints
            for net in r.sample(candidates, min(len(candidates), r.randint(1, 3))):
                circuit.nodesets.append(f'v({net})={r.choice([0.0, circuit.vdd]):.4g}')
        uic = r.random() < 0.5
        step = circuit.tstop / r.choice([200, 500, 1000])
        circuit.classify_resolution(step)
        analyses = self._analyses(circuit)
        temperature = r.choice([-40, 27, 27, 85])
        title = f'random circuit seed {seed}' + (f' scale {self.scale}' if self.scale > 1 else '')
        library = [f'.lib "fuzzlib.lib" {circuit.library_section}'] if circuit.library_section else []
        lines = [title,
                 *library,
                 f'.temp {temperature}',
                 f'.option {options}',
                 f'.options seed={seed % 100000 + 1}',
                 f'.param vsupply={circuit.vdd:.4g}', *circuit.functions.values(), *circuit.params,
                 *circuit.models,
                 '.model smith_n nmos level=54 version=4.8',
                 '.model smith_p pmos level=54 version=4.8']
        for kind in ('nmos', 'pmos'):
            model = 'smith_n' if kind == 'nmos' else 'smith_p'
            lines += [f'.subckt smith_{kind} d g s b w=0.5u l=0.13u ng=1 m=1',
                      f'M1 d g s b {model} w={{w*ng}} l={{l}} m={{m}}', '.ends']
        for subcircuit in circuit.subcircuits.values():
            lines += subcircuit.lines
        lines += circuit.elements
        lines += [f'.ic {ic}' for ic in circuit.ics]
        lines += [f'.nodeset {nodeset}' for nodeset in circuit.nodesets]
        outputs = ' '.join(f'v({n})' for n in circuit.observed(self.max_observed))
        transient = f'tran {eng(step)} {eng(circuit.tstop)}' + (' uic' if uic else '')
        for analysis, name in [*analyses, (transient, 'waveform.txt')]:
            lines += ['.' + analysis, f'* spicesmith-output {name} {outputs}']
        lines += ['.save ' + outputs, '.end']
        return Deck('\n'.join(lines) + '\n', dict(circuit.files), circuit.info(self.scale))


    @staticmethod
    def _analyses(circuit: CircuitBuilder) -> list[tuple[str, str]]:
        """Analyses besides the transient, with their output files: an
        operating point, a DC sweep of an input source, an AC sweep from one source made the
        AC input."""
        r = circuit.rng
        sources = [i for i, line in enumerate(circuit.elements) if line.startswith('Vs')]
        analyses = []
        if r.random() < 0.25:
            analyses.append(('op', 'op.txt'))
        if sources and r.random() < 0.25:
            name = circuit.elements[r.choice(sources)].split()[0]
            analyses.append((f'dc {name} 0 {circuit.vdd:.4g} {circuit.vdd / r.choice([6, 12, 24]):.4g}', 'dc.txt'))
        if sources and r.random() < 0.2:
            i = r.choice(sources)
            words = circuit.elements[i].split(None, 3)
            circuit.elements[i] = ' '.join(words[:3] + ['AC 1'] + words[3:])
            analyses.append((f'ac dec {r.choice([3, 5, 10])} 1meg {r.choice(["1g", "10g", "100g"])}', 'ac.txt'))
        return analyses


def generate(seed: int, scale: int = 1) -> Deck:
    """The deck of one seed, from the default generator."""
    return Generator(scale=scale).generate(seed)
