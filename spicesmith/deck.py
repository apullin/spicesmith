"""Immutable netlists and their analysis/observation contracts.

Plain decks use standard analysis directives and .save. Generated output comments
retain per-analysis vector requests; optional ngspice control decks are also accepted.
Transformations (tighter tolerances, a parse-only variant) return new decks.
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

OUT = '__OUT__'
ANALYSES = ('op', 'dc', 'ac', 'noise', 'tran')
SMALL_SIGNAL = ('ac', 'noise')
_OPTION_LINE = re.compile(r'(?im)^\.options?\b.*$')
_WRDATA = re.compile(rf'(?m)^wrdata {OUT}/(\S+)')

# The accuracy reference: tighter Newton and truncation tolerances. The first level is the
# original target; some decks cannot converge branch currents to abstol=1e-15 ("Timestep too
# small; ... trouble with node vdd#branch"), so the harness falls back level by level.
TIGHT_LEVELS: Sequence[Mapping[str, str]] = (
    {'reltol': '1e-6', 'abstol': '1e-15', 'vntol': '1e-8', 'trtol': '1'},
    {'reltol': '1e-6', 'abstol': '1e-14', 'vntol': '1e-8', 'trtol': '1'},
    {'reltol': '1e-6', 'abstol': '1e-13', 'vntol': '1e-8', 'trtol': '1'},
    {'reltol': '1e-5', 'abstol': '1e-13', 'vntol': '1e-7', 'trtol': '1'},
)


@dataclass(frozen=True)
class BlockInfo:
    """One block of a generated circuit: its kind, the circuit classes it belongs to (for
    per-class reporting) and the top-level nets it owns."""
    kind: str
    classes: frozenset[str]
    nets: tuple[str, ...]
    attributes: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CircuitInfo:
    """What the generator knows about a deck it made."""
    seed: int
    generator: int  # generator version
    vdd: float
    tstop: float
    blocks: tuple[BlockInfo, ...]
    scale: int = 1  # copies per block (size knob)

    def block_of(self, net: str) -> Optional[BlockInfo]:
        return next((b for b in self.blocks if net in b.nets), None)

    def to_json(self) -> dict[str, Any]:
        return {'seed': self.seed, 'generator': self.generator, 'scale': self.scale, 'vdd': self.vdd,
                'tstop': self.tstop,
                'blocks': [{'kind': b.kind, 'classes': sorted(b.classes), 'nets': list(b.nets),
                            **({'attributes': dict(b.attributes)} if b.attributes else {})}
                           for b in self.blocks]}


@dataclass(frozen=True)
class Deck:
    """A netlist with the auxiliary files it includes (written next to it) and, for generated
    decks, the generator's description of the circuit."""
    text: str
    files: Mapping[str, str] = field(default_factory=dict)
    info: Optional[CircuitInfo] = None

    # --- Reading ---------------------------------------------------------------------------

    @property
    def lines(self) -> list[str]:
        return self.text.split('\n')

    @property
    def outputs(self) -> list[str]:
        """Output files the deck writes, in order."""
        return list(self.output_requests)

    @property
    def output_requests(self) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
        """Each wrdata output's preceding analysis and requested vectors, in source order."""
        requests: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
        analysis: tuple[str, ...] = ()
        control = self._control()[1]
        explicit = any(line.startswith('* spicesmith-output ') for line in self.lines)
        vectors = tuple(v.lower() for line in self.lines if line.lower().startswith('.save ')
                        for v in line.split()[1:])
        for line in control or self.lines:
            words = line.split()
            if _is_analysis(line):
                analysis = tuple(words)
            elif words and words[0].lower() in {'.' + a for a in ANALYSES}:
                analysis = (words[0][1:].lower(), *words[1:])
                if vectors and not explicit:
                    name = analysis[0] + '.txt'
                    if name in requests:
                        name = f'{analysis[0]}-{len(requests)}.txt'
                    requests[name] = (analysis, vectors)
            elif len(words) >= 4 and words[:2] == ['*', 'spicesmith-output']:
                requests[words[2]] = (analysis, tuple(v.lower() for v in words[3:]))
            elif len(words) >= 3 and words[0].lower() == 'wrdata':
                path = words[1]
                name = path[len(OUT) + 1:] if path.startswith(OUT + '/') else Path(path).name
                requests[name] = (analysis, tuple(v.lower() for v in words[2:]))
        return requests

    @property
    def options(self) -> dict[str, Union[str, bool]]:
        """Values of the .option/.options lines (later lines win), lower-case keys."""
        values: dict[str, Union[str, bool]] = {}
        for line in _OPTION_LINE.findall(self.text):
            for token in line.split()[1:]:
                key, eq, value = token.partition('=')
                values[key.lower()] = value if eq else True
        return values

    @property
    def analyses(self) -> list[str]:
        """Analysis commands of the .control block, e.g. ['tran 10p 2n uic']."""
        control = self._control()[1]
        if control:
            return [line.strip() for line in control if _is_analysis(line)]
        return [line.strip()[1:] for line in self.lines
                if line.split() and line.split()[0].lower() in {'.' + a for a in ANALYSES}]

    @property
    def has_small_signal_analysis(self) -> bool:
        return any(a.split()[0].lower() in SMALL_SIGNAL for a in self.analyses)

    @property
    def tstop(self) -> Optional[float]:
        """Stop time of the first transient analysis."""
        for analysis in self.analyses:
            words = analysis.split()
            if words[0].lower() == 'tran':
                return spice_number(words[2])
        return None

    @property
    def vdd(self) -> Optional[float]:
        match = re.search(r'(?im)^\.param vsupply=(\S+)', self.text)
        return spice_number(match.group(1)) if match else None

    def bound_to(self, directory: Path) -> str:
        """The text as run in `directory`: outputs go there."""
        return self.text.replace(OUT, str(directory))

    # --- Files -----------------------------------------------------------------------------

    def save(self, directory: Path, name: str = 'deck.sp') -> Path:
        """Write the deck and the files it includes into `directory`."""
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(self.text)
        self.write_files(directory)
        return directory / name

    def write_files(self, directory: Path) -> None:
        """Write the files the deck includes next to it (nested paths get their directories);
        a path that is absolute or climbs out of the directory is refused."""
        for file, content in self.files.items():
            path = Path(file)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError(f'an included file must stay inside the deck directory: {file}')
            (directory / path).parent.mkdir(parents=True, exist_ok=True)
            (directory / path).write_text(content)

    @classmethod
    def load(cls, directory: Path, name: str = 'deck.sp') -> Deck:
        """A deck and the local files it includes (relative .lib and .include paths, followed
        into the included files)."""
        text = (directory / name).read_text()
        files: dict[str, str] = {}
        pending = list(_local_includes(text))
        while pending:
            file = pending.pop()
            if file not in files and (directory / file).is_file():
                files[file] = (directory / file).read_text()
                pending += _local_includes(files[file])
        return cls(text, files)

    # --- Transformations -------------------------------------------------------------------

    def with_text(self, text: str) -> Deck:
        return dataclasses.replace(self, text=text)

    def with_options(self, values: Mapping[str, str]) -> Deck:
        """Replace or add `key=value` options: existing keys are rewritten in place wherever
        they appear, missing keys go at the end of the first option line."""
        lines = self.lines
        pending = dict(values)
        first: Optional[int] = None
        for i, line in enumerate(lines):
            if not _OPTION_LINE.fullmatch(line):
                continue
            first = i if first is None else first
            tokens = line.split()
            for j, token in enumerate(tokens[1:], 1):
                key = token.partition('=')[0].lower()
                if key in values:
                    tokens[j] = f'{key}={values[key]}'
                    pending.pop(key, None)
            lines[i] = ' '.join(tokens)
        if pending:
            extra = ' '.join(f'{k}={v}' for k, v in pending.items())
            if first is None:
                lines.insert(1, f'.option {extra}')
            else:
                lines[first] += ' ' + extra
        return self.with_text('\n'.join(lines))

    def tightened(self, level: int = 0) -> Deck:
        """The accuracy reference deck at one of TIGHT_LEVELS."""
        return self.with_options(TIGHT_LEVELS[level])

    def front_end(self) -> Deck:
        """Parsed but not simulated: the .control block keeps its pre_ commands and settings,
        drops analyses and output commands, and quits."""
        from .ngspice import prepare
        adapted = prepare(self)
        head, body, tail = adapted._control()
        keep = [line for line in body if not _is_analysis(line)
                and not line.lower().startswith(('wrdata', 'rusage', 'quit'))]
        return self.with_text('\n'.join(head + ['.control', *keep, 'quit', '.endc'] + tail))

    # --- Internals -------------------------------------------------------------------------

    def _control(self) -> tuple[list[str], list[str], list[str]]:
        """(lines before .control, the .control body, lines after .endc)."""
        lines = self.lines
        try:
            start = next(i for i, line in enumerate(lines) if line.strip().lower() == '.control')
            end = next(i for i, line in enumerate(lines) if i > start and line.strip().lower() == '.endc')
        except StopIteration:
            return lines, [], []
        return lines[:start], lines[start + 1:end], lines[end + 1:]


_INCLUDE = re.compile(r'(?im)^\.(?:lib|include|inc)\s+"?([^"\s]+)"?')


def _local_includes(text: str) -> list[str]:
    """Relative paths of the files a netlist includes (.lib FILE SECTION, .include FILE)."""
    return [path for path in _INCLUDE.findall(text) if not path.startswith(('/', '$', '~'))]


def _is_analysis(line: str) -> bool:
    words = line.split()
    return bool(words) and words[0].lower() in ANALYSES


_SUFFIX = {'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3, 'k': 1e3, 'meg': 1e6, 'g': 1e9,
           't': 1e12}
_NUMBER = re.compile(r'([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)(meg|[fpnumkgt])?[a-z]*')


def spice_number(token: str) -> float:
    """A SPICE number such as '10p', '1.5meg', '2e-9' or '10pF'."""
    match = _NUMBER.fullmatch(token.lower())
    if not match:
        raise ValueError(f'not a SPICE number: {token}')
    return float(match.group(1)) * _SUFFIX.get(match.group(2), 1.0)


def eng(value: float) -> str:
    """A SPICE number with a scale suffix, exact enough for a deck."""
    for scale, suffix in ((1e-15, 'f'), (1e-12, 'p'), (1e-9, 'n'), (1e-6, 'u'), (1e-3, 'm'),
                          (1, ''), (1e3, 'k'), (1e6, 'meg'), (1e9, 'g')):
        if abs(value) < scale * 1000:
            return f'{value / scale:.4g}{suffix}'
    return f'{value:.4g}'
