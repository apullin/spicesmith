"""What produced a result: binaries, frozen inputs, host and this tool (see README.md)."""
from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from . import __version__
from .config import Configuration, Simulators, executable
from .deck import Deck
from .generator import VERSION

REPO = Path(__file__).resolve().parent.parent


@functools.lru_cache(maxsize=None)
def _sha256(path: str, size: int, mtime: int) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    """sha256 of a file, cached by path, size and modification time."""
    st = os.stat(path)
    return _sha256(str(path), st.st_size, st.st_mtime_ns)


@dataclass(frozen=True)
class BinaryRecord:
    """A simulator binary and, for candidates, the build record next to it."""
    path: str
    sha256: str
    build: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def of(cls, path: Path) -> BinaryRecord:
        path = executable(path).resolve()
        build: dict[str, Any] = {}
        record = path.parent / 'build.json'
        if record.exists():
            build['build_json_sha256'] = sha256(record)
            try:
                build['metadata'] = json.loads(record.read_text())
            except (OSError, ValueError):
                pass
        return cls(str(path), sha256(path), build)


def frozen_inputs(deck: Deck | None = None) -> Mapping[str, str]:
    """Hash only this deck's external dependencies, following nested includes.

    Bundled files are part of the deck identity. Missing external dependencies
    cannot silently become reproducible evidence.
    """
    if deck is None:
        return {}
    hashes: dict[str, str] = {}
    seen: set[str] = set()

    def visit(text: str, base: Path) -> None:
        for line in text.splitlines():
            if not line.lower().lstrip().startswith(('.include ', '.inc ', '.lib ', 'pre_osdi ')):
                continue
            words = shlex.split(line)
            if len(words) < 2 or (words[0].lower() == '.lib' and len(words) == 2):
                continue  # .lib SECTION declares a section, not a dependency
            path = base / words[1]
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            if not path.is_absolute() and key in deck.files:
                visit(deck.files[key], path.parent)
            else:
                if not path.is_absolute():
                    raise ValueError(f'unbundled relative dependency: {path}; load it with Deck.load')
                path = path.resolve()
                hashes[str(path)] = sha256(path)
                if words[0].lower() != 'pre_osdi':
                    visit(path.read_text(), path.parent)

    visit(deck.text, Path('.'))
    return dict(sorted(hashes.items()))


@functools.lru_cache(maxsize=None)
def host() -> Mapping[str, Any]:
    cpu = ''
    try:
        with open('/proc/cpuinfo') as f:
            cpu = next(line.split(':', 1)[1].strip() for line in f if line.startswith('model name'))
    except (OSError, StopIteration):
        pass
    return {'hostname': platform.node(), 'kernel': platform.release(), 'cpu': cpu, 'cpus': os.cpu_count(),
            'python': sys.version.split()[0]}


@functools.lru_cache(maxsize=None)
def tool() -> Mapping[str, Any]:
    """This package: version, a hash of its code (what decides results), and the git revision
    of its checkout; when the checkout is modified, which tracked files and a hash of the
    diff, so the source state of a run can be told apart from its commit."""
    record: dict[str, Any] = {'version': __version__, 'code_sha256': code_sha256()}
    try:
        def git(*args: str) -> str:
            return subprocess.run(['git', '-C', str(REPO), *args], capture_output=True, text=True,
                                  timeout=10).stdout
        head = git('rev-parse', 'HEAD').strip()
        dirty = [line[3:] for line in git('status', '--porcelain', '--untracked-files=no').splitlines()]
        if head:
            record['commit'] = head + ('-dirty' if dirty else '')
        if dirty:
            record['dirty_files'] = dirty
            record['diff_sha256'] = hashlib.sha256(git('diff', 'HEAD').encode()).hexdigest()
    except (OSError, subprocess.SubprocessError):
        pass
    return record


def code_sha256() -> str:
    """sha256 over the package's Python sources: the code that generates, runs and judges."""
    digest = hashlib.sha256()
    for source in sorted(Path(__file__).resolve().parent.glob('*.py')):
        digest.update(source.name.encode() + b'\0' + source.read_bytes() + b'\0')
    return digest.hexdigest()


@dataclass(frozen=True)
class Provenance:
    """Everything shared by the cases of a batch that could change a result."""
    reference: BinaryRecord
    candidate: BinaryRecord
    extra_binaries: tuple[BinaryRecord, ...]  # the binaries extra configurations name
    inputs: Mapping[str, str]
    host: Mapping[str, Any]
    tool: Mapping[str, str]
    generator: int
    scale: int = 1  # the generator's size knob

    @classmethod
    def collect(cls, simulators: Simulators, configurations: Mapping[str, Configuration],
                scale: int = 1) -> Provenance:
        """The reference and candidate, and every other binary one of `configurations` names."""
        extra = sorted({c.binary for c in configurations.values() if isinstance(c.binary, Path)})
        return cls(BinaryRecord.of(simulators.reference), BinaryRecord.of(simulators.candidate),
                   tuple(BinaryRecord.of(path) for path in extra), frozen_inputs(), host(), tool(), VERSION, scale)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
