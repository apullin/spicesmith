import os
import stat
import shutil
from pathlib import Path

import pytest

from spicesmith.deck import Deck

FAKE = Path(__file__).parent / 'fake_ngspice.py'


@pytest.fixture
def fake() -> Path:
    """Path of the stand-in simulator (see fake_ngspice.py)."""
    os.chmod(FAKE, os.stat(FAKE).st_mode | stat.S_IXUSR)
    return FAKE


@pytest.fixture
def real_settings():
    """Opt-in real tests use a caller-selected ngspice, or the installed command."""
    from spicesmith.config import Simulators
    from spicesmith.harness import Settings
    binary = os.environ.get('SPICESMITH_NGSPICE') or shutil.which('ngspice')
    if not binary:
        pytest.skip('set SPICESMITH_NGSPICE or install ngspice for opt-in simulator tests')
    return Settings(Simulators(Path(binary), Path(binary)), timeout=10, front_end=False)


def fake_deck(*lines: str, outputs=('waveform.txt',)) -> Deck:
    """A minimal deck for the fake simulator."""
    body = ['fake deck', '.param vsupply=1.2', *lines, 'R1 a 0 1k', '.control', 'tran 1p 1n']
    body += [f'wrdata __OUT__/{name} v(a) v(b)' for name in outputs]
    return Deck('\n'.join(body + ['quit', '.endc', '.end']) + '\n')
