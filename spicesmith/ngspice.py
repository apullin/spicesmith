"""Optional ngspice adapter. Circuit emission itself uses ordinary SPICE directives."""
from __future__ import annotations

from .deck import ANALYSES, Deck


def prepare(deck: Deck, threads: int = 1) -> Deck:
    """Add output collection to a plain netlist; existing control decks are left intact.

    Generated output comments associate each analysis with its requested vectors.
    Hand-written decks can instead use .save; without an observation contract we
    refuse execution rather than report success without collecting any evidence.
    """
    if type(threads) is not int or threads < 1:
        raise ValueError('threads must be a positive integer')
    if any(line.strip().lower() == '.control' for line in deck.lines):
        return deck
    requests = deck.output_requests
    if not requests:
        raise ValueError('ngspice observation requires analyses and .save vectors or spicesmith-output comments')
    lines = [line for line in deck.lines if line.strip().lower() != '.end'
             and not (line.split() and line.split()[0].lower().lstrip('.') in ANALYSES)
             and not line.lower().startswith('.save ')]
    lines += ['.control', 'set noaskquit', f'set num_threads={threads}',
              'set wr_vecnames', 'set wr_singlescale', 'set numdgt=15']
    for name, (analysis, vectors) in requests.items():
        lines += [' '.join(analysis), f'wrdata __OUT__/{name} ' + ' '.join(vectors)]
    return deck.with_text('\n'.join(lines + ['rusage all', 'quit', '.endc', '.end', '']))
