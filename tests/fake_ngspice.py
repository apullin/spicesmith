#!/usr/bin/env python3
"""A stand-in for ngspice in the harness tests: `fake_ngspice.py -n -b [-D ngdebug] deck.sp`.

It writes every `wrdata FILE v(a) ...` output as a small deterministic table and prints a
log with rusage counters. Comment lines in the deck steer it:

  * fake: abort            the transient aborts (exit status still 0, no output)
  * fake: abort-if TOKEN   the same, if an .option line contains TOKEN
  * fake: exit N           exit with status N after writing the outputs
  * fake: sleep S          sleep S seconds first
  * fake: warn TEXT        print 'Warning: TEXT'

A relative .lib or .include file that is not next to the deck stops it, as in ngspice.

and so do environment variables, to make one configuration differ from another:

  FAKE_SHIFT=x             add x to every output value
  FAKE_WARN=TEXT           print 'Warning: TEXT'
  FAKE_COMMENT_MODELS=1    with -D ngdebug, comment out .model lines inside subcircuits
                           the way the candidate does ('*model NAME ...', scope dropped)
"""
import os
import math
import re
import sys
import time


def main():
    args = sys.argv[1:]
    debug = 'ngdebug' in args
    deck = open(args[-1]).read().splitlines()
    directives = [line[len('* fake:'):].split(None, 1) for line in deck if line.startswith('* fake:')]
    log = ['', 'Circuit: ' + deck[0], '']
    status, aborted = 0, False
    for word, *rest in directives:
        if word == 'sleep':
            time.sleep(float(rest[0]))
        elif word == 'abort':
            aborted = True
        elif word == 'abort-if':
            aborted = aborted or any(rest[0] in line.split() for line in deck if line.lower().startswith('.option'))
        elif word == 'exit':
            status = int(rest[0])
        elif word == 'warn':
            log.append('Warning: ' + rest[0])
    if os.environ.get('FAKE_WARN'):
        log.append('Warning: ' + os.environ['FAKE_WARN'])
    for line in deck:  # like ngspice, relative includes resolve next to the deck
        words = line.replace('"', ' ').split()
        if words and words[0].lower() in ('.lib', '.include') and len(words) > 1 and not words[1].startswith('/'):
            if not os.path.exists(words[1]):
                print(f'Error: Could not find include file {words[1]}')
                sys.exit(1)
    if debug:
        write_debug(deck, os.environ.get('FAKE_COMMENT_MODELS') == '1')
    shift = float(os.environ.get('FAKE_SHIFT', '0'))
    if aborted:
        log += ['doAnalyses: TRAN:  Timestep too small; time = 1e-09, timestep = 1e-23: trouble with node "x"',
                'tran simulation(s) aborted']
    else:
        analysis = ['tran', '1p', '1n']
        for line in deck:
            words = line.split()
            if words and words[0].lower() in ('op', 'tran', 'dc', 'ac'):
                analysis = words
            if line.startswith('wrdata '):
                write_table(line.split()[1], line.split()[2:], shift, analysis)
    elements = [line for line in deck[1:] if line[:1].isalpha()
                and not line.lower().startswith(('wrdata', 'set', 'tran', 'quit'))]
    log += [f'Number of lines in the deck = {len(deck)}', f'Circuit Equations = {len(elements)}',
            'Transient iterations = 100', 'Total elapsed time (seconds) = 0.01']
    print('\n'.join(log))
    sys.exit(status)


def number(word):
    match = re.fullmatch(r'([-+0-9.eE]+)(meg|[fpnumkg]?)', word)
    return float(match[1]) * {'': 1, 'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6,
                             'm': 1e-3, 'k': 1e3, 'meg': 1e6, 'g': 1e9}[match[2]]


def write_table(path, vectors, shift, analysis):
    kind = analysis[0].lower()
    scale = {'op': vectors[0], 'tran': 'time', 'ac': 'frequency', 'dc': 'v-sweep'}[kind]
    if kind == 'tran':
        start = number(analysis[3]) if len(analysis) > 3 and analysis[3].lower() != 'uic' else 0.0
        points = [start + (number(analysis[2]) - start) * i / 10 for i in range(11)]
    elif kind == 'dc':
        start, stop, step = map(number, analysis[2:5])
        points = [start + step * i for i in range(math.floor((stop - start) / step + 1e-8) + 1)]
    elif kind == 'ac':
        count, start, stop = int(analysis[2]), number(analysis[3]), number(analysis[4])
        if analysis[1] == 'lin':
            points = [start + (stop - start) * i / (count - 1) for i in range(count)]
        else:
            base = 10 if analysis[1] == 'dec' else 2
            points = [start * base ** (i / count)
                      for i in range(math.floor(math.log(stop / start, base) * count + 1e-8) + 1)]
    else:
        points = [0.0]
    names = [name for name in vectors for _ in range(2 if kind == 'ac' else 1)]
    rows = [' ' + scale + ' ' + ' '.join(names)]
    for i, point in enumerate(points):
        values = [(k + 1) * 0.1 * (i % 3) + shift for k in range(len(vectors))]
        if kind == 'ac':
            values = [v for value in values for v in (value, 0.0)]
        rows.append(' '.join(f'{v:.15e}' for v in [point, *values]))
    with open(path, 'w') as f:
        f.write('\n'.join(rows) + '\n')


def write_debug(deck, comment_models):
    scope = None
    lines = []
    for n, line in enumerate(deck, 1):
        words = line.split()
        if line.lower().startswith('.subckt'):
            scope = words[1]
        elif line.lower().startswith('.ends'):
            scope = None
        if scope and line.lower().startswith('.model'):
            scoped = f'.model x1:{words[1]} ' + ' '.join(words[2:])
            line = ('*model ' + ' '.join(words[1:])) if comment_models else scoped
        lines.append(f'{n:6d}  {n:6d}  {line}')
    for name in ('debug-out2.txt', 'debug-out3.txt'):
        with open(name, 'w') as f:
            f.write('**************** uncommented deck **************\n\n')
            f.write('\n'.join(line for line in lines if not line[16:].startswith('*')) + '\n')
            f.write('\n****************** complete deck ***************\n\n')
            f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
