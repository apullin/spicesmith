"""Command line (see README.md for all workflows and usage contracts).

  spicesmith gen SEED OUT.sp
  spicesmith run --out DIR [--start N] [--cases N | --cases 0] [--resume] [--jobs N] ...
  spicesmith reduce CASE_DIR CHECK [--max-trials N] ...
  spicesmith judge DIR...
  spicesmith split CASE_DIR CHECK
  spicesmith report DIR... [--json FILE]
  spicesmith regress [--corpus DIR] [--candidate BIN]
  spicesmith adopt REDUCED_DIR NAME --expect pass|known --origin TEXT [--notes TEXT] [--requires NAME]
"""
from __future__ import annotations

import argparse
import functools
import json
import shutil
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Optional, Sequence

from . import generator, oracles
from .attribution import FlagSplit
from .corpus import CORPUS, Corpus, Entry, Expectation, Status, normalized
from .config import CANDIDATE, CONFIGURATIONS, CPUS, REFERENCE, TIMEOUT, Simulators, load_extra_configurations
from .deck import Deck
from .harness import Batch, Settings, Testbench, load_evidence
from .host import check_cpus
from .provenance import Provenance
from .reduce import FindingTest, Reducer, Reproducer, TrialBudget
from .report import Report


class Command(ABC):
    """One subcommand: its arguments and what it does."""
    name: ClassVar[str]
    help: ClassVar[str]

    @abstractmethod
    def configure(self, parser: argparse.ArgumentParser) -> None:
        """Add this command's arguments."""

    @abstractmethod
    def execute(self, args: argparse.Namespace) -> None:
        """Run the command."""


def add_simulator_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--candidate', type=Path, default=CANDIDATE,
                        help='candidate ngspice binary (default: ngspice on PATH)')
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--extra-configurations', type=Path, metavar='JSON',
                        help='configurations to run besides the built-in ones (see docs/USAGE.md)')
    parser.add_argument('--timeout', type=float, default=TIMEOUT, help='seconds per simulator run')
    parser.add_argument('--lock-wait', type=float, default=0.0,
                        help="retry a configuration's busy lock for up to this many seconds (default: fail fast)")
    parser.add_argument('--threads', type=int, default=1, help='threads per simulator run (default: 1)')


def settings_from(args: argparse.Namespace, check_reference: bool = True) -> Settings:
    """Caller-selected binaries, resources and environments."""
    check_cpus(args.cpus)
    simulators = Simulators(args.reference, args.candidate)
    configurations = dict(CONFIGURATIONS)
    if args.extra_configurations:
        configurations.update(load_extra_configurations(args.extra_configurations))
    settings = Settings(simulators, configurations, args.lock_wait,
                        args.timeout, front_end=not getattr(args, 'no_front', False),
                        repeat_every=getattr(args, 'repeat_every', 10), keep_outputs=getattr(args, 'keep', False),
                        threads=args.threads)
    if check_reference:
        settings.check_reference()
    return settings


class Generate(Command):
    name = 'gen'
    help = 'write the deck of one seed'

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('seed', type=int)
        parser.add_argument('out', type=Path)
        parser.add_argument('--scale', type=int, default=1, help='copies per block (size knob)')
        from .profiles import RECIPES
        parser.add_argument('--profile', choices=['legacy', *RECIPES], default='heterogeneous-continuous')
        parser.add_argument('--profile-config', type=Path, help='JSON overrides for a structural profile')
        parser.add_argument('--transistors', type=int, help='structural target; independent of repetition')
        parser.add_argument('--disposition', choices=['digital', 'analog', 'mixed'])

    def execute(self, args: argparse.Namespace) -> None:
        """The deck and the files it includes, side by side: a runnable bundle."""
        if args.profile == 'legacy':
            if args.profile_config or args.transistors is not None or args.disposition:
                raise SystemExit('structural controls require an explicit non-legacy --profile')
            generator.generate(args.seed, args.scale).save(args.out.parent, args.out.name)
            return
        from .extension_cli import generate_to, profile_from
        if args.scale != 1:
            raise SystemExit('--scale is legacy-only; structural profiles separate size and repetition')
        generate_to(args.seed, profile_from(args), args.out)


class Run(Command):
    name = 'run'
    help = 'run a batch of random cases'

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('--out', type=Path, required=True, help='batch directory')
        parser.add_argument('--start', type=int, default=1, help='first seed (default %(default)s)')
        parser.add_argument('--cases', type=int, default=20, help='number of seeds; 0 runs until interrupted')
        parser.add_argument('--resume', action='store_true', help='continue after the last finished seed')
        parser.add_argument('--fresh', action='store_true', help='rerun seeds that already have a summary')
        parser.add_argument('--force', action='store_true',
                            help='add to a batch made with another candidate or generator')
        parser.add_argument('--jobs', type=int, default=2, help='cases at a time (default: 2)')
        parser.add_argument('--cpus', default=CPUS, help='optional CPUs to split into sets of --threads CPUs')
        parser.add_argument('--repeat-every', type=int, default=10, help='run ref twice every N-th seed; 0 never')
        parser.add_argument('--no-front', action='store_true', help='skip the front-end identity runs')
        parser.add_argument('--keep', action='store_true', help='keep the outputs of passing cases')
        parser.add_argument('--scale', type=int, default=1,
                            help='copies per block: 1 gives tens of devices, 50 about a thousand')
        add_simulator_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        settings = settings_from(args)
        provenance = Provenance.collect(settings.simulators, settings.configurations, args.scale)
        batch = Batch(args.out, settings, provenance, functools.partial(generator.generate, scale=args.scale))
        batch.record({k: str(v) for k, v in vars(args).items() if k != 'command'}, args.force)
        batch.run(batch.seeds(args.start, args.cases, args.resume, args.fresh), args.jobs, args.cpus)


class Reduce(Command):
    name = 'reduce'
    help = 'reduce a failing case to a small reproducer'

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('case', type=Path, help='case directory (with deck.sp)')
        parser.add_argument('check', help='finding prefix, e.g. "exact:" or "approx: crossing"')
        parser.add_argument('--out', type=Path, help='output directory (default CASE/reduced)')
        parser.add_argument('--max-trials', type=int, default=2000,
                            help='maximum search trials; initial verification and final replay are additional')
        parser.add_argument('--cpus', help='optional CPU affinity for trial runs')
        add_simulator_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        settings = settings_from(args)
        case = args.case.resolve()
        deck = Deck.load(case)
        out = (args.out or case / 'reduced').resolve()
        test = FindingTest(Testbench(settings, args.cpus), args.check, out / 'work')
        initial = test.evidence(deck)
        original = oracles.judge(initial)
        matching = [f for f in original.findings if f.startswith(args.check)]
        if not matching:
            raise SystemExit(f'{case}: no finding starts with {args.check!r}; findings: {list(original.findings)}')
        test.baseline = initial.reference.diagnostics
        print(f'reducing {len(deck.lines)} lines for: {matching[0]}', flush=True)
        reduced = deck
        budget = TrialBudget(test, args.max_trials)
        for step in Reducer().steps(deck, budget):
            reduced = step.deck
            print(f'{step.reduction}: {step.size} lines ({test.trials} trials)', flush=True)
        evidence = test.evidence(reduced)
        final = [f for f in oracles.judge(evidence).findings if f.startswith(args.check)]
        if not final:
            reduced.save(out, 'reduced.sp')
            (out / 'reduce.json').write_text(json.dumps({'case': str(case), 'check': args.check,
                                                        'finding': matching[0], 'reduced_finding': [],
                                                        'state': 'not-reproduced', 'search_trials': budget.used},
                                                       indent=2))
            raise SystemExit('final verification did not reproduce; evidence retained, no verified bundle issued')
        reproducer = Reproducer(test, final[0], evidence.tight_level or 0)
        for name, content in reproducer.files(reduced).items():
            (out / name).parent.mkdir(parents=True, exist_ok=True)
            (out / name).write_text(content)
        (out / 'reduce.json').write_text(json.dumps({
            'case': str(case), 'check': args.check, 'finding': matching[0], 'reduced_finding': final,
            'lines': [len(deck.lines), len(reduced.lines)], 'trials': test.trials, 'search_trials': budget.used,
            'provenance': Provenance.collect(settings.simulators,
                                             {n: settings.configurations[n] for n in test.configurations()}).to_json()},
            indent=1))
        shutil.rmtree(out / 'work', ignore_errors=True)
        print(out / 'reduced.sp')


class Judge(Command):
    name = 'judge'
    help = 'judge the runs stored in a case or reproducer directory again'

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('directories', type=Path, nargs='+', help='case or reproducer directories')

    def execute(self, args: argparse.Namespace) -> None:
        for directory in args.directories:
            verdict = oracles.judge(load_evidence(directory.resolve()))
            failed = [k for k, v in verdict.checks.items() if v is oracles.Outcome.FAIL]
            print(f'{directory}: {len(verdict.checks)} checks, {len(failed)} failed')
            for heading, lines in (('finding', verdict.findings), ('note', verdict.notes),
                                   ('inconclusive', verdict.inconclusive)):
                for line in lines:
                    print(f'  {heading}: {line}')


class Split(Command):
    name = 'split'
    help = "find which of a configuration's flags a finding needs"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('case', type=Path, help='case directory (with deck.sp)')
        parser.add_argument('check', help='finding prefix, e.g. "approx: mid-rail"')
        parser.add_argument('--base', default='exact', help='configuration whose flags to start from')
        parser.add_argument('--cpus', help='optional CPU affinity for runs')
        add_simulator_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        case = args.case.resolve()
        split = FlagSplit(Testbench(settings_from(args), args.cpus), args.check, args.base)
        trials = []
        for trial in split.trials(Deck.load(case), case / 'split'):
            trials.append(trial)
            print(f"{trial.variant.label:45s} {'reproduces' if trial.reproduces else 'passes'}", flush=True)
        print(split.conclusion(trials))
        (case / 'split.json').write_text(json.dumps({
            'check': args.check, 'base': args.base, 'conclusion': split.conclusion(trials),
            'trials': [{'variant': t.variant.label, 'flags': dict(t.variant.flags), 'reproduces': t.reproduces,
                        'findings': list(t.findings)} for t in trials]}, indent=1))
        shutil.rmtree(case / 'split', ignore_errors=True)


class Aggregate(Command):
    name = 'report'
    help = 'aggregate batches: pass rates, accuracy distributions, finding clusters'

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('batches', type=Path, nargs='+', help='batch directories')
        parser.add_argument('--json', type=Path, help='also write the aggregates as JSON')

    def execute(self, args: argparse.Namespace) -> None:
        report = Report.load(args.batches)
        print(report.markdown(), end='')
        if args.json:
            args.json.write_text(json.dumps(report.to_json(), indent=1))


class Regress(Command):
    name = 'regress'
    help = 'run the regression corpus on a candidate (exit status 1 on a regression)'

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('--corpus', type=Path, default=CORPUS)
        parser.add_argument('--work', type=Path, default=Path('runs/regress'), help='run directory')
        parser.add_argument('--cpus', help='optional CPU affinity for runs')
        add_simulator_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        testbench = Testbench(settings_from(args), args.cpus)
        regressions = unresolved = evaluated = 0
        for result in Corpus(args.corpus).run(testbench, args.work.resolve()):
            regressions += result.status is Status.REGRESSION
            unresolved += result.status is Status.INCONCLUSIVE
            evaluated += result.status is not Status.SKIPPED
            print(f'{result.status.value:10s} {result.entry.name}', flush=True)
            for finding in result.unexpected:
                print(f'           {finding}')
            for reason in result.unresolved:
                print(f'           {reason}')
        if regressions:
            print(f'{regressions} regression(s): not promotable')
            sys.exit(1)
        if unresolved or not evaluated:
            print(f'{unresolved} inconclusive, {evaluated} evaluated: not promotable')
            sys.exit(2)
        print('promotable')


class Adopt(Command):
    name = 'adopt'
    help = 'add a reduced reproducer to the regression corpus'

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('reduced', type=Path, help='output directory of the reduce command')
        parser.add_argument('entry', help='name of the corpus entry')
        parser.add_argument('--expect', choices=[e.value for e in Expectation], required=True,
                            help='pass: a fixed bug; known: an accepted weakness')
        parser.add_argument('--origin', required=True, help='batch, seed and candidate it came from')
        parser.add_argument('--notes', default='')
        parser.add_argument('--requires', action='append', default=[], metavar='NAME',
                            help='a configuration the entry needs, e.g. one from --extra-configurations (repeatable)')
        parser.add_argument('--corpus', type=Path, default=CORPUS)

    def execute(self, args: argparse.Namespace) -> None:
        record = json.loads((args.reduced / 'reduce.json').read_text())
        expected = tuple(sorted({normalized(f) for f in record.get('reduced_finding') or [record['finding']]}))
        entry = Entry(args.entry, Deck.load(args.reduced, 'reduced.sp'), record['check'],
                      Expectation(args.expect), record['finding'], args.origin, args.notes, tuple(args.requires),
                      expected if args.expect == Expectation.KNOWN.value else ())
        print(entry.save(args.corpus))


COMMANDS: Sequence[Command] = (Generate(), Run(), Reduce(), Judge(), Split(), Aggregate(), Regress(), Adopt())


def main(argv: Optional[Sequence[str]] = None) -> None:
    from .extension_cli import COMMANDS as EXTENSION_COMMANDS
    parser = argparse.ArgumentParser(prog='spicesmith', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)
    commands = {}
    for command in (*COMMANDS, *EXTENSION_COMMANDS):
        command.configure(subparsers.add_parser(command.name, help=command.help))
        commands[command.name] = command
    args = parser.parse_args(argv)
    try:
        commands[args.command].execute(args)
    except (OSError, ValueError) as error:
        parser.exit(2, f'spicesmith: {error}\n')
