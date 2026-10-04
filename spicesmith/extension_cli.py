"""Opt-in extension commands. Existing run/reduce/regress remain the legacy workflow."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from . import benchmark, experiment, population, profiles
from .circuit import Circuit
from .cli import Command, add_simulator_arguments, settings_from
from .config import Binary
from .deck import Deck
from .harness import Testbench
from .plans import ExecutionPlan
from .policies import POLICIES, Assessment, Policy
from .predicates import Predicate, PredicateTest
from .reduce import Reducer, TrialBudget, bundle_files


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def profile_from(args: argparse.Namespace) -> profiles.Profile:
    controls = json.loads(args.profile_config.read_text()) if args.profile_config else {}
    if getattr(args, 'transistors', None) is not None:
        controls['target_transistors'] = args.transistors
    if getattr(args, 'disposition', None):
        controls['disposition'] = args.disposition
        if 'weights' not in controls:
            controls['weights'] = profiles.WEIGHTS
    return profiles.profile(args.profile, controls)


def generate_to(seed: int, p: profiles.Profile, path: Path) -> profiles.Generated:
    try:
        generated = profiles.generate(seed, p)
    except profiles.GenerationError as error:
        write_json(path.with_suffix('.generation.json'), {'seed': seed, 'profile': p.to_json(),
                                                        'state': 'rejected', 'attempts': error.history})
        raise
    generated.circuit.lower().save(path.parent, path.name)
    write_json(path.with_suffix('.circuit.json'), generated.circuit.to_json())
    write_json(path.with_suffix('.generation.json'), generated.metadata())
    return generated


def policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--policy', choices=POLICIES, default='observe')
    parser.add_argument('--baseline', default='ref')
    parser.add_argument('--policy-options', type=Path, help='JSON difference thresholds: absolute, relative, floor')


def policy_from(args: argparse.Namespace) -> Policy:
    return Policy(args.policy, args.baseline,
                  json.loads(args.policy_options.read_text()) if args.policy_options else {})


def execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--configurations', default='ref,exact', help='explicit comma-separated execution plan')
    parser.add_argument('--cpus', help='optional CPU affinity; unpinned by default')
    parser.add_argument('--front', action='store_true', help='include ref/exact parse-only comparison')
    parser.add_argument('--repeat-reference', action='store_true')
    parser.add_argument('--repetitions', type=int, default=1, help='timing samples per configuration')
    parser.add_argument('--tight-level', type=int, choices=range(4), help='freeze an explicitly requested tight run')
    policy_arguments(parser)
    add_simulator_arguments(parser)


def execution_from(args: argparse.Namespace) -> tuple[Testbench, ExecutionPlan]:
    plan = ExecutionPlan(tuple(n.strip() for n in args.configurations.split(',')), args.front,
                         args.repeat_reference, args.repetitions, args.tight_level)
    bench = Testbench(settings_from(args, check_reference=False), args.cpus)
    unknown = set(plan.names) - bench.configurations.keys()
    if unknown:
        raise ValueError(f'unknown configurations: {sorted(unknown)}')
    if any(bench.configurations[n].binary is Binary.REFERENCE for n in plan.names):
        bench.settings.check_reference()
    return bench, plan


def display(result: Assessment) -> None:
    print(f'{result.policy.name}: {result.state}')
    for reason in (*result.findings, *result.reasons, *result.selected):
        print(f'  {reason}')
    if result.state in ('fail', 'inconclusive'):
        raise SystemExit(1 if result.state == 'fail' else 2)


class Observe(Command):
    name = 'observe'
    help = 'execute an explicit plan; default policy records observations, not accuracy verdicts'

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('deck', type=Path)
        parser.add_argument('--out', type=Path, required=True)
        execution_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        bench, plan = execution_from(args)
        path = args.deck / 'deck.sp' if args.deck.is_dir() else args.deck
        meta = path.with_suffix('.generation.json')
        metadata = json.loads(meta.read_text()) if meta.exists() else {}
        deck = Deck.load(path.parent, path.name)
        display(experiment.run(deck, bench, plan, args.out, policy_from(args), metadata))


class Evaluate(Command):
    name = 'evaluate'
    help = 'apply a policy to saved evidence without rerunning the simulator'

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('case', type=Path)
        policy_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        display(experiment.evaluate(args.case, policy_from(args)))


class Transform(Command):
    name = 'transform'
    help = 'transform a saved circuit graph, retaining parent provenance'

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('circuit', type=Path)
        parser.add_argument('operation', choices=['duplicate', 'diversify', 'reconnect', 'excitation'])
        parser.add_argument('--seed', type=int, required=True)
        parser.add_argument('--out', type=Path, required=True)

    def execute(self, args: argparse.Namespace) -> None:
        parent = Circuit.from_json(json.loads(args.circuit.read_text()))
        result = profiles.transform(parent, args.operation, args.seed)
        result.circuit.lower().save(args.out.parent, args.out.name)
        write_json(args.out.with_suffix('.circuit.json'), result.circuit.to_json())
        write_json(args.out.with_suffix('.generation.json'), result.metadata())


class Explore(Command):
    name = 'explore'
    help = 'stratified generation with optional execution and bounded coverage feedback'

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('spec', type=Path, help='JSON with a strata list (name, profile, quota, target)')
        parser.add_argument('--out', type=Path, required=True)
        parser.add_argument('--seed', type=int, default=1)
        parser.add_argument('--budget', type=int, help='maximum cases including rejections; default sum of quotas')
        parser.add_argument('--feedback', action='store_true', help='spend extra budget on unmet coverage targets')
        parser.add_argument('--execute', action='store_true', help='run each case; default is generation only')
        parser.add_argument('--resume', action='store_true')
        execution_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        strata = tuple(population.Stratum.from_json(s) for s in json.loads(args.spec.read_text())['strata'])
        bench, plan = execution_from(args) if args.execute else (None, ExecutionPlan())
        if not args.execute and args.policy != 'observe':
            raise ValueError('a comparison policy needs --execute')
        budget = args.budget if args.budget is not None else sum(s.quota for s in strata)
        result = population.Campaign(args.out, strata, args.seed, budget, args.feedback, bench, plan,
                                     policy_from(args)).run(args.resume)
        print(f'{result["issued_cases"]} cases; states: {result["states"]}; coverage: {args.out / "coverage.json"}')


class Freeze(Command):
    name = 'freeze'
    help = 'freeze all generated campaign cases into a separate identified benchmark population'

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('campaign', type=Path)
        parser.add_argument('--out', type=Path, required=True)

    def execute(self, args: argparse.Namespace) -> None:
        manifest = population.freeze(args.campaign, args.out)
        print(f'{len(manifest["cases"])} frozen workloads; {len(manifest["excluded"])} cases without decks excluded')


class Benchmark(Command):
    name = 'benchmark'
    help = 'repeated comparisons on a frozen population; incomplete runs never count as speedups'

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('population', type=Path)
        parser.add_argument('--out', type=Path, required=True)
        execution_arguments(parser)
        parser.set_defaults(repetitions=3)

    def execute(self, args: argparse.Namespace) -> None:
        bench, plan = execution_from(args)
        benchmark.run(args.population, args.out, bench, plan, policy_from(args), args.baseline)
        print(args.out / 'benchmark.json')


class ReplayCase(Command):
    name = 'replay-case'
    help = 'replay sealed evidence with its saved execution plan, independently of the scheduler'

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('case', type=Path)
        parser.add_argument('--out', type=Path, required=True)
        policy_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        display(experiment.replay(args.case, args.out, policy_from(args)))


class Minimize(Command):
    name = 'minimize'
    help = 'reduce under an explicit JSON predicate and execution plan, then package a verified replay'

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('case', type=Path)
        parser.add_argument('predicate', type=Path)
        parser.add_argument('--out', type=Path, required=True)
        parser.add_argument('--max-trials', type=int, default=200,
                            help='search calls, excluding baseline/final verification')
        execution_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        if args.out.exists():
            raise ValueError('minimize destination must not exist')
        if args.policy != 'observe':
            raise ValueError('minimize selects interpretation through the predicate JSON, not --policy')
        predicate = Predicate.from_json(json.loads(args.predicate.read_text()))
        bench, plan = execution_from(args)
        if predicate.kind == 'slowdown' and plan.repetitions < 3:
            raise ValueError('slowdown reduction needs --repetitions >=3')
        path = args.case / 'deck.sp' if args.case.is_dir() else args.case
        deck = Deck.load(path.parent, path.name)
        test = PredicateTest(bench, plan, predicate, args.out / 'work')
        initial = predicate.evaluate(test.evidence(deck))
        if initial.state != 'matched':
            raise SystemExit(f'initial predicate {initial.state}: {initial.reasons}')
        budget = TrialBudget(test, args.max_trials)
        reduced = Reducer().reduce(deck, budget)
        evidence = test.evidence(reduced)
        final = predicate.evaluate(evidence)
        reduced.save(args.out, 'reduced.sp')
        write_json(args.out / 'minimize.json', {'predicate': predicate.to_json(), 'search_calls': budget.used,
                                               'simulated_trials': test.trials, 'final': final.state,
                                               'reasons': final.reasons, 'search': test.results})
        if final.state != 'matched':
            raise SystemExit('final predicate did not reproduce; evidence retained, no verified bundle issued')
        if any(bench.configurations[n].tight for n in plan.names):
            plan = replace(plan, tight_level=evidence.tight_level)
        for name, content in bundle_files(reduced, bench, plan, predicate=predicate.to_json()).items():
            (args.out / name).parent.mkdir(parents=True, exist_ok=True)
            (args.out / name).write_text(content)
        print(args.out / 'repro.sh')


COMMANDS: Sequence[Command] = (Observe(), Evaluate(), Transform(), Explore(), Freeze(), Benchmark(), ReplayCase(),
                               Minimize())
