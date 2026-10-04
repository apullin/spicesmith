import dataclasses
import json
import subprocess
from pathlib import Path

import pytest
from conftest import fake_deck

from spicesmith import benchmark, experiment, profiles
from spicesmith.cli import main
from spicesmith.config import Simulators
from spicesmith.deck import Deck
from spicesmith.harness import Settings, Testbench
from spicesmith.oracles import Evidence
from spicesmith.plans import ExecutionPlan
from spicesmith.policies import Policy
from spicesmith.population import Campaign, Stratum, freeze, load_frozen, numerical
from spicesmith.predicates import Predicate
from spicesmith.simulation import RunResult


def strata():
    return (Stratum('impossible', profiles.Profile(max_devices=5, max_attempts=2)),
            Stratum('ordinary', profiles.profile('heterogeneous-continuous')))


def test_deficit_scheduler_replays_and_counts_every_rejection(tmp_path):
    campaign = Campaign(tmp_path / 'resumed', strata(), seed=17, budget=4, feedback=True)
    first = campaign.run(limit=2)
    assert first['issued_cases'] == 2
    final = campaign.run(resume=True)
    assert final['issued_cases'] == 4
    assert final['states'] == {'rejected': 3, 'generated': 1}
    assert final['strata']['impossible']['generation_rejections'] == 6
    assert final['strata']['impossible']['remaining'] == 1
    full = dataclasses.replace(campaign, directory=tmp_path / 'full')
    assert full.run() == final
    left = json.loads((campaign.directory / 'campaign.json').read_text())
    right = json.loads((full.directory / 'campaign.json').read_text())
    assert left == right
    assert left['cases'][2]['decision']['reason'] == 'observed-deficit'
    with pytest.raises(ValueError, match='incompatible'):
        dataclasses.replace(campaign, seed=18).run(resume=True)
    with pytest.raises(ValueError, match='already exists'):
        campaign.run()


def test_interrupted_cases_remain_unresolved_in_the_population(tmp_path):
    campaign = Campaign(tmp_path, (strata()[1],), budget=2, feedback=True)
    campaign.run(limit=0)
    state = json.loads((tmp_path / 'campaign.json').read_text())
    state['cases'].append({'index': 0, 'seed': 3, 'path': 'case-000000', 'stratum': 'ordinary', 'state': 'pending'})
    (tmp_path / 'campaign.json').write_text(json.dumps(state))
    report = campaign.run(resume=True)
    assert report['issued_cases'] == 2
    assert report['states'] == {'interrupted': 1, 'generated': 1}


def test_observed_regimes_do_not_inherit_hint_intentions(fake, tmp_path):
    bench = Testbench(Settings(Simulators(fake, fake)), '0-3')
    s = Stratum('quiet-target', profiles.profile('coupled-feedback', {'motifs': 2}), quota=1, target='quiescent')
    report = Campaign(tmp_path, (s,), budget=3, feedback=True, bench=bench).run()
    reached = report['strata'][s.name]
    assert reached['issued'] == 3 and reached['remaining'] == 1
    assert reached['observed_regimes'] == {'active': 3}
    assert reached['assessment_states'] == {'unassessed': 3}
    assert reached['metric_availability']['equations'] == 6
    assert reached['metric_availability']['fillin'] == 0
    assert numerical({})['regime'] == 'incomplete'


def test_freeze_preserves_failed_simulations_and_discloses_rejected_cases(fake, tmp_path):
    bench = Testbench(Settings(Simulators(fake, Path('/usr/bin/false'))), '0-3')
    campaign = Campaign(tmp_path / 'source', strata(), budget=2, bench=bench)
    report = campaign.run()
    assert report['strata']['ordinary']['completed'] == 0
    result = freeze(campaign.directory, tmp_path / 'frozen')
    assert len(result['cases']) == 1 and len(result['excluded']) == 1
    assert result['cases'][0]['source_state'] == 'observed'
    assert load_frozen(tmp_path / 'frozen') == result
    deck = tmp_path / 'frozen' / result['cases'][0]['path'] / 'deck.sp'
    deck.write_text(deck.read_text() + '* changed\n')
    with pytest.raises(ValueError, match='workload changed'):
        load_frozen(tmp_path / 'frozen')


def test_incomplete_or_short_runs_never_count_as_speedups():
    def sample(seconds, complete=True):
        return {'seconds': seconds, 'complete': complete, 'outcome': 0 if complete else 'aborted'}
    reference = {'samples': [sample(10), sample(12), sample(11)]}
    for samples in ([sample(0.001, False)] * 3, [sample(0.001)] * 2, [sample(0)] * 3):
        result = benchmark.comparison({'ref': reference, 'candidate': {'samples': samples}}, 'ref')
        assert result['comparisons']['candidate']['speedup'] is None
    observed = {'ref': reference, 'candidate': {'samples': [sample(20), sample(22), sample(24)]}}
    result = benchmark.comparison(observed, 'ref')
    assert result['comparisons']['candidate']['speedup'] == 0.5
    assert result['timings']['candidate']['stdev'] == 2
    grouped = benchmark.summarize([{'assessment': 'unassessed', 'groups': {'composition': 'mixed'},
                                   'performance': result}])
    assert grouped['composition=mixed']['configurations']['candidate']['slowdowns'] == 1


def test_freezing_cannot_attribute_an_edited_deck_to_the_old_generated_structure(tmp_path):
    c = Campaign(tmp_path / 'source', (strata()[1],))
    c.run()
    path = c.directory / 'case-000000/deck.sp'
    path.write_text(path.read_text().replace('.end\n', 'Rextra a 0 1k\n.end\n'))
    with pytest.raises(ValueError, match='changed before freezing'):
        freeze(c.directory, tmp_path / 'frozen')
    assert benchmark.groups({}, {})['composition'] == 'unclassified'


def test_frozen_benchmark_retains_all_repeats_and_separate_policy(fake, tmp_path):
    campaign = Campaign(tmp_path / 'source', (Stratum('small', profiles.profile('nonlinear-small', {'motifs': 1})),))
    campaign.run()
    freeze(campaign.directory, tmp_path / 'frozen')
    bench = Testbench(Settings(Simulators(fake, fake)), '0-3')
    report = benchmark.run(tmp_path / 'frozen', tmp_path / 'bench', bench, ExecutionPlan(repetitions=3))
    assert report['all_cases_attempted'] and report['completed_workloads'] == 1
    assert report['cases'][0]['assessment'] == 'unassessed'
    assert len(report['cases'][0]['performance']['timings']['exact']['samples']) == 3
    assert report['cases'][0]['performance']['comparisons']['exact']['speedup'] > 0
    broken = Testbench(Settings(Simulators(fake, Path('/usr/bin/false'))), '0-3')
    bad = benchmark.run(tmp_path / 'frozen', tmp_path / 'broken', broken, ExecutionPlan(repetitions=3))
    assert bad['cases'][0]['performance']['comparisons']['exact']['speedup'] is None
    assert bad['all_cases_attempted'] and bad['completed_workloads'] == 0


def test_sealed_replay_and_evaluation_detect_changed_evidence(fake, tmp_path):
    bench = Testbench(Settings(Simulators(fake, fake)), '0-3')
    experiment.run(fake_deck(), bench, ExecutionPlan(), tmp_path / 'source')
    result = experiment.replay(tmp_path / 'source', tmp_path / 'replay', Policy('exact'))
    assert result.state == 'pass'
    artifact = tmp_path / 'source/exact/waveform.txt'
    artifact.write_bytes(artifact.read_bytes() + b'1 1 1\n')
    with pytest.raises(ValueError, match='evidence changed'):
        experiment.evaluate(tmp_path / 'source', Policy('exact'))


def test_predicates_preserve_outcomes_differences_structure_and_repeated_timings():
    deck = fake_deck()
    good = RunResult('ref', 0, outputs={'waveform.txt': b'time v(a) v(b)\n0 0 0\n1e-9 0 0\n'}, seconds=1)
    bad = dataclasses.replace(good, status=-11)
    assert Predicate('outcome', outcome=-11).evaluate(Evidence(deck, {'exact': bad})).state == 'matched'
    changed = dataclasses.replace(good, outputs={'waveform.txt': b'time v(a) v(b)\n0 0.2 0\n1e-9 0.2 0\n'})
    e = Evidence(deck, {'ref': good, 'exact': changed})
    assert Predicate('difference', threshold=0.1).evaluate(e).state == 'matched'
    assert Predicate('difference', threshold=0.3).evaluate(e).state == 'not-matched'
    assert Predicate('exact').evaluate(e).state == 'matched'
    assert Predicate('structure', minimum_devices=2).evaluate(e).state == 'not-matched'
    assert Predicate('structure', connections=(('a', '0'),)).evaluate(e).state == 'matched'
    assert Predicate('structure', connections=(('a', 'b'),)).evaluate(e).state == 'not-matched'
    slow = dataclasses.replace(good, seconds=2)
    for repeats, expected in ((2, 'unresolved'), (3, 'matched')):
        e = Evidence(deck, {'ref': good, 'exact': slow}, samples={'ref': (good,) * repeats, 'exact': (slow,) * repeats})
        assert Predicate('slowdown', threshold=1.5).evaluate(e).state == expected


def test_minimize_bundles_and_executes_its_actual_predicate(fake, tmp_path):
    candidate = tmp_path / 'candidate'
    candidate.write_text(f'#!/bin/sh\nexport FAKE_SHIFT=0.2\nexec {fake} "$@"\n')
    candidate.chmod(0o755)
    fake_deck('R2 b 0 2k', 'C1 a b 1p').save(tmp_path / 'input')
    predicate = tmp_path / 'predicate.json'
    predicate.write_text(json.dumps(Predicate('difference', threshold=0.1, minimum_devices=1).to_json()))
    out = tmp_path / 'reduced'
    main(['minimize', str(tmp_path / 'input'), str(predicate), '--out', str(out), '--max-trials', '3',
          '--reference', str(fake), '--candidate', str(candidate), '--cpus', '0-3'])
    record = json.loads((out / 'minimize.json').read_text())
    assert record['search_calls'] <= 3 and record['final'] == 'matched'
    shadow = tmp_path / 'shadow'
    (shadow / 'spicesmith').mkdir(parents=True)
    (shadow / 'spicesmith/__init__.py').write_text('raise RuntimeError("wrong package")\n')
    result = subprocess.run(['sh', str(out / 'repro.sh')], cwd=shadow, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((out / 'replay-result.json').read_text())['status'] == 'reproduced'
    assert Deck.load(out, 'reduced.sp').outputs
