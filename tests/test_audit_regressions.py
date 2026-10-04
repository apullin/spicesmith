import fcntl
import json
import subprocess
from dataclasses import replace

import pytest
from conftest import fake_deck

from spicesmith import oracles
from spicesmith.cli import main
from spicesmith.config import CONFIGURATIONS, Binary, Claim, Configuration, Simulators
from spicesmith.corpus import Corpus, Entry, EntryResult, Expectation, Status
from spicesmith.deck import Deck
from spicesmith.harness import Settings, Testbench, load_evidence
from spicesmith.integrity import read_output
from spicesmith.oracles import AccuracyOracle, Evidence, Verdict
from spicesmith.plans import ExecutionPlan
from spicesmith.reduce import FindingTest, Reducer, Reproducer, TrialBudget
from spicesmith.simulation import RunResult
from spicesmith.waveform import Waveform

DECK = Deck('audit\n.param vsupply=1.2\n.control\ntran 1p 1n\n'
            'wrdata __OUT__/waveform.txt v(d)\nquit\n.endc\n.end\n')
GOOD = b'time v(d)\n0 0\n5e-10 0\n1e-9 0\n'


@pytest.mark.parametrize('bad', [b'time v(d)\n0 0\n5e-10 0\n',
                                     b'time v(d)\n5e-10 0\n1e-9 0\n',
                                     b'time v(d)\n0\n0 5e-10 0 1e-9 0\n',
                                     b'time v(d) v(d)\n0 0 999\n5e-10 0 999\n1e-9 0 999\n'])
def test_bad_tables_cannot_pass_or_serve_as_a_reference(bad):
    runs = {name: RunResult(name, 0, outputs={'waveform.txt': GOOD}) for name in ('ref', 'tight', 'approx')}
    bad_run = replace(runs['approx'], outputs={'waveform.txt': bad})
    assert AccuracyOracle().judge(Evidence(DECK, {**runs, 'approx': bad_run})).findings
    for name in ('ref', 'tight'):
        judged = AccuracyOracle().judge(Evidence(DECK, {**runs, name: bad_run}))
        assert judged.inconclusive and not judged.findings


def test_ac_vector_representation_and_duplicate_names_are_checked():
    complex_table = Waveform.parse(b'frequency v(a) v(a)\n1 1 0\n2 1 0\n')
    real = Waveform.parse(b'frequency v(a)\n1 1\n2 1\n')
    assert 'representation' in complex_table.mismatch(real)
    assert not Waveform.read(b'frequency v(a) v(b) v(a)\n1 1 1 1\n').usable


def test_dc_grid_and_each_transient_request_have_their_own_contract():
    deck = Deck('d\n.control\ndc v1 1 0 -0.5\nwrdata __OUT__/dc.txt v(a)\n'
                'tran 1p 1n\nwrdata __OUT__/first.txt v(a)\n'
                'tran 1p 2n\nwrdata __OUT__/second.txt v(a)\n.endc\n.end\n')
    assert read_output(deck, 'dc.txt', b'v-sweep v(a)\n1 0\n0.5 0\n0 0\n').usable
    assert not read_output(deck, 'dc.txt', b'v-sweep v(a)\n1 0\n0.5 0\n').usable
    one_ns = b'time v(a)\n0 0\n1e-9 0\n'
    assert read_output(deck, 'first.txt', one_ns).usable
    assert not read_output(deck, 'second.txt', one_ns).usable


def test_fixed_ac_grid_must_not_silently_drop_interior_points():
    deck = Deck('ac\n.control\nac lin 3 1 3\nwrdata __OUT__/ac.txt v(a)\n.endc\n.end\n')
    assert read_output(deck, 'ac.txt', b'frequency v(a) v(a)\n1 1 0\n2 1 0\n3 1 0\n').usable
    assert not read_output(deck, 'ac.txt', b'frequency v(a) v(a)\n1 1 0\n3 1 0\n').usable


def test_an_entry_without_its_configuration_is_skipped_and_a_busy_lock_is_inconclusive(fake, tmp_path):
    corpus = Corpus(tmp_path / 'corpus')
    entry = Entry('mine', fake_deck(), 'mine:', Expectation.KNOWN, 'mine: bad', 'audit', requires=('mine',))
    entry.save(corpus.directory)
    plain = Testbench(Settings(Simulators(fake, fake)), '0-3')
    assert [result.status for result in corpus.run(plain, tmp_path / 'w')] == [Status.SKIPPED]
    lock = tmp_path / 'device.lock'
    mine = Configuration('mine', Binary.CANDIDATE, Claim.APPROXIMATE, locks=(str(lock),))
    with_mine = Testbench(Settings(Simulators(fake, fake), {**CONFIGURATIONS, 'mine': mine}), '0-3')
    with lock.open('w') as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        [result] = corpus.run(with_mine, tmp_path / 'w')
    assert result.status is Status.INCONCLUSIVE and f'mine: {oracles.BUSY_NOTE}' in result.unresolved


def test_inconclusive_and_unassessed_corpus_cases_are_not_fixed():
    entry = Entry('case', DECK, 'approx:', Expectation.KNOWN, 'approx: bad', 'audit')
    for verdict in (Verdict({}, (), (), (), {}), Verdict({}, (), (), ('tight aborted',), {})):
        assert EntryResult(entry, verdict).status is Status.INCONCLUSIVE


def test_completely_broken_simulators_cannot_be_promoted(tmp_path, capsys):
    Entry('case', DECK, 'approx:', Expectation.PASS, 'approx: bad', 'audit').save(tmp_path / 'corpus')
    with pytest.raises(SystemExit) as error:
        main(['regress', '--corpus', str(tmp_path / 'corpus'), '--work', str(tmp_path / 'work'),
              '--reference', '/usr/bin/false', '--candidate', '/usr/bin/false', '--cpus', '0-3'])
    assert error.value.code == 2
    assert 'not promotable' in capsys.readouterr().out


def test_all_rejected_trials_obey_the_budget_and_keep_the_original_deck():
    calls = []
    def reject(deck):
        calls.append(deck)
        return False
    deck = Deck('audit\n' + '\n'.join(f'R{i} n{i} 0 1k' for i in range(100)) + '\n.end\n')
    assert Reducer().reduce(deck, TrialBudget(reject, 1)) == deck
    assert len(calls) == 1
    assert Reducer().reduce(deck, TrialBudget(reject, 0)) == deck
    assert len(calls) == 1


def bundle(out, test, deck):
    verdict = oracles.judge(test.evidence(deck))
    finding = next(f for f in verdict.findings if f.startswith(test.check))
    for name, content in Reproducer(test, finding).files(deck).items():
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_text(content)
    done = subprocess.run(['sh', str(out / 'repro.sh')], capture_output=True, text=True, timeout=15)
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads((out / 'replay-result.json').read_text())['status'] == 'reproduced'
    return load_evidence(out / 'repro')


def test_front_end_and_timeout_reproducers_rerun_the_finding(fake, tmp_path):
    for kind, body, check in (
        ('front', 'case "$*" in *ngdebug*) exit 3;; esac', 'front-end:'),
        ('slow', 'sleep 1', 'exact: exit status timeout'),
    ):
        candidate = tmp_path / kind
        candidate.write_text(f'#!/bin/sh\n{body}\nexec {fake} "$@"\n')
        candidate.chmod(0o755)
        settings = Settings(Simulators(fake, candidate), timeout=0.5)
        test = FindingTest(Testbench(settings, '0-3'), check, tmp_path / f'work-{kind}')
        evidence = bundle(tmp_path / f'bundle-{kind}', test, fake_deck())
        assert any(f.startswith(check) for f in oracles.judge(evidence).findings)


def test_determinism_reproducer_includes_the_second_run(fake, tmp_path):
    reference = tmp_path / 'reference'
    reference.write_text(f'#!/bin/sh\ncase "$PWD" in *ref-repeat) export FAKE_WARN=repeat;; esac\n'
                         f'exec {fake} "$@"\n')
    reference.chmod(0o755)
    test = FindingTest(Testbench(Settings(Simulators(reference, fake)), '0-3'),
                       'ref: not deterministic', tmp_path / 'work')
    evidence = bundle(tmp_path / 'bundle', test, fake_deck())
    assert len(evidence.repeats) == 1


def test_explicit_plan_can_run_and_reload_without_a_reference(fake, tmp_path):
    plan = ExecutionPlan(('exact',), repetitions=2)
    bench = Testbench(Settings(Simulators(fake, fake)))
    first = bench.run(fake_deck(), tmp_path, plan=plan)
    second = load_evidence(tmp_path)
    assert first.runs.keys() == second.runs.keys() == {'exact'}
    assert len(second.samples['exact']) == 2 and second.front_end is None
