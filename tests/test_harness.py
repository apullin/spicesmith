import dataclasses
import json
import stat

import pytest
from conftest import fake_deck

from spicesmith import generator
from spicesmith.config import CONFIGURATIONS, Binary, Claim, Configuration, Simulators
from spicesmith.harness import Batch, CaseSummary, Settings, Tally, Testbench
from spicesmith.provenance import BinaryRecord, Provenance


def wrapper(tmp_path, fake, name, **env):
    """A candidate that runs the fake simulator with extra environment variables."""
    script = tmp_path / name
    exports = ' '.join(f'{k}={v!r}' for k, v in env.items())
    script.write_text(f'#!/bin/sh\nexec env {exports} {fake} "$@"\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def provenance(simulators):
    record = BinaryRecord(str(simulators.candidate), 'x' * 64)
    return Provenance(record, record, (), {}, {}, {}, 1)


@pytest.fixture
def settings(fake):
    return Settings(Simulators(fake, fake), repeat_every=2)


def test_testbench_gathers_runs_front_end_dumps_and_repeats(settings, tmp_path):
    evidence = Testbench(settings).run(fake_deck(), tmp_path, repeat=True)
    assert set(evidence.runs) == {'ref', 'exact', 'approx', 'tight'}
    assert all(run.ok for run in evidence.runs.values()) and evidence.tight_level == 0
    assert (tmp_path / 'front-ref/debug-out3.txt').exists()
    assert evidence.repeats[0].second.outputs == evidence.runs['ref'].outputs
    assert evidence.runs['exact'].environment['OMP_NUM_THREADS'] == '1'


def test_the_accuracy_reference_falls_back_to_looser_tolerances(settings, tmp_path):
    deck = fake_deck('* fake: abort-if abstol=1e-15')
    evidence = Testbench(settings).run(deck, tmp_path, names=['ref', 'tight'])
    assert evidence.tight_level == 1 and evidence.runs['tight'].ok
    assert (tmp_path / 'tight/log.txt').exists() and (tmp_path / 'tight-1/log.txt').exists()


def test_a_batch_of_passing_cases(settings, tmp_path):
    batch = Batch(tmp_path / 'b', settings, provenance(settings.simulators))
    lines = []
    tally = batch.run(batch.seeds(1, 3), jobs=2, cpus='0-7', echo=lines.append)
    assert tally == Tally(cases=3, failed_checks={})
    assert sorted(batch.finished()) == [1, 2, 3] and len(lines) == 4
    summary = CaseSummary.from_json(json.loads((tmp_path / 'b/case-000002/summary.json').read_text()))
    assert summary.findings == () and summary.checks['ref:determinism'] == 'pass'
    assert summary.deck['generator'] == generator.VERSION and summary.deck['blocks'][0]['kind'] == 'supply'
    assert summary.provenance['runs']['exact']['env']['OMP_NUM_THREADS'] == '1'
    assert not (tmp_path / 'b/case-000002/ref/waveform.txt').exists()  # passing cases drop outputs
    assert list(batch.seeds(1, 5)) == [4, 5] and list(batch.seeds(1, 2, resume=True)) == [4, 5]


def test_a_failing_case_is_kept_whole(fake, tmp_path):
    noisy = wrapper(tmp_path, fake, 'noisy', FAKE_WARN="can't find model 'p3'")
    settings = Settings(Simulators(fake, noisy), repeat_every=0)
    batch = Batch(tmp_path / 'b', settings, provenance(settings.simulators))
    summary = batch.run_case(7)
    assert "exact: diagnostics differ from ref (new: 'Warning: can't find model 'p3'')" in summary.findings
    assert summary.checks['exact:diagnostics'] == 'fail' and summary.checks['exact:outputs'] == 'pass'
    assert (tmp_path / 'b/case-000007/exact/waveform.txt').exists()


def test_a_harness_error_does_not_stop_the_batch(settings, tmp_path):
    def broken(seed):
        raise RuntimeError(f'no deck for {seed}')
    batch = Batch(tmp_path / 'b', settings, provenance(settings.simulators), generate=broken)
    lines = []
    tally = batch.run(batch.seeds(1, 2), jobs=1, cpus='0-3', echo=lines.append)
    assert tally.harness_errors == 2 and tally.cases == 0
    assert 'HARNESS ERROR RuntimeError: no deck for 1' in lines[0]


def test_a_batch_refuses_every_material_change_unless_forced(settings, tmp_path):
    base = provenance(settings.simulators)
    Batch(tmp_path / 'b', settings, base).record({'start': 1})
    other_binary = BinaryRecord('/x/ngspice', 'y' * 64)
    changes = {
        'reference': (dataclasses.replace(base, reference=other_binary), settings),
        'candidate': (dataclasses.replace(base, candidate=other_binary), settings),
        'inputs': (dataclasses.replace(base, inputs={'models.lib': 'z' * 64}), settings),
        'generator': (dataclasses.replace(base, generator=2), settings),
        'scale': (dataclasses.replace(base, scale=10), settings),
        'code': (dataclasses.replace(base, tool={'code_sha256': 'c' * 64}), settings),
        'extra_binaries': (dataclasses.replace(base, extra_binaries=(other_binary,)), settings),
        'configurations': (base, dataclasses.replace(settings, configurations={
            **CONFIGURATIONS, 'mine': Configuration('mine', Binary.CANDIDATE, Claim.EXACT, {'MY_FLAG': '1'})})),
        'front_end': (base, dataclasses.replace(settings, front_end=False)),
        'repeat_every': (base, dataclasses.replace(settings, repeat_every=0)),
        'timeout': (base, dataclasses.replace(settings, timeout=0.001)),
        'lock_wait': (base, dataclasses.replace(settings, lock_wait=60)),
    }
    for field_name, (changed_provenance, changed_settings) in changes.items():
        with pytest.raises(SystemExit, match=rf'{field_name}:'):
            Batch(tmp_path / 'b', changed_settings, changed_provenance).record({'start': 1})
    Batch(tmp_path / 'b', settings, changes['generator'][0]).record({'start': 2}, force=True)
    record = json.loads((tmp_path / 'b/batch.json').read_text())
    assert record['identity']['generator'] == 1 and len(record['sessions']) == 2  # the first identity stays
    assert record['sessions'][0]['identity'] != record['sessions'][1]['identity']


def test_a_legacy_batch_record_needs_force(settings, tmp_path):
    (tmp_path / 'b').mkdir()
    (tmp_path / 'b/batch.json').write_text(json.dumps({'provenance': {}, 'args': {}}))
    with pytest.raises(SystemExit, match='predates'):
        Batch(tmp_path / 'b', settings, provenance(settings.simulators)).record({})
    Batch(tmp_path / 'b', settings, provenance(settings.simulators)).record({}, force=True)


def test_summaries_name_their_identity_and_the_report_flags_a_mixture(settings, tmp_path):
    from spicesmith.report import Report
    first = Batch(tmp_path / 'b', settings, provenance(settings.simulators))
    first.run(first.seeds(1, 1), jobs=1, cpus='0-3', echo=lambda line: None)
    second = Batch(tmp_path / 'b', dataclasses.replace(settings, repeat_every=0), provenance(settings.simulators))
    second.record({}, force=True)
    second.run(second.seeds(2, 1), jobs=1, cpus='0-3', echo=lambda line: None)
    summaries = list(first.summaries())
    assert [s.provenance['identity'] for s in summaries] == [first.identity.digest(), second.identity.digest()]
    assert '**Mixed identities**' in Report(tuple(summaries)).markdown()


def failing(tmp_path, fake, name, when='', status=3):
    """A binary that exits `status` without writing anything (when its arguments contain
    `when`), and otherwise runs the fake simulator."""
    script = tmp_path / name
    script.write_text(f'#!/bin/sh\ncase "$*" in *{when}*) exit {status};; esac\nexec {fake} "$@"\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def test_a_rerun_front_end_cannot_reuse_an_earlier_dump(fake, tmp_path):
    from spicesmith import oracles
    first = Testbench(Settings(Simulators(fake, fake))).run(fake_deck(), tmp_path / 'w', ['ref', 'exact'])
    assert oracles.judge(first).checks['front-end:identity'].value == 'pass'
    broken = failing(tmp_path, fake, 'broken', when='ngdebug')
    second = Testbench(Settings(Simulators(fake, broken))).run(fake_deck(), tmp_path / 'w', ['ref', 'exact'])
    assert not (tmp_path / 'w/front-exact/debug-out3.txt').exists()
    verdict = oracles.judge(second)
    assert 'front-end: candidate parse-only run 3' in verdict.findings
    assert 'front-end: candidate wrote no debug-out3.txt' in verdict.findings


def test_a_failed_reference_front_end_is_inconclusive(fake, tmp_path):
    from spicesmith import oracles
    broken = failing(tmp_path, fake, 'broken', when='ngdebug', status=2)
    evidence = Testbench(Settings(Simulators(broken, fake))).run(fake_deck(), tmp_path / 'w', ['ref', 'exact'])
    verdict = oracles.judge(evidence)
    assert verdict.inconclusive == ('front-end: ref parse-only run 2',)
    assert verdict.checks['front-end:identity'].value == 'skip'


def test_a_rerun_accuracy_reference_leaves_no_stale_fallback_level(settings, tmp_path):
    Testbench(settings).run(fake_deck('* fake: abort-if abstol=1e-15'), tmp_path, names=['ref', 'tight'])
    assert (tmp_path / 'tight-1/log.txt').exists()
    evidence = Testbench(settings).run(fake_deck(), tmp_path, names=['ref', 'tight'])
    assert evidence.tight_level == 0 and not (tmp_path / 'tight-1').exists()


def test_an_inconclusive_case_keeps_its_outputs(settings, tmp_path):
    deck = fake_deck('* fake: abort-if trtol=1')  # tight aborts at every tolerance level
    batch = Batch(tmp_path / 'b', settings, provenance(settings.simulators), generate=lambda seed: deck)
    summary = batch.run_case(1)
    assert summary.inconclusive == ('tight aborted',) and not summary.findings
    assert (tmp_path / 'b/case-000001/ref/waveform.txt').exists()  # kept for diagnosis
