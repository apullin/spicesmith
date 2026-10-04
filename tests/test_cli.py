"""The command line, driven through main() with the fake simulator."""
import fcntl
import json

from spicesmith import generator
from spicesmith.cli import main
from spicesmith.deck import Deck


def test_gen_writes_a_runnable_bundle(tmp_path):
    seed = next(s for s in range(1, 100) if generator.generate(s).files)
    main(['gen', str(seed), str(tmp_path / 'case.sp'), '--profile', 'legacy'])
    deck = Deck.load(tmp_path, 'case.sp')
    assert deck.text == generator.generate(seed).text
    assert deck.files == generator.generate(seed).files and (tmp_path / 'fuzzlib.lib').exists()


def test_every_simulating_command_checks_the_supplied_reference(tmp_path, capsys):
    import pytest
    impostor = tmp_path / 'ngspice'
    impostor.write_text('not the reference')
    (tmp_path / 'case').mkdir()
    for argv in (['run', '--out', str(tmp_path / 'b')], ['regress', '--corpus', str(tmp_path)],
                 ['reduce', str(tmp_path / 'case'), 'flags:'], ['split', str(tmp_path / 'case'), 'approx:']):
        with pytest.raises(SystemExit) as error:
            main([*argv, '--reference', str(impostor), '--cpus', '0-3'])
        assert error.value.code == 2
        assert 'simulator executable not found' in capsys.readouterr().err


def fake_simulators(fake):
    return ['--reference', str(fake), '--candidate', str(fake), '--cpus', '0-3']


def test_run_report_and_judge_a_small_batch(fake, tmp_path, capsys):
    out = tmp_path / 'b'
    main(['run', '--out', str(out), '--cases', '2', '--jobs', '1', '--keep', *fake_simulators(fake)])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith('seed      1 ok') and '2 case(s)' in lines[-1]
    main(['report', str(out), '--json', str(tmp_path / 'report.json')])
    report = capsys.readouterr().out
    assert '2 cases: 0 failing' in report and '| exact:outputs | 2 | 0 | 0 |' in report
    main(['judge', str(out / 'case-000001')])
    assert capsys.readouterr().out.startswith(f'{out / "case-000001"}: ')
    main(['run', '--out', str(out), '--cases', '3', '--jobs', '1', *fake_simulators(fake)])  # resumes
    assert 'seed      3' in capsys.readouterr().out


def test_a_batch_runs_extra_configurations_and_notes_a_busy_lock(fake, tmp_path, capsys):
    lock, extra = tmp_path / 'device.lock', tmp_path / 'extra.json'
    extra.write_text(json.dumps([{'name': 'mine', 'binary': str(fake), 'claim': 'approximate', 'extends': 'approx',
                                  'locks': [str(lock)]}]))
    argv = ['run', '--cases', '1', '--jobs', '1', '--extra-configurations', str(extra), *fake_simulators(fake)]
    main([*argv, '--out', str(tmp_path / 'free')])
    summary = json.loads((tmp_path / 'free/case-000001/summary.json').read_text())
    assert summary['status']['mine'] == 0 and summary['checks']['mine:outputs'] == 'pass'
    with lock.open('w') as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        main([*argv, '--out', str(tmp_path / 'busy')])
    assert 'ok (note: mine: lock busy, not run)' in capsys.readouterr().out


def test_regress_runs_a_corpus(fake, tmp_path, capsys):
    from spicesmith.corpus import Entry, Expectation
    from conftest import fake_deck
    Entry('tiny', fake_deck(), 'flags:', Expectation.PASS, 'flags: x', 'test').save(tmp_path / 'corpus')
    main(['regress', '--corpus', str(tmp_path / 'corpus'), '--work', str(tmp_path / 'w'), *fake_simulators(fake)])
    assert capsys.readouterr().out.splitlines() == ['pass       tiny', 'promotable']
