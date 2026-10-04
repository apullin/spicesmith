import fcntl

from conftest import fake_deck

from spicesmith.deck import Deck
from spicesmith.simulation import BUSY, RunResult, Simulation, SimulatorLog


def test_a_clean_run(fake, tmp_path):
    run = Simulation(fake).run(fake_deck(), tmp_path / 'r', 'ref')
    assert run.ok and run.status == 0 and run.outcome == 0 and run.configuration == 'ref'
    assert run.outputs['waveform.txt'].startswith(b' time v(a) v(b)')
    assert run.diagnostics == ()
    assert run.stats['equations'] == 1 and run.stats['tran_iterations'] == 100
    assert (tmp_path / 'r/log.txt').exists() and str(tmp_path / 'r') in (tmp_path / 'r/deck.sp').read_text()


def test_an_aborted_transient_is_not_ok_although_ngspice_exits_0(fake, tmp_path):
    run = Simulation(fake).run(fake_deck('* fake: abort'), tmp_path / 'r')
    assert run.status == 0 and run.aborted and not run.ok and run.outcome == 'aborted'
    assert run.outputs == {'waveform.txt': None}
    assert run.diagnostics == ('doAnalyses: TRAN:  Timestep too small; time = 1e-09, timestep = 1e-23: '
                               'trouble with node "x"', 'tran simulation(s) aborted')


def test_diagnostics_keep_warnings_and_drop_environment_noise():
    log = SimulatorLog("Warning: can't find the initialization file spinit.\nNote: Starting dynamic gmin stepping\n"
                       "Total elapsed time (seconds) = 0.05\nwarning, can't find model 'p12' from line\n"
                       '{"n":45,"phases":{"factor":{"failures":0}}}\nCircuit Equations = 13\n')
    assert log.diagnostics == ('Note: Starting dynamic gmin stepping', "warning, can't find model 'p12' from line")
    assert log.stats == {'equations': 13.0} and not log.aborted


def test_exit_status_timeout_and_environment(fake, tmp_path):
    run = Simulation(fake, {'FAKE_WARN': 'x'}).run(fake_deck('* fake: exit 3'), tmp_path / 'a')
    assert run.status == 3 and not run.ok and run.diagnostics == ('Warning: x',)
    assert run.environment['FAKE_WARN'] == 'x'
    assert run.environment['LC_ALL'] == 'C' and 'PYTHONPATH' not in run.environment
    slow = Simulation(fake, timeout=0.5).run(fake_deck('* fake: sleep 5'), tmp_path / 'b')
    assert slow.status == 'timeout' and slow.seconds < 4


def test_stale_outputs_are_removed_before_a_run(fake, tmp_path):
    Simulation(fake).run(fake_deck(), tmp_path / 'r')
    assert Simulation(fake).run(fake_deck('* fake: abort'), tmp_path / 'r').outputs['waveform.txt'] is None


def test_auxiliary_files_go_next_to_the_deck(fake, tmp_path):
    deck = Deck(fake_deck().text, files={'lib.inc': '* lib\n'})
    Simulation(fake).run(deck, tmp_path / 'r')
    assert (tmp_path / 'r/lib.inc').read_text() == '* lib\n'


def test_a_lock_held_elsewhere_makes_a_run_busy_without_running_it(fake, tmp_path):
    lock = tmp_path / 'device.lock'
    with lock.open('w') as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        busy = Simulation(fake, locks=(str(lock),)).run(fake_deck(), tmp_path / 'a')
    assert busy.status == BUSY and busy.outputs == {'waveform.txt': None} and not busy.ok
    assert Simulation(fake, locks=(str(lock),)).run(fake_deck(), tmp_path / 'b').ok


def test_a_stored_run_loads_back(fake, tmp_path):
    run = Simulation(fake, {'FAKE_WARN': 'x'}).run(fake_deck('* fake: exit 2'), tmp_path / 'r', 'flags')
    loaded = RunResult.load(tmp_path / 'r')
    assert loaded.configuration == 'flags' and loaded.status == 2 and loaded.outputs == run.outputs
    assert loaded.diagnostics == run.diagnostics and loaded.stats == run.stats
    assert loaded.environment == run.environment and loaded.command == run.command
