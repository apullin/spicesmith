from pathlib import Path

import pytest

from spicesmith.cli import main
from spicesmith.deck import Deck
from spicesmith.ngspice import prepare
from spicesmith.provenance import frozen_inputs, sha256
from spicesmith.simulation import Simulation


def test_default_generation_does_not_start_a_process(tmp_path, monkeypatch):
    import subprocess
    def forbidden(*args, **kwargs):
        raise AssertionError('generation must not start a process')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    main(['gen', '31', str(tmp_path / 'deck.sp')])
    deck = Deck.load(tmp_path)
    assert '.control' not in deck.text and '__OUT__' not in deck.text
    assert '.tran ' in deck.text and '.model n nmos' in deck.text
    assert frozen_inputs(deck) == {}
    assert (tmp_path / 'deck.circuit.json').exists()


def test_optional_adapter_retains_the_generated_observation_contract(fake, tmp_path):
    main(['gen', '31', str(tmp_path / 'deck.sp')])
    deck = Deck.load(tmp_path)
    adapted = prepare(deck)
    assert adapted.outputs == deck.outputs
    assert adapted.output_requests == deck.output_requests
    assert adapted.text.count('tran ') == 1
    assert prepare(adapted) == adapted
    result = Simulation(fake, threads=3).run(deck, tmp_path / 'run')
    assert result.ok and set(result.outputs) == set(deck.outputs)
    assert '__OUT__' not in (tmp_path / 'run/deck.sp').read_text()
    assert 'set num_threads=3' in (tmp_path / 'run/deck.sp').read_text()
    assert result.environment['OMP_NUM_THREADS'] == '3'


def test_plain_saved_netlist_can_be_observed(fake, tmp_path):
    deck = Deck('RC\nV1 a 0 1\nR1 a b 1k\nC1 b 0 1p\n.tran 1p 1n\n.save v(b)\n.end\n')
    assert Simulation(fake).run(deck, tmp_path).outputs['tran.txt'] is not None
    with pytest.raises(ValueError, match='observation requires'):
        prepare(Deck('no analysis\nR1 a 0 1k\n.end\n'))


def test_identical_analyses_keep_distinct_output_contracts():
    deck = Deck('two outputs\n.tran 1p 1n\n* spicesmith-output a.txt v(a)\n'
                '.tran 1p 1n\n* spicesmith-output b.txt v(b)\n.save v(a) v(b)\n.end\n')
    assert deck.outputs == ['a.txt', 'b.txt']
    assert prepare(deck).output_requests == deck.output_requests


def test_external_dependency_hashes_are_deck_specific_and_follow_nested_includes(tmp_path):
    child = tmp_path / 'child.inc'
    child.write_text('.model n nmos level=1\n')
    parent = tmp_path / 'parent.lib'
    parent.write_text('.lib test\n.include "child.inc"\n.endl test\n')
    deck = Deck(f'models\n.lib "{parent}" test\n.end\n')
    assert frozen_inputs(deck) == {str(child): sha256(child), str(parent): sha256(parent)}
    assert frozen_inputs(Deck('independent\n.end\n')) == {}
    child.unlink()
    with pytest.raises(FileNotFoundError):
        frozen_inputs(deck)


def test_command_is_found_on_caller_path(fake, tmp_path, monkeypatch):
    monkeypatch.setenv('PATH', str(fake.parent))
    assert Simulation(Path(fake.name)).command()[0] == str(fake)
    monkeypatch.chdir(fake.parent)
    monkeypatch.setenv('PATH', '/nonexistent')
    assert Simulation(Path('./' + fake.name)).command()[0] == str(fake)


@pytest.mark.simulator
def test_portable_corpus_with_real_ngspice(tmp_path, real_settings):
    from spicesmith.corpus import Corpus, Status
    from spicesmith.harness import Testbench
    results = list(Corpus().run(Testbench(real_settings), tmp_path))
    assert results and all(r.status is Status.PASS for r in results)
