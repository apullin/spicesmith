import shutil
import subprocess
import json

from conftest import fake_deck

from spicesmith import oracles
from spicesmith.config import CONFIGURATIONS, Claim, Configuration, Simulators
from spicesmith.plans import definitions_from_json
from spicesmith.deck import Deck
from spicesmith.reduce import (DropOutputs, FindingTest, Layout, Reducer, RemoveLines, RemoveSubcircuitLines,
                               RemoveSubcircuits, Reproducer, ShortenTransient, SimplifyNumbers)
from spicesmith.harness import Settings, Testbench, load_evidence

DECK = Deck('\n'.join([
    'title',
    '.param p1=4.7k',
    '.subckt inv a b',
    'R1 a b 1k',
    'C1 b 0 1f',
    '.ends',
    '.subckt unused x',
    'R9 x 0 1',
    '.ends',
    'Xi n1 n2 inv',
    'R2 n2 0 p1',
    'C2 n2 0 2.2f',
    'V1 n1 0 PULSE(0 1 1n 10p 10p 1n 2n)',
    '.control',
    'tran 10p 20n',
    'wrdata __OUT__/waveform.txt v(n1) v(n2)',
    'quit',
    '.endc',
    '.end']))


def test_layout_finds_subcircuits_control_and_top_level_lines():
    layout = Layout.of(DECK)
    assert layout.subcircuits == ((2, 5), (6, 8)) and layout.control == (13, 17)
    assert layout.top_level == (1, 9, 10, 11, 12) and layout.subcircuit_body() == (3, 4, 7)


def test_ddmin_keeps_exactly_what_the_test_needs():
    needs = lambda deck: 'R2 n2 0 p1' in deck.text and '.param p1' in deck.text  # noqa: E731
    reduced = Reducer((RemoveLines(),)).reduce(DECK, needs)
    assert [reduced.lines[i] for i in Layout.of(reduced).top_level] == ['.param p1=4.7k', 'R2 n2 0 p1']


def test_unused_subcircuits_and_their_lines_go():
    needs = lambda deck: 'Xi n1 n2 inv' in deck.text and 'C1 b 0' in deck.text  # noqa: E731
    reduced = Reducer((RemoveSubcircuits(), RemoveSubcircuitLines())).reduce(DECK, needs)
    assert '.subckt unused' not in reduced.text and 'R1 a b 1k' not in reduced.text and 'C1 b 0 1f' in reduced.text


def test_outputs_drop_to_the_vectors_that_matter_but_never_to_none():
    reduced = Reducer((DropOutputs(),)).reduce(DECK, lambda deck: 'v(n2)' in deck.text)
    assert 'wrdata __OUT__/waveform.txt v(n2)' in reduced.text
    assert 'v(n1)' in DropOutputs().remove(DECK, DropOutputs().units(DECK)).text  # one always stays


def test_transient_halves_down_to_ten_steps():
    reduced = Reducer((ShortenTransient(),)).reduce(DECK, lambda deck: True)
    assert reduced.tstop == 156.2e-12 or 'tran 10p 156.2p' in reduced.text


def test_numbers_round_parameter_and_element_values_but_not_nodes():
    candidates = [d.text for d in SimplifyNumbers().candidates(DECK)]
    assert any('.param p1=5k' in text for text in candidates)
    assert any('C2 n2 0 2f' in text for text in candidates)
    assert not any('n3' in text or 'Xi n1 n2 inv' not in text for text in candidates)


def test_finding_test_selects_the_configurations_a_check_needs(fake, tmp_path):
    testbench = Testbench(Settings())
    assert FindingTest(testbench, 'exact: waveform', tmp_path).configurations() == ['ref', 'exact']
    assert FindingTest(testbench, 'approx: crossing', tmp_path).configurations() == ['ref', 'approx', 'tight']
    assert FindingTest(testbench, 'front-end:', tmp_path).configurations() == ['ref', 'exact']


def test_a_reproducer_reruns_the_finding_with_its_include_files(fake, tmp_path):
    noisy = tmp_path / 'noisy'
    noisy.write_text(f'#!/bin/sh\nexec env FAKE_WARN=extra {fake} "$@"\n')
    noisy.chmod(0o755)
    settings = Settings(Simulators(fake, noisy))
    deck = Deck(fake_deck('.lib "fuzzlib.lib" typ').text,
                files={'fuzzlib.lib': '.include "sub/inner.inc"\n', 'sub/inner.inc': '* nested\n'})
    test = FindingTest(Testbench(settings, '0-3'), 'exact: diagnostics', tmp_path / 'work')
    expected = oracles.judge(test.evidence(deck)).findings
    assert expected == ("exact: diagnostics differ from ref (new: 'Warning: extra')",)
    out = tmp_path / 'reduced'
    for name, content in Reproducer(test, expected[0]).files(deck).items():
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_text(content)
    manifest = json.loads((out / 'repro.json').read_text())
    assert manifest['cpus'] == '0-3'
    done = subprocess.run(['sh', str(out / 'repro.sh')], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert (out / 'repro/exact/sub/inner.inc').exists() and 'exact: exit 0' in done.stdout
    assert oracles.judge(load_evidence(out / 'repro')).findings == expected
    assert "finding: exact: diagnostics differ from ref (new: 'Warning: extra')" in done.stdout


def test_a_reproducer_refuses_changed_binaries_and_include_files(fake, tmp_path):
    copy = tmp_path / 'fake'
    shutil.copy(fake, copy)
    test = FindingTest(Testbench(Settings(Simulators(copy, copy)), '0-3'), 'exact:', tmp_path / 'work')
    deck = Deck(fake_deck().text, files={'lib.inc': '* lib\n'})
    for name, content in Reproducer(test, 'exact: x').files(deck).items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(content)
    (tmp_path / 'lib.inc').write_text('* edited\n')
    done = subprocess.run(['sh', str(tmp_path / 'repro.sh')], capture_output=True, text=True, timeout=60)
    assert done.returncode == 2 and 'lib.inc changed' in done.stderr


def test_a_reproducer_replays_an_extra_configuration_with_its_binary_and_locks(fake, tmp_path):
    binary = tmp_path / 'mine-ngspice'
    shutil.copy(fake, binary)
    mine = Configuration('mine', binary, Claim.EXACT, {'FAKE_WARN': 'extra'}, locks=(str(tmp_path / 'mine.lock'),))
    settings = Settings(Simulators(fake, fake), {**CONFIGURATIONS, 'mine': mine}, lock_wait=12)
    test = FindingTest(Testbench(settings, '0-3'), 'mine: diagnostics', tmp_path / 'work')
    assert test.configurations() == ['ref', 'mine']
    deck = fake_deck()
    [finding] = oracles.judge(test.evidence(deck)).findings
    out = tmp_path / 'bundle'
    for name, content in Reproducer(test, finding).files(deck).items():
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_text(content)
    manifest = json.loads((out / 'repro.json').read_text())
    assert manifest['settings']['lock_wait'] == 12 and str(binary.resolve()) in manifest['binaries']
    assert definitions_from_json(manifest['plan'])['mine'] == mine
    done = subprocess.run(['sh', str(out / 'repro.sh')], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and f'finding: {finding}' in done.stdout, done.stderr
