import pytest

from spicesmith import generator
from spicesmith.deck import TIGHT_LEVELS, Deck, spice_number

DECK = generator.generate(5)


def test_outputs_analyses_and_parameters_of_a_generated_deck():
    assert DECK.outputs == ['waveform.txt']
    assert len(DECK.analyses) == 1 and DECK.analyses[0].startswith('tran ')
    assert DECK.tstop == pytest.approx(DECK.info.tstop) and DECK.vdd == pytest.approx(DECK.info.vdd)
    assert not DECK.has_small_signal_analysis


def test_tightened_rewrites_tolerances_and_keeps_everything_else():
    tight = DECK.tightened()
    options = tight.options
    assert {k: options[k] for k in TIGHT_LEVELS[0]} == TIGHT_LEVELS[0]
    assert 'klu' not in options and options['method'] in ('gear', 'trap') and 'seed' in options
    assert DECK.tightened(3).options['reltol'] == '1e-5'
    assert tight.info is DECK.info
    unchanged = [line for line in tight.lines if not line.startswith('.option')]
    assert unchanged == [line for line in DECK.lines if not line.startswith('.option')]


def test_with_options_rewrites_in_place_and_adds_missing_keys():
    deck = Deck('title\n.option reltol=1e-3 method=gear\n.options trtol=7 seed=3\nR1 a 0 1k\n')
    assert deck.with_options({'trtol': '1', 'abstol': '1e-15'}).text == \
        'title\n.option reltol=1e-3 method=gear abstol=1e-15\n.options trtol=1 seed=3\nR1 a 0 1k\n'
    assert Deck('title\nR1 a 0 1k').with_options({'trtol': '1'}).text == 'title\n.option trtol=1\nR1 a 0 1k'


def test_front_end_deck_parses_but_does_not_simulate():
    front = DECK.front_end()
    head, body, _ = front._control()
    assert front.analyses == [] and front.outputs == []
    assert body[-1] == 'quit' and 'set wr_vecnames' in body
    assert not any(line.lower().startswith('.tran') for line in head)
    assert '.model smith_n nmos' in front.text


def test_small_signal_analyses_are_recognized():
    assert Deck('t\n.control\nac dec 10 1 1g\nquit\n.endc\n').has_small_signal_analysis


@pytest.mark.parametrize('token, expected', [('10p', 1e-11), ('1.5meg', 1.5e6), ('2e-9', 2e-9), ('4.7k', 4.7e3),
                                             ('1m', 1e-3), ('20n', 2e-8), ('10pF', 1e-11), ('-3u', -3e-6)])
def test_spice_numbers(token, expected):
    assert spice_number(token) == pytest.approx(expected)


def test_included_files_may_be_nested_but_not_escape(tmp_path):
    deck = Deck('title\n.include "models/cell.inc"\n', files={'models/cell.inc': '* cell\n'})
    deck.save(tmp_path / 'a')
    assert (tmp_path / 'a/models/cell.inc').read_text() == '* cell\n'
    assert Deck.load(tmp_path / 'a').files == deck.files
    for bad in ('/etc/cell.inc', '../cell.inc'):
        with pytest.raises(ValueError, match='inside the deck directory'):
            Deck('t\n', files={bad: ''}).save(tmp_path / 'b')
