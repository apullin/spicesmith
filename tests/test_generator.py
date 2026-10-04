import hashlib
import re

from spicesmith import generator
from spicesmith.deck import eng, spice_number

# The seed -> deck mapping of this generator version; a change needs a new VERSION.
DIGESTS = {1: 'a1e9216fed92fddf', 2: 'e5649218785df43c', 3: 'bd2fc72380c873b1',
           35: 'b3eed11bbd9d41bf', 228: '3c567c821610174e'}


def test_same_seed_same_deck():
    assert generator.generate(7) == generator.generate(7)
    assert generator.generate(7).text != generator.generate(8).text


def test_the_seed_to_deck_mapping_is_pinned_per_version():
    assert generator.VERSION == 7
    digests = {seed: hashlib.sha256(generator.generate(seed).text.encode()).hexdigest()[:16] for seed in DIGESTS}
    assert digests == DIGESTS


def test_rings_have_an_odd_number_of_inversions():
    for seed in range(1, 300):
        deck = generator.generate(seed)
        for block in deck.info.blocks:
            if block.kind == 'ring':
                stages = [net for net in block.nets if net.startswith('r')]
                assert len(stages) % 2 == 1, seed  # the NAND plus an even number of inverters


def test_latches_hold_a_state_and_are_never_released_at_mid_rail():
    for seed in range(1, 300):
        deck = generator.generate(seed)
        for block in deck.info.blocks:
            if block.kind != 'latch':
                continue
            q, qb, bl, _, wl = block.nets
            assert f'.ic v({q})=' in deck.text and f'.ic v({qb})=0' in deck.text
            word = pulse(deck, wl)
            bit = pulse(deck, bl)
            assert bit['period'] == 2 * word['period'] or abs(bit['period'] / word['period'] - 2) < 1e-3
            word_low = word['delay'] + word['rise'] + word['width'] + word['fall']  # end of the first write
            assert word['delay'] > bit['delay'] + bit['rise'] and word_low < bit['delay'] + word['period'], seed


def pulse(deck, net):
    line = next(line for line in deck.lines if re.match(rf'V\S+ {re.escape(net)} 0 (AC 1 )?PULSE', line))
    values = [spice_number(t) for t in line[line.index('(') + 1:line.index(')')].split()]
    return dict(zip(('low', 'high', 'delay', 'rise', 'fall', 'width', 'period'), values, strict=True))


def test_resonances_are_labelled_by_resolution():
    labels = set()
    for seed in range(1, 200):
        for block in generator.generate(seed).info.blocks:
            if block.kind == 'rlc':
                a = block.attributes
                assert ('under-resolved' in block.classes) == (a['steps_per_period'] < generator.UNDER_RESOLVED)
                assert ('high-q' in block.classes) == (a['q'] > generator.HIGH_Q)
                labels |= block.classes
    assert {'under-resolved', 'resolved', 'high-q'} <= labels


def test_counter_collection_belongs_to_the_optional_adapter():
    from spicesmith.ngspice import prepare
    deck = generator.generate(4)
    assert 'rusage all' not in deck.text and '.control' not in deck.text
    assert 'rusage all' in prepare(deck).text


def test_circuit_info_attributes_every_net_to_one_block():
    deck = generator.generate(228)
    info = deck.info
    nets = [net for block in info.blocks for net in block.nets]
    assert len(nets) == len(set(nets)) and info.blocks[0].kind == 'supply'
    assert all(f'v({net})' in deck.text for net in nets)
    assert info.block_of(info.blocks[1].nets[0]) is info.blocks[1]
    assert (info.vdd, info.tstop) == (deck.vdd, deck.tstop) and info.generator == generator.VERSION


def test_blocks_are_an_open_set():
    class Resistor(generator.Block):
        kind = 'resistor'

        def build(self, circuit, vdd):
            circuit.elements.append(f'R{circuit.name("r")} {vdd} {circuit.net("x")} 1k')

    deck = generator.Generator(blocks=(Resistor(),)).generate(3)
    assert [b.kind for b in deck.info.blocks][1:] == ['resistor'] * (len(deck.info.blocks) - 1)


def test_eng_formats_with_scale_suffixes():
    assert eng(1.5e-15) == '1.5f'
    assert eng(10e-12) == '10p'
    assert eng(4.7e3) == '4.7k'
    assert eng(2e6) == '2meg'


def test_the_size_knob_builds_arrays_of_identical_copies():
    small, large = generator.generate(12), generator.generate(12, scale=10)
    assert large.info.scale == 10 and 'scale 10' in large.lines[0]
    blocks = [b for b in large.info.blocks if b.kind != 'supply']
    assert len(blocks) == 10 * (len(small.info.blocks) - 1)
    first, second = blocks[0], blocks[1]
    assert first.kind == second.kind and len(first.nets) == len(second.nets)
    def sizes(block):  # the parameter assignments of the block's subcircuit instances
        lines = [line for line in large.lines if line.startswith('X')
                 and any(f' {net} ' in f' {line} ' for net in block.nets)]
        return [' '.join(word for word in line.split() if '=' in word) for line in lines]
    assert sizes(first) == sizes(second)  # same sizes, other nets


def test_large_decks_observe_a_bounded_sample_of_nets():
    deck = generator.Generator(scale=50, max_observed=60).generate(12)
    observed = deck.output_requests['waveform.txt'][1]
    assert len(observed) == 60 and all(f'{net[2:-1]} 0' in deck.text for net in observed)


def test_the_local_library_travels_with_its_deck(tmp_path):
    deck = next(d for d in map(generator.generate, range(1, 100)) if d.files)
    assert set(deck.files) == {'fuzzlib.lib', 'fuzzcell.inc'} and '.lib "fuzzlib.lib" ' in deck.text
    deck.save(tmp_path)
    from spicesmith.deck import Deck
    assert Deck.load(tmp_path).files == deck.files


def test_decks_add_operating_points_dc_ac_sweeps_and_binned_models():
    texts = [generator.generate(seed).text for seed in range(1, 120)]
    for marker in ('* spicesmith-output op.txt', '* spicesmith-output dc.txt', '* spicesmith-output ac.txt',
                   '.model nbin.1 nmos level=54'):
        assert any(marker in text for text in texts), marker
    for text in texts:
        if '* spicesmith-output ac.txt' in text:
            assert ' AC 1 ' in text


def normalized_copies(deck):
    """Each block's element lines with generated names (letters and a counter) reduced to
    their prefix: what must be the same in every copy of an array. Leak resistors and the
    AC input marker are deck-wide choices made after the blocks, so they are left out."""
    elements = deck.text.split('.control')[0].splitlines()
    copies = []
    for block in deck.info.blocks[1:]:
        lines = [line.replace(' AC 1 ', ' ') for line in elements if line[:1].isalpha()
                 and not line.startswith('Rleak') and any(re.search(rf'\b{re.escape(net)}\b', line)
                                                         for net in block.nets)]
        copies.append([re.sub(r'\b([A-Za-z]+)\d+\b', r'\1#', line) for line in lines])
    return copies


def test_every_kind_of_block_copies_identically_at_scale():
    for block in generator.BLOCKS:
        for seed in range(1, 6):
            copies = normalized_copies(generator.Generator(blocks=(block,), scale=3).generate(seed))
            for array in range(0, len(copies), 3):
                first, *others = copies[array:array + 3]
                assert first and all(other == first for other in others), (block.kind, seed)
