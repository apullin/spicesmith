from spicesmith.corpus import Corpus, Entry, EntryResult, Expectation, Status, normalized
from spicesmith.deck import Deck
from spicesmith.oracles import Outcome, Verdict


def verdict(*findings):
    return Verdict({'exact:outputs': Outcome.PASS}, tuple(findings), (), (), {})


def entry(expect, check='approx: mid-rail'):
    return Entry('e', Deck('title\n'), check, expect,
                 'approx: mid-rail crossing count differs from tight on v(l20), ref agrees', 'test')


def test_a_fixed_bug_must_pass_every_check():
    assert EntryResult(entry(Expectation.PASS), verdict()).status is Status.PASS
    result = EntryResult(entry(Expectation.PASS), verdict('flags: waveform.txt differs from ref'))
    assert result.status is Status.REGRESSION and result.unexpected == ('flags: waveform.txt differs from ref',)


def test_an_accepted_weakness_may_show_only_its_own_finding():
    known = entry(Expectation.KNOWN)
    own = 'approx: mid-rail crossing count differs from tight on v(l20), ref agrees'
    assert EntryResult(known, verdict(own)).status is Status.KNOWN
    assert EntryResult(known, verdict()).status is Status.FIXED
    elsewhere = 'approx: mid-rail crossing count differs from tight on v(l19), ref agrees'  # same prefix
    assert EntryResult(known, verdict(own, elsewhere)).unexpected == (elsewhere,)
    worse = EntryResult(known, verdict(own, 'exact: diagnostics differ'))
    assert worse.status is Status.REGRESSION and worse.unexpected == ('exact: diagnostics differ',)
    assert EntryResult(known, None).status is Status.SKIPPED


def test_accepted_findings_match_with_their_numbers_abstracted():
    shifted = Entry('e', Deck('t\n'), 'approx: crossing', Expectation.KNOWN,
                    'approx: crossing shift 6.43 ps vs ref 1.13 ps on v(c46)', 'test')
    assert EntryResult(shifted, verdict('approx: crossing shift 7.10 ps vs ref 1.20 ps on v(c46)')).status \
        is Status.KNOWN
    assert EntryResult(shifted, verdict('approx: crossing shift 7.10 ps vs ref 1.20 ps on v(c47)')).status \
        is Status.REGRESSION
    assert normalized('approx: crossing shift 6.43 ps vs ref 1.13 ps on v(c8)@waveform-125.txt') == \
        'approx: crossing shift # ps vs ref # ps on v(c8)@waveform-125.txt'


def test_entries_round_trip(tmp_path):
    original = Entry('rlc', Deck('title\nR1 a 0 1\n'), 'approx:', Expectation.KNOWN, 'approx: x', 'seed 35',
                     'notes', requires=('mine',))
    original.save(tmp_path)
    assert Corpus(tmp_path).entries() == [original]


def test_the_shipped_corpus_loads():
    entries = Corpus().entries()
    assert entries and all(e.deck.outputs for e in entries)
