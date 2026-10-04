import numpy as np

from spicesmith import oracles
from spicesmith.deck import Deck
from spicesmith.oracles import (AccuracyOracle, BitExactOracle, DeterminismOracle, Evidence, FrontEnd,
                                FrontEndOracle, Outcome, Repeat)
from spicesmith.simulation import RunResult

DECK = Deck('deck\n.param vsupply=1.2\n.control\ntran 1p 1n\nwrdata __OUT__/waveform.txt v(d)\nquit\n.endc\n')


def run(name='x', status=0, wave=b' time v(d)\n0 0\n', diagnostics=(), stats=None, aborted=False, env=None):
    return RunResult(name, status, 0.1, {'waveform.txt': wave}, tuple(diagnostics), dict(stats or {}), aborted, (),
                     dict(env or {}))


def square_wave(delay=0.0, ringing=0.0) -> bytes:
    t = np.linspace(0, 1e-9, 4001)
    phase = np.mod(t - 100e-12 - delay, 400e-12)
    v = 1.2 * (np.clip(phase / 10e-12, 0, 1) - np.clip((phase - 200e-12) / 10e-12, 0, 1))
    v = np.where(t < 100e-12 + delay, 0.0, v) + ringing * np.sin(2 * np.pi * t / 20e-12)
    rows = [' time v(d)'] + [f'{a:.15e} {b:.15e}' for a, b in zip(t, v, strict=True)]
    return ('\n'.join(rows) + '\n').encode()


def evidence(**runs) -> Evidence:
    return Evidence(DECK, runs)


def test_bit_exact_passes_identical_runs():
    judgement = BitExactOracle().judge(evidence(ref=run(stats={'equations': 5}), exact=run(stats={'equations': 5})))
    assert judgement.findings == () and {c.outcome for c in judgement.checks} == {Outcome.PASS}


def test_bit_exact_flags_status_output_diagnostics_and_counters():
    ref = run(stats={'equations': 5, 'tran_iterations': 100})
    exact = run(wave=b' time v(d)\n0 1\n', diagnostics=["warning, can't find model 'p3'"],
                stats={'equations': 6, 'tran_iterations': 100})
    assert BitExactOracle().judge(evidence(ref=ref, exact=exact)).findings == (
        'exact: waveform.txt differs from ref',
        "exact: diagnostics differ from ref (new: 'warning, can't find model 'p3'')",
        'exact: counters differ from ref (equations 6 vs 5)')
    judgement = BitExactOracle().judge(evidence(ref=ref, exact=run(aborted=True)))
    assert judgement.findings[0] == 'exact: exit status aborted vs ref 0'
    assert {c.key: c.outcome for c in judgement.checks}['exact:outputs'] is Outcome.SKIP


def test_accuracy_passes_a_run_as_close_as_ref():
    judgement = AccuracyOracle().judge(evidence(ref=run(wave=square_wave(1e-12)),
                                                approx=run(wave=square_wave(-1.5e-12)),
                                                tight=run(wave=square_wave())))
    assert judgement.findings == () and judgement.inconclusive == ()
    assert judgement.metrics['ref_shift_ps'] > 0.9 and judgement.metrics['approx_shift_ps'] > 1.4


def test_accuracy_flags_a_larger_shift_and_a_failed_run():
    tight, ref = run(wave=square_wave()), run(wave=square_wave(0.5e-12))
    shifted = AccuracyOracle().judge(evidence(ref=ref, approx=run(wave=square_wave(5e-12)), tight=tight))
    assert shifted.findings == ('approx: crossing shift 5.00 ps vs ref 0.50 ps on v(d)',)
    failed = AccuracyOracle().judge(evidence(ref=ref, approx=run(aborted=True), tight=tight))
    assert failed.findings == ('approx: exit status aborted, ref succeeded',)
    assert {c.key: c.outcome for c in failed.checks}['approx:dv'] is Outcome.SKIP


def test_accuracy_flags_extra_crossings_when_ref_agrees():
    judgement = AccuracyOracle().judge(evidence(ref=run(wave=square_wave()), tight=run(wave=square_wave()),
                                                approx=run(wave=square_wave(ringing=0.8))))
    assert judgement.findings == ('approx: mid-rail crossing count differs from tight on v(d), ref agrees',)


def test_cases_where_ref_fails_are_inconclusive_but_still_bit_exact_checked():
    verdict = oracles.judge(evidence(ref=run('ref', aborted=True), exact=run('exact', aborted=True),
                                     approx=run('approx'), tight=run('tight', wave=square_wave())))
    assert verdict.inconclusive == ('ref aborted',) and not verdict.failed
    assert verdict.checks['exact:status'] is Outcome.PASS and verdict.checks['approx:status'] is Outcome.SKIP


def test_determinism():
    repeat = Repeat('ref', run(), run(wave=b' time v(d)\n0 2\n'))
    judgement = DeterminismOracle().judge(Evidence(DECK, {'ref': run()}, repeats=(repeat,)))
    assert judgement.findings == ('ref: not deterministic (outputs differ between two runs)',)


REF_DUMP = ['**************** uncommented deck **************', '',
            '     1       1  title',
            '    20      20  .model x1:dclamp d is=1e-14',
            '    21      21  .model x1:dunused d is=1e-15',
            '    22      22  d.x1.d1 0 a x1:dclamp',
            '', '****************** complete deck ***************', '',
            '     1       1  title',
            '    20      20  .model x1:dclamp d is=1e-14',
            '    21      21  .model x1:dunused d is=1e-15',
            '    22      22  d.x1.d1 0 a x1:dclamp']


def candidate(dump, commented=('dunused',)):
    """What the candidate dumps: the models in `commented` turned into comments (unscoped,
    before substitution) and dropped from the uncommented listing."""
    out = []
    for line in dump:
        words = line.split()
        if len(words) > 3 and words[2] == '.model' and words[3].split(':')[1] in commented:
            if 'complete' in ''.join(out):
                out.append(line.replace('.model ' + words[3], '*model ' + words[3].split(':')[1]))
            continue
        out.append(line)
    return out


def test_debug_dumps_may_differ_by_commented_unused_models():
    comparison = FrontEndOracle.compare(REF_DUMP, candidate(REF_DUMP))
    assert comparison.difference is None and comparison.commented == {'x1:dunused'}


def test_debug_dumps_catch_other_changes_and_referenced_models():
    changed = candidate(REF_DUMP)
    changed[-1] = changed[-1].replace('0 a', '0 b')
    assert FrontEndOracle.compare(REF_DUMP, changed).difference.startswith('line 12: ref')
    comparison = FrontEndOracle.compare(REF_DUMP, candidate(REF_DUMP, commented=('dclamp', 'dunused')))
    assert comparison.difference is None
    assert comparison.commented & FrontEndOracle.references(REF_DUMP) == {'x1:dclamp'}


def test_front_end_oracle_on_dump_directories(tmp_path):
    for who, dump in (('ref', REF_DUMP), ('cand', candidate(REF_DUMP))):
        (tmp_path / who).mkdir()
        for name in oracles.DEBUG_FILES:
            (tmp_path / who / name).write_text('\n'.join(dump) + '\n')
    judgement = FrontEndOracle().judge(Evidence(DECK, {'ref': run()}, FrontEnd(tmp_path / 'ref', tmp_path / 'cand')))
    assert [(c.key, c.outcome) for c in judgement.checks] == [('front-end:identity', Outcome.PASS)]


def table(scale_name, scale, **columns):
    rows = [f' {scale_name} ' + ' '.join(f'{n} {n}' if np.iscomplexobj(v) else n for n, v in columns.items())]
    for i, x in enumerate(scale):
        values = []
        for v in columns.values():
            values += [v[i].real, v[i].imag] if np.iscomplexobj(v) else [v[i]]
        rows.append(' '.join(f'{z:.15e}' for z in [x] + values))
    return ('\n'.join(rows) + '\n').encode()


def test_operating_points_and_dc_sweeps_compare_vector_by_vector():
    sweep = np.linspace(0, 1.2, 7)

    def dc(offset):
        return RunResult('x', 0, 0.1, {'dc.txt': table('v-sweep', sweep, **{'v(a)': 1.2 - sweep + offset})}, (), {})
    deck = Deck('d\n.param vsupply=1.2\n.control\ndc v1 0 1.2 0.2\nwrdata __OUT__/dc.txt v(a)\n'
                'tran 1p 1n\nquit\n.endc\n')
    ok = AccuracyOracle().judge(Evidence(deck, {'ref': dc(1e-4), 'approx': dc(-2e-4), 'tight': dc(0.0)}))
    assert ok.findings == () and {c.key: c.outcome for c in ok.checks}['approx:dc'] is Outcome.PASS
    bad = AccuracyOracle().judge(Evidence(deck, {'ref': dc(1e-4), 'approx': dc(0.01), 'tight': dc(0.0)}))
    assert bad.findings == ('approx: dc.txt v(a) off by 0.01 V vs ref 0.0001 V',)


def test_ac_sweeps_compare_relative_complex_errors():
    frequency = np.logspace(6, 10, 9)
    response = 1 / (1 + 1j * frequency / 1e9)
    def ac(scale):
        return RunResult('x', 0, 0.1, {'ac.txt': table('frequency', frequency, **{'v(o)': response * scale})}, (), {})
    deck = Deck('d\n.param vsupply=1.2\n.control\nac dec 2 1meg 10g\nwrdata __OUT__/ac.txt v(o)\n'
                'tran 1p 1n\nquit\n.endc\n')
    ok = AccuracyOracle().judge(Evidence(deck, {'ref': ac(1.0001), 'approx': ac(0.9998), 'tight': ac(1.0)}))
    assert ok.findings == ()
    bad = AccuracyOracle().judge(Evidence(deck, {'ref': ac(1.0001), 'approx': ac(1.01), 'tight': ac(1.0)}))
    assert bad.findings == ('approx: ac.txt v(o) off by 0.01 relative vs ref 0.0001 relative',)


# Output integrity (audit A1): a table that cannot be compared never passes quietly.

GOOD = square_wave()


def accuracy(candidate: bytes, tight: bytes = GOOD, ref: bytes = GOOD):
    return AccuracyOracle().judge(evidence(ref=run(wave=ref), approx=run(wave=candidate), tight=run(wave=tight)))


def outputs_check(judgement):
    return {c.key: c.outcome for c in judgement.checks}['approx:outputs']


def test_unusable_candidate_tables_are_findings():
    rows = GOOD.decode().splitlines()
    cases = {
        b'': 'empty',
        b'not-a-table\n': 'no vectors in the header',
        b' time v(d)\n0 x\n': 'not a numeric table',
        b' time v(d)\n0 1 2\n': 'ragged rows',
        b' time v(d)\n0 nan\n1e-9 1\n': 'non-finite values',
        b' time v(d)\n0 inf\n1e-9 1\n': 'non-finite values',
        b' time v(d)\n0 0\n-inf 1\n': 'non-finite values',
        b' time v(d)\n0 0\n0 1\n1e-9 1\n': 'time does not increase',
        b' time v(other)\n0 0\n1e-9 1\n': 'requested vectors differ (missing v(d); extra v(other))',
        b' time v(d) v(e)\n0 0 0\n1e-9 1 1\n': 'requested vectors differ (extra v(e))',
        ('\n'.join([' frequency v(d)'] + rows[1:]) + '\n').encode(): 'scale frequency where tran requires time',
    }
    for data, problem in cases.items():
        judgement = accuracy(data)
        assert judgement.findings == (f'approx: waveform.txt unusable ({problem})',), data
        assert outputs_check(judgement) is Outcome.FAIL and judgement.inconclusive == ()


def test_unusable_reference_tables_make_the_case_inconclusive():
    cases = (('tight', b'garbage', 'tight: waveform.txt unusable (no vectors in the header)'),
             ('ref', b' time v(d)\n0 nan\n1e-9 0\n', 'ref: waveform.txt unusable (non-finite values)'),
             ('ref', b' time v(e)\n0 0\n1e-9 1\n',
              'ref: waveform.txt unusable (requested vectors differ (missing v(d); extra v(e)))'))
    for who, data, reason in cases:
        judgement = accuracy(GOOD, **{who: data})
        assert judgement.inconclusive == (reason,)
        assert judgement.findings == () and outputs_check(judgement) is Outcome.SKIP
        assert all(c.outcome is not Outcome.PASS for c in judgement.checks if c.key != 'approx:status')


def test_missing_dc_and_ac_vectors_are_findings():
    sweep = np.linspace(0, 1.2, 7)
    frequency = np.logspace(6, 10, 9)
    deck = Deck('d\n.param vsupply=1.2\n.control\ndc v1 0 1.2 0.2\nwrdata __OUT__/dc.txt v(a) v(b)\nac dec 2 1meg 10g\n'
                'wrdata __OUT__/ac.txt v(a)\ntran 1p 1n\nquit\n.endc\n')
    def outputs(dc_vectors, ac_points):
        dc = table('v-sweep', sweep, **{name: 1.2 - sweep for name in dc_vectors})
        ac = table('frequency', ac_points, **{'v(a)': 1 / (1 + 1j * ac_points / 1e9)})
        return RunResult('x', 0, 0.1, {'dc.txt': dc, 'ac.txt': ac}, (), {})
    good = outputs(('v(a)', 'v(b)'), frequency)
    judgement = AccuracyOracle().judge(Evidence(deck, {'ref': good, 'tight': good,
                                                       'approx': outputs(('v(a)',), frequency[:-1])}))
    assert judgement.findings == ('approx: dc.txt unusable (requested vectors differ (missing v(b)))',
                                  'approx: ac.txt unusable (AC sweep has 8 rows, expected 9)')


def test_summaries_refuse_non_finite_numbers():
    import json
    import pytest
    with pytest.raises(ValueError):
        json.dumps({'dv': float('nan')}, allow_nan=False)


def test_every_counter_ref_prints_is_required_including_fill_in():
    ref = run(stats={'equations': 5, 'fillin': -2, 'tran_iterations': 100})
    judgement = BitExactOracle().judge(evidence(ref=ref, exact=run(stats={'fillin': -1, 'tran_iterations': 100})))
    assert [f for f in judgement.findings if 'counters' in f] == [
        'exact: counters missing (equations)', 'exact: counters differ from ref (fillin -1 vs -2)']
    assert {c.key: c.outcome for c in judgement.checks}['exact:matrix'] is Outcome.FAIL
