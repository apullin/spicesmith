from spicesmith.harness import CaseSummary
from spicesmith.report import Report, pattern_of

BLOCKS = [{'kind': 'supply', 'classes': [], 'nets': ['vdd1']},
          {'kind': 'rlc', 'classes': ['linear', 'rlc'], 'nets': ['l2', 'l3']},
          {'kind': 'inverter-chain', 'classes': ['digital'], 'nets': ['in4', 'c5']}]


def summary(seed, findings=(), checks=None, nodes=None, status=None):
    nodes = nodes or {}
    metrics = {'nodes': nodes, 'ref_shift_ps': 1.0, 'approx_shift_ps': 1.0, 'ref_tran_iterations': 100.0,
               'approx_tran_iterations': 90.0}
    return CaseSummary(seed, tuple(findings), (), (), status or {'ref': 0, 'approx': 0}, {'ref': 0.2, 'approx': 0.3},
                       None, (), {'generator': 1, 'blocks': BLOCKS}, checks or {'approx:dv': 'pass'}, metrics,
                       {'candidate': {'sha256': 'ae77fb55f296aa'}})


def node(timing, ref, approx):
    return {'timing': timing, 'crossings': 3, 'ref': list(ref), 'approx': list(approx)}


def test_patterns_abstract_numbers_and_nodes():
    assert pattern_of('approx: crossing shift 5.00 ps vs ref 0.50 ps on v(l3)') == \
        'approx: crossing shift # ps vs ref # ps on <node>'
    assert pattern_of("exact: diagnostics differ from ref (new: 'warning, can't find model 'p12'')") == \
        "exact: diagnostics differ from ref (new: 'warning, can't find model 'p#'')"


def test_check_rates_and_clusters():
    report = Report((summary(1, ['approx: crossing shift 5.00 ps vs ref 0.50 ps on v(l3)'], {'approx:dv': 'fail'}),
                     summary(2, ['approx: crossing shift 7.00 ps vs ref 0.20 ps on v(c5)']),
                     summary(3, checks={'approx:dv': 'skip'})))
    [rate] = report.check_rates()
    assert (rate.key, rate.passed, rate.failed, rate.skipped) == ('approx:dv', 1, 1, 1)
    [cluster] = report.clusters()
    assert cluster.pattern == 'approx: crossing shift # ps vs ref # ps on <node>' and cluster.seeds == (1, 2)


def test_accuracy_compares_per_case_and_per_circuit_class():
    report = Report((
        summary(1, nodes={'v(l3)': node(True, (True, 2.0, 0.1), (True, 5.0, 0.1)),
                          'v(c5)': node(True, (True, 1.0, 0.0), (True, 0.5, 0.0))}),
        summary(2, nodes={'v(c5)': node(True, (True, 3.0, 0.0), (True, 1.0, 0.0)),
                          'v(vdd1)': node(False, (True, 0.0, 0.002), (True, 0.0, 0.001))})))
    comparisons = {(c.scope, c.measure): c for c in report.comparisons('approx')}
    overall = comparisons['all', 'crossing shift (ps)']
    assert overall.pairs == ((2.0, 5.0), (3.0, 1.0)) and (overall.better, overall.worse) == (1, 1)
    rlc = comparisons['rlc', 'crossing shift (ps)']
    assert rlc.pairs == ((2.0, 5.0),) and rlc.worse == 1
    digital = comparisons['digital', 'crossing shift (ps)']
    assert digital.better == 2 and digital.candidate.median == 0.75
    assert comparisons['all', 'max dV (mV)'].pairs == ((2.0, 1.0),)


def test_a_custom_configuration_is_measured_against_its_declared_base():
    def case(seed, stage_one, stage_two, status):
        entry = {'timing': True, 'crossings': 3, 'ref': [True, 0.5, 0.0], 'stage-one': list(stage_one),
                 'stage-two': list(stage_two)}
        return CaseSummary(seed, (), (), (), status, {}, None, (),
                           {'vdd': 1.2, 'tstop': 5e-9, 'blocks': BLOCKS}, {}, {'nodes': {'v(c5)': entry}},
                           {'bases': {'stage-two': 'stage-one'}})
    # Seed 1: 1 ps worse than stage-one, within the 2 ps slack although 2.5 ps beyond ref's 0.5 ps.
    # Seed 2: 4 ps beyond stage-one. Seed 3: stage-one finished, stage-two aborted.
    report = Report((case(1, (True, 2.0, 0.0), (True, 3.0, 0.0), {'ref': 0, 'stage-one': 0, 'stage-two': 0}),
                     case(2, (True, 1.0, 0.0), (True, 5.0, 0.0), {'ref': 0, 'stage-one': 0, 'stage-two': 0}),
                     case(3, (True, 1.0, 0.0), (True, 1.0, 0.0), {'ref': 0, 'stage-one': 0, 'stage-two': 'aborted'})))
    [shift] = [c for c in report.comparisons('stage-two') if c.scope == 'all']
    assert shift.base == 'stage-one' and shift.pairs == ((2.0, 3.0), (1.0, 5.0), (1.0, 1.0))
    assert shift.delta.maximum == 4.0 and shift.worst() == [(2, 4.0), (1, 1.0)]
    budget = report.ablation('stage-two')
    assert (budget.base_finished, budget.both_finished, budget.beyond_base) == (3, 2, (2, 3))


def test_markdown_and_json_render():
    report = Report((summary(1), summary(2, status={'ref': 'aborted', 'approx': 0})), ('runs/x',))
    text = report.markdown()
    assert '2 cases: 0 failing' in text and 'ref failed in 1' in text and 'ae77fb55f296' in text
    assert 'Transient iterations approx/ref: median 0.900' in text
    assert report.to_json()['health']['statuses']['ref'] == {'0': 1, 'aborted': 1}


def test_dv_counts_wherever_the_oracle_checks_it():
    report = Report((summary(1, nodes={'v(l3)': node(True, (False, 9.0, 0.3), (True, 1.0, 0.2))}),))
    comparisons = {(c.scope, c.measure): c for c in report.comparisons('approx')}
    assert comparisons['all', 'max dV (mV)'].pairs == ((300.0, 200.0),)  # ref's count disagreed: a dV node
    assert ('all', 'crossing shift (ps)') not in comparisons
