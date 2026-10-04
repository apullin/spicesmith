from spicesmith.attribution import FlagSplit, Trial, Variant
from spicesmith.config import CONFIGURATIONS, Binary, Claim, Configuration
from spicesmith.harness import Settings, Testbench


def test_variants_add_one_extra_flag_or_leave_one_out():
    base = {'COMMON': '1'}
    configs = {**CONFIGURATIONS,
               'exact': Configuration('exact', Binary.CANDIDATE, Claim.EXACT, base),
               'approx': Configuration('approx', Binary.CANDIDATE, Claim.APPROXIMATE,
                                       {**base, 'PREDICT': '2', 'SAFETY': '0.7'})}
    split = FlagSplit(Testbench(Settings(configurations=configs)), 'approx: crossing')
    assert set(split.extra_flags()) == {'PREDICT', 'SAFETY'}
    variants = {v.label: v.flags for v in split.variants()}
    assert len(variants) == 4
    only = variants['only PREDICT']
    assert only.items() >= base.items() and only['PREDICT'] == '2'
    assert 'SAFETY' not in only
    without = variants['without PREDICT']
    assert 'PREDICT' not in without and without['SAFETY'] == '0.7'


def test_conclusion_names_sufficient_and_necessary_flags():
    def trial(label, flag, reproduces):
        return Trial(Variant(f'{label} {flag}', flag, {}), reproduces, ())
    trials = [trial('only', 'A', True), trial('without', 'A', False), trial('only', 'B', False),
              trial('without', 'B', True)]
    assert FlagSplit.conclusion(trials) == 'sufficient alone: A; necessary: A'
    assert FlagSplit.conclusion([trial('only', 'A', False), trial('without', 'A', True)]).startswith('no single flag')
