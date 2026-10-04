import json

import pytest

from spicesmith.config import CONFIGURATIONS, Binary, Claim, CpuSet, Simulators, load_extra_configurations


def test_cpu_sets_split_requested_cpus_into_jobs():
    assert [str(s) for s in CpuSet.split('8-31', 4)] == ['8-11', '12-15', '16-19', '20-23', '24-27', '28-31']


def test_cpu_sets_join_lists_and_drop_leftovers():
    assert [str(s) for s in CpuSet.split('8-13,20,22', 4)] == ['8-11', '12,13,20,22']
    assert CpuSet.split('8-10', 4) == []


def test_extra_configurations_extend_flags_and_resolve_binaries_next_to_their_file(tmp_path):
    (tmp_path / 'extra.json').write_text(json.dumps([
        {'name': 'mine', 'binary': 'builds/ngspice', 'claim': 'approximate', 'extends': 'approx',
         'flags': {'MY_FLAG': '1'}, 'locks': ['/tmp/a.lock', '/tmp/b.lock']},
        {'name': 'theirs', 'binary': 'candidate', 'claim': 'exact', 'extends': 'mine'}]))
    extra = load_extra_configurations(tmp_path / 'extra.json')
    assert extra['mine'].binary == tmp_path / 'builds/ngspice' and extra['mine'].locks == ('/tmp/a.lock', '/tmp/b.lock')
    assert extra['mine'].flags == {'MY_FLAG': '1'}
    assert extra['theirs'].binary is Binary.CANDIDATE and extra['theirs'].flags == extra['mine'].flags
    assert extra['theirs'].claim is Claim.EXACT and extra['theirs'].base == 'ref'


@pytest.mark.parametrize('definition, problem', [
    ({'name': 'approx', 'binary': 'candidate', 'claim': 'exact'}, 'new name'),  # would replace a built-in
    ({'name': 'mine', 'binary': 'candidate', 'claim': 'accuracy'}, 'claim'),
    ({'name': 'mine', 'binary': 'candidate', 'claim': 'exact', 'extends': 'nonesuch'}, 'unknown configuration'),
    ({'name': 'mine', 'binary': 'candidate', 'claim': 'exact', 'lock': ['/tmp/a.lock']}, 'unknown keys'),
    ({'name': 'mine', 'binary': 'candidate', 'claim': 'exact', 'flags': {'X': 1}}, 'strings'),
])
def test_extra_configurations_refuse_definitions_that_would_corrupt_a_batch(tmp_path, definition, problem):
    (tmp_path / 'extra.json').write_text(json.dumps([definition]))
    with pytest.raises(ValueError, match=problem):
        load_extra_configurations(tmp_path / 'extra.json')


def test_claims_and_binaries(fake):
    claims = {name: c.claim for name, c in CONFIGURATIONS.items()}
    assert claims == {'ref': Claim.REFERENCE, 'exact': Claim.EXACT,
                      'approx': Claim.APPROXIMATE, 'tight': Claim.ACCURACY}
    assert all(not c.flags for c in CONFIGURATIONS.values())
    simulators = Simulators(fake, fake)
    assert simulators.path(CONFIGURATIONS['tight'].binary) == simulators.reference
    assert simulators.path(CONFIGURATIONS['approx'].binary) == simulators.candidate


def test_caller_flags_are_not_rewritten_for_an_analysis(tmp_path):
    (tmp_path / 'extra.json').write_text(json.dumps([
        {'name': 'mine', 'binary': 'candidate', 'claim': 'exact', 'flags': {'MY_FLAG': '1'}}]))
    mine = load_extra_configurations(tmp_path / 'extra.json')['mine']
    assert mine.flags_for(True) == mine.flags_for(False) == {'MY_FLAG': '1'}
