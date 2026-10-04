import dataclasses
import json
import shutil
from pathlib import Path

import pytest
from conftest import fake_deck

from spicesmith import experiment, profiles
from spicesmith.circuit import Circuit, fingerprint
from spicesmith.cli import main
from spicesmith.config import CONFIGURATIONS, Binary, Claim, Configuration, Simulators
from spicesmith.deck import Deck
from spicesmith.harness import Settings, Testbench, load_evidence
from spicesmith.integrity import output_problems
from spicesmith.oracles import CheckResult, Evidence, Judgement, Outcome, Verdict
from spicesmith.plans import ExecutionPlan
from spicesmith.policies import Policy, assess
from spicesmith.simulation import RunResult


@pytest.mark.parametrize('name,digest', [('feedback', '8b1c2ad9660d2f94'), ('switching', '5d0d19a95c65a00d')])
def test_versioned_examples_pin_structural_seed_mapping(name, digest):
    directory = Path(__file__).resolve().parent.parent / 'examples'
    meta = json.loads((directory / f'{name}.generation.json').read_text())
    circuit = profiles.generate(meta['seed'], profiles.Profile(**meta['profile'])).circuit
    assert circuit.version == 1
    assert fingerprint(circuit.to_json()) == meta['circuit_sha256']
    assert fingerprint(circuit.to_json()).startswith(digest)
    assert circuit.lower().text == (directory / f'{name}.sp').read_text()


def test_observation_does_not_need_a_reference_or_tight_run(fake, tmp_path):
    bench = Testbench(Settings(Simulators(Path('/does/not/exist'), fake)), '0-3')
    result = experiment.run(fake_deck(), bench, ExecutionPlan(('exact',)), tmp_path / 'case')
    assert result.state == 'unassessed' and not result.checks
    assert not (tmp_path / 'case/ref').exists() and not (tmp_path / 'case/tight').exists()
    assert experiment.evaluate(tmp_path / 'case', Policy('exact')).state == 'inconclusive'
    assert json.loads((tmp_path / 'case/observations.json').read_text())['exact']['samples'][0]['complete']
    with pytest.raises(ValueError, match='already exists'):
        experiment.run(fake_deck(), bench, ExecutionPlan(('exact',)), tmp_path / 'case')


def test_saved_evidence_can_be_reinterpreted_with_arbitrary_baseline(fake, tmp_path, monkeypatch):
    configs = dict(CONFIGURATIONS)
    configs['shifted'] = Configuration('shifted', Binary.CANDIDATE, Claim.EXACT, {'FAKE_SHIFT': '0.2'})
    bench = Testbench(Settings(Simulators(fake, fake), configs), '0-3')
    directory = tmp_path / 'case'
    experiment.run(fake_deck(), bench, ExecutionPlan(('exact', 'shifted')), directory)
    monkeypatch.setattr(Testbench, 'run', lambda *a, **kw: pytest.fail('reevaluation must not run a simulator'))
    exact = experiment.evaluate(directory, Policy('exact', 'exact'))
    difference = experiment.evaluate(directory, Policy('difference', 'exact', {'absolute': 0.1}))
    assert exact.state == 'fail' and exact.findings
    assert difference.state == 'unassessed' and difference.selected and not difference.findings
    assert difference.metrics['shifted']['outputs']['waveform.txt']['v(a)']['max_absolute'] == pytest.approx(0.2)
    assert len(list((directory / 'assessments').glob('*.json'))) == 3


def test_bad_artifacts_are_observed_not_silently_compared():
    deck = fake_deck()
    bad = RunResult('exact', 0, outputs={'waveform.txt': b'time v(a) v(b)\n0 0 0\n'})
    evidence = Evidence(deck, {'ref': bad, 'exact': bad})
    assert assess(evidence).state == 'unassessed'
    assert assess(evidence).reasons
    assert assess(evidence, Policy('difference')).state == 'inconclusive'
    assert assess(evidence, Policy('exact')).state == 'inconclusive'
    def custom(e):
        return Verdict.combine((Judgement((CheckResult('custom', Outcome.PASS),)),))
    # A passing user predicate cannot turn unusable artifacts into a successful assessment.
    assert assess(evidence, Policy('custom-property'), custom).state == 'inconclusive'


def test_explicit_plan_ignores_stale_tight_fallback_and_front_end(fake, tmp_path):
    bench = Testbench(Settings(Simulators(fake, fake)), '0-3')
    bench.run(fake_deck(), tmp_path, plan=ExecutionPlan(('ref', 'exact', 'tight'), True))
    fallback = tmp_path / 'tight-2'
    shutil.copytree(tmp_path / 'tight', fallback)
    (fallback / 'run.json').write_text(json.dumps(RunResult('tight', 3).record()))
    bench.run(fake_deck(), tmp_path, plan=ExecutionPlan(('exact', 'tight'), tight_level=1))
    loaded = load_evidence(tmp_path)
    assert loaded.front_end is None
    assert loaded.runs['tight'].ok and loaded.tight_level == 1


def test_a_named_tight_configuration_roundtrips_without_becoming_another_run(fake, tmp_path):
    config = Configuration('custom-tight', Binary.REFERENCE, Claim.ACCURACY, tight=True)
    bench = Testbench(Settings(Simulators(fake, fake), {'custom-tight': config}), '0-3')
    plan = ExecutionPlan(('custom-tight',), repetitions=2)
    bench.run(fake_deck(), tmp_path, plan=plan)
    result = load_evidence(tmp_path)
    assert result.runs['custom-tight'].configuration == 'custom-tight'
    assert result.tight_level == 0 and len(result.samples['custom-tight']) == 2


@pytest.mark.parametrize('kind', profiles.HINTS)
def test_every_hint_has_a_serializable_connected_graph(kind):
    p = profiles.Profile(weights={kind: 1}, disposition='mixed', motifs=4, coupling=1, conditioning=False)
    generated = profiles.generate(17, p)
    circuit = generated.circuit
    assert Circuit.from_json(json.loads(json.dumps(circuit.to_json()))) == circuit
    assert circuit.lower() == profiles.generate(17, p).circuit.lower()
    assert circuit.structural()['connected_components'] == 1
    assert len([d for d in circuit.devices if d.role == 'coupling']) >= 3
    assert not any(d.role == 'conditioning' for d in circuit.devices)
    for d in (d for d in circuit.devices if d.role == 'coupling'):
        target = next(m for m in circuit.motifs if m.id == d.owner)
        assert d.nodes[1] == target.ports['in']
        assert d.nodes[0] in {m.ports['out'] for m in circuit.motifs if m != target}
        assert d.kind == 'R' and d.value > 0


def test_analog_scaffolds_have_real_bias_interstage_feedback_and_switching_edges():
    for kind, required in {'bias': {'bias-reference', 'bias-branch', 'bias-distribution'},
                           'gain': {'tail-bias', 'differential-pair', 'interstage', 'load'},
                           'feedback': {'gain-scaffold', 'feedback', 'compensation'},
                           'rc': {'bridge', 'graph-edge'}, 'rlc': {'resonator', 'resonant-coupling'},
                           'nonlinear': {'clamp', 'memory'},
                           'switching': {'switching-injection', 'switching-load', 'tail-bias'}}.items():
        c = profiles.generate(21, profiles.Profile(weights={kind: 1}, disposition='mixed', motifs=1)).circuit
        assert required <= {d.role for d in c.devices}
        if kind == 'switching':
            m = c.motifs[0]
            assert any(d.nodes == (m.ports['switching'], m.ports['in']) for d in c.devices)
            assert any(d.role == 'regional-supply' and d.nodes[1] == m.ports['supply'] for d in c.devices)


def test_independent_size_reuse_excitation_and_excursion_controls():
    base = profiles.profile('repeated-analog', {'motifs': 12})
    c = profiles.generate(2, base).circuit
    assert c.structural()['distinct_topology_draws'] == c.structural()['distinct_parameter_draws'] == 1
    assert c.structural()['distinct_shapes'] < len(c.motifs)
    assert c.structural()['distinct_source_draws'] == 1
    diverse = profiles.generate(2, dataclasses.replace(base, topology_reuse=0, parameter_reuse=0)).circuit
    assert diverse.structural()['distinct_parameter_draws'] == 12
    assert diverse.structural()['distinct_source_draws'] == 1
    independent = profiles.generate(2, dataclasses.replace(base, source_correlation=0)).circuit
    assert independent.structural()['distinct_source_draws'] == 12
    small = profiles.generate(3, profiles.profile('nonlinear-small')).circuit
    large = profiles.generate(3, profiles.profile('nonlinear-large')).circuit
    assert [(d.kind, d.nodes, d.parameters) for d in small.devices] == [
        (d.kind, d.nodes, d.parameters) for d in large.devices]
    assert max(d.stimulus.get('amplitude', 0) for d in large.devices) > 10 * max(
        d.stimulus.get('amplitude', 0) for d in small.devices)


def test_changing_sources_or_parameter_reuse_does_not_resample_connectivity():
    p = profiles.profile('mixed-time-scales', {'motifs': 20})
    a = profiles.generate(7, p).circuit
    for variation in ({'source_correlation': 1.0}, {'parameter_reuse': 1.0}):
        b = profiles.generate(7, dataclasses.replace(p, **variation)).circuit
        assert [(d.kind, d.nodes, d.role) for d in a.devices] == [(d.kind, d.nodes, d.role) for d in b.devices]


@pytest.mark.parametrize('axis,role', [('loading', 'load'), ('bias_current', 'bias-reference'),
                                      ('compensation', 'compensation'), ('supply_impedance', 'regional-supply')])
def test_analog_parameter_controls_change_the_generated_values(axis, role):
    p = profiles.profile('coupled-feedback', {'motifs': 20})
    a = profiles.generate(9, p).circuit
    b = profiles.generate(9, dataclasses.replace(p, **{axis: getattr(p, axis) * 2})).circuit
    av = [d.value for d in a.devices if d.role == role and d.value]
    bv = [d.value for d in b.devices if d.role == role and d.value]
    assert av and av != bv
    assert [d.nodes for d in a.devices] == [d.nodes for d in b.devices]


def test_generation_rejections_are_bounded_and_retained():
    p = profiles.Profile(max_devices=5, max_attempts=3)
    with pytest.raises(profiles.GenerationError) as error:
        profiles.generate(7, p)
    assert len(error.value.history) == 3
    assert all('budget' in h['reason'] for h in error.value.history)
    with pytest.raises(ValueError, match='transistor-bearing'):
        profiles.Profile(weights={'rc': 1}, target_transistors=23000)


def test_normalized_profile_identity_does_not_depend_on_json_number_spelling():
    a = profiles.Profile(weights={'gain': 1}, coupling=1, vdd=1)
    b = profiles.Profile(weights={'gain': 1.0}, coupling=1.0, vdd=1.0)
    assert (fingerprint(profiles.generate(13, a).circuit.to_json())
            == fingerprint(profiles.generate(13, b).circuit.to_json()))


@pytest.mark.parametrize('disposition', ['digital', 'analog', 'mixed'])
def test_macro_scale_generation_without_simulation(disposition):
    p = profiles.profile(f'large-{disposition}')
    c = profiles.generate(42, p).circuit
    stats = c.structural()
    assert 23000 <= stats['transistors'] < 23100
    assert stats['devices'] <= p.max_devices and stats['nets'] <= p.max_nets
    assert stats['observation_scope']['observed_nets'] <= p.max_observed
    assert stats['observation_scope']['unobserved_nets'] > 1000
    assert stats['roles']['coupling'] > 100 and stats['hierarchy']['regions'] > 1
    assert stats['distinct_parameter_draws'] > 100
    assert len(c.lower().text) > 100000
    if disposition == 'mixed':
        assert {'logic', 'gain', 'switching'} <= set(stats['motifs'])


def test_transforms_preserve_actual_parent_structure_and_lineage():
    c = profiles.generate(13, profiles.profile('mixed-time-scales')).circuit
    for operation in ('diversify', 'excitation', 'reconnect', 'duplicate'):
        changed = profiles.transform(c, operation, 99).circuit
        assert changed.history[-1]['parent'] == fingerprint(c.to_json())
        assert changed == profiles.transform(c, operation, 99).circuit
        if operation == 'duplicate':
            assert changed.devices[:len(c.devices)] == c.devices
            assert len(changed.motifs) == 2 * len(c.motifs)
        else:
            assert [(d.id, d.kind) for d in changed.devices] == [(d.id, d.kind) for d in c.devices]
            if operation != 'reconnect':
                assert [d.nodes for d in changed.devices] == [d.nodes for d in c.devices]
    with pytest.raises(ValueError, match='budget'):
        profiles.transform(dataclasses.replace(c, profile={**c.profile, 'max_devices': len(c.devices)}), 'duplicate', 2)
    twice = profiles.transform(profiles.transform(c, 'duplicate', 1).circuit, 'duplicate', 2).circuit
    assert len(twice.motifs) == 4 * len(c.motifs)


def test_cli_profile_generation_and_observation(fake, tmp_path, capsys):
    path = tmp_path / 'sample.sp'
    main(['gen', '7', str(path), '--profile', 'coupled-feedback'])
    assert path.with_suffix('.circuit.json').exists()
    main(['observe', str(path), '--out', str(tmp_path / 'run'), '--configurations', 'exact',
          '--candidate', str(fake), '--reference', '/does/not/exist'])
    assert 'unassessed' in capsys.readouterr().out
    assert Deck.load(tmp_path, path.name).outputs


def test_observation_metrics_do_not_overflow_on_finite_large_samples():
    deck = fake_deck()
    data = b'time v(a) v(b)\n0 1e200 0\n1e-9 1e200 0\n'
    e = Evidence(deck, {'ref': RunResult('ref', 0, outputs={'waveform.txt': data}),
                        'exact': RunResult('exact', 0, outputs={'waveform.txt': data})})
    result = assess(e, Policy('difference'))
    assert result.state == 'unassessed'
    assert result.metrics['exact']['outputs']['waveform.txt']['v(a)']['max_absolute'] == 0
    from spicesmith.policies import observations
    assert observations(e)['ref']['samples'][0]['artifacts']['waveform.txt']['signals']['v(a)']['rms'] == 1e200


@pytest.mark.simulator
def test_small_switching_profile_uses_portable_models(tmp_path, real_settings):
    p = profiles.profile('switching-disturbance', {'motifs': 1,
                                                  'time_scale_spread': 0, 'tstop': 20e-9, 'maxstep': 100e-12})
    deck = profiles.generate(19, p).circuit.lower()
    assert '.model n nmos' in deck.text and '.control' not in deck.text
    e = Testbench(real_settings).run(deck, tmp_path, plan=ExecutionPlan(('ref', 'exact')))
    assert all(r.ok and not output_problems(deck, r) for r in e.runs.values())


@pytest.mark.simulator
@pytest.mark.parametrize('kind', profiles.HINTS)
def test_small_structural_simulator_smoke(kind, tmp_path, real_settings):
    p = profiles.Profile(weights={kind: 1}, disposition='mixed', motifs=2, depth=2,
                         time_scale_spread=1, tstop=100e-9, maxstep=1e-9)
    deck = profiles.generate(31, p).circuit.lower()
    evidence = Testbench(real_settings).run(deck, tmp_path, plan=ExecutionPlan(('ref',)))
    run = evidence.runs['ref']
    assert run.ok, run.diagnostics
    assert not output_problems(deck, run)
