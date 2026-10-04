import numpy as np
import pytest

from spicesmith.waveform import AccuracyMeasure, Crossings, Trace, Waveform

VDD = 1.2


def waveform(t, **columns) -> Waveform:
    names = list(columns)
    rows = [' time ' + ' '.join(names)]
    for i, ti in enumerate(t):
        rows.append(' '.join(f'{x:.15e}' for x in [ti] + [columns[n][i] for n in names]))
    return Waveform.parse(('\n'.join(rows) + '\n').encode())


def square(t, period, delay=0.0, rise=10e-12):
    """A trapezoidal 0..VDD square wave starting low."""
    phase = np.mod(t - delay, period)
    up = np.clip(phase / rise, 0, 1)
    down = np.clip((phase - period / 2) / rise, 0, 1)
    return np.where(t < delay, 0.0, VDD * (up - down))


def test_parse_real_and_complex_tables():
    table = Waveform.parse(b' frequency v(a) v(a) v(b)\n1e6 1.0 2.0 3.0\n1e7 4.0 5.0 6.0\n')
    assert table.scale_name == 'frequency' and table.names == ('v(a)', 'v(b)') and not table.is_transient
    assert table.column('v(a)').tolist() == [1 + 2j, 4 + 5j] and table.column('v(b)').real.tolist() == [3.0, 6.0]
    assert Waveform.parse(b'') is None and Waveform.parse(b' time v(a)\n1 2 3\n') is None


def test_crossings_interpolate_at_mid_rail_with_direction():
    t = np.linspace(0, 1e-9, 1001)
    crossings = Trace(t, square(t, 400e-12, 100e-12)).crossings(VDD / 2, 0.1 * VDD)
    assert crossings.directions.tolist() == [1, -1, 1, -1, 1]
    assert crossings.times[0] == pytest.approx(105e-12, abs=1e-15)
    assert crossings.times[1] == pytest.approx(305e-12, abs=1e-15)


def test_round_off_around_mid_rail_is_not_a_crossing():
    t = np.linspace(0, 1e-9, 101)
    assert len(Trace(t, VDD / 2 + 1e-15 * np.sin(np.arange(101))).crossings(VDD / 2, 0.1 * VDD)) == 0


def test_a_crossing_is_timed_between_leaving_one_band_edge_and_clearing_the_other():
    v = np.array([0.0, 0.7, 0.5, 0.65, 1.2, 1.2, 0.0, 0.0])  # wobbles inside the band, then rises
    crossings = Trace(np.arange(8) * 1.0, v).crossings(0.6, 0.12)
    assert crossings.directions.tolist() == [1, -1]
    left, entered = 0.48 / 0.7, 3 + 0.07 / 0.55  # crosses 0.48 going up, then clears 0.72
    assert crossings.times[0] == pytest.approx((left + entered) / 2) and crossings.times[1] == pytest.approx(5.5)


def test_a_slow_edge_wobbling_across_mid_rail_keeps_its_time():
    t = np.linspace(0, 1.0, 1001)
    ramp = np.clip((t - 0.2) / 0.6, 0, 1)  # 0..1 V, mid-rail at 0.5 s
    wobble = 0.004 * np.sin(2 * np.pi * t / 0.05) * np.exp(-((t - 0.5) / 0.03) ** 2)  # +-4 mV near mid-rail
    still, wobbly = Trace(t, ramp).crossings(0.5, 0.1), Trace(t, ramp + wobble).crossings(0.5, 0.1)
    assert abs(wobbly.times[0] - still.times[0]) < 1e-3


def test_settling_from_between_the_rails_is_not_a_crossing():
    t = np.linspace(0, 1e-9, 1001)
    trace = Trace(t, np.where(t < 1e-11, 0.78, 0.0) + square(t, 400e-12, 100e-12))  # uic start at 0.78 V
    narrow = trace.crossings(VDD / 2, 0.05 * VDD, 0.2 * VDD)
    wide = trace.crossings(VDD / 2, 0.2 * VDD, 0.2 * VDD)
    assert narrow.directions.tolist() == wide.directions.tolist() == [1, -1, 1, -1, 1]


def test_match_pairs_in_order_and_forgives_one_edge_leaving_the_window():
    reference = Crossings(np.array([1.0, 2.0, 3.0]), np.array([1, -1, 1]))
    shifted = reference.match(Crossings(reference.times + 0.1, reference.directions), 3.5, 0.0)
    assert shifted.agree and shifted.shift == pytest.approx(0.1)
    late = Crossings(np.array([1.1, 2.1]), np.array([1, -1]))  # the third edge, ~3.1, is past tend
    assert reference.match(late, 3.05, 0.0).agree
    assert not reference.match(late, 3.5, 0.0).agree  # not near the end: a missing edge
    assert not reference.match(Crossings(reference.times, -reference.directions), 3.5, 0.0).agree


def test_a_drifting_oscillator_may_end_crossings_behind():
    tight = Crossings(np.arange(1, 33) * 0.15, np.tile([1, -1], 16))  # 32 crossings by 4.9
    slow = Crossings(np.arange(1, 31) * 0.16, np.tile([1, -1], 15))  # 30, ending 0.3 behind
    match = tight.match(slow, 4.9, 0.0)
    assert match.agree and match.shift == pytest.approx(0.3)
    early = Crossings(np.arange(1, 29) * 0.15, np.tile([1, -1], 14))  # stops early: missing edges
    assert not tight.match(early, 4.9, 0.0).agree


def test_node_errors_classify_timing_and_analog_nodes():
    t = np.linspace(0, 1e-9, 2001)
    reference = waveform(t, d=square(t, 400e-12, 100e-12), a=0.3 + 0.01 * np.sin(2e10 * t))
    other = waveform(t + 2e-12, d=square(t, 400e-12, 100e-12), a=0.3 + 0.012 * np.sin(2e10 * t))
    errors = AccuracyMeasure(VDD, 1e-9, 2e-12).node_errors(reference, other)
    d, a = errors['d'], errors['a']
    assert d.timing and d.counts and d.shift == pytest.approx(2e-12, rel=1e-3) and d.crossings == 5
    assert not a.timing and a.dv == pytest.approx(0.002, rel=0.05)  # amplitude, plus the 2 ps shift


def test_a_reference_peak_near_the_band_edge_makes_the_count_fragile():
    t = np.linspace(0, 1e-9, 1001)
    bump = VDD / 2 - 0.3 + (0.3 + 0.1 * VDD) * np.exp(-((t - 5e-10) / 5e-11) ** 2)  # peaks at mid-rail + 10%
    errors = AccuracyMeasure(VDD, 1e-9, 2e-12).node_errors(waveform(t, b=bump), waveform(t, b=bump + 0.01))
    assert not errors['b'].timing


def test_an_extra_glitch_pairs_out_of_step_and_disagrees():
    reference = Crossings(np.array([1.0, 3.0]), np.array([1, -1]))
    glitchy = Crossings(np.array([1.0, 1.05, 3.0, 3.05]), np.array([1, -1, 1, -1]))
    assert not reference.match(glitchy, 3.1, 0.01).agree


def test_a_power_up_dip_does_not_decide_whether_a_later_crossing_counts():
    t = np.linspace(0, 2e-9, 2001)

    def trace(dip):  # uic power-up spike, a dip to `dip`, then a slow rise through mid-rail
        rise = np.clip((t - 0.2e-9) / 0.6e-9, 0, 1) * 0.5
        spike = np.where(t < 2e-11, 0.25, 0.0) - np.where((t > 2e-11) & (t < 4e-11), 0.344 - dip, 0.0)
        return Trace(t, 0.344 + rise + spike)

    for dip in (0.301, 0.299):  # either side of the old 20% settle threshold
        assert trace(dip).crossings(0.5, 0.1, 0.1).directions.tolist() == [1]
    error = AccuracyMeasure(1.0, 2e-9, 2e-12).node_error(trace(0.301), trace(0.299))
    assert error.counts


def test_a_power_up_glitch_may_go_unpaired_at_the_start_of_a_uic_transient():
    reference = Crossings(np.array([0.28, 0.85, 1.31]), np.array([1, -1, 1]))
    glitchy = Crossings(np.array([0.04, 0.28, 0.85, 1.31]), np.array([-1, 1, -1, 1]))
    assert not reference.match(glitchy, 2.0, 0.01).agree  # without a power-up window
    assert reference.match(glitchy, 2.0, 0.01, startup=0.05).agree
    assert not reference.match(glitchy, 2.0, 0.01, startup=0.03).agree  # the glitch is later than that
