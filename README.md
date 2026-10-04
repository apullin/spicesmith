# SpiceSmith

Csmith, for netlists: reproducible random-circuit workloads for SPICE simulators.

SpiceSmith is intended to exercise SPICE, physical-design (PD), and layout tools and
fixtures. Generate reproducible circuits, compare tool behavior, explore workload
coverage, and reduce interesting cases to small reproducers.

The built-in runner currently handles SPICE simulation; generated netlists and circuit
graphs are also available to downstream PD and layout workflows. Structural validity
is not a promise of convergence, useful circuit behavior, or correct analog results.

### Sections

[Install](#install) · [Generate circuits](#generate-circuits) ·
[Compare behavior](#compare-behavior) · [More workflows](#more-workflows) ·
[Development](#development) · [Full usage guide](docs/USAGE.md)

## Install

Python 3.11+ and NumPy; Python 3.14 is preferred for development.

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
python3 -m spicesmith --help
```

Generation needs no simulator and never launches one. The optional execution backend
uses ngspice on Linux, found on `PATH` or supplied with `--reference` and `--candidate`.
It is not installed by pip. `spicesmith` and `python3 -m spicesmith` are equivalent.

## Generate circuits

```sh
# Just generate a self-contained circuit.
python3 -m spicesmith gen 31 circuit.sp

# Small analog fixture; generation only.
python3 -m spicesmith gen 31 runs/example/deck.sp --profile heterogeneous-continuous \
  --profile-config examples/smoke-profile.json

# A connected mixed-signal workload near 23k transistors; generation only.
python3 -m spicesmith gen 42 runs/large/deck.sp --profile large-mixed --transistors 23000
```

Profiles span logic networks, biased gain stages, feedback, passive networks,
resonators, nonlinear recovery, and switching disturbances. Size, digital/analog/mixed
composition, repetition, parameter diversity, coupling, excitation correlation, and
time scales are independently configurable.

Each structured case includes a SPICE deck, a `.circuit.json` graph, and
`.generation.json` provenance. Transform saved graphs with `duplicate`, `diversify`,
`reconnect`, or `excitation`, retaining their ancestry.

Netlists contain standard analysis directives, built-in models, and no runner paths
or control blocks. Use them directly with your tools; the optional ngspice adapter
adds waveform collection only when you request execution. SPICE dialect compatibility
still depends on the receiving tool. `--profile legacy --scale N` selects the older
block-array generator, now also self-contained.

To run a generated file yourself: `ngspice -b -r circuit.raw circuit.sp`.

Large digital, analog, and mixed profiles are available. Macro-scale **generation** is
tested; large simulations still need resource and convergence checks. Observed nets are
bounded, and there is no byte-level simulator-output cap. Start small before long runs.
See [profiles and controls](docs/USAGE.md#structured-generation-and-large-circuits).

## Compare behavior

Replace the binary paths below with your reference and candidate builds:

```sh
python3 -m spicesmith observe runs/example/deck.sp --out runs/observed \
  --configurations ref,exact --reference /path/to/reference --candidate /path/to/candidate
python3 -m spicesmith evaluate runs/observed --policy exact
python3 -m spicesmith evaluate runs/observed --policy difference \
  --policy-options examples/difference-policy.json
python3 -m spicesmith replay-case runs/observed --out runs/replayed
```

Execution and interpretation are separate:

| Policy | Purpose |
|---|---|
| `observe` | Record outputs, outcomes, integrity and solver observations without a correctness verdict |
| `exact` | Check exact equivalence against the selected baseline |
| `difference` | Measure and select discrepancies without deciding which result is correct |
| `legacy-accuracy` | Apply explicit accuracy contracts using reference and tighter-tolerance runs |

Policies never silently add executions. Missing evidence is inconclusive; **unassessed
is not pass**, and exit 0 alone does not mean all simulations succeeded. Saved evidence
is sealed and can be reevaluated without rerunning the simulator. See
[policies and replay](docs/USAGE.md#observation-policies-and-replay).

No build-specific optimization flags or CPU reservations are imposed. Execution
defaults to one thread per run, without affinity; choose `--threads`/`--cpus` and
define extra binary/environment/lock configurations in JSON when needed.

## More workflows

| Task | Commands |
|---|---|
| Stratified generation, optional execution and coverage feedback | `explore` |
| Separate discovery from repeated performance measurement | `freeze`, `benchmark` |
| Reduce a discrepancy, outcome, structure or repeated slowdown | `minimize` |
| Preset acceptance batches and reports | `run`, `judge`, `report` |
| Finding-based reduction and flag attribution | `reduce`, `split` |
| Maintain and run a regression corpus | `adopt`, `regress` |

```sh
# Generate a population without running a simulator, then freeze its workloads.
python3 -m spicesmith explore examples/strata.json --out runs/discovery --seed 17
python3 -m spicesmith freeze runs/discovery --out runs/frozen
```

Benchmarks retain variability, failures and slowdowns; incomplete runs cannot become
speedups. Reproducers retain their execution plan and dependency hashes. Use fresh output
directories after code or binary changes, and reserve uncontended resources for timings.

The [usage guide](docs/USAGE.md) covers all commands, profiles, resource controls,
artifacts, Python interfaces, compatibility rules, and exit-code semantics.

## Development

```sh
python3 -m pip install -e '.[dev]'
python3 -m pytest
ruff check spicesmith tests
mypy spicesmith
```

The default suite uses unit tests and fake simulators. Real simulator tests are opt-in:
`SPICESMITH_NGSPICE=/path/to/ngspice python3 -m pytest -m simulator`.
