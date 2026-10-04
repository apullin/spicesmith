# Usage reference

[Back to the overview](../README.md).

This guide covers the implemented generation, execution, comparison and reduction
interfaces. Generating a circuit does not certify its function, convergence or accuracy.

## Choose a workflow

| Need | Commands | Interpretation |
|---|---|---|
| Preset acceptance and regression checks | `run`, `judge`, `report`, `reduce`, `split`, `regress`, `adopt` | Legacy exact/accuracy contracts, reference and tight runs |
| Controlled generation and investigation | `gen --profile`, `transform`, `observe`, `evaluate`, `replay-case`, `minimize` | Explicit execution plan; observation-only unless a policy is selected |
| Population coverage and performance | `explore`, `freeze`, `benchmark` | Discovery separated from frozen measurement; accuracy is an independent choice |

`gen SEED OUT.sp` defaults to the `heterogeneous-continuous` structured profile,
versioned as **structural-1**. `gen --profile legacy` and `run` use block generator
**7**, with self-contained models and a new seed/deck mapping.
`run` does not accept structural profiles: use `gen` then `observe`, or `explore`.

## Simulator setup

Install the package as described in the [README](../README.md#install). Generation
needs no simulator. The built-in runner currently targets ngspice on Linux, using
`taskset` for optional CPU affinity and `flock` for optional cooperative locks.
It is not a generic multi-simulator adapter or a sandbox for untrusted inputs.

Pass `--reference /path/to/reference` and `--candidate /path/to/candidate` when running
comparisons. Both roles default to `ngspice` on the caller's `PATH`; without distinct
binaries an exact comparison is only a self-check.
Use `--extra-configurations JSON` to add further configurations, e.g. another build
(see below). Binary and external-input hashes are recorded; do not replace dependencies
used by a running experiment. There is no fixed reference hash, model directory,
build-specific flag set or host reservation.

Keep a source checkout for the shipped `examples/` and `corpus/`, which are not wheel
package data. With a non-editable install, pass `regress --corpus /path/to/corpus`.
All shipped generated circuits and fixtures use built-in device models. Netlists
contain ordinary `.tran`/`.save` (and, for block profiles, `.op`/`.dc`/`.ac`) directives,
not `.control` blocks or runner output paths. The optional ngspice adapter adds
waveform collection at execution time. Generated `spicesmith-output` comments preserve
the per-analysis observation contract. Hand-written decks can use `.save` instead,
or an explicit ngspice `.control`/`wrdata __OUT__/NAME` contract. Other tool dialects
may need translation; generation is not a promise of universal SPICE compatibility.

## Quick start

The first three commands generate artifacts only:

```sh
python3 -m spicesmith gen 31 runs/gain/deck.sp --profile heterogeneous-continuous \
  --profile-config examples/smoke-profile.json
python3 -m spicesmith explore examples/strata.json --out runs/discovery --seed 17
python3 -m spicesmith freeze runs/discovery --out runs/frozen

# Replace these paths with your simulator builds.
python3 -m spicesmith observe runs/gain/deck.sp --out runs/observed --configurations ref,exact \
  --reference /path/to/reference --candidate /path/to/candidate
python3 -m spicesmith evaluate runs/observed --policy exact
python3 -m spicesmith evaluate runs/observed --policy difference \
  --policy-options examples/difference-policy.json
python3 -m spicesmith replay-case runs/observed --out runs/replayed
```

Use fresh experiment, freeze, benchmark and minimize destinations. `evaluate` reuses
saved evidence; `run --resume` and `explore --resume` continue their respective campaigns.
Generating to an existing filename overwrites it and its sidecars: use separate
directories for independent cases and local include files.

The quick-start [smoke-profile.json](../examples/smoke-profile.json) uses a small,
CPU-smoke-tested gain scaffold. Unrestricted profiles can abort or fail to converge;
that is retained evidence, not necessarily a generator defect. `observe` still records
such a run, while a comparison may return inconclusive (exit 2).

## Execution plans and resources

`--configurations` selects an ordered execution plan, not a policy. `ref` runs the
reference; `exact` runs the candidate without extra flags; `tight` uses tighter reference
tolerances. `approx` runs the candidate without extra flags under an accuracy claim.
Those four are the only built-ins; further configurations come from
`--extra-configurations`. Use `ref,exact` for a basic two-binary comparison.
Configuration names do not establish that a simulator feature ran. Caller environment
flags are passed unchanged; there are no analysis-dependent private flag rewrites.

Runs default to one thread (`--threads N`) and no CPU affinity. `run` defaults to two
jobs; with explicit `--cpus LIST`, it splits the allocation into disjoint sets of N CPUs,
drops leftovers and limits concurrency to the available sets. Other commands use the
requested CPU set as a whole. Custom `.control` thread settings remain the caller's
responsibility. Configurations within a case run sequentially. There are no host-specific
CPU reservations or scans for other projects' processes. Reserve uncontended resources
yourself for timing comparisons.

### Extra configurations

`--extra-configurations JSON` (all simulating commands) names a JSON list of
configurations to run besides the built-in ones:

```json
[{"name": "mine", "binary": "builds/x/ngspice", "claim": "approximate",
  "extends": "approx", "flags": {"MY_FLAG": "1"}, "locks": ["/tmp/device.lock"]}]
```

- `name`: a new configuration name (not a built-in one; a simple directory name).
- `binary`: `"reference"`, `"candidate"`, or a simulator path (relative to the JSON file).
- `claim`: `"exact"` (byte-identical to ref, like `exact`) or
  `"approximate"` (no less accurate than ref against tight, like `approx`).
- `extends` (optional): a built-in configuration, or one defined earlier in the file,
  whose flags this one starts from; `flags` add to or override them.
- `flags` (optional): environment variables (string values).
- `locks` (optional): cooperative lock files each run holds with `flock -n`, first
  listed taken first.

Unknown keys are rejected. `run` and `regress` run extra configurations on every
case/entry alongside the built-ins; `reduce` and `split` target them by check prefix
(e.g. `mine: crossing`); extension commands select them by name in `--configurations`.
Reports measure them against ref. The simulator environment is the fixed base set
(HOME, USER, TMPDIR, PATH, LC_ALL, thread variables) plus the configuration's flags.

Runs holding locks go one at a time within a SpiceSmith process. If another process
holds a lock the run's status is `busy`; `--lock-wait SECONDS` (default 0, fail fast)
retries for up to that long. A busy run is a note (`NAME: lock busy, not run`) in
batches, not a finding, and leaves corpus entries inconclusive. Do not remove lock
files or interfere with foreign jobs.

`--timeout` defaults to 300 seconds per simulator run, not per case or batch. Each
simulator owns a process group, killed together on timeout or interruption. The runner
constructs a controlled environment rather than inheriting arbitrary simulator flags;
run records retain actual commands and environments.

## Legacy acceptance, reporting and regression

`run` executes all configurations above, tight-reference fallback, parse-only
front-end comparisons and a second `ref` every tenth seed, plus any extra configurations.

```sh
python3 -m spicesmith run --out runs/batch --start 1 --cases 100 --jobs 2 --cpus 0-7 --keep
python3 -m spicesmith run --out runs/batch --resume --cases 100 --keep
python3 -m spicesmith judge runs/batch/case-000007
python3 -m spicesmith report runs/batch --json runs/report.json
python3 -m spicesmith reduce runs/batch/case-000007 'exact:' --timeout 60 --max-trials 100
python3 -m spicesmith split runs/batch/case-000007 'mine:' --base exact --extra-configurations extra.json
```

Reduction/splitting examples require an actual matching finding; the seed is illustrative.
`--cases 0` keeps generating until interrupted. `--resume` starts after the last finished
seed; existing summaries are skipped unless `--fresh` is requested. `--fresh` replaces
case results, so preserve evidence first. Ctrl-C/SIGTERM stops scheduling and lets
in-flight cases finish; harness exceptions become case `error.txt` files. `--no-front`
and `--repeat-every 0` omit their checks and reduce coverage. `--keep` retains clean
cases' raw outputs too.

Block generator 7 combines inverter chains, gated odd-stage rings, initialized
latches, mirrors, differential pairs, RC/RLC networks, controlled-source loops, diode
clamps, behavioral sources, nested buffers, local libraries and binned models. It varies
physical temperature, pulse/PWL/sine/EXP/SFFM sources, `.param`/`.func`/
`.if`, subcircuit multiplicity, local `.include`/`.lib`, solver options, initial conditions
and nodesets. Some decks include operating point, DC and AC in addition to the
transient. Noise analysis and string-parameter synthesis are not provided.

`--scale N` repeats selected blocks N times with the same block-level draws: a
repetition/compute-scaling knob, not independent diversity. Legacy windows are 2–20 ns;
every net has a 1–10 GΩ leak, and observation is capped at 200 nets spread across copies.
Use structural profiles for coupled, heterogeneous large circuits and independent scales.

### Legacy oracle contracts

Success requires exit 0, no analysis-abort diagnostics and all requested outputs.
ngspice can exit 0 after `Timestep too small`; partial waveforms are not successful runs.
Validation rejects ragged rows, duplicate/missing vectors, non-finite values, wrong
real/complex shape, invalid scales and missing requested interval coverage. DC and AC
require their requested grids; adaptive transient grids may differ. Missing coverage
cannot pass through interpolation or endpoint extrapolation.

- **Exact:** outcome, every output's bytes, relevant diagnostics and all reference-reported
  matrix/iteration counters must match. Self-check errors fail. Parse-only `-D ngdebug`
  compares `debug-out2.txt` and `debug-out3.txt`, permitting only intended unused-model
  removal/commenting and directory normalization. Reference repeats check outcome,
  outputs and diagnostics for determinism.
- **Approximate:** if `ref` succeeds, the candidate must succeed. Compare nodewise errors
  against `tight`, using ordinary `ref` error as the baseline. Stable hysteretic mid-rail
  crossings have timing allowance `max(2 ps, 2e-4*tstop, 5% of ref's shift)`. Other
  transient voltage errors may exceed reference error by `max(2% VDD, ref error)`;
  OP/DC use `max(0.2% VDD, ref error)`; AC uses complex relative error with a peak floor
  and allowance `max(1e-3, ref error)`. Crossing classification includes count stability,
  direction, drift and bounded startup/window-edge allowances; see
  [waveform.py](../spicesmith/waveform.py) and [oracles.py](../spicesmith/oracles.py).
- **Inconclusive/integrity:** failed reference/tight runs or unusable data cannot certify
  accuracy. Qualitative reference/tight crossing disagreement is inconclusive for those
  nodes; other usable nodes can still be checked. Build-specific checks may additionally
  require expanded-deck information. Recovered solver rejections remain visible as notes.

Tight levels 0–3 are `(reltol, abstol, vntol, trtol)`:
`(1e-6,1e-15,1e-8,1)`, `(1e-6,1e-14,1e-8,1)`, `(1e-6,1e-13,1e-8,1)`,
`(1e-5,1e-13,1e-7,1)`. The first completing level is recorded. Tight is a numerical
reference, **not** a universal definition of correct analog behavior.

`report` accepts one or more legacy batches: findings/clusters, per-check pass rates,
accuracy by class/block, completion rates and incremental ablation error budgets.
It flags mixed identities rather than presenting them as one compatible experiment.
Process timing includes startup, which can dominate small decks.

### Corpus and promotion

```sh
python3 -m spicesmith regress --corpus corpus --work runs/regress
python3 -m spicesmith adopt runs/batch/case-000007/reduced fixed-example \
  --expect pass --origin 'batch, seed, candidate' --notes 'why this is a regression test'
```

`adopt` consumes a legacy `reduce` bundle, not an arbitrary `minimize` bundle. Choose
`--expect known` only for a deliberately accepted weakness: only that entry's exact
normalized findings are exempted, not all findings sharing a prefix. A disappearing
known finding is reported as `fixed`. `adopt --requires NAME` (repeatable) marks an
entry as needing that configuration (typically one from `--extra-configurations`); it is
skipped when the configuration is not defined, and a busy configuration is inconclusive.

`regress` exits **0** for evaluated, resolved evidence without regressions, **1** for
a regression, **2** for inconclusive or zero evaluated coverage. Empty corpora, missing
outputs and broken binaries are not promotable. This is not general analog correctness
certification or a substitute for representative validation. Accepted findings are
specific to an entry; new findings still fail. See [corpus/README.md](../corpus/README.md).

## Structured generation and large circuits

```sh
python3 -m spicesmith gen 17 runs/feedback/deck.sp --profile coupled-feedback
python3 -m spicesmith gen 17 runs/mixed/deck.sp --profile switching-disturbance
python3 -m spicesmith gen 42 runs/large/deck.sp --profile large-mixed --transistors 23000
```

Generation writes `deck.sp`, `deck.circuit.json` (typed graph) and
`deck.generation.json` (normalized profile, seed/version, actual structure, hash,
attempt history and transformation ancestry). Exhausted generation writes rejection
metadata and exits unsuccessfully; no simulator/accuracy oracle decides admission.
No `--scale` with structural profiles: size and repetition are independent.

### Named profiles

| Profile | Requested population |
|---|---|
| `heterogeneous-continuous` | Diverse analog hints with low reuse |
| `coupled-feedback` | Coupled gain, bias and feedback/compensation scaffolds |
| `irregular-passive` | RC/RLC and primitive graphs with broad component spread |
| `mixed-time-scales` | Mixed families with broader characteristic time scales |
| `repetitive-switching`, `asynchronous-repeated` | Reused logic shapes/parameters with synchronized versus independent sources |
| `repeated-analog` | Repeated gain scaffolds with common source draws |
| `nonlinear-small`, `nonlinear-large`, `nonlinear-recovery` | Small, large or pulsed recovery excitation of nonlinear structures |
| `switching-disturbance` | Switching loads coupled to analog paths |
| `large-digital`, `large-analog`, `large-mixed` | Approximately 23k transistors, independently variable reuse/coupling |

Hint `weights` select `bias` (mirror/bias trees), `gain` (differential and loaded gain
stages), `feedback` (labelled controlled-source scaffolds), `rc` (irregular passive
networks), `rlc` (coupled resonators), `nonlinear` (clamping/recovery), `primitive`
(unhinted graph scaffolds), `switching` (analog paths disturbed by switching), and
`logic` (inverter/NAND networks). They construct finite signal links, shared bias,
loads, compensation and regional supply/return impedances, not just adjacent blocks.
Controlled-source feedback is not mislabelled as a transistor-level op amp.

### Independent controls

`--profile-config FILE.json` overrides the named recipe. CLI `--transistors N` and
`--disposition digital|analog|mixed` override that JSON. CLI disposition without explicit
JSON weights starts with equal family weights, then filters them: digital admits `logic`,
analog the seven non-switching families, mixed all nine. Explicit weights still apply
after filtering. All fields below are serialized; the schema is
[Profile](../spicesmith/profiles.py). Values use SI units; `name` and `version: 1` identify
the recipe/schema.

| Axis | Controls and base defaults |
|---|---|
| Composition / size | `weights`, `disposition`, `motifs` (6), `target_transistors` (0 means use motif count), `depth` (3) |
| Repetition / variation | `topology_reuse` (0.1), `parameter_reuse` (0), `diversity` (1, log10 spread around nominal values) |
| Connectivity | `coupling` (0.8), `interconnect` (`chain`, `tree`, `mesh`; default mesh), `fanout` (2), `region_size` (32) |
| Excitation | `source_correlation` (0), `vdd` (1.2 V), `common_mode` (0.6 V), `amplitude` (0.05 V), `excursion` (`small`, `large`, `recovery`) |
| Analog structure | `bias_current` (20 µA), `loading` (10 kΩ), `feedback` (0.5), `feedback_polarity` (-1 or +1), `compensation` (1 pF), `nominal_q` (3) |
| Time scales | `time_scale_spread` (2, log10 spread), `tstop` (1 µs), `maxstep` (2 ns) |
| Supply / conditioning | `supply_impedance` (0.5 Ω), `return_impedance` (0.1 Ω), `conditioning` (true), `shunt` (1 TΩ), `model_family` (built-in models by default) |
| Graph budgets | `max_devices` (200,000), `max_nets` (150,000), `max_motifs` (12,000), `max_observed` (48) |
| Work budgets | `max_requested_points` (100,000, bounds `tstop/maxstep`), `max_attempts` (4) |

Recipes override these base defaults. Reuse, correlation, coupling and feedback are in
`[0,1]`; required physical magnitudes must be finite/positive and counts/budgets positive
integers. Hard caps also bound depth (32), fanout (16), attempts (100), devices/nets
(2 million each) and motifs (100,000). Unknown controls, invalid ranges and impossible
dispositions/targets are rejected. Purely passive weights cannot satisfy a positive
transistor target. Target generation overshoots by at most one motif and still obeys
resource budgets.

Topology, parameter and source choices use separate seeded streams. Changing parameter
reuse or source correlation does not redraw connectivity. Requested reuse/correlation
is not measured waveform correlation: metadata records actual shapes/parameter sets,
and observations record limited signal correlations. Physical temperature is distinct
from sampling diversity; the structured API has no sampling-temperature or
physical-temperature knob. `model_family` currently admits `bsim4` only; the graph's
`models` mapping can be replaced in Python with caller-provided SPICE model definitions.
No external model library is selected implicitly.

Large circuits default to **48 observed nets**, not every simulated net; metadata reports
the unobserved fraction. Seed 42 at default large digital/analog/mixed profiles emits
23,004 / 23,001 / 23,003 transistors (62,612 / 137,503 / 94,246 total devices, including
shunts). These are synthetic connected workloads, not a functional design or
extracted-parasitic model. Macro-scale generation is tested; large
simulation convergence, memory use and timing distributions are not yet validated.
Smaller adaptive simulator steps can exceed the requested sample count; there is **no
byte-level simulator-output cap**. Start small with explicit timeouts before unattended
large runs.

Saved examples: [feedback.sp](../examples/feedback.sp), [switching.sp](../examples/switching.sp),
with profiles/structure summaries in adjacent `.generation.json` files. The IR emits
flat numeric R/C/L/V/I/M/D/E primitives with motif/region ownership and transient analysis.
Arbitrary SPICE expressions and parameterized subcircuit synthesis are not
implemented by this API; legacy `Deck` retains its subcircuit/DC/AC capabilities.

```sh
python3 -m spicesmith transform runs/feedback/deck.circuit.json diversify \
  --seed 23 --out runs/diversified/deck.sp
```

Operations are `duplicate`, `diversify`, `reconnect`, `excitation`, recording the parent
hash. Duplication preserves the original graph and adds a linked copy; others vary their
named axis. Transform the graph, not a hand-edited deck with stale metadata. Freezing
verifies graph/deck correspondence.

## Observation, policies and replay

`observe DECK_OR_CASE --out DIR` defaults to configurations `ref,exact`, policy `observe`,
one sample each, no front-end comparison and no reference repeat. Directory input means
its `deck.sp`; adjacent `.generation.json` metadata is retained. Candidate-only
observation uses `--configurations exact`; no reference/tight run is silently inserted.
`--front` requires `ref,exact`; `--repeat-reference` requires `ref`. `--repetitions N`
requests N timing samples of **each** configuration, independent of the special repeat.

| Policy | Required evidence | Result |
|---|---|---|
| `observe` (default) | Whatever was explicitly run | `unassessed`; records integrity problems/outcomes, never accuracy pass |
| `exact` | Baseline plus a comparison configuration | Exact contract on every selected configuration/sample, plus requested front/repeat checks |
| `difference` | Baseline plus a comparison configuration | Metrics/selection; `unassessed` or `inconclusive`, never accuracy pass |
| `legacy-accuracy` | `ref,tight` plus configurations/checks to assess | Original oracles, including declared supply/horizon requirements |

`--baseline NAME` defaults to `ref` for exact/difference. Exact policy applies even to
a configuration named `approx` if selected. Legacy accuracy still uses `ref`/`tight` and
configuration claims; `--baseline` does not redefine it. For example, request
`--configurations ref,approx,tight --policy legacy-accuracy`; add `exact` and `--front`
when expanded-deck/front-end evidence is needed. Choosing a policy never adds
its prerequisite executions.

Difference compares primary runs on baseline samples in the common covered interval,
linearly interpolating without extrapolation. It records maximum absolute, RMS and
baseline-peak-relative errors, outcomes, diagnostics and solver counters. All timing
samples are integrity-checked, but difference metrics use only the primary run; exact
also checks repeated numerical results. `--policy-options` accepts JSON thresholds:

```json
{"absolute": 0.001, "relative": 0.01, "floor": 1e-12}
```

Both absolute and relative thresholds must be **exceeded** to select a vector. Defaults
are zero thresholds and normalization floor `1e-12`. Absolute values use the vector's
units; volts, currents and other outputs have no universal shared tolerance. This is
triage, not analog correctness. See [difference-policy.json](../examples/difference-policy.json).

`evaluate CASE --policy ...` verifies the seal and writes an assessment without rerunning
the simulator. Assessments are keyed by serialized/versioned policy: different thresholds get
another file; the same key is updated. It cannot recover outputs pruned by legacy batches;
use `observe` for sealed, reevaluable evidence.

`replay-case CASE --out NEW_DIR` uses recorded plan, configurations, simulator paths,
CPUs and limits without the discovery scheduler. It verifies evidence and input/code
identities and refuses changed dependencies. To retest a new candidate, `observe` the
saved deck into a fresh directory with explicit new settings. Replay's policy defaults
to `observe`; pass `--policy` again for a comparison verdict.

### Existing decks and Python consumers

The file interface is `Deck`, not an arbitrary SPICE frontend. Use a `.control` block
with analyses and `wrdata __OUT__/waveform.txt v(node) ...`, plus `set wr_singlescale`,
`set wr_vecnames`, `set numdgt=15`; `rusage all` supplies solver counters. The harness
binds `__OUT__` to each run directory. Local `.include`/`.lib` files are staged; external
inputs remain dependencies. Declare `.param vsupply=...` for the legacy voltage
oracle. Generated decks demonstrate the format; other output conventions do not supply
comparable waveform data automatically.

The same interfaces are importable. This example performs **generation only**:

```python
from pathlib import Path
from spicesmith.profiles import generate, profile

recipe = profile("large-mixed", {"target_transistors": 1000, "parameter_reuse": 0.0})
generated = generate(42, recipe)
generated.circuit.lower().save(Path("runs/python-generated"))
metadata = generated.metadata()  # retain this and circuit.to_json() for provenance
```

[ExecutionPlan](../spicesmith/plans.py), [Testbench/Settings](../spicesmith/harness.py),
[experiment.run/evaluate/replay](../spicesmith/experiment.py),
[Policy/assess](../spicesmith/policies.py) and [Predicate](../spicesmith/predicates.py) expose
execution/interpretation separately. API callers must choose CPUs/settings and apply
the CLI's host/reference checks themselves; `Testbench` alone is not the full CLI safety
wrapper. `assess(evidence, Policy("my-check", version=1), custom=evaluator)` accepts an
explicit callable returning an oracle `Verdict`. Saved files never implicitly import
custom code. These are initial versioned interfaces, not a promise of stable internals.

## Coverage, frozen populations and benchmarks

`explore SPEC.json --out DIR` is generation-only unless `--execute` is supplied. JSON
contains `strata`: each has a unique `name`, profile name/object, positive `quota`, and
optional `target` (default `generated`):

```json
{
  "strata": [
    {"name": "coupled", "profile": "coupled-feedback", "quota": 10},
    {"name": "independent", "profile": {"name": "repeated-analog", "source_correlation": 0}, "quota": 10}
  ]
}
```

[strata.json](../examples/strata.json) spans initial analog/switching recipes. Targets
`completed`, `active`, `quiescent`, `unclassified` require execution. `--budget N` caps
issued cases including rejections (default sum of quotas). `--feedback` uses extra budget
for deterministic deficit scheduling after initial quotas; quotas do not guarantee
success. `--resume` requires unchanged strata, seed, budget, feedback, code, execution
and policy identity. The scheduler is sequential, not distributed. Failures, rejections,
interruptions and unmet targets stay in `campaign.json`/`coverage.json` denominators.

Activity is coarse: completed observed voltage traces with span above **1 µV** are
`active`, otherwise `quiescent`; unavailable classification is `unclassified` and
incomplete execution is `incomplete`. Correlation is measured on at most eight adjacent
saved-vector pairs. This is not an oscillator, settling, transistor-region or
condition-number classifier and says nothing about unobserved nodes.

```sh
python3 -m spicesmith explore examples/strata.json --out runs/executed-discovery \
  --seed 17 --execute --configurations ref,exact --feedback --budget 30
python3 -m spicesmith freeze runs/executed-discovery --out runs/measurement-population
python3 -m spicesmith benchmark runs/measurement-population --out runs/benchmark \
  --configurations ref,exact,approx --repetitions 3 --policy difference
```

Freeze copies **all generated decks**, even failed simulations, verifies evidence and
graph/deck hashes, and lists attempts excluded for having no deck. A frozen
discovery-selected population remains selection-biased; freezing does not make it
representative. Freeze the population before comparing speed rather than adapting
workloads during measurement.

Benchmarking defaults to three samples per configuration; ratios need at least three
matched, complete, positive-duration samples. `speedup` is baseline median wall time
divided by candidate median wall time (>1 is faster). Startup is included; configuration
order alternates between repetitions. Reports retain samples, median/range/standard
deviation, slowdowns and unresolved cases, grouped by composition, size, repetition,
coupling, requested/observed correlation and regime. Accuracy remains independently
assessed or explicitly unassessed. Timeouts, aborts, truncation and missing timings
cannot become speedups.

`benchmark.json` distinguishes all cases attempted from completed workloads and retains
partial progress. New sessions need fresh directories; no timing-session resume/combination.
Timing `tight` requires explicit `--tight-level 0|1|2|3` to avoid timing fallback searches.
Recommended experiments vary size, disposition, reuse, coupling and correlation
independently: small reserved CPU runs first, then bounded macro-scale populations after
resource use is known. The tools themselves imply no macro-scale performance claim.

## Reduction and standalone reproduction

Legacy `reduce CASE CHECK` selects a finding prefix and runs only required configurations,
preserving reference diagnostics and tight-reference validity where needed. `split CASE
CHECK --base exact` tries each added flag alone and omitted from the full set, recording
sufficient/necessary flags in `split.json`.

`minimize CASE_OR_DECK PREDICATE.json --out DIR` uses an explicit predicate and plan:

| Field | Meaning |
|---|---|
| `kind` | `outcome`, `exact`, `difference`, `structure`, `slowdown` |
| `configuration`, `baseline` | Configuration to retain and comparison baseline (defaults `exact`, `ref`) |
| `outcome` | Status such as `timeout`, `aborted`, `no-output`, or integer exit code |
| `threshold`, `relative` | Difference absolute/relative thresholds; slowdown `threshold` is candidate/baseline time ratio |
| `finding` | Optional finding/selection prefix |
| `minimum_devices`, `minimum_transistors`, `connections` | Structural guards; connections are pairs of direct primitive terminal nets, not arbitrary reachability |
| `regime` | Optional `active`, `quiescent`, `unclassified` observation guard |
| `version` | Predicate schema version, currently 1 |

```sh
# The selected discrepancy must first reproduce with this same plan.
python3 -m spicesmith minimize runs/observed examples/discrepancy.json \
  --out runs/minimized --configurations ref,approx --max-trials 100 --timeout 60
sh runs/minimized/repro.sh
```

Minimization reruns the deck with the supplied plan; the quick-start observation did not
execute `approx`. Inspect the initial result rather than assuming any seed exhibits a
discrepancy. Use predicate JSON, not `--policy`, to select minimization. A `structure`
predicate also requires execution completion. Guards count emitted top-level primitives
(including recognized transistor-instance forms), not arbitrary subcircuit internals. Slowdown needs
`--repetitions 3` or more; a single fast failure cannot preserve a slowdown claim.

Reduction first shortens transient duration, then removes top-level items, subcircuits,
their contents and observed vectors, then simplifies parameters/component values. Every
search predicate call, even a structurally rejected candidate, counts against
`--max-trials`; initial/final verification are additional. Defaults: 2,000 search calls
for legacy reduction, 200 for `minimize`. Shorter per-run timeouts help with intermediate
simulator convergence failures. Failed final verification retains evidence but issues **no**
purportedly verified bundle.

Bundles contain reduced decks/local files, a frozen runner, full plan, selected
finding/predicate, hashes and `repro.sh`. They preserve front-end runs, repeats, tight
level, process deadlines and lock/retry semantics; their plan records extra
configurations' definitions (binary path, locks) and their settings record `lock_wait`.
Python/NumPy, recorded binaries and external inputs remain runtime dependencies. Replay
checks identities and exits
**0** for reproduction, **1** for non-reproduction, **2** for unavailable/changed evidence.
Run scripts from trusted bundles only.

## Artifacts, compatibility and automation

| Workflow | Principal artifacts |
|---|---|
| Legacy batch | `batch.json` identity/sessions; `tally.json`; `case-000001/deck.sp`, `summary.json`, configuration directories, optional `error.txt` |
| Simulator run | Bound `deck.sp`, `log.txt`, `run.json` (command/environment/outcome/wall time), requested outputs |
| Special executions | `front-ref`, `front-exact`, `ref-repeat`, tight fallbacks, `NAME-sample-N`, `plan.json` |
| Structured generation | `.sp`, `.circuit.json`, `.generation.json` |
| Observation / replay | `experiment.json`, `plan.json`, `observations.json`, sealed `evidence.json`, `assessments/POLICY_HASH.json`, raw runs |
| Discovery / freeze | `campaign.json`, `coverage.json`, case `case.json`; frozen `population.json` plus decks/graphs/metadata |
| Benchmark | `benchmark.json` and each case's sealed experiment |
| Reduction | `reduced.sp`, optional tightened deck, `reduce.json` or `minimize.json`, frozen runtime/manifest, `repro.sh` |

Legacy cases with findings, notes or inconclusive reasons retain raw evidence. Only
clean cases drop outputs/dumps, unless `--keep` is used. Observation keeps and seals
raw evidence. Do not edit/prune sealed files or hand-edit manifests to make replay pass;
use a new experiment. External absolute dependencies are identified, not automatically
made portable.

Batch compatibility includes binary/input hashes, code hash, generator version, scale,
configuration definitions/ancestry (including extra configurations), extra-configuration
binary hashes (`extra_binaries`), front/repeat settings, timeout, `lock_wait` and threads.
Old metadata lacking required fields is incompatible. `--force` allows a mixed legacy
batch with separately recorded identities; it does not make the evidence uniform.
Prefer new output directories after upgrades or simulator changes. Discovery and replay
also verify identities; a package version string alone is insufficient.

Downstream adopters should start new processes and campaign directories. Running Python
processes do not become the new implementation merely because a checkout changed. Keep
ongoing old-code campaigns in their original worktree; do not resume under changed code
and call the results one experiment. Content hashes, not version or revision labels
alone, identify executable evidence.

Shell exit 0 is not scientific success:

- `observe`, `evaluate`, `replay-case`: 1 for policy failure, 2 for inconclusive/error;
  0 includes **unassessed**, even observation-only runs with recorded problems and valid
  difference selections. Inspect assessment `state`, `reasons`, `selected`, completeness.
- `regress`: 0 resolved/promotable corpus, 1 regression, 2 inconclusive/empty/error.
- `run`, `judge`, `report`, `explore`, `benchmark`: normal completion is not an all-pass
  gate. Inspect summaries, coverage, policy states and completed workloads.
- `repro.sh`: 0 reproduced, 1 did not reproduce, 2 unavailable/changed evidence.

## Validation and limits

Run the development checks in the [README](../README.md#development). Unit/fake-simulator
tests cover deterministic generation, output integrity, policies, coverage accounting,
replay, reduction and benchmark semantics. Real simulator tests are opt-in and need
configured binaries and fixtures; a skipped test is not validation of that path.

Macro-scale generation is tested, but large-circuit convergence, resource use and
performance need separate measurements. The current implementation does not provide
universal analog correctness certification, learned guidance, a distributed scheduler,
rich operating-regime/condition-number estimation, structured hierarchical synthesis,
a byte-level simulator-output cap, or built-in physical-design/layout execution.

Start with small kept cases, vary structure and scale independently, then freeze a
population for repeatable measurements on reserved resources. Retain failures and
slowdowns, reduce interesting cases and adopt explicit regression expectations.
