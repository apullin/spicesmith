# Regression corpus

Minimized decks from SpiceSmith findings. `spicesmith regress` runs
every entry in every configuration on a candidate, judged by all oracles, and exits with
status 1 on a regression or 2 on inconclusive/empty coverage. Only resolved, evaluated
evidence without regressions is promotable (exit 0). Run it before promoting a candidate;
see the [usage reference](../docs/USAGE.md#legacy-acceptance-reporting-and-regression).

The shipped entry is a self-contained RC smoke fixture. Project-specific findings
belong in caller-maintained corpora, supplied with `--corpus`.

Each entry is a directory with `deck.sp` (the reduced reproducer) and `entry.json`:

- `check`: the prefix of the entry's finding, e.g. `exact: diagnostics`;
- `expect`: `pass` for a fixed bug (every check must pass) or `known` for an accepted
  weakness of an approximate change (its own finding may reproduce, nothing else may fail;
  if it stops reproducing, `regress` says `fixed` and the entry should be updated);
- `finding`, `origin`, `notes`: what was found, where, and what it means;
- `requires` (optional): a list of configuration names the entry needs, typically from
  `--extra-configurations` (set with `adopt --requires NAME`); skipped when a name is not
  defined, and a busy configuration is inconclusive, not a pass.

Add an entry from the output of `spicesmith reduce` with `spicesmith adopt`.
