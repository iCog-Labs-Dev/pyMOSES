# GOMEA inner optimizer for pyMoses — user guide

`-a gomea` swaps MOSES's inner optimizer (the hill climber, `-a hc`) for a **GP-GOMEA-style
population optimizer**. Everything else MOSES does is unchanged:

* exemplar selection;
* representation (knob) building;
* the metapopulation, merging and termination.

GOMEA only replaces the search *inside* each representation (each "deme").

```text
MOSES picks an exemplar ─► builds one representation (knobs) ─► [inner optimizer] ─► merge
                                                                   │
                        hill climbing (-a hc, default) ───────────┤
                        GOMEA         (-a gomea)       ───────────┘
```

Inside one deme, GOMEA works like this:

1. It samples a population of N distinct knob settings around the exemplar, using MOSES's
   own neighbourhood sampler at a mix of distances.
2. It learns which knobs interact. It does this by building a *linkage tree* from the
   population's statistics.
3. It improves every individual by **gene-pool optimal mixing** (GOM). It copies groups of
   linked knob values from other individuals, keeping a change only if the candidate does
   not get worse.

Every candidate it evaluates goes into the deme, exactly as with hill climbing, so MOSES
merges the best of them back into the metapopulation as usual.

> **Status: experimental, Phase 1 (discrete only).** It works for **Boolean** problems,
> i.e. representations made of discrete/logical knobs. Continuous (regression) problems
> are rejected with a clear error; that is Phase 2.
>
> **Measured result, 1,140 pre-registered runs: under equal evaluation budgets it did
> *not* beat the stock hill climber on any tested task.**
>
> | Task | Stock HC | GOMEA with the default linkage tree | GOMEA with `--gomea-fos uni` |
> |---|---|---|---|
> | maj5 | 20/30 solved | 6/30 | 19/30 |
> | mux6 | 30/30 in about 1.7k evals | 30/30 in about 4.9k evals | 30/30 in about 3.0k evals |
> | pa4 | 2/20 | 0/20 | 0/20 |
>
> If you use it, **cap each deme** with `--gomea-max-evals-per-deme 500`. In Phase 1d that
> alone lifted LT-B on maj5 from 6/30 to 20/30 solved, a tie with hill climbing
> ([`PHASE1D-FINDINGS.md`](PHASE1D-FINDINGS.md)). Uncapped, prefer `--gomea-fos uni`
> ([`PHASE1-FINDINGS.md`](PHASE1-FINDINGS.md)).
> A follow-up ([`PHASE1B-FINDINGS.md`](PHASE1B-FINDINGS.md)) ran the authors' own
> stand-alone GP-GOMEA on the same tasks. It also lost to MOSES hill climbing at equal
> evaluations. On larger problems ([`PHASE1C-FINDINGS.md`](PHASE1C-FINDINGS.md)) the
> stand-alone GP-GOMEA scaled better than classic MOSES on majority and worse on
> multiplexer and parity.

## Quick start

```bash
# 3-input parity with GOMEA, population 16, bias-corrected linkage tree (the default FOS)
moses -H pa -k 3 -m 3000 -r 1 -a gomea            # population 16, linkage-tree FOS
moses -H pa -k 3 -m 3000 -r 1 -a gomea --gomea-fos uni   # the best GOMEA variant in Phase 1

# The same problem, stock hill climber, for comparison
moses -H pa -k 3 -m 3000 -r 1

# A CSV dataset with a Boolean target
moses -i tests/fixtures/table_data/trainingBoolean.dat -u y -m 3000 -a gomea --gomea-pop-size 16

# Record what GOMEA did in every deme (JSON lines)
moses -H mux -k 2 -m 5000 -a gomea --gomea-pop-size 16 --gomea-log gomea-events.jsonl
```

Output looks exactly as it does with `-a hc`: `<score> <program>` lines, best first. All
classic flags (`-m`, `-r`, `-c`, `-x`, `-W`, …) behave as usual.

## Options

Each option only has an effect with `-a gomea`. They are validated whatever `-a` is set
to, so an invalid value is always rejected.

| Option | Default | Meaning |
|---|---|---|
| `--gomea-pop-size N` | `16` | Population size per deme. Each generation costs about `N × 2 × (number of knobs)` trial evaluations. The pilot found 16 best of {16, 32, 64} under a 20k-evaluation budget; larger populations run out of budget after very few generations. |
| `--gomea-fos {lt-b,lt-mi,rt,uni}` | `lt-b` | How knobs are grouped for mixing. `lt-b` is a linkage tree built from initialization-bias-corrected mutual information (Virgolin et al. 2021, Eq. 2). `lt-mi` uses ordinary mutual information. `rt` is a random hierarchical tree (a control). `uni` mixes one knob at a time (no linkage). |
| `--gomea-fi {0,1}` | `0` | Forced improvements: when an individual stagnates, mix it with the best-so-far solution. This is a labelled reconstruction; neither GP-GOMEA nor gpg ships it. |
| `--gomea-radii SPEC` | `1:0.4,2:0.3,4:0.2,max:0.1` | The mixture of neighbourhood distances used for the initial population, as `radius:weight` pairs. `max` means "all knobs". |
| `--gomea-max-evals-per-deme N` | `0` (no cap) | The evaluation cap per deme. With no cap a deme may spend the whole remaining `-m` budget. If it stops, it is because it converged or stagnated. **Recommended: 500**, the main fix found in Phase 1d. |
| `--gomea-stall-gens N` | `5` | Return to MOSES after N generations without a strict improvement of the population's best. |
| `--gomea-max-gens N` | `100` | The hard limit on generations per deme. |
| `--gomea-skip-inactive {0,1}` | `0` | Skip evaluating a trial proven to decode to the same program. This is correct, but on MOSES representations it costs about 10 % of time and skips almost nothing, so it is off by default. |
| `--gomea-log PATH` | none | Append JSON-lines telemetry for every deme. See below. |

The default population was 64 during development. It became 16 after the pilot
(`docs/gomea/pilot-a/`); the confirmatory runs passed 16 explicitly.

## Telemetry (`--gomea-log`)

The file gets one JSON object per line, and each object has a `"type"`. For every deme
(one GOMEA invocation) it writes:

| type | Key fields |
|---|---|
| `invocation_start` | `schema_id` (a hash of this representation's knob layout), `dim` (number of knobs), `domains`, `max_evals`, `params` |
| `init` | the initial population: `attempts`, `duplicates_rejected`, `coverage_replacements`, `feasible_short`, and `diagnostics` (per-knob entropy and observed values, `constant_fields_before/after_coverage`, `radius_histogram`) |
| `generation` | `generation`, `fos_size`, per-generation GOM `stats` (trials, evaluated, accepted_improving / accepted_neutral / rejected, noop_skips, fi_*), `nis` (the no-improvement stretch), `best_raw`/`best_pen`/`best_cpx`, a `ledger` snapshot, and diversity: `unique_genotypes`, `mean_entropy_bits`, `n_varying_fields`, `best_raw_count` |
| `end` | `termination_reason`, `generations`, `ledger` (requests = evaluations charged, invalid, duplicate_requests, …), `timings` (init / eval / linkage / variation seconds), `best` |

A deme ends with one of these `termination_reason`s: `target_reached`, `converged`,
`stagnation`, `max_generations`, `global_budget`, `inner_budget`, `time_limit`,
`no_search_fields`, `no_valid_moves`, or `interrupted`.

## Using it from Python

The simplest route is the CLI entry point:

```python
from pymoses.moses.main.moses_exec import moses_exec
moses_exec(["moses", "-H", "pa", "-k", "3", "-m", "3000", "-r", "1",
            "-a", "gomea", "--gomea-pop-size", "16"])
```

You can also run the optimizer on a single representation, bypassing the MOSES outer loop:

```python
from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymoses.comboreduct.reduct.logical_reduction import logical_reduction
from pymoses.comboreduct.type_checker.type_signature import gen_signature, type_node
from pymoses.moses.representation.representation import representation
from pymoses.moses.representation.instance_set import instance_set
from pymoses.moses.representation.instance_scorer import complexity_based_scorer
from pymoses.moses.scoring.behave_cscore import behave_cscore
from pymoses.moses.moses.optimization.optimization import optim_parameters
from pymoses.moses.main.demo_problems import even_parity, logical_bscore
from pymoses.gomea import gomea_optimization, gomea_parameters
from pymoses.util.rng import randGen

randGen().seed(1)
rule = logical_reduction(frozenset())(2)
rep = representation(rule, rule, str_to_combo_tree("and($1 $2)"),
                     gen_signature(type_node.boolean_type, 3))
scorer = complexity_based_scorer(behave_cscore(logical_bscore(even_parity, 3)), rep, True)
deme = instance_set(rep.fields())
opt = gomea_optimization(optim_parameters(), gomea_parameters(pop_size=16, fos="lt-b"))
opt(deme, scorer, 2000, 60)                 # deme, scorer, max_evals, max_time (s)
print(opt.last_result["termination_reason"], opt.last_result["best"], deme.n_evals)
```

The building blocks are exported from `pymoses.gomea`:

* `FrozenSchema`: the knobs of a representation as categorical fields;
* `EvaluationOracle`: budgeted evaluation into the deme;
* `LinkageModel`: the FOS learners;
* `GomeaEngine`, `EngineParams`: the GOM generation loop, independent of MOSES;
* `build_population`: sampler-based initialization;
* `gomea_optimization`, `gomea_parameters`.

The engine works on any tuple-of-categories problem through a small `Evaluator` protocol.
See `pymoses/gomea/engine.py` and `tests/gomea/diagnostic_objectives.py` for an OneMax /
deceptive-trap example.

## Reproducibility and guarantees

* **Deterministic given `-r`.** GOMEA draws three independent RNG streams from MOSES's
  generator at the start of every deme.
* **Every evaluated candidate is charged to `-m`** and appended to the deme, just as with
  hill climbing. `deme.n_evals == len(deme) ==` the number of scorer calls. GOMEA never
  exceeds its budget.
* **One representation = one frozen search space.** The population, linkage calibration
  and statistics are rebuilt for every deme and never shared between representations.
* **`-a hc` is byte-identical to classic MOSES as before.** No classic flag changed
  meaning. `--help` only gains the `--gomea-*` lines.

## Limitations

* **Discrete (Boolean) representations only.** A representation with continuous knobs
  raises `UnsupportedRepresentation`, and the message suggests `-a hc`.
* **Pure Python.** Evaluation costs about 6–12 ms per candidate on the demo problems, so
  keep `-m` modest.
* **Mixing cannot invent knob values.** It can only recombine values present in the
  population. The initial-population coverage pass mitigates this.
* The demo problems (`-H pa|maj|mux|…`) run without a score cache in both optimizers.
  Table problems use `-s/--cache-size` as usual.

## Where things are

| Path | What |
|---|---|
| `pymoses/gomea/` | the implementation (schema, oracle, stats, fos, gom, engine, population, optimizer, telemetry) |
| `tests/gomea/` | contract, parity, engine and integration tests (`pytest tests/gomea`) |
| `tools/gomea_golden/` | the g++ harness producing golden fixtures from verbatim GP-GOMEA code |
| `benchmarks/gomea/` | the experiment runner and analysis (`python -m benchmarks.gomea.runner --help`) |
| `configs/gomea/phase1/` | the pre-registered experiment manifests |
| `docs/gomea/` | audit, provenance, pre-registration, findings, runbook, Phase 2 handoff |
