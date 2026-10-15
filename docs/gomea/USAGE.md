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
                        GOMEA         (-a gomea)       ───────────┤
                        HC, then GOMEA (-a hc-gomea)   ───────────┘
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

> **Status: experimental.** It handles **discrete** (Boolean/logical) knobs natively.
> Numeric constants need the opt-in `--gomea-contin copy`; a real-valued model is Phase 2.
> By default, continuous (regression) problems are rejected with a clear error. The opt-in `--gomea-contin copy` runs them
> with the constants handled as in GP-GOMEA's paper: copied between individuals, never
> mutated (see [Continuous knobs](#continuous-knobs---gomea-contin-copy)). A real-valued
> GOMEA is Phase 2.
>
> **Phase 1 result, 1,140 pre-registered runs (uncapped demes): under equal evaluation
> budgets it did *not* beat the stock hill climber on any tested task.** Later phases
> changed this; see the Phase 1d–1f notes below.
>
> | Task | Stock HC | GOMEA with the default linkage tree | GOMEA with `--gomea-fos uni` |
> |---|---|---|---|
> | maj5 | 20/30 solved | 6/30 | 19/30 |
> | mux6 | 30/30 in about 1.7k evals | 30/30 in about 4.9k evals | 30/30 in about 3.0k evals |
> | pa4 | 2/20 | 0/20 | 0/20 |
>
> If you use it, **cap each deme**. In Phase 1d, `--gomea-max-evals-per-deme 500` alone
> lifted LT-B on maj5 from 6/30 to 20/30 solved, a tie with hill climbing
> ([`PHASE1D-FINDINGS.md`](PHASE1D-FINDINGS.md)). Uncapped, prefer `--gomea-fos uni`
> ([`PHASE1-FINDINGS.md`](PHASE1-FINDINGS.md)).
>
> **Best setup found (Phase 1e):**
> `-a gomea --gomea-fos lt-b --gomea-max-evals-per-deme 250 --gomea-genotype-memo 1`.
> On mux11 it solved all 20 seeds in a median of 4,633 evaluations against HC's 8,757.
> That is a pre-registered *better* verdict, the first for embedded GOMEA against hill climbing. It tied HC on maj7
> and pa5. It costs about 3–4× HC's CPU time on mux11 and maj7, so prefer it only when scorer calls are
> the expensive part, and give large problems more than 512 MB of memory
> ([`PHASE1E-FINDINGS.md`](PHASE1E-FINDINGS.md)).
>
> **On the GP-GOMEA paper's regression benchmarks (Phase 1f):**
> * Add `--gomea-contin copy` so constants become genes.
> * The same capped setup beat hill climbing on test error on 4 of 6 tested datasets and
>   lost on none, at 2.3–5.4× the CPU time (up to 23× on Tow, descriptive).
> * On very wide data (Dow, 57 features), 2 of 4 linkage-tree runs needed more than 5 GB.
> * The authors' stand-alone C++ GP-GOMEA is far stronger than classic C++ MOSES there
>   ([`PHASE1F-FINDINGS.md`](PHASE1F-FINDINGS.md)).
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

# The best setup found in Phase 1e (capped demes, lt-b, genotype memo)
moses -H mux -k 2 -m 20000 -r 1 -a gomea --gomea-fos lt-b \
      --gomea-max-evals-per-deme 250 --gomea-genotype-memo 1

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

Each option only has an effect with `-a gomea` or `-a hc-gomea`. They are validated whatever `-a` is set
to, so an invalid value is always rejected.

| Option | Default | Meaning |
|---|---|---|
| `--gomea-pop-size N` | `16` | Population size per deme. Each generation costs about `N × 2 × (number of knobs)` trial evaluations. The pilot found 16 best of {16, 32, 64} under a 20k-evaluation budget; larger populations run out of budget after very few generations. |
| `--gomea-fos {lt-b,lt-mi,rt,uni}` | `lt-b` | How knobs are grouped for mixing. `lt-b` is a linkage tree built from initialization-bias-corrected mutual information (Virgolin et al. 2021, Eq. 2). `lt-mi` uses ordinary mutual information. `rt` is a random hierarchical tree (a control). `uni` mixes one knob at a time (no linkage). |
| `--gomea-fi {0,1}` | `0` | Forced improvements: when an individual stagnates, mix it with the best-so-far solution. This is a labelled reconstruction; neither GP-GOMEA nor gpg ships it. |
| `--gomea-radii SPEC` | `1:0.4,2:0.3,4:0.2,max:0.1` | The mixture of neighbourhood distances used for the initial population, as `radius:weight` pairs. `max` means "all knobs". `uniform` (or a `uniform:<weight>` component) makes every distance from 1 to all knobs equally likely. |
| `--gomea-max-evals-per-deme N` | `0` (no cap) | The evaluation cap per deme. With no cap a deme may spend the whole remaining `-m` budget. If it stops, it is because it converged or stagnated. **Recommended: 250 with `--gomea-fos lt-b --gomea-genotype-memo 1`** (the Phase 1e winner); 500 was the Phase 1d fix. |
| `--gomea-stall-gens N` | `5` | Return to MOSES after this many generations without improvement. |
| `--gomea-max-gens N` | `100` | The hard limit on generations per deme. |
| `--gomea-skip-inactive {0,1}` | `0` | Skip evaluating a trial proven to decode to the same program. This is correct, but on MOSES representations it costs about 10 % of time and skips almost nothing, so it is off by default. |
| `--gomea-log PATH` | none | Append JSON-lines telemetry for every deme. See below. |
| `--gomea-init {sampler,uniform}` | `sampler` | The initial population: `sampler` uses MOSES's neighbourhood sampler around the exemplar; `uniform` draws each knob uniformly over its domain (Phase 1d). |
| `--gomea-accept {nonworse,strict}` | `nonworse` | GOM acceptance: `nonworse` keeps equal-score trials; `strict` keeps only strict improvements (Phase 1d). |
| `--gomea-genotype-memo {0,1}` | `0` | `1` answers a genotype already evaluated in this deme from a memo, uncharged (a GOMEA-only mechanism, Phase 1d). **Recommended: 1** with the Phase 1e setup. |
| `--gomea-stall-evals N` | `0` (off) | End a GOMEA deme after N charged evaluations without a strict improvement of its best (Phase 1e). |
| `--gomea-contin {reject,copy}` | `reject` | Contin (numeric constant) knobs. `reject` refuses a representation that has any (the default, unchanged behaviour). `copy` makes each contin knob one gene that GOM only copies between individuals, never mutates. See [Continuous knobs](#continuous-knobs---gomea-contin-copy). |
| `--gomea-contin-stats {bin,all,none}` | `bin` | With `--gomea-contin copy` and a linkage-tree FOS (`lt-b`, `lt-mi`): how constants enter the linkage statistics. `bin` (the paper's bin-const, γ = 100), `all` (all-const) or `none` (no-const). |
| `--gomea-pop-fraction F` | `0` (off) | Derive each deme's population from its budget: `N = max(2, floor(F × deme budget))`, where the deme budget is `min(per-deme cap, evaluations left for this deme)`. Overrides `--gomea-pop-size`. The sampler draws the N programs once, at the start of the deme. See [Budget-derived demes](#budget-derived-demes). |
| `--gomea-pop-max M` | `0` (no limit) | With `--gomea-pop-fraction`: clamp N to at most M (M ≥ 2). An error without `--gomea-pop-fraction`. |
| `--gomea-deme-share D` | `0` (off) | Set the per-deme cap to `max(1, floor(-m / D))`, so a run gets about D demes. It is the same cap `--gomea-max-evals-per-deme` sets, and also applies to `-a hc-capped`. Giving both is an error unless they give the same cap. |

The default population was 64 during development. It became 16 after the pilot
(`docs/gomea/pilot-a/`); the confirmatory runs passed 16 explicitly.

## Budget-derived demes

Two opt-in options size each deme from the evaluation budget instead of fixed numbers:

* `--gomea-deme-share D` sets the per-deme cap `C = max(1, floor(-m / D))`. It is exactly the
  cap `--gomea-max-evals-per-deme C` would set (so `-a hc-capped` uses it too); the two
  options may be given together only if they agree.
* `--gomea-pop-fraction F` sets each deme's population `N = max(2, floor(F × B))` with
  `B = min(C, evaluations MOSES hands this deme)` (`B` is just the evaluations left when
  there is no cap), optionally clamped by `--gomea-pop-max M`. `--gomea-pop-size` is then
  ignored. In `-a hc-gomea` the GOMEA phase uses the budget left after hill climbing.

The population is drawn once by the usual sampler and evolved for the deme's whole life; each
generation's linkage model is learned from the current N programs. For example
`-m 4000 --gomea-deme-share 4 --gomea-pop-fraction 0.04` gives `C = 1000` and `N = 40` for
every deme that has at least 1000 evaluations left. With both options off (the default),
nothing changes. The effective values are logged per deme in `invocation_start` (below).

## Continuous knobs (`--gomea-contin copy`)

A regression representation has contin knobs: numeric constants that MOSES encodes as a
path of up to `depth` Left/Right steps from the knob's mean. By default GOMEA refuses
them. `--gomea-contin copy` handles them the way GP-GOMEA handles constants
(Virgolin et al. 2021, Sec 6): each constant is a gene that gene-pool optimal mixing
**copies** between individuals, and nothing ever **mutates**. There is no Gaussian model
and no coefficient mutation; that is Phase 2.

* **One knob, one gene.** The gene's code packs all the knob's raw step fields (two bits
  each), so copying a gene copies the donor's whole constant. The gene counts as one knob
  for the initial-population radius, for the FOS and for telemetry (`domains` shows
  `4 ** depth` for it; `invocation_start` also reports `n_contin_genes`).
* **Initial population.** `--gomea-init sampler` (the default) keeps MOSES's own
  neighbourhood sampler, which may step a constant to a neighbouring value.
  `--gomea-init uniform` draws each constant uniformly over the knob's legal step paths,
  i.e. over the constants it can represent. (Keeping the exemplar's constant would leave
  the gene the same in every individual, and mixing could then never change it.)
* **Only constants already in the population can appear.** Mixing cannot invent a
  constant, just as it cannot invent a discrete value.
* **Linkage statistics** (`--gomea-contin-stats`, linkage-tree FOS only; `uni`/`rt` take
  no statistics):
  * `bin` (default; the paper uses it for all its later experiments): the first 100
    distinct constants met, individual by individual and gene by gene, keep their own
    symbol. Every other constant takes the symbol of the nearest of them. While the
    population holds at most 100 distinct constants this equals `all`.
  * `all`: every distinct constant is its own symbol.
  * `none`: constants carry no linkage information. Each constant gene stays out of the
    linkage tree and is mixed as its own singleton subset.

```bash
moses -i data.csv -u y -m 3000 -r 1 -a gomea --gomea-contin copy \
      --gomea-max-evals-per-deme 250 --gomea-genotype-memo 1
```

The hybrid (`-a hc-gomea`) takes the same options.

## The hybrid (`-a hc-gomea`)

`-a hc-gomea` (Phase 1e) runs MOSES's own hill climber first in every deme, with the full
budget, exactly as `-a hc` does. GOMEA continues the **same** deme only if hill climbing
stopped at a local optimum, not on the budget, the time limit or the target. GOMEA then
starts from the instances hill climbing already evaluated: they are not charged or scored
again, and its initial population is the best distinct ones among them. A deme where GOMEA
does not run is identical to `-a hc`.

It did **not** help in Phase 1e stage 1: the hybrids solved maj5 5/10 times, against HC's
9/10 ([`PHASE1E-FINDINGS.md`](PHASE1E-FINDINGS.md)).

## Telemetry (`--gomea-log`)

The file gets one JSON object per line, and each object has a `"type"`. For every deme
(one GOMEA invocation) it writes:

| type | Key fields |
|---|---|
| `invocation_start` | `schema_id` (a hash of this representation's knob layout), `dim` (number of knobs), `domains`, `max_evals`, `params`; with `--gomea-pop-fraction` or `--gomea-deme-share` also `deme_budget`, `deme_cap` (the per-deme cap, `null` for none) and `deme_pop_size` (this deme's N) |
| `init` | the initial population: `attempts`, `duplicates_rejected`, `coverage_replacements`, `feasible_short`, and `diagnostics` (per-knob entropy and observed values, `constant_fields_before/after_coverage`, `radius_histogram`) |
| `generation` | `generation`, `fos_size`, per-generation GOM `stats` (trials, evaluated, accepted_improving / accepted_neutral / rejected, noop_skips, fi_*), `nis` (the no-improvement stretch), `best_raw`/`best_pen`/`best_cpx`, a `ledger` snapshot, and diversity: `unique_genotypes`, `mean_entropy_bits`, `n_varying_fields`, `best_raw_count` |
| `end` | `termination_reason`, `generations`, `ledger` (requests = evaluations charged, invalid, duplicate_requests, …), `timings` (init / eval / linkage / variation seconds), `best` |
| `hybrid` | `-a hc-gomea` only, one per deme: `deme_id`, `max_evals`, `hc_evals`, `hc_stop` (`budget`, `time_limit`, `target` or `local_optimum`), `gomea_ran`, `gomea_evals` (charged), `gomea_termination_reason` |

A deme ends with one of these `termination_reason`s: `target_reached`, `converged`,
`stagnation`, `max_generations`, `global_budget`, `inner_budget`, `time_limit`,
`no_search_fields`, `no_valid_moves`, `interrupted`, or `stall_evals` (the
`--gomea-stall-evals` rule).

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

* **Discrete (Boolean) representations by default.** A representation with continuous
  knobs raises `UnsupportedRepresentation`, and the message suggests `-a hc` or
  `--gomea-contin copy`. With `--gomea-contin copy` constants are only recombined, never
  tuned: a constant not in the initial population can never appear in that deme.
* **Pure Python.** Evaluation costs about 6–12 ms per candidate on the small demo
  problems, so keep `-m` modest. On larger representations it costs more: in Phase 1e the
  median CPU per scorer call was 92 ms (HC) vs 525 ms (GOMEA) on mux11, 129 vs 466 ms on
  maj7 and 66 vs 69 ms on pa5. On Phase 1f's regression datasets GOMEA's CPU time was
  2–23× HC's ([`PHASE1E-FINDINGS.md`](PHASE1E-FINDINGS.md),
  [`PHASE1F-FINDINGS.md`](PHASE1F-FINDINGS.md)).
* **Memory.** GOMEA can need far more memory than HC. In Phase 1f it peaked at up to
  1.35 GB on the six tested datasets and 2.9 GB on Tow, and 2 of 4 Dow linkage-tree runs
  exceeded 5 GB; HC stayed under 0.1 GB. Not profiled.
* **Recursion limit.** pyMoses's recursive
  `neighborhood_sampling.count_neighborhood_size_from_index` exceeds Python's default
  recursion limit once a representation has more than about 1,000 knobs. This affects HC
  too. The harnesses work around it: `--recursion-limit` in `benchmarks/gomea/runner.py`,
  and a wrapper that raises the limit in `tools/gomea/moses_sr_runner.py`.
* **Mixing cannot invent knob values.** It can only recombine values present in the
  population. The initial-population coverage pass mitigates this.
* The demo problems (`-H pa|maj|mux|…`) run without a score cache in both optimizers.
  Table problems use `-s/--cache-size` as usual.

## Where things are

| Path | What |
|---|---|
| `pymoses/gomea/` | the implementation (schema, oracle, stats, fos, gom, engine, population, optimizer, hybrid, telemetry) |
| `tests/gomea/` | contract, parity, engine and integration tests (`pytest tests/gomea`) |
| `tools/gomea_golden/` | the g++ harness producing golden fixtures from verbatim GP-GOMEA code |
| `benchmarks/gomea/` | the experiment runner and analysis (`python -m benchmarks.gomea.runner --help`) |
| `tools/gomea/` | per-phase run drivers and report tools, GP-GOMEA build scripts (including the LT-MI patch), and the Phase 1f regression tooling (dataset splits, scoring, runners, run watchdog) |
| `configs/gomea/phase1/` | the pre-registered experiment manifests |
| `docs/gomea/` | audit, provenance, pre-registration, findings, runbook, Phase 2 handoff |
