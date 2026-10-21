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
| `--gomea-accept-complexity-ratio R` | `0` (off) | Inside a GOMEA deme, rank candidates by `raw score − complexity / R` (the form of `-z`, so one complexity unit costs `1/R` score units) instead of MOSES's penalized score. MOSES's own scoring, merge and exemplar selection are unchanged. See [Deme-local complexity penalty](#deme-local-complexity-penalty). |
| `--gomea-deme-store {all,accepted}` | `all` | Which evaluated GOMEA trials the deme hands back to MOSES: `all` of them (the default), or `accepted`: only the initial population and the trials GOM or forced improvement accepted. Every evaluation is still charged to `-m` and counted. |
| `--simplify-exemplar-numeric TOL` | `0` (off) | Port-only, regression only, any optimizer: before knobs are built, replace exemplar subtrees whose training-output range is below `TOL × sd(target)` by their mean, keeping a replacement only if the training score is not worse. See [Numerical simplification of exemplars](#numerical-simplification-of-exemplars---simplify-exemplar-numeric). |
| `--gomea-fos-topk K` | `0` (off) | Each generation, GOM (and forced improvement) visits only K FOS subsets instead of all of them (about 2 per knob), so a generation costs about `N × K` trials instead of `N × 2 × knobs`. See [Top-K FOS subsets](#top-k-fos-subsets---gomea-fos-topk). |
| `--gomea-fos-topk-mode {linkage,random}` | `linkage` | With `--gomea-fos-topk K`: `linkage` keeps the K linkage-tree merges with the strongest linkage (needs `--gomea-fos lt-b` or `lt-mi`); `random` draws K subsets uniformly each generation (the control). |
| `--gomea-long-deme-cap C` | `0` (off) | With `-a gomea` and a per-deme cap: a deme chosen by `--gomea-long-deme-rule` runs with cap C instead of `--gomea-max-evals-per-deme`, still bounded by the evaluations left; N is unchanged. See [Long demes](#long-demes-for-promising-exemplars---gomea-long-deme-cap). |
| `--gomea-long-deme-rule {top,random}` | `top` | `top`: a deme is long when its exemplar has the metapopulation's highest penalized score when chosen (ties count) and has not had a long deme before. `random`: each deme is long with probability `--gomea-long-deme-prob p`, drawn from the run's RNG. |
| `--gomea-long-deme-prob p` | `0` | With `--gomea-long-deme-rule random`. p = 0 (and p = 1) draws nothing. |
| `--reserve-empty-every K` | `0` (off) | Port-only, any optimizer: every K-th expansion picks its exemplar only among the programs descending from an initial exemplar that is the empty program (`+`/`and`), falling back to the normal choice when none is selectable. See [Reserved expansions for the empty program](#reserved-expansions-for-the-empty-program---reserve-empty-every). |
| `--linear-scaling {0,1}` | `0` (off) | Port-only, regression only (squared error), any optimizer: score each program `f` on `a + b·f`, the least-squares fit of `f` to the training target. See [Linear scaling](#linear-scaling---linear-scaling). |

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

## Deme-local complexity penalty

MOSES's complexity penalty (`-z`, default 3.5, i.e. 1/3.5 ≈ 0.29 score units per complexity
unit) hardly binds on a problem like mux11 (raw score = minus the wrong rows out of 2,048), so a
capped GOMEA deme, which keeps the first trial that is not worse, lets the exemplar grow quickly
(`docs/gomea/exemplar-growth-2026-10-15/REPORT.md`). `--gomea-accept-complexity-ratio R` adds a
stronger penalty that applies only inside GOMEA:

* Two candidates are compared by `raw − complexity / R`; on equal values the lower complexity
  wins, and NaN is worse than anything except NaN, exactly as MOSES's own comparator. MOSES's
  stored complexity and uniformity penalties are left out. `R = 0.125` means one complexity unit
  must buy 8 raw-score units.
* It is used by every comparison GOMEA makes inside a deme: GOM acceptance, forced
  improvements, the elite, the engine's best and its no-improvement count
  (`--gomea-stall-gens`), the `--gomea-stall-evals` rule, and the `-a hc-gomea` choice of
  seeds from hill climbing's deme.
* It is not used by anything MOSES sees: the scores stored in the deme, `n_best_evals`, the
  `end` event's `best`, and MOSES's merge, exemplar selection and output all keep `-z`.

With the default `--gomea-deme-store all` the deme still holds every evaluated trial, including
the rejected ones, and MOSES ranks them under `-z`; rejected trials that GOMEA judged too large
can then come back as candidates. `--gomea-deme-store accepted` keeps only the initial
population and the accepted trials (the elite and the final population are always among them).
Under `accepted`, a trial that reaches the target score but is rejected by the deme-local
penalty is not stored either, so MOSES does not see it.

## Knob probing baseline (`--knob-probe-baseline`)

`--knob-probe-baseline {classic,reduced}` (default `reduced`) is a knob-building option, so it
applies to every `-a` (hill climbing, `hc-capped`, GOMEA and the hybrid alike). The default
`reduced` is a **divergence from C++ MOSES**, adopted as the default on 2026-10-21 (backlog
item 15; it ran in every arm of Phases 1u–1ab without a loss attributable to it). With
`--knob-probe-baseline classic` knob building is classic MOSES's, unchanged. The ported
`build_knobs` and `representation` constructors still default to `classic` when called
directly from Python.

When it builds a logical knob, MOSES turns the knob to each setting, reduces the whole decorated
exemplar and disallows the setting if the reduction lowers its complexity
(`build_knobs.cc:311`, `disc_probe`). That assumes the exemplar with every knob at 0 (the
baseline) is already a fixpoint of the knob-building reducer. Effort-2 logical reduction is not
confluent, so after knob building has wrapped a literal in a single-child junctor the baseline
itself can become reducible; from then on every setting of every later knob looks reducible and
almost all later knobs are dropped (BACKLOG section 4 item 1: e.g. 135 knobs on a 48-node mux11
exemplar, where a comparable exemplar gets about 830).

`reduced` disallows a setting only when it adds reducibility, i.e. when the setting's tree loses
more complexity under reduction than the baseline does:
`c(T_setting) − c(R(T_setting)) > c(T_0) − c(R(T_0))`. When the baseline is a fixpoint this is
exactly classic's test, so the two options build the same representation on every exemplar
whose baseline is not reducible, and differ only on the demes that would collapse. The extra
cost is one baseline reduction per junctor (about 2 % more reductions).

## Top-K FOS subsets (`--gomea-fos-topk`)

A GOM generation tries every FOS subset on every individual. A linkage tree has about two
subsets per knob, and demes have 100–2,000 knobs, so with `--gomea-max-evals-per-deme 250`
a deme usually never finishes its first generation, and the linkage model is learned only
once, from the unselected initial population. `--gomea-fos-topk K` (K > 0) narrows the FOS
learned at the start of every generation to K subsets, so one generation costs at most
`N × K` GOM trials and a capped deme can finish several generations (and relearn the
linkage from a selected population).

* **`linkage` (default).** The candidates are the linkage tree's merges: every subset of
  two or more knobs the tree builds (the root, the set of all knobs, is already left out of
  the FOS). Each merge's strength is the UPGMA similarity, at the moment of merging, between
  the two clusters it joins: the cardinality-weighted average of the pairwise similarities
  the tree is built from (`lt-b`: initialization-bias-corrected MI; `lt-mi`: MI). Higher is
  more strongly linked. The K merges with the highest strength are kept, ties going to the
  earlier merge. Singletons are not candidates: they carry no linkage, and GOM over a
  singleton is univariate mixing. Since UPGMA never inverts, a kept merge's sub-merges are
  kept too (barring ties), so the selection is the bottom of the tree: its tightest
  clusters. If the tree has fewer than K merges (fewer than K + 2 knobs), the remaining
  slots go to singletons in FOS order. Needs `--gomea-fos lt-b` or `lt-mi`.
  **Caveat (`lt-b`, first generation):** the bias-corrected similarity is measured against
  the initial population, so in a deme's first generation every pair scores 0 (up to
  rounding) and every merge has strength 0. The first generation's K merges are therefore
  just the earliest merges (decided by ties and rounding), close to arbitrary pairs; the
  ranking becomes informative from the second generation. `lt-mi` ranks the first
  generation by plain mutual information.
* **`random`.** K subsets drawn uniformly, without replacement, from the whole FOS
  (singletons included) each generation: the control for "fewer subsets" against "the best
  subsets". Works with every `--gomea-fos`.

Each individual's GOM visits the K subsets in its own shuffled order, as before. With
`K ≥` the FOS size the whole FOS is used. **RNG:** with K = 0 (the default) nothing is
drawn and runs are byte-identical to runs without the option. `linkage` mode draws nothing
extra (the strengths come from the tree already built). `random` mode draws exactly K
`randint_n` values per generation from the deme's linkage stream (a partial Fisher–Yates
shuffle), after the linkage tree is learned and before any GOM, whenever `K <` the FOS
size. In both modes GOM's own per-individual shuffle and donor draws are fewer, because
it visits fewer subsets. Every `generation` event then also logs `fos_topk`,
`fos_topk_mode`, `fos_effective_size`, `fos_effective_mean_subset_size`, `fos_selected`
(the subsets, as knob positions) and `fos_selected_strength` (their strengths, `null` for a
subset without one); `fos_size` stays the full FOS size.

**CPU cost:** every generation relearns the linkage tree (an entropy matrix over all knob
pairs), which takes seconds per generation on a 1,000-knob deme. Without top-K a capped deme
rarely finishes a generation, so it learns the tree about once; with top-K it runs many short
generations, so CPU time per evaluation rises sharply even though each generation costs fewer
evaluations.

## Numerical simplification of exemplars (`--simplify-exemplar-numeric`)

Port-only and opt-in: classic MOSES has no counterpart, and with the default `0` nothing
changes. It is not GOMEA-specific: it acts on the exemplar, so it applies to `-a hc`,
`hc-capped`, `gomea` and the others alike. Regression only (`-H it` on a table with a contin
output); any other problem refuses the option.

With `--simplify-exemplar-numeric TOL` (TOL > 0), every time MOSES has picked an exemplar and
before it builds knobs around it (`deme_expander.create_demes`), each non-leaf subtree of the
exemplar is evaluated on the training rows. A subtree whose outputs are all finite numbers and
whose range (max − min) is below `TOL × sd(training target)` is replaced by the mean of its
outputs, and the tree is reduced with the candidate reducer. Candidates are tried one at a time,
largest first; a replacement is kept only if the training composite score (raw score with the
complexity penalty, the score MOSES ranks candidates by) is not worse. A rejected subtree's own
subtrees are still tried. There is no validation split inside MOSES, so this guard uses the
training score. Its extra scorer calls are not charged to `-m`.

The scale is the target's standard deviation, so TOL is relative and in the units NMSE uses: a
training NMSE floor `e` corresponds to residuals of about `sqrt(e) × sd`, and the literature
puts the best threshold between one and five times the noise level. A constant target disables
the pass. Each exemplar logs one INFO line, `Numeric exemplar simplification: candidates=…
accepted=… rejected=… nodes=A->B … | totals: calls=… fired=… … nodes_removed=…`.
See `pymoses/moses/deme/numeric_simplify.py`.

## Linear scaling (`--linear-scaling`)

Port-only and opt-in: classic MOSES has no counterpart, and with the default `0` nothing
changes. It sets the regression scorer (`contin_bscore`), which every optimizer shares, so it
applies to `-a hc`, `hc-capped`, `gomea` and the others alike. Regression only (`-H it` on a
table with a contin output) with squared error; any other problem, and `--it-abs-err`, refuse
it.

With `--linear-scaling 1` a program `f` is scored on `a + b·f`, where `(a, b)` is the
least-squares fit of `f`'s predictions to the training target (Keijzer 2003):
`b = cov(y, f) / var(f)`, `a = mean(y) − b·mean(f)`, and `b = 0`, `a = mean(y)` for a constant
`f`. If a prediction is not finite the program is scored unscaled, as without the option. The
behavioural score keeps its shape, one `−(a + b·f(x) − y)²` per training row, and the
complexity penalty, best possible score and everything else are unchanged. The printed
programs are `f` itself, not `a + b·f`. `tools/gomea/sr_eval.py` reports NMSE with the same
scaling refitted on the training split and applied to the validation and test splits, so its
reported metrics mean the same with and without the option. See `contin_bscore.
set_linear_scaling` and `linear_scaling` in `pymoses/moses/scoring/bscores.py`.

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
| `invocation_start` | `schema_id` (a hash of this representation's knob layout), `dim` (number of knobs), `domains`, `max_evals`, `params` (`accept_complexity_ratio`, `deme_store`, `fos_topk` and `fos_topk_mode` appear only when not at their defaults); with `--gomea-pop-fraction` or `--gomea-deme-share` also `deme_budget`, `deme_cap` (the per-deme cap, `null` for none) and `deme_pop_size` (this deme's N); with `--gomea-long-deme-cap` also `long`, `deme_cap_used`, `long_deme_rule`, `n_demes`, `n_long_demes` |
| `init` | the initial population: `attempts`, `duplicates_rejected`, `coverage_replacements`, `feasible_short`, and `diagnostics` (per-knob entropy and observed values, `constant_fields_before/after_coverage`, `radius_histogram`) |
| `generation` | `generation`, `fos_size` (with `--gomea-fos-topk` also `fos_topk`, `fos_topk_mode`, `fos_effective_size`, `fos_effective_mean_subset_size`, `fos_selected`, `fos_selected_strength`), per-generation GOM `stats` (trials, evaluated, accepted_improving / accepted_neutral / rejected, noop_skips, fi_*), `nis` (the no-improvement stretch), `best_raw`/`best_pen`/`best_cpx` (of the engine's best, under the deme-local comparator when `--gomea-accept-complexity-ratio` is on), a `ledger` snapshot, and diversity: `unique_genotypes`, `mean_entropy_bits`, `n_varying_fields`, `best_raw_count` |
| `end` | `termination_reason`, `generations`, `ledger` (requests = evaluations charged, invalid, duplicate_requests, …), `timings` (init / eval / linkage / variation seconds), `best`; with `--gomea-long-deme-cap` also the long-deme keys of `invocation_start` |
| `hybrid` | `-a hc-gomea` only, one per deme: `deme_id`, `max_evals`, `hc_evals`, `hc_stop` (`budget`, `time_limit`, `target` or `local_optimum`), `gomea_ran`, `gomea_evals` (charged), `gomea_termination_reason` |

A deme ends with one of these `termination_reason`s: `target_reached`, `converged`,
`stagnation`, `max_generations`, `global_budget`, `inner_budget`, `time_limit`,
`no_search_fields`, `no_valid_moves`, `interrupted`, or `stall_evals` (the
`--gomea-stall-evals` rule).

## Long demes for promising exemplars (`--gomea-long-deme-cap`)

Opt-in (Phase 1af, backlog item 19). With cap 250 and N = 16 almost every deme learns its
linkage once, from the unselected initial population. `--gomea-long-deme-cap C` gives a few
demes cap C, so GOM can run several generations and relearn the linkage:

* The decision is made in `expand_deme`, after the exemplar is chosen and its deme built,
  and before the deme is optimized (`gomea_optimization.prepare_deme`,
  `pymoses/gomea/long_deme.py`).
* **`top`:** long when the exemplar's penalized score (`get_penalized_score()`, the score
  `select_exemplar` samples by) is finite and no lower than any finite penalized score in the
  metapopulation at that moment, and the exemplar (keyed by its printed combo tree) has not
  had a long deme before in the run.
* **`random`:** long with probability p, from `randGen().randdouble()`, one draw per deme
  (none when p is 0 or 1, so p = 0 gives the same run as the option off).
* A long deme still stops on the engine's own rules (convergence, `--gomea-stall-gens`,
  `--gomea-max-gens`) before C when they fire first.
* Needs `-a gomea` and a normal cap (`--gomea-max-evals-per-deme` or `--gomea-deme-share`).
  `--gomea-pop-fraction` still sizes N from the normal cap.
* **Telemetry** (only when on): `invocation_start` and `end` carry `long` (true/false),
  `deme_cap_used`, `long_deme_rule`, and the run's counts so far, this deme included:
  `n_demes` and `n_long_demes` (the last deme's values are the run's totals).

## Reserved expansions for the empty program (`--reserve-empty-every`)

Port-only and opt-in: classic MOSES has no counterpart, and with the default `0` nothing
changes (no lineage is tracked). It is not GOMEA-specific: it acts on exemplar selection,
so it applies to every `-a`. It is meant for seeded runs (`-e SEED1 ... -e +`): Phase 1r
found that MOSES never expanded the empty program's lineage once seeds were present.

With `--reserve-empty-every K` (K > 0) the metapopulation records which initial exemplar
each program descends from: the k-th `-e` is `exemplar<k>`, and a program merged for the
first time inherits the lineage of the exemplar whose deme produced it. An initial
exemplar counts as the empty program if it was `+` or `and` before reduction (reduction
turns them into `0` and `true`). Expansions K, 2K, 3K, ... then pick their exemplar only
among the empty program's descendants, by MOSES's usual Boltzmann selection (score with
the complexity penalty, `-v`, `--revisit`) restricted to that subset. If none of the
subset is selectable (all visited, or none left after trimming), the expansion makes the
normal choice. `K = 1` expands the empty lineage whenever it can. Without an empty
initial exemplar the option has no effect. It costs no evaluations; on reserved turns
each candidate's printed form is computed once.

In `tools/gomea/moses_sr_runner.py`, a seed block's `"reserve_empty_every": K` passes the
flag. The summary's `reserve_empty` (from `lineage.json`) gives `expansions_by_lineage`,
`empty_lineage_expansions`, `reserved_turns` and `reserved_fallbacks`. See
`pymoses/moses/metapopulation/lineage.py`.

## Re-seeding on stagnation (runner seed block)

Tooling-only and opt-in, for seeded regression runs of `tools/gomea/moses_sr_runner.py`
with pyMoses (backlog item 9, later extension). A seed block's `"reseed_after": S`
installs `tools/gomea/reseed.py` in the MOSES process. Optional keys:
`"reseed_share": f` (default `0.2`), `"max_reseeds": n` (default `1`),
`"reseed_min_budget"` (default: the initial seeding's budget, `share × -m`). The other seed-block keys
(`height`, `k`, `popsize`, `functions`, `gpgomea_argv`, `tol`) are reused.

After every expansion the hook compares the metapopulation's best score with the best so
far. After S consecutive expansions without a strict improvement it runs GP-GOMEA again
with `floor(f × (-m − evaluations so far))` evaluations and a new random seed
(`split + 1000 × r` for the r-th re-seed), verifies the elites with the running MOSES's
own scorer, reduces them and merges them into the metapopulation (lineage `reseed<r>`).
GP-GOMEA's evaluations (its own count, which can exceed the share by up to one
generation) are added to MOSES's evaluation count, so they come out of `-m` and the
run's total budget stays fixed; MOSES's printed `n_evals` includes them. GP-GOMEA checks
its limit only between generations, so it overspends a small share (on Con a share of 128
cost 684 evaluations); a share below the minimum budget therefore skips that re-seed (a
skip does not count towards `max_reseeds`, and is retried after the next S stagnant
expansions), and an overspend larger than what MOSES has left is recorded as
`over_budget`. The verification and merge scorer calls are not
counted, as in the initial seeding.

GP-GOMEA's initial population is not seeded from the metapopulation: GP-GOMEA works on a
fixed tree template of height H with its own function set, so reading MOSES programs
(larger, with `sin`/`exp`/`log` and arbitrary constants) into it would need substantial
C++ changes. Each re-seed is a fresh GP-GOMEA run instead.

The summary adds `n_reseeds`, `reseed_evals`, `reseed_cpu_s` and `moses_own_evals`
(`n_evals − reseed_evals`); `reseed.json` records every re-seed (`at_expansion`, `budget`,
`evaluations_used`, the seeds, the best score before and after). The hook point is
`pymoses.moses.moses.local_moses.expansion_hooks`, a port-only list of callables run after
each expansion, empty by default.

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
  hill climbing. `deme.n_evals == len(deme) ==` the number of scorer calls (with
  `--gomea-deme-store accepted`, `len(deme)` is smaller, but `deme.n_evals` still counts
  every scorer call). GOMEA never exceeds its budget.
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
