# Phase 2 handoff (mixed discrete / real-valued GOMEA)

This note is for a separate execution run. It needs explicit approval that names the
approved Phase 1 commit (see `02_PHASE2_MIXED.md` in the untracked handoff pack
`MOSES_GOMEA_LOCAL_AGENT_HANDOFF/moses_gomea_plan/`, not committed, and the Phase 2 section
of `docs/superpowers/plans/2026-09-24-moses-gomea.md`). The real-valued model has not been
implemented. Copy-only constants exist (`--gomea-contin copy`, `bcccf8f`, Phase 1f).

## Adapter APIs Phase 2 builds on (as committed)

* `pymoses.gomea.schema.FrozenSchema(rep, contin="reject")`
  * Attributes: `.rep`, `.fields`, `.contin_mode`, `.contin_positions`, `.contin_depths`,
    `.dim`, `.domains`, `.schema_id`.
  * Methods: `.default_genotype()`, `.to_genotype(inst)`, `.to_instance(g)`,
    `.contin_value(pos, code)`, `.sample_contin_code(pos, rng)`, `.phenotype_key(g)` (a
    key of the unreduced decoded tree), `.decode(g)` (the reduced tree).
  * With `contin="reject"` (the default) **it raises `UnsupportedRepresentation` when
    contin fields are present.** With `contin="copy"` each contin knob is one gene at the
    front of the genotype, domain `4 ** depth`, whose code packs the knob's raw step
    fields; GOM copies it whole and never mutates it. Phase 2 must extend or wrap this
    into the mixed schema `(z, c)`.
* `pymoses.gomea.oracle.EvaluationOracle(schema, iscorer, deme, max_evals, max_time,
  inner_cap=None, clock=time.monotonic, genotype_memo=False, stall_evals=0)`
  * Methods: `check()` and `evaluate(g)`; `evaluate` appends to the deme and raises
    `BudgetExhausted`. `preload(scored, prior_evals)` marks genotypes another optimizer
    already evaluated (used by `-a hc-gomea`); `finalize()` writes `deme.n_evals`.
  * The ledger counts requests, invalid, no-op skips, phenotype skips, duplicate requests,
    memo hits and cache deltas.
  * For Phase 2 it must key on `(schema_id, z, c)` and evaluate a recipient's own
    constants.
* `pymoses.gomea.fos.LinkageModel(mode, rng)`: `.calibrate(pop, domains)` is called once,
  then `.learn(pop)` returns the FOS over genotype indices. This is the discrete side of
  the mixed coordinator.
* `pymoses.gomea.gom.gom(...)`, `forced_improvement(...)` and
  `pymoses.gomea.engine.GomeaEngine`.
  * The generation loop does per-recipient FOS order ([P1] style).
  * Phase 2's subset-major GP step is a different schedule (see `02_PHASE2_MIXED.md` and
    the plan's Task P2.3). Add a discrete-only subset-major control.
* `pymoses.gomea.optimizer.gomea_optimization` / `gomea_parameters`: the MOSES seam. A
  mixed optimizer should be a sibling class selected by a new `-a` value, reusing
  `population.build_population` for z.
* Harness: `benchmarks/gomea/*`, with manifests, the runner (memory limit, code check,
  resume) and the analysis.
  * Phase 2 needs a correlated/hierarchical analysis for CV folds (protocol §8). That is
    not in `analyze.py`.

## Continuous pieces that are missing

* **Constants.**
  * There are no per-individual real-valued constant pools. `--gomea-contin copy` only
    recombines the step-encoded codes present in the population; nothing tunes them.
  * The native MOSES contin knobs are **step-encoded** (`contin_spec`: depth × {Stop, Left,
    Right} raw fields). They are not real coordinates and must not be fed to RV-GOMEA.
  * `field_set.get_contin` / `set_contin` decode and encode them.
* **Regression side of MOSES.** The pyMoses port does support contin scoring
  (`contin_bscore`), contin reduction, and the `+ - * / sin log exp` interpreter.
  Division semantics must be pinned before any gradient or SymPy baseline.
* **RV engine.** Nothing exists yet for AMS/AVS, the Gaussian model, RV-FI, the
  intron-aware moments, the evaluation-balanced scheduler, or linear scaling.

## Dependency choices made in Phase 1 and their implications

* **Pure stdlib, no numpy.** The Phase 1 statistics were O(N·d²) in Python and cheap: LT-B
  learning took 0.12 s at d = 150, N = 256. An RV component needs covariance factorizations
  (K × K, K ≈ 10). That is still feasible in pure Python for K ≤ 20, but porting [C3]/[C4]
  numerics without numpy is a real cost. **Decide at P2.0 whether a numpy dependency is
  acceptable.** The project currently promises zero runtime dependencies.
* **Reuse by narrow port with golden parity** worked well for [C1]. The same approach fits
  [C4]'s C RV-GOMEA (build a g++/gcc harness from the archive).

## Measured bottlenecks (Phases 1–1f)

* **Evaluation dominates** at about 6–12 ms per candidate on the Phase 1 Boolean demo
  problems. The score cache is off for demo problems.
* **Per-evaluation cost grows with the representation** (Phase 1e, median CPU per scorer
  call, HC vs GOMEA): mux11 92 vs 525 ms, maj7 129 vs 466 ms, pa5 66 vs 69 ms. In the
  original maj7 GOMEA runs (median 472 knobs), time inside the GOMEA code was about 16 %
  of the CPU; of that, scoring took about 84 % and linkage learning about 9 %. The rest
  was MOSES's own per-deme work outside the optimizer.
* **GOMEA's per-generation cost is about N × |FOS| trials**, where |FOS| ≈ 2d. On MOSES
  representations with 30–150 fields, that makes a single generation cost thousands of
  evaluations. In the pilot, larger N was uniformly worse under a 20k budget.
* **Representation turnover is expensive for both optimizers.** In Phase 1 HC ran many
  expansions and showed more CPU per evaluation end to end. From Phase 1d on, capped GOMEA
  runs more demes than HC (maj7: 161 vs 43 per run, median), and on larger representations
  it costs more CPU per evaluation, not less (above).
* **Phenotype-proof skips are nearly useless on MOSES representations**, because neutral
  single-field changes are rare. They are off by default.

## What Phases 1d–1f answered

Phase 1 found LT-B GOMEA **worse** than HC end to end, with half of the tree-FOS demes
ending by population convergence and 32–47 % of requests re-scoring duplicates. The
follow-ups answered most of that:

* **Phase 1d:** the main cause was per-deme budget use. A 500-evaluation cap tied HC on
  maj5; a more diverse start and strict acceptance did not help; a larger population was
  worse end to end ([`PHASE1D-FINDINGS.md`](PHASE1D-FINDINGS.md)).
* **Phase 1e:** a 250 cap with `lt-b` and the genotype memo needed about half HC's
  evaluations on mux11 (*better*) and tied on maj7 and pa5, at 3–4× the CPU time on mux11
  and maj7. The memo's effect was mixed ([`PHASE1E-FINDINGS.md`](PHASE1E-FINDINGS.md)).
* **Phase 1f:** on the GP-GOMEA paper's regression benchmarks, the same setup with
  `--gomea-contin copy` beat HC on test error on 4 of 6 tested datasets and lost on none
  ([`PHASE1F-FINDINGS.md`](PHASE1F-FINDINGS.md)).

Build the mixed optimizer on the capped setup, not on uncapped demes.

## Learned in Phases 1e–1f

* **Constants.** `--gomea-contin copy` and `--gomea-contin-stats` (bin-const, γ = 100) are
  Phase 2's starting point: constants already take part in GOM and in linkage learning,
  and Phase 2 adds the real-valued model on top.
* **Reduct fix.** A port bug in `reduce_factorize_fraction` crashed contin reduction; it is
  fixed and pinned against the C++ output (`484e302`). Every regression run depends on it.
* **Recursion limit.** `neighborhood_sampling.count_neighborhood_size_from_index` recurses
  once per knob and exceeds Python's default limit past about 1,000 knobs, in HC too. The
  harnesses raise the limit (`--recursion-limit` in `benchmarks/gomea/runner.py`,
  `63023d9`; a wrapper in `tools/gomea/moses_sr_runner.py`, `8b23deb`). Backlog: make it
  iterative.
* **Memory growth.** GOMEA peaked at 805 MB on mux11 against HC's 44 MB (Phase 1e), and at
  up to 1.35 GB on the six tested regression datasets, 2.9 GB on Tow and over 5 GB on 2 of
  4 Dow LTB runs, against about 0.1 GB for HC (Phase 1f). It is not profiled. Profile it
  before adding per-individual real-valued state.
* **Speed.** pyMoses is about 20–35× slower than C++ MOSES. GOMEA used 2–23× HC's CPU
  time on the regression datasets. At 1,000 s per run, C++ GP-GOMEA did 3.3–27.7 million
  evaluations per run against 19–362 thousand for C++ MOSES.
* **Tooling to reuse** (`tools/gomea/`): `sr_datasets.py` (splits), `sr_eval.py` (NMSE
  scoring of MOSES programs), `moses_sr_runner.py`, `gpgomea_sr_runner.py`,
  `watch_runs.py` (a run watchdog) and `phase1f_report.py`. The datasets live outside the
  repository (`../gomea-results/phase1f/datasets/`).
* **Unfinished Phase 1f work.**
  * Dow LTB exceeded 5 GB on 2 of 4 splits and was dropped; Dow RT stopped at 6 of 10
    splits.
  * Tow, WiR and WiW are descriptive only.
  * LTMI ran only on Air, Bos and Con.
  * Whether the linkage-model advantage appears at a paper-scale budget (about 10⁶
    evaluations) is untested.
* **The `-a hc-gomea` hybrid did not help** (maj5 5/10 vs HC 9/10 in Phase 1e stage 1).
* **MOSES vs GP-GOMEA setup differences** that stay in any comparison: MOSES has no
  analytic quotient, no linear scaling in training (the target is rescaled instead), and
  its constants are knobs rather than ERCs.
