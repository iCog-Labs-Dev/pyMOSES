# Phase 2 handoff (mixed discrete / real-valued GOMEA)

This note is for a separate execution run. It needs explicit approval that names the
approved Phase 1 commit (see `02_PHASE2_MIXED.md` and the plan's Phase 2 section). Nothing
here has been implemented.

## Adapter APIs Phase 2 builds on (as committed)

* `pymoses.gomea.schema.FrozenSchema(rep)`
  * Attributes: `.fields`, `.dim`, `.domains`, `.schema_id`.
  * Methods: `.default_genotype()`, `.to_genotype(inst)`, `.to_instance(g)`,
    `.phenotype_key(g)` (a key of the unreduced decoded tree), `.decode(g)` (the reduced
    tree).
  * **It raises `UnsupportedRepresentation` when contin fields are present.** This is the
    first thing Phase 2 must extend or wrap. The mixed schema is `(z, c)`.
* `pymoses.gomea.oracle.EvaluationOracle(schema, iscorer, deme, max_evals, max_time, inner_cap)`
  * Methods: `check()` and `evaluate(g)`; `evaluate` appends to the deme and raises
    `BudgetExhausted`.
  * The ledger counts requests, invalid, no-op skips, phenotype skips, duplicate requests
    and cache deltas.
  * For Phase 2 it must key on `(schema_id, z, c)` and evaluate a recipient's own
    constants.
* `pymoses.gomea.fos.LinkageModel(mode, rng)`: `.calibrate(pop, domains)` is called once,
  then `.learn(pop)` returns the FOS over genotype indices. This is the discrete side of
  the mixed coordinator.
* `pymoses.gomea.gom.gom(...)`, `forced_improvement(...)` and
  `pymoses.gomea.engine.GomeaEngine`.
  * The generation loop does per-recipient FOS order ([P1] style).
  * Phase 2's subset-major GP step is a different schedule (see `02_PHASE2_MIXED.md`).
    Add a discrete-only subset-major control.
* `pymoses.gomea.optimizer.gomea_optimization` / `gomea_parameters`: the MOSES seam. A
  mixed optimizer should be a sibling class selected by a new `-a` value, reusing
  `population.build_population` for z.
* Harness: `benchmarks/gomea/*`, with manifests, the runner (memory limit, code check,
  resume) and the analysis.
  * Phase 2 needs a correlated/hierarchical analysis for CV folds (protocol §8). That is
    not in `analyze.py`.

## Continuous pieces that are missing

* **Constants.**
  * There are no per-individual constant pools.
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

## Measured bottlenecks (Phase 1)

* **Evaluation dominates** at about 6–12 ms per candidate on Boolean demo problems. The
  score cache is off for demo problems.
* **GOMEA's per-generation cost is about N × |FOS| trials**, where |FOS| ≈ 2d. On MOSES
  representations with 30–150 fields, that makes a single generation cost thousands of
  evaluations. In the pilot, larger N was uniformly worse under a 20k budget.
* **HC's cost is dominated by representation turnover.** It runs many expansions with
  large representations, and knob building plus reduction costs time. In the end-to-end
  comparisons that shows up as more CPU per evaluation for HC.
* **Phenotype-proof skips are nearly useless on MOSES representations**, because neutral
  single-field changes are rare. They are off by default.

## Open Phase 1 questions that affect Phase 2

The Phase 1 confirmation results (`PHASE1-FINDINGS.md`, once written) determine whether the
discrete GOMEA inner loop is worth extending. If LT-B does not beat HC end-to-end, Phase 2
should first establish why: per-deme cost, representation turnover, or linkage quality.
The frozen-representation comparison is the main evidence for that.
