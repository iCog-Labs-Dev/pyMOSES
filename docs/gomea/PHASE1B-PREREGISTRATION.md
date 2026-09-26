# Phase 1b pre-registration: stand-alone GP-GOMEA vs MOSES hill climbing

Written 2026-09-26, after the Phase 1b pilot and **before any Phase 1b confirmatory run**.
The commit that adds this file is the timestamp.

**Requested by the user:** "An experiment where we check if the normal GP-GOMEA without
being integrated into MOSES beats the normal hillcimbing integrated into MOSES on similar
problems ... and do similar analysis." (`APPROVALS.md` #5). The criteria below are
agent-proposed.

## Question

Does the original GP-GOMEA, run as its own program on its own fixed-height tree
representation, solve the same Boolean synthesis tasks more reliably or more cheaply than
MOSES with its stock hill climber, under the same evaluation budget?

## Systems

**GP-GOMEA** is the authors' C++ code, [C1] `github.com/marcovirgolin/GP-GOMEA@750220c`
(Apache-2.0).
* **Build:** unmodified sources, compiled out of tree against Armadillo 14.0.2 (source
  sha256 `a067616d…d207`, used header-only) with the system BLAS/LAPACK and Boost 1.92.
  The recipe is `tools/gomea/build_gpgomea.sh`.
* **Binary for all runs:** an *instrumented* copy that differs from the source by one
  `cout` field. It prints the cumulative evaluation count on each generation line
  (`phase1b/gpgomea-build/instrumentation.diff`). The search is unchanged.
* **Hashes:** stock `bfbfbf47…d1d4`, instrumented `480a0c89…a548`.

**MOSES HC** means the Phase 1 confirmation's `HC` runs, reused unchanged: stock `-a hc`
at code `c6301cb`, the same tasks, the same 20k budget, seeds 1001–1030 (pa4 1001–1020).
The MOSES-embedded GOMEA arms from Phase 1 (`UNI`, `LT-B`) are shown for context.

## Tasks and budget

* **Truth tables:** generated from MOSES's own target functions
  (`tools/gomea/make_truth_tables.py`, checksums in `configs/gomea/phase1b/data/`).
  * pa4: `even_parity`, 16 rows;
  * maj5: `majority(5)`, 32 rows;
  * mux6: `multiplex(2)`, 64 rows.
* **Primitives:** AND, OR, NOT in both systems. No constants.
* **Error:** the number of wrong rows. GP-GOMEA's error rate × rows equals MOSES's −score.
* **Budget:** 20,000 full-table evaluations per run in both systems.
* **How GP-GOMEA is measured.** It checks its limit only between generations and can
  overshoot. Every metric is therefore read at the **last generation boundary within 20k
  evaluations**.
  * *success* means a zero-error elitist by then;
  * *evaluations-to-success* is the cumulative count at the end of the first such
    generation, which is conservative by less than one generation.
  * A run that completes no generation within 20k has no result, and counts as the worst
    error and a failure.

## GP-GOMEA arms (fixed from the pilot)

| Arm | Settings | Why |
|---|---|---|
| **GPG-DEFAULT** | the authors' defaults from `pyGPGOMEA`: population 500, IMS `5_1`, LT FOS, `initmaxtreeheight 4`, `syntuniqinit 1000`; `--functions AND_OR_NOT --prob accuracy` | "normal GP-GOMEA" out of the box |
| **GPG-TUNED** | the same, but population **64** and height **5** | best of the pilot grid {64, 500} × {4, 5, 6}: 6/15 solved vs ≤ 4/15 for the others; 5 seeds (501–505) per task. Rule: most successes, then lowest median error |

**Tuning asymmetry, disclosed:** GP-GOMEA gets a small pilot tuning; MOSES HC runs with its
defaults. This favours GP-GOMEA.

Pilot results:

* the defaults solved 0/15;
* population 500 at heights 5–6 never finished a generation within 20k;
* P64-H5 solved maj5 4/5, mux6 2/5 and pa4 0/5.

The pilot data is in `phase1b-pilot/` (summaries in `docs/gomea/phase1b-results/pilot/`).

**Secondary, descriptive only:** GPG-TUNED with a **200k** budget, to see where GP-GOMEA
gets with ten times the evaluations. There is no HC counterpart, so it is not tested.

## Seeds and runs

GP-GOMEA uses seeds 1001–1030 (pa4: 1001–1020). The two systems' random generators are
unrelated, so **seed numbers do not pair runs**. All comparisons are unpaired.

## Metrics, tests and the decision rule

* **Metrics:** success (zero error within budget); final error (wrong rows); censored
  evaluations-to-success (failure = budget + 1). CPU seconds and program size (GP-GOMEA
  nodes; MOSES complexity) are descriptive only. C++ vs pure Python CPU is **not**
  comparable.
* **Tests, per task and per contrast:**
  * Fisher's exact test, two-sided, on success;
  * Mann–Whitney U, two-sided, on final error;
  * Mann–Whitney U on censored evaluations-to-success.

  Holm correction runs across all contrasts × metrics within a task. We also report
  success rates with Wilson 95 % CIs, medians and IQRs, and a bootstrap 95 % CI of the
  difference in medians (10,000 resamples, seed 12345).
* **Contrasts:** primary, **GPG-TUNED vs HC** and **GPG-DEFAULT vs HC**. Secondary,
  GPG-TUNED vs MOSES-UNI and GPG-TUNED vs MOSES-LT-B.
* **Decision rule (same as Phase 1):** an arm is *better* on a task if some Holm-adjusted
  p < 0.05 **and** it either has a success rate at least 10 percentage points higher or a
  median censored evaluations-to-success at least 20 % lower. *Worse* is the mirror image.
  Otherwise it is a *tie*.

## Notes recorded after the runs

* **The instrumentation was redundant.** Stock GP-GOMEA already writes cumulative
  evaluations per generation to `stats_generations.txt` in its working directory. The
  one-line log change, which leaves the search untouched, was therefore not needed.
* **Shared working directory.** The confirmation runs used the repo root as their working
  directory, so those files were shared across runs and discarded. Every metric comes from
  each run's own stdout. `gpgomea_runner.py` now gives each run its own working directory.
