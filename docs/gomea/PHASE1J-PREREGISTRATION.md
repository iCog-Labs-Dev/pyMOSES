# Phase 1j pre-registration: GOMEA vs hill climbing at equal wall-clock time

Written 2026-10-12, before any Phase 1j run. Amended the same day, still before any run, to
the two-stage design (10 pairs, then 10 more if needed).

**Requested by the user:** "Does our implementation support time base termination to run
without any evaluation cap just like the classic version? ... follow the classic way
closely", then "schedule for a benchmark on one symbolic regression and one complex boolean
problem and then let's see if there's any difference in the result after that change."

## Question

Phases 1e and 1f compared tuned GOMEA with hill climbing (HC) at **equal evaluations**.
GOMEA won on mux11 and on 4 of 6 regression datasets, but it used 2.3–5.4× the CPU. Does
it still beat HC at **equal wall-clock time**, the way classic MOSES is usually run?

## How classic MOSES terminates on time (checked in the code)

* **C++ MOSES** (`asmoses/moses/main/problem-params.cc:262–266`): `-m`/`--max-evals`
  defaults to 10,000 and `--max-time` to INT_MAX.
* **The main loop** (`moses/local_moses.cc:131–137`) runs while evaluations < max-evals
  **and** elapsed time < max-time.
* **pyMoses** ports both defaults and the loop unchanged (`problem_params.py:607–608`,
  `local_moses.py`).
* **So a time-only run sets `--max-time T` and a very large `-m`.** Phase 1f part B already
  ran C++ MOSES this way (`--max-time 1000 -m 1000000000`). Both arms here do the same.

## Design

* **Arms:**
  * `HC`: `-a hc`.
  * `LTB-C250-M`: the tuned GOMEA of Phases 1e and 1f: `-a gomea --gomea-fos lt-b
    --gomea-pop-size 16 --gomea-skip-inactive 0 --gomea-max-evals-per-deme 250
    --gomea-genotype-memo 1`, plus `--gomea-contin copy` on regression.
* **Problems:**
  * **Con, symbolic regression.** GOMEA was better at equal evaluations in Phases 1f and 1g.
    * Setup: the Phase 1f setup (`tools/gomea/moses_sr_runner.py`), paired by split: splits
      1–10, extended to 11–20 if needed (see Analysis).
    * Budget: `--max-time 1000 -m 1000000000`, the same time limit as the Phase 1f part B
      classic-MOSES runs.
    * Manifests: `configs/gomea/phase1j/con.json` (stage 1) and `con-ext.json` (extension).
  * **mux11, Boolean.** GOMEA was better at equal evaluations in Phase 1e.
    * Setup: `benchmarks/gomea/runner.py`, paired by seed: seeds 3001–3010, extended to
      3011–3020 if needed.
    * Budget: `--max-time 3600 -m 1000000000`. In Phase 1e, HC's median CPU time to solve was
      about 700 s and GOMEA's about 2,800 s, so both can solve within the limit.
    * Manifests: `configs/gomea/phase1j/mux11.json` and `mux11-ext.json`, code `7860f64`.
    * Options: `--recursion-limit 100000`, as in the Phase 1e re-runs.
* **Load control:** wall-clock limits depend on machine load.
  * The launcher (`tools/gomea/phase1j_run.sh`) waits for Phase 1h step 3 to finish, then
    runs Con and then mux11.
  * Each uses 10 workers on 22 cores, so every run has a core to itself.
  * CPU time is recorded too, to check fairness.
* **Estimate:** stage 1 about 2.5 h: Con about 0.6 h (20 runs × 1,000 s / 10 workers) and
  mux11 at most 2 h. Each extension doubles its problem's time.

## Analysis (`tools/gomea/phase1j_report.py`)

* **Metrics:**
  * **mux11:** wall-clock time to a solution; an unsolved or failed run counts as 3,600 s.
  * **Con:** test NMSE of the program chosen on validation; a failed run counts as infinite.
* **Test:** per problem, a paired Wilcoxon signed-rank test of LTB-C250-M vs HC, with Holm
  correction over the problems tested at that stage.
* **Two stages, each at α = 0.025** (Bonferroni over the two looks, so 0.05 overall):
  * **Stage 1, 10 pairs per problem:**
    * **better:** Holm p < 0.025 and a median at least 10 % lower than HC's;
    * **worse:** the mirror image;
    * **extend:** 0.025 ≤ Holm p < 0.25. Run 10 more pairs for that problem;
    * otherwise a **tie**.
  * **Stage 2, all 20 pairs, extended problems only:** the same rule, without "extend".
    Holm correction is over the extended problems.
  * The launcher (`tools/gomea/phase1j_run.sh`) applies this automatically.
* **Why two stages:** the user's default from 2026-10-12 is 10 seeds first, and 10 more only
  when a result needs confirmation.
* **Reading against equal evaluations:** compare each verdict with the earlier equal-evaluation
  verdict (mux11: better, Phase 1e; Con: better, Phases 1f and 1g).
  * If both stay "better", GOMEA's advantage survives its CPU cost.
  * A tie or "worse" means the CPU overhead cancels the gain.
* **Descriptive:** solved counts (mux11), median evaluations and CPU time per arm, and
  failures.
