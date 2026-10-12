# Phase 1j pre-registration: GOMEA vs hill climbing at equal wall-clock time

Written 2026-10-12, before any Phase 1j run.

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
    * Setup: the Phase 1f setup (`tools/gomea/moses_sr_runner.py`), splits 1–20, paired.
    * Budget: `--max-time 1000 -m 1000000000`, the same time limit as the Phase 1f part B
      classic-MOSES runs.
    * Manifest: `configs/gomea/phase1j/con.json`.
  * **mux11, Boolean.** GOMEA was better at equal evaluations in Phase 1e.
    * Setup: `benchmarks/gomea/runner.py`, seeds 3001–3020, paired.
    * Budget: `--max-time 3600 -m 1000000000`. In Phase 1e, HC's median CPU time to solve was
      about 700 s and GOMEA's about 2,800 s, so both can solve within the limit.
    * Manifest: `configs/gomea/phase1j/mux11.json`, code `7860f64`.
    * Options: `--recursion-limit 100000`, as in the Phase 1e re-runs.
* **Load control:** wall-clock limits depend on machine load.
  * The launcher (`tools/gomea/phase1j_run.sh`) waits for Phase 1h step 3 to finish, then
    runs Con and then mux11.
  * Each uses 10 workers on 22 cores, so every run has a core to itself.
  * CPU time is recorded too, to check fairness.
* **Estimate:** Con about 1.2 h (40 runs × 1,000 s / 10 workers). mux11 at most about 4 h.

## Analysis (`tools/gomea/phase1j_report.py`)

* **Metrics:**
  * **mux11:** wall-clock time to a solution; an unsolved or failed run counts as 3,600 s.
  * **Con:** test NMSE of the program chosen on validation; a failed run counts as infinite.
* **Test:** per problem, a paired Wilcoxon signed-rank test of LTB-C250-M vs HC, with Holm
  correction over the 2 problems.
* **Verdict per problem** (the Phase 1e/1f rule):
  * **better:** Holm p < 0.05 and a median at least 10 % lower than HC's;
  * **worse:** the mirror image;
  * otherwise a **tie**.
* **Reading against equal evaluations:** compare each verdict with the earlier equal-evaluation
  verdict (mux11: better, Phase 1e; Con: better, Phases 1f and 1g).
  * If both stay "better", GOMEA's advantage survives its CPU cost.
  * A tie or "worse" means the CPU overhead cancels the gain.
* **Descriptive:** solved counts (mux11), median evaluations and CPU time per arm, and
  failures.
