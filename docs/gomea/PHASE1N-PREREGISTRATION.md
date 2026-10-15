# Phase 1n pre-registration: GOMEA vs hill climbing with the same per-deme cap

Written 2026-10-15, before any Phase 1n run.

**Requested by the user:** "When comparing the GOMEA and HC from now on, we should use the
same deme cap for the both of them. If you can, verify one of the boolean and regression
problems using a 10 seed please."

## Question

Our GOMEA-vs-HC wins compared capped GOMEA (250 evaluations per deme) with uncapped HC:

* **mux11 (Phase 1e):** median 4,633 against 8,757 evaluations to solve.
* **Regression (Phase 1f part D):** better on 4 of 6 datasets.

Phase 1g compared capped HC (`-a hc-capped`) with capped GOMEA only on regression at 2,000
evaluations, where LT-B's lead over capped HC was not significant.

**With the same per-deme cap for both, does GOMEA still beat HC?**

## Design

* **Arms** (equal evaluations):
  * `HC-C250`: `-a hc-capped --gomea-max-evals-per-deme 250`.
  * `LTB-C250-M`: `-a gomea --gomea-fos lt-b --gomea-pop-size 16 --gomea-skip-inactive 0
    --gomea-max-evals-per-deme 250 --gomea-genotype-memo 1`, plus `--gomea-contin copy` on Con.
* **Problems:**
  * **mux11 (Boolean):** `-m 40000` (Phase 1e's budget), seeds 3001–3010.
    `benchmarks/gomea/runner.py`, `configs/gomea/phase1n/mux11.json`.
  * **Con (regression):** `-m 10000` (Phase 1f part D's budget), splits 1–10.
    `tools/gomea/moses_sr_runner.py`, `configs/gomea/phase1n/con.json`.
* **Extension if borderline:** seeds 3011–3020 and splits 11–20.
* **Code:** `8b1e728`.
* **Run:** after Phase 1m finishes (memory is too tight to run both), 10 workers.

## Analysis (`tools/gomea/phase1n_report.py`)

* **Metrics:**
  * **mux11:** evaluations to a solution; an unsolved run counts as 40,000.
  * **Con:** test NMSE of the program chosen on validation.
* **Test:** paired Wilcoxon, GOMEA vs capped HC, with Holm correction over the 2 problems.
* **Two looks at α = 0.025:**
  * **Better / worse:** Holm p < 0.025 and a median at least 10 % lower / higher.
  * **Extend:** 0.025 ≤ Holm p < 0.25 at stage 1.
  * Otherwise a **tie**.
* **Descriptive:**
  * solved counts;
  * CPU per run;
  * the earlier uncapped-HC results (Phases 1e and 1f), for comparison.
