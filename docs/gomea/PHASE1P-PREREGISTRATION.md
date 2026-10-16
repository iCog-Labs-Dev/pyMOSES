# Phase 1p pre-registration: GOMEA vs capped hill climbing at equal wall-clock time

Written 2026-10-16, before any Phase 1p run.

**Requested by the user:** make GOMEA at least tie hill climbing (HC); always compare under the
same per-deme cap; continue through the backlog. This is the decisive equal-time test under
the same cap.

## Background

* **Phase 1n (equal evaluations, cap 250 for both):** GOMEA beat capped HC on mux11 and Con,
  but cost about 2.75× the CPU per evaluation. On mux11 its CPU to solve was about 1.8× capped
  HC's (descriptive).
* **Phase 1o:** a complexity penalty in GOMEA's acceptance cut CPU (−37 % on Con) but worsened
  Con's test error. It is included here as an exploratory arm.
* **Phases 1j and 1l** compared against *uncapped* HC at equal time. GOMEA tied or won on Con
  and lost on mux11.

## Design

* **Arms** (all `--max-time T -m 1000000000`, the classic way, and cap 250 per deme):
  * `HC-C250`: `-a hc-capped --gomea-max-evals-per-deme 250`.
  * `LTB-C250-M`: `-a gomea --gomea-fos lt-b --gomea-pop-size 16 --gomea-skip-inactive 0
    --gomea-max-evals-per-deme 250 --gomea-genotype-memo 1`, plus `--gomea-contin copy` on Con.
  * `PEN` (exploratory): LTB-C250-M + `--gomea-accept-complexity-ratio` (0.125 on mux11,
    0.005 on Con).
* **Problems:**
  * **mux11:** T = 3,600 s, seeds 3001–3010, extension 3011–3020.
  * **Con:** T = 1,000 s, splits 1–10, extension 11–20.
* **Manifests:** `configs/gomea/phase1p/`. Code: the commit in the mux11 manifest.
* **Load:** 10 workers on 22 cores. CPU time is recorded.
  * **Launch note (2026-10-16 17:25):** launched while two agent pilots used about 5 cores, so
    about 15 busy processes on 22 cores and every run still had a core to itself. The plan
    had been to wait for an idle machine.

## Analysis (`tools/gomea/phase1p_report.py`)

* **Metrics:**
  * **mux11:** wall time to a solution; an unsolved run counts as 3,600 s.
  * **Con:** test NMSE.
* **Contrasts:**
  * **Primary,** with Holm correction over 2: `vs-hc`, LTB-C250-M vs HC-C250, per problem.
  * **Secondary,** with Holm correction over 4: `pen-vs-hc` and `pen-vs-current`, per problem.
* **Rule:** two looks at α = 0.025.
  * **Better / worse:** Holm p < 0.025 and a median at least 10 % lower / higher.
  * **Extend:** for primary contrasts at stage 1, 0.025 ≤ Holm p < 0.25.
  * Otherwise a **tie**.
* **Stage 2:** the Holm family stays complete. A problem that was not extended enters with its
  stage-1 data, and only contrasts that were "extend" at stage 1 take a stage-2 verdict.
* **Reading:** the user's goal ("at least tie") is met on a problem if `vs-hc` is "tie" or
  "better".
