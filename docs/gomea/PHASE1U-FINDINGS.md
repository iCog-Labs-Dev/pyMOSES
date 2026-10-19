# Phase 1u findings: GP-GOMEA seeding on mux11

**Status:** complete, on 2026-10-19.

* **Stage 1:** 13:24 to 15:04 (40 runs).
* **Extension:** 15:04 to 16:24 (seeds 3011–3020, 40 runs).

All 80 runs ended `ok` and solved. Protocol: `PHASE1U-1AA-PREREGISTRATION.md`. Code: the
merged HEAD `311d8eb` (knob-building speed-ups + seeding options; every arm has
`--knob-probe-baseline reduced`). Results: `phase1u-results/` and
`../gomea-results/1/phase1u-bool*/`.

**Answer:** seeding makes **GOMEA** solve mux11 with 41 % less CPU (registered **better**).
It does nothing for **HC** (tie). With seeding, GOMEA comes close to HC on time but does not
beat it.

## Setup

* **Arms:** `HC`, `HC-S`, `GOMEA`, `GOMEA-S`, all capped at 250 with `-m 40000`.
* **Seeding:**
  * `-S` arms run GP-GOMEA first with 20 % of the budget (median 8,748 evaluations, a 9 %
    overshoot because it finishes its generation). Settings: height 3, top 5 elites,
    AND/OR/NOT.
  * MOSES then starts from the seeds plus the empty program with the rest.
  * The seeding's evaluations and CPU are charged to the run.
* **Metric:** CPU seconds to solve (unsolved counts as +∞; none were unsolved).

## Results

**Medians:**

| Arm | Seeds | CPU to solve (s) | Evaluations to solve | CPU per evaluation (s) | Demes per run |
|---|---|---|---|---|---|
| HC | 20 | 664 | 7,866 | 0.088 | 32 |
| HC-S | 20 | 627 | 15,258 | 0.041 | 26 |
| GOMEA | 20 | 1,321 | 5,166 | 0.246 | 21 |
| GOMEA-S | 20 | **785** | 13,251 | 0.060 | 19 |

* **Seeds:**
  * The best seed scored −640 at the median, i.e. 640 of 2,048 rows wrong, so the seeds
    were far from a solution.
  * Every seeded run's final best program descends from a seed (20 of 20 in each seeded
    arm).
* **CPU per evaluation, seeded arms:** this averages GP-GOMEA's nearly free evaluations
  with MOSES's own.

**Contrasts:**

| Contrast | Family | Stage | Lower / tie / higher | Holm p | Verdict |
|---|---|---|---|---|---|
| GOMEA-S vs GOMEA (`seed-gomea`) | primary | 1 | 6 / 0 / 4 | 0.21 | extend |
| GOMEA-S vs GOMEA | primary | 2 (20 seeds) | **15 / 0 / 5** | **0.0097** | **seeded better** (−41 %) |
| HC-S vs HC (`seed-hc`) | primary | 1 | 6 / 0 / 4 | 0.77 | **tie** (not extended) |
| GOMEA-S vs HC-S | secondary | 1 | 4 / 0 / 6 | 0.11 | **tie** |

**Descriptive, with all 20 seeds:**

* **HC-S vs HC:** 13 / 7, p 0.33.
* **GOMEA-S vs HC-S:** 6 / 14, p 0.030; seeded HC still faster, median 627 against 785 s.
* **Unseeded GOMEA vs HC:** 3 / 17, p 0.0005; HC about 2× faster.

## What this shows

1. **Seeding works for GOMEA on the Boolean problem too, through cost.**
   * Seeded GOMEA needs about 2.6× the evaluations to solve, about 8,700 of them from
     GP-GOMEA.
   * But MOSES then builds knobs around small exemplars, and its CPU per evaluation falls
     about 4×.
   * The net is 41 % less CPU to solve. This matches Phase 1r, where seeding cut GOMEA's
     CPU per run 4–6× on regression.
2. **HC gains nothing:** its evaluations are already cheap, so starting from seeds just
   doubles its evaluations.
3. **The Boolean time gap narrows but remains.** GOMEA against HC went from about 2× slower
   unseeded (1,321 against 664 s) to about 1.25× slower seeded (785 against 627 s).
4. **The knob-building speed-ups also show:** unseeded GOMEA solved mux11 in a median of
   1,321 s of CPU here, against 1,872 s (Phase 1o) and 3,693 s (Phase 1s, a heavier load) on the old code.
   This is a cross-phase comparison under different loads, so it is indicative only.

## Limits

* One Boolean problem, one seeding share (20 %) and one tree height (3).
* CPU time on this shared machine varies up to about 1.6× between identical runs (Phase 1t).
  The GOMEA contrast survives this (15 of 20 pairs); the smaller contrasts may not.
