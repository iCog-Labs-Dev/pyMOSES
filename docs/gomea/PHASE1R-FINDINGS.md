# Phase 1r findings: seeding the metapopulation with GP-GOMEA elites

**Status:** complete, on 2026-10-18.

* **Stage 1:** 17:50 to 20:46 (Con and EnH, 80 runs).
* **Con extension:** to 22:21 (splits 11–20, 40 runs).

All 120 runs ended `ok`, and none had an infinite test error. Protocol:
`PHASE1R-PREREGISTRATION.md`, including its launch note. Code `a1c2ce7`. Results:
`phase1r-results/` and `../gomea-results/phase1r/`.

**Answer:** yes for hill climbing, and mostly yes for GOMEA.

* **Capped HC:** seeding MOSES with GP-GOMEA's elites, at the same total budget, lowered test
  error on both problems: by 39 % on Con and 72 % on EnH.
* **GOMEA:** seeding lowered error on EnH (−46 %). On Con it was a registered tie, with
  seeded GOMEA worse in 14 of 20 splits.
* **Once seeded, GOMEA and HC tie.**
* **Where the gain comes from:** most of it is the seeds themselves. MOSES improves on the
  best seed only slightly.

## Setup

* **Budget:** every arm has 10,000 evaluations in total and a cap of 250 per deme.
* **Seeded arms (`-S`):**
  * GP-GOMEA runs first, with about 20 % of the budget (median 2,390–2,509 evaluations,
    because it finishes its last generation), tree height 3, functions `+ - *`.
  * MOSES then gets the remainder and starts from the top 5 distinct elites plus the empty
    program.
* **Arms:** `HC-C250`, `HC-C250-S`, `LTB-C250-M` (current GOMEA), `LTB-C250-M-S`.
* **Splits:** Con 1–20, EnH 1–10.

## Results

**Median test NMSE** (Con: 20 splits; EnH: 10):

| Problem | HC-C250 | HC-C250-S | LTB-C250-M | LTB-C250-M-S | Best seed alone |
|---|---|---|---|---|---|
| Con | 70.2 | 42.6 | **35.2** | 42.3 | 46.5 |
| EnH | 28.9 | **8.1** | 15.7 | 8.5 | 8.9 |

**Size and cost** (medians):

| Problem | Arm | Final size (nodes) | CPU per run (s) |
|---|---|---|---|
| Con | HC-C250 / HC-C250-S | 51 / 24.5 | 585 / 286 |
| Con | LTB-C250-M / LTB-C250-M-S | 229.5 / 28.5 | 2,673 / 581 |
| EnH | HC-C250 / HC-C250-S | 89 / 48.5 | 769 / 453 |
| EnH | LTB-C250-M / LTB-C250-M-S | 366.5 / 64.5 | 3,897 / 747 |

* **GP-GOMEA's own cost:** about 0.1 s of CPU per run. It is compiled C++, so the seeding
  step is almost free in time.
* **Lineage:** in all 60 seeded runs the reported program descends from a seed. MOSES never
  chose to build on the empty program.
* **Improvement over the seed:** the reported program was better on test than the best seed
  in 50 of 60 seeded runs. The margin was small: median 42.3–42.6 against 46.5 on Con, and
  8.1–8.5 against 8.9 on EnH.

**Primary contrasts** (seeded vs unseeded; Holm over 4):

| Problem | Contrast | Stage | Seeded lower / tie / higher | Holm p | Verdict |
|---|---|---|---|---|---|
| Con | seed-hc | 1 | 9 / 0 / 1 | 0.012 | **seeded better** |
| Con | seed-gomea | 1 | 3 / 0 / 7 | 0.13 | extend |
| Con | seed-gomea | 2 (20 splits) | 6 / 0 / 14 | 0.036 | **tie** |
| EnH | seed-hc | 1 | 10 / 0 / 0 | 0.0078 | **seeded better** |
| EnH | seed-gomea | 1 | 10 / 0 / 0 | 0.0078 | **seeded better** |

* **Con seed-gomea:** Holm p 0.036 is above the per-look α of 0.025, so the verdict is a
  tie. The direction favours unseeded GOMEA (median 35.2 against 42.3).
* **Other contrasts:** the verdicts listed for them are from stage 1. Con's seed-hc contrast,
  descriptive with 20 splits, was 18 / 0 / 2.

**Secondary contrasts** (seeded GOMEA vs seeded HC; Holm over 2): both ties. On Con the split
was 4 / 2 / 4 at stage 1 (8 / 3 / 9 over 20 splits). On EnH it was 4 / 0 / 6.

## What this shows

1. **GP-GOMEA's small fixed-shape programs are a strong start for MOSES.**
   * Seeded capped HC beat unseeded capped HC by a wide margin on both problems.
   * On EnH, both seeded arms reached about half of unseeded GOMEA's error.
   * The seeding step cost about 2,500 evaluations and almost no CPU.
2. **MOSES mostly polishes the seeds.**
   * The final programs descend from seeds in every run, and are only modestly better than
     the best seed.
   * What GP-GOMEA finds with 2,500 evaluations is close to what the whole seeded run
     reports.
3. **On Con, unseeded GOMEA remains the best arm** (35.2).
   * Seeding pulled GOMEA towards the seeds' basin (42.3), close to the seeds' own 46.5.
   * The empty-program control was never built on. A seed that is good but not the best
     region can trap the run.
   * This was a tie at the registered threshold, not a significant loss.
4. **Seeding also gives much smaller and cheaper programs.**
   * Final size fell 1.8–8×, for example GOMEA on Con from 229.5 to 28.5 nodes.
   * CPU per run fell 1.7–5.2×, because knob building scales with exemplar size.
   * This addresses the cost problem from Phases 1n and 1p directly.
5. **Once seeded, the optimizer inside MOSES no longer matters:** seeded GOMEA and seeded HC
   tie on both problems.

## Limits

* Two regression problems, one seeding share (20 %) and one tree height (3).
* No Boolean problem. Earlier, a single mux11 pilot with seeding doubled the evaluations to
  solve.
* EnH was not extended: its contrasts were decided at stage 1.
* At stage 2 the report recomputed Holm over the full family. The EnH verdicts are unchanged
  (Holm p 0.0059).
* **Possible follow-ups,** not registered:
  * reserve some expansions for the empty program;
  * re-seed when MOSES stagnates;
  * vary the share and the tree height;
  * a Boolean phase.
