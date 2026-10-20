# Phase 1y findings: can seeding stop hurting GOMEA on Con?

**Status:** complete, on 2026-10-20.

* **Stage 1:** 07:49 to 09:35 (40 runs).
* **Extension:** splits 11–20 to 10:57 (30 runs: GOMEA, GOMEA-S-RES and GOMEA-S10).

All runs ended `ok`. Protocol: `PHASE1Y-1AB-PREREGISTRATION.md`. Code: merged HEAD `311d8eb`
(+ docs). Results: `phase1y-results/` and `../gomea-results/phase1y/`.

* **Overlap disclosure:** six orphaned Phase 1aa runs were running on the machine until
  09:25 (see `QUEUE.md`). The metric, test NMSE at a fixed budget, is not affected by load.

**Answer: no.** Neither refinement brings seeded GOMEA's test error on Con down to unseeded
GOMEA's. Both stay about 15–20 % worse by median: registered **ties**, leaning worse. They
keep seeding's large savings in CPU and program size.

## Results (medians)

| Arm | Splits | Test NMSE | CPU per run (s) | Final size (nodes) |
|---|---|---|---|---|
| GOMEA (unseeded) | 20 | **35.2** | 2,370 | 229.5 |
| GOMEA-S (Phase 1r seeding) | 10 | 44.9 | 536 | 28.5 |
| GOMEA-S-RES (every 3rd expansion reserved for the empty program) | 20 | 42.4 | 484 | 28 |
| GOMEA-S10 (10 % seeding share) | 20 | 40.6 | 580 | 32 |

**Contrasts:**

| Contrast | Family | Stage | Lower / tie / higher | Holm p | Verdict |
|---|---|---|---|---|---|
| RES vs unseeded | primary | 1 | 3 / 0 / 7 | 0.074 | extend |
| RES vs unseeded | primary | 2 (20 splits) | 6 / 0 / 14 | 0.053 | **tie** |
| S10 vs unseeded | primary | 1 | 3 / 0 / 7 | 0.43 | **tie** (not extended) |
| RES vs seeded | secondary | 1 | 4 / 2 / 4 | 0.46 | tie |
| S10 vs seeded | secondary | 1 | 7 / 1 / 2 | 0.20 | tie |

* **S10 at stage 2, descriptive:** with 20 splits, S10 vs unseeded was 5 / 15, raw p 0.040.
* **Seeded vs unseeded, descriptive (10 splits):** 44.9 against 37.1, 3 / 7, p 0.13. This
  repeats Phase 1r's direction exactly; these are deterministic reruns of the same arms.

## What this shows

1. **The reserve barely reached the empty program.**
   * It was meant to give the empty program's lineage every 3rd expansion (about 33 %). It
     got a median of 6 % of expansions; the rest fell back to the normal choice, presumably
     because no program of that lineage was selectable (not checked).
   * Every final program still descended from a seed (20 of 20).
   * So this phase tests the option as built, but only weakly tests the idea "let MOSES grow
     its own solution alongside the seeds".
2. **A smaller seeding share helps a little but not enough.** At 10 % (median 1,226 seeding
   evaluations) the median test error is 40.6 against 44.9 at 20 %. That is still above
   unseeded GOMEA's 35.2.
3. **Seeding's cost advantage holds in every seeded arm.**
   * About 4–5× less CPU per run (484–580 s against 2,370 s).
   * About 7–8× smaller final programs (28–32 against 229.5 nodes).
4. **Practical reading:**
   * On Con, seeding trades about 15–20 % higher test error for much smaller, cheaper
     programs.
   * On EnH (Phase 1r) it improved both.
   * Whether to seed depends on the dataset and on how much program size matters.

## Limits

* One dataset, one reserve interval (3), one alternative share (10 %).
* The reserve's low effective share means a stronger version would need the empty lineage
  kept selectable, e.g. protected in the metapopulation.
