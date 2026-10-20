# Phase 1ab findings: re-seed from GP-GOMEA when MOSES stagnates

**Status:** complete. The run was on 2026-10-20 from 16:27 to 17:10 (30 runs, 16 workers),
all `ok`. No contrast was borderline, so no extension ran. Protocol:
`PHASE1Y-1AB-PREREGISTRATION.md`. Code: merged HEAD with the effort-3 fix (`571b596`), which
regression at default effort does not use. Results: `phase1ab-results/` and
`../gomea-results/phase1ab/`.

**Answer: no.** Re-running GP-GOMEA after 5 expansions without improvement, with a fresh random
seed, 30 % of the remaining evaluations and at most twice per run, and injecting its elites,
did not change seeded GOMEA's test error on Con.

## Results (Con, 10 splits, medians)

| Arm | Test NMSE | CPU per run (s) | Final program from a seed / re-seed |
|---|---|---|---|
| GOMEA (unseeded) | **37.1** | 1,720 | — |
| GOMEA-S | 44.9 | 475 | 10 / 0 |
| GOMEA-S-RESEED | 44.6 | 330 | 8 / 2 |

| Contrast | Family | Lower / tie / higher | Holm p | Verdict |
|---|---|---|---|---|
| RESEED vs seeded | primary | 1 / 7 / 2 | 1.0 | **tie** |
| RESEED vs unseeded | secondary | 3 / 0 / 7 | 0.16 | tie |
| Seeded vs unseeded (descriptive) | — | 3 / 0 / 7 | 0.13 | — |

**Re-seeding activity:**

* Re-seeds fired in 7 of 10 runs, once each. The second allowed re-seed never fired, because
  the remaining budget's 30 % fell below the 1,000-evaluation minimum or the run ended.
* Each re-seed used about 2,000–2,300 evaluations and injected 5 programs.
* In 2 runs the final program descends from re-seeded programs, but the test error was not
  lower. In 7 of 10 splits the result is identical to plain seeding.

## What this shows

1. **A second GP-GOMEA run rarely beats what MOSES already has.**
   * GP-GOMEA's height-3 programs are good starting points.
   * Once MOSES has refined the first seeds, fresh height-3 elites are usually worse than
     the current best. On split 1, for example, the best re-seeded program scored −25,797
     against MOSES's −18,686.
   * So they are rarely chosen for expansion.
2. **It does not fix seeding's accuracy cost on Con:** 44.6 against unseeded GOMEA's 37.1.
3. **It keeps seeding's cost advantage**, with a median CPU of 330 s against 1,720 s.
4. **Practical reading:** re-seeding as built is not worth adding. Seeding GP-GOMEA's
   population with MOSES's current programs, so that re-seeds improve on MOSES's best rather
   than start over, would need C++ changes. It remains untested.

## Limits

* One dataset (Con), one re-seed setting, 10 splits.
