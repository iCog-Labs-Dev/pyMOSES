# Phase 1v findings: uniform-distance start, end to end

**Status:** complete. The run was on 2026-10-19 from 16:24 to 19:06 (mux11 20 runs, then Con
20 runs). All runs ended `ok`, and no primary contrast was borderline, so no extension ran.
Protocol: `PHASE1U-1AA-PREREGISTRATION.md`. Code `311d8eb`. Results: `phase1v-results/`,
`../gomea-results/phase1v/` and `../gomea-results/1/phase1v-bool-*/`.

**Answer:**

* **Regression: yes.** With a starting population spread evenly over all distances from the
  exemplar (`--gomea-radii uniform`), GOMEA's test error on Con fell from 37.1 to 29.2.
  * Lower on all 10 splits, Holm p 0.0039 (**better**).
  * The same CPU per run.
* **Boolean: no.** mux11 CPU to solve was a tie.

## Results (medians, 10 pairs each)

| Problem | Metric | GOMEA | GOMEA-UNI | UNI lower / tie / higher | Holm p | Verdict |
|---|---|---|---|---|---|---|
| Con | test NMSE | 37.1 | **29.2** | 10 / 0 / 0 | 0.0039 | **better** |
| mux11 | CPU to solve (s) | 1,471 | 1,692 | 4 / 0 / 6 | 0.63 | tie |

**Descriptive:**

| | GOMEA | GOMEA-UNI |
|---|---|---|
| Con train NMSE | 33.9 | 24.2 |
| Con final program size (nodes) | 187 | 309 |
| Con CPU per run (s) | 2,315 | 2,394 |
| mux11 evaluations to solve | 5,538 | 4,115 (UNI lower in 6 of 10, p 0.63) |
| mux11 CPU per evaluation (s) | 0.25 | 0.35 |
| mux11 final complexity | 36 | 36 |

Con per-split change in test NMSE (UNI − GOMEA), splits 1–10: −16.6, −14.5, −19.4, −8.1,
−3.7, −11.9, −4.2, −10.8, −9.8, −1.7.

## What this shows

1. **On regression, a wider start improves search quality at no extra cost.**
   * Phase 1i saw the uniform start help on frozen maj5 at 6–8× the evaluations. Inside full
     MOSES runs, with a fixed 10,000-evaluation budget, it lowered both train and test error
     on every split, at the same CPU.
   * **Likely mechanism (untested):** an initial population far from the exemplar puts more
     knob settings in play. That gives the first linkage model more variation and lets GOM
     reach distant improvements a narrow start cannot.
2. **The programs are larger** (309 against 187 nodes at the median). Interpretability-sized
   programs may need the acceptance penalty or seeding alongside it.
3. **On mux11 it does not help on time.** It needed fewer evaluations at the median, but they
   cost more, and seed-by-seed it was a tie. Phase 1m also found a wider start useless or
   harmful on mux11 with long demes.

## Limits

* One regression dataset (Con) and 10 splits. The size of the gain makes it unlikely to be
  noise, but EnH and EnC were not tested.
* Not combined with seeding or the penalty.
* **Candidate default for regression:** `--gomea-radii uniform`, pending a check on a second
  dataset.
