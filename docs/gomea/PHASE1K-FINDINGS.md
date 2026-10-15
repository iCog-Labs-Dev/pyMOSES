# Phase 1k findings: fewer pair knobs and learned vs random linkage

**Status:** complete. Stage 1 ran on 2026-10-15 from 04:59 to 05:46 (160 runs, all `ok`).
Neither primary contrast was borderline, so no extension ran. Protocol:
`PHASE1K-PREREGISTRATION.md`. Code `d875294`. Results: `phase1k-results/` and
`../gomea-results/1/phase1k-stage1-*/` (relative to the repository root, outside it).

**Answer:** no. Cutting the pair knobs from 5 to 1 per and/or node (about 40 % fewer knobs)
did not change how learned linkage (LT-B) compares with a random tree (RT), at N = 64 or
256. It made both arms worse.

## Setup

* **Problem:** frozen maj5, fixtures `maj5-x0` to `x3`.
* **Arms:**
  * LT-B and RT with `--logical-perm-ratio -0.75` (one pair knob per node), at N = 64 and 256.
  * The default-knob arms are reused from Phase 1h step 1.
* **Budget and pairs:** `-m 300000`; seeds 4001–4010 per fixture, 40 pairs per N.

## Results

| Arm | Mean error | Solved (of 40) |
|---|---|---|
| LT-B, reduced knobs, N = 64 | 1.95 | 0 |
| RT, reduced knobs, N = 64 | 1.88 | 0 |
| LT-B, default knobs, N = 64 | 1.63 | 1 |
| RT, default knobs, N = 64 | 1.53 | 1 |
| LT-B, reduced knobs, N = 256 | 1.68 | 2 |
| RT, reduced knobs, N = 256 | 1.63 | 1 |
| LT-B, default knobs, N = 256 | 1.20 | 5 |
| RT, default knobs, N = 256 | 1.30 | 3 |

**Primary contrast, `interaction`:** (LT-B − RT) with reduced knobs minus (LT-B − RT) with
default knobs, Holm-corrected over the two N.

| N | Negative / zero / positive | Mean | p | Holm p | Verdict |
|---|---|---|---|---|---|
| 64 | 7 / 28 / 5 | −0.03 | 1.00 | 1.00 | no difference |
| 256 | 8 / 21 / 11 | +0.15 | 0.24 | 0.47 | no difference |

**Secondary measures (uncorrected):**

| Measure | N | Reduced knobs | Default knobs |
|---|---|---|---|
| LT-B − RT error (lower / tie / higher) | 64 | 2 / 35 / 3, p 0.44 | 2 / 32 / 6, p 0.29 |
| LT-B − RT error (lower / tie / higher) | 256 | 5 / 29 / 6, p 0.79 | 8 / 28 / 4, p 0.39 |
| Improvement rate, LT-B higher than RT (pairs of 40) | 64 | 37, p < 10⁻⁶ | 37, p < 10⁻⁶ |
| Improvement rate, LT-B higher than RT (pairs of 40) | 256 | 38, p < 10⁻⁶ | 38, p < 10⁻⁷ |

## What this shows

1. **Fewer knobs did not help the linkage model.**
   * The learned-vs-random gap is unchanged at both population sizes.
   * About two-thirds fewer knob pairs to estimate (65–67 %) did not help LT-B.
2. **Both arms got worse with fewer pair knobs** (exploratory: not a registered contrast).
   * Paired, uncorrected, reduced minus default knobs:
     * LT-B N = 64: +0.33, p 0.002;
     * RT N = 64: +0.35, p 0.003;
     * LT-B N = 256: +0.48, p 0.001;
     * RT N = 256: +0.33, p 0.013.
   * At N = 256, LT-B's solved runs fell from 5 to 2.
   * The reduced-knob runs also stopped much earlier. Median evaluations were 2,458 / 3,002
     (LT-B / RT) at N = 64 and 12,073 / 15,566 at N = 256, against 4,249 / 5,213 and
     19,284 / 25,200 with default knobs.
   * At N = 64, 13 of 40 runs per reduced arm ended on convergence, against 3–4 with default
     knobs.
   * One explanation (a hypothesis, not tested): pair knobs are how a deme adds new
     two-input subtrees, so with one per node fewer useful edits exist and the population
     converges sooner.
3. **The improvement-rate gap is unchanged.** LT-B's knob-group copies improve the program
   more often than RT's in 37–38 of 40 pairs, with either knob set.
   * The mean gap was 0.018 against 0.014 at N = 64 and 0.014 against 0.015 at N = 256;
     these were not tested.
   * Absolute rates rose with reduced knobs, e.g. LT-B 0.064 → 0.094 at N = 64.
   * The learned groups are better building blocks per trial in both settings, but that does
     not translate into lower final error here.
4. **Together with Phase 1i:** a more diverse sample made learned linkage pay off (at about
   6× the evaluations); one random cut of pair knobs did not. This points to the sample
   rather than the knob count as the limit, but both phases test one setting each.

## Limits

* One problem (maj5), frozen representations, one reduction level (−0.75), and 10 seeds per
  fixture.
* The reduction removes pair knobs at random. A smarter reduction, such as feature
  selection or stronger simplification during knob building, is untested; see `BACKLOG.md`.
