# Phase 1i findings: a starting population spread over all distances

**Status:** complete. Stage 1 ran 2026-10-13 15:08 to 2026-10-14 11:23 (80 runs). One primary
contrast was borderline, so the registered extension ran until 2026-10-15 04:59 (160 runs).
All runs ended `ok`, as did the identity check. The first launch (20 workers) was stopped
about 15 s in and restarted with 12. The 21 runs it had begun (20 stage-1 uniform runs and
the identity check) restarted from scratch before their first generation, so their logs show
two starts but a single, clean generation sequence. Results are unaffected; the identity
check still matches. Protocol: `PHASE1I-PREREGISTRATION.md`.
Code `d875294`. Results: `phase1i-results/` and `../gomea-results/1/phase1i-*/` (relative to
the repository root, outside it).

**Answer:** yes, on both questions. With a starting population spread evenly over all
distances from the exemplar (`--gomea-radii uniform`), LT-B reaches lower error, and LT-B's
lead over a random tree (RT) widens. **The catch is cost:** most uniform-start runs ran
until the 150,000-evaluation budget (49 of 80 LT-B, 57 of 80 RT). The uniform start used
about 6–8× the evaluations of the default start: 7.7× (LT-B) and 6.2× (RT) by median, 6.2×
and 5.4× by mean.

## Setup

* **Problem:** frozen maj5, the Phase 1h fixtures `maj5-x0` to `x3`, at N = 256.
* **Arms:**
  * LT-B and RT with the uniform start (new runs);
  * LT-B and RT with the default start (radius mixture `1:0.4,2:0.3,4:0.2,max:0.1`). Seeds
    4001–4010 are reused from Phase 1h step 1. Seeds 4011–4020 ran in the extension.
* **Budget:** `-m 150000`.
* **Pairs:** 80, as 4 fixtures × seeds 4001–4020.
* **Workers:** 12. Twenty were planned but cut at launch for lack of free memory.
* **Identity check:** the default start under the Phase 1i budget reproduced step 1 exactly
  for maj5-x0, LT-B, seed 4001 (error 2, 18,056 evaluations). Reusing step 1's runs is
  therefore valid.

## Results (stage 2, all 80 pairs)

| Arm | Mean error | Solved (of 80) | Median evaluations | Median wall time (s) |
|---|---|---|---|---|
| LT-B, default start | 1.30 | 7 | 19,523 | 723 |
| **LT-B, uniform start** | **0.89** | **15** | 150,000 | 8,770 |
| RT, default start | 1.25 | 8 | 24,355 | 815 |
| RT, uniform start | 1.14 | 3 | 150,000 | 10,255 |

**Primary contrasts.** These are paired Wilcoxon tests with Holm correction over the two;
each look is decided at α = 0.025.

| Contrast | Negative / zero / positive | Mean | Holm p | Verdict |
|---|---|---|---|---|
| `start`: LT-B uniform − LT-B default | 38 / 32 / 10 | −0.41 | 0.00007 | **uniform better** |
| `interaction`: (LT-B − RT) uniform − (LT-B − RT) default | 28 / 40 / 12 | −0.30 | 0.0071 | **uniform better** |

At stage 1 (40 pairs), `start` was already decided (Holm p 0.021). `interaction` was
borderline (Holm p 0.17), so it was extended; with 80 pairs it is decided.

**Descriptive contrasts** (80 pairs, uncorrected p):

| Contrast | Negative / zero / positive | Mean | p |
|---|---|---|---|
| LT-B − RT, uniform start | 23 / 54 / 3 | −0.25 | 0.00009 |
| LT-B − RT, default start | 14 / 50 / 16 | +0.05 | 0.51 |
| RT uniform − RT default | 21 / 46 / 13 | −0.11 | 0.14 |

**Per fixture.** Each fixture has 20 pairs. "Better / tie / worse" is the uniform start
relative to the default. Solved counts are given as uniform start vs default.

| Fixture | `start` | `interaction` | Solved, LT-B | Solved, RT |
|---|---|---|---|---|
| maj5-x0 | 13 / 6 / 1 | 10 / 8 / 2 | 6 vs 1 | 2 vs 1 |
| maj5-x1 | 7 / 8 / 5 | 7 / 9 / 4 | 7 vs 5 | 1 vs 6 |
| maj5-x2 | 9 / 7 / 4 | 6 / 10 / 4 | 2 vs 1 | 0 vs 1 |
| maj5-x3 | 9 / 11 / 0 | 5 / 13 / 2 | 0 vs 0 | 0 vs 0 |

Both effects point the same way on every fixture.

## What this shows

1. **With a diverse start, learned linkage pays off at N = 256.**
   * With the default start (70 % of individuals 1–2 knobs from the exemplar), LT-B and RT do
     not differ at N = 256: LT-B was lower in 14 pairs and higher in 16 here (Phase 1h step 1:
     8 and 4, p 0.39).
   * With the uniform start, LT-B is lower in 23 pairs and higher in 3.
   * Phase 1h saw a smaller lead (−0.13 rows) only at N = 1,024 with the default start, at a
     comparable evaluation count (median 89–102k). That comparison is across phases and was
     not registered.
   * This suggests that sample diversity, not only sample size, limits the linkage model. It
     is not proven: the uniform arms also had about 6× the evaluations.
2. **LT-B gains more from the diverse start than RT does.** The registered `interaction`
   test supports this.
   * RT did not improve significantly with the uniform start (p 0.14). It solved fewer runs
     (3 against 8), almost entirely on `maj5-x1` (1 against 6).
   * LT-B improved strongly. That is consistent with LT-B learning from the more varied
     sample, but more evaluations also contributed.
3. **It is expensive.**
   * The uniform-start runs ran until the 150,000-evaluation budget: 49 of 80 LT-B runs and
     57 of 80 RT runs ended on the budget.
   * Per generation, the uniform start costs about 37,000 evaluations (150,000 over a median
     of 4 generations), against about 3,000–4,000 for the default start (median 6
     generations).
   * Median wall time was about 12× that of the default start, but the default-start times
     come partly from Phase 1h (different day and load). Within the extension alone, where
     both starts ran together, the ratio is about 15×.
   * This phase therefore shows the effect at an unequal cost. Whether it pays off at equal
     evaluations is not tested.

## Limits

* One problem (maj5), frozen representations and one population size.
* Unequal evaluation budgets, as above.
* `maj5-x3` was never solved by any arm. Only `maj5-x0` shows large solved-count gains for
  LT-B, and RT's solved-count drop comes almost entirely from `maj5-x1`.
* The manifests (and their archived copies) still say 20 workers. The runs used 12, as
  `environment.json` records.

## Next

* **Equal-cost test:** give the default start the same 150,000 evaluations (e.g. N ≈ 2,048,
  or restarts), or cap the uniform start at the default's evaluations. This asks whether
  the uniform start is a better use of the budget.
* **Inside full MOSES runs:** test whether the uniform start helps end-to-end, where demes
  are short and capped. Its per-generation cost may be prohibitive there.
* Both are listed in `BACKLOG.md`.
