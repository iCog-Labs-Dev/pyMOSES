# Phase 1h findings: does learned linkage work with a bigger sample?

**Status:** step 1 complete (360 runs, all `ok`). Step 2 (N = 1,024, exploratory) is
running; this file will be extended when it finishes. Protocol:
`PHASE1H-PREREGISTRATION.md`. Code `7860f64`. Results: `phase1h-results/step1__*` and
`../gomea-results/1/phase1h-step1-2e6f467ecb9c/` (relative to the repository root, outside
it).

## Setup

* **Problem:** maj5, frozen. Phase 1's four maj5 fixtures (`maj5-x0` to `x3`), each a
  single deme on a fixed representation (`-e … -g 1`).
* **Arms:** GOMEA with the bias-corrected linkage tree (LT-B), the plain mutual-information
  linkage tree (LT-MI) or a random tree (RT).
* **Sample size:** the population size N (16, 64 or 256). Each generation's linkage model
  is learned from the N individuals in the population.
* **Budget:** `-m 300000`, no per-deme cap. Every run ended on GOMEA's own criteria
  (stagnation, convergence or reaching the target) well before the budget.
* **Pairing:** 4 fixtures × 10 seeds (4001–4010) = 40 pairs per contrast. The same seed
  gives the same initial population for every FOS.
* **Metric:** the final error, i.e. wrong rows out of 32. Lower is better.

## Results

| N | Arm | Mean error | Median error | Solved (of 40) | Median evaluations | Median wall time (s) |
|---|---|---|---|---|---|---|
| 16 | LT-B | 1.85 | 2 | 0 | 854 | 47 |
| 16 | LT-MI | 1.83 | 2 | 1 | 692 | 37 |
| 16 | RT | 1.85 | 2 | 1 | 1,055 | 48 |
| 64 | LT-B | 1.63 | 2 | 1 | 4,249 | 223 |
| 64 | LT-MI | 1.73 | 2 | 0 | 4,226 | 211 |
| 64 | RT | 1.53 | 2 | 1 | 5,213 | 210 |
| 256 | LT-B | **1.20** | 1 | 5 | 19,284 | 1,034 |
| 256 | LT-MI | 1.25 | 1 | 7 | 24,687 | 1,056 |
| 256 | RT | 1.30 | 1 | 3 | 25,200 | 1,168 |

Wall times are with 14 runs in parallel on 22 cores.

**Paired contrasts.** "A lower / tie / B lower" counts pairs. Mean diff is A − B in wrong
rows, with a 95 % bootstrap CI. Holm correction covers the 6 learned-vs-RT contrasts.

| N | Contrast | A lower / tie / B lower | Mean diff [95 % CI] | p | Holm p |
|---|---|---|---|---|---|
| 16 | LT-B vs RT | 5 / 30 / 5 | +0.00 [−0.15, +0.15] | 1 | 1 |
| 16 | LT-MI vs RT | 4 / 32 / 4 | −0.03 [−0.20, +0.13] | 1 | 1 |
| 64 | LT-B vs RT | 2 / 32 / 6 | +0.10 [−0.03, +0.23] | 0.29 | 1 |
| 64 | LT-MI vs RT | 0 / 32 / 8 | +0.20 [+0.08, +0.33] | 0.0078 | **0.047** |
| 256 | LT-B vs RT | 8 / 28 / 4 | −0.10 [−0.25, +0.08] | 0.39 | 1 |
| 256 | LT-MI vs RT | 7 / 28 / 5 | −0.05 [−0.23, +0.13] | 0.77 | 1 |
| 16 | LT-B vs LT-MI | 3 / 34 / 3 | +0.03 [−0.13, +0.18] | 1 | – |
| 64 | LT-B vs LT-MI | 4 / 35 / 1 | −0.10 [−0.25, +0.03] | 0.31 | – |
| 256 | LT-B vs LT-MI | 7 / 27 / 6 | −0.05 [−0.25, +0.15] | 0.81 | – |

## Verdict under the pre-registered rules

**MIXED.** LT-MI is significantly worse than RT at N = 64. No learned arm is significantly
better than RT at any N.

The MIXED verdict is weak:
* **It rests on one contrast.** In 8 pairs LT-MI ended exactly one row worse than RT, with
  no pair the other way and 32 ties. It clears the threshold only just (Holm p 0.047).
  LT-B is not significantly different from RT anywhere.
* **FALSE's literal condition was also met.** At the largest N (256), nothing is
  significant and both learned-vs-RT CIs lie within ±0.25, LT-B's exactly at the −0.25
  endpoint. The pre-registration does not say which rule wins when both hold.
  `phase1h_report.py` lets MIXED win, because it allows FALSE only when nothing is
  significant at *any* N. That precedence is the script's, not the written protocol's.
* **It depends on the Holm family.** The pre-registration corrects across every step
  together. Once step 2 adds its 2 contrasts, the N = 64 LT-MI contrast gets a Holm p of at
  least 0.0078 × 8 = 0.0625, which is no longer significant.

Read fairly, step 1 shows **no reliable difference between learned and random linkage at
N ≤ 256**, with one marginal result against learned linkage.

## What the data do show

1. **The hypothesis is not supported up to N = 256.** Growing the linkage model's sample from
   16 to 256 did not make learned linkage beat random linkage.
2. **Population size helps every arm about equally.** From N = 16 to 256, the mean error fell
   by 0.65 (LT-B), 0.58 (LT-MI) and 0.55 (RT) rows, and solved runs rose from 0–1 to 3–7.
   Random linkage gained about as much as learned linkage, so the gain looks like it comes
   from the bigger population rather than from better linkage.
3. **The direction flips at N = 256.** At N = 64 both learned arms trail RT; at N = 256 both
   lead it (LT-B 8 wins to 4 losses, mean −0.10). This is the only hint for the hypothesis,
   and it is not significant.
4. **LT-B stops sooner at N = 256.** Its median evaluations were 19,284, against 24,687
   (LT-MI) and 25,200 (RT), for an error that is no worse. Fewer evaluations per run can mean
   faster convergence or earlier stagnation; these data do not separate the two.
5. **The size of any effect is bounded but not zero.** At N = 256 the CIs are [−0.25, +0.08]
   (LT-B vs RT) and [−0.23, +0.13] (LT-MI vs RT). LT-B's CI touches the −0.25 margin, so a
   learned-linkage advantage of a quarter of a row cannot be ruled out. A disadvantage larger
   than about 0.13 rows can be.
6. **No fixture shows a consistent learned-linkage advantage.** The per-fixture means move
   by at most 0.4 rows between arms, in both directions.

## Limits

* **One problem.** maj5 was chosen because Phase 1 showed no linkage effect on it, so it may
  simply have little structure for a linkage model to find.
* **Generations are few.** Runs stopped after a median of 4–6 generations at every N, so each
  run learned only a handful of linkage models.
* **The sampler start limits diversity.** Every individual starts in the exemplar's
  neighbourhood. The uniform start, which would give more diverse samples, was too slow to
  use (pilot: no generation in 10 min at N = 256).

## Next

Step 2 (N = 1,024, same 40 pairs) tests whether the N = 256 lead grows with a larger sample.
It was registered as exploratory in `PHASE1H-PREREGISTRATION.md` before it ran. Step 1's
verdict above uses step 1's 6 contrasts. The combined analysis will report the Holm values
over all 8 contrasts as well.

At N = 1,024 the sampler start sometimes already contains a perfect program, for example
`maj5-x1`, seed 4008. Every arm shares that starting population, so such pairs tie
regardless of linkage.
