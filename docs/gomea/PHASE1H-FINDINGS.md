# Phase 1h findings: does learned linkage work with a bigger sample?

**Status:** complete. Steps 1, 2 and 3 ran 360, 120 and 320 runs, all `ok`. Protocol:
`PHASE1H-PREREGISTRATION.md`. Code `7860f64`. Results: `phase1h-results/step{1,2,3}__*` and
`../gomea-results/1/phase1h-step{1,2,3}-*/` (relative to the repository root, outside it).

**Answer:** yes for LT-B, the bias-corrected linkage tree. With a population of 1,024, LT-B
beats a random tree on frozen maj5: the registered step 3 test gives p = 0.0033 on 160 new
pairs. The effect is small, a mean of 0.13 fewer wrong rows, and comes mostly from two of
the four fixtures.
* **LT-MI**, the uncorrected tree, showed no advantage at N = 1,024 (step 2).
* **At N = 16 to 256** there was no significant difference for LT-B. Step 1's registered
  verdict was MIXED: LT-MI was worse than RT at N = 64 (Holm p 0.047; 0.0625 under the later
  8-contrast correction).

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

## Step 2: N = 1,024 (exploratory)

Same 4 fixtures × 10 seeds, 40 pairs per contrast, 18 runs in parallel. The step took about
10 h (20:21 to 06:20). No run reached the 300,000-evaluation budget: 85 ended on stagnation
and 35 on reaching the target. In 3 runs (one fixture and seed, all three arms) the starting
sample already held a perfect program.

| Arm | Mean error | Median error | Solved (of 40) | Median evaluations | Median wall time (s) |
|---|---|---|---|---|---|
| LT-B | **0.65** | 1 | **16** | 83,130 | 3,704 |
| LT-MI | 0.93 | 1 | 8 | 138,596 | 6,194 |
| RT | 0.85 | 1 | 11 | 105,510 | 4,815 |

| Contrast | A lower / tie / B lower | Mean diff [95 % CI] | p | Holm p (8 contrasts) |
|---|---|---|---|---|
| LT-B vs RT | 9 / 29 / 2 | −0.20 [−0.38, −0.03] | 0.056 | 0.39 |
| LT-MI vs RT | 7 / 23 / 10 | +0.08 [−0.13, +0.28] | 0.63 | 1 |
| LT-B vs LT-MI | 13 / 23 / 4 | −0.28 [−0.50, −0.05] | 0.030 | – |

Mean error / solved runs (of 10) per fixture:

| Fixture | LT-B | LT-MI | RT |
|---|---|---|---|
| maj5-x0 | 0.7 / 3 | 1.1 / 1 | 0.9 / 2 |
| maj5-x1 | 0.3 / 7 | 0.4 / 6 | 0.3 / 7 |
| maj5-x2 | 0.4 / 6 | 1.1 / 1 | 0.9 / 2 |
| maj5-x3 | 1.2 / 0 | 1.1 / 0 | 1.3 / 0 |

**Combined verdict (steps 1 and 2, 8 contrasts in the Holm family): INCONCLUSIVE.** Nothing
is significant after correction; the N = 64 LT-MI result becomes Holm p 0.0625. Step 2 is
exploratory, so this does not replace step 1's pre-registered MIXED verdict; both are
reported.

The 3 perfect-start runs (`maj5-x1`, seed 4008) count as solved in every arm and as a tie in
every contrast.

**What step 2 shows:**
1. **LT-B has led RT since N = 256, and the lead is largest at N = 1,024.**
   * The mean paired difference is +0.00, +0.10, −0.10 and −0.20 rows at N = 16, 64, 256 and
     1,024, so the series is not monotonic: LT-B trailed at N = 64.
   * The steps between Ns are well inside each CI.
   * At N = 1,024 the bootstrap CI excludes zero, but the Wilcoxon p is 0.056 (0.39 after
     Holm), so the lead is not significant.
2. **LT-B solves more runs.** It solved 16 of 40 against 11 (RT) and 8 (LT-MI).
   * Its median evaluations are lower overall (83k against 106k and 139k), partly because
     solved runs stop early.
   * Among unsolved runs LT-B is still the lowest (median 125k against 135k for RT and 147k
     for LT-MI).
   * Among solved runs LT-B used more than RT (73k against 52k).
3. **The bias correction seems to matter at this N.** LT-B beats LT-MI in 13 pairs and loses
   in 4 (p 0.030, uncorrected, descriptive). LT-MI does not beat RT. At N = 256 the solve
   counts went the other way (LT-MI 7, LT-B 5).
4. **The lead comes mostly from x2.** Summed over pairs, LT-B made 8 fewer wrong rows than
   RT: −5 on x2, −2 on x0, −1 on x3 and 0 on x1. On x2 it solved 6 runs against RT's 2.
   x3 was never solved by any arm.

This is the pattern the hypothesis predicts, a learned-linkage advantage that appears only
once the linkage model has a big enough sample, but 40 pairs are not enough to establish it.

## Step 3: replication at N = 1,024 (confirmatory)

* **Design:** LT-B vs RT at N = 1,024, 4 fixtures × seeds 4011–4050 = 160 new pairs. The
  setup is otherwise as in step 2.
* **Registered test:** one two-sided Wilcoxon test on the new pairs only, at α = 0.05.
  * **Confirmed:** p < 0.05 with a negative mean difference.
  * **Refuted:** p ≥ 0.05 with the 95 % CI inside ±0.25 rows.
  * Anything else is inconclusive.
* **Run:** 2026-10-12 06:26 to 2026-10-13 11:50 (about 29 h), 18 workers.

| Arm | Mean error | Median error | Solved (of 160) | Median evaluations | Median wall time (s) |
|---|---|---|---|---|---|
| LT-B | **0.67** | 1 | **63** | 89,185 | 5,050 |
| RT | 0.80 | 1 | 47 | 101,626 | 5,522 |

**LT-B vs RT:**

| Pairs | LT-B lower / tie / RT lower | Mean diff [95 % CI] | p |
|---|---|---|---|
| 160 | 36 / 109 / 15 | −0.13 [−0.21, −0.04] | **0.0033** |

**Verdict: confirmed.** At N = 1,024, LT-B beats the random tree on frozen maj5. The
registered reversal rule (p < 0.05 with a positive mean) did not apply.

**Per fixture** (LT-B lower / tie / RT lower in 40 pairs; solved runs LT-B vs RT):

| Fixture | Pairs | Solved |
|---|---|---|
| maj5-x0 | 16 / 21 / 3 | 21 vs 9 |
| maj5-x1 | 4 / 29 / 7 | 30 vs 33 |
| maj5-x2 | 15 / 21 / 4 | 12 vs 5 |
| maj5-x3 | 1 / 38 / 1 | 0 vs 0 |

**Notes:**
1. **The effect is uneven.** It comes from x0 and x2. On x1 the random tree is slightly ahead,
   and x3 was never solved by either arm. In step 2 the lead came mostly from x2. Here x0
   contributes as much.
2. **It is small.** It amounts to 0.13 wrong rows on average, or 16 more solved runs out of
   160.
3. **Steps 2 and 3 together** (200 pairs at N = 1,024, descriptive): LT-B lower in 45 pairs
   and higher in 17, mean −0.145, p = 0.0003.
4. **Termination:**
   * LT-B: 59 runs reached the target and 101 stagnated.
   * RT: 47 reached the target, 112 stagnated and 1 hit the 300,000-evaluation budget.
   * As in step 2, one fixture and seed had a perfect program in its starting sample, in
     both arms.
   * Four LT-B runs reached zero error without ending on `target_reached`. The solved
     counts above are by final error.
5. **Evaluations:** LT-B's median is lower overall (89k vs 102k), partly because solved runs
   stop early. Among solved runs LT-B used more (76k vs 62k), and among unsolved runs fewer
   (119k vs 126k).

## What Phase 1h shows

* **LT-B beat random linkage only with a large sample.** It made no significant difference
  at N = 16 to 256 and helped at N = 1,024. This is consistent with linkage learning needing
  more data than a deme provides, but it rests on one larger population size and one learned
  model; LT-MI did not beat RT at N = 1,024.
* **MOSES demes normally use N = 16.** With the 250-evaluation cap, demes rarely finish one
  generation (Phase 1e logs: 7.6 % of capped demes reached a second).
  There, the linkage model cannot pay off. This matches Phases 1f and 1g, where learned and
  random linkage tied.
* **For linkage learning to matter in practice,** each model needs far more data than one
  deme provides. The options are a much larger population (expensive), a linkage model
  shared across demes, fewer knobs per deme, or a more diverse starting sample.
* **Limits:** one problem (maj5), frozen representations, and a modest effect driven by two
  of four fixtures.
