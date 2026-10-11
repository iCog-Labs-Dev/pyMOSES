# Phase 1h pre-registration: does learned linkage work with a bigger sample?

Written 2026-10-11, before any Phase 1h run. Each later step is added here before it runs.

**Requested by the user:** "increase the number of candidates the linkage learning gets so
we can see it in action. For this just use one problem from our history that didn't show
significant difference between random, linkage and biased linkage tree and reengineer
somethings so each linkage learning gets bigger sample size ... Repeat the process until it
is conclusive if either the thing is true or not is correct."

## Hypothesis

Learned linkage did not beat a random FOS inside pyMoses because each linkage model was
learned from too few individuals (16). **H:** with a larger sample per linkage model,
learned linkage (LT-B or LT-MI) beats the random tree (RT).

## Problem and setup

* **Problem: maj5, frozen.** These are Phase 1's four frozen maj5 fixtures (`maj5-x0` to
  `x3`, from `configs/gomea/phase1/confirm-frozen.json`). Each run is a single deme on the
  fixed representation around the fixture's exemplar (`-e … -g 1`, 119 knobs for x0).
  * **Why maj5:** in Phase 1, LT-B, LT-MI and RT tied there, both frozen and end to end.
  * **Why frozen:** a single deme isolates linkage learning from MOSES's deme rebuilding.
* **How the sample grows.** Each generation's linkage model is learned from the population,
  so the **population size N** is the sample.
  * There is no per-deme cap; the budget is `-m 300000`.
  * A deme ends on GOMEA's own criteria: stagnation, convergence or the budget.
  * The default sampler start is kept. A pilot showed that the uniform start is too slow:
    no generation completed in 10 min at N = 256.
  * Distinct individuals do grow with N: about 16, 92 and 442 at N = 64, 256 and 1,024.
* **Arms:** `--gomea-fos` lt-b, lt-mi or rt, at each N, with `--gomea-skip-inactive 0` and
  everything else at its default.
* **Pairing:** by (fixture, seed). The same seed gives the same initial population for
  every FOS.
* **Code:** the code at `7860f64`, run with `benchmarks/gomea/runner.py`.

## Escalation ladder

1. **Step 1:** N = 16, 64 and 256; 4 fixtures × 10 seeds (4001–4010) = 40 pairs per
   contrast. Estimated at about 1 h.
2. **Step 2, if step 1 is inconclusive:** N = 1,024, the same 40 pairs. Estimated at about
   5 h.
3. **Step 3, if still inconclusive:** 40 more seeds (4011–4050) at the N with the largest
   learned-vs-RT gap.
4. **Stop after step 3** and report "inconclusive with these resources" if no rule below
   has fired.

## Analysis (`tools/gomea/phase1h_report.py`)

* **Metric:** final error, i.e. wrong rows out of 32. A run that does not end `ok` counts
  as 32.
* **Contrasts, per N:** LT-B vs RT and LT-MI vs RT (learned vs random), and LT-B vs LT-MI
  (the bias correction, descriptive).
* **Tests:** a paired Wilcoxon signed-rank test, with Holm correction across *all*
  learned-vs-RT contrasts tested so far, every N and every step together. The report also
  gives the mean paired difference with a 95 % bootstrap CI.

**Conclusive rules.**
* **TRUE:** a learned arm beats RT (Holm p < 0.05, negative mean difference) at some N, and
  no learned arm is significantly worse than RT anywhere.
* **FALSE:** at the largest N tested, nothing is significant, and the 95 % CI of the mean
  paired difference lies within ±0.25 wrong rows for both learned arms.
* **MIXED:** a learned arm is significantly *worse* than RT somewhere.
* **Otherwise:** INCONCLUSIVE, so escalate.

Descriptive: errors, success counts, evaluations used and wall time per arm and N; the
trend of the gap with N.

## Step 1 outcome and step 2 (added 2026-10-11, before step 2 ran)

**Step 1 verdict, by the rules above: MIXED.** LT-MI was worse than RT at N = 64 (8 losses,
0 wins, 32 ties; Holm p = 0.047). Nothing else was significant. At N = 256 both learned arms
led slightly but not significantly. See `PHASE1H-FINDINGS.md`.

MIXED is one of the conclusive rules, so the ladder above would stop here. The user asked to
continue testing, so **step 2 runs as an exploratory follow-up.** It cannot overturn the
step 1 verdict, and it is reported separately.

* **Step 2:** N = 1,024, arms LT-B, LT-MI and RT, the same 4 fixtures × 10 seeds (4001–4010)
  = 40 pairs. Manifest `configs/gomea/phase1h/step2.json`, code `7860f64`, `-m 300000`.
* **Deviations from step 1's manifest:**
  * `wall_cap_s` is 43,200 instead of 7,200. At step 1's per-process speed (about 23
    evaluations/s with 14 workers), a 100k–300k-evaluation run needs 1–4 h.
  * 18 workers instead of 14. The peak RSS at N = 256 was 96 MB, so memory is not the limit.
* **Estimate:** roughly 10–20 h, depending on how many evaluations N = 1,024 uses before it
  stagnates.
* **Analysis:** `tools/gomea/phase1h_report.py` over both result directories. Holm correction
  runs over all 8 learned-vs-RT contrasts (N = 16, 64, 256 and 1,024).
* **Exploratory reading:**
  * At N = 1,024, does a learned arm beat RT (Holm p < 0.05, negative mean difference)?
  * Is the N = 1,024 gap in the same direction as the N = 256 lead, and is it larger?
* **Step 3, if step 2 shows a learned-arm lead that is not significant:** 40 more seeds
  (4011–4050) at the N with the largest learned-vs-RT gap. It is registered here before
  running.
