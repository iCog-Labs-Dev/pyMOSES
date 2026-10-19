# GOMEA in pyMoses: findings log

One row per pre-registered phase: the question, the registered verdict and the key numbers.
The per-phase `PHASE1x-FINDINGS.md` files have the details, caveats and deviations; where
this log and a findings file differ, the findings file is right.

* **GOMEA:** the GOMEA optimizer embedded in MOSES as the per-deme search, with LT-B linkage
  unless noted.
* **HC:** MOSES hill climbing. **Capped HC** (`-a hc-capped`) has GOMEA's per-deme cap; it
  has been the primary comparison since 2026-10-15.
* **Problems:**
  * mux11, maj5/7 and pa4/5 are Boolean.
  * Con, EnH, EnC, Air, Bos and Yac are regression datasets (Virgolin et al. 2021), scored
    by test NMSE.
* **Verdicts** follow each phase's registered rule: Holm-corrected, and "better" also needs a
  10 % lower median.

## Standing conclusions (as of 2026-10-19)

1. **Regression: GOMEA beats HC, including at equal time.**
   * Shown in Phases 1f, 1g, 1j, 1n and 1p.
   * Con at equal time, same cap: 43.2 against 68.4 (1p).
2. **Boolean: GOMEA needs fewer evaluations but loses on time.**
   * mux11 evaluations to solve: 4,633 against 7,978 for capped HC (1n).
   * At equal time: about 2× slower than capped HC (1p) and about 7× slower than uncapped HC
     (1l: 2,010 s against 278 s).
3. **The cost is per evaluation, mostly knob building.**
   * GOMEA costs about 2.75–3.5× capped HC's CPU per evaluation (1n, 1p).
   * Knob building is about 68 % of GOMEA's CPU, and it grows with exemplar size.
   * GOMEA's exemplars grow about 2× faster than HC's.
4. **Short demes matter, and so does moving knobs in groups.**
   * A per-deme cap (250–500) turned GOMEA from worse than HC into competitive (1d, 1e).
   * GOMEA without linkage groups did not beat HC (1g).
   * *Learned* linkage beats a random tree only with large samples on a frozen problem
     (N = 1,024, 1h). Inside MOSES's short demes it rarely matters (1g, 1m).
5. **Seeding MOSES with GP-GOMEA elites is the biggest lever found so far.**
   * It lowered test error for capped HC on Con and EnH, and for GOMEA on EnH (1r).
   * It cut GOMEA's CPU per run about 4–6×.
   * The seeds do most of the work.
6. **Smaller exemplars are what cut cost:** seeding (1r) and the acceptance penalty (1o, 1p).
   Fixes that don't shrink exemplars did not change cost (1q).

## Phase register

| Phase | Date | Question | Verdict and key numbers |
|---|---|---|---|
| 1 | 2026-09-26 | Does GOMEA inside MOSES beat HC at equal evaluations (maj5, mux6, pa4)? | **No; worse on all three.** E.g. mux6 median evaluations to solve 1,627 for HC against 4,829 for LT-B. No linkage (UNI) beat LT-B on maj5 and mux6. |
| 1b | 2026-09 | Does stand-alone GP-GOMEA (C++) beat MOSES HC at equal evaluations? | **No, on any task.** With 10× the budget it solves maj5 and mux6, at about 3× and 20× HC's evaluations. |
| 1c | 2026-09 | How do GP-GOMEA and classic C++ MOSES scale on Boolean families? | **Mixed.** GP-GOMEA scales better on majority, worse on multiplexer (mux11: about 359k evaluations against 8.8k for MOSES) and on parity. |
| 1d | 2026-10 | Which GOMEA change closes the gap to HC? | **Capping each deme (500 evaluations).** That alone moved maj5 from worse to a tie (20/30 each) and nearly halved mux6's evaluations. A larger population, uniform start and strict acceptance did not help. |
| 1e | 2026-10 | On larger Boolean tasks, does capped GOMEA beat HC? | **Better on mux11 in evaluations:** 4,633 against 8,757 (Holm p 0.0063), the first "better". Ties on maj7 and pa5. About 3–4× HC's CPU. |
| 1f | 2026-10 | Regression: does the paper replicate, and does pyMoses GOMEA beat HC? | **A:** paper levels replicate, the pattern only partly. **B:** stand-alone C++ GP-GOMEA beats C++ MOSES on all 10 datasets. **D:** GOMEA **better** than HC on Air, Con, EnC and EnH, ties on Bos and Yac, at 2.3–5.4× HC's CPU. |
| 1g | 2026-10-11 | Is the regression gain from short demes or from linkage? | **Neither alone.** Capping HC barely changed its error. GOMEA without linkage groups (UNI) did not beat HC. LT-B was the only arm to beat HC; random trees did as well in 1f, so grouping rather than *learned* grouping seems to matter. |
| 1h | 2026-10-12 | Does a bigger linkage sample make learned linkage pay off (frozen maj5)? | **Yes, but only at N = 1,024:** LT-B beat a random tree, p 0.0033, a small effect (0.13 fewer wrong rows). No difference at N = 16–256. |
| 1i | 2026-10-13 | Does a uniform-distance starting population help learned linkage? | **Yes, at a cost:** lower error (LT-B mean 0.89 against 1.30) and a wider LT-B lead, at about 6–8× the evaluations. |
| 1j | 2026-10-13 | GOMEA vs uncapped HC at equal time? | **Con better** (43.8 against 54.0). **mux11 worse:** 2,498 s against 615 s to solve, 7/10 against 10/10 solved. |
| 1k | 2026-10-15 | Do fewer pair knobs make learned linkage matter? | **No.** Both arms got worse; no interaction. |
| 1l | 2026-10-15 | After the speed-ups, does GOMEA still lose mux11 at equal time? | **Yes, worse:** 2,010 s against 278 s. The speed-ups helped HC more (2.3× against 1.2×). Results byte-identical. |
| 1m | 2026-10-15 | Budget-derived demes (cap = budget/10, N up to 200)? | **Regression: tie** at 37–64 % less CPU. **mux11: much worse** (about 5.6× the evaluations). Learned and random linkage tie. A wider start did not help. |
| 1n | 2026-10-15 | GOMEA vs capped HC, same cap 250, equal evaluations? | **GOMEA better on both:** Con 37.1 against 60.4; mux11 4,633 against 7,978 evaluations (20 seeds). About 2.75× CPU per evaluation. |
| 1o | 2026-10-16 | Complexity penalty inside GOMEA's acceptance (PEN), equal evaluations? | **Con CPU −37 % (better).** Con accuracy a registered tie, but descriptively worse in 15 of 20 splits. mux11: ties. |
| 1p | 2026-10-17 | GOMEA vs capped HC at equal time, same cap? | **Con better** (43.2 against 68.4, 16/20). **mux11 worse** (2,568 s against 1,311 s, 15/20). PEN (exploratory) ties on both. |
| 1q | 2026-10-18 | Does fixing knob collapse (`--knob-probe-baseline reduced`) help GOMEA? | **Tie on accuracy and CPU** (Holm p 1.0), although collapsed demes fell to 0 (35 → 0 mux11, 367 → 0 pa5). Kept opt-in. |
| 1r | 2026-10-18 | Seed MOSES with GP-GOMEA elites at the same total budget? | **Capped HC better** on Con (70.2 → 42.6) and EnH (28.9 → 8.1). **GOMEA better** on EnH (15.7 → 8.5) and a **tie** on Con (35.2 → 42.3, worse in 14/20). GOMEA CPU per run about 4–6× lower. |

## Descriptive comparisons (not pre-registered)

| Date | Comparison | Result |
|---|---|---|
| 2026-10-19 | PEN (1p) against unpenalized GOMEA (1p) and uncapped HC (1j, 1l), equal time, 10 pairs | **Con:** PEN 43.3, GOMEA 45.4, uncapped HC 54.0. PEN beats uncapped HC in 9/10 (p 0.020). **mux11:** PEN 1,800 s (10/10 solved), GOMEA 2,568 s (8/10), uncapped HC 278 s. PEN is faster than GOMEA in 8/10 (p 0.11) but about 6.5× slower than uncapped HC. Uncorrected p. |
| 2026-10-19 | GOMEA CPU, seeded against unseeded (1r) | Seeded cheaper in every pair: Con 20/20, median about 4×; EnH 10/10, about 6.3×. GOMEA/HC CPU ratio fell from 4.6–5.2× to 1.65–1.9×. |

## Open questions (see `BACKLOG.md`)

* Does seeding close the Boolean time gap? Seeded GOMEA on mux11 at equal time (item 14).
* Does PEN beat unpenalized GOMEA on mux11 at equal time, under a registered test?
* Cheaper knob building (items 1.x).
