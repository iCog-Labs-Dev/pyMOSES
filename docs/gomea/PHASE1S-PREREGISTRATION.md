# Phase 1s pre-registration: does the acceptance penalty make GOMEA faster on mux11?

Written 2026-10-19, before any Phase 1s run.

**Requested by the user:** "a short but conclusive experiment … to settle the problem of the
penalization on GOMEA", designed for quick results and not run on 20 new seeds.

## Question

On mux11, does PEN (`--gomea-accept-complexity-ratio 0.125`) reach the solution with less
CPU time than plain GOMEA, at the same per-deme cap?

## What is already known

* **Con:** settled.
  * At equal time PEN tied plain GOMEA (Phase 1p, 20 splits, 10 / 10).
  * At equal evaluations it cut CPU by 37 %, probably at some accuracy cost (Phase 1o).
* **mux11:** open.
  * Phase 1o, equal evaluations, 10 seeds: CPU tie (7 / 3), Holm p 0.39.
  * Phase 1p, equal wall time, 20 seeds: PEN faster in 15 of 20, an exploratory arm, Holm
    p 0.098.

## Design: quick, and conclusive because the sample is pooled

**Metric:** CPU seconds to reach the solution (`cpu_to_target`); a run that never solves counts
as +∞.

* CPU time is used rather than wall time because the machine is shared with other jobs.
  Phase 1q showed wall-time caps fail under load; CPU time is much less sensitive. The arms
  interleave, so both see the same conditions.
* The budget is `-m 40000`, which every mux11 run solved within in Phases 1n, 1o and 1q.

**Arms:** `LTB-C250-M` and `PEN`, exactly as in Phase 1o (LT-B, N = 16, cap 250, memo, plus
R = 0.125 for PEN).

**New runs:** seeds 3021–3030, 10 seeds × 2 arms = 20 runs, with 8 workers.

**Analysis** (`tools/gomea/phase1s_report.py`):

* **Primary: pooled.** One paired Wilcoxon test over 20 seeds: the 10 new ones (3021–3030)
  plus Phase 1o's 10 (3001–3010).
  * Phase 1o used the same arms, settings, budget and metric.
  * It was pre-registered and fully reported before this phase was designed.
  * Its pairs ran under one set of conditions per pair, as here.
* **Verdict:** a single look at α = 0.025.
  * **PEN better:** p < 0.025 and PEN's median at least 10 % lower.
  * **PEN worse:** the mirror image.
  * Otherwise a **tie**.
  * No extension.
* **Replication check:** the 10 new seeds must agree with the pooled direction (more pairs
  favouring the winning arm). If they do not, the result is reported as "pooled better, not
  replicated" and PEN is not adopted.
* **Descriptive:** solved count, evaluations to solve, CPU per evaluation, demes per run,
  best complexity, for the new seeds and for Phase 1o separately.

**Why this is conclusive with only 10 new seeds:**

* The pooled test has 20 pairs.
* With 20 pairs, an effect like Phase 1p's (15 of 20 pairs) reaches p ≈ 0.03 to 0.01,
  depending on the sizes of the differences.
* A clear tie is also a usable answer: PEN would then be a cost option only.

**Reading:**

| Result | Decision |
|---|---|
| PEN better and replicated | PEN becomes the recommended GOMEA setting for Boolean problems. On regression it is neutral at equal time. |
| Tie | PEN does not make GOMEA faster to solve; it remains an opt-in CPU saver. |
| PEN worse | Drop PEN for Boolean problems. |

**Related:** Phase 1t asks whether the same penalty also speeds up capped hill climbing.
