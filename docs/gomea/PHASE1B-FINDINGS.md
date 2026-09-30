# Phase 1b findings: stand-alone GP-GOMEA vs MOSES hill climbing

**Status: complete.** There were 160 pre-registered confirmatory GP-GOMEA runs and 60
descriptive runs at a 200k budget, all `ok`. They are compared with the 80 Phase 1 MOSES
hill-climbing runs.

## Implementation and comparison

* **GP-GOMEA:** the authors' C++ code, GP-GOMEA `750220c`, built out of tree against
  Armadillo 14.0.2 (`tools/gomea/build_gpgomea.sh`, with a pinned sha256). Its search is
  unmodified; only one log field was added to report the evaluation count
  (`phase1b-results/instrumentation.diff`).
* **Baseline:** stock MOSES `-a hc` (`c6301cb`), the Phase 1 confirmation runs.
* **Tasks:** identical truth tables, generated from MOSES's own target functions. Both
  systems use AND/OR/NOT and no constants.
* **Budget:** 20,000 evaluations per run. GP-GOMEA is read at its last generation boundary
  within the budget, which is conservative against it.
* **Tests:** unpaired Fisher and Mann–Whitney with Holm correction, per
  `PHASE1B-PREREGISTRATION.md`.

| Task | MOSES HC | GP-GOMEA tuned (N=64, h=5) | GP-GOMEA defaults (N=500, h=4) | Tuned vs HC (pre-registered rule) | Tuned at 200k (descriptive) |
|---|---|---|---|---|---|
| maj5 | 20/30 solved | 11/30 | 0/30 | **worse** (evals p_holm 0.02; −30 pp) | 20/20; 14/20 by 30k |
| mux6 | 30/30; median evals-to-solve 1,627 | 2/30 | 0/30 | **worse** (p_holm ≤ 1e-10) | 20/20; 9/20 by 30k |
| pa4 | 2/20; median error 2 | 0/20; median error 3 | 0/20; median error 5 | **worse** (error p_holm 0.048) | 3/20; median 150k when solved |

## What the evidence supports

**At equal evaluations, stand-alone GP-GOMEA did not beat MOSES hill climbing on any
task.**

* **Defaults.** With population 500, the first generation alone costs 14–17k evaluations,
  so only one generation fits in the budget. The defaults solved 0 of 80 runs.
* **Tuned setting.** It narrows the gap on maj5 (11 vs 20 of 30) and loses badly on mux6.
  There, HC's median evals-to-solve is 1,627, while GP-GOMEA's earliest success was at
  12,402. It solved only seeds 1007 and 1017.
* **More budget.** With ten times the budget, GP-GOMEA solves maj5 and mux6 on every seed.
  Its medians are 27.7k and 33.1k evaluations, about 3× and 20× the median of HC's
  successful runs. The slowest runs needed 94.7k and 114.1k.

**Against GOMEA inside MOSES (Phase 1):**

* against MOSES+UNI, stand-alone GP-GOMEA is *worse* on maj5 and mux6 and *ties* on pa4;
* against MOSES+LT-B, it *ties* on maj5 and pa4 and is *worse* on mux6;
* on pa4 its error is lower than LT-B's (3 vs 4, p_holm 0.027), but that is a *tie* under the
  practical rule.

## Strengths, weaknesses and costs

**Strength: CPU cost.** A GP-GOMEA run took a median of 0.32 CPU-seconds, including its
budget overshoot (a median of 30.4k evaluations executed). Per-task medians for pyMoses HC
were 27–767 s. *Interpretation:* this is C++ against pure Python, not a measured
algorithmic advantage. No equal-wall-time runs were made.

**Weaknesses:**

* **Sample efficiency.** It is poor on these tasks; the logs show IMS doubling the
  population.
* **Sensitivity.** Results depend strongly on population and template height. In the pilot
  P64-H5 solved 6/15, P64-H6 4/15, and P64-H4 and P500-H4 0/15.
* **Program size.** Median final-program size for the tuned arm is 44.5–51 nodes,
  unsimplified and measured after the overshoot. HC's median reduced complexity is 11–12.
  These are different measures.
* **Tuning favoured GP-GOMEA.** It had a pilot; HC had none.

*Interpretation:* the mux6 gap plausibly reflects MOSES's exemplar-driven representation
against GP-GOMEA's fixed-height templates.

## Decision

*Superseded in part by Phases 1d–1f; see [`PHASE1E-FINDINGS.md`](PHASE1E-FINDINGS.md) and
[`PHASE1F-FINDINGS.md`](PHASE1F-FINDINGS.md).*

At equal evaluations, **MOSES hill climbing remains the stronger method** on these Boolean
tasks. Next: an equal-wall-time comparison against classic C++ MOSES, which removes the
language confound.

Evidence: `phase1b-results/` (CSVs, `findings.json`, checksums, figures, pilot) and
`../gomea-results/phase1b/`. Reproduce with `tools/gomea/build_gpgomea.sh`, then
`gpgomea_runner.py` and `phase1b_report.py`.
