# Phase 1 findings

**Status: complete.** 1,140 pre-registered confirmatory runs; all `ok`, with 0 timeouts,
0 errors and 0 memory warnings. They ran locally on 2026-09-26 with 14 workers.

## Implementation and comparison

* **Implementation commit:** `c6301cb`. The runs were checked against it, and later commits
  touch only docs and tools.
* **Matched HC baseline:** stock `-a hc` at the same commit. It needed no repairs and is
  byte-identical to classic MOSES.
* **Representation mode:** the MOSES logical discrete knobs of one frozen representation
  per deme.
* **Reused components:** GP-GOMEA `750220c` (Apache-2.0), ported narrowly with golden
  parity: entropy/MI/biased similarity, the UPGMA/RNN linkage tree, univariate and random
  FOS, and GOM (`PROVENANCE.md`).
* **Material deviations from the papers:**
  * a MOSES-sampler initial population instead of random trees;
  * MOSES's composite comparator;
  * forced improvement is a labelled reconstruction;
  * N = 16, chosen in the pilot.
* **Manifests and protocol:** `confirm-e2e-bbfe0e929452`, `confirm-e2e-pa4-b932c6bd8748`
  and `confirm-frozen-490ea023700c`, run under `PREREGISTRATION.md`.

End-to-end runs, 20,000 evaluations each (`phase1-results/`):

| Task | HC solved | LT-B solved | Best GOMEA (UNI) | Primary effect, LT-B vs HC (pre-registered rule) | Median evals / CPU s (HC vs LT-B) | Failures |
|---|---|---|---|---|---|---|
| maj5 (30 seeds) | 20/30 | 6/30 | 19/30 | **worse**: W/T/L 0/15/15, p_holm 6e-4 | 13.6k vs 20k / 632 vs 735 | 0 |
| mux6 (30 seeds) | 30/30 | 30/30 | 30/30 | **worse**: evals-to-solve median 1,734 vs 4,854, p_holm 2e-5 | 1.7k vs 4.9k / 27 vs 99 | 0 |
| pa4 (20 seeds) | 2/20 | 0/20 | 0/20 | **worse**: median error 2 vs 4, W/T/L 1/4/15, p_holm 0.02 | 20k vs 20k / 767 vs 429 | 0 |

## What the evidence supports

Under equal evaluation budgets, **the GOMEA inner optimizer did not improve MOSES on any
task**. It was significantly worse than stock hill climbing on all three:

* maj5: LT-B solved only seeds 1001, 1005, 1006, 1014, 1027 and 1029 (e.g. `maj5__LT-B__s1005`); HC failed only on 10 seeds (e.g. `maj5__HC__s1011`);
* mux6: `mux6__HC__s1001` solved at 1,809 evaluations, against 5,239 for LT-B;
* pa4: only HC solved it (`pa4__HC__s1001`, `pa4__HC__s1002`).

**Learned linkage did not beat no linkage.** Univariate GOMEA (UNI) was significantly
better than LT-B on maj5 (19/30 vs 6/30) and on mux6 (fewer evaluations), and tied on pa4.
LT-B vs ordinary-MI linkage (LT-MI) was a **tie** everywhere, so the initialization-bias
correction had no measurable effect. Forced improvements (LT-B-FI) were also a tie. Random
trees were the worst (maj5: 3/30).

On **identical frozen representations** (secondary, descriptive), LT-B did find
lower-error programs than HC on pa4: W/T/L 16/21/3, p_holm 0.016. It spent about 6× the
evaluations to do so (a median of 1,065 vs 179), because HC stops at its first local
optimum. On mux6 fixtures HC solved more (20/30 vs 13/30) with fewer evaluations.

## Strengths, weaknesses and costs

**Strength.** GOMEA searches a single representation more thoroughly than HC.

**Weakness.** Inside MOSES, that thoroughness costs more budget per deme than it returns.
The telemetry (`findings.json → gomea_behaviour`) shows why:

* **Premature convergence.** With N = 16, the hierarchical FOS variants end about half of
  their demes by *population convergence*. For example, `maj5__LT-B__s1001` had 6 of 17
  demes converge.
* **Wasted evaluations.** 32–47 % of their requests re-score a genotype already evaluated
  in the same deme (duplicate requests), against 13–19 % for UNI. The demo problems run
  uncached in both arms.
* **Deme turnover depends on the task.** HC's deme count was not recorded here. In the separate 2-seed cost probes, HC used about 70 demes per run on pa4, against about 21 for LT-B here. On maj5, HC used only 5–7 demes against LT-B's 18. So turnover explains the pa4 gap but not the maj5 one; there, the per-deme search quality of the tree FOS is the likelier cause.

Larger populations (32, 64) were worse still in the pilot, because each generation costs
about N × 2d trials.

## Decision

**Do not adopt LT-B GOMEA as a replacement for HC** under the tested budgets. Keep it as an
opt-in, documented extension (`USAGE.md`); prefer `--gomea-fos uni` if using it. The
remaining uncertainty covers larger budgets, other task families, and multi-deme or
probing settings, none of which were tested.

The next justified experiment is a **per-deme genotype cache**, which removes the 13–47 %
duplicate requests, applied identically to HC. It should be combined with a
diversity-preserving population policy (IMS or immigrants). Then re-test UNI and LT-B
against HC.

Evidence: `phase1-results/` (the checksums of every run summary, the manifests and
environments, `findings.json`, `e2e_verdicts.csv`, and the figures). The bulk traces are in
`../gomea-results/1/`. Configuration: `configs/gomea/phase1/`. Reproduce at tag `gomea-phase1-confirmation` with
`tools/gomea/run_phase1.sh OUT 14 512`, then
`python tools/gomea/phase1_report.py OUT/1 docs/gomea/phase1-results`.
