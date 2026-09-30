# Phase 1 findings

**Status: complete.** There were 1,140 pre-registered confirmatory runs, all `ok`: 0
timeouts, 0 errors, 0 memory warnings. They ran locally on 2026-09-26 with 14 workers.

## Implementation and comparison

* **Implementation commit:** `c6301cb`. The runner's code check passed for every manifest.
  Later commits add docs and tools, and in `9550bbc` help text and a population default of
  16. The runs passed `--gomea-pop-size 16` explicitly, so they are unaffected.
* **Matched baseline:** stock `-a hc`. This work does not change it; its output is
  byte-identical to the pre-extension port, per the tests in `tests/gomea/test_integration.py`.
* **Representation:** the MOSES logical knobs of one frozen representation per deme.
* **Reused code:** GP-GOMEA `750220c` (Apache-2.0). Linkage statistics, the UPGMA/RNN tree
  and GOM are ported with golden parity (`PROVENANCE.md`).
* **Deviations from the papers:**
  * the initial population comes from MOSES's sampler;
  * acceptance uses MOSES's composite comparator;
  * forced improvement is a reconstruction;
  * N = 16, from the pilot.
* **Manifests:** `confirm-e2e-bbfe0e929452`, `confirm-e2e-pa4-b932c6bd8748` and
  `confirm-frozen-490ea023700c`, run under `PREREGISTRATION.md`.

End-to-end results, 20,000 evaluations per run (`phase1-results/`):

| Task | HC solved | LT-B solved | Best GOMEA (UNI) | LT-B vs HC (pre-registered rule) | Median evals-to-solve, HC vs LT-B |
|---|---|---|---|---|---|
| maj5 (30 seeds) | 20/30 | 6/30 | 19/30 | **worse**: W/T/L 0/15/15, p_holm 6e-4 | 9,189 vs 10,579 (solved runs only) |
| mux6 (30 seeds) | 30/30 | 30/30 | 30/30 | **worse**: evals-to-solve p_holm 2e-5 | 1,627 vs 4,829 |
| pa4 (20 seeds) | 2/20 | 0/20 | 0/20 | **worse**: median error 2 vs 4, p_holm 0.02 | 17,881 (HC only) |

## What the evidence supports

At equal evaluation budgets, **GOMEA inside MOSES did not improve MOSES on any task**. LT-B
was significantly worse than stock hill climbing on all three tasks:

* on maj5, LT-B solved only seeds 1001, 1005, 1006, 1014, 1027 and 1029;
* `mux6__HC__s1001` solved at 1,809 evaluations, against 5,239 for LT-B;
* only HC solved pa4 (`pa4__HC__s1001` and `pa4__HC__s1002`).

Learned linkage did not beat no linkage:

* **UNI vs LT-B:** UNI was better on maj5 (19 vs 6 of 30) and on mux6 (fewer evaluations),
  and tied on pa4.
* **LT-B vs LT-MI:** tied on every task, so the initialization-bias correction had no
  measurable effect.
* **Forced improvements:** tied on every task.
* **Random-tree FOS (RT):** lowest on maj5 (3/30), but not significantly different from
  LT-B.

**Frozen representations** (secondary, descriptive). On pa4 fixtures, LT-B reached lower
error than HC (W/T/L 16/21/3, p_holm 0.016) using about 6× the evaluations (a median of
1,065 vs 179). On mux6 fixtures, HC solved more (20/30 vs 13/30) with fewer evaluations. On
maj5 fixtures there was no significant difference.

## Strengths, weaknesses and costs

**Strength.** The only strength is on frozen pa4 representations: better programs than HC,
at much higher cost.

**Weaknesses,** from the GOMEA telemetry (`findings.json → gomea_behaviour`):

* The hierarchical FOS variants end 23–57 % of demes by *population convergence*. For
  example, 6 of 17 demes converged in `maj5__LT-B__s1001`.
* 32–47 % of their evaluation requests re-score a genotype already evaluated in that deme,
  against 13–19 % for UNI.
* GOMEA does not have a score cache, confirmed by its telemetry. HC's demo-problem scorer is
  also built without one (`demo_problems.py`).

**Interpretation, not measured:**

* Fewer demes may explain the pa4 gap. In separate 2-seed probes HC used about 70 demes per
  run, against LT-B's median of 21.5 here.
* That cannot explain maj5, where HC used 5–7 demes in the probes against LT-B's 18. There,
  per-deme search quality is the more plausible cause.

## Decision

*Superseded in part by Phases 1d–1f; see [`PHASE1E-FINDINGS.md`](PHASE1E-FINDINGS.md) and
[`PHASE1F-FINDINGS.md`](PHASE1F-FINDINGS.md).*

**Do not adopt LT-B GOMEA in place of HC** at these budgets. Keep it as a documented opt-in
(`USAGE.md`), with `--gomea-fos uni` preferred.

The next experiment: a per-deme genotype cache applied identically to both arms, plus a
diversity-preserving population policy, then re-test. Untested: larger budgets, other task
families, multi-deme and probing settings.

Evidence: `phase1-results/` and `../gomea-results/1/`. Reproduce at tag
`gomea-phase1-confirmation` with `tools/gomea/run_phase1.sh OUT 14 512`, then
`tools/gomea/phase1_report.py`.
