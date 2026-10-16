# Phase 1n findings: GOMEA vs hill climbing with the same per-deme cap

**Status:** complete. Run times (all 2026-10-15/16):

* **Stage 1:** 20:54 to 23:35. Con 20 runs, mux11 20 runs.
* **mux11 extension:** to 01:01, 20 more runs.

All 60 runs ended `ok`. Protocol: `PHASE1N-PREREGISTRATION.md`. Code `8b1e728`. Results:
`phase1n-results/`, `../gomea-results/phase1n/` and `../gomea-results/1/phase1n-mux11*/`.

**Answer:** yes. With the same 250-evaluation cap per deme for both, GOMEA (LT-B, memo) is
**better** than hill climbing (capped HC, `-a hc-capped`) on both problems at equal evaluations.

## Results

| Problem (metric) | Pairs | GOMEA lower / tie / HC lower | Median GOMEA | Median capped HC | Holm p | Verdict |
|---|---|---|---|---|---|---|
| Con (test NMSE), stage 1 | 10 | 9 / 0 / 1 | 37.1 | 60.4 | 0.0078 | **better** |
| mux11 (evaluations to solve), stage 1 | 10 | 7 / 0 / 3 | 5,345 | 7,306 | 0.16 | extend |
| mux11, stage 2 (seeds 3001–3020) | 20 | 17 / 0 / 3 | 4,633 | 7,978 | 0.0012 | **better** |

* Both arms solved mux11 in all 20 runs.
* Stage-1 Holm correction covers the two problems; stage 2 tests the extended problem alone,
  as registered.

**Cost, equal evaluations** (median CPU per run; mux11 over all 20 seeds, Con over 10 splits):

| Problem | GOMEA | Capped HC |
|---|---|---|
| mux11 | 1,965 s | 1,118 s |
| Con | 1,525 s | 343 s |

On mux11 that is 363 ms against 132 ms of CPU per evaluation, about 2.75× (median of per-run
ratios). Against uncapped HC the ratio was about 10× (Phase 1l). Capped HC also rebuilds knobs
more often (median 32.5 demes per run against 5 uncapped), so it pays more per evaluation.

## What this shows

1. **GOMEA's earlier wins were not an artefact of comparing against uncapped HC.**
   * **Con:** capping HC did not help it. Capped HC's median test NMSE was 60.4, against 54.9
     for uncapped HC on the same splits (Phase 1m); it was lower in 5 of 10 splits, p 0.77.
   * **mux11, evaluations:** capped HC needed about as many evaluations as uncapped HC (median
     7,978 against 7,953 in Phase 1l).
   * **mux11, time:** capped HC was much slower than uncapped HC (median CPU to solve 1,113 s
     against 278 s).
   * GOMEA beats it on both problems.
   * The gain comes from how GOMEA searches a deme, not from shorter demes alone. This
     experiment cannot attribute it to linkage: in Phase 1g, GOMEA without linkage did not
     beat capped HC.
   * It agrees with Phase 1g on regression (LT-B vs capped HC, 2,000 evaluations), where the
     direction was the same but not significant.
2. **The cost gap is smaller under equal caps but still real.**
   * On mux11, GOMEA needs about 0.58× the evaluations at about 2.75× the CPU per evaluation.
   * In CPU to solve, recorded per run, GOMEA was about 1.8× slower: median 1,959 s against
     1,113 s, with HC faster in 15 of 20 seeds. This is descriptive, not a registered test.
   * Lowering GOMEA's cost per evaluation (knob building, exemplar growth; see `BACKLOG.md`
     items 13 and 1.x) remains the route to winning at equal time.
3. **Determinism check:** the GOMEA arms repeat earlier runs exactly.
   * **Con:** the same test NMSE as Phase 1m's current-GOMEA arm and Phase 1f part D on all 10
     splits.
   * **mux11:** the same evaluations to solve as Phase 1e on all 20 seeds, which is why the
     stage-2 median (4,633) equals Phase 1e's.

## Limits

* One Boolean and one regression problem, 10–20 pairs, one cap (250).
* Equal evaluations, not equal time.
