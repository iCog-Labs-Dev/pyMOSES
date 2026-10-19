# Phase 1s findings: does the acceptance penalty make GOMEA faster on mux11?

**Status:** complete. The run was on 2026-10-19 from 10:55 to 12:30: 20 runs, all `ok`, all
solved. Protocol: `PHASE1S-PREREGISTRATION.md`. Code `ebd77c0`. Results:
`phase1s-results/` and `../gomea-results/1/phase1s-mux11-*/`.

**Answer: a registered tie.**

* Pooled over 20 seeds, PEN (`--gomea-accept-complexity-ratio 0.125`) solved mux11 with less
  CPU than plain GOMEA in 13 of 20 pairs, median 1,889 s against 2,998 s.
* That is not significant: p 0.097 against the registered α 0.025.
* The direction is the same in the new seeds and in Phase 1o's. The margin per run is too
  variable to call PEN reliably faster.

## Results

**Median CPU seconds to solve** (all runs solved):

| Pairs | Plain GOMEA | PEN | PEN lower / tie / higher | p | Verdict |
|---|---|---|---|---|---|
| **Pooled, 20 seeds (primary)** | 2,998 | 1,889 | 13 / 0 / 7 | 0.097 | **tie** |
| New seeds 3021–3030 | 3,693 | 2,671 | 6 / 0 / 4 | 0.43 | (replication check) |
| Phase 1o seeds 3001–3010 | 1,872 | 1,341 | 7 / 0 / 3 | 0.13 | (descriptive) |

**Descriptive, medians:**

| | Plain GOMEA | PEN |
|---|---|---|
| Evaluations to solve, new seeds / 1o seeds | 5,966 / 5,345 | 6,993 / 7,145 |
| CPU per evaluation, new / 1o (s) | 0.53 / 0.34 | 0.37 / 0.20 |
| Demes per run, new / 1o | 24 / 22 | 28.5 / 29.5 |
| Final best complexity, new / 1o | 32.5 / 37 | 29 / 29.5 |

* **CPU per evaluation is higher on the new seeds** for both arms. Those runs shared the
  machine with other jobs at load 25–30. Both arms were interleaved, so pairs remain
  comparable.

## What this shows

1. **PEN trades more evaluations for cheaper ones.**
   * Smaller exemplars make each evaluation about 30–40 % cheaper.
   * But PEN needs about 15–35 % more evaluations to solve, and more demes.
   * The net effect on CPU to solve is −28 % to −37 % at the median, but it varies widely
     by seed: in 7 of 20 pairs PEN was slower.
2. **The decision rule** in the pre-registration reads: "Tie: PEN does not make GOMEA faster
   to solve; it remains an opt-in CPU saver."
3. **On regression it stays neutral** at equal time (Phase 1p), with a likely small accuracy
   cost at equal evaluations (Phase 1o).

## Limits and follow-up

* **Single look by design,** so no extension ran. The result is a registered tie, not
  evidence of no effect.
* **Possible confirmation:** a dedicated test on the merged, faster code would show whether
  the median gain is real. It would use 20 fresh seeds and the same metric. This is in
  `BACKLOG.md`, not queued.
* **One problem** (mux11) and one penalty strength (R = 0.125).
