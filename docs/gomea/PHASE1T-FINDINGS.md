# Phase 1t findings: does the acceptance penalty help capped hill climbing?

**Status:** complete. The run was on 2026-10-19 from 12:30 to 13:24: 20 runs, all `ok`, all
solved. Protocol: `PHASE1T-PREREGISTRATION.md`. Code `ebd77c0`. Results:
`phase1t-results/` and `../gomea-results/1/phase1t-mux11-*/`.

**Answer: no. The penalty never changes hill climbing's search on mux11.**

* With `--gomea-accept-complexity-ratio 0.125`, capped HC produced exactly the same run as
  without it on all 10 seeds: the same evaluations to solve and the same final complexity.
* The registered verdict is a tie (p 0.32). The CPU differences are measurement noise.

## Results

| Seed | Evaluations to solve (both arms) | Final complexity (both) | CPU to solve, HC-C250 / HC-PEN (s) |
|---|---|---|---|
| 3021 | 7,458 | 30 | 1,080 / 1,702 |
| 3022 | 6,121 | 24 | 769 / 945 |
| 3023 | 6,032 | 28 | 1,326 / 949 |
| 3024 | 5,764 | 24 | 965 / 985 |
| 3025 | 7,755 | 24 | 1,326 / 1,326 |
| 3026 | 5,553 | 29 | 1,116 / 1,116 |
| 3027 | 6,742 | 26 | 1,375 / 988 |
| 3028 | 6,666 | 31 | 1,557 / 1,591 |
| 3029 | 4,752 | 29 | 1,220 / 1,228 |
| 3030 | 4,826 | 29 | 901 / 907 |

* **Medians:** 6,077 evaluations for both arms; CPU to solve 1,168 s (HC-C250) and
  1,052 s (HC-PEN).
* **Primary contrast** (HC-PEN vs HC-C250, CPU to solve): lower in 2 of 10, higher in 8,
  p 0.32, **tie**.

## What this shows

1. **The penalty works only through GOMEA's acceptance.**
   * HC moves to the best neighbour of each batch. MOSES's default composite score
     (0.29 per complexity unit) already breaks ties between equal raw scores towards the
     simpler program.
   * A penalty of 8 wrong rows per unit (R = 0.125) never outweighed a mux11 improvement
     step in these runs.
   * GOMEA accepts any trial that is not worse, so the same penalty changes its decisions on
     neutral moves (Phase 1o: smaller exemplars and cheaper evaluations).
   * So the penalty's cost savings come from changing GOMEA's search, not from a general
     property of the problem.
2. **Measurement noise in CPU time on this machine.**
   * Identical computations differed in CPU time by up to 1.6× (seed 3021: 1,080 s against
     1,702 s), because the machine was shared with other jobs.
   * CPU-to-solve comparisons can detect large effects but are noisy. Later phases report
     evaluations to solve alongside, and findings note this.

## Limits

* One problem (mux11) and one strength. A much stronger penalty would eventually change HC's
  choices; this phase used the strength PEN uses for GOMEA.
