# Phase 1ac findings: confirm PEN's CPU-to-solve gain on mux11 (backlog 16)

**Status:** complete.

* **Run:** 2026-10-21 from 10:09 to 11:30, 20 runs, 16 workers. All runs ended `ok` and all
  solved.
* **No extension:** stage-1 Holm p was 0.56, outside the extension band.
* **Code:** `2c5f24c`, code check ok.
* **Protocol:** `PHASE1AC-1AG-PREREGISTRATION.md`.
* **Results:** `phase1ac-results/` and `../gomea-results/1/phase1ac-bool-7a4259c69be4/`.
* **Load disclosure:** during the run, two subagents ran identity checks in separate clones,
  each a single job at nice 19. One of them used the regression runner with 2 workers for a
  few minutes before 1ac started. Another project's script was also running on the machine.
  The metric is CPU time, not wall time.

**Answer: no.** On fresh seeds, the acceptance penalty
(`--gomea-accept-complexity-ratio 0.125`) does not cut GOMEA's CPU to solve mux11. The result
is a registered **tie**, and by median PEN was slower.

## Results (mux11, seeds 3031–3040)

| Arm | Solved | Median CPU to solve | Median evaluations to solve | CPU per evaluation | Demes per run | Final size |
|---|---|---|---|---|---|---|
| GOMEA | 10 / 10 | **1,055 s** | 4,389 | 0.25 s | 18.5 | 32.5 |
| PEN | 10 / 10 | 1,628 s | 7,035 | 0.18 s | 29 | 29 |

| Contrast | PEN lower / higher | p (Holm) | Verdict |
|---|---|---|---|
| PEN vs GOMEA | 3 / 7 | 0.56 | **tie** |

## What this shows

1. **The 1o/1s lean towards PEN does not replicate.**
   * On the 20 earlier seeds, PEN was lower in 13 of 20 pairs (median 1,889 against 2,998 s,
     p 0.097).
   * On 10 fresh seeds it was lower in only 3 of 10 pairs.
   * Pooled over 30 seeds, PEN was lower in 16 of 30 pairs. There is no consistent effect.
2. **The mechanism is the same as before, but it nets out to nothing:**
   * PEN makes each evaluation cheaper (0.18 against 0.25 s), because its programs are
     smaller.
   * It needs more evaluations and more demes to solve (7,035 against 4,389).
3. **Practical reading:** PEN is not adopted for Boolean problems for speed. On regression it
   remains a D1 size lever (1o: Con CPU −37 %). It is tested again combined with the uniform
   start in 1ad.

## Limits

* One problem and 10 fresh seeds. The comparison with 1o/1s is across phases, run on older,
  slower code.
