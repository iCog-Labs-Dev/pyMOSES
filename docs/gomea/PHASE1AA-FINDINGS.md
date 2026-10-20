# Phase 1aa findings: `-B 3` (effort-3 reduction in knob building)

**Status:** complete. The rerun was on 2026-10-20 from 10:57 to 16:27: 40 runs, 8 workers at
nice 19 while the machine was shared.

* **Code:** the merged effort-3 fix (`17a7bd0`, via `571b596`). A first attempt was stopped
  at 07:48 because of that bug; its results are discarded.
* **Early stop:** applied under a documented deviation (`phase1aa_earlystop.py`).

Protocol: `PHASE1U-1AA-PREREGISTRATION.md`. Results: `phase1aa-results/` and
`../gomea-results/1/phase1aa-bool-e9012d291ff4/`.

**Answer: no.** A stronger, effort-3 reducer during knob building (`-B 3`) makes both
optimizers far slower on mux11. Each evaluation costs about 4.5× more CPU, and evaluations to
solve do not fall. Verdict: **worse** for HC and for GOMEA, 10 of 10 pairs each.

## Results (mux11, CPU seconds to solve; unsolved or ended = +∞)

| Arm | Solved within the run | Median CPU to solve | CPU per evaluation | Demes per run |
|---|---|---|---|---|
| HC | 10 / 10 | 424 s | 0.058 s | 29 |
| HC-B3 | 3 / 10 (7 ended by the rule) | — | 0.26 s | 30 |
| GOMEA | 10 / 10 | 891 s | 0.18 s | 22.5 |
| GOMEA-B3 | 2 / 10 (8 ended by the rule) | — | 0.82 s | 37.5 |

| Contrast | Lower / tie / higher | Holm p | Verdict |
|---|---|---|---|
| HC-B3 vs HC | 0 / 0 / 10 | 0.0039 | **worse** |
| GOMEA-B3 vs GOMEA | 0 / 0 / 10 | 0.0039 | **worse** |

**The `-B 3` runs that solved, against their pairs:**

| Run | Evaluations to solve | CPU to solve | Base run: evaluations / CPU |
|---|---|---|---|
| HC-B3 s3001 | 7,341 | 1,939 s | 7,652 / 474 s |
| HC-B3 s3002 | 11,617 | 5,463 s | 6,827 / 468 s |
| HC-B3 s3008 | 6,903 | 1,791 s | 6,828 / 379 s |
| GOMEA-B3 s3001 | 8,789 | 6,756 s | 8,332 / 1,684 s |
| GOMEA-B3 s3002 | 9,745 | 8,604 s | 5,989 / 813 s |

## The early stop (deviation)

* **The problem:** at 15:41, 20 of 40 runs were done, and 8 slow `-B 3` runs held every slot.
* **The rule:**
  * `phase1aa_earlystop.py` ended each `-B 3` run whose CPU already exceeded its paired base
    run's CPU to solve.
  * It did so by terminating that run's own process, while the runner stayed alive and
    recorded it as `error`, i.e. unsolved (+∞), the outcome the registered metric gives.
  * 15 runs were ended this way, each at 1.1–17.6× its pair's CPU to solve.
* **No orphans:** the runner was never stopped, and none were found afterwards.
* **Ranking caveat:** treating the ended runs as +∞ only enlarges differences that already
  favour the base arm.

## What this shows

1. **Effort-3 reduction is too expensive for knob building.**
   * Every probe of every knob runs the stronger reducer. CPU per evaluation rises about
     4.4× for HC and 4.6× for GOMEA.
   * It does not save evaluations: the `-B 3` runs that solved used as many evaluations as
     their pairs, or more.
2. **It also exposed a latent upstream bug** (an empty-junctor crash in an effort-3 rule,
   also in C++). It is now fixed in the port as a disclosed DIVERGENCE (`BACKLOG.md` section
   4, item 2).
3. **Practical reading:** keep the default `-B 2`. This closes the backlog item.

## Limits

* One problem (mux11) and 10 seeds; CPU was measured on a shared machine at nice 19.
* With the early stop, the size of the slowdown per pair is a lower bound.
