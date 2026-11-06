# Server round 4 findings: knob caps with and without the evaluation cap (phase 1as)

**Status:** complete.

* **Run:** 2026-11-03 13:20 to 2026-11-04 07:37 UTC, on the same pod as rounds 1 to 3
  (`jupyter-yderese`, cgroup quota 12 CPUs, 64 GB). W = 11 workers (limited by CPU). One
  orchestrator session. S0 took 1.34 h and 1as took 16.95 h.
* **Runs:** 490 (350 regression, 140 mux11), all `ok`, 0 failed. Total CPU 168.4 h (the sum of
  `cpu_s` over all run summaries).
* **Code:** `26aa93e`. The tree was dirty only in seven configs of earlier rounds (1af, 1ao, 1ar),
  which the scripts of rounds 1 and 3 wrote on the server. No phase of this round uses them.
* **Protocol:** `SERVER-ROUND4-PREREGISTRATION.md`.
* **Results (outside git):** `../server-results-round4/gomea-results/`. The round log is
  `server-run4/round-summary.md`. Report: `phase1as/report-stage2`.
* **How the extra numbers were computed:** a read-only script read each `summary.json`. "Gap" is
  the median over runs of (test NMSE − train NMSE). Deme counts, generations per deme and deme
  end shares are from `aggregate.csv`, which counts only demes with at least one evaluation.

**Answer in short:**

* **No arm qualifies under the adoption rule.** No arm is better than BASE on any regression
  dataset, so the rule keeps BASE.
* **mux11: no evaluation cap is better.** NOCAP solved mux11 in 225 s of CPU against 619 s for
  BASE (18/20 pairs, Holm p 0.0011). It used about 3× more evaluations (14,841 against 5,166), but
  each evaluation cost about 7× less CPU.
* **Knob caps do not help.** K8 and K16 are worse on mux11, and all knob caps tie on regression
  with higher medians than BASE.
* **The demes did run longer.** Without the evaluation cap, demes ran 5–17 generations, against
  1 for BASE. On regression, this gave no gain in accuracy.
* **This repeats 1ae.** In server round 1, cap 1,000 was better on mux11 (278 against 627 s) and
  worse on Con. I did not point this out when we designed round 4.
* **The user's decision (2026-11-06):** no evaluation cap becomes the base for all problems. This
  goes against the adoption rule (see "Decision").

## S0. Environment check

* **pytest:** 2 failed, 8,944 passed, 13 skipped.
  * `test_toolchain_drift_is_clean_on_the_real_repo`: the same environmental failure as in rounds
    1 to 3 (the pod's g++ 13.3.0 against the fixtures' g++ 16.2.1).
  * `test_tune_arms_wait_for_the_choice[phase]`: round 3's choice script wrote `lm` into the 1ao
    configs on the server, and the test expects the placeholder. No phase of rounds 4 or 5 uses
    these configs.
* **Reproducibility:** 4 of 4 reference runs matched exactly, the mux11 run included.

## 1as. Knob caps with and without the evaluation cap

All arms have linear scaling and LT-B, N = 16. BASE has the cap of 250 and no knob cap. NOCAP has
neither cap. K8 to K64 have a knob cap (explore, contrast) and no evaluation cap. K16-C250 has
both. Regression runs have 1,000 s, and mux11 runs have 3,600 s and stop when solved.

### mux11 (20 seeds, CPU to solve)

| Arm | Solved | CPU to solve (s) | Evaluations used | Demes | Generations per deme | Size | Verdict vs BASE |
|---|---|---|---|---|---|---|---|
| BASE | 20 | 619 | 5,188 | 21 | 1 | 37 | |
| **NOCAP** | 20 | **225** | 14,860 | 4 | 8 | 29 | **better** (18/2, Holm p 0.0011) |
| K8 | 6 | not reached | 13,810 | 50 | 5 | 55 | **worse** (1/19) |
| K16 | 17 | 2,268 | 24,860 | 45 | 5 | 34.5 | **worse** (0/20) |
| K32 | 20 | 1,310 | 25,610 | 16.5 | 7 | 39 | tie (5/15, Holm p 0.073) |
| K64 | 20 | 778 | 25,690 | 8 | 10 | 34.5 | tie (8/12, Holm p 1) |

* **CPU per evaluation (solved runs):** BASE 0.116 s, NOCAP 0.016 s, K64 0.027 s.
* **Secondary:** K16-C250 is **worse** than BASE (3/20 solved), and K16 is **better** than
  K16-C250 (16/1).
* **Wall time:** the longest NOCAP run took 498 s. The longest K8 and K16-C250 runs took 3,695 s
  and 3,807 s, inside the wall cap of 5,400 s.

### Regression (test NMSE, 1,000 s)

| Dataset | Splits | BASE | NOCAP | K8 | K16 | K32 | K64 |
|---|---|---|---|---|---|---|---|
| Con | 10 | 24.93 | 27.49 (+10 %) | 30.75 (+23 %) | 28.93 (+16 %) | 28.97 (+16 %) | 30.26 (+21 %) |
| EnH | 20 | 8.26 | 8.43 (+2 %) | 8.81 (+7 %) | 8.57 (+4 %) | 8.45 (+2 %) | 8.50 (+3 %) |
| Air | 20 | 37.87 | 41.13 (+9 %) | 39.97 (+6 %) | 39.90 (+5 %) | 41.13 (+9 %) | 41.50 (+10 %) |

* **Primary (15):** all ties. Every median is higher than BASE.
* **Air:** K8 (Holm p 0.017) and K64 (Holm p 0.019) are statistically worse, but the median
  difference is below 10 %, so the registered verdict is a tie. NOCAP was worse in 16/20 pairs
  (Holm p 0.073).
* **Con:** no contrast reached the extension range, so Con has 10 splits only.
* **Secondary:** K16-C250 ties BASE and K16 on all three datasets.

| Dataset | Arm | Size | Gap | Demes | Generations per deme | Main deme end |
|---|---|---|---|---|---|---|
| Con | BASE | 162 | 3.0 | 45 | 1 | evaluation cap (97 %) |
| | NOCAP | 153 | 2.8 | 5.5 | 7 | converged (73 %) |
| | K16 | 136 | 3.5 | 12.5 | 9 | stagnation (88 %) |
| EnH | BASE | 247.5 | 1.8 | 41.5 | 1 | evaluation cap (98 %) |
| | NOCAP | 150 | 1.7 | 6 | 8 | converged (73 %) |
| | K16 | 97 | 1.2 | 12 | 14 | stagnation (87 %) |
| Air | BASE | 261.5 | −0.5 | 35.5 | 1 | evaluation cap (97 %) |
| | NOCAP | 213.5 | 1.3 | 7 | 7 | converged (75 %) |
| | K16 | 166 | 0.3 | 8 | 17 | stagnation (75 %) |

Size was not a registered contrast. The smaller programs of the knob caps are descriptive. LM
(1ao, 1av) gives programs of similar or smaller size with lower error.

## Interpretation

* **Why BASE ran 1 generation:** about 97 % of BASE demes ended on the cap of 250. The first
  generation used the whole cap. Without the cap, demes ended when the population converged or
  stopped improving, after 7–8 generations.
* **Why NOCAP is cheaper per evaluation (inference):** knob building is about 68 % of GOMEA's CPU
  on mux11 (`BACKLOG.md`, section 1), and MOSES builds the knobs again for each deme. NOCAP built
  4 demes per mux11 run against 21. This round did not measure the knob-building time.
* **Evaluations against CPU:** the cap saves evaluations (5,166 against 14,841 to solve mux11).
  Phases 1d and 1e, where the cap won, counted evaluations. This round counted CPU. With a slow
  fitness function the cap is the better choice, and with fast scorers no cap is better on mux11.
* **Knob caps:** on mux11, fewer knobs gave worse results (K8 6/20 solved, K64 20/20). The cap
  removes knobs that the search needs. On regression, the extra generations of the knob caps
  did not lower the error.

## Decision

The user decided (2026-11-06): "lets make the no cap the default for all problems for now". The
registered rule keeps BASE, because NOCAP is not better on any regression dataset, and its
regression medians are 2–10 % higher. The code default of `--gomea-max-evals-per-deme` is already
no cap, so the code does not change. The base arm of later rounds has no cap. Knob caps stay off
by default and are not recommended.

## Deviations

1. **The design missed 1ae.** The NOCAP result on mux11 was predictable from 1ae (cap 1,000
   better). The round still adds two results: no cap gives no further gain over cap 1,000, and
   knob caps do not help with the evaluation cap removed.
2. **The dirty tree** (see "Code") is from earlier rounds. It caused the second pytest failure.

## Limits

* **One Boolean problem:** the NOCAP gain is from mux11 only. 1ae used the same problem.
* **Regression with no cap** is a tie with higher medians, and on Air it is close to "worse".
* **One budget:** 1,000 s for regression and 3,600 s for mux11.
* **LM and the penalty with no cap** are not tested. LM runs once per deme, so with 5–7 demes a
  run gets far fewer LM calls (`BACKLOG.md`, section 7).
