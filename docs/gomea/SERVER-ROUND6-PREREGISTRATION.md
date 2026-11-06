# Server round 6: pre-registration (2026-11-06)

**Status:** the design is approved by the user (see `APPROVALS.md`). Not implemented yet.

**Question:** is the fixed GOMEA population of 16 a bottleneck? Two phases: population size and
linkage (1ay), then the knob caps of round 4 again with a dynamic population (1az).

This round runs first. Round 7 (LM-VAL, phase 1aw) and round 8 (confirmation, phase 1ax) follow
in the same chain, without a pause, and use the population setting that 1ay picks.

## Rules

* As in rounds 1 to 5: on the server, one phase at a time, W from `resources.py`, state in
  `server-run6/`. No round overlaps with another.
* **Statistics:** paired Wilcoxon signed-rank, Holm across a phase's primary contrasts, α = 0.025.
  "Better" also needs a 10 % lower median. Holm p in [0.025, 0.25) at stage 1 extends that problem
  (regression splits 11–20, mux11 seeds 3011–3020). Secondary contrasts never extend. Failed runs
  count as infinitely bad.
* **Reported (D1):** test NMSE or CPU to solve, program size, train–test gap, CPU, deme count,
  generations per deme, deme end reasons, and the population size of each deme (median, range).
* **S0:** full pytest and the reproducibility check.
* **No evaluation cap in any arm** (the base since 2026-11-06), except where an arm says so.

## New code: a dynamic population from the classic hill-climbing code (off by default)

* **Flag:** `--gomea-pop-size dynamic`. The default stays 16, and a fixed number keeps its meaning.
* **Count:** N for a deme is the number of new instances that classic hill climbing would take at
  distance 1: `estimate_neighborhood(1, fields)` and then `n_new_instances(1, ...)`
  (`pymoses/moses/moses/optimization/hill_climbing.py`), with the classic options
  `--hc-fraction-of-nn` (default 2.0) and `--hc-max-nn-evals` (default 20,000), the classic minimum
  of 100 (`MINIMUM_DEME_SIZE`) and the evaluations left. No other limit.
* **Drawing:** the existing population builder, which uses MOSES's `sample_from_neighborhood`.
* **Identity:** with `--gomea-pop-size 16` (or not set), results are byte-identical to the code
  before the change.

## Base flags

* **Regression:** `-a gomea --gomea-skip-inactive 0 --gomea-genotype-memo 1 --gomea-contin copy
  --knob-probe-baseline reduced --linear-scaling 1`, 1,000 s per run.
* **mux11:** `-H mux -k 3`, `-a gomea --gomea-skip-inactive 0 --gomea-max-evals-per-deme 0
  --gomea-genotype-memo 1 --knob-probe-baseline reduced`, `--max-time 3600`, stop when solved.
* Each arm adds its FOS (`--gomea-fos lt-b` or `rt`) and its population (`--gomea-pop-size 16`, or
  `--gomea-pop-size dynamic --hc-fraction-of-nn F`).

## 1ay. Population size and linkage

* **Arms (8):** N16-LTB, N16-RT (N = 16); DYN2-LTB, DYN2-RT (dynamic, F = 2.0); DYN025-LTB,
  DYN025-RT (F = 0.25); DYN005-LTB, DYN005-RT (F = 0.05).
* **Problems:** mux11 (CPU to solve, seeds 3001–3010), Con and EnH (test NMSE, splits 1–10).
* **Primary (9):** DYN2-LTB, DYN025-LTB and DYN005-LTB, each against N16-LTB, on each problem.
* **Secondary (12):** LT-B against RT at each of the 4 population settings, on each problem.
* **The pick for rounds 7 and 8 (POP):** a dynamic fraction qualifies if its LT-B arm is better
  than N16-LTB on at least 2 of the 3 problems and worse on none. If several qualify, POP is the one
  with the best median rank. If none qualifies, POP is N = 16.
* **The pick for 1az and the control arms (DYN):** the dynamic fraction whose LT-B arm has the best
  median rank, whether or not it qualifies.
* **Median rank:** on each problem, rank the 3 dynamic LT-B arms by their median metric (1 = best);
  then take the median of the 3 ranks. Ties go to the smaller F.
* **Runs:** 240 (+ up to 240).

## 1az. The knob caps of round 4 again, with the dynamic population

* **Arms (6):** NOCAP-DYN (no knob cap); K8-DYN, K16-DYN, K32-DYN, K64-DYN (`--gomea-knob-cap K
  --gomea-knob-select explore --gomea-knob-rank0 contrast`, as in 1as); NOCAP-N16 (reference). All
  use LT-B and no evaluation cap. The DYN arms use the fraction DYN from 1ay.
* **Problems:** mux11 (3,600 s), Con, EnH, Air (1,000 s), as in 1as.
* **Primary (16):** each K-DYN arm against NOCAP-DYN, on each problem.
* **Secondary (4):** NOCAP-DYN against NOCAP-N16, on each problem.
* **Adoption rule (as 1as):** a knob cap qualifies if it is better than NOCAP-DYN on at least 2 of
  the 3 regression datasets and worse on none of the 4 problems. If several qualify, recommend the
  one with the smallest median size over the regression datasets. If none qualifies, no knob cap.
* **Runs:** 240 (+ up to 240).

## Changes to rounds 7 and 8 (user, 2026-11-06)

* The LM-VAL round (phase 1aw) becomes round 7 and the confirmation (phase 1ax) becomes round 8.
* Every GOMEA arm of rounds 7 and 8 uses POP. The NOCAP and HC-LS arms keep their other flags.
* **Control arm, round 7:** LMVAL-DYN, LMVAL with the population DYN. Secondary: test NMSE,
  LMVAL-DYN vs LMVAL, on each dataset (6).
* **Control arm, round 8:** FINAL-DYN, FINAL with the population DYN. Secondary: test NMSE,
  FINAL-DYN vs FINAL, pooled over the 100 pairs, as the primary test of round 8.
* **If POP equals DYN,** the control arms use N = 16 instead, so each round compares the two
  population settings.
* The control arms do not change any adoption rule or the choice of FINAL.

## Size and time

* 1ay: 240 runs, about 11 hours at W = 11; with extensions about 22 hours.
* 1az: 240 runs, about 12 hours; with extensions about 24 hours.
* S0: about 1.3 hours. Rounds 7 and 8 then take about 17–24 hours with the control arms.
* **Risks:** at F = 2.0, N can be several thousand, so GOMEA may finish few generations. A large N
  with many knobs needs more memory; the smoke test runs mux11 at F = 2.0.
