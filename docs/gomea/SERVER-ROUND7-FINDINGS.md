# Server round 7 findings: LM-VAL with no cap, and LM-VAL with PEN125 (phase 1aw)

**Status:** complete.

* **Run:** 2026-11-07 17:18 to 2026-11-08 05:39 UTC, started by round 6 (the chain). W = 11. S0
  took 1.33 h and 1aw 11.03 h.
* **Runs:** 400, all `ok`. Total CPU 116.3 h.
* **Code:** `e6c0038`, clean tree.
* **Protocol:** `SERVER-ROUND7-PREREGISTRATION.md`. POP = N16 from round 6, so the four main arms
  use `--gomea-pop-size 16`, and the control arm LMVAL-DYN uses the dynamic population, F 0.05.
* **Results (outside git):** `../server-results-round7/gomea-results/`. Report:
  `phase1aw/report-stage2`. Extended: Air and EnH for LMVAL vs NOCAP, Air for the PEN125 size.

**Answer in short:**

* **Rule 1 passes: LM-VAL helps with no cap.** Better than NOCAP on EnH, Air and Yac, tie on WiR,
  Con and Tow, worse on none.
* **Rule 2 fails: PEN125 does not make LM-VAL's programs smaller on enough datasets.** Its size is
  better only on Air by the registered verdict (EnH also, but only in extension data that the
  stage rule does not use for that contrast). Its NMSE is within −4 % to +2 %.
* **Rule 3 does not apply, but there is a hint.** LMVAL with the cap of 250 ties with LMVAL on all
  six datasets, but its medians are lower on 4 of 6. On Tow it is lower in 10 of 10 splits (Holm
  p 0.031).
* **The control arm LMVAL-DYN** is worse than LMVAL on EnH and Air, with smaller programs.
* **The user's decisions (2026-11-09):** LM-VAL is the default and PEN125 is kept
  (`APPROVALS.md`).

## S0

* pytest: 1 failed (the same toolchain test as in round 6), 9,182 passed, 13 skipped.
* Reproducibility: 4 of 4 matched.

## Results

Median test NMSE (size in brackets). Splits: 20 on Air and EnH, 10 on the others.

| Dataset | NOCAP | LMVAL | LMVAL-PEN125 | LMVAL-C250 | LMVAL-DYN |
|---|---|---|---|---|---|
| WiR | 65.8 (128) | 64.0 (114) | 65.3 (130.5) | 64.6 (159.5) | 65.9 (82.5) |
| Con | 27.5 (156.5) | 23.0 (119) | 22.1 (98) | 19.3 (131.5) | 29.6 (93) |
| EnH | 8.43 (150) | **6.74** (117) | 6.87 (85.5) | 6.03 (118) | 8.27 (72.5) |
| Air | 41.1 (216) | **33.5** (157) | 34.1 (**116**) | 33.0 (148.5) | 39.7 (106.5) |
| Yac | 1.21 (94) | **0.58** (75) | 0.57 (72.5) | 0.59 (81) | 0.67 (67.5) |
| Tow | 20.2 (99) | 17.6 (108) | 17.1 (74) | 11.3 (152.5) | 18.6 (30) |

Bold: registered "better" against the arm on its left in the primary contrast.

**Primary verdicts:**

| Dataset | LMVAL vs NOCAP (NMSE) | LMVAL-PEN125 vs LMVAL (size) |
|---|---|---|
| WiR | tie (7/2) | tie (5/5) |
| Con | tie (7/3) | tie (5/4) |
| EnH | better (17/3, Holm p 0.0009) | tie at stage 1 (6/3); not extended |
| Air | better (19/1, Holm p 0.0009) | better (19/1, Holm p 0.0005) |
| Yac | better (10/0, Holm p 0.023) | tie (5/5) |
| Tow | tie (6/3) | tie (7/2) |

Pairs that do not add up to 10 or 20 have ties (equal values).

**Secondary:** LMVAL vs LMVAL-C250, all 6 tie. LMVAL-PEN125 vs LMVAL on NMSE, all 6 tie.
LMVAL-DYN vs LMVAL, worse on EnH (3/17) and Air (0/20), tie on the others.

**Other numbers:**

* **Train–test gap:** the guard keeps the gap small with no cap. WiR LMVAL 2.50, against 9.69 for
  plain LM in round 5 (1at).
* **Demes and LM calls:** with no cap a run has 2–28 demes (median 6.5–8 on most datasets), and LM
  runs once at each deme end. With the cap of 250 a run has 11.5–107.5 demes, so many more LM
  calls. That is a probable cause of the lower C250 medians. No run measured it directly.

## Deviations

* None. All 400 runs ok.
* The choice script of round 8 printed the stage-2 Holm p (0.009) next to the stage-1 verdict
  ("tie") for the EnH size contrast. The verdict follows the stage rule; only the printed p is
  confusing.

## Limits

* 10 splits on 4 datasets, so most contrasts can only tie after the Holm correction.
* LMVAL-C250 is secondary. The hint needs its own primary test.
