# Server round 8 findings: confirmation on fresh splits (phase 1ax)

**Status:** complete.

* **Run:** 2026-11-08 05:39 to 17:54 UTC, started by round 7 (the chain). W = 11. S0 took 1.34 h
  and 1ax 10.90 h.
* **Runs:** 400 (397 ok, 3 error). Total CPU 118.2 h.
* **Code:** `e6c0038`, clean tree.
* **Protocol:** `SERVER-ROUND8-PREREGISTRATION.md`.
* **FINAL (chosen by rule 2 of 1aw, which failed):** LMVAL of 1aw: `-a gomea --gomea-fos lt-b
  --gomea-pop-size 16 --gomea-skip-inactive 0 --gomea-genotype-memo 1 --gomea-contin copy
  --knob-probe-baseline reduced --linear-scaling 1 --const-opt lm --const-opt-guard val`.
* **Results (outside git):** `../server-results-round8/gomea-results/`. Report:
  `phase1ax/report-stage1`. An independent check with the standard library only:
  `../round8-check/round8_check.py` gives the same numbers.

**Answer in short:**

* **FINAL is better than HC-LS and NOCAP in about 3 of 4 pairs, but by less than 10 %.** Against
  HC-LS: 73 of 100 pairs lower, median ratio 0.925 (7.5 % lower), Holm p 9.5e-7. Against NOCAP: 72
  of 100 lower, ratio 0.939, Holm p 9.5e-7. Both registered verdicts are "tie", because the rule
  also needs a ratio ≤ 0.9.
* **FINAL-DYN vs FINAL (control):** FINAL-DYN lower in 37 of 100, ratio 1.03, tie. Its programs are
  25–70 % smaller.
* **Dow has a memory problem with no cap.** 3 Dow runs ended with an error at a peak of about 3.0–
  3.1 GB, which is most likely the 3 GB limit per run.

## S0

* pytest: 1 failed (the same toolchain test), 9,182 passed, 13 skipped.
* Reproducibility: 4 of 4 matched.

## Results

**Primary (pooled, 100 pairs, r = log(NMSE_A / NMSE_B)):**

| Contrast | A lower | B lower | Equal | Ratio exp(median r) | Holm p | Verdict |
|---|---|---|---|---|---|---|
| FINAL vs HC-LS | 73 | 27 | 0 | 0.925 | 9.5e-7 | tie (ratio > 0.9) |
| FINAL vs NOCAP | 72 | 26 | 2 | 0.939 | 9.5e-7 | tie (ratio > 0.9) |
| FINAL-DYN vs FINAL (secondary) | 37 | 63 | 0 | 1.030 | 0.0006 | tie |

**Median test NMSE per dataset (size in brackets), splits 21–30:**

| Dataset | FINAL | NOCAP | HC-LS | FINAL-DYN | FINAL lower than HC-LS |
|---|---|---|---|---|---|
| Air | 32.9 (149.5) | 37.3 (209.5) | 39.4 (130.5) | 38.7 (113.5) | 10 of 10 |
| Bos | 25.6 (164.5) | 27.8 (206.5) | 28.8 (131) | 25.6 (78) | 8 of 10 |
| Con | 22.6 (123.5) | 24.9 (148.5) | 20.8 (75.5) | 31.6 (79) | 4 of 10 |
| Dow | 31.1 (86) | 32.6 (91) | 38.1 (56) | 33.9 (36.5) | 6 of 10 |
| EnC | 10.3 (92.5) | 11.1 (120.5) | 11.0 (78.5) | 10.9 (65.5) | 8 of 10 |
| EnH | 7.12 (110.5) | 7.73 (128.5) | 7.60 (96.5) | 7.47 (58.5) | 8 of 10 |
| Tow | 16.3 (105.5) | 19.2 (138) | 18.9 (49) | 17.2 (31) | 7 of 10 |
| WiR | 63.3 (120) | 63.0 (167.5) | 64.7 (113) | 63.4 (74) | 5 of 10 |
| WiW | 75.4 (109.5) | 74.9 (141) | 75.8 (103.5) | 73.8 (83) | 7 of 10 |
| Yac | 0.61 (97) | 1.48 (100) | 1.26 (94) | 0.73 (65) | 10 of 10 |

* **Where FINAL wins:** Yac and Air in 10 of 10 splits; most splits on Bos, EnC, EnH and Tow.
* **Where FINAL loses:** Con, where HC-LS is more accurate with smaller programs.
* **LM in FINAL:** median 6 LM calls per run (2–36), 771 in total, 105 rejected by the guard.
* **Size:** FINAL's programs are smaller than NOCAP's on every dataset, and larger than HC-LS's on
  every dataset.

## Deviations

* **3 Dow runs failed** (FINAL splits 22 and 25, NOCAP split 29): exit code 1 after 880–1,016 s,
  peak RSS 3.07–3.11 GB against the limit of 3,072 MB. They count as the worst result, as the
  protocol says. The pre-registration named this risk: no Dow run without the cap existed before.
* **HC-LS on WiW split 21** has test NMSE `inf` with status ok: its program gives an infinite value
  on a test row. It counts as the worst result for HC-LS.

## Limits

* One budget (1,000 s) and 10 splits per dataset.
* FINAL is not the setting that the user later chose as the base (dynamic population, LM-VAL and
  PEN125). That combination never ran together (`APPROVALS.md`, 2026-11-09).
