# Server round 11: findings (2026-11-16)

**Pre-registration:** `SERVER-ROUND11-PREREGISTRATION.md`. **Results:** bundle
`bundle-final-20261112T180228Z.tar.gz` (local copy: `../server-results-round11/`). **Code:**
`653d131`. **Decisions:** `APPROVALS.md` (results only, no adoption rule; see "User decisions").

Test NMSE in the tables is × 100. "Pairs" is the number of splits where A is lower / where B is
lower, with the ties in brackets. "Ratio" is exp(median of log(A / B)) over the pairs: above 1
means that A has the higher value (higher NMSE, larger program or more CPU). A failed run counts as
infinitely bad in the contrasts. The per-dataset tables use the ok runs only; a number in brackets
after a value is the number of ok runs when it is less than 10.

**Datasets** (`data/gomea-sr/README.md`; rows in the training part of a split in brackets):

| Abbr. | Dataset | Input features | Rows (train) |
|---|---|---|---|
| Air | Airfoil Self-Noise | 5 | 1,503 (753) |
| Bos | Boston Housing | 13 | 506 (254) |
| Con | Concrete Compressive Strength | 8 | 1,030 (516) |
| Dow | Dow Chemical | 57 | 1,066 (534) |
| EnC | Energy Efficiency, cooling load | 8 | 768 (384) |
| EnH | Energy Efficiency, heating load | 8 | 768 (384) |
| Tow | Tower | 25 | 4,999 (2,501) |
| WiR | Wine Quality, red | 11 | 1,599 (801) |
| WiW | Wine Quality, white | 11 | 4,898 (2,450) |
| Yac | Yacht Hydrodynamics | 6 | 308 (154) |

## Summary

| Contrast | Family | Pairs | Holm p | Ratio | Verdict |
|---|---|---|---|---|---|
| 1. BASE-R vs BASE | primary | 62/20 (18) | 2.7e-6 | 0.984 | tie |
| 2. BASE-D5 vs BASE | primary | 67/29 (4) | 7.1e-5 | 0.961 | tie |
| 3. BASE-D10 vs BASE | primary | 70/29 (1) | 4.5e-6 | 0.915 | tie |
| 4. BASE-D20 vs BASE | primary | 72/28 | 4.7e-6 | 0.899 | **A better** |
| BASE vs HC-LM | gap to HC-LM | 29/71 | 5.7e-5 | 1.056 | tie |
| BASE-R vs HC-LM | gap to HC-LM | 37/63 | 0.0081 | 1.040 | tie |
| BASE-D5 vs HC-LM | gap to HC-LM | 44/56 | 0.23 | 1.018 | tie |
| BASE-D10 vs HC-LM | gap to HC-LM | 59/41 | 0.23 | 0.973 | tie |
| BASE-D20 vs HC-LM | gap to HC-LM | 60/40 | 0.23 | 0.971 | tie |

* The verdicts are descriptive. The user asked for results only (2026-11-11).
* All four arms are lower than BASE in most pairs (Holm p < 0.025 for all four). The ratio falls
  as the deme count rises: 0.984 (BASE-R, 5 demes), 0.961 (D5, 5), 0.915 (D10, 10), 0.899 (D20,
  18). Only D20 passes the 10 % rule.
* D10 and D20 are lower than HC-LM in 59 and 60 of 100 pairs (Holm p 0.23, ties). BASE is lower
  than HC-LM in 29 of 100 pairs.
* The D arms have larger programs (pooled median 74.5, 85.5 and 93.5 against 48) and use more
  CPU (1.66, 1.76 and 2.12 times BASE). BASE-R has the same CPU as BASE and programs 4 % larger.
* 4 of 600 runs failed at the memory limit, all D-arm runs on Dow and Tow. The phase took
  26.6 h, not the estimated 12–18 h.

## Run health

* **Steps:** S0 1.34 h, phase1be 26.64 h (95,918 s), one orchestrator session. W = 11; W = 7 for
  Dow (`manifest_workers` in `state.json`). The pre-registration estimated 12–18 h.
* **S0:** reproducibility 4 of 4 matched. pytest: 9,420 passed, 1 failed
  (`test_toolchain_drift_is_clean_on_the_real_repo`), 13 skipped. The same test failed in rounds
  6 to 10: the server's toolchain differs from the toolchain that the fixtures record. It is not
  a code fault.
* **Runs:** 596 ok, 4 failed (see "Failed runs"). Each ok run used exactly 20,000 evaluations.
* **Code:** `653d131` on the server, tree clean (`git-head.txt`; `round-summary.md`: dirty False).
* **CPU:** 224.4 h in total over the 600 runs. BASE 16.9 h, BASE-R 16.9 h, BASE-D5 41.7 h,
  BASE-D10 61.7 h, BASE-D20 64.7 h, HC-LM 22.6 h. The three D arms used 75 %.
* **Time:** the longest run took 30,035 s (Dow, BASE-D20), below the wall cap of 36,000 s. The D
  arms on Dow and Tow caused the long phase (see "Knob counts").

### Memory

Peak RSS in GB is `ru_maxrss_kb` / 1e6. The share of the limit is `ru_maxrss_kb` / (limit ×
1,024). The runner sets the limit on the address space (`RLIMIT_AS`), not on the RSS.

| Manifest | `mem_limit_mb` | W | Highest peak RSS | Run | Share of the limit |
|---|---|---|---|---|---|
| Dow | 8,192 | 7 | 8.36 GB | BASE-D5, split 40 (failed) | 100 % |
| 9 other datasets | 5,120 | 11 | 5.21 GB | Tow BASE-D5, split 35 (failed) | 99 % |

* Five runs reached 90 % of the limit or more: the 4 failed runs and Tow BASE-D10 split 37 (4.99
  GB, 95 %, ok).
* Highest peak RSS for each arm: BASE 2.31 GB, BASE-R 2.25 GB, HC-LM 3.97 GB, BASE-D5 8.36 GB,
  BASE-D10 6.07 GB, BASE-D20 8.24 GB. All arm maxima are on Dow.
* The pre-registration set these limits from round 10, where all arms except NOCAP stayed below
  3.4 GB. The memory of the D arms at 20,000 evaluations was not measured before the round (RV16
  finding 11, `APPROVALS.md`).

### Failed runs

| Run | Wall s | Peak RSS | Share of the limit | Demes that ended | Largest deme (knobs) |
|---|---|---|---|---|---|
| Dow BASE-D5, split 40 | 6,273 | 8.36 GB | 100 % | 3 | 4,811 |
| Dow BASE-D20, split 37 | 3,007 | 8.24 GB | 98 % | 7 | 814 |
| Tow BASE-D5, split 35 | 8,316 | 5.21 GB | 99 % | 4 | 2,555 |
| Tow BASE-D5, split 37 | 3,823 | 5.14 GB | 98 % | 2 | 262 |

* Each run ended with exit code 1 and the error "lost sys.stderr". The phase log gives no other
  text. All four reached 98–100 % of the address-space limit. The agent reads them as memory
  failures.
* The contrasts count them as infinitely bad against BASE-D5 (3 runs) and BASE-D20 (1 run).

## Main experiment (contrasts 1–4)

Pooled over 100 pairs (10 datasets × 10 splits). The size and CPU columns give the pooled medians
from `contrasts.csv` (BASE / other arm) and the verdict of the secondary contrast.

| Contrast | Pairs | Holm p | Ratio | Verdict | Size BASE / X | Size verdict | CPU s BASE / X | CPU verdict |
|---|---|---|---|---|---|---|---|---|
| BASE-R vs BASE | 62/20 (18) | 2.7e-6 | 0.984 | tie | 48 / 52.5 | tie | 458 / 458 | tie |
| BASE-D5 vs BASE | 67/29 (4) | 7.1e-5 | 0.961 | tie | 48 / 74.5 | A worse | 458 / 877 | A worse |
| BASE-D10 vs BASE | 70/29 (1) | 4.5e-6 | 0.915 | tie | 48 / 85.5 | A worse | 458 / 903 | A worse |
| BASE-D20 vs BASE | 72/28 | 4.7e-6 | 0.899 | A better | 48 / 93.5 | A worse | 458 / 1,155 | A worse |

The datasets have different NMSE scales, so this table gives no pooled NMSE medians. Use the ratio
and the per-dataset values.

### Program size and CPU (secondary, Holm over 8)

"A better" means a smaller program or less CPU.

| Metric | Contrast | Pairs (ties) | Holm p | Ratio | Verdict | Median A / B |
|---|---|---|---|---|---|---|
| Size | BASE-R vs BASE | 15/55 (30) | 8.4e-7 | 1.040 | tie | 52.5 / 48 |
| Size | BASE-D5 vs BASE | 20/74 (6) | 4.0e-9 | 1.490 | A worse | 74.5 / 48 |
| Size | BASE-D10 vs BASE | 18/80 (2) | 1.4e-10 | 1.462 | A worse | 85.5 / 48 |
| Size | BASE-D20 vs BASE | 17/82 (1) | 2.9e-13 | 1.711 | A worse | 93.5 / 48 |
| CPU s | BASE-R vs BASE | 42/58 | 0.15 | 1.005 | tie | 458 / 458 |
| CPU s | BASE-D5 vs BASE | 8/92 | < 1e-15 | 1.658 | A worse | 877 / 458 |
| CPU s | BASE-D10 vs BASE | 8/92 | < 1e-15 | 1.764 | A worse | 903 / 458 |
| CPU s | BASE-D20 vs BASE | 7/93 | < 1e-15 | 2.121 | A worse | 1,155 / 458 |

* `contrasts.csv` gives Holm p = 0.0 for the rows with "< 1e-15". The normal approximation
  computes 1 − Φ(z), and this value rounds to 0 in floating point (as in round 10).
* The size and CPU medians in `contrasts.csv` count a failed run as infinite. With the ok runs only,
  the pooled median size of BASE-D5 is 74 and of BASE-D20 is 93.

## The gap to HC-LM (Holm over 5)

| Contrast | Pairs | Holm p | Ratio | Verdict |
|---|---|---|---|---|
| BASE vs HC-LM | 29/71 | 5.7e-5 | 1.056 | tie |
| BASE-R vs HC-LM | 37/63 | 0.0081 | 1.040 | tie |
| BASE-D5 vs HC-LM | 44/56 | 0.23 | 1.018 | tie |
| BASE-D10 vs HC-LM | 59/41 | 0.23 | 0.973 | tie |
| BASE-D20 vs HC-LM | 60/40 | 0.23 | 0.971 | tie |

* BASE vs HC-LM repeats the round 10 result on new splits: round 10 gave 19/81 and ratio 1.06.
* HC-LM against BASE, pooled (agent computation, ok pairs; the report has no such contrast): size
  ratio 1.61 (median 72 against 48), CPU ratio 1.25 (663 against 458 s).

## Trade-off table (pre-registered)

Pooled over the 100 pairs or runs of each arm. The NMSE, size and CPU ratios against BASE and the
NMSE ratio against HC-LM come from `contrasts.csv`. The rows marked * are agent computations.
"Demes" and "LM share" are medians over the ok runs of each arm. "LM share" is the LM evaluations
/ the run's evaluations.

| Arm | NMSE vs BASE | NMSE vs HC-LM | Size vs BASE | CPU vs BASE | Demes | Knobs per deme | Total knobs | LM share |
|---|---|---|---|---|---|---|---|---|
| BASE | 1 | 1.056 | 1 | 1 | 2 | 74 | 217 | 0.2 % |
| BASE-R | 0.984 | 1.040 | 1.040 | 1.005 | 5 | 170 | 785 | 1.5 % |
| BASE-D5 | 0.961 | 1.018 | 1.490 | 1.658 | 5 | 181 | 822 | 1.9 % |
| BASE-D10 | 0.915 | 0.973 | 1.462 | 1.764 | 10 | 196 | 1,741 | 4.9 % |
| BASE-D20 | 0.899 | 0.971 | 1.711 | 2.121 | 18 | 246 | 4,116 | 10.0 % |
| HC-LM | 0.947* | 1 | 1.606* | 1.247* | 18 | – | – | 11.3 % |

* HC-LM NMSE vs BASE is 1 / 1.056 (the inverse of BASE vs HC-LM).
* "Knobs per deme" is, for each run, the median knob count of its demes, and then the median over
  the ok runs. "Total knobs" is the sum of the knob counts of a run's demes, median over the ok
  runs. The private chart page of 2026-11-16 (not in the repo) uses the same definitions. HC-LM logs no knob count (its "end"
  events have no `dim` field, and the bundle has no MOSES logs).

## Per dataset (exact values)

The user asked for the exact values on 2026-11-16. Bold marks the lowest median test NMSE. With
the failed runs counted as infinite (as in `aggregate.csv`), the medians are Dow BASE-D5 30.62,
Dow BASE-D20 24.21 and Tow BASE-D5 15.46.

### Test NMSE (median)

| Dataset | BASE | BASE-R | BASE-D5 | BASE-D10 | BASE-D20 | HC-LM |
|---|---|---|---|---|---|---|
| Air | 42.09 | 38.35 | 36.97 | 33.68 | 36.30 | **33.26** |
| Bos | **21.46** | 23.06 | 25.38 | 23.12 | 23.83 | 23.32 |
| Con | 30.91 | 23.33 | 23.79 | 23.55 | 19.51 | **18.89** |
| Dow | 39.29 | 35.04 | 30.10 (9) | 22.15 | **21.51** (9) | 33.70 |
| EnC | 11.17 | 10.95 | 11.11 | 10.67 | 10.62 | **10.58** |
| EnH | 7.71 | 7.58 | 7.56 | 7.37 | **7.07** | 7.57 |
| Tow | 18.36 | 16.14 | 14.17 (8) | **10.63** | 10.75 | 13.92 |
| WiR | 68.13 | 65.85 | **65.36** | 67.24 | 66.95 | 65.58 |
| WiW | 71.21 | 70.81 | **70.21** | 73.88 | 70.74 | 71.42 |
| Yac | 0.93 | 0.93 | **0.78** | 0.83 | 0.85 | 0.82 |

* Lowest median: BASE-D5 and HC-LM on 3 datasets each, BASE-D20 on 2, BASE and BASE-D10 on 1 each,
  BASE-R on none.
* Four ok runs have a test NMSE above 100: Bos BASE-R split 35 (60,308.97), WiR BASE-D10 split 32
  (2,184.86), WiW BASE-D10 split 35 (805.95) and Con HC-LM split 33 (699.93). All other ok runs
  are below 100. See "Observations", item 6.

### Program size (median)

| Dataset | BASE | BASE-R | BASE-D5 | BASE-D10 | BASE-D20 | HC-LM |
|---|---|---|---|---|---|---|
| Air | 70.5 | 78.5 | 83 | 106 | 105 | 144.5 |
| Bos | 56 | 48.5 | 73.5 | 86 | 82 | 93.5 |
| Con | 53.5 | 81 | 94.5 | 85 | 109.5 | 53 |
| Dow | 26 | 29 | 80 (9) | 114.5 | 113 (9) | 55 |
| EnC | 45.5 | 60.5 | 46.5 | 66 | 79.5 | 69 |
| EnH | 45.5 | 53 | 72 | 50.5 | 79.5 | 82.5 |
| Tow | 32.5 | 35 | 84.5 (8) | 129.5 | 104.5 | 58.5 |
| WiR | 63 | 74.5 | 59.5 | 73 | 90 | 96 |
| WiW | 62.5 | 73 | 102.5 | 106.5 | 116.5 | 122 |
| Yac | 41 | 41.5 | 37 | 46.5 | 42 | 70.5 |

### Train–test gap (median of test NMSE − training NMSE)

| Dataset | BASE | BASE-R | BASE-D5 | BASE-D10 | BASE-D20 | HC-LM |
|---|---|---|---|---|---|---|
| Air | 2.71 | 2.79 | 1.90 | 1.47 | 2.42 | 3.39 |
| Bos | 3.75 | 4.58 | 10.48 | 9.62 | 7.66 | 8.81 |
| Con | 2.01 | 2.07 | 1.22 | 0.36 | 1.78 | 1.56 |
| Dow | 1.29 | 2.91 | 1.52 (9) | 2.61 | 4.34 (9) | 2.27 |
| EnC | 1.01 | 1.11 | 0.88 | 0.62 | 1.11 | 0.89 |
| EnH | 0.17 | 0.34 | 0.35 | 0.39 | 0.26 | 0.64 |
| Tow | -0.54 | -0.45 | 0.03 (8) | -0.34 | 0.06 | -0.09 |
| WiR | 3.83 | 4.75 | 3.64 | 5.37 | 5.72 | 5.84 |
| WiW | -1.33 | -0.71 | -0.58 | 2.27 | 0.00 | 1.46 |
| Yac | 0.29 | 0.40 | 0.28 | 0.15 | 0.28 | 0.36 |

### CPU s per run (median)

Regression has no "solved" target. This is the time to use the 20,000 evaluations. The wall time
medians are the same within 0.05 %.

| Dataset | BASE | BASE-R | BASE-D5 | BASE-D10 | BASE-D20 | HC-LM |
|---|---|---|---|---|---|---|
| Air | 640 | 622 | 796 | 1,045 | 1,062 | 886 |
| Bos | 367 | 363 | 648 | 684 | 744 | 512 |
| Con | 449 | 445 | 895 | 764 | 1,091 | 293 |
| Dow | 338 | 353 | 1,279 (9) | 3,237 | 4,916 (9) | 776 |
| EnC | 379 | 369 | 545 | 593 | 816 | 402 |
| EnH | 341 | 334 | 583 | 466 | 719 | 421 |
| Tow | 1,100 | 1,100 | 3,600 (8) | 6,354 | 5,184 | 1,347 |
| WiR | 606 | 591 | 970 | 1,011 | 1,245 | 921 |
| WiW | 1,453 | 1,439 | 2,855 | 2,864 | 2,855 | 2,153 |
| Yac | 285 | 276 | 286 | 291 | 274 | 292 |

### Demes per run: median [min–max]

The count includes only demes with evaluations (`ledger.requests` > 0).

| Dataset | BASE | BASE-R | BASE-D5 | BASE-D10 | BASE-D20 | HC-LM |
|---|---|---|---|---|---|---|
| Air | 3 [2–4] | 5 [4–6] | 6 [6–6] | 10 [10–10] | 18 [17–20] | 13.5 [11–18] |
| Bos | 3 [2–4] | 6 [4–7] | 5 [5–6] | 10 [9–10] | 19 [18–20] | 21.5 [14–27] |
| Con | 2 [2–4] | 5 [4–9] | 5 [5–6] | 10 [10–10] | 19 [16–20] | 29.5 [17–35] |
| Dow | 1 [1–2] | 5 [3–6] | 5 [5–5] (9) | 10 [9–10] | 19 [15–20] (9) | 10 [7–12] |
| EnC | 3 [3–5] | 5.5 [5–7] | 5.5 [5–6] | 10 [10–10] | 18.5 [17–20] | 22 [13–30] |
| EnH | 3 [2–5] | 5 [4–7] | 5 [5–6] | 10 [10–10] | 18.5 [18–19] | 19 [12–23] |
| Tow | 1.5 [1–2] | 4.5 [3–6] | 5 [5–5] (8) | 10 [8–10] | 18 [16–20] | 15 [13–23] |
| WiR | 2 [2–3] | 4 [3–7] | 5 [5–6] | 10 [10–10] | 18 [17–19] | 20 [17–23] |
| WiW | 2 [2–2] | 4 [3–5] | 5 [5–5] | 10 [9–10] | 17.5 [16–19] | 16 [11–20] |
| Yac | 6 [4–8] | 8.5 [6–11] | 6.5 [6–10] | 10 [10–10] | 19 [19–20] | 23.5 [13–27] |

### Knob counts

HC-LM logs no knob count, so its column is empty.

Knobs per deme (for each run, the median knob count of its demes; then the median over the runs;
rounded to whole knobs):

| Dataset | BASE | BASE-R | BASE-D5 | BASE-D10 | BASE-D20 | HC-LM |
|---|---|---|---|---|---|---|
| Air | 56 | 106 | 109 | 135 | 124 | – |
| Bos | 117 | 184 | 233 | 235 | 283 | – |
| Con | 56 | 219 | 223 | 193 | 298 | – |
| Dow | 118 | 813 | 814 (9) | 2,147 | 2,319 (9) | – |
| EnC | 61 | 134 | 118 | 179 | 190 | – |
| EnH | 61 | 120 | 143 | 93 | 202 | – |
| Tow | 125 | 430 | 418 (8) | 1,127 | 950 | – |
| WiR | 92 | 168 | 187 | 202 | 249 | – |
| WiW | 56 | 185 | 372 | 225 | 277 | – |
| Yac | 72 | 86 | 74 | 64 | 71 | – |

Total knobs per run (sum over the demes, median over the runs):

| Dataset | BASE | BASE-R | BASE-D5 | BASE-D10 | BASE-D20 | HC-LM |
|---|---|---|---|---|---|---|
| Air | 190.5 | 417.5 | 568 | 1,156 | 2,043 | – |
| Bos | 314 | 989.5 | 1,012 | 2,356 | 4,988 | – |
| Con | 112 | 871.5 | 913 | 1,632 | 5,032 | – |
| Dow | 118 | 3,370 | 4,306 (9) | 17,410 | 38,551 (9) | – |
| EnC | 303.5 | 655 | 578 | 1,456 | 3,188 | – |
| EnH | 256.5 | 623.5 | 693.5 | 897.5 | 3,310 | – |
| Tow | 223 | 1,555 | 2,103 (8) | 9,048 | 15,478 | – |
| WiR | 183.5 | 719 | 892.5 | 1,912 | 4,087 | – |
| WiW | 112 | 826.5 | 1,368 | 1,966 | 4,473 | – |
| Yac | 351.5 | 605 | 399.5 | 610 | 1,284 | – |

Largest deme of a run, in knobs (median over the runs [highest]):

| Dataset | BASE | BASE-R | BASE-D5 | BASE-D10 | BASE-D20 | HC-LM |
|---|---|---|---|---|---|---|
| Air | 124 [158] | 132 [159] | 141 [206] | 157 [225] | 161 [260] | – |
| Bos | 184 [295] | 280.5 [381] | 357.5 [707] | 397 [966] | 375.5 [569] | – |
| Con | 91.5 [162] | 290 [429] | 288.5 [695] | 224 [497] | 366.5 [734] | – |
| Dow | 118 [929] | 987 [2,204] | 1,567 [3,195] (9) | 2,696 [5,742] | 2,724 [6,140] (9) | – |
| EnC | 176 [313] | 198.5 [313] | 225 [553] | 212 [385] | 254.5 [441] | – |
| EnH | 150 [207] | 207.5 [465] | 232.5 [522] | 145.5 [377] | 242 [781] | – |
| Tow | 196 [521] | 534.5 [712] | 626 [1,770] (8) | 1,327 [4,853] | 1,109 [1,566] | – |
| WiR | 157.5 [338] | 258.5 [567] | 297.5 [1,139] | 275.5 [456] | 316.5 [745] | – |
| WiW | 86 [133] | 444 [716] | 463 [1,198] | 293 [713] | 380 [688] | – |
| Yac | 104.5 [190] | 121 [190] | 100 [147] | 101 [149] | 102.5 [164] | – |

* The failed runs had larger demes: Dow BASE-D5 split 40 reached 4,811 knobs and Tow BASE-D5
  split 35 reached 2,555 (see "Failed runs").

### Per-dataset contrasts on test NMSE (descriptive, Holm over 40)

Each cell: pairs (A lower / B lower) · Holm p · pair ratio. Paired Wilcoxon on the test NMSE
differences, as pre-registered. With 10 pairs and Holm over 40, the lowest possible p is 0.078.
The pair ratio is an agent computation (a failed run is infinite).

| Dataset | BASE-R vs BASE | BASE-D5 vs BASE | BASE-D10 vs BASE | BASE-D20 vs BASE |
|---|---|---|---|---|
| Air | 9/0 (1) · 0.14 · 0.87 | 9/1 · 0.31 · 0.87 | 8/2 · 1.0 · 0.80 | 9/1 · 0.21 · 0.86 |
| Bos | 3/3 (4) · 1.0 · 1.00 | 4/6 · 1.0 · 1.07 | 4/6 · 1.0 · 1.02 | 4/6 · 1.0 · 1.04 |
| Con | 9/1 · 0.21 · 0.81 | 8/2 · 1.0 · 0.75 | 7/3 · 1.0 · 0.67 | 7/3 · 1.0 · 0.59 |
| Dow | 8/1 (1) · 0.79 · 0.96 | 9/1 · 1.0 · 0.85 | 10/0 · 0.078 · 0.55 | 8/2 · 1.0 · 0.62 |
| EnC | 6/2 (2) · 1.0 · 0.97 | 5/5 · 1.0 · 0.99 | 9/1 · 1.0 · 0.93 | 9/1 · 0.21 · 0.86 |
| EnH | 5/2 (3) · 1.0 · 0.99 | 9/1 · 0.42 · 0.97 | 8/2 · 1.0 · 0.92 | 9/1 · 0.21 · 0.88 |
| Tow | 7/1 (2) · 0.47 · 0.90 | 8/2 · 1.0 · 0.87 | 10/0 · 0.078 · 0.61 | 10/0 · 0.078 · 0.57 |
| WiR | 4/4 (2) · 1.0 · 1.00 | 6/4 · 1.0 · 0.99 | 5/5 · 1.0 · 1.00 | 5/5 · 1.0 · 1.00 |
| WiW | 6/4 · 1.0 · 1.00 | 5/5 · 1.0 · 1.00 | 3/7 · 1.0 · 1.03 | 6/4 · 1.0 · 1.00 |
| Yac | 5/2 (3) · 1.0 · 1.00 | 4/2 (4) · 1.0 · 1.00 | 6/3 (1) · 1.0 · 0.91 | 5/5 · 1.0 · 1.01 |

### Per-dataset pair ratio of test NMSE against HC-LM (descriptive, not tested)

| Dataset | BASE | BASE-R | BASE-D5 | BASE-D10 | BASE-D20 |
|---|---|---|---|---|---|
| Air | 1.27 | 1.13 | 1.12 | 1.02 | 1.10 |
| Bos | 1.06 | 1.07 | 1.15 | 1.05 | 0.91 |
| Con | 1.57 | 1.31 | 1.26 | 1.18 | 1.05 |
| Dow | 1.17 | 0.99 | 0.85 | 0.62 | 0.67 |
| EnC | 1.03 | 1.02 | 1.06 | 0.96 | 0.90 |
| EnH | 1.05 | 1.04 | 1.00 | 0.96 | 0.96 |
| Tow | 1.24 | 1.15 | 1.01 | 0.76 | 0.81 |
| WiR | 1.03 | 1.01 | 0.99 | 1.02 | 1.00 |
| WiW | 0.97 | 0.97 | 0.98 | 1.02 | 0.99 |
| Yac | 1.01 | 1.19 | 0.91 | 0.91 | 0.89 |

## BASE-R details

* **Same result as BASE:** in 18 of 100 pairs, BASE-R has the same final program and the same test
  NMSE as BASE (Bos 4, Yac 3, EnH 3, EnC 2, Tow 2, WiR 2, Air 1, Dow 1). These are the 18 ties of
  contrast 1.
* **LM in the first deme:** the median LM evaluations in the first deme are 12 (Con) to 41.5 (WiW)
  of the 500 reserved, with a highest value of 70. MOSES gave the rest to more demes, as in the
  pilot: 5 demes per run against 2 for BASE.
* **LM budget:** BASE got 0 LM evaluations in 12 runs (Dow 7, Tow 5); BASE-R in none. LM calls
  with 0 evaluations: BASE 100 of 277, BASE-R 2 of 536.
* **LM effect:** LM improved the training error in 494 of 536 calls (92 %). The guard rejected 101
  of 424 checked results (24 %). LM share of the evaluations: 1.5 % (BASE 0.2 %).
* **Size and CPU:** the same CPU as BASE (ratio 1.005); programs 4 % larger (52.5 against 48).

## D-arm details

* **Generations:** the first deme of a run had a median of 3 (D5), 2 (D10) and 1 (D20)
  generations. In the later demes, the median is 1 generation for all three arms. Later demes with
  more than 1 generation: D5 140 of 427 (33 %), D10 28 of 892 (3 %), D20 1 of 1,720. BASE has 9
  generations in the first deme (median) and 5 in the later demes.
* **LM:** LM calls 533 (D5), 992 (D10) and 1,826 (D20); improved 79 %, 88 % and 90 %; guard
  rejected 21 %, 24 % and 26 % of the checked results. LM calls with 0 evaluations: 95, 98 and 88,
  about one in each run (the last deme). LM share: 1.9 %, 4.9 % and 10.0 %.
* **Knob counts:** see "Knob counts". On Dow and Tow, the knob count of a deme grows with each new
  exemplar. The total knobs per run reach 38,551 (Dow D20) and 15,478 (Tow D20), against 118 and
  223 for BASE. On Air, EnC, EnH and Yac, the largest deme stays below 800 knobs in every run.

## HC-LM details

* Demes: 18 per run (median, range 7–35). LM share: 11.3 % (Dow 1.0 %, Tow 6.6 %, the other
  datasets 11.3–12.9 %).
* LM calls 1,858; improved 1,726 (93 %); guard rejected 449 of 1,720 checked results (26 %); 90
  calls with 0 evaluations (the last deme of a run). Round 10 gave 1,825 calls, 93 % and 26 %.

## Observations

Facts only. This round does not test them, and they give no recommendation.

1. **Deme count and test NMSE.** The pooled NMSE ratio against BASE falls as the deme count rises:
   BASE-R 0.984 (5 demes), D5 0.961 (5), D10 0.915 (10), D20 0.899 (18). Program size and CPU rise
   in the same order, except that BASE-R has the CPU of BASE.
2. **GOMEA and HC-LM.** With a deme count in the HC-LM range (10 and 18 against 18), D10 and D20
   are lower than HC-LM in 59 and 60 of 100 pairs (ties). Their programs are 0.94 and 1.05 times
   the HC-LM size (pooled medians 85.5 and 93.5 against 72; agent computation, ok pairs). Their
   CPU is 1.50 and 1.60 times HC-LM.
3. **Where the effect is.** The large effects are on Dow, Tow and Con (D20 pair ratio against
   BASE 0.57–0.62), and Air (0.80–0.87). On Bos, WiR, WiW and Yac, the pair ratios of the D arms
   against BASE are 0.91–1.07. On Dow and Tow, D10 and D20 are also lower than HC-LM (pair ratios
   0.62–0.81).
4. **Bos.** BASE has the lowest median test NMSE on Bos. The D arms have larger train–test gaps
   on Bos (7.7–10.5 against 3.8 for BASE), as HC-LM has (8.8).
5. **Knob growth.** On Dow and Tow, the D arms build demes with thousands of knobs. This
   coincides with the 4 memory failures, the CPU (Dow D20 4,916 s median against 338 s for BASE)
   and the 26.6 h phase.
6. **Single bad runs.** Four runs have a test NMSE far above their arm's range:

   | Run | Test NMSE | Training NMSE | Validation NMSE |
   |---|---|---|---|
   | Bos BASE-R, split 35 | 60,308.97 | 14.65 | 22.90 |
   | WiR BASE-D10, split 32 | 2,184.86 | 61.56 | 59.78 |
   | WiW BASE-D10, split 35 | 805.95 | 68.38 | 71.78 |
   | Con HC-LM, split 33 | 699.93 | 15.46 | 16.15 |

   Their training and validation NMSE are in the normal range. Thus these programs give very wrong
   values on some test rows, and the validation guard did not see it. The medians and the paired
   tests are not much affected by one run.
7. **LM without budget in BASE.** BASE got 0 LM evaluations in 12 of 100 runs (Dow 7, Tow 5),
   against 17 in round 10 (splits 1–10).

## Limits

* One evaluation budget (20,000). Equal evaluations are not equal CPU: the pooled median CPU goes
  from 458 s (BASE) to 1,155 s (D20).
* Splits 31–40 were not used before. No setting was chosen on them.
* After the first deme, a D-arm deme rarely has more than 1 generation (D10 28 of 892, D20 1 of
  1,720 later demes). The D arms test restarts plus LM, with a short GOMEA search in each deme
  (pre-registration, corrected 2026-11-11).
* BASE-R does not change only the LM budget: it also adds tiny demes with repeated LM.
* The D arms make more LM calls than BASE. This round cannot separate the effect of more demes from
  the effect of more LM calls.
* The 4 failed runs count as infinitely bad against D5 and D20 in the contrasts. A larger memory
  limit can change those 4 pairs.
* HC-LM logs no knob count, so the knob tables have no HC-LM column.
* The 10 % rule hides large per-dataset effects (pair ratios 0.55–1.07 against BASE). The
  per-dataset contrasts cannot give a verdict (lowest possible Holm p 0.078).
* `--const-opt-reserve` and the deme-share fix (709b27f) are new code. This round is their first
  use, after local tests only.

## User decisions

The pre-registration has no adoption rule. On 2026-11-16 the user asked: "Due to a lower program
size and a better running time, is it fair to pick the base configuration for the time being? While
the HC-LM being next in line incase the GOMEA can't find further tailored improvements?" The agent
answered yes, with the limits above (large losses on Con, Air and Tow; no LM budget in 12 BASE runs
on Dow and Tow). The user confirmed: "Yes, record the decision".

* **BASE stays the default** for the time being: smaller programs and less CPU than HC-LM.
* **HC-LM is next in line** if GOMEA finds no further improvements for these problems.
* **Next action:** task T1.5 in `ROADMAP.md`. Explain the BASE-R failure case (Bos split 35), and
  check BASE-R as a replacement for BASE on the datasets where BASE is weak (user, 2026-11-16).
