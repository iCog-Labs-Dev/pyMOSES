# Server round 12: findings (2026-11-18)

**Pre-registration:** `SERVER-ROUND12-PREREGISTRATION.md`. **Results:** bundle
`bundle-final-20261117T103333Z.tar.gz` (local copy: `../server-results-round12/`). **Code:**
`c26a3c5`. **Decisions:** `APPROVALS.md` (results only, no adoption rule; see "User decisions").

Test NMSE in the tables is × 100. "Pairs" is the number of splits where A is lower / where B is
lower; the remaining pairs are ties. "Ratio" is exp(median of log(A / B)) over the pairs: above 1
means that A has the higher value (higher NMSE, larger program or more CPU). A verdict of "better"
or "worse" needs Holm p < 0.025 and a ratio at least 10 % from 1. No run failed, so no run counts
as infinitely bad.

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

**Arms** (all at 20,000 evaluations, splits 41–50):

| Arm | Task | Change against BASE |
|---|---|---|
| BASE | reference | the round 11 BASE (dynamic population, LT-B, linear scaling, LM-VAL, PEN125) |
| HC-LM | accuracy baseline | hill climbing with LM-VAL |
| BASE-R | T1.5 | `--const-opt-reserve 500` |
| BASE-K5 | T1.6 | `--const-opt-reserve 500 --const-opt-top-k 5` |
| BASE-NP | T2.1 | `--gomea-fos np` in place of `lt-b` |
| BASE-TLS | T2.2 | `--linear-scaling terms --const-opt none` (no LM, no guard) |
| BASE-LME | T2.3 | `--const-opt-during elite` (LM on new deme bests, non-zero constants only) |
| BASE-DIV | T2.4 | `--diversity-pressure {6*n_train} --diversity-autoscale 0` |

## Summary

| Contrast | Family | Pairs | Holm p | Ratio | Verdict |
|---|---|---|---|---|---|
| BASE-R vs BASE | primary | 63/11 | 1.9e-9 | 0.979 | tie |
| BASE-K5 vs BASE-R | primary | 53/44 | 0.20 | 0.998 | tie |
| BASE-NP vs BASE | primary | 28/72 | 1.9e-6 | 1.058 | tie |
| **BASE-TLS vs BASE** | primary | **73/27** | **1.9e-6** | **0.879** | **A better** |
| BASE-LME vs BASE | primary | 73/27 | 1.9e-6 | 0.929 | tie |
| BASE-DIV vs BASE | primary | 49/32 | 0.20 | 1.000 | tie |

**Answers to the three questions:**

1. **T1.6, LM on the top 5 programs of each deme: no gain.** BASE-K5 ties with BASE-R (0.998,
   53/44). Against BASE, BASE-K5 has a ratio of 0.959 (70/29), which is about the BASE-R gain.
2. **The four tier 1 ideas:**
   * **T2.2, per-term least squares (BASE-TLS): "A better", the only verdict of the round.** The
     test NMSE is 12 % lower at the median pair. The programs are 18 % larger in nodes, and 40 %
     larger when the fitted term weights are counted. The CPU is the same (1.047, a tie).
   * **T2.3, LM during the search (BASE-LME): 7 % lower, a tie under the 10 % rule.** It is lower
     in 73 of 100 pairs. Its programs are 27 % larger ("A worse" on size).
   * **T2.1, node-proximity linkage (BASE-NP): 6 % higher, a tie.** It is cheaper: 26 % less CPU
     ("A better" on CPU) and 8 % smaller programs (a tie). Against HC-LM it is "A worse" (1.167).
     On mux11 it had the lowest median CPU to solve, but all three Boolean contrasts tie.
   * **T2.4, diversity pressure (BASE-DIV): no effect on accuracy** (1.000). It used 2.44 times the
     CPU of BASE ("A worse").
3. **BASE-R confirms its round 11 result on new splits:** 0.979 (round 11: 0.984), lower than BASE
   in 63 of the 74 pairs that differ, at the same CPU. It stays a tie under the 10 % rule.

* The verdicts are descriptive. The round reports results only (pre-registration, "Rules").
* Against HC-LM, BASE has a ratio of 1.065 (28/72, a tie, as in rounds 10 and 11). BASE-TLS has
  0.968 (54/46, a tie). In round 11, the deme caps D10 and D20 also had medians below HC-LM (0.973
  and 0.971, ties).
* All 840 runs ended ok. The round took 20.0 h, at the top of the estimate (17–20 h).

## Run health

* **Steps:** S0 1.34 h (4,807 s), phase1bf 17.62 h (63,445 s), phase1bg 1.04 h (3,740 s). One
  orchestrator session, 2026-11-16 14:33Z to 2026-11-17 10:33Z. W = 11; W = 7 for the Dow
  manifest (limited by memory). `bundle.err` is empty.
* **S0:** reproducibility 4 of 4 matched. pytest: 9,740 passed, 1 failed
  (`test_toolchain_drift_is_clean_on_the_real_repo`), 13 skipped. The same test failed in rounds
  6 to 11: the server's toolchain differs from the toolchain that the fixtures record. It is not
  a code fault.
* **Runs:** phase1bf 800 ok, 0 failed; phase1bg 40 ok, 0 failed. Every phase1bf run used exactly
  20,000 evaluations.
* **Memory:** the highest peak RSS was 2,688 MB (52 % of the 5,120 MB limit; Bos BASE-DIV s47)
  and 3,592 MB on Dow (44 % of 8,192 MB; Dow HC-LM s41). Round 11 had 4 memory failures in the
  deme-cap arms; round 12 has none. Peak RSS per arm: BASE 1,571 MB, HC-LM 3,592, BASE-R 1,499,
  BASE-K5 2,418, BASE-NP 751, BASE-TLS 1,558, BASE-LME 776, BASE-DIV 2,688.
* **CPU:** 176.7 CPU hours in phase1bf. BASE-DIV used 57.1 h of it; the other arms used 12.0–22.4 h
  each (BASE 16.3, HC-LM 22.4, BASE-R 16.5, BASE-K5 16.7, BASE-NP 12.0, BASE-TLS 18.3, BASE-LME
  17.4).

## Phase 1bf: the gap to HC-LM (Holm over 7)

| Contrast | Pairs | Holm p | Ratio | Verdict |
|---|---|---|---|---|
| BASE vs HC-LM | 28/72 | 4.9e-6 | 1.065 | tie |
| BASE-R vs HC-LM | 35/65 | 0.0022 | 1.043 | tie |
| BASE-K5 vs HC-LM | 38/62 | 0.016 | 1.021 | tie |
| BASE-NP vs HC-LM | 19/81 | 1.0e-9 | 1.167 | **A worse** |
| BASE-TLS vs HC-LM | 54/46 | 0.027 | 0.968 | tie |
| BASE-LME vs HC-LM | 46/54 | 0.69 | 1.006 | tie |
| BASE-DIV vs HC-LM | 33/67 | 6.9e-5 | 1.051 | tie |

## Phase 1bf: program size and CPU (Holm over 12)

| Contrast | Size pairs | Size ratio | Size verdict | CPU pairs | CPU ratio | CPU verdict |
|---|---|---|---|---|---|---|
| BASE-R vs BASE | 11/54 | 1.022 | tie | 40/60 | 1.004 | tie |
| BASE-K5 vs BASE-R | 48/42 | 1.000 | tie | 66/34 | 0.978 | tie |
| BASE-NP vs BASE | 58/40 | 0.915 | tie | 88/12 | 0.744 | **A better** |
| BASE-TLS vs BASE | 34/63 | 1.178 | **A worse** | 41/59 | 1.047 | tie |
| BASE-LME vs BASE | 34/64 | 1.268 | **A worse** | 43/57 | 1.041 | tie |
| BASE-DIV vs BASE | 34/45 | 1.000 | tie | 1/99 | 2.440 | **A worse** |

* Size is the node count of the final program. For BASE-TLS it does not count the fitted term
  weights (`terms` record). With the weights, the BASE-TLS median is 71.0 against 48.5, a pair
  ratio of 1.398 (not a registered contrast).
* **Descriptive, BASE-K5 vs BASE:** 70/29, ratio 0.959, Holm p 4.5e-7 in a family of one. The
  `verdict.json` value ("tie") is descriptive only.

## Trade-off table (pre-registered)

Pooled over the 100 pairs. Ratios are pair ratios against BASE (or HC-LM). Demes, knobs and the LM
share are medians over the runs. "Knobs per deme" is the median of the per-run medians.

| Arm | NMSE / BASE | NMSE / HC-LM | Size / BASE | CPU / BASE | Demes | Knobs per deme | Total knobs | LM share |
|---|---|---|---|---|---|---|---|---|
| BASE | – | 1.065 | – | – | 3 | 68.5 | 257 | 0.40 % |
| HC-LM | 0.939 | – | 1.675 | 1.349 | 18 | not logged | not logged | 12.37 % |
| BASE-R | 0.979 | 1.043 | 1.022 | 1.004 | 5 | 176.0 | 778.5 | 1.65 % |
| BASE-K5 | 0.959 | 1.021 | 1.061 | 0.996 | 3 | 86.0 | 394.5 | 4.35 % |
| BASE-NP | 1.058 | 1.167 | 0.915 | 0.744 | 2 | 56.0 | 118 | 0.16 % |
| BASE-TLS | 0.879 | 0.968 | 1.178 (1.398 with weights) | 1.047 | 2 | 67.5 | 158 | none (no LM) |
| BASE-LME | 0.929 | 1.006 | 1.268 | 1.041 | 3 | 76.5 | 204 | 3.50 % |
| BASE-DIV | 1.000 | 1.051 | 1.000 | 2.440 | 3 | 73.8 | 236 | 0.42 % |

**Other medians (pooled over 100 runs):**

| Arm | Test NMSE | Train NMSE | Gap | Size | CPU s | LM calls | Guard rejected (all LM results) |
|---|---|---|---|---|---|---|---|
| BASE | 27.60 | 21.20 | 1.00 | 48.5 | 453 | 3 | 40 of 192 (20.8 %) |
| HC-LM | 20.80 | 15.99 | 1.21 | 79.0 | 663 | 18 | 462 of 1,704 (27.1 %) |
| BASE-R | 23.36 | 20.11 | 0.85 | 58.0 | 451 | 5 | 102 of 439 (23.2 %) |
| BASE-K5 | 23.90 | 19.95 | 0.77 | 57.0 | 441 | 14 | 317 of 1,446 (21.9 %) |
| BASE-NP | 34.23 | 28.18 | 0.85 | 42.5 | 302 | 2 | 29 of 123 (23.6 %) |
| BASE-TLS | 16.59 | 13.80 | 1.15 | 61.0 | 410 | – | – |
| BASE-LME | 22.70 | 17.94 | 0.60 | 58.5 | 457 | 10 | 258 of 882 (29.3 %) |
| BASE-DIV | 26.75 | 22.99 | 0.80 | 55.0 | 1,513 | 3 | 45 of 187 (24.1 %) |

The pooled medians mix datasets with different NMSE scales. Use the pair ratios for comparisons.

## Per dataset (exact values)

### Test NMSE (median)

| Dataset | BASE | HC-LM | BASE-R | BASE-K5 | BASE-NP | BASE-TLS | BASE-LME | BASE-DIV |
|---|---|---|---|---|---|---|---|---|
| Air | 37.57 | 34.19 | 36.49 | 39.04 | 41.09 | 36.94 | 35.72 | 37.74 |
| Bos | 23.03 | 24.74 | 22.72 | 21.59 | 24.78 | 24.31 | 21.66 | 22.16 |
| Con | 34.16 | 18.03 | 24.16 | 25.42 | 37.44 | 26.36 | 29.19 | 32.24 |
| Dow | 39.61 | 33.83 | 37.44 | 35.67 | 43.30 | **14.88** | 30.07 | 39.61 |
| EnC | 10.90 | 11.66 | 10.92 | 10.56 | 10.88 | **5.54** | 9.79 | 10.81 |
| EnH | 7.99 | 6.88 | 7.68 | 7.21 | 8.02 | **2.84** | 7.14 | 7.60 |
| Tow | 18.41 | 15.21 | 16.19 | 15.49 | 20.95 | **8.16** | 13.49 | 18.60 |
| WiR | 64.99 | 65.41 | 64.58 | 64.66 | 64.42 | 63.96 | 64.51 | 65.50 |
| WiW | 72.91 | 71.63 | 72.88 | 72.44 | 74.46 | 78.71 | 73.35 | 73.19 |
| Yac | 0.83 | 0.64 | 0.83 | 0.68 | 1.22 | 1.22 | 0.83 | 0.87 |

### Program size (median nodes; BASE-TLS without its term weights)

| Dataset | BASE | HC-LM | BASE-R | BASE-K5 | BASE-NP | BASE-TLS | BASE-LME | BASE-DIV |
|---|---|---|---|---|---|---|---|---|
| Air | 78.0 | 146.0 | 88.0 | 77.5 | 71.0 | 70.0 | 72.5 | 77.5 |
| Bos | 54.0 | 105.0 | 64.0 | 65.5 | 55.0 | 60.5 | 68.5 | 62.5 |
| Con | 61.5 | 88.5 | 66.0 | 64.0 | 67.0 | 65.5 | 51.0 | 71.0 |
| Dow | 23.5 | 49.0 | 28.5 | 28.0 | 22.0 | 32.0 | 53.5 | 23.5 |
| EnC | 46.0 | 70.5 | 54.0 | 60.0 | 34.0 | 60.5 | 65.0 | 56.0 |
| EnH | 47.0 | 76.5 | 55.5 | 51.5 | 43.5 | 55.5 | 61.0 | 44.0 |
| Tow | 31.5 | 67.0 | 36.0 | 45.0 | 26.5 | 146.5 | 53.5 | 29.0 |
| WiR | 62.5 | 85.5 | 62.5 | 56.5 | 27.0 | 63.0 | 54.0 | 55.0 |
| WiW | 77.0 | 102.0 | 81.0 | 78.0 | 82.5 | 70.5 | 64.0 | 77.5 |
| Yac | 35.0 | 61.0 | 35.0 | 37.5 | 37.5 | 47.0 | 40.5 | 41.0 |

### Train–test gap (median of test NMSE − training NMSE)

| Dataset | BASE | HC-LM | BASE-R | BASE-K5 | BASE-NP | BASE-TLS | BASE-LME | BASE-DIV |
|---|---|---|---|---|---|---|---|---|
| Air | 1.68 | 5.15 | 1.59 | 3.72 | 1.98 | 3.61 | -0.02 | 2.01 |
| Bos | 6.46 | 9.54 | 6.18 | 6.01 | 4.96 | 7.33 | 5.93 | 5.29 |
| Con | 2.29 | 2.99 | 2.25 | 2.02 | 2.57 | 3.56 | 3.46 | 2.25 |
| Dow | 2.42 | 1.42 | 1.71 | 2.40 | 2.29 | 1.92 | 0.51 | 2.42 |
| EnC | 1.08 | 1.11 | 0.89 | 0.81 | 1.03 | 0.60 | 0.64 | 0.81 |
| EnH | 0.39 | 0.40 | 0.23 | 0.31 | 0.18 | 0.07 | 0.39 | 0.49 |
| Tow | 0.18 | 0.90 | 0.21 | -0.03 | 0.64 | 0.44 | 0.52 | 0.21 |
| WiR | 1.24 | 3.31 | 1.22 | 0.48 | 0.06 | 4.39 | 0.58 | 1.22 |
| WiW | 2.21 | 2.17 | 2.80 | 1.69 | 1.53 | **11.93** | 2.16 | 1.68 |
| Yac | 0.09 | 0.04 | 0.17 | 0.06 | 0.17 | 0.06 | 0.22 | 0.03 |

### CPU s per run (median)

| Dataset | BASE | HC-LM | BASE-R | BASE-K5 | BASE-NP | BASE-TLS | BASE-LME | BASE-DIV |
|---|---|---|---|---|---|---|---|---|
| Air | 617 | 979 | 617 | 614 | 561 | 573 | 631 | 2,752 |
| Bos | 353 | 611 | 355 | 359 | 271 | 343 | 338 | 751 |
| Con | 437 | 420 | 437 | 421 | 296 | 422 | 415 | 742 |
| Dow | 303 | 781 | 326 | 314 | 287 | 358 | 344 | 348 |
| EnC | 374 | 361 | 373 | 412 | 242 | 360 | 407 | 2,012 |
| EnH | 390 | 414 | 392 | 323 | 217 | 322 | 361 | 1,708 |
| Tow | 945 | 1,374 | 953 | 932 | 841 | 1,589 | 1,124 | 1,564 |
| WiR | 493 | 900 | 501 | 480 | 380 | 568 | 528 | 1,736 |
| WiW | 1,367 | 1,839 | 1,416 | 1,434 | 958 | 1,581 | 1,571 | 7,205 |
| Yac | 269 | 317 | 266 | 237 | 266 | 335 | 198 | 430 |

### Demes per run (median)

| Dataset | BASE | HC-LM | BASE-R | BASE-K5 | BASE-NP | BASE-TLS | BASE-LME | BASE-DIV |
|---|---|---|---|---|---|---|---|---|
| Air | 3 | 14.5 | 5 | 3 | 3 | 3 | 3 | 3 |
| Bos | 3 | 17.5 | 5 | 3.5 | 2 | 2.5 | 2 | 3 |
| Con | 3 | 23 | 5.5 | 4 | 2 | 2 | 3 | 2.5 |
| Dow | 1 | 10 | 5 | 3 | 1 | 1 | 1 | 1 |
| EnC | 4 | 23 | 6 | 4 | 2 | 3 | 4 | 4 |
| EnH | 3 | 20 | 5 | 3 | 2 | 2 | 3 | 3 |
| Tow | 1 | 15 | 4.5 | 2.5 | 1 | 2 | 1 | 1 |
| WiR | 3 | 20 | 5 | 3 | 2 | 2 | 2 | 3 |
| WiW | 2 | 17.5 | 4 | 2 | 2 | 2 | 2 | 2 |
| Yac | 6 | 20.5 | 8 | 6.5 | 5.5 | 5 | 7 | 5.5 |

### Knob counts (GOMEA arms; HC-LM logs no knob count, `BACKLOG.md` 10.4)

Knobs per deme (median of the per-run medians):

| Dataset | BASE | BASE-R | BASE-K5 | BASE-NP | BASE-TLS | BASE-LME | BASE-DIV |
|---|---|---|---|---|---|---|---|
| Air | 55.5 | 75.5 | 47.5 | 50.0 | 44.0 | 56.0 | 59.0 |
| Bos | 113.5 | 260.5 | 152.2 | 82.2 | 103.2 | 96.2 | 120.5 |
| Con | 69.5 | 210.2 | 107.2 | 53.8 | 55.8 | 92.0 | 67.0 |
| Dow | 118.0 | 783.5 | 784.5 | 118.0 | 118.0 | 118.0 | 118.0 |
| EnC | 83.0 | 118.0 | 91.8 | 44.8 | 60.5 | 74.0 | 69.2 |
| EnH | 60.5 | 142.5 | 76.8 | 42.5 | 47.0 | 56.0 | 60.5 |
| Tow | 54.0 | 469.5 | 287.5 | 54.0 | 313.8 | 54.0 | 54.0 |
| WiR | 121.5 | 168.8 | 124.8 | 73.8 | 68.0 | 82.5 | 100.8 |
| WiW | 58.8 | 190.0 | 73.5 | 59.0 | 82.8 | 68.0 | 56.0 |
| Yac | 65.0 | 72.0 | 63.2 | 58.2 | 62.5 | 44.0 | 61.5 |

Total knobs per run (median):

| Dataset | BASE | BASE-R | BASE-K5 | BASE-NP | BASE-TLS | BASE-LME | BASE-DIV |
|---|---|---|---|---|---|---|---|
| Air | 201.5 | 351.0 | 198.5 | 180.0 | 170.5 | 174.0 | 212.0 |
| Bos | 383.0 | 1,022.5 | 620.5 | 164.5 | 243.0 | 192.5 | 385.0 |
| Con | 270.0 | 927.5 | 396.0 | 107.5 | 111.5 | 226.5 | 166.0 |
| Dow | 118.0 | 3,399.0 | 1,745.0 | 118.0 | 118.0 | 118.0 | 118.0 |
| EnC | 321.0 | 634.5 | 406.5 | 89.5 | 240.0 | 329.0 | 349.0 |
| EnH | 328.5 | 737.0 | 291.0 | 85.0 | 94.0 | 253.5 | 243.0 |
| Tow | 54.0 | 1,556.5 | 575.0 | 54.0 | 627.5 | 54.0 | 54.0 |
| WiR | 331.5 | 789.5 | 346.5 | 147.5 | 136.0 | 165.0 | 299.5 |
| WiW | 117.5 | 825.5 | 147.0 | 118.0 | 165.5 | 136.0 | 112.0 |
| Yac | 366.5 | 529.5 | 385.5 | 304.0 | 323.5 | 332.0 | 341.0 |

### Per-dataset contrasts on test NMSE (descriptive, no verdicts)

Each cell: A lower / B lower, then the pair ratio over the 10 splits.

| Dataset | BASE-R / BASE | BASE-K5 / BASE-R | BASE-NP / BASE | BASE-TLS / BASE | BASE-LME / BASE | BASE-DIV / BASE | BASE / HC-LM | BASE-TLS / HC-LM |
|---|---|---|---|---|---|---|---|---|
| Air | 7/1 0.97 | 2/8 1.04 | 2/8 1.10 | 7/3 0.97 | 7/3 0.92 | 5/3 1.00 | 1/9 1.10 | 1/9 1.10 |
| Bos | 5/2 1.00 | 4/5 1.00 | 4/6 1.03 | 5/5 1.02 | 7/3 0.95 | 8/2 0.98 | 4/6 1.06 | 5/5 1.01 |
| Con | 9/1 0.88 | 4/6 1.01 | 2/8 1.22 | 7/3 0.83 | 6/4 0.92 | 5/5 0.98 | 0/10 1.47 | 0/10 1.41 |
| Dow | 9/0 0.93 | 5/4 1.00 | 2/8 1.09 | 10/0 0.39 | 10/0 0.84 | 0/0 1.00 | 3/7 1.06 | 10/0 0.47 |
| EnC | 5/2 0.99 | 7/3 0.98 | 5/5 1.00 | 10/0 0.51 | 8/2 0.92 | 8/2 0.98 | 5/5 1.00 | 10/0 0.52 |
| EnH | 7/0 0.97 | 7/3 1.00 | 3/7 1.02 | 10/0 0.40 | 8/2 0.85 | 5/4 1.00 | 3/7 1.09 | 9/1 0.45 |
| Tow | 10/0 0.92 | 8/2 0.95 | 1/9 1.13 | 10/0 0.44 | 10/0 0.79 | 1/3 1.00 | 1/9 1.19 | 9/1 0.55 |
| WiR | 3/3 1.00 | 5/4 1.00 | 4/6 1.00 | 8/2 0.99 | 6/4 0.99 | 6/4 0.99 | 5/5 1.00 | 6/4 0.99 |
| WiW | 6/2 1.00 | 4/6 1.00 | 2/8 1.02 | 4/6 1.07 | 6/4 0.99 | 5/5 1.00 | 4/6 1.01 | 4/6 1.10 |
| Yac | 2/0 1.00 | 7/3 0.91 | 3/7 1.30 | 2/8 1.61 | 5/5 0.95 | 6/4 0.90 | 2/8 1.18 | 0/10 1.91 |

The per-dataset Holm p values in `contrasts.csv` (Holm over 60) are at least 0.117, so no
per-dataset contrast is significant. With 10 pairs the lowest possible two-sided p is 0.00195.

## Runs with a test NMSE above 100

| Run | Test | Train | Validation | Size |
|---|---|---|---|---|
| Dow HC-LM s44 | 118.8 | 21.71 | 29.14 | 72 |
| WiW BASE-LME s43 | 144.2 | 70.78 | 69.28 | 128 |

Both runs have normal training and validation NMSE. The report (`phase_report.py`) does not make
this list; it comes from the run summaries (RV16 finding 1 of the round 12 build). Round 11 had 4
such runs.

## Arm details

### BASE-TLS (T2.2)

* All 100 runs wrote a `terms` record. No final program used the fallback to plain scaling. The
  share of fallbacks during the search is not in the telemetry.
* Terms of the final program, median [min–max]:

  | Air | Bos | Con | Dow | EnC | EnH | Tow | WiR | WiW | Yac |
  |---|---|---|---|---|---|---|---|---|---|
  | 8.5 [7–10] | 1 [1–13] | 10.5 [8–11] | 25 [21–30] | 9.5 [7–11] | 10 [8–11] | 21.5 [12–26] | 10 [10–12] | 12.5 [9–14] | 1 [1–2] |

* The largest absolute term weight, median (max): Air 132 (417), Bos 21.1 (89.9), Con 57.2 (346),
  Dow 27.7 (79), EnC 1,350 (5,820), EnH 1,750 (7,480), Tow 9,760 (22,600), WiR 6.75 (23.3), WiW
  160 (284), Yac 21.8 (60.5).
* The gains are on the datasets whose programs have many terms: Dow (25 terms, ratio 0.39), Tow
  (21.5, 0.44), EnH (10, 0.40), EnC (9.5, 0.51) and Con (10.5, 0.83).
* On Bos and Yac the median final program has 1 term. A one-term program gets the plain `a + b·f`
  score, so there BASE-TLS is BASE without LM. Yac is worse (1.61, 2/8), Bos ties (1.02).
* WiW is worse (1.07, 4/6) with a large train–test gap (11.93 against 2.21 for BASE). BASE-TLS has
  no validation guard.
* Tow programs are large: 146.5 nodes at the median, plus about 21 weights.

### BASE-LME (T2.3)

| Dataset | Elite LM calls per run (median) | Kept | Guard rejected | Worse | LM share (median) |
|---|---|---|---|---|---|
| Air | 8.5 | 72 of 89 | 16 | 0 | 4.08 % |
| Bos | 9.0 | 54 of 94 | 39 | 1 | 5.13 % |
| Con | 6.0 | 55 of 70 | 14 | 0 | 3.34 % |
| Dow | 3.5 | 34 of 39 | 4 | 0 | 1.94 % |
| EnC | 7.0 | 47 of 61 | 11 | 1 | 3.74 % |
| EnH | 6.5 | 55 of 72 | 13 | 2 | 3.85 % |
| Tow | 3.5 | 33 of 41 | 0 | 0 | 2.04 % |
| WiR | 8.5 | 52 of 87 | 35 | 0 | 4.55 % |
| WiW | 7.0 | 45 of 78 | 27 | 1 | 4.37 % |
| Yac | 7.5 | 53 of 75 | 12 | 10 | 3.46 % |

* The LM share stays far below the 50 % limit of the pilot gate (median 3.50 %).
* The largest gains are on the one-deme datasets, Tow (0.79, 10/0) and Dow (0.84, 10/0), and on
  EnH (0.85). The programs grow on six datasets (Dow 23.5 → 53.5, Tow 31.5 → 53.5); they shrink on
  Air, Con, WiR and WiW.

### BASE-K5 (T1.6)

* Each deme made 5 LM picks at the median (all datasets). LM share: median 4.35 % (1.35 % on Dow,
  7.31 % on Yac).
* The per-dataset ratios of BASE-K5 against BASE-R are 0.95–1.01, except Air (1.04, 2/8) and Yac
  (0.91, 7/3). None is significant.

### BASE-R (T1.5)

* 26 of 100 runs gave the same final program as BASE (Yac 8, WiR 4).
* Gains on Con (0.88, 9/1), Tow (0.92, 10/0) and Dow (0.93, 9/0), as in round 11.
* BASE-R has more demes (5 against 3) and more knobs per deme (176 against 68.5; Dow 783.5 against
  118).

### BASE-NP (T2.1)

* Fewer demes (2 against 3) and fewer knobs per run (118 against 257).
* Worse on Con (1.22), Yac (1.30), Tow (1.13) and Air (1.10); no dataset is better.

### BASE-DIV (T2.4)

* It gave the same final program as BASE in 19 runs: Dow 10 (one deme, as expected), Tow 6, Air 2,
  EnH 1.
* The CPU rise is largest on WiW (7,205 s against 1,367) and Air (2,752 against 617). The
  diversity step recomputes behavioural scores in the merge; that CPU is not counted as
  evaluations (RV16 finding 6).

### HC-LM

* 18 demes per run, 18 LM calls, guard rejected 27.1 % of the LM results, LM share 12.4 %.

## Phase 1bg (mux11, CPU to solve)

All 40 runs solved mux11 within the 3,600 s budget.

| Arm | Solved | CPU to solve, median [min–max] s | Evaluations to solve (median) | CPU per evaluation (median) |
|---|---|---|---|---|
| NP | 10 | 424 [258–959] | 60,962 | 0.0078 s |
| LT-B | 10 | 547 [269–822] | 58,492 | 0.0121 s |
| UNI | 10 | 815 [258–1,886] | 65,666 | 0.0145 s |
| RT | 10 | 979 [401–1,273] | 86,724 | 0.0123 s |

| Contrast (Holm over 3) | Pairs | Holm p | Verdict |
|---|---|---|---|
| NP vs RT | 8/2 | 0.041 | tie |
| NP vs UNI | 8/2 | 0.074 | tie |
| NP vs LT-B | 6/4 | 0.49 | tie |

* NP needs about the same evaluations as LT-B. It is faster because each evaluation costs less:
  the node-proximity FOS needs no population statistics.

## Observations (agent reading)

1. **Per-term least squares is the strongest single idea so far.** It gave the only "better"
   verdict of the round. Its gain comes from programs with many top-level terms. Where the
   programs have one term (Bos, Yac), it loses the LM step and is worse or equal.
2. **BASE-TLS has two risks that this round did not show as failures:** very large term weights
   on EnC, EnH and Tow (cancellation), and no validation guard (the WiW gap). None of its 100 runs
   had a test NMSE above 100.
3. **LM during the search gives a consistent but smaller gain** (lower in 73 of 100 pairs), mostly
   on the one-deme datasets, with larger programs.
4. **More LM at deme end (top 5) adds nothing to the reserve.** The BASE-R gain repeats on new
   splits.
5. **Node-proximity linkage is a cost saver, not an accuracy gain**, on regression and on mux11.
   It gives 6 % higher NMSE for 26 % less CPU and 8 % smaller programs.
6. **Diversity pressure in the merge does not help** at P = 6 × training rows, and it is expensive.
7. **The winners do not combine without new code.** `--linear-scaling terms` needs `--const-opt
   none` (`problem_params.py`), so BASE-TLS cannot take the reserve, top-k LM or LM during the
   search. BASE-TLS with `--gomea-fos np` needs no new code. LM on the inner constants of each term
   with least squares for the term weights is tier 2 task T3.1 (`ROADMAP.md`).

## Deviations from the pre-registration

* None in the arms, budgets, splits, seeds or contrasts.
* The pre-registration asks the report to list the runs with a test NMSE above 100. The report
  does not; this file gives the list (from the run summaries).
* The share of BASE-TLS fallbacks during the search is not recorded (known before the round, RV16
  finding 2).

## Limits

* One budget (20,000 evaluations) and one value for each new option (K = 5, P = 6 × rows).
* BASE-TLS changes two things against BASE: per-term weights in place of `a + b·f`, and no LM.
* The BASE-TLS size does not count its weights; the size contrast understates its program size.
* The BASE-TLS result is from one set of splits (41–50). No fresh splits exist after split 50.
* BASE-NP ran at the dynamic population size (N about 100), not at the N of its source paper
  (1,024).
* HC-LM logs no knob count, so its deme size cannot be compared with GOMEA (`BACKLOG.md` 10.4).
* The pooled medians mix datasets with different NMSE scales; the pair ratios are the comparison.

## User decisions

* **2026-11-18:** the agent presented the results. The user answered: "Yes, write the findings and
  update the records. And set up the next set of tests we had planned next."
  * User-approved: this findings file and the record updates, committed locally.
  * Not decided yet: which round 12 settings change the reference. BASE stays the reference until
    the user decides (agent reading, as for earlier rounds).
