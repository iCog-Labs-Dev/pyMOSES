# Server round 14: findings (2026-11-20)

**Pre-registration:** `SERVER-ROUND14-PREREGISTRATION.md` (with its "Correction" and "Amendment"
blocks). **Results:** bundle `bundle-final-20261120T003450Z.tar.gz` (SHA-256 `5e5b9221…72f59b5`;
local copy in the MOSES folder, next to the repo). **Code:** `d6dadac`. **Decisions:**
`APPROVALS.md`.

Test NMSE in the tables is × 100. "Pairs" is the number of splits where A is lower / where B is
lower; the remaining pairs are ties. "Ratio" is exp(median of log(A / B)) over the pairs: above 1
means that A has the higher value (higher NMSE, larger program or more CPU). A verdict of "better"
or "worse" needs Holm p < 0.025 and a ratio at least 10 % from 1. All 1,000 runs ended ok.

**Size** (pre-registered): the node count, plus the number of terms when the final model uses the
per-term fit. A linear baseline with k inputs has the size 2k + 2 (erratum of the
pre-registration). The node count alone is also given.

**Datasets:** Air, Bos, Con, Dow, EnC, EnH, Tow, WiR, WiW, Yac. Fresh splits 61–70.

**Arms** (the MOSES arms at 20,000 evaluations):

| Arm | Question | Change |
|---|---|---|
| BASE-TLS | reference | the round 13 BASE-TLS (`--linear-scaling terms --const-opt none`) |
| TLS-NP | next in line (Q4) | BASE-TLS with `--gomea-fos np` |
| HC-LM | accuracy baseline | hill climbing with LM-VAL |
| TLS-LMR | Q1 (T3.1 with a budget) | BASE-TLS + `--const-opt lm --const-opt-guard val --const-opt-reserve 500` |
| TLSNP-LMR | Q2, per-term | TLS-NP + the same three LM options |
| NP-LMR | Q2, plain scaling | the round 12 BASE-NP + `--const-opt-reserve 500` |
| NP-NOLM | Q2, plain scaling | NP-LMR with `--const-opt none`, no guard, no reserve |
| NP-LMR-PSR | Q3 (relative pole rule) | NP-LMR + `--const-opt-pole-check box-rel` |
| LIN-OLS | Q5 | ordinary least squares on all inputs, no MOSES run |
| LIN-RIDGE | Q5 | ridge on all inputs, λ chosen on the validation rows, no MOSES run |

## Summary

| Contrast | Family | Pairs | Ties | Holm p | Ratio | Verdict |
|---|---|---|---|---|---|---|
| TLS-LMR vs BASE-TLS | primary | 63/23 | 14 | 2.6e-6 | 0.980 | tie |
| TLSNP-LMR vs TLS-NP | primary | 64/22 | 14 | 1.5e-6 | 0.974 | tie |
| NP-LMR vs NP-NOLM | primary | 75/21 | 4 | 7.2e-8 | 0.950 | tie |
| NP-LMR-PSR vs NP-LMR | primary | 7/16 | 77 | 0.056 | 1.000 | tie |
| TLS-NP vs BASE-TLS | primary | 37/63 | 0 | 0.0068 | 1.029 | tie |

**Answers to the questions of the round:**

1. **Q1, LM inside the terms with a budget (TLS-LMR): a tie with a small gain.** The test NMSE is
   2 % lower at the median pair, and lower in 63 of the 86 pairs that differ. The Holm p is below
   0.025, but the ratio is inside the 10 % band. The reserve solved the budget problem of round 13:
   2 of 509 LM calls had no budget (round 13: 99 of 239). CPU 1.009, size 1.039 (both ties).
2. **Q2, LM under node-proximity linkage (the user's question): a small gain in both forms, below
   the band.**
   * With plain scaling, NP-LMR is 5 % lower than NP-NOLM (0.950, 75/21). This is the largest LM
     effect of the round. CPU 1.005, size 1.072 (ties).
   * With per-term least squares, TLSNP-LMR is 2.6 % lower than TLS-NP (0.974, 64/22). CPU 1.016,
     size 1.097 (ties; the size ratio is near the band).
   * Each LM arm also has the reserve, and the reserve adds demes (median 5 against 2). So each
     contrast measures LM plus the extra demes, not LM alone (see "Limits").
3. **Q3, the relative pole rule (NP-LMR-PSR): no gain.** A tie (1.000, 7/16, Holm p 0.056). It
   rejected 15 of 386 checked tunings (3.9 %; the round 13 rule: 19 of 171, 11 %). In the 12 runs
   with a rejection, PSR was lower in 3, higher in 7 and the same in 2. It did not catch the two
   runs of its arm with a test NMSE above 100 (Bos and WiW split 67; no rejection in either run).
4. **Q4, TLS-NP against BASE-TLS on new splits: within 10 % again.** 2.9 % higher NMSE (1.029,
   37/63), 18 % less CPU ("A better", 0.817), 8 % smaller (0.920, a tie). Round 13 gave 1.023, 0.843
   and 1.000.
5. **Q5, the linear baselines: BASE-TLS is 25 % lower than a plain linear model, but not on Dow.**
   BASE-TLS vs LIN-OLS 0.748 (86/14); TLS-NP vs LIN-RIDGE 0.746 (84/16); HC-LM vs LIN-OLS 0.919.
   **On Dow the linear models are better than every MOSES arm:** LIN-RIDGE 15.05 and LIN-OLS 15.72
   against BASE-TLS 17.16 (BASE-TLS is higher in 10 of 10 splits, ratio 1.114) and HC-LM 34.07.
   Ridge and OLS give the same result (median ratio 1.000).

* **Reference rule:** no arm with `--linear-scaling terms` is "A better" than BASE-TLS. The added
  descriptive contrast TLSNP-LMR vs BASE-TLS gives 0.996 (52/48, Holm p 0.64), so its reporting
  rule does not apply. **BASE-TLS stays the reference, and TLS-NP stays next in line.**
* **The gap to HC-LM:** 6 of 7 contrasts tie; NP-NOLM is "A worse" (1.146). TLS-LMR (0.908) and
  TLSNP-LMR (0.911) are the nearest to the band.
* **9 runs had a test NMSE above 100** (round 13: none). BASE-TLS had none. See "Runs with a test
  NMSE above 100".
* 1,000 of 1,000 runs ended ok. The round took 13.3 h (estimate 14–16 h).

## Run health

* **Steps:** S0 1.33 h, phase1bi 11.99 h. One orchestrator session, 2026-11-19 11:15Z to
  2026-11-20 00:34Z. W = 11; W = 7 for the Dow manifest. `bundle.err` is empty. Git HEAD
  `d6dadac`, clean.
* **S0:** reproducibility 4 of 4 matched. pytest: 10,203 passed, 1 failed
  (`test_toolchain_drift_is_clean_on_the_real_repo`), 13 skipped. The same test failed in rounds 6
  to 13 (the server's toolchain differs from the toolchain that the fixtures record). S0 is not a
  gate.
* **Runs:** 1,000 ok, 0 failed. Every MOSES run used exactly 20,000 evaluations. The new splits
  61–70 have no column-type mismatch (pre-registration, correction of RV16 finding 1), so the
  server run did not exercise the 12.1 fix.
* **Memory:** the highest peak RSS was 1,282 MB on the 5,120 MB manifest (25 %; BASE-TLS) and 1,983
  MB on Dow (24 % of 8,192 MB; HC-LM). Peak RSS per arm, other datasets / Dow: BASE-TLS 1,282 / 550,
  TLS-NP 866 / 563, HC-LM 635 / 1,983, TLS-LMR 1,272 / 586, TLSNP-LMR 854 / 594, NP-LMR 909 / 650,
  NP-NOLM 839 / 560, NP-LMR-PSR 909 / 654, LIN-OLS 31 / 25, LIN-RIDGE 29 / 23.
* **CPU:** 125.5 CPU hours (estimate 125–140; the pilot estimate 104): BASE-TLS 18.7, TLS-NP 14.1,
  HC-LM 23.0, TLS-LMR 18.8, TLSNP-LMR 14.5, NP-LMR 12.2, NP-NOLM 12.2, NP-LMR-PSR 12.1. The two
  linear arms used 3 CPU seconds together.

## The gap to HC-LM (Holm over 7)

| Contrast | Pairs | Holm p | Ratio | Verdict |
|---|---|---|---|---|
| BASE-TLS vs HC-LM | 60/40 | 0.0012 | 0.947 | tie |
| TLS-NP vs HC-LM | 63/37 | 0.0054 | 0.950 | tie |
| TLS-LMR vs HC-LM | 68/32 | 8.6e-6 | 0.908 | tie |
| TLSNP-LMR vs HC-LM | 63/37 | 1.9e-4 | 0.911 | tie |
| NP-LMR vs HC-LM | 28/72 | 2.4e-6 | 1.069 | tie |
| NP-NOLM vs HC-LM | 20/80 | 1.3e-8 | 1.146 | **A worse** |
| NP-LMR-PSR vs HC-LM | 26/74 | 5.0e-7 | 1.078 | tie |

## Program size and CPU (Holm over 10)

| Contrast | Size pairs | Size ratio | Size verdict | CPU pairs | CPU ratio | CPU verdict |
|---|---|---|---|---|---|---|
| TLS-LMR vs BASE-TLS | 28/54 | 1.039 | tie | 27/73 | 1.009 | tie |
| TLSNP-LMR vs TLS-NP | 15/66 | 1.097 | tie | 29/71 | 1.016 | tie |
| NP-LMR vs NP-NOLM | 29/61 | 1.072 | tie | 43/57 | 1.005 | tie |
| NP-LMR-PSR vs NP-LMR | 13/5 | 1.000 | tie | 80/20 | 0.996 | tie |
| TLS-NP vs BASE-TLS | 61/36 | 0.920 | tie | 89/11 | 0.817 | **A better** |

* In nodes only (not registered): TLS-LMR vs BASE-TLS 1.047, TLSNP-LMR vs BASE-TLS 1.057, TLS-NP vs
  BASE-TLS 0.930, NP-LMR vs BASE-TLS 0.795, HC-LM vs BASE-TLS 1.524.
* The LM arms cost about the same CPU as their partners (1.005–1.016). The LM share is about 1 %.

## Descriptive contrasts (their own Holm family of 14; no verdicts)

| Contrast | Pairs | Holm p | Ratio |
|---|---|---|---|
| BASE-TLS vs LIN-OLS | 86/14 | 5.0e-13 | 0.748 |
| BASE-TLS vs LIN-RIDGE | 86/14 | 1.1e-12 | 0.748 |
| TLS-NP vs LIN-OLS | 84/16 | 1.2e-11 | 0.754 |
| TLS-NP vs LIN-RIDGE | 84/16 | 2.5e-11 | 0.746 |
| HC-LM vs LIN-OLS | 73/27 | 7.5e-4 | 0.919 |
| HC-LM vs LIN-RIDGE | 72/28 | 7.5e-4 | 0.916 |
| NP-NOLM vs TLS-NP | 10/88 | 5.6e-13 | 1.323 |
| TLS-LMR vs TLSNP-LMR | 64/36 | 0.0030 | 0.970 |
| **TLSNP-LMR vs BASE-TLS** (amendment) | 52/48 | 0.64 | 0.996 |
| TLS-LMR vs BASE-TLS, clipped | 61/25 | 4.3e-5 | 0.983 |
| TLSNP-LMR vs TLS-NP, clipped | 63/23 | 4.7e-6 | 0.976 |
| NP-LMR vs NP-NOLM, clipped | 76/20 | 2.1e-8 | 0.943 |
| NP-LMR-PSR vs NP-LMR, clipped | 7/16 | 0.12 | 1.000 |
| TLS-NP vs BASE-TLS, clipped | 39/61 | 0.0074 | 1.022 |

* "Clipped" is the test NMSE with the predictions clipped to the training target range widened by
  10 % on each side. The clipped contrasts have the same direction as the plain ones.
* NP-NOLM vs TLS-NP: per-term least squares lowers the NMSE by 24 % under NP without LM (TLS-NP is
  1 / 1.323 = 0.756 of NP-NOLM).
* TLSNP-LMR vs BASE-TLS: the reporting rule of the amendment (Holm p < 0.025 and a ratio at least
  10 % below 1) does not apply.

## Trade-off table (pre-registered)

Pooled over the 100 pairs. Ratios are pair ratios. Demes, knobs and the LM share are medians over
the runs. "Knobs per deme" is the median of the per-run medians.

| Arm | NMSE / BASE-TLS | NMSE / HC-LM | Size / BASE-TLS | CPU / BASE-TLS | Demes | Knobs per deme | Total knobs | LM share |
|---|---|---|---|---|---|---|---|---|
| BASE-TLS | – | 0.947 | – | – | 2 | 74 | 160 | none (no LM) |
| TLS-NP | 1.029 | 0.950 | 0.920 | 0.817 | 2 | 61 | 140 | none (no LM) |
| HC-LM | 1.056 | – | 1.292 | 1.288 | 17 | not logged | not logged | 11.56 % |
| TLS-LMR | 0.980 | 0.908 | 1.039 | 1.009 | 5 | 195.5 | 796.5 | 1.03 % |
| TLSNP-LMR | 0.996 | 0.911 | 1.038 | 0.823 | 5 | 181 | 770.5 | 1.03 % |
| NP-LMR | 1.164 | 1.069 | 0.673 | 0.724 | 5 | 136.25 | 598.5 | 1.20 % |
| NP-NOLM | 1.304 | 1.146 | 0.616 | 0.715 | 2 | 56 | 118 | none (no LM) |
| NP-LMR-PSR | 1.180 | 1.078 | 0.632 | 0.724 | 5 | 133.5 | 588 | 1.23 % |
| LIN-OLS | 1.337 | 1.088 | 0.300 | ≈ 0 | – | – | – | – |
| LIN-RIDGE | 1.337 | 1.092 | 0.300 | ≈ 0 | – | – | – | – |

**Other medians (pooled over 100 runs):**

| Arm | Test NMSE | Train NMSE | Gap | Size | Nodes | CPU s |
|---|---|---|---|---|---|---|
| BASE-TLS | 19.04 | 15.04 | 1.26 | 70 | 60 | 424 |
| TLS-NP | 18.83 | 15.82 | 1.02 | 60 | 48 | 377 |
| HC-LM | 19.60 | 17.56 | 1.17 | 79.5 | 79.5 | 676 |
| TLS-LMR | 17.32 | 14.03 | 0.93 | 74 | 61 | 432 |
| TLSNP-LMR | 18.47 | 14.64 | 0.93 | 74.5 | 65 | 394 |
| NP-LMR | 26.55 | 21.09 | 0.86 | 41.5 | 41.5 | 318 |
| NP-NOLM | 29.88 | 23.33 | 0.56 | 40 | 40 | 320 |
| NP-LMR-PSR | 26.55 | 21.09 | 0.86 | 39.5 | 39.5 | 317 |
| LIN-OLS | 34.53 | 29.72 | 1.57 | 21 | 11.5 | 0.01 |
| LIN-RIDGE | 34.54 | 30.48 | 1.50 | 21 | 11.5 | 0.01 |

The pooled medians mix datasets with different NMSE scales. Use the pair ratios for comparisons.

## LM calls (summed over the 100 runs of each arm; from `summary.json`)

| Arm | Calls | No budget | No constants | Singular | Not improved | Improved | Guard checked | Guard rejected | Pole checked | Pole rejected |
|---|---|---|---|---|---|---|---|---|---|---|
| HC-LM | 1,815 | 85 | 0 | 43 | 0 | 1,687 | 1,672 | 446 (26.7 %) | – | – |
| TLS-LMR | 509 | 2 | 69 | 11 | 26 | 401 | 328 | 84 (25.6 %) | – | – |
| TLSNP-LMR | 485 | 1 | 62 | 7 | 16 | 399 | 316 | 93 (29.4 %) | – | – |
| NP-LMR | 493 | 0 | 0 | 8 | 21 | 464 | 388 | 76 (19.6 %) | – | – |
| NP-LMR-PSR | 489 | 0 | 0 | 6 | 24 | 459 | 371 | 72 (19.4 %) | 386 | 15 (3.9 %) |

* "No budget" is `const_opt.lm_budget` and "No constants" is `const_opt.lm_no_constants` (RV16
  finding 3: counted from `summary.json`, not from `runs.csv`).
* **The last LM call of most reserve runs is lost.** In 74 (TLS-LMR), 83 (TLSNP-LMR), 79 (NP-LMR)
  and 75 (NP-LMR-PSR) of the 100 runs, the run's budget ended during the last LM call of the run,
  before the guard could check it. That result is "improved" but not written back. It is always
  the last call of the run (per-call records in `events.jsonl`). HC-LM: 15 runs.
* Written back (per-call records): TLS-LMR 243, TLSNP-LMR 223, NP-LMR 309, NP-LMR-PSR 297, HC-LM
  1,226.
* **Dow guard rejections** (the pilot observation was 3 of 4): TLS-LMR 13 of 34, TLSNP-LMR 14 of 32,
  NP-LMR 12 of 34, NP-LMR-PSR 11 of 31, HC-LM 28 of 92. This is 30–44 %, above the arm shares of
  19–29 %, but not 75 %. The highest share is on WiR (TLS-LMR 15 of 20, TLSNP-LMR 15 of 21).

## Per dataset (exact values)

### Test NMSE (median)

| Dataset | BASE-TLS | TLS-NP | HC-LM | TLS-LMR | TLSNP-LMR | NP-LMR | NP-NOLM | NP-LMR-PSR | LIN-OLS | LIN-RIDGE |
|---|---|---|---|---|---|---|---|---|---|---|
| Air | 34.81 | 35.71 | 32.47 | 33.23 | 35.49 | 36.68 | 42.68 | 37.21 | 48.30 | 48.08 |
| Bos | 25.64 | 23.92 | 24.34 | 25.33 | 23.66 | 25.92 | 25.63 | 25.92 | 29.36 | 29.57 |
| Con | 24.19 | 24.95 | 18.83 | 24.08 | 24.04 | 28.35 | 33.95 | 28.27 | 39.44 | 39.48 |
| Dow | 17.16 | 17.06 | 34.07 | 16.25 | 16.81 | 40.84 | 42.74 | 40.84 | **15.72** | **15.05** |
| EnC | 5.20 | 7.04 | 10.80 | 4.58 | 6.35 | 10.79 | 10.89 | 10.79 | 11.28 | 11.30 |
| EnH | 2.08 | 3.77 | 7.12 | 1.90 | 2.31 | 7.42 | 7.81 | 7.42 | 8.47 | 8.48 |
| Tow | 8.77 | 11.66 | 15.26 | 8.77 | 10.12 | 20.12 | 21.83 | 20.58 | 11.67 | 11.76 |
| WiR | 62.74 | 62.31 | 64.97 | 62.50 | 63.62 | 64.95 | 65.18 | 64.64 | 64.76 | 64.48 |
| WiW | 70.82 | 71.08 | 74.02 | 70.55 | 70.82 | 72.90 | 72.90 | 73.34 | 72.88 | 72.97 |
| Yac | 1.06 | 0.92 | 0.76 | 0.75 | 0.72 | 0.86 | 1.20 | 0.86 | 36.60 | 36.31 |

### Program size (median; TLS arms with the term weights)

| Dataset | BASE-TLS | TLS-NP | HC-LM | TLS-LMR | TLSNP-LMR | NP-LMR | NP-NOLM | NP-LMR-PSR | LIN-OLS | LIN-RIDGE |
|---|---|---|---|---|---|---|---|---|---|---|
| Air | 78.5 | 77 | 158 | 79 | 72.5 | 80 | 50 | 80 | 12 | 12 |
| Bos | 56.5 | 54.5 | 126.5 | 73 | 74 | 45 | 51 | 45 | 28 | 28 |
| Con | 63 | 70 | 60.5 | 71.5 | 81.5 | 56.5 | 54 | 56.5 | 18 | 18 |
| Dow | 58.5 | 64 | 44 | 77.5 | 85 | 27 | 22.5 | 26.5 | 116 | 116 |
| EnC | 66.5 | 51 | 75 | 68 | 61 | 39 | 36 | 36 | 18 | 18 |
| EnH | 68 | 57 | 84 | 68 | 60 | 40 | 42 | 37.5 | 18 | 18 |
| Tow | 180.5 | 43 | 60 | 175.5 | 108.5 | 27.5 | 25 | 27.5 | 52 | 52 |
| WiR | 79 | 72 | 91.5 | 78.5 | 78.5 | 43.5 | 47 | 43.5 | 24 | 24 |
| WiW | 75 | 79.5 | 119 | 77 | 85.5 | 77.5 | 67.5 | 80 | 24 | 24 |
| Yac | 43 | 37.5 | 65 | 41 | 43 | 33.5 | 32 | 33 | 14 | 14 |

Median nodes:

| Dataset | Air | Bos | Con | Dow | EnC | EnH | Tow | WiR | WiW | Yac |
|---|---|---|---|---|---|---|---|---|---|---|
| BASE-TLS | 70 | 52 | 54.5 | 33.5 | 55.5 | 58.5 | 160.5 | 66.5 | 63.5 | 43 |
| TLS-NP | 69.5 | 52 | 61.5 | 36.5 | 42.5 | 48 | 24 | 63 | 68.5 | 37.5 |
| HC-LM | 158 | 126.5 | 60.5 | 44 | 75 | 84 | 60 | 91.5 | 119 | 65 |
| TLS-LMR | 71.5 | 67.5 | 62 | 51 | 59 | 58 | 153.5 | 65.5 | 65 | 37.5 |
| TLSNP-LMR | 66.5 | 67.5 | 72 | 58 | 51.5 | 51.5 | 90 | 68 | 75 | 41 |
| NP-LMR | 80 | 45 | 56.5 | 27 | 39 | 40 | 27.5 | 43.5 | 77.5 | 33.5 |
| NP-NOLM | 50 | 51 | 54 | 22.5 | 36 | 42 | 25 | 47 | 67.5 | 32 |
| NP-LMR-PSR | 80 | 45 | 56.5 | 26.5 | 36 | 37.5 | 27.5 | 43.5 | 80 | 33 |
| LIN (k inputs) | 7 (5) | 15 (13) | 10 (8) | 59 (57) | 10 (8) | 10 (8) | 27 (25) | 13 (11) | 13 (11) | 8 (6) |

### Train–test gap (median of test NMSE − training NMSE)

| Dataset | BASE-TLS | TLS-NP | HC-LM | TLS-LMR | TLSNP-LMR | NP-LMR | NP-NOLM | NP-LMR-PSR | LIN-OLS | LIN-RIDGE |
|---|---|---|---|---|---|---|---|---|---|---|
| Air | 2.36 | 2.12 | 2.94 | 1.59 | 2.36 | 1.62 | 1.10 | 1.93 | 1.79 | 1.65 |
| Bos | 7.44 | 3.40 | 8.54 | 8.61 | 4.78 | 7.19 | 4.34 | 7.19 | 3.03 | 2.60 |
| Con | 1.64 | 3.07 | 0.79 | 2.95 | 2.97 | 2.28 | 0.89 | 2.37 | 2.25 | 1.89 |
| Dow | 3.17 | 2.92 | 1.86 | 2.92 | 2.93 | 2.65 | 3.22 | 2.68 | 5.04 | 3.84 |
| EnC | 0.06 | 0.07 | 0.21 | 0.15 | 0.17 | −0.29 | −0.49 | −0.46 | 0.23 | 0.17 |
| EnH | 0.32 | 0.47 | 0.38 | 0.22 | 0.47 | 0.29 | −0.05 | 0.23 | 0.02 | 0.02 |
| Tow | 0.82 | 0.67 | 1.22 | 0.78 | 0.51 | 1.27 | 0.67 | 1.13 | 0.66 | 0.65 |
| WiR | 3.47 | 3.01 | 3.41 | 3.62 | 3.84 | 1.57 | 1.73 | 1.57 | 1.75 | 1.21 |
| WiW | 3.09 | 3.28 | 3.45 | 3.02 | 3.01 | 2.06 | 1.74 | 2.06 | 1.31 | 1.16 |
| Yac | 0.08 | −0.02 | 0.25 | 0.12 | 0.08 | 0.23 | −0.01 | 0.23 | 3.51 | 2.48 |

### CPU s per run (median)

| Dataset | BASE-TLS | TLS-NP | HC-LM | TLS-LMR | TLSNP-LMR | NP-LMR | NP-NOLM | NP-LMR-PSR |
|---|---|---|---|---|---|---|---|---|
| Air | 592 | 492 | 886 | 603 | 494 | 524 | 456 | 507 |
| Bos | 366 | 280 | 614 | 399 | 302 | 314 | 309 | 314 |
| Con | 418 | 371 | 343 | 419 | 379 | 275 | 293 | 274 |
| Dow | 370 | 342 | 831 | 425 | 402 | 310 | 296 | 308 |
| EnC | 330 | 266 | 402 | 338 | 271 | 237 | 231 | 237 |
| EnH | 341 | 267 | 476 | 338 | 271 | 235 | 230 | 232 |
| Tow | 1,475 | 885 | 1,396 | 1,493 | 930 | 821 | 802 | 821 |
| WiR | 630 | 440 | 751 | 639 | 473 | 356 | 392 | 354 |
| WiW | 1,608 | 1,379 | 2,055 | 1,596 | 1,369 | 1,056 | 1,070 | 1,052 |
| Yac | 282 | 246 | 283 | 295 | 219 | 202 | 269 | 200 |

### Demes per run (median)

| Dataset | BASE-TLS | TLS-NP | HC-LM | TLS-LMR | TLSNP-LMR | NP-LMR | NP-NOLM | NP-LMR-PSR |
|---|---|---|---|---|---|---|---|---|
| Air | 3 | 3 | 13.5 | 5.5 | 5 | 4.5 | 2 | 4.5 |
| Bos | 2 | 2 | 16.5 | 5 | 4 | 5 | 2 | 5 |
| Con | 2 | 2 | 25 | 4.5 | 4.5 | 5.5 | 2 | 5.5 |
| Dow | 1 | 1 | 10 | 4.5 | 4 | 4 | 1 | 4 |
| EnC | 2 | 2 | 19.5 | 5.5 | 5 | 5 | 2 | 5 |
| EnH | 2 | 2.5 | 15.5 | 5 | 5 | 4 | 2 | 4 |
| Tow | 2 | 1 | 15 | 4 | 3 | 5 | 1 | 5 |
| WiR | 2 | 2 | 22 | 4 | 4 | 5 | 2 | 5 |
| WiW | 2 | 2 | 15.5 | 5 | 4 | 4 | 2 | 4 |
| Yac | 6 | 5.5 | 21.5 | 8 | 8.5 | 8 | 5.5 | 8 |

Knobs per deme on Dow (median of the per-run medians): BASE-TLS 118, TLS-LMR 1,742, TLSNP-LMR
2,002, NP-LMR 538, NP-NOLM 118.

### Per-dataset contrasts on test NMSE (descriptive, Holm over 50, no verdicts)

Pairs A lower / B lower. Holm p is 1 unless given.

| Contrast | Air | Bos | Con | Dow | EnC | EnH | Tow | WiR | WiW | Yac |
|---|---|---|---|---|---|---|---|---|---|---|
| TLS-LMR vs BASE-TLS | 9/1 (0.18) | 5/5 | 6/1 | 6/3 | 8/2 (0.78) | 8/2 | 5/3 | 3/3 | 4/2 | 9/1 (0.41) |
| TLSNP-LMR vs TLS-NP | 5/3 | 7/1 | 7/2 | 6/4 | 7/0 (0.64) | 8/2 (0.78) | 10/0 (0.098) | 2/4 | 5/3 | 7/3 |
| NP-LMR vs NP-NOLM | 10/0 (0.098) | 5/5 | 9/1 (0.18) | 6/2 | 8/1 (0.34) | 9/1 (0.18) | 9/1 (0.26) | 6/3 | 6/4 | 7/3 |
| NP-LMR-PSR vs NP-LMR | 2/1 | 0/0 | 1/2 | 1/2 | 0/3 | 0/3 | 1/2 | 1/0 | 1/1 | 0/2 |
| TLS-NP vs BASE-TLS | 5/5 | 7/3 | 3/7 | 4/6 | 2/8 (0.78) | 1/9 (0.78) | 0/10 (0.098) | 4/6 | 4/6 | 7/3 |

### The linear baselines per dataset (not registered; pairs and median ratio)

| Contrast | Air | Bos | Con | Dow | EnC | EnH | Tow | WiR | WiW | Yac |
|---|---|---|---|---|---|---|---|---|---|---|
| BASE-TLS vs LIN-RIDGE | 10/0 0.725 | 9/1 0.831 | 10/0 0.605 | **0/10 1.114** | 10/0 0.481 | 10/0 0.235 | 10/0 0.765 | 9/1 0.971 | 8/2 0.977 | 10/0 0.028 |
| TLS-NP vs LIN-RIDGE | 10/0 0.723 | 10/0 0.775 | 10/0 0.617 | **0/10 1.177** | 10/0 0.661 | 10/0 0.453 | 8/2 0.993 | 8/2 0.980 | 8/2 0.974 | 10/0 0.026 |
| HC-LM vs LIN-RIDGE | 10/0 0.668 | 8/2 0.716 | 10/0 0.477 | **0/10 2.475** | 8/2 0.955 | 10/0 0.836 | 4/6 1.169 | 7/3 0.979 | 5/5 1.003 | 10/0 0.020 |

## Runs with a test NMSE above 100

| Dataset | Arm | Split | Test NMSE | Clipped | Train NMSE | Size |
|---|---|---|---|---|---|---|
| WiR | HC-LM | 63 | 1,744.5 | 69.5 | 61.0 | 61 |
| Bos | NP-LMR | 67 | 138.7 | 46.1 | 16.6 | 53 |
| Bos | NP-LMR-PSR | 67 | 138.7 | 46.1 | 16.6 | 53 |
| WiW | NP-LMR | 67 | 104.8 | 77.0 | 69.4 | 101 |
| WiW | NP-LMR-PSR | 67 | 104.8 | 77.0 | 69.4 | 101 |
| WiW | TLS-NP | 63 | 104.2 | 70.5 | 65.5 | 82 |
| WiW | TLSNP-LMR | 61 | 102.1 | 71.0 | 63.6 | 90 |
| WiW | TLS-NP | 61 | 101.5 | 71.0 | 63.8 | 87 |
| WiW | TLSNP-LMR | 63 | 101.4 | 70.8 | 65.0 | 133 |

* NP-LMR and NP-LMR-PSR printed the same final program on Bos split 67 and on WiW split 67. On Bos
  split 67, NP-NOLM gave 25.95.
* Clipping brings each of these runs back to the range of the other runs. The WiW runs are only
  slightly above 100 (the WiW medians are 70–74).

## Arm details

### TLS-LMR (Q1)

* 63/23 against BASE-TLS (14 identical pairs). The 86 pairs that differ have a median ratio of
  0.959 (range 0.308–1.609). Gains on Air (9/1), Yac (9/1), EnC and EnH (8/2 each).
* It has the lowest pooled median test NMSE of all arms (17.32) and is the nearest to the band
  against HC-LM (0.908).
* 69 of 509 LM calls had no constant inside the terms.

### TLSNP-LMR (Q2, per-term)

* 64/22 against TLS-NP. Gains on Tow (10/0; 10.12 against 11.66) and on EnC (7/0).
* Against BASE-TLS (descriptive): 0.996 (52/48). Not registered: CPU 0.823 (85/15), size 1.038.
  So TLSNP-LMR has the accuracy of BASE-TLS at 18 % less CPU, with slightly larger programs.

### NP-LMR and NP-NOLM (Q2, plain scaling; the user's question)

* NP-LMR is lower than NP-NOLM in 75 of 96 pairs that differ (0.950). Gains on Air (10/0), Con
  (9/1), EnH (9/1) and Tow (9/1).
* Both are far behind the TLS arms (NP-LMR 1.164 against BASE-TLS; NP-NOLM 1.304) but make the
  smallest MOSES programs (size 0.673 and 0.616 against BASE-TLS) at the lowest CPU (0.72).
* Round 12 (BASE-NP with LM-VAL and no reserve) gave LM share 0.16 %; with the reserve it is 1.20 %.

### NP-LMR-PSR (Q3, the relative pole rule)

* 386 pole checks, 15 rejections in 12 runs. In each rejection the tuned spread was 12 to 340
  times the old spread (S(old) 26–2,675; S(new) 1,398–96,692).
* 23 pairs differ from NP-LMR (7 lower, 16 higher):
  * 12 runs with a rejection: 3 lower, 7 higher, 2 identical.
  * 13 runs without a rejection: 4 lower, 9 higher; 75 runs without a rejection are identical.
    Each check charges 2 evaluations, and the reserve sets the stop point of each deme from the
    budget that is left; the agent reading is that these charges move the stop points and so
    change the search path.
* It did not prevent the two runs of its arm with a test NMSE above 100 (Bos and WiW split 67;
  0 rejections in either run).
* Status: an agent idea, designed after the round 13 data (post hoc). Not approved as a setting
  (see "User decisions").

### TLS-NP (Q4)

* 37/63 against BASE-TLS. It loses on Tow (0/10; 11.66 against 8.77), EnH (1/9) and EnC (2/8). It
  wins on Bos and Yac (7/3 each).
* On Tow its programs are much smaller (43 against 180.5; 24 against 160.5 nodes), as in round 13.

### LIN-OLS and LIN-RIDGE (Q5)

* OLS dropped `x4` on EnC and EnH (aliased with the other inputs; the Gram–Schmidt fallback) in all
  10 splits.
* The ridge λ that the validation rows chose varies: 10 or 100 on 7 or 8 of the 10 splits of WiR,
  Yac, Air and Bos; 10^-6 on 6 of 10 WiW splits and 5 of 10 EnH splits. The test NMSE is the same
  as OLS (median ratio 1.000).
* On Yac (6 inputs) the linear models are very poor (36.6 against 1.06): the target is strongly
  non-linear in one input. On WiR and WiW they are within 2–3 % of BASE-TLS.
* On Dow (57 inputs) they are the best models of the round, with programs of 57 inputs (size 116).
  Their train–test gap on Dow (5.04 OLS, 3.84 ridge) is larger than that of BASE-TLS (3.17).

## Observations (agent reading)

1. **LM with a budget helps a little, every time.** All three LM-with-reserve contrasts favour LM
   (2–5 %, 63–75 of the pairs that differ), at no CPU cost and with 4–10 % larger programs. None
   reaches the 10 % band.
2. **The reserve changes more than the LM budget.** The LM arms have 5 demes against 2, and
   2.4–2.6 times more knobs per deme (Dow: 1,742 against 118). Round 11 saw the same for BASE-R
   (`SERVER-ROUND11-FINDINGS.md`). The likely cause: LM uses about 50 of the 500 reserved
   evaluations; MOSES gives the rest to a new deme, which again stops early; the chain ends when
   the budget is gone, so the last LM call has no budget for the guard. The round cannot separate
   the LM effect from the effect of the extra small demes.
3. **The last LM tuning of a run is usually lost.** In 74–83 % of the reserve runs, the last LM
   call ran out of the run's budget before the guard check, so the result was not written back.
   This call tunes the best program of the last deme. The agent did not check how often that
   program becomes the final program. An agent idea: end the deme chain when the budget that is
   left is too small for one LM call and the guard.
4. **On Dow, the search does not reach a plain linear model.** BASE-TLS is 11 % higher than ridge
   on 10 of 10 Dow splits, after 20,000 evaluations. The BASE-TLS programs on Dow are in effect
   linear regressions on selected inputs (round 13), but with fewer inputs than ridge uses (size
   58.5 against 116). On the other 9 datasets, the search gives a large gain over the linear model
   (Yac 0.028, EnH 0.235, EnC 0.481).
5. **The relative pole rule does not earn its place.** It rejects fewer tunings than the round 13
   rule, but where it rejects, the result is worse in 7 of 10 runs that differ, and it missed the
   poles of its own arm. The clipped NMSE handled those runs.
6. **TLS-NP repeats its round 13 profile.** About 3 % higher NMSE, 18 % less CPU, slightly smaller
   programs. TLSNP-LMR gives the accuracy of BASE-TLS at the CPU of TLS-NP, but with larger programs
   (not registered).
7. **No arm passes HC-LM by 10 %.** TLS-LMR (0.908) and TLSNP-LMR (0.911) are the nearest, with
   smaller programs (0.852 and 0.780 of HC-LM, not registered) and less CPU (0.770 and 0.638).

## Faults and deviations

1. **S0 pytest failure:** the known toolchain test (see "Run health"). Not a code fault.
2. **No run failed.** The splits 61–70 do not exercise the 12.1 fix (see "Run health").
3. **Pilot estimate:** the pilot gave 104 CPU hours and 10–12 h. The round used 125.5 CPU hours and
   13.3 h, inside the pre-registered estimate (125–140 CPU hours, 14–16 h).
4. **The pilot observation on Dow** (the guard rejected 3 of 4 checked LM results) did not repeat
   at that rate: 30–44 % on the full round.

## Limits

* One budget (20,000 evaluations) and one reserve (500).
* Each LM arm differs from its partner by LM **and** by the reserve, and the reserve adds demes. No
  arm had the reserve without LM (the option needs `--const-opt lm`).
* NP-LMR-PSR tests the pole rule only under NP with plain scaling, and only one factor (10).
* The linear baselines use all inputs; no feature selection apart from ridge shrinkage. No
  interaction terms.
* 9 runs had a test NMSE above 100; the plain test NMSE counts them as they are.

## User decisions

* **Reference (rule result, 2026-11-20):** no arm with `--linear-scaling terms` is "A better" than
  BASE-TLS, and the amendment contrast TLSNP-LMR vs BASE-TLS does not trigger a report. BASE-TLS
  stays the reference and the default; TLS-NP stays next in line.
* **The relative pole rule (PSR, `box-rel`): rejected (2026-11-20).** An agent idea, designed post
  hoc on the round 13 data (origin: `BACKLOG.md` 12.3). The agent recommended to reject it as a
  setting (observation 5). The user answered: "Yes let's reject it".
* **Pending: the other agent ideas of this round** (none is a setting until the user decides):
  * the reserve of 500 for the LM arms (an agent choice);
  * the linear baselines as report baselines (`BACKLOG.md` 12.6);
  * the new ideas of `BACKLOG.md` section 13. The user (2026-11-20): "Don't start any of these";
    the next ideas are to be discussed first.
