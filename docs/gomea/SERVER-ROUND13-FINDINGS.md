# Server round 13: findings (2026-11-19)

**Pre-registration:** `SERVER-ROUND13-PREREGISTRATION.md` (with its section "Rules that the approved
text did not cover"). **Results:** bundle `bundle-final-20261119T042106Z.tar.gz` (SHA-256
`70c3af6e…5b489`; local copy in the MOSES folder, next to the repo). **Code:** `593b600`.
**Decisions:** `APPROVALS.md`. The confirmation rule gives a proposal to the user (see "User
decisions"); no decision yet.

Test NMSE in the tables is × 100. "Pairs" is the number of splits where A is lower / where B is
lower; the remaining pairs are ties. "Ratio" is exp(median of log(A / B)) over the pairs: above 1
means that A has the higher value (higher NMSE, larger program or more CPU). A verdict of "better"
or "worse" needs Holm p < 0.025 and a ratio at least 10 % from 1. One run failed (Dow HC-LM split
56, see "Faults"); it counts as infinitely bad against HC-LM.

**Size** (pre-registered): the node count, plus the number of terms when the final model uses the
per-term fit (2 or more terms, no fallback; for TLS-G only when the guard kept `terms`). The node
count alone is also given.

**Datasets:** as in round 12 (`SERVER-ROUND12-FINDINGS.md`): Air, Bos, Con, Dow, EnC, EnH, Tow,
WiR, WiW, Yac. Fresh splits 51–60.

**Arms** (all at 20,000 evaluations):

| Arm | Task | Change |
|---|---|---|
| BASE | reference | the round 12 BASE (dynamic population, LT-B, linear scaling, LM-VAL, PEN125) |
| HC-LM | accuracy baseline | hill climbing with LM-VAL |
| BASE-TLS | T2.2, confirmation | `--linear-scaling terms --const-opt none` (no LM, no guard) |
| TLS-NP | T2.1 + T2.2 | BASE-TLS with `--gomea-fos np` |
| TLS-G | TLS guard | BASE-TLS; the runner keeps per-term or plain scaling by validation NMSE |
| TLS-LM | T3.1 | BASE-TLS with `--const-opt lm --const-opt-guard val` |
| BASE-LEX | A4, ε-lexicase | BASE + `--exemplar-selection lexicase` |
| BASE-PS | pole safeguard | BASE + `--const-opt-pole-check box` |

## Summary

| Contrast | Family | Pairs | Ties | Holm p | Ratio | Verdict |
|---|---|---|---|---|---|---|
| **BASE-TLS vs BASE** | primary | **80/20** | 0 | **9.7e-10** | **0.784** | **A better** |
| TLS-NP vs BASE-TLS | primary | 38/62 | 0 | 0.0037 | 1.023 | tie |
| TLS-G vs BASE-TLS | primary | 0/0 | 100 | 1 | 1.000 | tie |
| TLS-LM vs BASE-TLS | primary | 10/10 | 80 | 1 | 1.000 | tie |
| BASE-LEX vs BASE | primary | 26/55 | 19 | 0.0062 | 1.012 | tie |
| BASE-PS vs BASE | primary | 1/12 | 87 | 0.0037 | 1.000 | tie |

**Answers to the questions of the round:**

1. **Confirmation: BASE-TLS is "A better" than BASE again, by more than in round 12.** The test
   NMSE is 22 % lower at the median pair (round 12: 12 %). It is lower in 80 of 100 pairs. The
   programs are 26 % larger with the term weights (8 % in nodes). The CPU is the same (1.082, a
   tie). The confirmation rule applies (see "User decisions").
2. **TLS-NP (node-proximity linkage on top of BASE-TLS): a tie, 2.3 % higher NMSE, 16 % less CPU**
   ("A better" on CPU, 0.843). The size is the same (1.000). The Holm p is below 0.025, but the
   ratio is inside the 10 % band.
3. **TLS-G (the validation guard at the final choice): no effect.** The test NMSE is the same in
   all 100 pairs. The guard kept the per-term model in 82 runs. It kept plain scaling in 18 runs,
   all with a one-term program, where the two models are the same.
4. **TLS-LM (T3.1, LM on the inner constants): no effect at this budget.** 80 pairs are identical,
   10 lower and 10 higher. LM wrote back a tuning in 44 of 239 calls.
5. **BASE-LEX (ε-lexicase exemplar selection): a tie with a small loss.** It is higher in 55 of the
   81 pairs that differ (ratio 1.012). The Holm p is below 0.025, but the ratio is inside the band.
   The median run made one lexicase selection.
6. **BASE-PS (the pole safeguard): a tie, but harmful where it acts.** The 13 pairs that differ all
   had a pole rejection, and BASE-PS is worse in 12 of them. The Holm p is below 0.025; the ratio is
   1.000 because 87 pairs are identical.

* **No run had a test NMSE above 100**, in any arm. The clipped test NMSE gives the same verdicts as
  the test NMSE. So the round had no pole problem for the safeguard to fix.
* **The gap to HC-LM:** all 7 contrasts tie. The TLS arms are 4–6 % lower than HC-LM (BASE-TLS
  0.949, 64/36); BASE is 6 % higher (1.060, 32/68).
* 799 of 800 runs ended ok. The round took 14.5 h (estimate 15–18 h).

## Run health

* **Steps:** S0 1.33 h, phase1bh 13.12 h. One orchestrator session, 2026-11-18 13:53Z to
  2026-11-19 04:21Z. W = 11; W = 7 for the Dow manifest. `bundle.err` is empty. Git HEAD `593b600`,
  clean.
* **S0:** reproducibility 4 of 4 matched. pytest: 9,985 passed, 1 failed
  (`test_toolchain_drift_is_clean_on_the_real_repo`), 13 skipped. The same test failed in rounds 6
  to 12 (the server's toolchain differs from the toolchain that the fixtures record). The 48 runner
  and report identity tests of round 13 ran; none was skipped (RV16 nit 2 did not occur).
* **Runs:** 799 ok, 1 failed (Dow HC-LM split 56, 3.8 s, see "Faults"). Every ok run used exactly
  20,000 evaluations.
* **Memory:** the highest peak RSS was 1,188 MB on the 5,120 MB manifest (23 %; BASE-LEX) and 2,838
  MB on Dow (35 % of 8,192 MB; HC-LM). Peak RSS per arm, other datasets / Dow: BASE 886 / 560,
  HC-LM 599 / 2,838, BASE-TLS 1,038 / 552, TLS-NP 842 / 571, TLS-G 1,038 / 552, TLS-LM 1,054 / 552,
  BASE-LEX 1,188 / 562, BASE-PS 886 / 560.
* **CPU:** 138.0 CPU hours (estimate 140–160): BASE 15.8, HC-LM 22.0, BASE-TLS 18.2, TLS-NP 13.9,
  TLS-G 18.0, TLS-LM 18.1, BASE-LEX 16.4, BASE-PS 15.6.

## The gap to HC-LM (Holm over 7)

| Contrast | Pairs | Holm p | Ratio | Verdict |
|---|---|---|---|---|
| BASE vs HC-LM | 32/68 | 8.7e-6 | 1.060 | tie |
| BASE-TLS vs HC-LM | 64/36 | 3.7e-4 | 0.949 | tie |
| TLS-NP vs HC-LM | 65/35 | 0.0036 | 0.958 | tie |
| TLS-G vs HC-LM | 64/36 | 3.7e-4 | 0.949 | tie |
| TLS-LM vs HC-LM | 64/36 | 1.1e-4 | 0.943 | tie |
| BASE-LEX vs HC-LM | 25/75 | 6.7e-7 | 1.080 | tie |
| BASE-PS vs HC-LM | 29/71 | 2.1e-6 | 1.062 | tie |

The failed HC-LM run (Dow split 56) counts against HC-LM in each of these 7 contrasts (1 pair of
100).

## Program size and CPU (Holm over 12)

| Contrast | Size pairs | Size ratio | Size verdict | CPU pairs | CPU ratio | CPU verdict |
|---|---|---|---|---|---|---|
| BASE-TLS vs BASE | 23/75 | 1.261 | **A worse** | 35/65 | 1.082 | tie |
| TLS-NP vs BASE-TLS | 49/49 | 1.000 | tie | 82/18 | 0.843 | **A better** |
| TLS-G vs BASE-TLS | 0/0 | 1.000 | tie | 86/14 | 0.995 | tie |
| TLS-LM vs BASE-TLS | 8/12 | 1.000 | tie | 70/30 | 0.996 | tie |
| BASE-LEX vs BASE | 48/30 | 1.000 | tie | 47/53 | 1.010 | tie |
| BASE-PS vs BASE | 9/3 | 1.000 | tie | 64/36 | 0.998 | tie |

* In nodes only (not registered): BASE-TLS vs BASE 1.082 (38/58), TLS-NP vs BASE 1.030, TLS-LM vs
  BASE 1.089, HC-LM vs BASE 1.568.
* TLS-G and BASE-TLS ran the same search (`same_search.json`: 100 of 100 pairs printed the same
  candidates). Their CPU difference (0.995, 86/14) comes from the server load, not from the arm.

## Descriptive contrasts (their own Holm family of 9; the verdicts are descriptive only)

| Contrast | Pairs | Holm p | Ratio | Report verdict |
|---|---|---|---|---|
| TLS-NP vs BASE | 76/24 | 9.4e-8 | 0.839 | A better |
| TLS-G vs BASE | 80/20 | 1.1e-9 | 0.784 | A better |
| TLS-LM vs BASE | 82/16 | 6.3e-11 | 0.790 | A better |
| BASE-TLS vs BASE, clipped | 84/16 | 1.1e-10 | 0.786 | A better |
| TLS-NP vs BASE-TLS, clipped | 35/65 | 0.0023 | 1.019 | tie |
| TLS-G vs BASE-TLS, clipped | 0/0 | 1 | 1.000 | tie |
| TLS-LM vs BASE-TLS, clipped | 10/10 | 1 | 1.000 | tie |
| BASE-LEX vs BASE, clipped | 27/54 | 0.0095 | 1.012 | tie |
| BASE-PS vs BASE, clipped | 1/12 | 0.0049 | 1.000 | tie |

"Clipped" is the test NMSE with the predictions clipped to the training target range widened by
10 % on each side.

## Trade-off table (pre-registered)

Pooled over the 100 pairs. Ratios are pair ratios against BASE (or HC-LM). Demes, knobs and the LM
share are medians over the runs. "Knobs per deme" is the median of the per-run medians.

| Arm | NMSE / BASE | NMSE / HC-LM | Size / BASE | CPU / BASE | Demes | Knobs per deme | Total knobs | LM share |
|---|---|---|---|---|---|---|---|---|
| BASE | – | 1.060 | – | – | 2.5 | 65.0 | 190 | 0.26 % |
| HC-LM | 0.944 | – | 1.572 | 1.344 | 18 | not logged | not logged | 11.58 % |
| BASE-TLS | 0.784 | 0.949 | 1.261 (nodes 1.082) | 1.082 | 2 | 65.0 | 148 | none (no LM) |
| TLS-NP | 0.839 | 0.958 | 1.191 (nodes 1.030) | 0.908 | 2 | 57.0 | 124 | none (no LM) |
| TLS-G | 0.784 | 0.949 | 1.261 | 1.078 | 2 | 65.0 | 148 | none (no LM) |
| TLS-LM | 0.790 | 0.943 | 1.270 (nodes 1.089) | 1.070 | 2 | 65.0 | 146 | 0.005 % |
| BASE-LEX | 1.012 | 1.080 | 1.000 | 1.010 | 2 | 63.0 | 150 | 0.20 % |
| BASE-PS | 1.000 | 1.062 | 1.000 | 0.998 | 2.5 | 63.5 | 182.5 | 0.28 % |

**Other medians (pooled over 100 runs):**

| Arm | Test NMSE | Train NMSE | Gap | Size | Nodes | CPU s |
|---|---|---|---|---|---|---|
| BASE | 27.65 | 24.14 | 1.03 | 53.0 | 53.0 | 443 |
| HC-LM | 20.70 | 16.27 | 1.15 | 77.0 | 77.0 | 697 |
| BASE-TLS | 17.89 | 14.51 | 0.96 | 67.5 | 59.0 | 404 |
| TLS-NP | 18.81 | 15.80 | 0.90 | 64.0 | 50.5 | 372 |
| TLS-G | 17.89 | 14.51 | 0.96 | 67.5 | 59.0 | 401 |
| TLS-LM | 18.35 | 15.09 | 0.96 | 70.0 | 59.5 | 405 |
| BASE-LEX | 29.68 | 25.86 | 0.72 | 44.5 | 44.5 | 400 |
| BASE-PS | 27.65 | 24.14 | 1.00 | 52.0 | 52.0 | 443 |

The pooled medians mix datasets with different NMSE scales. Use the pair ratios for comparisons.
HC-LM has 99 ok runs.

## Per dataset (exact values)

### Test NMSE (median)

| Dataset | BASE | HC-LM | BASE-TLS | TLS-NP | TLS-G | TLS-LM | BASE-LEX | BASE-PS |
|---|---|---|---|---|---|---|---|---|
| Air | 41.51 | 32.94 | 35.69 | 36.67 | 35.69 | 35.69 | 40.13 | 42.53 |
| Bos | 24.97 | 22.91 | 23.01 | 23.64 | 23.01 | 25.54 | 25.68 | 24.97 |
| Con | 29.81 | 18.17 | 25.47 | 25.57 | 25.47 | 25.47 | 32.76 | 29.81 |
| Dow | 41.26 | 34.83 | **14.88** | **14.70** | **14.88** | **14.88** | 41.26 | 41.26 |
| EnC | 10.25 | 10.23 | **6.01** | 6.80 | **6.01** | **5.85** | 10.66 | 10.25 |
| EnH | 7.70 | 7.12 | **2.48** | 3.06 | **2.48** | **2.44** | 7.91 | 7.70 |
| Tow | 17.41 | 15.93 | **8.64** | 11.70 | **8.64** | **8.64** | 17.41 | 17.43 |
| WiR | 67.69 | 66.28 | 65.25 | 65.43 | 65.25 | 65.25 | 66.28 | 67.69 |
| WiW | 73.12 | 73.70 | 73.05 | 72.75 | 73.05 | 73.05 | 74.00 | 73.12 |
| Yac | 0.88 | 0.70 | 1.22 | 1.16 | 1.22 | 0.81 | 1.01 | 0.97 |

### Program size (median; TLS arms with the term weights)

| Dataset | BASE | HC-LM | BASE-TLS | TLS-NP | TLS-G | TLS-LM | BASE-LEX | BASE-PS |
|---|---|---|---|---|---|---|---|---|
| Air | 73 | 147.5 | 81 | 83 | 81 | 81 | 79 | 71 |
| Bos | 51 | 113 | 54 | 45.5 | 54 | 67 | 46.5 | 51 |
| Con | 72 | 67 | 73 | 77 | 73 | 73 | 68 | 72 |
| Dow | 26 | 55 | 57.5 | 65 | 57.5 | 57.5 | 26 | 26 |
| EnC | 57 | 68.5 | 64.5 | 57 | 64.5 | 62 | 43.5 | 50 |
| EnH | 48 | 80 | 60.5 | 56.5 | 60.5 | 60.5 | 39 | 48 |
| Tow | 41.5 | 50 | 175 | 46.5 | 175 | 175 | 41.5 | 41.5 |
| WiR | 51.5 | 90 | 73 | 79.5 | 73 | 73 | 47.5 | 51.5 |
| WiW | 65 | 103 | 78 | 85 | 78 | 78 | 77 | 65 |
| Yac | 42.5 | 69 | 52 | 44 | 52 | 37.5 | 34 | 42.5 |

Median nodes of the TLS arms (TLS-G as BASE-TLS):

| Dataset | Air | Bos | Con | Dow | EnC | EnH | Tow | WiR | WiW | Yac |
|---|---|---|---|---|---|---|---|---|---|---|
| BASE-TLS | 72 | 53 | 62.5 | 32.5 | 54.5 | 51 | 155 | 63.5 | 66.5 | 52 |
| TLS-NP | 75 | 45.5 | 68 | 38 | 47 | 48 | 26 | 69.5 | 73 | 44 |
| TLS-LM | 72 | 67 | 62.5 | 32.5 | 51 | 51 | 155 | 63.5 | 66.5 | 37.5 |

### Train–test gap (median of test NMSE − training NMSE)

| Dataset | BASE | HC-LM | BASE-TLS | TLS-NP | TLS-G | TLS-LM | BASE-LEX | BASE-PS |
|---|---|---|---|---|---|---|---|---|
| Air | 1.50 | 3.01 | 3.17 | 2.49 | 3.17 | 3.17 | 2.31 | 1.84 |
| Bos | 8.60 | 10.07 | 6.46 | 6.92 | 6.46 | 10.11 | 7.04 | 8.60 |
| Con | 1.62 | 1.14 | 2.77 | 2.75 | 2.77 | 2.77 | 2.50 | 1.62 |
| Dow | 2.66 | 3.06 | 0.96 | 0.82 | 0.96 | 0.96 | 2.66 | 2.66 |
| EnC | 0.69 | 0.46 | 0.53 | 0.57 | 0.53 | 0.56 | 0.13 | 0.61 |
| EnH | 0.07 | 0.25 | 0.24 | 0.16 | 0.24 | 0.24 | 0.10 | 0.07 |
| Tow | −0.62 | −0.44 | 1.04 | 0.92 | 1.04 | 1.04 | −0.62 | −0.62 |
| WiR | 4.32 | 6.40 | 6.08 | 4.93 | 6.08 | 6.08 | 3.28 | 4.32 |
| WiW | 1.17 | 1.61 | 5.90 | 5.19 | 5.90 | 5.90 | 1.98 | 1.17 |
| Yac | 0.14 | 0.22 | 0.10 | −0.02 | 0.10 | 0.11 | 0.01 | 0.16 |

### CPU s per run (median)

| Dataset | BASE | HC-LM | BASE-TLS | TLS-NP | TLS-G | TLS-LM | BASE-LEX | BASE-PS |
|---|---|---|---|---|---|---|---|---|
| Air | 638 | 921 | 601 | 529 | 600 | 599 | 609 | 593 |
| Bos | 370 | 591 | 357 | 256 | 355 | 337 | 319 | 370 |
| Con | 462 | 330 | 410 | 402 | 408 | 419 | 399 | 459 |
| Dow | 307 | 914 | 364 | 340 | 363 | 359 | 311 | 303 |
| EnC | 338 | 380 | 317 | 277 | 314 | 317 | 368 | 349 |
| EnH | 337 | 435 | 316 | 279 | 312 | 311 | 317 | 342 |
| Tow | 971 | 1,320 | 1,698 | 941 | 1,700 | 1,712 | 999 | 948 |
| WiR | 531 | 804 | 614 | 431 | 612 | 613 | 448 | 522 |
| WiW | 1,363 | 1,991 | 1,447 | 1,268 | 1,448 | 1,468 | 1,614 | 1,363 |
| Yac | 274 | 281 | 288 | 263 | 285 | 299 | 278 | 273 |

### Per-dataset contrasts on test NMSE (descriptive, Holm over 60, no verdicts)

Pairs A lower / B lower. Holm p is 1 unless given.

| Contrast | Air | Bos | Con | Dow | EnC | EnH | Tow | WiR | WiW | Yac |
|---|---|---|---|---|---|---|---|---|---|---|
| BASE-TLS vs BASE | 8/2 | 6/4 | 9/1 (0.22) | 10/0 (0.12) | 10/0 (0.12) | 10/0 (0.12) | 10/0 (0.12) | 8/2 (0.52) | 6/4 | 3/7 |
| TLS-NP vs BASE-TLS | 8/2 | 3/7 | 4/6 | 3/7 | 2/8 | 3/7 | 1/9 (0.22) | 4/6 | 5/5 | 5/5 |
| TLS-G vs BASE-TLS | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| TLS-LM vs BASE-TLS | 1/0 | 1/5 | 0/0 | 0/0 | 2/0 | 1/0 | 0/0 | 0/0 | 0/0 | 5/5 |
| BASE-LEX vs BASE | 2/7 | 3/7 | 5/5 | 0/0 | 1/9 (0.22) | 2/8 (0.52) | 1/1 | 3/7 | 5/5 | 4/6 |
| BASE-PS vs BASE | 1/5 | 0/1 | 0/0 | 0/0 | 0/3 | 0/1 | 0/1 | 0/0 | 0/0 | 0/1 |

## Runs with a test NMSE above 100

None, in any arm (`report-stage1/nmse_above_100.csv` has no data row).

## Arm details

### BASE-TLS (T2.2, the confirmation)

* Gains on Dow (14.88 against 41.26), EnH (2.48 against 7.70), Tow (8.64 against 17.41) and EnC
  (6.01 against 10.25), each lower in 10 of 10 splits. It loses on Yac (3/7; 1.22 against 0.88).
* **The structure of the programs** (agent measurement on the final programs): on Dow the median
  program has 26 top-level terms, and 96 % of them are single inputs (10 of 10 runs have at least
  80 % single-input terms). With one fitted weight per term, such a program is a linear
  regression on selected inputs. On Bos and Yac the median program has one term, so the per-term
  fit cannot act. On the other datasets the programs have 8.5–21.5 terms, 35–45 % of them single
  inputs.
* Size with the weights 1.261 ("A worse"); in nodes 1.082. Tow programs are large: 175 (155 nodes)
  against 41.5 for BASE.
* The pooled gap is the same as for BASE (0.96 against 1.03). Per dataset it is larger on WiW
  (5.90 against 1.17), Air, Con, Tow and WiR, and smaller on Dow and Bos.

### TLS-NP (T2.1 + T2.2)

* 2.3 % higher NMSE than BASE-TLS (38/62, a tie), 16 % less CPU ("A better"), the same size.
* On Tow it is worse (1/9; 11.70 against 8.64), but its programs are much smaller: 26 nodes
  against 155. On Air it is better (8/2).
* Against BASE: 0.839 (76/24, descriptive "A better"), size 1.191, CPU 0.908.

### TLS-G (the guard at the final choice)

* The same search as BASE-TLS (100 of 100 pairs printed the same candidates).
* The guard kept plain scaling for 182 of the 1,000 printed candidates. For the reported
  candidate, it kept `terms` in 82 runs and plain in 18 runs. All 18 have a one-term program, so
  the per-term fit falls back and the two models are the same (a tie keeps plain).
* So the per-term model of a selected program with 2 or more terms was never worse on the
  validation rows than plain `a + b·f`.

### TLS-LM (T3.1)

* LM calls: 239. 99 had no budget left at the deme end, 66 had no constant inside the terms, 9
  were singular, 1 did not improve, and 64 improved. The guard rejected 20 of the 64, so 44
  tunings were written back.
* For comparison, BASE: 278 calls, 100 with no budget, 4 singular, 174 improved, 40 rejected by
  the guard.
* 20 pairs differ from BASE-TLS: 10 lower, 10 higher (median ratio of these 20: 1.003; geometric
  mean 0.882; range 0.387–1.525). Bos 1/5, Yac 5/5, Air 1/0, EnC 2/0, EnH 1/0.
* The feasibility note hoped that LM gives the one-term programs of Bos and Yac their constant
  tuning back. The result does not support this (Bos 1/5, Yac 5/5).

### BASE-LEX (A4, ε-lexicase)

* Lexicase acted in 84 runs. It did not act in the 10 Dow runs and in 6 Tow runs (one deme, so no
  exemplar selection).
* 153 selections in total; the median acting run made 1 selection (maximum 8).
* Cases used per selection: median 67.75 of a median of 516 training rows. In 16 runs every
  selection used all rows, so the random pick among the survivors decided.
* Pool size: median 747 candidates. Behavioural-score recomputes: median 2,188 per run, median CPU
  26 s, 6.5 % of the run CPU (maximum 14 %). These recomputes are not counted as evaluations.
* 81 pairs differ from BASE: 26 lower, 55 higher (median ratio of the 81: 1.028). The median
  program is smaller (44.5 against 53, but the pair ratio is 1.000) and the gap is smaller (0.72
  against 1.03).

### BASE-PS (the pole safeguard)

* 171 pole checks, 19 rejections (11 %). All 19 came from the size rule (S(new) > 10R and
  S(new) > S(old)); no probe prediction was non-finite.
* In 14 of the 19 rejections, the old program already had S(old) > 10R. 81 of the 152 accepted
  checks also had S(old) > 10R. Most programs, old or tuned, extrapolate far beyond 10R somewhere
  in the widened input box.
* The 13 pairs that differ from BASE all had a rejection. BASE-PS is worse in 12 of them (ratios
  0.975–2.047; the largest is Yac split 56). In the other 87 pairs the result is identical to BASE:
  the 2 evaluations of each check did not change the search.
* At the pilot gate the rejection share was at most 50 % (here 11 %).

### HC-LM

* 99 ok runs; Dow split 56 failed (see "Faults"). 1,824 LM calls, LM share 11.58 %.

## Observations (agent reading)

1. **BASE-TLS is the strongest result so far, and it repeats.** 0.784 on splits 51–60 after 0.879 on
   splits 41–50. Its gains come from four datasets (Dow, EnC, EnH, Tow). On Dow the programs are in
   effect linear regressions. A plain linear-regression baseline (ordinary least squares, or ridge,
   on all inputs) would show how much of the Dow gain needs the search at all (backlog 12.6).
2. **TLS-NP is the cheap variant of BASE-TLS.** 16 % less CPU at 2.3 % higher NMSE, and the same
   size. On Tow it finds small linear programs (26 nodes) with a higher NMSE. This fits the user's
   preference for smaller and cheaper programs at a moderate accuracy cost. Under the confirmation
   rule it is not a candidate (it is not "A better" than BASE-TLS).
3. **The per-term weights did not overfit at the final choice.** TLS-G never changed a multi-term
   model. The larger gaps of BASE-TLS on WiW, Air and Con are therefore not a per-term-against-plain
   effect at the end; they come from the programs that the search finds.
4. **LM inside the terms does not act at this budget.** 41 % of the TLS-LM calls had no budget, and
   28 % had no inner constant. T3.1 needs a budget for LM (for example a reserve) to be tested.
5. **ε-lexicase does not fit MOSES at this budget.** With about one exemplar selection per run, the
   rule changes one deme choice; the effect is a different search path with a small loss.
6. **The pole safeguard rejects good tunings.** The box probes reach regions that the test rows do
   not reach, and most programs extrapolate wildly there. The rule "worse than before and above
   10R" then rejects small increases. A relative rule, or probes near the data, could fix it; but
   this round had no pole failure to fix.
7. **No arm passes HC-LM by 10 %.** The TLS arms are 4–6 % lower than HC-LM, with smaller programs
   and less CPU. Pair ratios against HC-LM over the 99 pairs where both ended ok (not registered):
   BASE-TLS size 0.847 and CPU 0.791; TLS-NP size 0.800 and CPU 0.629; BASE size 0.638 and CPU
   0.748.

## Faults and deviations

1. **Dow HC-LM split 56 failed** (`ValueError: Expecting boolean value, got 801.904`, after 3.8 s).
   * Cause (reproduced locally): `load_rows` (`pymoses/gomea/const_opt.py`) loads the validation
     file of the guard with its own type inference. The loader infers types from the first 20
     lines (as classic MOSES does). In the validation rows of Dow split 56, the first 19 values of
     `x13` are 0, so `x13` is typed boolean, and a later row has 801.904.
   * The code is from round 5 (c5faa2e). Round 13 did not cause it.
   * An arm fails only when its guard loads the file. HC-LM tunes early, so it failed. The other
     Dow split 56 arms with the guard ended ok; their guard never loaded the file.
   * A check of all 10 datasets × splits 1–60 found the same failure only on Dow splits 11 and 56,
     and no case where the validation types differ from the training types without a failure.
   * Effect: 1 of 100 pairs in each of the 7 gap contrasts counts against HC-LM. No primary contrast
     uses HC-LM. Fix: backlog 12.1.
2. **S0 pytest failure:** the known toolchain test (see "Run health"). Not a code fault.
3. **Agent errors in the server instructions (2026-11-18):**
   * The agent added a separate server smoke step that the server guide does not have. The user
     removed it ("I want to run everything in one go").
   * The agent wrote "If S0 fails, the phase does not start". This is wrong: the orchestrator records
     an S0 failure as a flag and still runs the phase.
4. **Rule completions** after the approval (non-finite probes count as +∞, boolean probe inputs, the
   probe seed, the clip rule, the TLS-G tie, the Holm family of the descriptive contrasts): listed in
   the pre-registration. None changes an arm, a contrast or a gate.

## Limits

* One budget (20,000 evaluations). At this budget, LM at the deme end gets no evaluations in many
  runs (TLS-LM 99 of 239 calls, BASE 100 of 278), so the LM arms have little power.
* BASE-LEX made about one selection per run. The round does not test ε-lexicase with many
  selections.
* BASE-PS: one rule, one margin (10 %), one factor (10).
* TLS-G guards only the final choice, not the search.
* No linear-regression baseline. The Dow gain of BASE-TLS is not compared with a plain linear model.

## User decisions

**Open (2026-11-19).** The confirmation rule applies: BASE-TLS is "A better" than BASE again (0.784,
80/20). No TLS variant is "A better" than BASE-TLS. So the agent proposes **BASE-TLS**
(`--linear-scaling terms --const-opt none`, no LM, no guard) as the new reference. The user
decides; until then BASE stays the reference.
