# Server round 10 (T1.2): findings (2026-11-11)

**Pre-registration:** `SERVER-ROUND10-PREREGISTRATION.md`. **Results:** bundle
`bundle-final-20261111T091824Z.tar.gz` (local copy: `../server-results-round10/`). **Code:**
`f022395`. **Decisions:** `APPROVALS.md`, 2026-11-11 (no adoption rule; see "User decisions").

Test NMSE in the tables is × 100 and is the median over the splits 1–10. "Pairs" is the number of
splits where A is lower / where B is lower. "Ratio" is exp(median of log(A / B)) over the pairs:
above 1 means that A has the higher value (higher NMSE, larger program or more CPU).

## Summary

| Contrast | Family | Pairs | Holm p | Ratio | Verdict |
|---|---|---|---|---|---|
| 1. BASE vs HC-LM | primary | 19/81 | 1.5e-6 | 1.06 | tie |
| 2. BASE vs HC-LM-W | primary | 70/30 | 1.8e-5 | 0.96 | tie |
| 3. BASE vs NOCAP | primary | 33/67 | 0.0055 | 1.03 | tie |
| 4. BASE-S vs BASE | seeding control | 66/34 | 0.0011 | 0.97 | tie |
| 5. BASE-S vs HC-LM | seeding control | 36/64 | 0.0068 | 1.04 | tie |
| 6. BASE-S vs HC-LM-W | seeding control | 82/18 | 1.5e-9 | 0.92 | tie |
| 7. BASE-S vs NOCAP | seeding control | 48/52 | 0.81 | 1.01 | tie |
| 8. HC-LM-W vs HC-LM | widen search | 13/87 | 3.2e-12 | 1.18 | **HC-LM-W worse** |

* The 10 % rule gives "tie" although the p values are small and the pair counts are uneven. A
  verdict needs a ratio of at most 0.9 or at least 1/0.9 = 1.11. Only contrast 8 reaches it.
* HC-LM is lower than BASE in 81 of 100 pairs. At the median pair, BASE is 6 % worse. HC-LM
  programs are larger (pooled median size 75.5 against 47). HC-LM uses more CPU (674 against
  430 s).
* NOCAP is lower than BASE in 67 of 100 pairs (ratio 1.03). NOCAP programs are about 4 times
  larger (191 against 47). NOCAP uses about 5.6 times the CPU (2,419 against 430 s).
* BASE is lower than HC-LM-W in 70 of 100 pairs (ratio 0.96). HC-LM-W has the smallest programs
  (32) and the lowest CPU (234 s).
* The size and CPU family (Holm over 10) gives a verdict for 8 of 10 contrasts. The two BASE-S vs
  BASE rows are ties.

## Run health

* **Steps:** S0 1.34 h, phase1bd 19.62 h, one orchestrator session. W = 11; W = 3 for Dow and
  W = 7 for Tow (`manifest_workers` in `state.json`). The pre-registration estimated 12–17 h plus
  1–2 h for Dow.
* **S0:** reproducibility 4 of 4 matched. pytest: 9,322 passed, 1 failed
  (`test_toolchain_drift_is_clean_on_the_real_repo`), 13 skipped. The same test failed in rounds
  6 to 9: the server's toolchain (Ubuntu g++ 13.3) differs from the toolchain that the fixtures
  record (g++ 16.2.1). It is not a code fault.
* **Runs:** 500 of 500 ok, 0 failed. Each run used 20,000 evaluations. For BASE-S, the GP-GOMEA
  seed evaluations are part of the 20,000.
* **Code:** `f022395` on the server, tree clean (`git-head.txt`; `round-summary.md`: dirty False).
* **CPU:** 152 h in total over the 500 runs. NOCAP used 85 h (56 %), HC-LM-W 9 h.

### Memory

Peak RSS in GB is `ru_maxrss_kb` / 1e6, as in round 9. The share of the limit is
`ru_maxrss_kb` / (limit × 1,024). The runner sets the limit on the address space (`RLIMIT_AS`),
not on the RSS.

| Manifest | `mem_limit_mb` | W | Highest peak RSS | Run | Share of the limit |
|---|---|---|---|---|---|
| Dow | 16,384 | 3 | 9.75 GB | NOCAP, split 3 | 58 % |
| Tow | 8,192 | 7 | 8.14 GB | NOCAP, split 5 | 97 % |
| 8 other datasets | 5,120 | 11 | 5.14 GB | Bos NOCAP, split 6 | 98 % |

* No run failed. Four NOCAP runs reached 90 % of the limit or more: Bos splits 6 and 10 (98 %,
  95 %) and Tow splits 5 and 2 (97 %, 91 %).
* The other arms stayed below 3.4 GB (highest: Dow HC-LM, split 6, 3.34 GB). BASE stayed below
  1 GB.
* Round 9 had memory failures at 5.09 GB (Tow, 5,120 MB) and 8.04 GB (Dow, 8,192 MB). In this
  round, Bos NOCAP split 6 reached 5.14 GB under the same 5,120 MB limit and did not fail.
* No NOCAP run failed, so the pre-registered form "with and without the failed pairs" is not
  necessary. The pre-registration calls this contrast 2; BASE vs NOCAP is contrast 3.

### Consistency

* **BASE:** all 100 runs (10 datasets, splits 1–10) have the same test NMSE and the same final
  program as in round 9 (1ba for 8 datasets, 1bb for Con and EnH). Thus the BASE medians equal
  round 9 on all 10 datasets.
* **BASE-S:** the 40 runs on Air, Con, EnH and Yac are identical to 1bb in the same way (Yac:
  splits 1–10 of the 20 splits of 1bb).
* The code changed from `efcea29` to `f022395` between the rounds. The GOMEA output did not
  change.

## Main experiment (contrasts 1–3)

Pooled over 100 pairs (10 datasets × 10 splits). Pairs: BASE lower / other arm lower. The size
and CPU columns give the pooled medians (BASE / other arm) and the verdict of the secondary
contrast.

| Contrast | Pairs | Holm p | Ratio | Verdict | Size BASE / X | Size verdict | CPU s BASE / X | CPU verdict |
|---|---|---|---|---|---|---|---|---|
| BASE vs HC-LM | 19/81 | 1.5e-6 | 1.06 | tie | 47 / 75.5 | BASE smaller | 430 / 674 | BASE lower |
| BASE vs HC-LM-W | 70/30 | 1.8e-5 | 0.96 | tie | 47 / 32 | BASE larger | 430 / 234 | BASE higher |
| BASE vs NOCAP | 33/67 | 0.0055 | 1.03 | tie | 47 / 191 | BASE smaller | 430 / 2,419 | BASE lower |

The datasets have different NMSE scales, so this table gives no pooled NMSE medians (the
`median_a` and `median_b` columns of `contrasts.csv`). Use the ratio and the per-dataset medians.

### Program size and CPU (secondary, Holm over 10)

"A better" means a smaller program or less CPU.

| Metric | Contrast | Pairs (ties) | Holm p | Ratio | Verdict | Median A / B |
|---|---|---|---|---|---|---|
| Size | BASE vs HC-LM | 86/14 | 2.6e-13 | 0.58 | A better | 47 / 75.5 |
| Size | BASE vs HC-LM-W | 19/78 (3) | 3.6e-10 | 1.57 | A worse | 47 / 32 |
| Size | BASE vs NOCAP | 99/1 | < 1e-15 | 0.26 | A better | 47 / 191 |
| Size | BASE-S vs BASE | 37/61 (2) | 0.00064 | 1.107 | tie | 62 / 47 |
| Size | HC-LM-W vs HC-LM | 97/3 | < 1e-15 | 0.37 | A better | 32 / 75.5 |
| CPU s | BASE vs HC-LM | 85/15 | 5.6e-11 | 0.72 | A better | 430 / 674 |
| CPU s | BASE vs HC-LM-W | 8/92 | < 1e-15 | 1.99 | A worse | 430 / 234 |
| CPU s | BASE vs NOCAP | 100/0 | < 1e-15 | 0.20 | A better | 430 / 2,419 |
| CPU s | BASE-S vs BASE | 43/57 | 0.014 | 1.08 | tie | 532 / 430 |
| CPU s | HC-LM-W vs HC-LM | 100/0 | < 1e-15 | 0.33 | A better | 234 / 674 |

* `contrasts.csv` gives Holm p = 0.0 for the rows with "< 1e-15". The normal approximation
  computes 1 − Φ(z), and this value rounds to 0 in floating point. A direct computation with
  erfc gives raw p values from 4e-18 to 2.4e-17.
* BASE-S vs BASE size: the ratio 1.107 is just below 1/0.9 = 1.111, so the verdict is "tie".

## Per dataset

### Test NMSE

Bold marks the lowest median. On Yac, BASE-S (0.66) and HC-LM (0.71) are equal at one decimal.

| Dataset | BASE | BASE-S | HC-LM | HC-LM-W | NOCAP |
|---|---|---|---|---|---|
| Air | 39.1 | 36.5 | **31.5** | 38.6 | 39.0 |
| Bos | 19.6 | 19.4 | 27.5 | 22.0 | **17.9** |
| Con | 36.1 | 30.9 | **20.7** | 38.1 | 25.2 |
| Dow | 39.8 | 33.8 | 33.3 | 42.8 | **30.5** |
| EnC | 11.5 | **11.1** | 11.5 | 12.2 | 12.1 |
| EnH | 8.3 | 8.0 | **7.4** | 8.8 | 8.1 |
| Tow | 18.8 | 18.2 | **16.6** | 19.1 | 16.9 |
| WiR | 66.0 | **64.7** | 65.0 | 65.6 | 65.4 |
| WiW | 72.1 | 73.1 | **70.4** | 74.2 | 71.6 |
| Yac | 0.8 | **0.7** | **0.7** | 1.0 | 1.2 |

### Program size

| Dataset | BASE | BASE-S | HC-LM | HC-LM-W | NOCAP |
|---|---|---|---|---|---|
| Air | 77 | 84.5 | 147 | 40.5 | 289.5 |
| Bos | 64.5 | 57.5 | 101 | 35 | 237 |
| Con | 47 | 85.5 | 57 | 30.5 | 186.5 |
| Dow | 26 | 67 | 50 | 30.5 | 171 |
| EnC | 45 | 41 | 62 | 24 | 130 |
| EnH | 44.5 | 51 | 75 | 26 | 185 |
| Tow | 26 | 58.5 | 53 | 34.5 | 174 |
| WiR | 54.5 | 65.5 | 106.5 | 31 | 157.5 |
| WiW | 76.5 | 76 | 126.5 | 38 | 275 |
| Yac | 33.5 | 34.5 | 72 | 27 | 93 |

### Demes per run: median [min–max]

The count includes only demes with evaluations (`ledger.requests` > 0). Every deme of the round
had evaluations.

| Dataset | BASE | BASE-S | HC-LM | HC-LM-W | NOCAP |
|---|---|---|---|---|---|
| Air | 3 [3–4] | 2.5 [2–3] | 12.5 [11–20] | 3 [3–4] | 8 [6–11] |
| Bos | 3 [2–4] | 2 [1–3] | 20 [16–22] | 4 [3–4] | 5.5 [4–10] |
| Con | 2 [2–3] | 1.5 [1–3] | 25.5 [21–37] | 3.5 [3–4] | 6.5 [4–16] |
| Dow | 1 [1–1] | 1 [1–1] | 9.5 [8–11] | 5 [3–6] | 3 [3–4] |
| EnC | 4 [2–5] | 3 [2–5] | 21.5 [11–32] | 4.5 [3–6] | 8 [7–11] |
| EnH | 3.5 [2–4] | 3.5 [2–4] | 20.5 [9–25] | 4 [3–5] | 7 [5–10] |
| Tow | 1 [1–2] | 1 [1–1] | 15 [12–19] | 4 [3–5] | 3 [2–4] |
| WiR | 2 [2–3] | 2 [1–3] | 19.5 [17–27] | 4 [3–5] | 9 [6–11] |
| WiW | 2 [2–3] | 1.5 [1–2] | 16.5 [14–18] | 3 [3–5] | 5 [4–8] |
| Yac | 6.5 [6–9] | 5 [2–6] | 23.5 [10–28] | 4 [3–4] | 20.5 [15–26] |

### Train–test gap

The median over the splits of (test NMSE − training NMSE), × 100.

| Dataset | BASE | BASE-S | HC-LM | HC-LM-W | NOCAP |
|---|---|---|---|---|---|
| Air | -0.1 | -0.8 | -0.1 | -1.1 | 0.1 |
| Bos | 1.1 | -1.5 | 10.2 | 0.4 | 2.4 |
| Con | 3.4 | 3.1 | 1.7 | 2.1 | 3.0 |
| Dow | 2.0 | 1.2 | 4.8 | 4.5 | 4.1 |
| EnC | 1.3 | 1.5 | 1.4 | 1.5 | 1.9 |
| EnH | 1.3 | 1.4 | 0.9 | 1.2 | 1.0 |
| Tow | 0.4 | 0.6 | 0.4 | 0.3 | 0.6 |
| WiR | 2.5 | 2.0 | 3.0 | 1.4 | 2.6 |
| WiW | -0.1 | 0.2 | 0.1 | -0.3 | 0.6 |
| Yac | 0.0 | 0.1 | 0.1 | 0.1 | 0.1 |

### CPU s

| Dataset | BASE | BASE-S | HC-LM | HC-LM-W | NOCAP |
|---|---|---|---|---|---|
| Air | 655 | 569 | 962 | 239 | 2,606 |
| Bos | 352 | 342 | 561 | 185 | 2,211 |
| Con | 433 | 508 | 331 | 203 | 2,315 |
| Dow | 314 | 818 | 862 | 283 | 2,762 |
| EnC | 344 | 364 | 377 | 95 | 1,723 |
| EnH | 346 | 371 | 414 | 110 | 1,706 |
| Tow | 926 | 1,524 | 1,258 | 844 | 7,073 |
| WiR | 537 | 597 | 956 | 348 | 2,484 |
| WiW | 1,455 | 1,336 | 2,162 | 869 | 5,784 |
| Yac | 259 | 189 | 343 | 91 | 661 |

### Per-dataset contrasts on test NMSE (descriptive, Holm over 40)

Each cell: pairs (A lower / B lower) · Holm p · pair ratio. Paired Wilcoxon on the test NMSE
differences, as pre-registered. These contrasts give no verdicts: with 10 pairs and Holm over 40,
the lowest possible p is 0.078.

| Dataset | BASE vs HC-LM | BASE vs HC-LM-W | BASE vs NOCAP | BASE-S vs BASE |
|---|---|---|---|---|
| Air | 0/10 · 0.078 · 1.25 | 5/5 · 1.0 · 1.00 | 2/8 · 1.0 · 1.05 | 8/2 · 1.0 · 0.95 |
| Bos | 5/5 · 1.0 · 0.92 | 8/2 · 1.0 · 0.90 | 5/5 · 1.0 · 1.01 | 4/6 · 1.0 · 1.05 |
| Con | 0/10 · 0.078 · 1.84 | 4/6 · 1.0 · 1.07 | 2/8 · 0.32 · 1.50 | 8/2 · 1.0 · 0.94 |
| Dow | 1/9 · 0.38 · 1.21 | 7/3 · 1.0 · 0.93 | 0/10 · 0.078 · 1.41 | 9/1 · 1.0 · 0.90 |
| EnC | 3/7 · 1.0 · 1.04 | 8/2 · 0.32 · 0.95 | 9/1 · 0.14 · 0.95 | 6/4 · 1.0 · 0.97 |
| EnH | 1/9 · 0.32 · 1.05 | 8/2 · 0.71 · 0.83 | 1/9 · 0.32 · 1.03 | 6/4 · 1.0 · 0.96 |
| Tow | 2/8 · 0.85 · 1.17 | 5/5 · 1.0 · 0.98 | 2/8 · 1.0 · 1.10 | 5/5 · 1.0 · 0.96 |
| WiR | 1/9 · 0.20 · 1.03 | 6/4 · 1.0 · 1.00 | 3/7 · 1.0 · 1.01 | 9/1 · 0.32 · 0.98 |
| WiW | 1/9 · 0.38 · 1.02 | 9/1 · 0.14 · 0.97 | 2/8 · 1.0 · 1.01 | 2/8 · 1.0 · 1.01 |
| Yac | 5/5 · 1.0 · 1.07 | 10/0 · 0.078 · 0.72 | 7/3 · 0.71 · 0.65 | 9/1 · 0.71 · 0.81 |

## Seeding control (contrasts 4–7)

The control family has its own Holm correction. It does not change the main result.

* **Against BASE:** BASE-S is lower in 66 of 100 pairs (ratio 0.97, Holm p 0.0011, tie). The
  BASE-S median is lower on 9 of 10 datasets. WiW is the exception (73.1 against 72.1). The
  per-dataset pair ratios go from 0.81 (Yac) to 1.05 (Bos).
* **Against the other arms:** HC-LM 36/64 (ratio 1.04), HC-LM-W 82/18 (0.92), NOCAP 48/52
  (1.01, Holm p 0.81). All three are ties.
* **Size and CPU:** 62 against 47 (ratio 1.107) and 532 against 430 s (ratio 1.08). Both are
  ties.
* **Seed cost:** GP-GOMEA used 4,001–4,655 of the 20,000 evaluations per run. Its CPU
  (`gpgomea_cpu_s`) was at most 0.34 s per run.
* **Lineage:** in 99 of 100 runs, the final program descends from a seed. In Bos split 2, the seed
  check rejected 16 GP-GOMEA programs and kept none. Thus the run started without a seed
  (lineage "empty"). Eight other runs had 1–4 seeds in place of 5.
* **Dow and Tow:** BASE-S had 1 deme in all 20 runs (on Dow with 1–3 generations), and LM got 0
  evaluations. See "Observations", item 2.

## Widen search (contrast 8)

* **LM reserve:** with `-a hc --const-opt lm --hc-widen-search 1`, the climb in each deme gets the
  deme budget minus a reserve of min(500, deme budget // 2). LM then runs on the best program with
  the rest. The user chose the reserve on 2026-11-10 ("Reserve LM budget"). Commit `493d2b8`
  implements it.
* **Why:** the local check before the round (Air, Bos, Tow, Yac, split 1) showed no LM without
  the reserve. One deme used all 20,000 evaluations, so the one LM call got no budget.
* **Result:** HC-LM-W is worse than HC-LM: 13/87 pairs, Holm p 3.2e-12, ratio 1.18. This is the
  only test NMSE verdict of the round.
* **Per dataset:** the HC-LM-W median is higher on 9 of 10 datasets. Bos is the exception (22.0
  against 27.5). Pairs (HC-LM-W lower / HC-LM lower): Air 0/10, Bos 5/5, Con 0/10, Dow 1/9,
  EnC 2/8, EnH 0/10, Tow 1/9, WiR 3/7, WiW 0/10, Yac 1/9. The pair ratios go from 1.03 (WiR) to
  1.69 (Yac). These per-dataset numbers are descriptive; the pre-registration has no test for them.
* **Deme structure:** in all 100 runs, the climb of the first deme used 19,500 evaluations and
  ended at the deme budget. With its LM call, the first deme used 19,512–19,623 evaluations
  (97.6–98.1 % of the run). The other 377–488 evaluations went to 2–5 small demes. Every HC-LM-W
  deme ended at its budget, none at a local optimum.
* **Size and CPU:** HC-LM-W programs are smaller than HC-LM programs (97/3, pooled median 32
  against 75.5). HC-LM-W uses less CPU (100/0, 234 against 674 s).

| Dataset | Demes | Steps in the first deme | LM calls | LM improved | Guard rejected / checked | LM share of evaluations |
|---|---|---|---|---|---|---|
| Air | 3 [3–4] | 39.5 [35–41] | 34 | 32 (94 %) | 0/24 (0 %) | 1.0 % [0.7–1.1] |
| Bos | 4 [3–4] | 37.5 [15–51] | 37 | 36 (97 %) | 17/27 (63 %) | 0.9 % [0.6–1.3] |
| Con | 3.5 [3–4] | 33.5 [19–50] | 35 | 32 (91 %) | 8/25 (32 %) | 1.1 % [0.6–1.3] |
| Dow | 5 [3–6] | 20 [15–48] | 49 | 41 (84 %) | 9/36 (25 %) | 0.5 % [0.3–1.4] |
| EnC | 4.5 [3–6] | 11 [10–41] | 44 | 41 (93 %) | 13/35 (37 %) | 0.7 % [0.4–1.3] |
| EnH | 4 [3–5] | 16.5 [12–24] | 41 | 38 (93 %) | 12/31 (39 %) | 0.9 % [0.7–1.1] |
| Tow | 4 [3–5] | 30.5 [18–45] | 39 | 39 (100 %) | 6/29 (21 %) | 0.9 % [0.5–1.4] |
| WiR | 4 [3–5] | 22 [17–35] | 40 | 38 (95 %) | 13/30 (43 %) | 0.9 % [0.5–1.3] |
| WiW | 3 [3–5] | 37 [12–42] | 33 | 32 (97 %) | 3/23 (13 %) | 1.2 % [0.6–1.3] |
| Yac | 4 [3–4] | 18.5 [13–21] | 37 | 37 (100 %) | 6/28 (21 %) | 0.8 % [0.7–1.0] |
| All | | | 389 | 366 (94 %) | 87/288 (30 %) | |

"Steps in the first deme" is the median [min–max] over the 10 runs. The other columns are as in
"HC-LM details" below.

## HC-LM details

| Dataset | Demes | Evaluations per deme | Steps per deme | LM calls | LM improved | Guard rejected / checked | LM share of evaluations |
|---|---|---|---|---|---|---|---|
| Air | 12.5 [11–20] | 1,106 | 6 [2–51] | 136 | 109 (80 %) | 10/107 (9 %) | 16.0 % [2.1–19.4] |
| Bos | 20 [16–22] | 1,039 | 5 [2–22] | 193 | 173 (90 %) | 89/171 (52 %) | 10.6 % [3.4–13.2] |
| Con | 25.5 [21–37] | 716 | 5 [2–20] | 270 | 261 (97 %) | 80/260 (31 %) | 11.7 % [7.0–12.6] |
| Dow | 9.5 [8–11] | 2,118 | 5 [2–9] | 95 | 85 (89 %) | 26/85 (31 %) | 1.7 % [0.9–2.7] |
| EnC | 21.5 [11–32] | 817 | 5 [2–41] | 210 | 202 (96 %) | 40/200 (20 %) | 11.2 % [6.7–13.9] |
| EnH | 20.5 [9–25] | 914 | 6 [2–59] | 188 | 181 (96 %) | 40/178 (22 %) | 12.4 % [6.2–14.2] |
| Tow | 15 [12–19] | 1,245 | 4 [2–28] | 155 | 146 (94 %) | 21/145 (14 %) | 5.8 % [4.2–9.0] |
| WiR | 19.5 [17–27] | 1,006 | 5 [2–16] | 199 | 190 (95 %) | 100/189 (53 %) | 13.6 % [9.9–16.6] |
| WiW | 16.5 [14–18] | 1,149 | 6 [2–29] | 165 | 158 (96 %) | 32/155 (21 %) | 13.1 % [11.8–15.4] |
| Yac | 23.5 [10–28] | 770 | 6 [2–38] | 214 | 201 (94 %) | 4/200 (2 %) | 15.7 % [1.4–18.0] |
| All | | | | 1,825 | 1,706 (93 %) | 442/1,690 (26 %) | |

* **Columns:** "Demes" is the median [min–max] per run. "Evaluations per deme" (median) and
  "Steps per deme" (median [min–max]) pool the demes of the 10 runs. A step is the `generations`
  field of the deme's "end" event. The LM counts are sums over the 10 runs. "LM share of
  evaluations" is the LM evaluations / 20,000, median [min–max] per run.
* **Deme ends:** 1,743 of 1,825 demes (96 %) ended at a local optimum. The other 82 demes are the
  last deme of 82 runs. They ended at the run budget, and their LM call had the status "budget".
* **LM:** LM improved the training error in 1,706 of 1,825 calls (93 %). The guard rejected 442
  of 1,690 checked results (26 %): about half on Bos (52 %) and WiR (53 %), 2 % on Yac.
* **LM share:** the LM share of the evaluations is 11–16 % (median per dataset) on 8 datasets,
  5.8 % on Tow and 1.7 % on Dow. The largest LM call used 465 evaluations (Air, split 1).
* **Bos:** on Bos, HC-LM has the largest train–test gap of the round (10.2; test 27.5, training
  16.1). The HC-LM median is above the BASE median on Bos (27.5 against 19.6) and EnC (11.54
  against 11.46).

## Observations

Facts only. This round does not test them, and they give no recommendation.

1. **Demes, test NMSE and program size.** The arms that spread the evaluations over more demes
   than BASE have lower test NMSE and larger programs than BASE.
   * HC-LM: 9.5–25.5 demes (median per dataset) against 1–6.5 for BASE. HC-LM has the lowest
     median test NMSE on 5 datasets (Air, Con, EnH, Tow, WiW). On Yac, it equals BASE-S at one
     decimal. Its pooled median size is 75.5 against 47.
   * Round 9 BASE-C250 (cap of 250 evaluations per deme): 50.5–59.5 demes against 1.5–6.5 for
     BASE. Its median test NMSE was lower on 5 of 6 datasets. Its median size was 54–107.5
     against 31–67.5.
   * NOCAP: 3–20.5 demes. It is lower than BASE in 67 of 100 pairs, with a pooled median size of
     191.
   * HC-LM-W puts 97.6–98.1 % of the evaluations in one deme. It has the highest median test NMSE
     of the five arms on 6 of 10 datasets, and the smallest programs (32).
   * BASE-S does not follow the pattern. Its deme count (1–5) is not above the BASE count on any
     dataset, but it is lower than BASE in 66 of 100 pairs.
   * The deme count is not a factor of this design. The arms also differ in the search method,
     the LM use and the complexity handling. This round cannot separate these effects.
2. **LM in BASE and BASE-S on Dow and Tow.** One deme used the whole budget, so LM got 0
   evaluations. This occurred for BASE on Dow in 10 of 10 runs and on Tow in 7 of 10. It occurred
   for BASE-S on Dow and Tow in 10 of 10 runs each.
   * It is the same budget effect that the LM reserve removed for HC-LM-W.
   * BASE-S also got 0 LM evaluations in 16 runs on Bos, Con, WiR and WiW.
   * On Dow and Tow, HC-LM is lower than BASE in 9/10 and 8/10 pairs (pair ratios 1.21 and 1.17).
3. **The 10 % rule and the per-dataset effects.** The pooled ratios of contrasts 1–3 are
   0.96–1.06. The per-dataset pair ratios go from 0.65 to 1.84.
   * BASE vs HC-LM: Con 1.84 (36.1 against 20.7), Air 1.25, Dow 1.21.
   * BASE vs NOCAP: Con 1.50, Dow 1.41, Yac 0.65.
   * BASE vs HC-LM-W: Yac 0.72.
4. **NOCAP.** Plain GOMEA without LM has the lowest median on Bos and Dow. It has the largest
   programs (pooled median 191) and the highest CPU (2,419 s). It used 56 % of the round's CPU.
5. **Memory.** NOCAP (N = 16, no cap) reached 97–98 % of the memory limit on Bos and Tow.

## Limits

* Splits 1–10 are selection data: earlier rounds chose settings on them. The results do not
  confirm the base on new data.
* One evaluation budget (20,000). Equal evaluations are not equal CPU: the pooled median CPU goes
  from 234 s (HC-LM-W) to 2,419 s (NOCAP).
* HC with LM is new code. This round is its first use, after local tests only.
* The LM reserve formula min(500, deme budget // 2) is an agent choice (`APPROVALS.md`). No other
  reserve value ran.
* The 10 % rule hides large per-dataset effects (pair ratios 0.65–1.84). The per-dataset
  contrasts cannot give a verdict (lowest possible Holm p 0.078).
* The arms differ in more than one factor (search method, LM, deme count, complexity handling).
  The contrasts compare whole settings, not single factors.

## User decisions

The pre-registration has no adoption rule. On 2026-11-11 the agent recommended three points. The
user answered: "Agreed to the 1 and 2. Not sure what you meant by 3". After an explanation of
point 3, the user answered: "Yes, check the code and write the draft".

* **BASE stays the compact reference** for stage 2.
* **HC-LM is the accuracy baseline.** Every GOMEA change is compared with it.
* **Next round:** a test of the deme count in GOMEA (`SERVER-ROUND11-PREREGISTRATION.md`, a draft
  that the user has not approved).
