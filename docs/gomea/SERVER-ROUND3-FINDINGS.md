# Server round 3 findings: constant tuning, equal CPU, size control, uniform start, new datasets (phases 1an to 1ar)

**Status:** complete.

* **Run:** 2026-10-30 12:41 to 2026-11-02 21:47 UTC, on the same pod as rounds 1 and 2
  (`jupyter-yderese`, cgroup quota 12 CPUs, 64 GB). W = 11 workers (limited by CPU). Two
  orchestrator sessions with a stop between them (see "Deviations"). The steps ran for about
  18 h. Most of the elapsed time was idle time before the restart.
* **Runs:** 640 (150 in 1an, 150 in 1ao, 90 in 1ap, 60 in 1aq, 190 in 1ar), all `ok`, 0
  failed. Total CPU 168.6 h (the sum of `cpu_s` over all run summaries).
* **Code:** 1an, 1ap and 1aq ran on `484d8e8`. 1ao and 1ar ran on `afa051b`. The difference is
  only `tools/gomea/set_const_opt.py` and its tests. The tree was dirty in two round-1 configs
  and in the 1ao and 1ar configs that the choice script writes.
* **Protocol:** `SERVER-ROUND3-PREREGISTRATION.md`.
* **Results (outside git):** `../server-results-round3/gomea-results/`. The round log is
  `server-run3/round-summary.md`. Reports: `phase1an/report-stage1`, `phase1ao/report-stage2`,
  `phase1ap/report-stage1`, `phase1aq/report-stage1`, `phase1ar/report-stage2`.
* **How the extra numbers were computed:** a read-only script read each `summary.json`. "Gap"
  is the median over runs of (test NMSE − train NMSE). The other numbers are from
  `aggregate.csv` and `contrasts.csv`.

**Answer in short:**

* **LM constant tuning is the main result.** At equal CPU it is **better** than GOMEA-LS on all
  three registered datasets: test NMSE −19 % on Con, −26 % on EnH, −12 % on Air. Its programs
  are also smaller.
* **At equal evaluations, LM is a tie** on all three datasets (1an). It used about half the CPU
  or less.
* **Plain GOMEA-LS against HC-LS at equal CPU:** better on Air, and a tie on Con and EnH (1ao).
  It is a tie on all five new datasets (1ar).
* **The size penalty does not qualify.** It makes programs 56–83 % smaller, but both strengths
  fail the accuracy limit of the adoption rule.
* **The uniform start gives no gain:** a tie on all three datasets, with larger programs and
  more CPU. Do not make it the default.
* **Recommendation:** make `--const-opt lm` a candidate default for regression, with linear
  scaling. The user decides. WiR needs care: there LM raised the gap from 1.2 to 9.7 with no
  gain in test NMSE.

## A correction to the design of 1ap

The pre-registration and my report to the user call the 0.001 arm a "weaker" penalty. **This
is wrong.** The penalty is `complexity / R` (`accept_worse`, `pymoses/gomea/optimizer.py`), so
a smaller R is a stronger penalty: 0.001 is 5× stronger than 0.005. The user chose this arm
to lower the accuracy cost, on my wrong description. The data agree with the formula: PEN1
gives smaller programs and a higher error than PEN5 on all three datasets. So 1ap did not test
a weaker penalty. That test is still open (`BACKLOG.md`, section 6).

## S0. Environment check

* **pytest:** 1 failed, 8,792 passed, 13 skipped. The failure is the same environmental test as
  in rounds 1 and 2, `test_toolchain_drift_is_clean_on_the_real_repo` (the pod's g++ 13.3.0
  against the fixtures' g++ 16.2.1).
* **Reproducibility:** 4 of 4 reference runs matched exactly. Thus with `--const-opt none`, the
  new code reproduces the earlier rounds.
* **A second correction:** I told the user that a failed S0 stops the run. It does not. The
  orchestrator only records a flag and continues.

## 1an. Constant tuning at 10,000 evaluations

| Dataset | Arm | Test NMSE | vs base | Size | Gap | CPU (s) | Tuning share of evaluations |
|---|---|---|---|---|---|---|---|
| Con | GOMEA-LS | 25.85 | | 160 | 3.2 | 922 | |
| | LM | 19.73 | −23.7 % | 112 | 4.0 | 463 | 31 % |
| | MUT | 23.67 | −8.4 % | 206 | 4.6 | 756 | 59 % |
| | POLISH | 22.30 | −13.7 % | 160 | 4.0 | 882 | 3 % |
| | HC-LS | 38.57 | | 32 | 2.0 | 114 | |
| EnH | GOMEA-LS | 7.49 | | 276 | 0.9 | 1,077 | |
| | LM | 6.54 | −12.8 % | 96 | 0.8 | 336 | 36 % |
| | MUT | 8.00 | +6.8 % | 293 | 0.9 | 1,025 | 57 % |
| | POLISH | 7.22 | −3.7 % | 262 | 0.8 | 1,025 | 3 % |
| | HC-LS | 9.49 | | 53 | 1.5 | 133 | |
| Air | GOMEA-LS | 35.21 | | 289 | −1.6 | 1,199 | |
| | LM | 35.00 | −0.6 % | 146 | 0.3 | 655 | 40 % |
| | MUT | 32.15 | −8.7 % | 695 | 0.9 | 3,210 | 49 % |
| | POLISH | 32.04 | −9.0 % | 281 | 1.6 | 1,177 | 5 % |
| | HC-LS | 47.05 | | 84 | 1.8 | 349 | |

* **Primary (6):** all ties. LM was lower in 8/10, 8/10 and 7/10 splits (Holm p 0.29, 0.29 and
  0.52). MUT had Holm p ≥ 0.58. No contrast reached the extension range.
* **Secondary:** GOMEA-LS beat HC-LS on EnH and Air (10/10, Holm p 0.018) and tied on Con (9/1,
  Holm p 0.049). POLISH tied on all three (Holm p 0.027–0.049), and validation accepted its
  program in 28 of 30 runs. LM vs MUT tied on all three.
* **The choice for 1ao and 1ar:** LM, by the registered rule (pooled median 19.73 against
  23.67). This is a rule result. It does not show that LM beats mutation.
* **LM calls:** 791 in 30 runs, 667 lowered the training error (84 %), 103 had a singular
  matrix, 0 failed.
* **My cost estimate was wrong:** I told the user that LM would use most of the budget. It used
  a median of 36 % (range 12–50 %).

## 1ao. Equal CPU (1,000 s)

| Dataset | Splits | Arm | Test NMSE | Size | Gap | Evaluations (tuning) |
|---|---|---|---|---|---|---|
| Con | 10 | GOMEA-LS | 24.93 | 162 | 3.0 | 10,810 |
| | | GOMEA-LS-TUNE | 20.12 | 140 | 5.5 | 17,450 (6,615) |
| | | HC-LS | 26.72 | 57 | 3.2 | 58,419 |
| EnH | 20 | GOMEA-LS | 8.26 | 242 | 1.8 | 10,277 |
| | | GOMEA-LS-TUNE | 6.09 | 105 | 0.9 | 22,562 (8,626) |
| | | HC-LS | 8.64 | 60 | 1.8 | 39,733 |
| Air | 20 | GOMEA-LS | 37.87 | 262 | −0.5 | 8,789 |
| | | GOMEA-LS-TUNE | 33.42 | 149 | 0.6 | 14,187 (6,250) |
| | | HC-LS | 44.50 | 102 | 3.5 | 23,750 |

| Contrast | Dataset | Stage | A lower / B lower | Holm p | Median change | Verdict |
|---|---|---|---|---|---|---|
| GOMEA-LS vs HC-LS | Con | 1 | 4 / 6 | 0.77 | −6.7 % | tie |
| | EnH | 2 | 15 / 5 | 0.034 | −4.4 % | tie |
| | Air | 1 | 9 / 1 | 0.020 | −14.9 % | **better** |
| TUNE vs GOMEA-LS | Con | 1 | 9 / 1 | 0.020 | −19.3 % | **better** |
| | EnH | 1 | 10 / 0 | 0.012 | −26.2 % | **better** |
| | Air | 2 | 19 / 1 | 0.00042 | −11.8 % | **better** |

* **Holm p on the extended datasets:** the stage-2 table recalculates over the same family. The
  verdicts of the contrasts that did not extend stay at stage 1, as registered.
* **Secondary:** TUNE beat HC-LS on all three (9/1, 20/0 and 20/0, median −25 %, −29 % and
  −25 %). Its programs are still 1.5–2.5× those of HC-LS.
* **Accuracy against the literature:** the review predicted a small gain or a tie, and that
  tuning does not make models smaller. In our runs LM gave both a clear gain at equal CPU and
  smaller programs.
* **What this does not test:** Harrison et al. 2023 compared LM with mutation at equal time. We
  did not run mutation at equal CPU. So this result does not contradict theirs.
* **The cause of the smaller programs and the low CPU of LM is not measured here.**

## 1ap. Size control (10,000 evaluations)

| Dataset | Arm | Size | vs base | Test NMSE | vs base | CPU (s) |
|---|---|---|---|---|---|---|
| Con | GOMEA-LS | 160 | | 25.85 | | 921 |
| | PEN5 (R 0.005) | 70 | −56 % | 27.35 | +5.8 % | 412 |
| | PEN1 (R 0.001, stronger) | 48 | −70 % | 29.64 | +14.6 % | 212 |
| EnH | GOMEA-LS | 276 | | 7.49 | | 1,076 |
| | PEN5 | 61 | −78 % | 9.30 | +24.1 % | 255 |
| | PEN1 | 47 | −83 % | 9.65 | +28.8 % | 243 |
| Air | GOMEA-LS | 289 | | 35.21 | | 1,207 |
| | PEN5 | 81 | −72 % | 40.63 | +15.4 % | 420 |
| | PEN1 | 51 | −83 % | 41.66 | +18.3 % | 280 |

* **Primary (size, 6):** all **better** (Holm p 0.012).
* **Secondary (test NMSE):** PEN1 is **worse** on EnH (0/10) and Air (1/9). All others are ties.
* **Adoption rule:** neither strength qualifies. The rule allows at most +10 % on each dataset.
  PEN5 passes only on Con, and PEN1 passes on none.

## 1aq. Uniform start (10,000 evaluations)

| Dataset | GOMEA-LS: NMSE, size, CPU | GOMEA-LS-UNIF: NMSE, size, CPU | Verdict |
|---|---|---|---|
| Con | 25.85, 160, 921 s | 27.95, 223, 1,008 s | tie (5/5, Holm p 1) |
| EnH | 7.49, 276, 1,076 s | 7.52, 351, 1,376 s | tie (5/5, Holm p 1) |
| EnC | 12.37, 144, 592 s | 12.36, 311, 1,394 s | tie (5/5, Holm p 1) |

With linear scaling, the gain of Phase 1v does not appear, and the programs are 1.3–2.2× larger.

## 1ar. New datasets at equal CPU (1,000 s)

Medians over splits 1–10, the splits that all three arms share:

| Dataset | GOMEA-LS | GOMEA-LS-TUNE | HC-LS | TUNE vs GOMEA-LS | Sizes (base, TUNE, HC) |
|---|---|---|---|---|---|
| Yac | 1.48 | 0.58 | 1.44 | −60.5 %, 10/0, **better** | 119, 89, 54 |
| Dow | 40.08 | 24.59 | 30.17 | −38.6 %, 8/2, tie (Holm p 0.082) | 190, 133, 61 |
| Tow | 18.19 | 12.30 | 18.64 | −32.4 %, 10/0, **better** | 186, 152, 50 |
| WiR | 65.24 | 66.23 | 64.91 | +1.5 %, 5/5, tie | 217, 202, 58 |
| WiW | 71.18 | 70.18 | 74.46 | −1.4 %, 8/2, tie | 286, 164, 50 |

The sizes of Dow and WiW for GOMEA-LS and HC-LS are medians over 20 splits.

**Primary (GOMEA-LS vs HC-LS, 5):** all ties.
* **Dow (20 splits):** 6/14, Holm p 0.28, medians 34.0 against 31.3.
* **WiW (20 splits):** 18/2, Holm p 0.000095, but the medians are 72.1 against 74.6 (−3.3 %), so
  the 10 % rule makes it a tie.
* **Yac, Tow, WiR:** ties at stage 1.

**Other points:**
* **Yac:** the earlier loss to HC (1ah: 8.0 against 4.5) does not appear with linear scaling (1.48
  against 1.44). With LM, GOMEA is lower than HC-LS in 10/10 splits.
* **WiR:** LM raised the gap from 1.2 to 9.7 with no gain in test NMSE. This is overfitting.
* **TUNE vs GOMEA-LS is a secondary contrast**, so it did not extend. TUNE has only 10 splits on
  all five datasets. TUNE vs HC-LS is descriptive.

## Deviations

1. **The choice script failed at first.** `set_const_opt.py` expected the runs of every
   `sr-ext-*` manifest, but `phase_run.sh` runs an extension only after an "extend" verdict. The
   first session thus skipped 1ao and 1ar, and ran 1ap and 1aq. The fix (`afa051b`) follows the
   rule of `phase_run.sh`. The restart (`RETRY_FAILED=1 ... --force`) ran only 1ao and 1ar.
2. **Order:** 1ao and 1ar ran after 1ap and 1aq, about three days later, not in the registered
   order. No phase uses the results of 1ap or 1aq.
3. **`duration_s`** of 1ao and 1ar in `round-summary.md` (74.7 h and 72.4 h) includes the idle
   time. The real durations are 4.2 h and 5.1 h.
4. **`const-opt-choice.json`** is not in the bundle. The choice and its medians are in
   `events.jsonl` and `round-summary.md`.

## Limits

* **Selection:** 1an chose LM on Con, EnH and Air, and 1ao tests it on the same datasets and
  splits 1–10. 1ar gives independent datasets, but only as a secondary contrast.
* **One pod and one budget:** 1,000 s CPU and 10,000 evaluations only.
* **Mutation** did not get an equal-CPU test, and LM was not tested with a size penalty.
