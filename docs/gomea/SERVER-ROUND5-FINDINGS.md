# Server round 5 findings: LM validation guard, weaker penalty, LM with a penalty (phases 1at to 1av)

**Status:** complete.

* **Run:** 2026-11-04 07:38 to 23:25 UTC, on the same pod as round 4. Round 4's orchestrator
  started this round 20 s after round 4 ended (the chain). W = 11 workers. S0 took 1.33 h, 1at
  4.78 h, 1au 3.26 h and 1av 6.42 h.
* **Runs:** 580 (180 in 1at, 160 in 1au, 240 in 1av), all `ok`, 0 failed. Total CPU 149.7 h.
* **Code:** `26aa93e`, with the same dirty configs of earlier rounds as round 4. No phase of this
  round uses them.
* **Protocol:** `SERVER-ROUND5-PREREGISTRATION.md`.
* **Results (outside git):** `../server-results-round5/gomea-results/`. The round log is
  `server-run5/round-summary.md`. Reports: `phase1at/report-stage1`, `phase1au/report-stage2`,
  `phase1av/report-stage2`.
* **How the extra numbers were computed:** as in round 4. Percentages are changes of the median.

**Answer in short:**

* **1at: LM-VAL qualifies.** The guard lowered the WiR gap from 9.69 to 4.04, and LM-VAL is not
  worse than LM on any dataset. All six accuracy contrasts are ties.
* **1au: PEN125 (R 0.125) qualifies.** It makes programs 58 % smaller on EnH and 48 % on Air, with
  test NMSE at most 8 % higher. It does not make programs smaller on Con.
* **1av: neither LM-PEN strength qualifies.** Both make LM's programs 16–27 % smaller with NMSE
  −2 % to +4 %, but the size contrast is "better" only on Air. LM-PEN5 on EnH missed by Holm p
  0.028 against 0.025.
* **The user's decisions (2026-11-06):** LM-VAL is the preferred LM variant and PEN125 the preferred
  strength. Neither is a default. LM-VAL with PEN125 and LM-VAL with no cap go to the backlog
  (`BACKLOG.md`, section 7).
* **All arms of this round used the cap of 250.** The user later chose no cap as the base, so
  these results need a check with no cap.

## S0. Environment check

* **pytest:** 2 failed, 8,944 passed, 13 skipped. The same two failures as in round 4
  (`SERVER-ROUND4-FINDINGS.md`, S0). Neither is from the code of this round.
* **Reproducibility:** 4 of 4 reference runs matched exactly. With the guard off, the new code
  reproduces the earlier rounds.

## 1at. LM with a validation guard (1,000 s, 10 splits)

LM-VAL is LM with `--const-opt-guard val`: the tuned constants replace the old ones only if the
NMSE on the validation split is not worse. Each check costs 2 evaluations.

| Dataset | GOMEA-LS NMSE | LM NMSE | LM-VAL NMSE | LM gap | LM-VAL gap | LM-VAL gap lower | Guard rejected (median) | LM size | LM-VAL size |
|---|---|---|---|---|---|---|---|---|---|
| WiR | 65.24 | 66.23 | 64.55 | 9.69 | **4.04** | 8/10 | 63 % | 202 | 159.5 |
| Con | 24.93 | 20.12 | 19.29 | 5.50 | 3.73 | 7/10 | 29 % | 140 | 131.5 |
| EnH | 7.70 | 6.54 | 6.03 | 0.89 | 0.75 | 5/10 | 4 % | 100.5 | 112 |
| Air | 37.76 | 34.47 | 33.41 | 0.35 | 0.31 | 3/10 | 15 % | 150 | 148.5 |
| Yac | 1.475 | 0.582 | 0.593 | 0.11 | 0.16 | 3/10 | 13 % | 89 | 81 |
| Tow | 18.19 | 12.30 | 11.33 | 0.80 | 0.80 | 3/10 | 0 % | 152 | 163 |

* **Primary (LM-VAL vs LM, test NMSE, 6):** all ties (Holm p ≥ 0.45). No contrast reached the
  extension range, so each dataset has 10 splits.
* **Secondary (LM-VAL vs GOMEA-LS, 6):** **better** on EnH, Air, Yac and Tow (9/1 to 10/0, Holm p
  0.012–0.018). Ties on WiR (8/2) and Con (8/2, Holm p 0.055).
* **Adoption rule:** LM-VAL qualifies. Its WiR gap is lower than LM's, and it is not worse on any
  dataset.
* **Guard telemetry:** over 60 LM-VAL runs, 2,925 LM calls, 2,611 checked and 596 rejected (23 %).
  Tuning used a median of 34–46 % of the evaluations per dataset, in both LM arms.
* **The effect is where LM overfitted:** on WiR and Con the gap fell. On the other four datasets
  the paired gap change is near zero. The WiR gap of LM-VAL (4.04) is still above GOMEA-LS's
  (1.22).

## 1au. A weaker size penalty (10,000 evaluations)

All penalty arms use `--gomea-deme-store accepted`. The penalty is `complexity / R`, so a larger R
is weaker.

| Dataset | Splits | Arm | Size | vs base | Test NMSE | vs base | CPU (s) |
|---|---|---|---|---|---|---|---|
| Con | 10 | GOMEA-LS | 160 | | 25.85 | | 920 |
| | | PEN25 (R 0.025) | 67.5 | −58 % | 29.48 | **+14.0 %** | 437 |
| | | PEN125 (R 0.125) | 155.5 | −3 % | 25.65 | −0.8 % | 711 |
| | | PEN625 (R 0.625) | 123.5 | −23 % | 28.16 | +8.9 % | 716 |
| EnH | 20 | GOMEA-LS | 228.5 | | 8.48 | | 938 |
| | | PEN25 | 86 | −62 % | 9.07 | +7.0 % | 383 |
| | | PEN125 | 96 | −58 % | 8.75 | +3.2 % | 508 |
| | | PEN625 | 147 | −36 % | 8.80 | +3.8 % | 671 |
| Air | 10 | GOMEA-LS | 289 | | 35.21 | | 1,200 |
| | | PEN25 | 102.5 | −65 % | 39.83 | **+13.1 %** | 534 |
| | | PEN125 | 149 | −48 % | 38.10 | +8.2 % | 772 |
| | | PEN625 | 260 | −10 % | 37.61 | +6.8 % | 1,123 |

* **Primary (size, 9):** PEN25 **better** on all three. PEN125 **better** on EnH and Air, tie on
  Con (5/5). PEN625 **better** on EnH only.
* **Secondary (test NMSE, 9):** all ties. PEN25 on EnH has Holm p 0.044 (4/16).
* **Adoption rule:** PEN125 qualifies: better on size on 2 of 3, NMSE at most +8.2 %. PEN25 fails
  the 10 % limit on Con and Air. PEN625 is better on size on only 1 of 3.
* **CPU:** PEN125 used 23–46 % less CPU than GOMEA-LS for the same evaluations.
* **Con is not stable:** PEN125 does not make programs smaller on Con, but the weaker PEN625
  gives a smaller median there.

## 1av. LM with a size penalty (1,000 s, 20 splits)

| Dataset | Arm | Size | vs LM | Test NMSE | vs LM | Gap |
|---|---|---|---|---|---|---|
| Con | GOMEA-LS | 185.5 | | 25.46 | | 2.85 |
| | LM | 123.5 | | 18.57 | | 3.08 |
| | LM-PEN5 (R 0.005) | 92 | −26 % | 18.97 | +2.1 % | 2.69 |
| | LM-PEN25 (R 0.025) | 104 | −16 % | 19.30 | +3.9 % | 2.93 |
| EnH | GOMEA-LS | 241.5 | | 8.26 | | 1.82 |
| | LM | 102 | | 6.09 | | 0.89 |
| | LM-PEN5 | 78.5 | −23 % | 5.99 | −1.7 % | 0.52 |
| | LM-PEN25 | 77.5 | −24 % | 6.33 | +3.9 % | 1.18 |
| Air | GOMEA-LS | 261.5 | | 37.87 | | −0.53 |
| | LM | 148.5 | | 33.42 | | 0.63 |
| | LM-PEN5 | 111.5 | −25 % | 34.65 | +3.7 % | 1.49 |
| | LM-PEN25 | 108 | −27 % | 33.37 | −0.1 % | 1.88 |

* **Primary (size against LM, 6):** **better** on Air for both strengths (19/1 and 18/2, Holm p <
  0.001). Ties on Con (Holm p 0.071 and 0.125) and EnH (0.028 and 0.071).
* **Secondary (test NMSE against LM, 6):** all ties (Holm p ≥ 0.55).
* **Adoption rule:** neither strength qualifies. Each is better on size on 1 of 3 datasets.
* **LM against GOMEA-LS (descriptive):** LM has lower NMSE on all three datasets with smaller
  programs, as in round 3. The splits are the same as in 1ao, so this is not independent evidence.
* **PEN125 was not in this phase.** The design was fixed before 1au chose PEN125.

## Deviations

1. **The dirty tree and the second pytest failure** are from earlier rounds (see round 4).
2. **1au extended only EnH.** Con and Air have 10 splits. This follows the registered rule.

## Limits

* **The cap of 250:** all arms used it. With no cap, a run builds 5–7 demes on regression instead
  of 35–45, so it gets far fewer LM calls and penalty decisions. The results can change.
* **10 splits in 1at**, so a small accuracy effect of the guard cannot show.
* **Different budgets:** 1au used equal evaluations and 1av equal CPU. Do not compare PEN125 with
  LM-PEN5 across these phases.
* **The guard uses the validation split.** WiR has duplicate rows (1,359 unique of 1,599), so a
  validation row can equal a training row. On split 30, 62 unique rows are in both.
