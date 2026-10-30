# Server round 1 findings: phases 1ae, 1af, 1ag, 1ah, 1ai and 1aj

**Status:** complete.

* **Run:** 2026-10-26 13:49 to 2026-10-27 07:49 UTC (18.0 h wall), on the Debian/Ubuntu
  Jupyter pod `jupyter-yderese` (AMD Threadripper 1920X, cgroup quota 12 CPUs, 64 GB).
  W = 11 workers (limited by CPU). One orchestrator session, no restarts.
* **Runs:** 704 (700 phase runs and 4 pilot runs), all `ok`, 0 failed. Total CPU 147.6 h
  (the sum of `cpu_s` over all run summaries). `SERVER-RUN.md` estimated about 300 CPU-hours.
* **Code:** git HEAD `65bd8e5` (the merge of `514f602`, `12718ec` and the tools), clean
  tree. Every arm passed `--knob-probe-baseline reduced`.
* **Protocol:** `SERVER-ROUND-PREREGISTRATION.md`. The phase order was as registered: S0,
  1ah, 1ai, 1af-pilot, 1af, 1aj, 1ag, 1ae.
* **Results (outside git):** `../server-results/gomea-results/`. The round log is
  `server-run/round-summary.md` and `server-run/events.jsonl`. Each phase has
  `phase1a*/report-stage*/{aggregate,contrasts}.csv` and `verdict.json`. Boolean runs are in
  `1/phase1a*-bool-*/runs/`.
* **How the extra numbers were computed:** train NMSE, the train–test gap and the deme
  telemetry are not in `aggregate.csv`. A read-only script computed them from each run's
  `summary.json` and `events.jsonl`. "Gap" is the median over runs of (test NMSE − train
  NMSE), not the difference of the two medians. All other numbers are from `aggregate.csv`
  and `contrasts.csv`.

**Answer in short:**

* **Regression:** GOMEA beats capped MOSES hill climbing at equal time on 3 of 5 new
  datasets (Air, EnC, EnH) and ties on Bos and Yac. Its programs are about 3× larger than
  capped HC's.
* **Against the authors' GP-GOMEA at equal evaluations:** plain GOMEA ties on Air and Con
  and is worse on EnC and EnH (0/10). With linear scaling, GOMEA ties GP-GOMEA on all four.
  GP-GOMEA's programs are 6–17× smaller than plain GOMEA's, and it uses under 1 s of CPU
  per run.
* **Linear scaling** in the fitness helps HC and GOMEA on Con (both "better").
* **Long demes, seeding at equal time and larger caps** do not help regression. Cap 1,000
  helps GOMEA on mux11 only.
* **Telemetry:** in the base setting, 95–99.5 % of GOMEA demes end inside generation 0. Thus
  the learned linkage tree is used only once per deme, from an unselected population.

## S0. Environment check

* **pytest:** 1 failed, 8,676 passed, 13 skipped. The failure is
  `tests/fixtures/test_fixture_manifests.py::test_toolchain_drift_is_clean_on_the_real_repo`.
  It compares the C++ toolchain that made the golden fixtures (g++ 16.2.1, Boost 1.92) with
  the pod's toolchain (g++ 13.3.0). It is environmental and does not touch the code under
  test.
* **Reproducibility:** 4 of 4 runs matched the local results exactly (1ad EnH GOMEA split 1,
  1ad EnC GOMEA-UNI split 1, 1ad Con GOMEA-UNI split 1, 1ae mux11 GOMEA seed 3001). Thus
  the pod's floating point agrees with the local machine for these runs. The Con GOMEA arm
  also gives 37.1 on splits 1–10 in 1ae, 1af, 1ag and 1ai, the same as 1v locally.

## 1ah. GOMEA vs MOSES hill climbing at equal time (1,000 s), beyond Con

**Verdict (primary, splits 1–20):** GOMEA **better** than HC-C250 on Air, EnC and EnH;
**tie** on Bos and Yac. All five problems were extended at stage 1 (Holm p 0.049–0.111).

| Problem | Arm | Test NMSE | Train NMSE | Gap | Size (nodes) | CPU per run | Evaluations used | CPU per evaluation |
|---|---|---|---|---|---|---|---|---|
| Air | GOMEA | **46.1** | 45.5 | 1.5 | 361.5 | 1,004 s | 7,268.5 | 0.138 s |
| Air | HC-C250 | 51.8 | 48.0 | 2.3 | 113 | 1,007 s | 25,250 | 0.040 s |
| Air | HC (10 splits) | 59.3 | 58.9 | 1.7 | 196.5 | 1,031 s | 21,017 | 0.050 s |
| Bos | GOMEA | 44.3 | 31.5 | 8.9 | 374 | 1,004 s | 7,990 | 0.126 s |
| Bos | HC-C250 | 73.4 | 58.4 | 12.5 | 96 | 1,009 s | 26,625 | 0.038 s |
| Bos | HC (10 splits) | 73.3 | 55.4 | 21.4 | 177 | 1,032 s | 20,117 | 0.051 s |
| EnC | GOMEA | **19.3** | 17.7 | 2.1 | 352.5 | 1,005 s | 8,511.5 | 0.118 s |
| EnC | HC-C250 | 24.6 | 23.4 | 1.2 | 115.5 | 1,009 s | 28,375 | 0.036 s |
| EnC | HC (10 splits) | 23.3 | 21.4 | 1.5 | 160.5 | 1,020 s | 22,514.5 | 0.045 s |
| EnH | GOMEA | **16.2** | 14.5 | 1.3 | 381.5 | 1,005 s | 7,876 | 0.128 s |
| EnH | HC-C250 | 22.8 | 21.3 | 2.2 | 121 | 1,008 s | 26,375 | 0.038 s |
| EnH | HC (10 splits) | 19.8 | 19.0 | 1.8 | 174 | 1,041 s | 21,158 | 0.049 s |
| Yac | GOMEA | 8.0 | 4.1 | 1.3 | 281 | 1,005 s | 13,588.5 | 0.074 s |
| Yac | HC-C250 | 4.5 | 5.0 | 0.4 | 93 | 1,008 s | 33,250 | 0.030 s |
| Yac | HC (10 splits) | 4.4 | 3.8 | 0.4 | 152.5 | 1,029 s | 22,717.5 | 0.045 s |

| Contrast | GOMEA lower / higher | p | p (Holm) | Verdict |
|---|---|---|---|---|
| Air: GOMEA vs HC-C250 | 16 / 4 | 0.0003 | 0.0016 | **A better** |
| Bos: GOMEA vs HC-C250 | 15 / 5 | 0.048 | 0.097 | tie |
| EnC: GOMEA vs HC-C250 | 15 / 5 | 0.0042 | 0.017 | **A better** |
| EnH: GOMEA vs HC-C250 | 14 / 6 | 0.0073 | 0.022 | **A better** |
| Yac: GOMEA vs HC-C250 | 8 / 12 | 0.064 | 0.097 | tie |
| *Secondary, splits 1–10:* Air: GOMEA vs HC | 8 / 2 | 0.014 | 0.041 | tie |
| Bos: GOMEA vs HC | 10 / 0 | 0.0020 | 0.0098 | **A better** |
| EnC: GOMEA vs HC | 8 / 2 | 0.105 | 0.105 | tie |
| EnH: GOMEA vs HC | 9 / 1 | 0.027 | 0.055 | tie |
| Yac: GOMEA vs HC | 1 / 9 | 0.0098 | 0.039 | tie |

**Measured costs (computed from the medians above):**

* GOMEA's programs are 3.0–3.9× the size of HC-C250's and 1.8–2.2× the size of HC's.
* GOMEA's CPU per evaluation is 2.4–3.5× HC-C250's and 1.6–2.8× HC's. Yac has the lowest
  ratio on both.
* In the same time GOMEA does 29–41 % of HC-C250's evaluations.
* The gap is not larger for GOMEA, except on Yac (1.3 against 0.4). On Bos, GOMEA's gap
  (8.9) is smaller than HC-C250's (12.5) and HC's (21.4).

**Interpretation:** the equal-time regression win of 1p (Con) holds on Air, EnC and EnH.
Bos leans to GOMEA (15/5 against HC-C250, 10/0 against HC). Yac leans the other way:
HC-C250 and HC are lower on 12/20 and 9/10 splits, and GOMEA's median error is about 1.8×
theirs, with a larger gap (1.3 against 0.4).

## 1ai. Integrated GOMEA vs stand-alone GP-GOMEA at equal evaluations

**Verdict (primary, splits 1–10):** GOMEA vs GPGOMEA-P16 is a **tie** on Air and Con and
**worse** on EnC and EnH (0/10 each). All secondary contrasts of GOMEA-LS against
GPGOMEA-P16 are **ties**.

| Problem | Arm | Test NMSE | Train NMSE | Gap | Size | CPU per run | Evaluations used |
|---|---|---|---|---|---|---|---|
| Air | GOMEA | 42.5 | 41.2 | 0.3 | 514.5 | 1,942 s | 10,000 |
| Air | GOMEA-LS | 35.2 | 38.1 | −1.6 | 289 | 1,197 s | 10,000 |
| Air | GPGOMEA-P16 | 41.0 | 42.5 | −0.9 | 29 | 0.40 s | 11,273.5 |
| Air | GPGOMEA (P500) | 44.1 | 44.4 | 0.0 | 24 | 0.61 s | 19,524 |
| Con | GOMEA | 37.1 | 33.9 | 4.0 | 187 | 859 s | 10,000 |
| Con | GOMEA-LS | 25.9 | 23.1 | 3.2 | 160 | 924 s | 10,000 |
| Con | GPGOMEA-P16 | 33.6 | 29.9 | 4.7 | 29 | 0.33 s | 12,031.5 |
| Con | GPGOMEA (P500) | 37.3 | 34.0 | 5.0 | 28 | 0.44 s | 18,007 |
| EnC | GOMEA | 20.4 | 17.1 | 2.4 | 522.5 | 1,403 s | 10,000 |
| EnC | GOMEA-LS | 12.4 | 10.5 | 2.1 | 143.5 | 592 s | 10,000 |
| EnC | GPGOMEA-P16 | 12.2 | 10.4 | 1.5 | 30 | 0.27 s | 11,035 |
| EnC | GPGOMEA (P500) | 12.4 | 10.8 | 1.3 | 25 | 0.39 s | 17,614.5 |
| EnH | GOMEA | 15.7 | 15.1 | 0.9 | 366.5 | 1,318 s | 10,000 |
| EnH | GOMEA-LS | 7.5 | 7.0 | 0.9 | 275.5 | 1,078 s | 10,000 |
| EnH | GPGOMEA-P16 | 9.0 | 7.3 | 1.7 | 29 | 0.28 s | 11,705.5 |
| EnH | GPGOMEA (P500) | 8.5 | 7.5 | 1.1 | 29 | 0.41 s | 17,658.5 |

| Contrast | A lower / higher | p | p (Holm) | Verdict |
|---|---|---|---|---|
| Air: GOMEA vs GPGOMEA-P16 | 6 / 4 | 0.70 | 0.70 | tie |
| Con: GOMEA vs GPGOMEA-P16 | 4 / 6 | 0.32 | 0.64 | tie |
| EnC: GOMEA vs GPGOMEA-P16 | 0 / 10 | 0.0020 | 0.0078 | **A worse** |
| EnH: GOMEA vs GPGOMEA-P16 | 0 / 10 | 0.0020 | 0.0078 | **A worse** |
| *Secondary:* GOMEA vs GPGOMEA (P500): Air / Con / EnC / EnH | 7/3, 4/6, 0/10, 1/9 | | 0.48, 1.0, 0.016, 0.027 | tie, tie, **A worse**, tie |
| GOMEA-LS vs GPGOMEA-P16: Air / Con / EnC / EnH | 9/1, 8/2, 5/5, 8/2 | | 0.027, 0.098, 1.0, 0.11 | tie on all four |

**Caveats (as registered):**

* GP-GOMEA overshoots its budget: P16 used 11,035–12,031.5 evaluations (median per dataset)
  and P500 used 17,614.5–19,524. The overshoot favours GP-GOMEA.
* CPU is not comparable across C++ and Python. It is descriptive only.
* GP-GOMEA's size is its own node count without the scaling terms. pyMoses counts atoms
  and parentheses. Thus the size ratio is approximate.

**Interpretation:**

* Per evaluation, plain GOMEA (no linear scaling) is weaker than GP-GOMEA on the two
  Energy datasets. The difference is the fitness: GP-GOMEA always uses linear scaling.
* With the same scaling (GOMEA-LS), GOMEA has a lower median than GPGOMEA-P16 on Air, Con
  and EnH and the same on EnC, but no contrast is significant. The integration matches
  GP-GOMEA on accuracy per evaluation, with programs 5–10× larger (GOMEA-LS 143.5–289 nodes
  against 29–30) and CPU about 2,200–3,900× higher (592–1,197 s against 0.27–0.40 s; ratio per dataset).

## 1af. Long demes for promising exemplars (and the LONG-RAND pilot)

**Pilot (2 LONG-TOP runs per problem, not data):** the median share of long demes set
LONG-RAND's p. mux11: p = 0.771 (shares 0.54 and 1.00). Con: p = 0.265 (shares 0.31 and
0.22). Log: `phase1af-pilot/longrand-p.json`.

**Verdict (stage 1, seeds 3001–3010 / splits 1–10):** all six contrasts are **ties**
(Holm p 0.77–1.0). Nothing was extended.

| Problem | Arm | Metric | Solved | Evaluations to solve | Size | CPU per run | Demes | Long demes per run | Long demes with ≥ 2 generations |
|---|---|---|---|---|---|---|---|---|---|
| mux11 | GOMEA | 536 s CPU to solve | 10/10 | 5,538 | 36 | 538 s | 22.5 | – | – |
| mux11 | LONG-TOP | 384 s | 9/10 | 13,690 | 34 | 387 s | 8 | 8 | 15 % |
| mux11 | LONG-RAND | 381 s | 10/10 | 11,914 | 35 | 388 s | 9 | 7 | 31 % |
| mux11 | LONG-TOP-RT | 323 s | 10/10 | 15,083.5 | 30 | 325 s | 8.5 | 8 | 26 % |

| Problem | Arm | Test NMSE | Train NMSE | Gap | Size | CPU per run | Long demes per run | Long demes with ≥ 2 generations |
|---|---|---|---|---|---|---|---|---|
| Con | GOMEA | 37.1 | 33.9 | 4.0 | 187 | 860 s | – | – |
| Con | LONG-TOP | 35.8 | 34.2 | 4.4 | 159.5 | 592 s | 5 | 80 % |
| Con | LONG-RAND | 40.4 | 37.2 | 1.8 | 123.5 | 778 s | 5 | 60 % |
| Con | LONG-TOP-RT | 38.0 | 35.1 | 4.3 | 105.5 | 435 s | 5 | 100 % |

The last two columns are medians over runs, from the `long` flag of each deme's telemetry.

| Contrast | A lower / higher | p (Holm) | Verdict |
|---|---|---|---|
| mux11: LONG-TOP vs GOMEA | 6 / 4 | 1.0 | tie |
| Con: LONG-TOP vs GOMEA | 5 / 5 | 1.0 | tie |
| mux11: LONG-TOP vs LONG-RAND / vs LONG-TOP-RT | 6/4, 4/6 | 0.86, 0.77 | tie, tie |
| Con: LONG-TOP vs LONG-RAND / vs LONG-TOP-RT | 6/4, 6/4 | 0.77, 0.86 | tie, tie |

**Interpretation:**

* Long demes do not change accuracy or CPU to solve in a way the test can detect.
* On mux11 the long arms need 2.2–2.7× GOMEA's evaluations to solve but each evaluation is
  cheaper (fewer demes, so less knob building). The two effects cancel.
* On mux11 only 15–31 % of long demes complete a second generation; most stop early.
* Where demes do relearn linkage (Con, 80–100 %), the learned tree (LONG-TOP) does not beat
  a random tree (LONG-TOP-RT).

## 1aj. Seeded GOMEA on mux11 at equal time (3,600 s)

**Verdict:** GOMEA-SEED vs HC-C250 is a **tie** (stage 2, 20 seeds, 7 / 13, Holm p 0.28).
GOMEA-SEED vs HC (secondary, 10 seeds) is **worse** (0 / 10, Holm p 0.0020).

| Arm | CPU to solve | Solved | Evaluations to solve | CPU per evaluation | Demes | Size |
|---|---|---|---|---|---|---|
| GOMEA-SEED (20 seeds) | 351 s | 20/20 | 13,251 | 0.025 s | 19 | 32 |
| HC-C250 (20 seeds) | 293 s | 20/20 | 7,865.5 | 0.037 s | 32 | 26 |
| HC (10 seeds) | 77 s | 10/10 | 7,953 | 0.011 s | 5 | 26.5 |

* GOMEA-SEED's CPU to solve includes the seeding step's GP-GOMEA CPU, as registered. The
  seeding step used a median of 8,748 evaluations (budget 8,000; GP-GOMEA overshoots).
* `aggregate.csv` reports `final_from_seed` = 20 for GOMEA-SEED: in all 20 runs the final
  program descends from a seed.

**Interpretation:** seeding brings GOMEA level with capped HC on mux11 (statistically), but
plain MOSES hill climbing is still about 4.6× faster (351 s against 77 s, medians). Seeding
does not close the Boolean time gap.

## 1ag. Linear scaling in the regression fitness (Con)

**Verdict (splits 1–10):** HC-LS vs HC **better** (10 / 0, Holm p 0.0039). GOMEA-LS vs GOMEA
**better** (8 / 2, Holm p 0.020). As registered, HC and GOMEA were rerun on the server.

| Arm | Test NMSE | Train NMSE | Gap | Size | CPU per run | CPU per evaluation |
|---|---|---|---|---|---|---|
| HC (capped 250) | 60.4 | 60.0 | 3.1 | 51.5 | 185 s | 0.019 s |
| HC-LS | **38.6** | 35.4 | 2.0 | 31.5 | 114 s | 0.011 s |
| GOMEA | 37.1 | 33.9 | 4.0 | 187 | 862 s | 0.086 s |
| GOMEA-LS | **25.9** | 23.1 | 3.2 | 160 | 916 s | 0.092 s |

**Interpretation:**

* Linear scaling lowers test NMSE by 36 % for HC and 30 % for GOMEA (computed from the
  medians), with smaller programs and a smaller gap for both.
* GOMEA-LS has the lowest error of all arms (25.9). But at equal evaluations it uses about
  8× HC-LS's CPU (916 s against 114 s). This round did not test GOMEA-LS against HC-LS at
  equal time (an open item).

## 1ae. Per-deme caps 250, 500 and 1,000

**Verdict (stage 2, 20 seeds or splits):**

* mux11: cap 1,000 **better** than 250 (19 / 1, Holm p 0.0002). Cap 500 is a **tie**
  (13 / 7, Holm p 0.033).
* Con: caps 500 and 1,000 are both **worse** than 250 (5 / 15 each, Holm p 0.0063 and
  0.0018).

| Problem | Arm | Metric | Evaluations to solve | CPU per evaluation | Demes | Size | Demes with ≥ 2 generations |
|---|---|---|---|---|---|---|---|
| mux11 | GOMEA (cap 250) | 627 s CPU to solve | 5,165.5 | 0.116 s | 21 | 37 | 0.7 % |
| mux11 | GOMEA-C500 | 384 s | 6,776 | 0.045 s | 14 | 34 | 1.9 % |
| mux11 | GOMEA-C1000 | **278 s** | 9,058.5 | 0.027 s | 10 | 31.5 | 12.6 % |

All mux11 arms solved 20/20.

| Problem | Arm | Test NMSE | Train NMSE | Gap | Size | CPU per run | Demes with ≥ 2 generations |
|---|---|---|---|---|---|---|---|
| Con | GOMEA (cap 250) | **35.2** | 32.4 | 2.3 | 229.5 | 927 s | 2.4 % |
| Con | GOMEA-C500 | 43.4 | 37.6 | 5.7 | 199 | 987 s | 9.3 % |
| Con | GOMEA-C1000 | 44.9 | 40.8 | 3.6 | 146 | 654 s | 44.7 % |

"Demes with ≥ 2 generations" is pooled over all demes of all runs of the arm.

**Interpretation:**

* This replicates the local 1ae stage 1 on mux11 (cap 1,000 better, 10 / 0 at stage 1 here
  too). The gain comes from cheaper evaluations: fewer demes, so less knob building. Cap
  1,000 needs 1.75× the evaluations of cap 250 to solve.
* On Con, longer demes give smaller programs but higher error. Cap 250 stays the
  regression default.

## Telemetry: how many generations a deme runs

The table uses the GOMEA arms with the base cap of 250 and N = 16. Values are pooled over
all demes of all runs of the arm. "Generation-0 FOS size" is the median number of FOS
subsets in the first generation.

| Phase | Problem | Arm | Demes | Demes that end inside generation 0 | Generation-0 FOS size |
|---|---|---|---|---|---|
| 1ah | Air / Bos / EnC / EnH / Yac | GOMEA | 650 / 689 / 769 / 678 / 1,203 | 94.8 / 98.7 / 96.7 / 96.8 / 97.1 % | 414 / 1,121 / 675 / 814 / 414 |
| 1ai | Air / Con / EnC / EnH | GOMEA | 402 / 402 / 405 / 402 | 96.0 / 97.5 / 96.5 / 97.3 % | 638 / 680 / 1,054 / 977 |
| 1ai | Air / Con / EnC / EnH | GOMEA-LS | 404 / 405 / 400 / 402 | 96.3 / 96.3 / 98.3 / 98.3 % | 370 / 786 / 467 / 979 |
| 1ae | Con / mux11 | GOMEA | 805 / 436 | 97.6 / 99.3 % | 796 / 2,308 |
| 1aj | mux11 | GOMEA-SEED | 370 | 99.5 % | 2,034 |

"Ends inside generation 0" means the deme's `end` event has `generations` = 1: the budget
of 250 evaluations ran out in the first generation.

**Interpretation:** a deme with N = 16 and a FOS of 400–2,300 subsets cannot finish one
generation in 250 evaluations. Thus the linkage tree is learned once per deme, from the
unselected initial population, and GOM never uses a tree learned from a selected
population. Round 2 (`SERVER-ROUND2-FINDINGS.md`) tested demes that do run several
generations.

## What this round shows

1. **Regression, equal time:** GOMEA beats capped HC on Air, EnC and EnH, and Con from 1p.
   It ties on Bos (leaning GOMEA) and Yac (leaning HC).
2. **Costs (D1):** GOMEA's programs are 1.8–3.9× larger than HC's and its CPU per
   evaluation is 1.6–3.5× higher (1ah). The gap is not worse, except on Yac.
3. **Against GP-GOMEA:** with linear scaling, GOMEA matches the authors' GP-GOMEA per
   evaluation on all four datasets. Without it, GOMEA loses on EnC and EnH. GP-GOMEA's
   programs are much smaller and its CPU is about three orders of magnitude lower.
4. **Linear scaling helps** both HC and GOMEA on Con (1ag), and makes programs smaller.
5. **Boolean:** seeding (1aj) does not close the gap to plain HC. Cap 1,000 (1ae) makes
   GOMEA faster on mux11, but plain HC in 1aj (77 s) is still faster than any GOMEA arm in
   this round (278 s for cap 1,000 in 1ae; measured in different phases).
6. **No regression gain** from long demes (1af), larger caps (1ae) or relearned linkage
   (LONG-TOP vs LONG-TOP-RT).

## Recommendation given to the user (after this round)

* **Regression:** offer GOMEA as an optional inner optimizer, with linear scaling on by
  default.
* **Boolean:** keep HC as the default.
* **For regression, do not use** long demes, seeding or larger caps.
* **Costs to state:** programs 2–4× larger and CPU per evaluation about 3–4× (measured in
  1ah: 1.8–3.9× and 1.6–3.5×, depending on the dataset and the HC variant).
* **Open items:** GOMEA-LS against HC-LS at equal time; Yac; size control.
* **Reading:** the evidence supports optimal mixing inside MOSES, not learned linkage.

## Deviations and limits

* **Code commit:** the pre-registration names `514f602` plus `12718ec` plus tools. The pod
  ran their merge, `65bd8e5`, with a clean tree.
* **1ae is a full rerun,** as registered; the local stage 1 stays a separate result.
* **Secondary contrasts never extend,** so HC in 1ah and HC in 1aj have 10 splits or seeds.
* **1ai budget overshoot** for GP-GOMEA (see the caveats above).
* **CPU on a shared pod:** runs did not share cores (W = 11 on 12 CPUs), but CPU times can
  still vary between identical runs (1t showed up to 1.6× locally).
* **Cross-phase comparisons** in "What this round shows", item 5, are descriptive only.
