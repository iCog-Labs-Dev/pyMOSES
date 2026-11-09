# Server round 6 findings: dynamic population and knob caps again (phases 1ay and 1az)

**Status:** complete.

* **Run:** 2026-11-06 10:35 to 2026-11-07 17:18 UTC. W = 11 workers. S0 took 1.33 h, 1ay 16.07 h
  and 1az 13.31 h. The chain then started round 7.
* **Runs:** 1ay 400 (381 ok, 19 not ok), 1az 300 (299 ok, 1 not ok).
* **Code:** `e6c0038`, clean tree (the dirty configs of earlier rounds were saved to a patch and
  restored before the round).
* **Protocol:** `SERVER-ROUND6-PREREGISTRATION.md`.
* **Results (outside git):** `../server-results-round6/gomea-results/`. The round log is
  `server-run6/round-summary.md`. Reports: `phase1ay/report-stage2`, `phase1az/report-stage2`. The
  choice is in `server-run6/phase1az-config/population-choice.json`.

**Answer in short:**

* **1ay: no dynamic fraction qualifies.** All three are worse than N = 16 on mux11 (10 of 10 seeds)
  and on Con (20 splits), and tie on EnH. The rule chose POP = N16 for rounds 7 and 8, and DYN =
  F 0.05 (best median rank) for 1az and the control arms.
* **The dynamic programs are much smaller:** median size 50–88 against 150–153 at N = 16 on Con
  and EnH.
* **LT-B against RT:** all 12 secondary contrasts tie, at every population setting.
* **1az: no knob cap qualifies.** With the dynamic population, all four caps are worse on mux11
  (20 of 20 seeds) and tie on Con, EnH and Air. The same result as round 4 (1as).
* **The user's decision (2026-11-09):** the population stays dynamic, against the pick rule. The
  user will test it on other datasets (`APPROVALS.md`).

## S0. Environment check

* **pytest:** 1 failed, 9,182 passed, 13 skipped. The failure is
  `test_toolchain_drift_is_clean_on_the_real_repo`, which compares the pod's compiler set-up with
  the fixtures (also in round 4). `test_plan3` passes now, because the configs were restored.
* **Reproducibility:** 4 of 4 reference runs matched exactly. With `--gomea-pop-size 16`, the new
  code reproduces the earlier rounds.

## 1ay. Population size and linkage

Median metric: CPU seconds to solve on mux11 (inf if not solved), test NMSE on Con and EnH.

| Problem | N16-LTB | DYN005-LTB | DYN025-LTB | DYN2-LTB |
|---|---|---|---|---|
| mux11, CPU to solve | 203 | 621 | 1,850 | inf (9 of 10 not ok) |
| mux11, solved | 10/10 | 10/10 | 9/10 | 1/10 |
| Con, test NMSE | 26.3 | 30.7 | 31.2 | 33.4 |
| EnH, test NMSE | 8.42 | 8.31 | 8.24 | 8.71 |
| Con size | 153 | 88.5 | 84.5 | 64.5 |
| EnH size | 150 | 72 | 71 | 50.5 |
| Con / EnH median N | 16 | 100 / 100 | 100 / 100 | 392 / 310 |
| Con / EnH demes per run | 6 / 6 | 2.5 / 3 | 2 / 3 | 2 / 2 |

**Registered verdicts (stage 2 for the extended contrasts):**

| Problem | DYN2 vs N16 | DYN025 vs N16 | DYN005 vs N16 |
|---|---|---|---|
| mux11 | worse (0/10, Holm p 0.012) | worse (0/10) | worse (0/10) |
| Con | worse (3/17, Holm p 0.0006) | worse (2/18) | worse (4/16) |
| EnH | tie (5/15, Holm p 0.098) | tie (5/15) | tie (8/12) |

* **Pick:** no fraction is better on 2 of 3 problems, so POP = N16. Median ranks: DYN005 1,
  DYN025 2, DYN2 3, so DYN = F 0.05. The control population of rounds 7 and 8 = F 0.05.
* **LT-B against RT (secondary):** 12 ties. On mux11 at N = 16, LT-B solved in 203 s against 371 s
  (9 of 10 seeds lower), Holm p 0.047. This closes backlog item 7.3: LT-B and RT do not differ at
  our scale, also with no cap.
* **Train–test gap:** about the same in all arms (Con 2.0–3.6, EnH 1.4–1.7).

**Interpretation.** With a time limit of 1,000 s, a population of 100 or more lets a run finish
only 2–3 demes, against about 6 at N = 16. So fewer exemplars get a deme. On mux11 this costs
much CPU to solve. On regression the programs are smaller, but on Con also less accurate.

## 1az. Knob caps with the dynamic population (F = 0.05)

| Problem | NOCAP-DYN | K8-DYN | K16-DYN | K32-DYN | K64-DYN | NOCAP-N16 |
|---|---|---|---|---|---|---|
| mux11, CPU to solve | 724 | inf (10/20 solved) | inf (8/20) | 2,941 (12/20) | 2,627 (16/20) | 228 |
| Con, test NMSE | 30.4 | 31.8 | 33.8 | 30.8 | 29.7 | 27.5 |
| EnH, test NMSE | 8.31 | 8.44 | 8.74 | 8.44 | 8.66 | 8.34 |
| Air, test NMSE | 41.6 | 41.3 | 43.5 | 40.9 | 40.7 | 41.3 |
| Size, Con / EnH / Air | 86.5 / 67 / 94 | 58 / 70.5 / 100.5 | 88.5 / 72.5 / 87.5 | 95 / 67 / 88.5 | 65.5 / 77.5 / 85 | 153 / 151.5 / 212.5 |

* **Primary:** all 4 caps are worse on mux11 (stage 2, 0–1 of 20 seeds lower, Holm p < 1e-4). All
  12 regression contrasts tie. No cap qualifies.
* **Secondary, NOCAP-DYN vs NOCAP-N16:** worse on mux11 (0/20), tie on Con, EnH and Air.

## Deviations

* **The DYN2 runs on mux11 hit the hard stop.** 18 of 20 DYN2 runs (LT-B and RT) and 1 DYN025-LTB
  run ended as `timeout` at the wall cap of 5,400 s. MOSES checks the 3,600 s limit only between
  steps, and one generation with N ≈ 1,400–3,450 is longer than the time left. These runs count as
  not solved, as the protocol says, but their size and evaluation counts are lost.
* **The dynamic arms used more CPU than the limit on regression.** Median CPU was 1,130–1,810 s
  against 1,010–1,170 s at N = 16, for the same reason. So the dynamic arms had more CPU, not
  less, and were still worse or equal.
* **1 run of K32-DYN on mux11** was not ok.

## Limits

* Only one budget (1,000 s regression, 3,600 s mux11) and 3 problems in 1ay. A larger budget gives
  a large population more demes. The user plans tests on other datasets.
* The fractions 2.0, 0.25 and 0.05 were agent-written. On Con, EnH and the first mux11 deme, 0.25
  and 0.05 both give the classic minimum of 100, so those two arms are often the same.
