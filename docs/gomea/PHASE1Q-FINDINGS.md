# Phase 1q findings: fixing knob collapse inside GOMEA

**Status:** complete. Stage 1 ran on 2026-10-18 from 13:45 to 20:48, including the reruns.
No contrast was borderline, so no extension ran. Protocol: `PHASE1Q-PREREGISTRATION.md`,
including its deviation section. Code `a1c2ce7`. Results: `phase1q-results/` and
`../gomea-results/1/phase1q-stage1*/`.

**Answer:** the knob-collapse fix (`--knob-probe-baseline reduced`, arm `KR`) does what it was
built for, but changes nothing that matters.

* **Collapse:** it removed every collapsed deme, from 35 to 0 on mux11 and from 367 to 0 on
  pa5.
* **Accuracy and CPU:** both tied current GOMEA on both problems, with Holm p 1.0 in all four
  primary contrasts.
* **Recommendation:** keep it as an opt-in option. It is not worth making the default.

## Setup

* **Arms:** `LTB-C250-M` (current GOMEA), `KR` (+ `--knob-probe-baseline reduced`) and
  `HC-C250` (capped hill climbing).
* **Budget:** all arms `-m 40000` with a cap of 250 evaluations per deme.
* **Problems:** mux11 and pa5, seeds 3001–3010.
* **Load:** 18 workers. The machine was shared with other jobs for much of stage 1, at load
  up to about 42 on 22 logical cores.

## Deviation: capped runs were rerun

Six mux11 runs hit the 7,200 s wall-clock safety cap: LTB-C250-M and KR, seeds 3003–3005
each. They were rerun with a 36,000 s cap, as the pre-registration's deviation section
describes, and all six then ended `ok`.

* **Determinism check:** the three LTB-C250-M reruns needed exactly the evaluations Phase 1o
  recorded for the same seeds (6,941, 13,734 and 12,310). So the reruns are faithful
  replacements, and only their CPU reflects the load.
* **No run was censored** in the analysis.

## Results (medians over 10 seeds)

| Problem (accuracy metric) | LTB-C250-M | KR | HC-C250 |
|---|---|---|---|
| mux11: evaluations to solve (all 30 runs solved) | **5,345** | 5,538 | 7,306 |
| mux11: CPU per run (s) | 3,197 | 3,006 | 1,769 |
| mux11: search fields per deme | 1,006 | 1,126 | — |
| mux11: collapsed demes | 35 of 278 | **0** of 237 | — |
| pa5: final error (wrong rows; none solved) | 8 | 7.5 | **6.5** |
| pa5: CPU per run (s) | 2,642 | 2,656 | 3,244 |
| pa5: search fields per deme | 112.5 | 172 | — |
| pa5: collapsed demes | 367 of 1,638 | **0** of 1,618 | — |

A deme counts as collapsed if it has fewer than 30 % of the most search fields any earlier
deme of the same run had.

**Primary contrasts** (KR vs LTB-C250-M; Holm over 4):

| Problem | Contrast | KR lower / tie / higher | Holm p | Verdict |
|---|---|---|---|---|
| mux11 | CPU | 6 / 0 / 4 | 1.0 | tie |
| mux11 | accuracy | 5 / 0 / 5 | 1.0 | tie |
| pa5 | CPU | 5 / 0 / 5 | 1.0 | tie |
| pa5 | accuracy | 4 / 3 / 3 | 1.0 | tie |

**Secondary contrasts** (KR vs HC-C250, accuracy; Holm over 2):

| Problem | KR lower / tie / HC lower | Holm p | Verdict |
|---|---|---|---|
| mux11 | 7 / 0 / 3 | 0.26 | tie |
| pa5 | 2 / 1 / 7 | 0.26 | tie |

## What this shows

1. **The fix works mechanically.** Collapse disappears, and the demes that used to collapse
   now search full-size neighbourhoods (median fields per deme +12 % on mux11, +53 % on pa5).
2. **Collapse was not costing GOMEA accuracy.**
   * With full neighbourhoods, GOMEA solved mux11 in about the same number of evaluations
     and reached the same pa5 error.
   * A collapsed deme still found improvements in its smaller neighbourhood, or the next
     deme recovered.
3. **No CPU penalty showed up either.**
   * The pilot had predicted 1.3–2× more GOMEA CPU. Here CPU tied, 6 / 4 and 5 / 5.
   * Larger demes cost more per deme, but KR ran fewer GOMEA demes on mux11 (237 against
     278).
   * The CPU numbers were measured under uneven load and are noisy.
4. **Against capped HC at equal evaluations:** no significant difference on either problem.
   * mux11: GOMEA lower in 7 of 10.
   * pa5: HC lower in 7 of 10, with median error 6.5 against 7.5.
   * Phase 1n's 20-seed mux11 result (GOMEA better at equal evaluations) is not contradicted;
     10 seeds are fewer.

## Limits

* Two Boolean problems and 10 seeds. The option does not apply to regression knobs.
* CPU was measured on a shared, unevenly loaded machine.
