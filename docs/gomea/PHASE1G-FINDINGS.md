# Phase 1g findings: short demes or linkage? (exploratory)

**Status:** complete. The quick run (48 runs) and the bigger run (240 runs, including the
48) all ended `ok`. Protocol: `PHASE1G-PREREGISTRATION.md`. Code `c3d3ad6` adds
`-a hc-capped`. Results: `phase1g-results/` and `../gomea-results/phase1g/` (relative to
the repository root, outside it).

## Setup

* **Arms:**
  * `HC`: plain hill climbing;
  * `HC-C250`: hill climbing capped at 250 evaluations per deme;
  * `LTB-C250`: capped GOMEA with the bias-corrected linkage tree;
  * `UNI-C250`: capped GOMEA with no linkage (each knob mixed on its own).
* **Datasets:** Con, EnH and EnC.
* **Budget and pairing:** 2,000 evaluations per run, 20 paired splits.
* **Metric:** test NMSE (lower is better).

| Median test NMSE | Con | EnH | EnC |
|---|---|---|---|
| HC | 75.2 | 37.6 | 38.0 |
| HC-C250 | 75.7 | 37.7 | 35.1 |
| LTB-C250 | **55.6** | **31.6** | **28.2** |
| UNI-C250 | 64.7 | 34.1 | 33.7 |

Paired over 20 splits, with Holm correction across the 3 datasets. "Lower in" counts the
splits where the first arm had the lower error; *leads* means Holm p < 0.05 with a negative
median difference.

| Contrast | Con | EnH | EnC |
|---|---|---|---|
| HC-C250 vs HC (short demes alone) | 11/20, p_H 0.67 | 14/20, p_H 0.23 | 16/20, p_H 0.028 **leads** |
| LTB-C250 vs HC-C250 | 14/20, p_H 0.052 | 15/20, p_H 0.066 | 12/20, p_H 0.23 |
| UNI-C250 vs HC-C250 | 12/20, p_H 0.65 | 13/20, p_H 0.70 | 12/20, p_H 0.78 |
| LTB-C250 vs UNI-C250 | 14/20, p_H 0.12 | 10/20, p_H 0.39 | 13/20, p_H 0.18 |
| LTB-C250 vs HC | 17/20, p_H 0.015 **leads** | 14/20, p_H 0.027 **leads** | 16/20, p_H 0.015 **leads** |
| UNI-C250 vs HC | 13/20, p_H 0.21 | 14/20, p_H 0.25 | 11/20, p_H 0.25 |

* **Median differences, LTB-C250 vs HC-C250:** −17.3, −10.8 and −4.2.
* **Median differences, LTB-C250 vs UNI-C250:** −10.2, −2.0 and −7.3.
* **CPU:** the median CPU time was 56–107 s for both HC arms and 208–332 s for GOMEA.

## Verdict under the pre-registered reading

**Inconclusive.**
* **Short demes alone:** capped HC leads plain HC on only 1 of 3 datasets.
* **GOMEA over capped HC:** neither GOMEA arm leads capped HC on 2 of 3. LTB misses on Con
  and EnH at Holm p 0.052 and 0.066.
* **Linkage over no linkage:** LTB does not lead UNI on any dataset.

## What the data do show

1. **The full package replicates Phase 1f at a fifth of the budget.** Capped LT-B GOMEA beats
   plain HC on all 3 datasets (Holm p ≤ 0.027).
2. **Short demes alone are not the explanation.** Capping HC changed its median error by
   less than 1 point on Con and EnH, and by −5.5 on EnC.
3. **GOMEA without grouping is not enough either.** UNI-C250, which mixes one knob at a
   time, beat neither plain HC nor capped HC significantly anywhere.
4. **Copying knobs in groups seems to matter.** LT-B was ahead of UNI by a median of 10 and
   7 points on Con and EnC (raw p 0.04 and 0.09), and it is the only arm that beats HC. In
   Phase 1f, the *random* tree did as well as LT-B. Together these suggest that what helps
   is moving groups of knobs at once, not that the groups are learned. This is not
   established here.

## Why learned linkage cannot matter much here (measured on one EnH run, split 1)

GOMEA's telemetry for one LTB-C250 run:
* **Every deme ended within its first generation,** so each linkage model was learned once
  per deme, from **16 individuals**.
* **Representations had 20–264 knobs,** over 200 in most demes, with up to 526 FOS subsets.
* **The 234 mixing evaluations after the 16-individual start reached only about 2–5 of the
  16 individuals** on large demes.

For comparison, the C++ GP-GOMEA in Table 3's setting learns each model from 2,000
individuals over 31 fixed variables, about 1.3–1.4 million evaluations per run. There,
the linkage tree measurably beats the random tree on EnH and Con (Phase 1f part A).

## Next

Learned linkage would need far more data per model. Pooling statistics across demes (the
proposed global model) is the cheapest route. Seeding demes with diverse good programs is
another, as is fewer, larger groups per deme. A follow-up should also test the random tree
against no linkage directly, to settle whether grouping, not learning, is the active
ingredient.
