# Phase 1k pre-registration: with fewer pair knobs, does learned linkage gain on random?

Written 2026-10-13, before any Phase 1k run.

**Requested by the user:** reduce the knob count per deme so the linkage model has fewer
knob pairs to estimate, "but with reduced pair knobs because without it, we can't build
more complex programs", with disc probing on; then "Go with phase 1i and the follow up with
the rest".

## Question

The linkage tree is built from the mutual information between every pair of knobs:
ℓ(ℓ−1)/2 pairs for ℓ knobs. With few samples, many pairs look linked by chance. Phase 1h
found that LT-B beats a random tree (RT) on frozen maj5 only at N = 1,024. **H:** with fewer
knobs, learned linkage gains on random linkage at the same population size.

## How knobs are reduced

* **Fewer pair knobs:** `--logical-perm-ratio -0.75` builds 1 pair knob `op($i $j)` per
  and/or node instead of 5 (maj5 has 5 inputs; `floor((1 + r) × 5)`).
  * New two-input subtrees can still be added, so programs can still grow.
  * −0.8 would give 0 pairs, because of floating-point rounding.
* **Measured knob counts** (default → −0.75): x0 111 → 64–65, x1 158 → 92–102,
  x2 137 → 77–79, x3 49 → 26–31. That is about 40 % fewer knobs and about 64 % fewer pairs.
* **Disc probing is unchanged.** It is already on for these problems and drops knobs whose
  settings simplify away (`-B 2`, the default).
  * `-B 3` was measured and rejected. It left 1–43 knobs, including a single knob on x3,
    which points to an artefact of exemplars that were only simplified at effort 2.

## Design

* **Problem:** the Phase 1h frozen maj5 fixtures `maj5-x0` to `x3`, `-m 300000`.
* **Arms:**
  * `LT-B-RK-N64`, `RT-RK-N64`, `LT-B-RK-N256` and `RT-RK-N256`:
    `-a gomea --gomea-pop-size N --gomea-skip-inactive 0 --gomea-fos {lt-b,rt}
    --logical-perm-ratio -0.75`.
  * The default-knob arms `LT-B-N{64,256}` and `RT-N{64,256}` are **reused from Phase 1h step
    1**: same seeds, same budget, and code identical on this path.
* **Seeds:** under the 10 + 10 rule, stage 1 runs seeds 4001–4010 per fixture (40 pairs per
  N). The extension runs seeds 4011–4020 with all eight arms (`configs/gomea/phase1k/ext.json`).
* **Manifests:** `configs/gomea/phase1k/stage1.json` and `ext.json`; code `d875294`;
  14 workers.
* **Launch:** after Phase 1i finishes (`tools/gomea/phase1k_run.sh` waits for its launcher).
* **Estimate:** stage 1 is 160 runs, about 3–4 h.

## Analysis (`tools/gomea/phase1k_report.py`)

* **Metric:** final error, the wrong rows out of 32. A run that does not end `ok` counts as 32.
* **Primary,** per N, with Holm correction over the two N:
  * **`interaction`:** (LT-B − RT) with reduced knobs minus (LT-B − RT) with default knobs,
    paired by (fixture, seed), tested with a Wilcoxon signed-rank test.
  * A negative value means fewer knobs widen learned linkage's lead.
* **Two looks, each at α = 0.025:**
  * **Stage 1:** Holm p < 0.025 decides the contrast: "reduced helps" with a negative mean,
    "reduced hurts" with a positive one. 0.025 ≤ Holm p < 0.25 means "extend". Otherwise
    "no difference".
  * **Stage 2:** all 80 pairs, the same rule without "extend".
* **Secondary** (with its own p, not part of the decision):
  * **Improvement rate:** the share of evaluated GOM trials that improved the individual,
    LT-B minus RT, per knob set and N. It asks directly whether learned groups are better
    building blocks, even when final errors tie.
  * **Descriptive:** LT-B − RT within each knob set; mean error and solved runs per arm.
* **Already visible before this phase** (Phase 1h step 1, default knobs): the improvement
  rate is higher for LT-B than for RT in 37 of 40 pairs at N = 64 and 38 of 40 at N = 256,
  although final errors tie. It was found while testing this report script. It is recorded
  here so it is not mistaken for a Phase 1k result.
