# Phase 1d findings: why MOSES hill climbing beat embedded GOMEA

**Status: complete.** Stage 1 ran 990 frozen-representation runs and stage 2 ran 240
end-to-end runs, all `ok`.

**Reuse check.** Six reruns of Phase 1 LT-B (maj5 and mux6, seeds 1001–1003) match their
Phase 1 originals in trajectory and results (only timing fields and new telemetry keys
differ). Reusing the Phase 1 `HC`, `LT-B` and `UNI` runs also relies on the code
regression tests in `ee1957e`; HC, UNI and pa4 were not re-run.

## Implementation and comparison

* **Code:** `ee1957e` adds opt-in options: uniform initialization, strict acceptance and a
  per-deme genotype memo. It also adds a harness duplicate counter that is identical for
  every arm.
* **Base configuration:** Phase 1 LT-B (N = 16). Each factor arm changes one factor;
  `LTB-COMBO` changes both carried ones.
* **Stage 1:** 11 frozen exemplars, 10 seeds, 5,000 evaluations.
* **Stage 2:** maj5 and mux6 with 30 seeds, pa4 with 20; 20,000 evaluations.
* **Protocol:** `PHASE1D-PREREGISTRATION.md`.

| Hypothesis | Stage 1 (one frozen deme, vs LT-B) | Stage 2 (end-to-end) | Verdict |
|---|---|---|---|
| A: too uniform a start | uniform start keeps a median of 16/16 genotypes distinct (LT-B 2–5); wider radii 2–16; neither improves error in any family (Holm p ≥ 0.36) | not carried | **not supported** |
| B: too small a population | N = 64 better by 5k evaluations in every family (e.g. pa4 W/T/L 15/23/2), not by 1k | N = 64 *worse* than LT-B on maj5 (1/30 vs 6/30) and mux6; tie on pa4 | **contradicted** end-to-end |
| C: duplicate re-scoring | memo: the same final result in 110/110 units, about ⅓ fewer charged evaluations | not carried | **undetermined** |
| D: too long per deme | carried by design | 500-eval cap: maj5 **20/30** vs LT-B 6/30 (*better*), **tie with HC**; mux6 evals-to-solve 2,569 vs 4,829 (*better*), *worse* than HC (1,627); pa4 tie with both | **supported** |
| E: neutral drift | strict acceptance: no significant effect (Holm p ≥ 0.31) | not carried | **not supported** |
| D+B combined | — | maj5 16/30 (tie with LT-B and HC); mux6 2,595 evals (*better* than LT-B, *worse* than HC); pa4 tie with both | no gain over D alone |

## What the evidence supports

Only one manipulation changed the end-to-end outcome for the better: **capping each deme
at 500 evaluations.** It raised the demes per run from 18 to 29.5 on maj5, from 4 to 6 on
mux6, and from 21.5 to 42 on pa4. That turned LT-B from significantly worse than hill
climbing on maj5 into a tie (20/30 each), and nearly halved evaluations-to-solve on mux6.

Population 64 points the same way. In a single frozen deme it reaches better programs, but
only at the 5,000-evaluation checkpoint, after using a median of 3,200–5,000 evaluations
there. End to end it gets 2–4 demes per run and is worse than LT-B on maj5 and mux6.

Neither a more diverse start nor strict acceptance improved error. With a uniform start,
the share of improving trials fell (pa4: 0.6 % vs 2.9 %).

*Interpretation (not directly tested):*
* Per-deme budget use is the main cause of the Phase 1 gap.
* MOSES's strength is rebuilding the representation around each new best program. Its
  knob building supplies the building blocks that GOMEA's linkage learning tries to infer
  from a small population.
* GOMEA's useful gains come in a deme's first few hundred evaluations. After that, a fresh
  representation is worth more.

## Remaining gap and limitations

* **mux6:** capped GOMEA still needs about 1.6× HC's evaluations.
* **pa4:** a tie only statistically; median error 4 vs 2.
* **Memo:** not tested end-to-end.
* **Cap:** only one value (500) was tried.
* **HC:** its deme counts were not recorded.

## Decision

Inside MOSES, run GOMEA **with a per-deme cap** (about 500 evaluations here). Next: sweep
the cap, and test the memo end-to-end.

Evidence: `phase1d-results/` and `../gomea-results/1/{phase1d-stage1-*,stage2-*}`.
