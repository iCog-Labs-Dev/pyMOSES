# Phase 1i pre-registration: does a starting population spread over all distances help?

Written 2026-10-13, before any Phase 1i run (the two pilot runs below excepted).

**Requested by the user:** "Wouldn't it be better to give all kinds of variations equal
chance instead of being biased to small variations first?", then "yes prepare option 1"
(equal chance per distance) and "Go with phase 1i".

## Question

GOMEA's starting population is sampled around the deme's exemplar with the radius mixture
`1:0.4,2:0.3,4:0.2,max:0.1`, so 70 % of individuals differ from the exemplar in only 1–2
knobs. Phase 1h found that learned linkage (LT-B) beats a random tree only with a large
population (N = 1,024). **H:** a starting population spread evenly over all distances gives
the linkage model more informative samples, so that:

1. LT-B reaches lower error;
2. LT-B's lead over the random tree (RT) grows.

Phase 1d tested a wider start (radii 2/4/8/max) and a uniform start at N = 16 and found no
gain (hypothesis A, "not supported"). This phase tests the idea at N = 256, where the
linkage model has more data to learn from.

## Code

* `--gomea-radii uniform` (commit `d875294`) gives every distance from 1 to all knobs the
  same probability. Within a distance, knobs and values are drawn exactly as before.
* The default mixture is unchanged.
  * **Pilot check:** maj5-x0, LT-B, N = 256, seed 4001 reproduced Phase 1h step 1 exactly
    (18,056 evaluations, error 2).
  * **Identity check:** one run repeats this under the Phase 1i budget and must match step 1
    (`configs/gomea/phase1i/identity-check.json`).
* **Pilot with the uniform start** (same fixture and seed, `-m 300000`):
  * about 40,000 evaluations per generation, against about 3,000 with the default start;
  * error 1 after 116,000 evaluations, unchanged up to 223,000.
  * This is the only uniform-start run seen before registration.

## Design

* **Problem:** the Phase 1h frozen maj5 fixtures `maj5-x0` to `x3` (`-e … -g 1`), N = 256.
* **Arms:**
  * `LT-B-UNI-N256`: `--gomea-fos lt-b --gomea-radii uniform`;
  * `RT-UNI-N256`: `--gomea-fos rt --gomea-radii uniform`;
  * `LT-B-N256` and `RT-N256`, the default start, **reused from Phase 1h step 1**: same seeds,
    and code that is identical on this path.
  * All arms use `-a gomea --gomea-pop-size 256 --gomea-skip-inactive 0`.
* **Budget:** `-m 150000`.
  * The uniform pilot reached its final error by 116,000 evaluations.
  * Default-start runs at N = 256 never used more than 46,883 (step 1), so the cap does not
    bind for them, and the step 1 results stand as they are.
  * The identity check confirms this.
* **Seeds:** following the 10 + 10 rule, stage 1 runs seeds 4001–4010 per fixture (40 pairs).
  The extension, seeds 4011–4020, runs all four arms
  (`configs/gomea/phase1i/ext.json`), because step 1 has no default-start runs for those
  seeds.
* **Manifests:** `configs/gomea/phase1i/stage1.json`, `identity-check.json` and `ext.json`;
  code `d875294`; 12 workers (20 were planned; cut to 12 at launch because only about 2.4 GB of RAM was free).
* **Estimate:** stage 1 is 80 runs at about 1.5–2 h each, about 11–13 h with 12 workers. The extension would add
  about the same.

## Analysis (`tools/gomea/phase1i_report.py`)

* **Metric:** final error, the wrong rows out of 32. A run that does not end `ok` counts as 32.
* **Primary contrasts,** paired by (fixture, seed), each a Wilcoxon signed-rank test, with
  Holm correction over the two:
  1. **`start`:** LT-B with the uniform start minus LT-B with the default start.
  2. **`interaction`:** (LT-B − RT) with the uniform start minus (LT-B − RT) with the default
     start. A negative value means the diverse start widens learned linkage's lead.
* **Two looks, each at α = 0.025** (0.05 overall):
  * **Stage 1:** Holm p < 0.025 decides the contrast: "uniform better" with a negative mean,
    "uniform worse" with a positive one.
  * 0.025 ≤ Holm p < 0.25 means "extend".
  * Otherwise "no difference".
  * **Stage 2:** all 80 pairs, the same rule without "extend".
* **Descriptive:**
  * RT uniform vs RT default;
  * LT-B vs RT within each start;
  * solved runs, evaluations, wall time and generations per arm.
* **Reading:** H holds if `start` or `interaction` is decided in the uniform start's favour.
* **Cost:** the uniform start uses many more evaluations (budget 150,000 against about
  20,000). Any gain must therefore be weighed against that, so evaluations are reported
  alongside error.

## Next

Phase 1k (fewer pair knobs per deme) is registered separately and queued after this phase.
