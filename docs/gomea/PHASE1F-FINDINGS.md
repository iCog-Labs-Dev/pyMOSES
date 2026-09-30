# Phase 1f findings: the GP-GOMEA paper's own benchmarks

**Status.**
* **Parts A and B:** complete.
* **Parts C and D:** stopped early at the user's request, after 181 of 330 runs. The
  tests use the three datasets with complete pairs; the rest is descriptive.

**Protocol and every deviation:** `PHASE1F-PREREGISTRATION.md`. **Results:**
`phase1f-results/`, and the raw runs in `../gomea-results/phase1f/`.

## The benchmarks

Virgolin et al. (2021) is the paper our port follows, and it tests only symbolic
regression.
* **Datasets:** Air, Bos, Con, Dow, EnC, EnH, Tow, WiR, WiW and Yac. Counts and sources
  are verified in `../gomea-results/phase1f/datasets/PROVENANCE.md`.
* **Splits:** 50/25/25 train/validation/test, paired across methods.
* **Settings:** functions {+, −, ×, ÷AQ} and linear scaling.
* **Metric:** test **NMSE = 100·MSE/var(y)**. Lower is better; 100 is no better than
  predicting the mean.

## Answers

| Part | Question | Answer |
|---|---|---|
| A | Does the authors' C++ GP-GOMEA reproduce the paper (Table 3)? | **Partly.** The error levels replicate on all 60 medians. The linkage tree's advantage over the random tree replicates on training error, but most of the bias correction's advantage does not. |
| B | Does stand-alone GP-GOMEA beat classic C++ MOSES? | **Yes, on all 10 datasets**, by large margins at equal wall time. |
| C | Is pyMoses GOMEA consistent with the literature? | **Consistent; no regression visible.** LTB is never significantly worse than LTMI or RT. But none shows the paper's advantage either at this budget, and its levels are far from the paper's. |
| D | Does pyMoses GOMEA beat pyMoses hill climbing? | **Yes, on 2 of the 3 complete datasets (Air, Con), with a tie on Bos.** It is lower in every available pair on 5 of the 7 partial datasets (descriptive, 1–3 pairs). It costs 3.3–5.4× HC's CPU time on the three tested datasets, and 2–23× across all ten. |

## A: replication of Table 3 (C++ GP-GOMEA, 30 splits × 10 datasets, 900 runs)

The settings are those of Table 3: population 2,000, height 4, 20 generations and no ERC.
LT–MI needed a one-line source patch, because upstream hard-wires the bias correction on.

* **Levels replicate.** All 60 medians (3 FOS × 10 datasets × train and test) are within
  0.91–1.15× of the published values (`partA_levels.csv`).
* **The pattern replicates only partly** (`partA_verdict.json`). "Better" means a paired
  Wilcoxon over 30 runs with p < 0.025, which is also how the paper counts.

  | LT–MIb̃ significantly better than … | Paper | Ours |
  |---|---|---|
  | RT, train | 10/10 | **10/10** |
  | RT, test | 9/10 | 4/10 |
  | LT–MI, train | 8/10 | 5/10, and 1 worse (Dow) |
  | LT–MI, test | 7/10 | 1/10 |

* **Possible reasons, not tested:**
  * the GP-GOMEA commit we built (upstream 750220c) is later than the paper;
  * the splits and seeds differ;
  * the differences are small (often < 1 NMSE point), so with 30 runs they are
    sensitive to both.

## B: GP-GOMEA vs classic C++ MOSES (1,000 s per run, 10 splits × 10 datasets)

* **GP-GOMEA:** the paper's Table 8 setting (IMS g = 4, height 4, ERC).
* **MOSES:** the regression setup that lets it run to budget, with its output chosen on
  validation.

| | Air | Bos | Con | Dow | EnC | EnH | Tow | WiR | WiW | Yac |
|---|---|---|---|---|---|---|---|---|---|---|
| GP-GOMEA test NMSE | 25.3 | 16.3 | 17.4 | 21.4 | 5.70 | 3.04 | 12.5 | 62.6 | 68.7 | 0.51 |
| C++ MOSES test NMSE | 46.7 | 72.5 | 51.7 | 94.1 | 20.5 | 17.7 | 60.5 | 76.4 | 88.2 | 3.99 |

* **Every dataset is *better* for GP-GOMEA.** It was lower on all 10 splits of every
  dataset: raw p = 0.00195, the minimum with 10 pairs, which is Holm p = 0.0195 over the 10
  datasets.
* **Throughput:** per-dataset medians of 3.7–24 million evaluations per run for GP-GOMEA
  (per run, 3.3–27.7 million), against 25–190 thousand for MOSES (per run, 19–362
  thousand).
* **Secondary Table 8 check:** our GP-GOMEA medians are within 20 % of the published
  IMS g = 4 medians on 8/10 datasets. The exceptions are EnC (1.29×) and EnH (1.48×).
  We used 10 runs, not 30, on faster hardware.
* **Differences between the two tools that we could not remove:**
  * MOSES has no analytic quotient and no size cap;
  * MOSES has no linear scaling in training (the target is rescaled instead);
  * MOSES constants are knobs, not ERCs.

## C and D: pyMoses GOMEA vs the literature and vs pyMoses hill climbing

**Setup: 10,000 evaluations per run, and the same MOSES regression setup for every arm.**
* **HC:** stock hill climbing.
* **LTB:** the Phase 1e winner, `--gomea-fos lt-b --gomea-max-evals-per-deme 250
  --gomea-genotype-memo 1`, plus the new `--gomea-contin copy`. That option makes
  constants genes that mixing copies but never mutates, as in the paper.
* **LTMI and RT:** the same, with the other linkage models.

**D: tests on Air, Bos and Con** (10 pairs each; Holm correction over the 3 datasets).

| Dataset | LTB test NMSE | HC test NMSE | LTB lower in | Holm p | Verdict |
|---|---|---|---|---|---|
| Air | 42.5 | 75.2 | 10/10 | 0.0059 | **better** |
| Bos | 41.6 | 77.4 | 9/10 | 0.065 | tie |
| Con | 37.1 | 54.9 | 10/10 | 0.0059 | **better** |

GOMEA's median CPU time was 3.3–5.4× HC's on these datasets: Air 7,710 vs 1,430 s,
Bos 6,410 vs 1,950 s, Con 2,980 vs 737 s. On the partial datasets the ratio reaches 2.2–23×;
Tow is 22.8× and WiR 8.9×.

**Descriptive only** (LTB vs HC test NMSE, with the number of pairs):

| Dataset | Pairs | LTB lower in | LTB | HC |
|---|---|---|---|---|
| EnC | 3 | 3 | 18.4 | 26.9 |
| EnH | 3 | 3 | 15.5 | 24.8 |
| Tow | 3 | 3 | 46.2 | 85.3 |
| WiR | 3 | 3 | 73.3 | 78.1 |
| WiW | 1 | 1 | 84.5 | 373 |
| Yac | 3 | 1 | 7.26 | 7.14 |

* **WiW:** the one pair is split 2. HC's program there extrapolates badly (train 92, test
  373). A second HC run with no LTB partner, split 3, reached about 1.45 million on test.
* **Dow:** both completed LTB runs beat HC (88.0 vs 95.9 and 84.5 vs 100.9). The other two
  hit the 3 GB memory cap and count as failures, i.e. infinite NMSE.
* **EnC:** HC's 26.9 is the median over the 3 paired runs; over all 4 HC runs it is 27.1.

**C: the linkage models inside pyMoses** (`partC_pattern.csv`).
* **Significance:** no contrast is significant on any of the 3 datasets (all p ≥ 0.06).
* **Train medians over the paired runs:** LTB's is lower than LTMI's on 2/3 datasets, and
  lower than RT's on 2/3. Over each arm's full set of runs, LTB is lower than RT on 3/3.
* **Not tested:** LTMI vs RT.
* **Verdict:** consistent under the pre-registered rule; nothing points to a regression.

This is weak evidence, though. At 10,000 evaluations, with 16-individual populations and
250-evaluation demes, the choice of linkage model makes little measurable difference. The
paper's advantage appeared at about 10⁶ evaluations with populations of 2,000.

**Levels.** pyMoses GOMEA is far from the C++ GP-GOMEA.
* **Test NMSE, pyMoses LTB vs C++ LT–MIb̃ on the same splits** (1–10 for Air, Bos and Con;
  1–3 for Tow and Yac): Air 42.5 vs 29.5, Bos 41.6 vs 18.3, Con 37.1 vs 20.2, Tow 46.2 vs
  13.0, Yac 7.26 vs 0.61.
* **Why this is expected and not a regression test:** pyMoses gets about 1 % of the
  evaluations, and it searches MOSES's representation rather than a fixed 31-node template.

## Bugs found and fixed along the way

All three were in shared infrastructure, not in the GOMEA code.

* **`reduce_factorize_fraction`** (`pymoses/comboreduct/reduct/contin_rules.py`): a port
  bug that crashed contin reduction. It had a missing placeholder argument, and it moved
  the wrong node. The fix is verified against the C++ build and covered by a regression
  test (`484e302`). It affects any pyMoses regression run, including hill climbing.
* **The regression scorer** (`tools/gomea/sr_eval.py`) mishandled boolean-typed columns
  (`impulse`). It now matches MOSES's own scores exactly (`3e41da7`).
* **The pyMoses recursion limit** on large representations: the runner raises it
  (`8b23deb`).

## Recommendation

* **For symbolic regression, stand-alone GP-GOMEA is far stronger than classic C++
  MOSES** on the paper's benchmarks, at equal wall time (part B). Python MOSES was not
  compared with it directly.
* **Inside pyMoses,** the capped GOMEA setup beat hill climbing on test error on 2 of the
  3 datasets tested, with a tie on the third. It pointed the same way on 5 of the 7
  partial datasets. It costs 3.3–5.4× the CPU time there, and up to 23× on the largest
  partial dataset. It is worth using when evaluations, not CPU, are the constraint.
* **Not established here:**
  * whether pyMoses GOMEA would show the paper's linkage-model advantage at a paper-scale
    budget;
  * the confirmatory results on the seven larger datasets. Finishing them needs about 2–3
    more days on this machine, or a bigger one.
