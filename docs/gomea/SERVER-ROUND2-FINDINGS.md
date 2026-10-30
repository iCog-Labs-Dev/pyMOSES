# Server round 2 findings: knob cap, linkage ablation and generation-0 ranking (phases 1ak, 1al, 1am)

**Status:** complete.

* **Run:** 2026-10-28 05:52 to 2026-10-30 01:16 UTC (43.4 h wall), on the same pod as
  round 1 (`jupyter-yderese`, cgroup quota 12 CPUs, 64 GB). W = 11 workers (limited by
  CPU). One orchestrator session, no restarts.
* **Runs:** 438 (18 pilot runs and 420 phase runs), all `ok`, 0 failed. Total CPU 289.5 h
  (the sum of `cpu_s` over all run summaries).
* **Code:** git HEAD `5fb393d`. The tree was dirty, but only in two round-1 configs (see
  "Deviations"). Every arm passed `--knob-probe-baseline reduced`.
* **Protocol:** `SERVER-ROUND2-PREREGISTRATION.md`. Order as registered: S0, 1ak-pilot, 1ak,
  1al, 1am.
* **Results (outside git):** `../server-results-round2/gomea-results/`. The round log is
  `server-run2/round-summary.md` and `server-run2/events.jsonl`. Reports:
  `phase1ak/report-stage2`, `phase1al/report-stage1`, `phase1am/report-stage1`. Boolean runs
  are in `1/phase1a*-bool-*/runs/`, regression runs in `phase1a*/phase1a*-sr-*/runs/`.
* **How the extra numbers were computed:** a read-only script read each run's
  `summary.json` and `events.jsonl`.
  * "Gap" is the median over runs of (test NMSE − train NMSE).
  * "Generations per deme" is the median over runs of each run's median `generations` (from
    the GOMEA `end` events). "Demes with ≥ 3 generations" is the median over runs of that
    share.
  * "Linkage share" is the linkage time as a share of the GOMEA engine's time (`timings` in
    the `end` events). The engine time does not include deme building (knob building) outside
    the engine.
  * All other numbers are from `aggregate.csv` and `contrasts.csv`.

**Answer in short:**

* **The knob cap works mechanically:** with K = 16, the median generations per deme rise
  from 1 to 3–5.2, and 61–99 % of demes run 3 or more generations (0 % in the base GOMEA).
* **But it makes GOMEA worse:** KC and KC-STATIC are **worse** than plain GOMEA on all
  three problems (mux11, Con, EnH).
* **Learned linkage still gives no benefit:** at 3–5 generations per deme, LT-B ties the
  univariate and random-tree FOS on all three problems, and N = 64 does not change this.
* **The generation-0 fitness ranking gives no benefit:** all 9 contrasts of 1am are ties.
* **Recommendation:** do not use a knob cap (keep the code off by default). The round-1
  recommendation does not change.

## S0. Environment check

* **pytest:** 1 failed, 8,749 passed, 13 skipped. The failure is the same environmental test
  as in round 1, `test_toolchain_drift_is_clean_on_the_real_repo` (the pod's g++ 13.3.0
  against the fixtures' g++ 16.2.1).
* **Reproducibility:** 4 of 4 round-1 reference runs matched exactly. Thus with the knob
  cap off, the new code reproduces round 1. The base GOMEA arm agrees too: mux11 needs 5,538
  evaluations to solve (round 1, 1ae stage 1: 5,538) and Con splits 1–10 give 37.1.

## 1ak-pilot. Choosing K

Arm KC (`explore`, LT-B, N = 16, cap 250) with K ∈ {8, 16, 32}, 2 runs per problem.
Rule: the largest K with a median of at least 3 generations per deme (over the demes of
both runs).

| Problem | K = 8 | K = 16 | K = 32 | Chosen K |
|---|---|---|---|---|
| mux11 | 5 (157 demes) | 5 (164) | 2 (139) | 16 |
| Con | 5 (86) | 3 (81) | 2 (82) | 16 |
| EnH | 5 (89) | 3 (81) | 2 (80) | 16 |

Values: median generations per deme. Log: `phase1ak-pilot/knob-cap.json`. The configs
already had K = 16, so the script changed no file. With K = 16 the default explore count is
E = 4, so the core has 12 knobs.

## 1ak. Does a knob cap help? (part A)

**Verdict (primary):** KC vs GOMEA and KC-STATIC vs GOMEA are **worse** on all three
problems. mux11 was decided at stage 1 (10 seeds). Con and EnH were extended to 20 splits
(stage-1 Holm p 0.055).

| Contrast | A lower / higher | p | p (Holm) | Verdict |
|---|---|---|---|---|
| mux11: KC vs GOMEA | 0 / 10 | 0.0020 | 0.0078 | **A worse** |
| mux11: KC-STATIC vs GOMEA | 0 / 10 | 0.0020 | 0.0078 | **A worse** |
| Con: KC vs GOMEA | 4 / 16 | 0.0007 | 0.0035 | **A worse** |
| Con: KC-STATIC vs GOMEA | 2 / 18 | 0.00004 | 0.0002 | **A worse** |
| EnH: KC vs GOMEA | 3 / 17 | 0.0094 | 0.0094 | **A worse** |
| EnH: KC-STATIC vs GOMEA | 3 / 17 | 0.0037 | 0.0078 | **A worse** |
| *Secondary:* KC vs KC-STATIC: mux11 / Con / EnH | 4/6, 12/8, 8/12 | | 1.0 each | tie |
| KC-STATIC vs KC-RANDOM: mux11 / Con / EnH | 5/5, 7/3, 4/6 | | 1.0 each | tie |

**mux11 (10 seeds; metric CPU to solve):**

| Arm | CPU to solve | Solved | Evaluations to solve | CPU per evaluation | Demes | Size | Generations per deme | Demes with ≥ 3 generations | Linkage share |
|---|---|---|---|---|---|---|---|---|---|
| GOMEA | **535 s** | 10/10 | 5,538 | 0.108 s | 22.5 | 36 | 1 | 0 % | 13 % |
| KC | 5,901 s | 10/10 | 19,448 | 0.318 s | 82 | 55.5 | 5 | 91 % | 50 % |
| KC-STATIC | 4,502 s | 10/10 | 9,973.5 | 0.531 s | 88.5 | 50 | 5 | 97 % | 0 % |
| KC-RANDOM | 4,174 s | 9/10 | 13,468 | 0.323 s | 108.5 | 49.5 | 5 | 99 % | 0 % |

**Regression (GOMEA, KC and KC-STATIC 20 splits; KC-RANDOM 10 splits):**

| Problem | Arm | Test NMSE | Train NMSE | Gap | Size | CPU per run | Generations per deme | Demes with ≥ 3 generations |
|---|---|---|---|---|---|---|---|---|
| Con | GOMEA | **35.2** | 32.4 | 2.3 | 229.5 | 929 s | 1 | 0 % |
| Con | KC | 49.2 | 46.4 | 2.4 | 157 | 883 s | 3 | 68 % |
| Con | KC-STATIC | 57.2 | 50.4 | 4.3 | 130 | 665 s | 3 | 74 % |
| Con | KC-RANDOM | 61.0 | 52.5 | 4.9 | 135.5 | 785 s | 3.8 | 74 % |
| EnH | GOMEA | **15.6** | 14.3 | 1.5 | 408.5 | 1,387 s | 1 | 0 % |
| EnH | KC | 20.6 | 19.1 | 0.6 | 204.5 | 965 s | 3 | 65 % |
| EnH | KC-STATIC | 21.2 | 19.0 | 1.8 | 184.5 | 755 s | 3 | 69 % |
| EnH | KC-RANDOM | 20.5 | 20.4 | 0.3 | 162 | 835 s | 3.2 | 75 % |

**Other telemetry:**

* **Candidate knobs** (search positions in generation 0, median): about 280–355 on Con and
  EnH, and 1,410–1,740 on mux11. Thus K = 16 uses about 1–6 % of them.
* **Knobs that enter the core** (KC only; mean per deme, median over runs): 19.5 on Con,
  20.2 on EnH and 38.3 on mux11, for a core of 12 knobs. The core is not stable: most of it
  changes in each generation.
* **How demes end on mux11:** 91–92 % of KC-STATIC and KC-RANDOM demes stop on stagnation,
  after a median of 94–99 evaluations. For KC it is 25 %, and for GOMEA almost none (98 %
  use the full 250).

**Measured costs and sizes (computed from the medians above):**

* **Regression programs are smaller** with the cap: 32–43 % smaller on Con and 50–55 % on
  EnH (KC and KC-STATIC against GOMEA, 20 splits). The gap does not change in a consistent
  direction.
* **mux11 programs are larger** with the cap: 38–54 % (49.5–55.5 against 36 nodes).
* **mux11 CPU:** 7.8–11× GOMEA's CPU to solve. The capped arms use 3.6–4.8× the demes and
  1.8–3.5× the evaluations to solve.

### The cost of the linkage score (a problem in the knob-cap code)

* In `explore`, each generation ranks the knobs by linkage strength
  (`LinkageModel.knob_strengths` in `pymoses/gomea/fos.py`). It builds the entropy matrix
  over **all** linked search positions, not only the K selected knobs. On mux11 there are
  about 1,600 of them, so this is a large pairwise computation in each generation.
* **Measured on mux11 (1ak):** linkage is 50 % of the GOMEA engine time for KC (13 % for
  GOMEA). The engine is 36 % of KC's total CPU, so linkage is about 19 % of KC's total CPU
  (median 1,127 s per run). On Con and EnH linkage is 2 % of engine time or less.
* **This does not explain the result.** KC-STATIC and KC-RANDOM compute no linkage score
  (linkage share 0 %), but they also use 8.4× and 7.8× GOMEA's CPU to solve mux11. Their
  engine time is 14 % of total CPU; the rest is mostly deme building for 3.9–4.8× the demes.
  The evaluations to solve (a measure that does not depend on CPU) show the same direction:
  all capped arms need more than GOMEA.
* **Effect on the round:** the linkage score made the explore arms slower in CPU and wall
  time (see "Deviations"). It did not change any verdict on Con or EnH, where the metric is
  test NMSE at fixed evaluations.

### Interpretation (1ak)

* A cap of 16 knobs makes demes run several generations, so the linkage tree is relearned
  from selected populations. But each generation can change only 16 of 300–1,700 knobs. The
  base GOMEA's FOS covers all knobs, although its single generation is partial. Restricting the search
  to a small subset costs more than relearning linkage gains.
* KC-STATIC ties KC-RANDOM: ranking knobs by fitness contrast is no better than a random
  choice. KC ties KC-STATIC: rotating explore knobs does not help.

## 1al. Does learned linkage help when demes run several generations? (part B)

**Verdict (stage 1, 10 seeds or splits):** all six primary contrasts are **ties** (Holm p
0.96–1.0). All six secondary contrasts are ties. Nothing was extended.

| Contrast | A lower / higher | p (Holm) | Verdict |
|---|---|---|---|
| KC vs KC-UNI: mux11 / Con / EnH | 7/3, 5/5, 4/6 | 1.0, 1.0, 1.0 | tie |
| KC vs KC-RT: mux11 / Con / EnH | 6/4, 7/3, 6/4 | 1.0, 0.96, 1.0 | tie |
| *Secondary:* KC-N64 vs KC-RT-N64: mux11 / Con / EnH | 4/5 (1 tied pair), 6/4, 5/5 | 1.0 each | tie |
| KC-N64 vs KC: mux11 / Con / EnH | 4/6, 3/7, 4/6 | 0.40, 1.0, 1.0 | tie |

**mux11:**

| Arm | CPU to solve | Solved | Evaluations to solve | CPU per run | Demes | Size | Generations per deme | Demes with ≥ 3 generations | Linkage share |
|---|---|---|---|---|---|---|---|---|---|
| KC (LT-B) | 5,916 s | 10/10 | 19,448 | 5,916 s | 82 | 55.5 | 5 | 91 % | 50 % |
| KC-UNI | 5,730 s | 10/10 | 16,414 | 5,731 s | 73.5 | 58 | 5 | 98 % | 52 % |
| KC-RT | 5,213 s | 10/10 | 19,589.5 | 5,213 s | 81 | 52.5 | 4.5 | 88 % | 49 % |
| KC-N64 | unsolved (median) | 4/10 | – | 3,083 s | 42 | 43 | 5 | 96 % | 38 % |
| KC-RT-N64 | 3,659 s | 8/10 | 32,815 | 3,057 s | 35.5 | 45.5 | 4 | 97 % | 38 % |

KC-UNI and KC-RT also spend about half their engine time on linkage, because the knob
ranking uses LT-B in every arm (as registered).

**Regression:**

| Problem | Arm | Test NMSE | Train NMSE | Gap | Size | CPU per run | Generations per deme | Demes with ≥ 3 generations |
|---|---|---|---|---|---|---|---|---|
| Con | KC | 49.2 | 43.8 | 4.9 | 157 | 877 s | 3 | 67 % |
| Con | KC-UNI | 53.9 | 52.2 | 4.8 | 115 | 741 s | 5.2 | 99 % |
| Con | KC-RT | 58.1 | 55.5 | 2.9 | 131 | 667 s | 3 | 63 % |
| Con | KC-N64 | 50.8 | 46.0 | 5.7 | 105 | 504 s | 3 | 70 % |
| Con | KC-RT-N64 | 55.6 | 47.7 | 6.0 | 53 | 360 s | 3 | 70 % |
| EnH | KC | 22.0 | 21.1 | 1.2 | 196.5 | 1,001 s | 3 | 67 % |
| EnH | KC-UNI | 19.6 | 18.4 | 1.1 | 215 | 1,645 s | 5 | 98 % |
| EnH | KC-RT | 21.1 | 20.0 | 3.0 | 247 | 1,182 s | 3 | 65 % |
| EnH | KC-N64 | 24.1 | 23.8 | 1.2 | 105.5 | 479 s | 3 | 65 % |
| EnH | KC-RT-N64 | 27.2 | 24.4 | 0.3 | 126 | 714 s | 3.2 | 80 % |

**Interpretation:**

* At 3–5 generations per deme, the learned linkage tree is no better than a univariate FOS
  or a random tree. This is the setting that round 1 could not test.
* A larger population (N = 64, deme cap 1,000) does not make learned linkage pay off. On
  mux11 it hurts: KC-N64 solved only 4/10 within the 40,000-evaluation budget.
* The univariate FOS gets more generations per deme (5–5.2 against 3–5), as the
  pre-registration predicted, and still ties.

## 1am. Does the fitness ranking in generation 0 help? (part C)

**Verdict (stage 1):** all six primary and all three secondary contrasts are **ties** (Holm
p 0.79–1.0).

| Contrast | A lower / higher | p (Holm) | Verdict |
|---|---|---|---|
| KC vs KC-R0-ENT: mux11 / Con / EnH | 4/6, 4/6, 5/5 | 1.0 each | tie |
| KC vs KC-R0-RAND: mux11 / Con / EnH | 2/8, 5/5, 5/5 | 0.79, 1.0, 1.0 | tie |
| *Secondary:* KC-R0-ENT vs KC-R0-RAND: mux11 / Con / EnH | 5/5, 5/5, 6/4 | 1.0 each | tie |

| Problem | Arm | Metric | Solved / Train NMSE | Gap | Size | CPU per run | Generations per deme | Demes with ≥ 3 generations |
|---|---|---|---|---|---|---|---|---|
| mux11 | KC (contrast) | 5,876 s CPU to solve | 10/10 | – | 55.5 | 5,876 s | 5 | 91 % |
| mux11 | KC-R0-ENT | 3,217 s | 10/10 | – | 35 | 3,217 s | 4.2 | 76 % |
| mux11 | KC-R0-RAND | 3,952 s | 10/10 | – | 40.5 | 3,953 s | 5 | 86 % |
| Con | KC (contrast) | 49.2 test NMSE | 43.8 | 4.9 | 157 | 876 s | 3 | 67 % |
| Con | KC-R0-ENT | 50.9 | 42.7 | 4.4 | 129 | 720 s | 3 | 64 % |
| Con | KC-R0-RAND | 51.5 | 48.6 | 4.5 | 144 | 848 s | 3 | 71 % |
| EnH | KC (contrast) | 22.0 test NMSE | 21.1 | 1.2 | 196.5 | 1,007 s | 3 | 67 % |
| EnH | KC-R0-ENT | 21.7 | 20.3 | 3.1 | 194 | 906 s | 3 | 61 % |
| EnH | KC-R0-RAND | 23.6 | 23.1 | 1.3 | 183 | 939 s | 3 | 69 % |

mux11 evaluations to solve: KC 19,448, KC-R0-ENT 17,553, KC-R0-RAND 14,636.

**Interpretation:** a fitness-based generation-0 core gives no benefit over an entropy
ranking or a random one. This agrees with 1ak's KC-STATIC vs KC-RANDOM tie. On mux11 the
fitness ranking is descriptively the slowest (KC lower in only 2 and 4 of 10 pairs).

## What this round shows

1. **Knob cap:** it gives demes several generations, but it is worse than plain GOMEA on
   all three problems. Do not use it. The code stays, off by default (K = 0 is
   byte-identical).
2. **Learned linkage:** no benefit at one generation per deme (round 1), at 3–5 generations
   (1al), or at N = 64 (1al). For these settings the linkage-learning question is closed.
3. **Generation-0 ranking:** fitness contrast, entropy and random give the same results
   (1am, and 1ak's static vs random).
4. **Reading:** GOMEA's regression gain inside MOSES comes from optimal mixing (GOM over
   a FOS that covers all knobs in each deme), not from learned linkage.
5. **Round-1 recommendation unchanged:** GOMEA as an optional optimizer for regression with
   linear scaling; HC stays the default for Boolean.

## Why learned linkage gives no benefit (investigation, 2026-10-30)

The user asked why learned linkage gave no benefit here, when it seems a large source of
gains in GP-GOMEA. Three investigations followed. They were not pre-registered. The
evidence is in `linkage-investigation-2026-10-30/` (A: literature notes; B: scripts and
tables; C: scripts and statistics). Paper PDFs and raw run outputs were not kept.

**A. What the GP-GOMEA papers measured** (`A/A-literature.md`). The large gain in those
papers is GOM with a tree-shaped FOS against univariate. Learned linkage (LT) against a
random tree (RT) gives a small gain, and only at a large scale:

| Source | Setting | LT against RT |
|---|---|---|
| GECCO 2017 | Boolean problems, IMS | The difference "is far less pronounced" than on artificial problems. No tests. |
| ECJ 2021 (SR) | N 2,000, about 2.4M evaluations | LT-B better on 10/10 datasets (train), 9/10 (test); median error about 6 % lower. Plain-MI LT about equal to RT. |
| Virgolin et al. 2020 | N 100–500, 10K–50K evaluations | LT not better than RT, or worse. Advice: RT unless N is "of the order of thousands or more". |
| Koch, Alderliesten, Bosman 2026 (arXiv 2602.02311) | 5 SR datasets, 10^7 evaluations | P(LT-B better than RT) 0.57; P(RT better than UNI) 0.92. |
| GOMGE 2018 | Grammatical evolution, N 500 | LT equal to RT. |

Our demes have populations 8–100× smaller and budgets 50–10,000× smaller than any setting
in which LT won. Our only LT win (Phase 1h, frozen maj5, N 1,024) is in the range that the
authors give.

**C. LT against RT and UNI in GP-GOMEA itself** (`C/analysis.txt`). GP-GOMEA 750220c,
Phase 1ai flags, `--gomfos LT|RT|U`, Con, EnH and Air, splits 1–10, 360 runs, all ok. LT
reproduced the Phase 1ai GPGOMEA and GPGOMEA-P16 arms on 60/60 runs.

| Setting | LT vs RT, wins/losses (Con, EnH, Air) | LT vs UNI | Holm p |
|---|---|---|---|
| N 500, 10k evaluations | 4/6, 6/4, 6/4 | 7/3, 8/2, 4/6 | all 1.0 |
| N 16, 10k evaluations | 4/6, 4/6, 4/6 | 3/7, 4/6, 6/4 | all 1.0 |
| N 2,000, 50k evaluations | 6/4, 7/3, 4/6 | 4/6, 7/3, 4/6 | ≥ 0.785 |

Over all 90 pairs, LT won 45 and lost 45 against RT. Thus the absence of a linkage gain is
not a fault of the port. With n = 10 only large effects are detectable.

**B. Diagnosis inside pyMoses** (`B/sum_A.txt`, `B/sum_B.txt`; instrumentation by wrapping
from outside the repo; mux11 seeds 3001–3003, Con splits 1–2; small samples, so read the
numbers as descriptive):

1. **LT-B is blind in generation 0.** The bias correction makes the similarity exactly 0
   for every pair when the population is the initial one. The tree is then a balanced
   tree over a random order. In round 1, 94.8–99.6 % of demes in the cap-250 arms stop inside generation 0, so
   the round-1 LT-B arm was in effect a random tree. (GP-GOMEA has the same property, but
   its runs last many generations.)
2. **The landscape gives linkage little to use.** Pairs of knobs are non-additive in
   60–90 % of random pairs, and top-linked pairs are not more epistatic. Pairs that
   improve only when both knobs change are rare (0–2.5 %), for learned and random pairs.
3. **GOM gains come mostly from one-knob changes.** 62–78 % of accepted improvements
   change one knob. Improvements per scored trial fall as the subset grows (mux11 LT-B:
   0.064 for size 1, 0.032 for size 16+). UNI had the most improvements per trial (mux11
   0.0099 against 0.0066 for LT-B; Con 0.0405 against 0.0334).
4. **At N 16 the learned matrix is noise plus sampler structure.** After generation 0,
   only 1–2 % of pairs exceed the 95th percentile of a null that keeps the sampler's row
   structure. At N 64 and above, more pairs exceed it (11–56 %), but the top pairs are not
   more epistatic, and only 0–7 % of tree subsets survive from one generation to the next.
5. Populations do not converge. With the cap, 90–100 % of knobs still vary after 2–7
   generations. Convergence is not the cause.

**Conclusion.** Learned linkage is a small, large-scale effect in GP-GOMEA too. In MOSES
demes the population is too small, the deme too short, and the useful improvements mostly
change one knob. The benefit of GOMEA in pyMoses comes from optimal mixing (copying values
from donors with strict acceptance), not from the learned linkage. No further
linkage-learning phases are recommended at MOSES deme scales.

**Not measured:** LT-B at N ≥ 64 without the cap over several generations; RT and UNI in
the diagnosis with the cap; interactions of three or more knobs; an epistasis probe with
donor values.

## Deviations from the pre-registration

1. **State folder:** round 2 wrote its state to `server-run2/`, not `server-run/`, because
   round 1's DONE marker and finished steps are in `server-run/`. This is the first
   implementation note of the pre-registration.
2. **Duration: 43.4 h wall, against an estimate of about 16 h** (about 15 h for the phases
   plus about 1.3 h for S0).
   * **Measured:** total CPU was 289.5 h, against an estimate of about 160 h (648 runs at
     about 900 CPU-seconds each). The actual run count was 438, lower than the 648 maximum.
   * **Why (measured):** mux11 runs used a mean of 5,548–5,922 CPU-seconds per phase, not
     about 900. mux11 used 197.2 of the 289.5 CPU-hours (68 %) with 126 of the 438 runs. The
     capped arms needed more evaluations and more demes (1ak), and the explore arms also
     paid for the linkage score.
   * **Why (interpretation):** 289.5 CPU-hours on 11 workers is about 26 h of full use. The
     other hours come from running one phase at a time: at the end of each phase, a few long
     mux11 runs kept the phase open while most workers were idle. The pilot alone took 4.5 h
     wall (07:12–11:41), because of its six mux11 runs (mean 5,922 CPU-seconds).
3. **Dirty tree:** `git status` at the round start shows two modified files,
   `configs/gomea/phase1af/bool.json` and `configs/gomea/phase1af/sr.json`. Round 1's pilot
   script wrote LONG-RAND's p into them. Round 2 does not use them.
4. **pytest:** 1 known environmental failure (see S0). As registered, this is logged, not
   fatal.
5. **Implementation notes added before the round** (from the end of the pre-registration):
   * the knob-cap random stream is seeded by one extra draw from MOSES's global generator,
     only when K > 0; K = 0 stays byte-identical (golden test against `04c2020`);
   * the generation-0 contrast uses the penalized score, and a NaN contrast scores 0;
   * 1al and 1am wait for `phase1ak-pilot/knob-cap.json`;
   * K can differ per problem, so the regression manifests are split per dataset.
6. **Secondary contrasts never extend:** KC-RANDOM has 10 splits on Con and EnH, so KC-STATIC
   vs KC-RANDOM uses 10 pairs while KC vs KC-STATIC uses 20.

## Limits

* One K (16) for all problems, chosen by the pilot rule. A larger K gives fewer generations
  (K = 32: median 2), so it would not test linkage in the intended setting.
* CPU and wall time of the explore arms include the linkage-score cost described above. Use
  evaluations to solve when you compare explore arms with static or random arms on mux11.
* 1al and 1am are stage 1 only (10 seeds or splits), because no contrast reached the
  extension band.
