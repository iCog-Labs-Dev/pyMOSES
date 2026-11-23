# Server round 15: findings (2026-11-23)

**Pre-registration:** `SERVER-ROUND15-PREREGISTRATION.md` (approved 2026-11-20, with the stop-rule
amendment and the RV16 corrections). **Plan:** `IT-LINKAGE-STUDY-PLAN.md`. **Results:** two bundles
in the MOSES folder, next to the repo: `bundle-final-20261120T231546Z.tar.gz` (`round15`, S0 and 500
runs; SHA-256 `690196ce…89e28`) and `bundle-final-20261121T033211Z.tar.gz` (`round15b`, all 1,000
runs and the final report; SHA-256 `f18ce8b8…6e99c4`). **Code:** `276e3f9`. **Decisions:**
`APPROVALS.md`.

NMSE values are × 100. "Pairs" is the number of (dataset, split) pairs where A is lower / where B is
lower; the remaining pairs are ties. "Ratio" is exp(median log ratio): below 1 means that A has the
lower value. For an interaction contrast, d = log(IT-f / IT-RT) − log(T-f / T-RT): a ratio below 1
means that f gains more over the random tree on IT than on the template; above 1, the template gains
more. A verdict of "better", "worse" or "gains more" needs Holm p < 0.025 and a ratio at least 10 %
from 1. All 1,000 runs ended ok.

## Why this round was run

* **The problem.** In pyMOSES, learned linkage (LT-B) never beat a random tree (RT) in a real run
  (findings 1g, 1m, 1al, 1ay). It beat RT only once, on one frozen representation with a population
  of 1,024 (1h). The likely cause: MOSES builds new knobs for each deme from the exemplar's shape, so
  a knob position means something else in every deme, and each deme must learn its linkage from
  zero, from a small population, in a few generations.
* **The research question (chosen by the user, 2026-11-20).** When every individual has the same
  genome for the whole run, does learned linkage pay off? If it does, a linkage model that lasts the
  whole MOSES run (phase B of the plan; T3.4 in pyMOSES) could be worth building.
* **The design (the user's).** To keep the MOSES loop out of the answer, the round runs GOMEA on a
  fixed genome with no MOSES loop: the IT genome (8 terms × 2 factors × input and power; per-term
  least squares) and GP-GOMEA's tree template (height 4; linear scaling), each under five linkage
  models: UNI (no linkage), RT, LT (mutual information), LT-B (bias-adjusted mutual information)
  and NP (node proximity, a fixed structural model). 10 datasets, fresh splits 71–80, N = 1,000,
  E = 500,000 evaluations.

## Summary

| Family | Contrast | Pairs | Holm p | Ratio | Verdict |
|---|---|---|---|---|---|
| P1 | UNI gain, IT vs template | 42/58 | 0.031 | 1.008 | tie |
| P1 | LT gain, IT vs template | 39/61 | 0.031 | 1.008 | tie |
| P1 | **LT-B gain, IT vs template** | **26/74** | **2.5e-5** | **1.035** | tie (the template gains more) |
| P1 | **NP gain, IT vs template** | **25/75** | **1.0e-5** | **1.030** | tie (the template gains more) |
| P2 | IT-UNI vs IT-RT | 69/20 | 1.4e-6 | 0.994 | tie |
| P2 | IT-LT vs IT-RT | 74/15 | 6.7e-8 | 0.993 | tie |
| P2 | IT-LTB vs IT-RT | 73/16 | 5.6e-7 | 0.994 | tie |
| P2 | IT-NP vs IT-RT | 78/11 | 1.8e-10 | 0.989 | tie |
| P2 | IT-NP vs IT-LTB | 40/38 | 0.073 | 1.000 | tie |
| S1 | T-UNI vs T-RT | 71/29 | 2.4e-5 | 0.986 | tie |
| S1 | T-LT vs T-RT | 75/25 | 2.1e-6 | 0.982 | tie |
| S1 | **T-LTB vs T-RT** | **87/13** | **4.6e-11** | **0.956** | tie |
| S1 | **T-NP vs T-RT** | **88/12** | **8.7e-13** | **0.956** | tie |
| S1 | T-NP vs T-LTB | 59/41 | 0.17 | 0.994 | tie |
| S2 | IT-LTB vs T-LTB (test NMSE) | 76/24 | 1.2e-9 | 0.738 | **A better** |
| S2 | IT-NP vs T-NP (test NMSE) | 78/22 | 1.2e-9 | 0.748 | **A better** |

P1–S1 are on training NMSE (the primary metric). The IT contrasts have 11 ties: on Yac, all IT arms
found the same model on all 10 splits.

**Answers to the questions of the round:**

1. **Q3, the replication: yes.** On the template, our engine reproduces the known result: LT-B has a
   lower training NMSE than RT in 87 of 100 pairs and on 10 of 10 datasets (median of the splits),
   as the authors' C++ code did in Phase 1F part A (10 of 10). The gain is 4.4 % at the median pair,
   inside the 10 % band. Node proximity gives the same gain (0.956, 88/12) and ties with LT-B (0.994,
   59/41).
2. **Q1, the main question: no.** Learned linkage does not help more on the IT genome. The direction
   is the opposite: LT-B and NP gain about 4.4 % over RT on the template, but only 0.6–1.1 % on IT.
   The interaction is clear (74/26 and 75/25, Holm p ≤ 2.5e-5) but smaller than the 10 % band, so the
   verdict is "tie". UNI and LT show no difference between the genomes (1.008).
3. **Q2, the best linkage model on IT: node proximity, by a small margin, and no better than LT-B at
   the end.** Every structured model beats RT on IT in most pairs (69–78 of the 89 that differ), but
   by 0.6–1.1 %. NP is the largest (0.989, 78/11). NP and LT-B end equal (1.000, 40/38), but NP gets
   there faster (anytime area 0.987 against LT-B, 74/26; descriptive).

* **The absolute level (S2):** with the same linkage model, IT has 26 % lower test NMSE than the
  template (0.738 and 0.748, "A better"). This compares two genomes with different scorers
  (per-term least squares against plain linear scaling) and sizes (IT median 75 nodes and terms,
  template 31), so it is not a linkage result.
* **On test NMSE the linkage effects vanish** (S3: the P1 and P2 ratios are 0.995–1.049; the
  smallest Holm p is 0.012, for the NP interaction, again on the template side).

## Run health

* **Steps:** `round15`: S0 1.33 h, phase1bj-a 5.45 h (500 runs); `round15b` started by itself and ran
  phase1bj 4.27 h (the other 500). From 2026-11-20 16:28Z to 2026-11-21 03:32Z: 11.1 h, half of the
  estimate (about 22 h). W = 11. Both `bundle.err` files are empty. Git HEAD `276e3f9`, clean.
* **CPU:** 106.3 CPU hours (estimate about 226): T-UNI 19.4, T-RT 17.5, T-LT 14.9, T-LTB 19.1, T-NP
  18.1, IT-UNI 3.6, IT-RT 3.2, IT-LT 3.0, IT-LTB 4.1, IT-NP 3.4. The local pilot ran on a loaded
  machine (load 14–45), so its CPU per evaluation was about twice the server's.
* **Runs:** 1,000 ok, 0 failed; every run passed the cross-check against `sr_eval`.
* **Memory:** the highest peak was 367 MB (IT-RT), 18 % of the 2,048 MB limit; template arms 128–132
  MB.
* **End reasons (the approved stop rule):**

  | Arm | Budget E | Stalled | Converged | 300 generations | Evaluations, median [min] | Generations, median [max] |
  |---|---|---|---|---|---|---|
  | T-UNI | 100 | | | | 500,000 | 50 [123] |
  | T-RT | 100 | | | | 500,000 | 15 [32] |
  | T-LT | 16 | 66 | 18 | | 316,850 [124,573] | 29 [51] |
  | T-LTB | 56 | 31 | 11 | 2 | 500,000 [198,552] | 20.5 [300] |
  | T-NP | 68 | 12 | | 20 | 500,000 [193,869] | 35 [300] |
  | IT-UNI | 80 | | | 20 | 500,000 [268,363] | 31.5 [300] |
  | IT-RT | 100 | | | | 500,000 | 11 [31] |
  | IT-LT | 16 | 69 | 9 | 6 | 405,175 [218,932] | 27 [300] |
  | IT-LTB | 79 | 10 | | 11 | 500,000 [286,746] | 12 [300] |
  | IT-NP | 91 | | | 9 | 500,000 [311,008] | 12 [300] |

  The stop rule ended 286 runs before E. The 300-generation guard ended 68 runs, most on Air and Yac
  (5 and 6 inputs, where most trials are cache hits) and on Bos and WiR for T-NP. So the arms did not
  all use the same number of evaluations; the comparison is at the end of each run, as
  pre-registered.

## Program size, generations and CPU (descriptive)

Median training NMSE per dataset:

| Dataset | T-UNI | T-RT | T-LT | T-LTB | T-NP | IT-UNI | IT-RT | IT-LT | IT-LTB | IT-NP |
|---|---|---|---|---|---|---|---|---|---|---|
| Air | 34.45 | 35.56 | 33.89 | 31.92 | 33.64 | 21.18 | 21.35 | 21.04 | 21.04 | 21.04 |
| Bos | 14.50 | 15.88 | 14.42 | 13.92 | 14.80 | 8.72 | 8.77 | 8.65 | 8.41 | 8.40 |
| Con | 19.88 | 21.04 | 19.13 | 19.20 | 17.85 | 15.39 | 15.69 | 15.69 | 15.52 | 15.64 |
| Dow | 22.95 | 25.44 | 20.44 | 23.87 | 22.43 | 14.83 | 15.28 | 13.59 | 15.31 | 14.32 |
| EnC | 10.07 | 10.36 | 10.32 | 10.15 | 9.21 | 3.13 | 3.21 | 3.31 | 3.23 | 3.11 |
| EnH | 6.67 | 6.66 | 6.61 | 6.55 | 6.57 | 1.22 | 1.24 | 1.17 | 1.11 | 1.12 |
| Tow | 14.30 | 13.86 | 13.41 | 13.07 | 13.19 | 7.66 | 7.77 | 7.41 | 7.76 | 7.59 |
| WiR | 60.74 | 61.11 | 60.32 | 59.96 | 59.51 | 57.42 | 57.75 | 57.43 | 57.31 | 57.35 |
| WiW | 68.81 | 69.08 | 68.26 | 68.64 | 68.37 | 64.93 | 65.01 | 64.66 | 64.74 | 64.92 |
| Yac | 0.39 | 0.46 | 0.44 | 0.43 | 0.39 | 0.31 | 0.31 | 0.31 | 0.31 | 0.31 |

Median test NMSE per dataset:

| Dataset | T-UNI | T-RT | T-LT | T-LTB | T-NP | IT-UNI | IT-RT | IT-LT | IT-LTB | IT-NP |
|---|---|---|---|---|---|---|---|---|---|---|
| Air | 36.66 | 38.90 | 35.72 | 35.28 | 37.35 | 25.01 | 24.80 | 23.91 | 24.50 | 25.01 |
| Bos | 26.84 | 23.66 | 24.82 | 23.61 | 23.31 | 18.33 | 18.55 | 19.27 | 19.18 | 19.76 |
| Con | 24.11 | 22.45 | 21.31 | 20.96 | 21.11 | 20.20 | 19.35 | 20.42 | 20.14 | 20.20 |
| Dow | 23.16 | 28.27 | 22.49 | 24.11 | 23.25 | 15.84 | 16.90 | 15.68 | 17.05 | 16.63 |
| EnC | 10.35 | 10.11 | 10.96 | 10.63 | 9.67 | 3.49 | 3.54 | 3.51 | 3.51 | 3.37 |
| EnH | 6.39 | 6.65 | 6.56 | 6.13 | 6.29 | 1.32 | 1.37 | 1.37 | 1.20 | 1.28 |
| Tow | 13.59 | 13.21 | 13.24 | 12.74 | 12.82 | 7.91 | 7.90 | 7.68 | 7.79 | 7.80 |
| WiR | 64.05 | 64.71 | 66.74 | 64.97 | 64.89 | 67.85 | 65.19 | 65.99 | 67.57 | 67.77 |
| WiW | 69.96 | 70.25 | 69.76 | 70.10 | 69.90 | 68.26 | 67.92 | 69.05 | 69.08 | 67.91 |
| Yac | 0.50 | 0.65 | 0.56 | 0.56 | 0.38 | 0.43 | 0.43 | 0.43 | 0.43 | 0.43 |

* **Size:** template median 29–31 active nodes (of 31); IT median 72–77.5 (nodes of the printed
  program plus terms; not comparable with the template, RV16 nit 5). Every IT model uses all 8 term
  slots, as gate 2 showed.
* **Generations:** RT reaches E in the fewest generations (template 15, IT 11). IT-LTB and IT-NP need
  12; T-LTB 20.5, T-NP 35, the UNI arms 31.5–50.
* **CPU per run (median):** template 251–381 s, IT 76–114 s.

## Per dataset (descriptive, Holm over 90, no verdicts)

Pairs A lower / B lower and the ratio; Holm p only when it is below 1.

| Contrast | Air | Bos | Con | Dow | EnC | EnH | Tow | WiR | WiW | Yac |
|---|---|---|---|---|---|---|---|---|---|---|
| LT-B gain IT vs T | 2/8 1.073 | 2/8 1.049 | 0/10 1.085 | 2/8 1.079 | 4/6 1.006 | 7/3 0.940 | 0/10 1.049 | 2/8 1.012 | 4/6 1.002 | 3/7 1.064 |
| NP gain IT vs T | 2/8 1.051 | 4/6 1.045 | 0/10 1.140 | 4/6 1.055 | 3/7 1.046 | 7/3 0.958 | 2/8 1.056 | 1/9 1.020 | 1/9 1.006 | 1/9 1.133 |
| IT-LTB vs IT-RT | 9/0 0.993 | 10/0 0.956 | 9/1 0.983 | 4/6 1.003 | 7/3 0.986 | 9/1 0.898 | 5/5 1.003 | 10/0 0.993 | 10/0 0.996 | 0/0 1.000 |
| IT-NP vs IT-RT | 9/0 0.990 | 9/1 0.952 | 8/2 0.992 | 10/0 0.946 | 6/4 0.974 | 9/1 0.918 | 8/2 0.983 | 10/0 0.996 | 9/1 0.998 | 0/0 1.000 |
| IT-NP vs IT-LTB | 1/0 1.000 | 5/4 1.000 | 0/8 1.006 | 10/0 0.955 | 9/1 0.975 | 3/7 1.021 | 9/1 0.975 | 0/10 1.003 | 3/7 1.002 | 0/0 1.000 |

The template gains more from LT-B and from NP on 9 of 10 datasets; EnH is the exception (IT gains more).

**Evaluations to reach RT's final result** (`evals_to_reference.csv`; per dataset, the median final
training NMSE of RT on the same genome): T-LTB reached it in 84 of 100 runs, T-NP 78, T-LT 71, T-UNI
66; IT-NP 65, IT-LT 61, IT-UNI 60, IT-LTB 57 (RT itself 50 by definition).

## Linkage telemetry (agent analysis of `events.jsonl`)

Median over the runs of each arm; "inside the structure" is the share of the 20 position pairs with
the highest mutual information that lie in the same factor or term (IT) or are parent and child or
siblings (template), as the harness records it.

| Arm | Top MI pairs inside the structure | FOS Jaccard to the last generation | Improvements by 1-position subsets | By subsets of 4 or more | Slot dispersion median / max (last generation) | Distinct monomials (last generation) |
|---|---|---|---|---|---|---|
| T-LT | 0.30 | 0.49 | 0.37 | 0.50 | – | – |
| T-LTB | 0.25 | 0.62 | 0.47 | 0.33 | – | – |
| T-RT | – | 0.35 | 0.57 | 0.17 | – | – |
| T-NP | – | 1.00 | 0.43 | 0.30 | – | – |
| IT-LT | 0.34 | 0.51 | 0.48 | 0.38 | 1 / 1 | 8 |
| IT-LTB | **0.92** | 0.80 | 0.51 | 0.22 | 3 / 8 | 262 |
| IT-RT | – | 0.35 | 0.57 | 0.15 | 3.5 / 8 | 326 |
| IT-NP | – | 1.00 | 0.49 | 0.22 | 3.75 / 8 | 289.5 |

* **On IT, LT-B learns the genome's own structure.** 92 % of its top mutual-information pairs lie
  inside one factor or one term (95 % at the last generation), mostly the two input boxes of one term
  (which two inputs interact). Its model is stable (Jaccard 0.80). Node proximity gives that grouping
  without learning, so learning adds nothing over it at the end.
* **On the template, the learned pairs are mostly not local** (25 % parent and child or siblings),
  but LT-B and NP still end equal.
* **Plain mutual information (LT) collapses the population.** IT-LT ends with a median of 8 distinct
  monomials, one model in effect (slot dispersion 1), against 262–340 in the other IT arms; LT stalled
  or converged in 84 template runs and 78 IT runs.
* **The slot-swap problem is real.** In the IT arms other than LT, the same monomial sits in a median
  of 3–3.75 slots (up to 8) at the end of a run, so one position does not mean one term across the
  population.
* About half of the accepted improvements come from 1-position subsets in every structured arm.

## Runs with a test NMSE above 100

11 runs, all IT; the template has none.

| Dataset | Arm | Split | Test NMSE | Training NMSE | Note |
|---|---|---|---|---|---|
| WiR | IT-RT | 73 | 35,357 | **25,314** | the score fault (see "Faults") |
| Con | IT-NP | 76 | 7,929 | 15.25 | validation NMSE 7,564 |
| Con | IT-LT | 76 | 7,855 | 15.20 | validation NMSE 7,494 |
| WiW | IT-UNI | 75 | 254.6 | 65.65 | |
| WiW | IT-LTB | 75 | 232.3 | 65.64 | |
| WiW | IT-LT | 75 | 221.9 | 65.67 | |
| Con | IT-LT | 73 | 200.8 | 17.01 | |
| Con | IT-NP | 73 | 200.8 | 17.01 | the same model as IT-LT |
| Con | IT-RT | 73 | 189.7 | 17.11 | |
| Bos | IT-LT | 72 | 136.3 | 9.46 | |
| Con | IT-LTB | 76 | 116.2 | 15.25 | |

Apart from WiR split 73, these are models with negative
powers that extrapolate badly outside the training rows, as the pre-registration's limits predicted.
The primary metric (training NMSE) does not see them.

## Faults and deviations

1. **S0 on the server: `/tmp` was full.** pytest gave 8,906 passed, 1,416 failed, 98 errors and 13
   skipped. 1,404 failures and all the errors are "OSError: [Errno 28] No space left on device" in
   `/tmp/pytest-of-jovyan/`. One more failure is the known toolchain test (rounds 6–15). One test,
   `test_univariate.py::test_tournament_selection_draw_order_is_exact`, failed; it passes locally on
   the same commit (8 of 8 in its file) and passed in the full suite before the push, so it was most
   likely a side effect of the failures before it (not proved). The reproducibility check matched 4
   of 4. No code in the run path writes to `/tmp` (the harness, the runner, the report and the
   orchestrator write under `gomea-results`, which had 189 GB free), so the 1,000 runs were not
   affected. S0 is not a gate.
2. **A score fault in the IT search (harness defect).** The search scores a model with the
   least-squares identity SSE = Syy − explained (`pymoses/gomea/standalone/evaluator.py`), which
   loses all precision when the term columns are almost the same. In WiR IT-RT split 73 the model has
   x8, x8², 1/x8 and 1/x8², and x8 (wine density) is always close to 1, so the four columns are nearly
   equal. The search gave the model a training NMSE of 0.0; the exact value is 25,314. The false
   perfect score made it the elite, and it was reported. The harness records the difference
   (`crosscheck.search_rel_diff`, here 1.0) but does not fail on it (RV16 nit 2 noted this). 13 other
   runs have differences above 1e-9, all at most 2.7e-4. **Effect:** one IT-RT pair of 100 in P1 and
   P2, in favour of the IT linkage models. Without WiR split 73 (agent sensitivity check, with the
   report's own statistics code): P1 LT-B 1.037 (25/74), NP 1.031 (24/75), UNI and LT 1.008–1.009
   (Holm p 0.017); P2 0.990–0.994 and NP vs LT-B 1.000. No verdict changes, and the P1 direction does
   not depend on that run.
3. **The stop rule ended 286 runs before E** (see "Run health"); this is the approved amendment.
4. **The round took half of the estimated time** (11.1 h against about 22 h): the local pilot
   overestimated the server's CPU per evaluation by about 2×.

## Observations (agent reading)

1. **The fixed IT genome does not make learned linkage more useful.** It makes the linkage easy to
   learn (LT-B finds the term structure), but the structure is already known, so a fixed structural
   model (NP) gives the same result, and faster.
2. **Why IT gains so little from any linkage model (agent hypotheses, not tested):** per-term least
   squares refits all weights for any set of terms, so the only dependency left is which inputs pair
   inside a term, a 2-position dependency that the RT subsets also catch often; the slot swap blurs
   what a position means across the population; and IT runs reach E in about 12 generations, so the
   model has little time to matter.
3. **The template does gain from linkage, and structure is enough there too.** T-LTB and T-NP both
   gain 4.4 %, in 87–88 of 100 pairs. Koch et al. (GECCO 2026) found node proximity better than the
   mutual-information models; we find it equal at the end, at zero learning cost.
4. **What this means for MOSES.** The hope behind phase B was a learned linkage model that lasts the
   whole MOSES run. This round, run in the most favourable case (one genome for the whole run, no
   MOSES loop, N = 1,000), shows that learned linkage gives no more than a fixed structural model.
   pyMOSES already has a structural model (`--gomea-fos np`; TLS-NP, next in line after BASE-TLS).
   So the linkage argument for phase B and for T3.4 (IT inside pyMOSES) is weak. The accuracy gap
   between the genomes (S2) comes mostly from per-term least squares, which BASE-TLS already uses.

## Limits

* No MOSES loop: the result is about GOMEA on a fixed genome.
* One size per genome (31 and 32 positions), no constants, one budget, one population size.
* Each genome has its own scorer, so S2 is not a linkage result.
* The stop rule ended 286 runs before E, so the arms did not use the same number of evaluations.
* The score fault of fault 2 can also give smaller false scores during a search; only the final
  model is checked (14 runs with a difference above 1e-9). The sensitivity check covers the final
  models only.
* Training NMSE is the primary metric; the test NMSE of IT is hurt by poles from negative powers.

## User decisions

* None yet. Open, for the user: whether to continue toward phase B (`IT-LINKAGE-STUDY-PLAN.md`,
  section 11) and T3.4; the agent ideas of `BACKLOG.md` section 14.
