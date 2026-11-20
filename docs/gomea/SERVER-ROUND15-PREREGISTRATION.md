# Server round 15: pre-registration (2026-11-20)

**Status:** a draft. Not approved. Built (2026-11-20) at a617702: clone K 99392fa, 2ce26b0,
3b18ca6 (the harness and the runner arm); clone G d4c257f (`--max-generations`); clone L e09f779,
d077826, b978642, d5187f0, 3c5bc87, 70bbf79, 01f3827, a617702 (splits 71–80, the report, the
configs, the two plans, the stop rule, the re-pin). The pilot is done (no NMSE of split 71 read).
Gate 1 is still running. Nothing of splits 71–80 is read before the user approves this text.

**Plan:** `IT-LINKAGE-STUDY-PLAN.md` (the implementation plan; the user accepted its decisions
D1–D7 as recommended on 2026-11-20). Backlog item: `BACKLOG.md` 13.9. Background:
`T3.4-FEASIBILITY.md`.

**The user's frame (2026-11-20, `APPROVALS.md`):**

* The reason: the linkage research question ("No I think I like the research question").
* The order: "first explore IT with GP-GOMEA before using it in our pyMOSES to reduce noise."
* The code: "Option b." (the GOMEA engine ported in `pymoses/gomea`, with a stand-alone harness).
* The design: "replace the GOMEA's template with IT to see the linkage learning it uses has an
  effect, keeping this switch we test it with different type of linkage learning, the UNI, RT,
  LT, LT-B and the structural [...] and if we don't have the numbers for the GOMEA's performance
  with the template used alone for those different tests, we also need to run that experiment as
  well. [...] The objective is to see if the IT approach is indeed better for the linkage learning
  than the template alone and find linkage models that might have a better result when used with
  IT as opposed to the current version of the linkage model."

The 2 × 5 design is the user's. The genome sizes, the start population of IT, the metrics, the
contrasts, the gates and the sizes are agent proposals that the user accepted as D1–D7. The rest
of this text is agent-written.

**Origin:** the IT form is from a paper (de França and Aldeia, Evolutionary Computation 2021). The
template genome is GP-GOMEA's (Virgolin et al.). The research question came from the agent's T3.4
discussion with the user; the user chose it. Two web searches (2026-11-20) found no paper that
combines the IT form with GOMEA or linkage learning (not a full literature check).

## Questions

| # | Question | Answered by |
|---|---|---|
| Q1 | Does learned linkage help more on the IT genome than on GP-GOMEA's template? | family P1 (the genome × linkage interaction) |
| Q2 | Which linkage model works best with IT? | family P2 |
| Q3 | Does our engine reproduce the known template result (learned linkage beats a random tree on training error)? | the replication gate (before the server) and family S1 |

Q4 (a global, run-long linkage model for MOSES) is phase B. It is not in this round.

## Background (plan, section 2)

* Phase 1F part A (the authors' C++ GP-GOMEA, template height 4, N 2,000, 20 generations): LT-MIb
  (our LT-B) beat RT on training NMSE on 10 of 10 datasets, on test NMSE on 4 of 10.
* Koch, Alderliesten and Bosman (GECCO 2026): P(learned better than random) = 0.57; P(node
  proximity better than the MI variants and random) ≈ 0.80.
* pyMOSES (1g, 1h, 1m, 1al, 1ay): inside MOSES, where the knobs change in every deme, LT-B ties RT
  in every real run. It beat RT only on one frozen representation with N = 1,024 (1h).
* No numbers exist for the template under our engine, for UNI and node proximity on the template
  with our datasets, or for any linkage model on an IT genome.

## Rules

* As in rounds 1 to 14: on the server, one phase at a time, W from `resources.py`, a state
  directory of its own. Round 15 never overlaps with another round. If the pilot shows more than
  20 h, the round is two plans that chain: the second starts by itself when the first ends (the
  user's rule).
* **No MOSES loop.** Each run is one GOMEA population on a fixed genome
  (`python -m pymoses.gomea.standalone`), started by the runner's arm key `"standalone"`.
* **Datasets:** Air, Bos, Con, Dow, EnC, EnH, Tow, WiR, WiW, Yac (D2: all 10, because the pilot fits
  two chained rounds). **Fresh splits 71–80.** Splits 1–70 stay byte-identical. The column types of
  training, validation and test agree on all 100 new splits (`tools/gomea/check_split_types.py`).
* **Two plans that chain:** `round15` runs S0 and the 5 datasets of the D2 fallback set (Air, Con,
  Dow, Tow, EnH; 500 runs), then starts `round15b` by itself, which runs the other 5 datasets (Bos,
  EnC, WiR, WiW, Yac; 500 runs) into the same result directory. The report covers all 1,000 runs. If
  `round15b` cannot run, `round15` alone gives the complete fallback design.
* **Population and budget (D1):** N = 1,000 and E = 500,000 evaluations per run (the pilot fits
  them). A score-cache hit is not an evaluation. No time limit; a wall cap of 36,000 s as a guard
  only.
* **Stop rule.** A run ends at the first of: (a) E evaluations; (b) convergence (all genotypes
  equal); (c) **10 generations in a row with no new evaluation** (end reason `stalled`); (d) **300
  generations** (end reason `max_generations`). Every run reports its end reason.
  **Amendment from the pilot (agent proposal, 2026-11-20; the plan had only (a) and (b)):** 3 of
  the 20 pilot runs never ended. Their populations stopped finding new models, but accepted
  neutral changes of inactive positions kept the genotypes different, so (b) never held, and cache
  hits do not count, so (a) never held. Con T-LT and Con IT-LT made no new evaluation after
  generation 20 and 18 and ran on for more than 300 generations; Dow T-NP added about 27
  evaluations per generation for 800 generations. A run that makes no new evaluation cannot find a
  better model, so (c) loses nothing. The 17 runs that ended used E within 10–70 generations, so
  (d) is a guard for runs that find new models very slowly. The pilot read no NMSE.
* **Statistics (D5):** as in rounds 12 to 14. A failed run counts as infinitely bad. Paired
  Wilcoxon signed-rank on log ratios over the (dataset, split) pairs, two-sided, Holm, α = 0.025.
  "Better" or "worse" also needs a ratio at least 10 % from 1. Every contrast also reports its
  direction (pairs A lower / B lower) and its Holm p, because linkage effects on training NMSE
  were small in Phase 1F.
* **The reported model** of a run is the best training score of the final population. Its
  validation NMSE is reported too. No selection on validation rows (an optimization study).
* **S0:** full pytest and the reproducibility check. S0 is not a gate.

## Arms (10)

| Arm | Genome | Linkage model (`LinkageModel` mode) |
|---|---|---|
| T-UNI | template | `uni` (univariate) |
| T-RT | template | `rt` (random tree) |
| T-LT | template | `lt-mi` (linkage tree on mutual information) |
| T-LTB | template | `lt-b` (linkage tree on bias-adjusted mutual information) |
| T-NP | template | `np` (node proximity on the template tree) |
| IT-UNI | IT | `uni` |
| IT-RT | IT | `rt` |
| IT-LT | IT | `lt-mi` |
| IT-LTB | IT | `lt-b` |
| IT-NP | IT | `np` (proximity on the IT tree: box, factor, term) |

**Fixed settings (D3, D4, D7; plan 3.2 and 5):**

| Setting | Template | IT |
|---|---|---|
| Genome | full binary tree of height 4: 31 positions; internal: `+ - * aq` or an input; leaves: an input | 8 term slots × 2 factors × (input box, power box) = 32 positions; input "none" or 1…n; power −2, −1, 1, 2 |
| Constants | none | none (least squares gives the weights) |
| Output scaling | linear scaling a + b·f on the training rows | per-term least squares w0 + Σ wi·termi, aliased columns dropped |
| Start population | GP-GOMEA's half-and-half at height 4 | plan 5.6 (slot active with probability 0.5; factor input with probability 0.5) |
| Acceptance, inactive changes | `nonworse`; a change that leaves the model unchanged is not evaluated | the same |
| Forced improvements | none: GP-GOMEA 750220c has none in `src/GOMEA/` (GOM, an optional coefficient mutation, elitism 0 by default; clone K's code check) | the same |

## Metrics

| Metric | Role |
|---|---|
| Training NMSE of the reported model at the end | **primary** (D6) |
| Test NMSE and clipped test NMSE | secondary |
| Anytime area: exp(A / W), where A is the trapezoid area under ln(best-so-far training NMSE) over log10 of the evaluation fraction at the checkpoints 1, 2, 5, 10, 20, 50 and 100 % of E, and W = 2 (the width of that axis); a weighted geometric mean of the best-so-far curve, in NMSE units. A run that ends before a checkpoint keeps its final best | secondary |
| Evaluations to reach the median final training NMSE of RT on the same genome and dataset | secondary |
| Size, number of terms (IT), CPU, generations | descriptive |

## Contrasts

* **P1, the interaction (Q1; 4 contrasts, primary, Holm across the 4):** for each linkage model f
  in {UNI, LT, LT-B, NP}, the per-pair value d = log(IT-f / IT-RT) − log(T-f / T-RT) on training
  NMSE. d < 0 means that f gains more over the random tree on IT than on the template. Ratio =
  exp(median d). "IT gains more" if Holm p < 0.025 and the ratio is at most 0.9; "the template
  gains more" if Holm p < 0.025 and the ratio is at least 1/0.9; else "tie".
* **P2, the linkage models on IT (Q2; 5 contrasts, primary, Holm across the 5):** IT-UNI, IT-LT,
  IT-LTB and IT-NP each against IT-RT; IT-NP against IT-LTB (structure against learned).
* **S1, the same 5 on the template (Q3; secondary, its own Holm family):** T-UNI, T-LT, T-LTB, T-NP
  each against T-RT; T-NP against T-LTB.
* **S2 (secondary, its own Holm family):** IT-LTB against T-LTB and IT-NP against T-NP, on test
  NMSE (the absolute level of the two genomes; confounded by the scorer, D4).
* **S3 (descriptive):** P1 and P2 on test NMSE and on the anytime area (18 rows).
* **Per dataset (descriptive, no verdicts):** P1 and P2 on each dataset (90 rows).
* Descriptive rows keep their raw p and the Holm p of their own family, with no verdict.
* The report code: `tools/gomea/linkage_report.py` (spec keys `contrast_families` and
  `evals_to_reference`; `configs/gomea/phase1bj/spec.json`).

## Reported (plan 6.2 and 6.3)

Training, validation, test and clipped test NMSE; size; nodes; terms (IT); evaluations; CPU;
generations; the end reason; cache hits; peak memory. Per generation: best and median training
NMSE; the FOS size, mean subset size and its Jaccard similarity to the previous generation; the
top 20 position pairs by mutual information and their structural alignment (LT, LT-B); the
accepted improvements by subset size; the share of active positions; for IT the slot dispersion
and the count of negative powers made positive.

## Gates before the server (plan 8.3; a gate, not a selection)

1. **Template replication:** Con, EnH and Tow × 5 splits from 1–30, template, N = 2,000, 20
   generations (the Phase 1F part A settings): T-LTB must have a lower median training NMSE than
   T-RT on each of the 3 datasets. If not, the study stops and the agent reports to the user.
2. **IT recovery:** a synthetic table (5 inputs, 500 rows, y = 3·x1·x2 + 5·x3 + small noise):
   IT-LTB finds the true terms in at least 4 of 5 seeds within 50,000 evaluations.
3. **Cross-check:** every pilot run passes the `sr_eval` cross-check (the two NMSE values agree to
   a relative 1e-9).
4. **Resources:** the peak memory of each arm is at most 50 % of its manifest limit.

**Results:**

* Gate 1: running (clone G; about 13 h on the shared local machine). The server run waits for it.
* Gate 2 (5 seeds): in all 5, the model contains x1·x2 (weight 2.98–3.01) and x3 (4.96–5.00), and
  also 6 extra terms; every model fills all 8 slots. By the plan's wording ("finds the true terms"):
  5 of 5, **pass**. By a strict reading ("exactly these two terms", the agent's brief to clone G):
  0 of 5. By the combined effect of the extra terms (all together below 1 % of sd(y), the noise
  level): 5 of 5. In 3 seeds some extra terms have large weights that cancel. Reason: least squares
  never loses training accuracy with one more term, and nothing penalizes size, so the search fills
  every slot. This is a property of the IT genome in this study, not a fault.
* Gate 3: passed. All 20 pilot runs pass the cross-check: the 17 runs that ended at first, and the
  3 runs that did not end, run again with the stop rule (Con T-LT `stalled` after 31 generations
  and 266,173 evaluations; Con IT-LT `stalled` after 29 and 249,829; Dow T-NP `max_generations`
  after 300 and 345,971). The stop rule would not have ended any of the 17 runs earlier: their
  longest stretch with no new evaluation was 0 generations in 16 runs and 2 in one, and the most
  generations was 70.
* Gate 4: passed. The highest peak was 371 MB (IT-RT on Dow), 18.1 % of the 2,048 MB limit; the
  template arms peaked at 108–128 MB.

## Memory

`mem_limit_mb` 2,048 for every run (the pilot peak is 18.1 % of it). W is limited by the CPUs.

## Implementation (by subagents, in separate clones)

| Clone | Content | Status |
|---|---|---|
| K | The stand-alone harness (`pymoses/gomea/standalone/`), the runner arm key `"standalone"`, the tests of plan 8.1 and 8.2 | Done |
| G | `--max-generations`; gates 1 and 2 | Option done; gate 2 done; gate 1 running |
| L | Splits 71–80; the report additions (`train_nmse`, `anytime_area`, interaction contrasts, Holm families, evaluations to RT; opt-in; `tools/gomea/linkage_report.py`); the configs of phase 1bj; the two plans; the pilot; gates 3 and 4; the stop rule (`--stall-generations`) and its runner keys; the re-pin of the 14 Boolean manifests; the dry run (1,000 jobs) and the local smoke run of the chain | Done |

* Then: merge, the full test suite, the RV16 review, and the push only with a fresh approval.

## Size and time

From the pilot (the real CPU of the 20 pilot runs, per training row, the mean of Con and Dow for
each arm, scaled by the training rows of each dataset; the machine load was 14–45 on 22 logical
CPUs, so the CPU values are on the high side):

| Plan | Datasets | Runs | CPU hours | Hours at W = 11 |
|---|---|---|---|---|
| `round15` | Air, Con, Dow, EnH, Tow | 500 | about 121 | about 11, plus S0 about 1.3 |
| `round15b` | Bos, EnC, WiR, WiW, Yac | 500 | about 104 | about 9.5 |
| Total | 10 | 1,000 | about 226 (template 174, IT 52) | about 22 |

* The server's speed per thread is about that of the local machine: round 14's S0 pytest took
  1,514 s on the server for 10,217 tests, and the local full suite takes about 20–35 minutes. On a
  server 2× slower the two plans take about 24 h and 19 h.
* With the higher of the two pilot datasets for each arm, the total is about 271 CPU hours.
* The IT column cache makes IT the cheap genome (0.24–0.75 ms per evaluation in the pilot); the
  template costs 1.3–1.95 ms.

## Limits

* No MOSES loop: the result says what linkage learning does on a fixed genome, not inside MOSES.
* One size for each genome (31 and 32 positions); no constants on either side.
* Each genome has its own scorer (D4). P1 compares gains within each genome, so the scorer
  difference affects it less than S2.
* The primary metric is training NMSE (an optimization effect); test NMSE is secondary.
* The term slots of IT can hold the same monomial in different slots (the slot-swap problem); the
  telemetry measures it, but the round does not fix it.
* IT allows the powers −2 and −1. The zero rule keeps every denominator non-zero on the training
  rows only, so an IT model can extrapolate badly on the test rows (a local check run on Con split
  1, not part of the study: training NMSE 17.0, test NMSE 275, clipped 37.0). The primary metric is
  the training NMSE; the clipped test NMSE is reported next to the test NMSE.
