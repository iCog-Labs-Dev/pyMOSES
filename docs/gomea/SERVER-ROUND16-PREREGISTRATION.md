# Server round 16: pre-registration (2026-11-23)

**Status:** draft. The user approved the frame on 2026-11-23 ("Yes I agree lets run it on the
server", `APPROVALS.md`). The agent wrote the details below. The user approves this text before
the run. Not yet implemented, not yet run.

**The user's question (2026-11-23):** "Can we test if increasing the number of evaluation for
everyone has a better effect in the long term? [...] I do want to know how much better the result
is going to be in the end and make sure we cover all of our bases so that we can understand this
better."

**Origin of the ideas:** the budget test is the user's idea (after the T3.2 spike 2 result). The
RV method is the RV-GOMEA core of Bouter et al. (GECCO 2017) with the intron-aware rules and the
convergence rule of GP-RV-GOMEA (Koch, Alderliesten, Bosman, PPSN 2024), as in
`T3.2-FEASIBILITY.md` (appendix). The MOSES controls (absorbed-gene skip, one sample per distinct
genotype, free codes only when kept), the arms, the budgets, the data sets, the contrasts, the
seed fix and the pilot gate are agent-written.

**Defaults that the user did not choose:** the agent proposed 7 data sets (or all 10) and asked
whether TLS-LMR should run at 100,000 evaluations. The user did not answer these two points. This
draft uses the proposal as written: 7 data sets and no TLS-LMR. The user can change both before
the run.

**Questions:**

1. At 100,000 evaluations, is real-valued GOMEA on the constants (option C) better than BASE-TLS?
2. Does the ratio of the RV arm to BASE-TLS fall when the budget grows from 20,000 to 100,000?
3. How much does BASE-TLS itself gain from 5 times the budget?

BASE-TLS (20,000 evaluations) stays the reference; TLS-NP stays next in line.

## Background (`T3.2-FEASIBILITY.md`, section "Spike 2 result")

* Spike 2 (the RV-GOMEA core in option C; old splits s1–s3 of Con, EnH, Yac, Dow, Tow; training
  NMSE; 20,000 evaluations) failed its gate in all three arms: `rvg` median ratio 1.234 (4 of 15
  lower), `rvg-cap` 1.285 (2 of 15), `rvg-late` 1.027 (6 of 15).
* The RV arms had fewer demes (41 for the reference, 28–32 for the RV arms). The second deme of
  `rvg` started at a median of 9,575 evaluations, against 4,787 for the reference.
* Best-so-far training error: the RV arms are level with the reference up to 7,500 evaluations
  and about 25 % behind from 10,000 on. In the last quarter (15,000 to 20,000), `rvg` improved a
  little faster than the reference (0.803 against 0.822 of the 15,000 value). This is a weak sign
  (post hoc, agent analysis), and it is the reason for this round.
* A confound in the spike: with the flag on, the code drew one number from the global random
  generator, so each RV run started from another initial population than its `off` run. This
  round fixes it (see "The seed fix").

## Rules

* As in rounds 1 to 15: on the server, one phase at a time, W from `resources.py`, state in
  `server-run16/`. Round 16 never overlaps with another round. One round if the pilot estimate is
  at most 20 h; else the agent splits it into two plans that chain, so that the second starts by
  itself when the first ends (the user's rule). The server round runs in one go; there is no
  separate server smoke step.
* **One phase (1bk, regression):** 7 data sets (the paper's five, Air, Con, EnC, EnH, Yac, plus
  Tow and Dow), **fresh splits 81–90**, a budget of 20,000 or 100,000 evaluations for each run
  (`-m`), no time limit, a wall cap of 36,000 s as a guard only.
* Splits 81–90 need `SEEDS = range(1, 91)` in `tools/gomea/sr_datasets.py`, the entries s81–s90 in
  `data/gomea-sr/<Abbr>/splits.json` of all 10 data sets, 720 split files in `setup.sh`, and the
  requirement `datasets:Air,Con,Dow,EnC,EnH,Tow,Yac:81-90`. Splits 1–80 stay byte-identical. The
  build runs `check_split_types.py` on the new splits.
* **Statistics:** as in rounds 12 to 15. Failed runs count as infinitely bad. Paired Wilcoxon
  signed-rank on log(A / B), two-sided, Holm in each family, α = 0.025. "Better" or "worse" also
  needs a ratio at least 10 % from 1. The interaction contrast is the one of round 15
  (`linkage_report.py`): d = log(A1 / B1) − log(A2 / B2) for each pair; "A1 gains more" if
  Holm p < 0.025 and exp(median d) ≤ 0.9; "A2 gains more" if Holm p < 0.025 and
  exp(median d) ≥ 1/0.9; else "tie".
* **Program size:** as in rounds 13 and 14 (the node count plus `n_terms` with the per-term fit).
* **S0:** full pytest and the reproducibility check. S0 is not a gate.

## The seed fix (before the round)

* The RV stream gets its seed from the run seed (`-r`) and the deme index. It never draws from
  the global random generator. So, with the flag on, everything up to the first RV generation is
  the same as with the flag off.
* Check: a test, and the pilot (the first deme's initial population of each RV run is identical
  to that of the BASE-TLS run with the same split, seed and budget).

## Arms (420 MOSES runs)

The RV option is opt-in: without it, the output of every existing setting stays byte-identical
(identity tests).

| Arm | Budget | Flags |
|---|---|---|
| **BASE-TLS** | 20,000 | the round 14 BASE-TLS flags |
| **BASE-TLS-100K** | 100,000 | the same |
| **TLS-RV** | 20,000 | BASE-TLS with `--gomea-rv-step rvg --gomea-rv-budget balance` |
| **TLS-RV-100K** | 100,000 | the same |
| **TLS-RVL** | 20,000 | BASE-TLS with `--gomea-rv-step rvg --gomea-rv-budget late` |
| **TLS-RVL-100K** | 100,000 | the same |

* `balance`: the paper's rule; after each GOM generation an RV generation runs with probability
  p_RV = 1 − 0.5 · (RV evaluations / GOM evaluations) of the deme.
* `late`: an RV generation runs only after a GOM generation that did not improve the deme's best,
  then with the same p_RV rule.
* The RV method is the spike 2 method (`T3.2-FEASIBILITY.md`, appendix, and the 14 departures in
  the spike 2 record) with the seed fix: a univariate model over the active genes (non-zero, not
  absorbed by the per-term fit); the best 35 %; AMS and the second AMS round; acceptance of a strict
  improvement, else with probability 0.05; SDR-AVS; forced improvements at NIS_MAX = 20; one
  elitist; structure-only convergence (the same zero/non-zero pattern, or no new evaluation in a
  full generation). Every RV evaluation counts in the budget.
* `rvg-cap` of spike 2 is left out (it was the worst arm).

## Contrasts (70 pairs each: 7 data sets × 10 splits)

* **Primary (4), test NMSE, Holm across the 4:**
  1. TLS-RV-100K vs BASE-TLS-100K (Q1);
  2. TLS-RVL-100K vs BASE-TLS-100K (Q1);
  3. interaction [TLS-RV-100K, BASE-TLS-100K, TLS-RV, BASE-TLS] (Q2; "A1 gains more" = the RV
     ratio to BASE-TLS is at least 10 % lower at 100,000 than at 20,000);
  4. interaction [TLS-RVL-100K, BASE-TLS-100K, TLS-RVL, BASE-TLS] (Q2).
* **Secondary (8), one Holm family:** TLS-RV vs BASE-TLS and TLS-RVL vs BASE-TLS at 20,000 (test
  NMSE; spike 2 on fresh splits); BASE-TLS-100K vs BASE-TLS (Q3, test NMSE); TLS-RV-100K vs TLS-RV
  (test NMSE); the 4 primary contrasts on the training NMSE.
* **Size and CPU (8), one Holm family:** program size and CPU for primary 1 and 2 and for the two
  20,000 contrasts.
* **Descriptive (no verdicts):** the 4 primary contrasts per data set; the primary 1 and 2 on the
  clipped test NMSE; BASE-TLS-100K vs BASE-TLS on the training NMSE.
* **Descriptive tables:** the best-so-far training NMSE of each run at 1, 2, 5, 10, 20, 50 and
  100 % of its budget (median ratio of each arm to BASE-TLS of the same budget, pooled and per data
  set); demes per run and the evaluation at which the second deme starts; the RV counters (RV
  evaluations and share, improving, kept by p_accept, AMS, forced improvements, skips); the
  train–test gap; the runs with a test NMSE above 100.

## What the agent proposes after the round (the user decides)

* **Primary 1 or 2 "A better":** RV helps at 100,000 evaluations. Proposal: RV as a long-budget
  option; option A becomes worth a test; a round at a larger budget.
* **Primary 1 and 2 tie, but primary 3 or 4 "A1 gains more":** RV gains with the budget but does
  not pass BASE-TLS at 100,000. Proposal: one round at 300,000 or more evaluations on the faster
  data sets.
* **Otherwise:** no sign that the budget helps RV. Proposal: stop T3.2 (options C and A).
* Each verdict is reported with the size, CPU and train–test gap (the user's accuracy-size rule).

## Local pilot before the server (a gate, not a selection)

* On the merged code: Con and Yac, old split 1, all 6 arms; Tow, old split 1, BASE-TLS-100K (the
  memory and time probe). Do not read the NMSE.
* **Gate:**
  1. every run ends ok;
  2. the first deme's initial population of each RV run is identical to that of its BASE-TLS run;
  3. the peak RSS of each run is at most 50 % of its limit; if a 100,000 run is above, the 100,000
     arms get a manifest of their own with a larger limit (the agent reports the new W and time);
  4. the RV share of the evaluations of each RV run is at most 50 %;
  5. the evaluation count of each run is recorded (a run that ends before its budget is reported).
* If a check fails, the agent reports to the user before the server run.

## Memory

As in round 14: Dow 8,192 MB, the other data sets 5,120 MB, unless the pilot (gate item 3) needs
more for the 100,000 arms.

## Implementation (by subagents, in separate clones)

| Clone | Content |
|---|---|
| K | The RV core under `pymoses` (from the spike 2 commit `be83e08`), with the seed fix; identity reference first; opt-in; tests; USAGE |
| L | Splits 81–90; the runner (`rv_summary`, the best-so-far checkpoints); `configs/gomea/phase1bk/`, `plan16.json`, the smoke plan, `test_plan16.py`, the SERVER-RUN section; the dry run; after K merges, the local smoke run and the pilot |

* Then: merge, the full test suite, the RV16 review, and the push only with a fresh approval.

## Size and time

* CPU per 100 runs in round 14 (20,000 evaluations, 10 data sets): BASE-TLS 18.7 h. In spike 2
  the RV arms used about 0.6 times the CPU of the reference.
* Estimate: BASE-TLS 13 h, TLS-RV and TLS-RVL 8 h each, BASE-TLS-100K about 65 h, TLS-RV-100K and
  TLS-RVL-100K about 39 h each: about 170 CPU-h, about 16 h at W = 11, plus S0 (about 1.3 h).
* **About 15–25 hours in total.** The cost of a 100,000 run is not known; the pilot gives a better
  estimate.

## Limits

* Two budgets only (20,000 and 100,000); the paper used 10,000,000.
* 7 of the 10 data sets; no TLS-LMR (see "Defaults").
* Option C only; option A is not tested.
* The RV parameters are the paper's values (τ = 0.35, p_accept = 0.05, NIS_MAX = 20); no tuning.
