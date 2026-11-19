# Server round 14: pre-registration (2026-11-19)

**Status:** draft, agent-written. It waits for the user's approval. Not implemented.

**The user's frame (2026-11-19, `APPROVALS.md`):**

* Reference: "I want to make BASE-TLS as a default for now. But the TLS-NP needs to follow it
  next."
* Scope: "Okay let's go with the fix and run all the smaller ideas in more than on round if we
  have to but the runs if they need to be more than one round, they need to run sequentially."
* The user's question: "Have we tested BASE-NP with LM? I want to see the effect there as well"
  (backlog 12.10).

Everything else below (arms, option rules, thresholds, contrasts, the pilot gate, the time
estimate) is agent-written.

**Questions:**

1. Does LM on the inner constants of each term (T3.1) help when LM gets a budget (the LM reserve)?
2. What does LM add under node-proximity linkage, with plain scaling and with per-term least
   squares?
3. Does a relative pole rule keep the good LM tunings that the round 13 rule rejected?
4. Does TLS-NP stay within 10 % of BASE-TLS on new splits?
5. How much of the BASE-TLS accuracy does a plain linear model (OLS, ridge) reach without a search?

BASE-TLS is the reference; TLS-NP is next in line; HC-LM is the accuracy baseline.

## Background (`SERVER-ROUND13-FINDINGS.md`, `BACKLOG.md` section 12)

* BASE-TLS was "A better" than BASE twice (0.879, 0.784). On Dow its programs are in effect linear
  regressions on selected inputs (26 terms, 96 % single inputs).
* TLS-LM (T3.1) tied with BASE-TLS: 99 of 239 LM calls had no budget left at the deme end, and 80
  pairs were identical (12.4). The LM reserve (`--const-opt-reserve`, round 11) stops the deme
  search R evaluations before the budget ends, so that LM can run; with `--linear-scaling terms` it
  is allowed (USAGE, T3.1 row).
* BASE-NP (round 12) had LM-VAL and tied with BASE (1.058) at 0.744× the CPU; its LM share was
  0.16 %. TLS-NP (round 13) had no LM. No round measured the LM effect under NP (12.10).
* The round 13 pole check rejected 19 of 171 tunings; BASE-PS was worse in 12 of the 13 pairs
  that differ. In 14 of the 19 rejections the old program already had S > 10R (12.3). The check
  is refused with `--linear-scaling terms`.
* ε-lexicase (12.5) is left out: about one exemplar selection per run at this budget. A hard S0
  gate (12.8) and the telemetry items (12.9) are not experiments; they stay in the backlog.

## Before the round: the fix (approved, in progress)

* Backlog 12.1: the guard loads its validation table with the training table's column types (Dow
  splits 11 and 56 crashed). Backlog 12.2: the four RV16 nits of round 13. Clone H, with identity
  tests. Round 14 runs on the fixed code.

## Rules

* As in rounds 1 to 13: on the server, one phase at a time, W from `resources.py`, state in
  `server-run14/`. Round 14 never overlaps with another round. **One round is enough** (about
  15 h); if the pilot shows that it needs more than 20 h, the agent splits it into two plans that
  chain, so that the second starts by itself when the first ends (the user's rule).
* **One phase (1bi, regression):** 10 datasets, **fresh splits 61–70**, 20,000 evaluations for
  each run (`-m 20000`), no time limit, a wall cap of 36,000 s as a guard only. LM passes and
  check passes count.
* Splits 61–70 need `SEEDS = range(1, 71)` in `tools/gomea/sr_datasets.py`, 560 split files in
  `setup.sh`, and the requirement `datasets:...:61-70`. Splits 1–60 stay byte-identical. The
  build checks that the training and validation column types agree on every new split.
* **Statistics:** as in rounds 12 and 13. Failed runs count as infinitely bad. Paired Wilcoxon
  signed-rank on log(A / B), two-sided, Holm, α = 0.025. "Better" or "worse" also needs a ratio at
  least 10 % from 1.
* **Reference rule (agent-written):** if an arm with `--linear-scaling terms` is "A better" than
  BASE-TLS, the agent proposes it as the new reference. The user decides. Otherwise BASE-TLS stays
  the reference and TLS-NP stays next in line. All other verdicts are descriptive.
* **Program size:** as in round 13: the node count, plus `n_terms` when the final model uses the
  per-term fit. A linear baseline with k inputs counts as the TLS program `+(x1 … xk)`: k + 1 nodes
  plus k terms, so 2k + 1. The node count is also reported.
* **Reported:** test NMSE, the clipped test NMSE, size, nodes, train–test gap, CPU, evaluations,
  demes, knobs per deme, total knobs, LM calls (improved, no budget, no constants, singular, guard
  rejections), LM share, pole checks and rejections, the runs with a test NMSE above 100, and the
  per-run table (`runs.csv`).
* **S0:** full pytest and the reproducibility check. S0 is not a gate (`BACKLOG.md` 12.8).

## Arms (800 MOSES runs + 200 linear fits)

Each new option is opt-in: without it, the output of every existing setting stays byte-identical
(identity tests).

| Arm | Question | Flags |
|---|---|---|
| **BASE-TLS** | reference | the round 13 BASE-TLS flags |
| **TLS-NP** | next in line (Q4) | the round 13 TLS-NP flags |
| **HC-LM** | accuracy baseline | the round 13 HC-LM flags |
| **TLS-LMR** | T3.1 with an LM budget (Q1) | BASE-TLS with `--const-opt lm --const-opt-guard val --const-opt-reserve 500` |
| **TLSNP-LMR** | LM under NP, per-term (Q2) | TLS-NP with `--const-opt lm --const-opt-guard val --const-opt-reserve 500` |
| **NP-LMR** | LM under NP, plain (Q2) | the round 12 BASE-NP flags (`--gomea-fos np --linear-scaling 1 --const-opt lm --const-opt-guard val`) with `--const-opt-reserve 500` |
| **NP-NOLM** | LM under NP, plain (Q2) | NP-LMR with `--const-opt none` and no guard, no reserve |
| **NP-LMR-PSR** | relative pole rule (Q3) | NP-LMR with `--const-opt-pole-check box-rel` (new) |
| **LIN-OLS** | linear baseline (Q5) | no MOSES run: ordinary least squares on all inputs (new, runner only) |
| **LIN-RIDGE** | linear baseline (Q5) | no MOSES run: ridge regression on all inputs, λ chosen on the validation rows (new, runner only) |

### TLS-LMR, TLSNP-LMR, NP-LMR: LM with a budget

* The reserve is 500 evaluations, as for BASE-R (rounds 11 and 12). LM runs at the end of each deme
  on the deme's best program, with the validation guard.
* **Agent choice:** the same reserve for all three LM arms, so that NP-LMR against NP-NOLM measures
  LM with a budget, not the budget accident of round 13.

### NP-LMR-PSR: the relative pole rule

* `--const-opt-pole-check box-rel` (new): the probes, the charges and the non-finite rule of `box`
  (round 13). The size rule changes: reject the tuned program if
  S(tuned) > 10 · max(S(old), R). So a tuning is rejected only when its spread grows to more than
  10 times the larger of the old spread and the target range.
* It is on NP-LMR because the check is refused with `--linear-scaling terms` and NP-LMR is the
  plain-scaling LM arm of this round.
* **Agent choices:** the rule form and the factor 10. "Probes near the data" (the other idea of
  12.3) is not tested.

### LIN-OLS and LIN-RIDGE: linear baselines

* Fitted by the runner on the training rows of each split, with no MOSES run. All inputs are used;
  a boolean input (Bos `x4`) counts as 0/1. The target is scaled as for the MOSES arms; the
  report scores them as the other arms.
* Ridge: λ from the grid 10^k, k = −6 … 3, on standardized inputs; the λ with the lowest validation
  NMSE is used, then refitted on the training rows. The intercept is not penalized.
* Each split takes a few seconds. Their CPU is reported but not compared.

## Contrasts

* **Primary (5), pooled over 100 pairs, test NMSE, Holm across the 5:**
  1. TLS-LMR vs BASE-TLS (Q1);
  2. TLSNP-LMR vs TLS-NP (Q2, per-term);
  3. NP-LMR vs NP-NOLM (Q2, plain; the user's question);
  4. NP-LMR-PSR vs NP-LMR (Q3);
  5. TLS-NP vs BASE-TLS (Q4).
* **The gap to HC-LM (7), its own Holm family:** each MOSES GOMEA arm against HC-LM.
* **Size and CPU (10), one Holm family:** program size and CPU for the 5 primary contrasts.
* **Descriptive (no verdicts; the new spec key of 12.2):** BASE-TLS, TLS-NP and HC-LM each
  against LIN-OLS and LIN-RIDGE (6); NP-NOLM vs TLS-NP (per-term least squares against plain
  scaling, both under NP and without LM); TLS-LMR vs TLSNP-LMR; the 5 primary contrasts on the
  clipped test NMSE.
* **Per dataset, descriptive (50):** the 5 primary contrasts on each dataset, no verdicts.
* **Trade-off table:** for each arm, the NMSE ratio against BASE-TLS and against HC-LM, the size
  and CPU ratios against BASE-TLS, demes, knobs, the LM counts and share, the pole counts.

## Local pilot before the server (a gate, not a selection)

* The new arms (TLS-LMR, TLSNP-LMR, NP-LMR, NP-NOLM, NP-LMR-PSR, LIN-OLS, LIN-RIDGE) on Con and Dow,
  split 1, 20,000 evaluations, on the clone's code. Do not read the NMSE.
* **Gate:** every run ends ok; in the three LM-reserve arms, fewer than half of the LM calls end
  with no budget; the LM share of each arm is at most 50 %; NP-LMR-PSR rejects at most 50 % of the
  LM results that it checks; the peak memory of each arm is at most 50 % of its limit. If a check
  fails, the agent reports to the user before the server run.

## Memory

As in rounds 12 and 13: Dow 8,192 MB, all other datasets 5,120 MB.

## Implementation (by subagents, in separate clones)

| Clone | Content |
|---|---|
| H | The fix (12.1) and the four nits (12.2). In progress. |
| I | `--const-opt-pole-check box-rel` under `pymoses` (identity reference first; opt-in; tests; USAGE) |
| J | After H and I merge: splits 61–70; the linear baselines in the runner; `configs/gomea/phase1bi/`, `plan14.json`, the smoke plan, `test_plan14.py`, the SERVER-RUN section; the pilot; the dry run; the local smoke run |

* Then: merge, the full test suite, the RV16 review, and the push only with a fresh approval.
* An arm whose code is not ready leaves the round; the others still run.

## Size and time

* CPU per 100 runs in round 13: BASE-TLS 18.2 h, TLS-NP 13.9 h, HC-LM 22.0 h. Estimates: TLS-LMR
  18–20 h, TLSNP-LMR 14–16 h, NP-LMR 12–14 h, NP-NOLM about 12 h, NP-LMR-PSR 12–14 h; the linear
  fits under 1 h. The phase: about 125–140 CPU-h, about 12–14 h at W = 11. S0: about 1.3 h.
* **About 14–16 hours in total, in one round.** The pilot gives a better estimate.

## Limits

* One budget (20,000 evaluations) and one reserve (500).
* NP-LMR-PSR tests the pole rule only under NP with plain scaling.
* The linear baselines use all inputs; no feature selection apart from ridge shrinkage.
* No ε-lexicase, no deme caps, no tier 2 idea in this round.
