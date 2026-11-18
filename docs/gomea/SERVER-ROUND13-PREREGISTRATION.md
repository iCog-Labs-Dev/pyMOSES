# Server round 13: pre-registration (2026-11-18)

**Status:** approved by the user on 2026-11-18 ("Yes.", `APPROVALS.md`), together with T3.1 as
the arm TLS-LM. Implemented in 22b2a63, 30669d7 and 1fcb749 (T3.1 and the pole safeguard);
6d81b98, 4a9d4fe and c106485 (ε-lexicase); cb1ccfe (Boolean manifest commits); 039ef88, fd7013a
and eb16cb2 (splits 51-60, the runner and the report, configs, plans, tests and the SERVER-RUN
section); b4ffc5b (boolean inputs of the pole check), 987da63 (Boolean manifest commits), 14145d7
and 0f6d818 (`test_nmse_clipped` only with the manifest key `clip_test_nmse`). The section
"Rules that the approved text did not cover" lists the agent's completions. The user decided the
frame on 2026-11-18:

* Reference: "Keep BASE, confirm TLS (Recommended)".
* Arms: "TLS + NP (Recommended), TLS + guard (Recommended), TLS + LM per term (T3.1)".
* Data: "Fresh splits 51–60 (Recommended)".
* New ideas from the backlog ("I also want us to add few more ideas from the backlog so we can
  start testing new ideas as well."): "ε-lexicase selection (Recommended), Pole safeguard
  (Recommended)".
* Tier 2: "No tier 2 note now" (only the T3.1 note, `T3.1-FEASIBILITY.md`).

Everything else below (option rules, thresholds, contrasts, the confirmation rule, the pilot gate,
the time estimate) is agent-written. The user approved it with the draft (commit 70a810a).

**Questions:**

1. Does per-term least squares (BASE-TLS) beat BASE again, on splits that round 12 did not use?
2. Do node-proximity linkage, a validation guard, or LM on the inner constants improve BASE-TLS?
3. Do ε-lexicase exemplar selection and a pole safeguard improve BASE?

HC-LM stays the accuracy baseline (user, 2026-11-11).

## Background

* Round 12 (`SERVER-ROUND12-FINDINGS.md`): BASE-TLS was better than BASE (0.879, 73/27), with
  gains on Dow, EnH, Tow, EnC and Con and a loss on Yac (1.61). Its programs were 1.18× larger in
  nodes (1.40× with the term weights). On WiW its train–test gap was 11.93 against 2.21. Its
  largest term weights reached 22,600 (Tow). Node-proximity linkage saved 26 % CPU at 6 % higher
  NMSE.
* `--linear-scaling terms` refuses LM today (`problem_params.py:1728`). T3.1 lifts this for LM
  with per-term weights refitted at each pass.
* Rare runs with a test NMSE above 100: 4 in round 11 (one an LM pole, Bos split 35), 2 in round
  12. The validation guard cannot see a pole that no validation row comes near (`BACKLOG.md`
  10.1).
* `--diversity-pressure` gave no gain in round 12. ε-lexicase is the second arm of backlog item A4.

## Rules

* As in rounds 1 to 12: on the server, one phase at a time, W from `resources.py`, state in
  `server-run13/`. Round 13 never overlaps with another round.
* **One phase (1bh, regression):** 10 datasets, **fresh splits 51–60**, 20,000 evaluations for
  each run (`-m 20000`), no time limit, a wall cap of 36,000 s as a guard only. LM passes and
  check passes count.
* Splits 51–60 need `SEEDS = range(1, 61)` in `tools/gomea/sr_datasets.py`, 480 split files in
  `setup.sh`, and the requirement `datasets:...:51-60`. Splits 1–50 stay byte-identical.
* **Statistics:** as in round 12 (the pooled test of round 8). Failed runs count as infinitely
  bad. Paired Wilcoxon signed-rank on log(A / B), two-sided, Holm, α = 0.025. "Better" or "worse"
  also needs a ratio at least 10 % from 1.
* **Confirmation rule for BASE-TLS (agent-written, from the user's choice "Keep BASE, confirm
  TLS"):** if BASE-TLS is "A better" than BASE again, the agent proposes a TLS arm as the new
  reference: BASE-TLS, or a TLS variant that is "A better" than BASE-TLS in this round. The user
  decides. Otherwise BASE stays the reference. All other verdicts are descriptive.
* **Program size of a TLS arm:** the node count plus `n_terms` when the final model uses the
  per-term fit (2 or more terms, no fallback). A final model with plain `a + b·f` adds 0, as for
  BASE. Round 12 counted only the nodes (its findings added `n_terms` to every program, also to
  one-term programs). The node count is also reported.
* **Reported (D1):** test NMSE, size, train–test gap, CPU, evaluations, demes, knobs per deme,
  total knobs, LM calls and LM share, guard rejections, the list of runs with a test NMSE above
  100 (with training and validation NMSE), and the extra values of each arm below.
* **Descriptive for every arm:** the test NMSE with the predictions clipped to the training target
  range widened by 10 % on each side (the second form of the pole safeguard; no search change, no
  extra runs).
* **S0:** full pytest and the reproducibility check.

## Arms (800 runs: 8 arms × 10 datasets × 10 splits)

Option names are proposals. Each new option is opt-in: without it, the output of every existing
setting stays byte-identical (identity tests).

| Arm | Task | Flags |
|---|---|---|
| **BASE** | reference | the round 12 BASE flags (`configs/gomea/phase1bf/`) |
| **HC-LM** | accuracy baseline | the round 12 HC-LM flags |
| **BASE-TLS** | T2.2, confirmation | the round 12 BASE-TLS flags (`--linear-scaling terms --const-opt none`) |
| **TLS-NP** | T2.1 + T2.2 | BASE-TLS with `--gomea-fos np` in place of `lt-b` (no new code) |
| **TLS-G** | TLS guard | the BASE-TLS argv; the runner chooses the final scaling by validation (`"terms_guard": "val"`, new, tools only) |
| **TLS-LM** | T3.1 | BASE-TLS with `--const-opt lm --const-opt-guard val` in place of `--const-opt none` (new) |
| **BASE-LEX** | A4, ε-lexicase | BASE + `--exemplar-selection lexicase` (new) |
| **BASE-PS** | pole safeguard | BASE + `--const-opt-pole-check box` (new) |

### TLS-G: the final-model guard (runner)

* For each printed candidate, the runner computes the validation NMSE twice, with the per-term
  weights and with plain `a + b·f`, both fitted on the training rows. It keeps the lower of the two
  (a tie keeps plain, the smaller model). Then it chooses the candidate with the lowest validation
  NMSE, as for every arm.
* The search is the same as BASE-TLS (same argv, same seeds). The report checks that both arms
  print the same candidates. TLS-G therefore tests only the guard.
* The summary records the chosen scaling (`terms.guard`: `terms` or `plain`). The runner also
  records the plain-scaling NMSE of each candidate for BASE-TLS, TLS-NP and TLS-LM (descriptive).
* **Agent choice:** a guard at the final choice, not in the search. A guard in the search needs the
  validation rows in every evaluation (about 1.5× the cost of each evaluation). The user's
  description ("fall back to plain a + b·f scaling when the per-term fit is worse on the validation
  rows") fits both; the final form is cheap and tests the overfit of the chosen model.

### TLS-LM: LM on the inner constants (T3.1)

* `T3.1-FEASIBILITY.md`. LM at deme end on the deme best, as in BASE. At each LM pass the term
  weights are fitted again by least squares, so LM tunes only the constants inside the terms. A
  one-term program, or a fit that falls back, uses plain `a + b·f`, as the scorer does.
* The validation guard in `terms` mode: weights from the training rows, applied to the validation
  rows.
* The CLI accepts `--linear-scaling terms` only with `--const-opt none` or `lm`. Top-k LM and LM
  during the search stay refused with `terms`.

### BASE-LEX: ε-lexicase exemplar selection

* `--exemplar-selection {boltzmann,lexicase}`, default `boltzmann` (the classic rule, unchanged).
* **Pool:** the candidates that the Boltzmann rule may select (finite penalized score, the revisit
  rule).
* **Selection:** a random order of the training rows (cases) from the run RNG. For each case, ε is
  the median absolute deviation (MAD) of the pool's errors on that case, computed once for each
  selection over the whole pool. Keep the candidates whose error on the case is at most the lowest
  error among the remaining candidates + ε. Stop when one candidate remains or after the last case;
  then pick one of the remaining candidates at random (run RNG). The visit count changes as for
  Boltzmann.
* **Errors:** the per-row errors of the behavioural score (the scorer's `-err` values, after linear
  scaling).
* **Behavioural scores:** the merge keeps them for this option through the existing recompute path
  (`merging.py`, as for `keep_bscore`). These recomputes are not counted as evaluations (as for
  BASE-DIV); the report gives their number and the CPU.
* The rule ignores the complexity penalty. The pool still comes from the metapopulation's score
  window. The report gives the program size.
* The run RNG is used only when the option is on.
* Source: La Cava, Spector and Danai, GECCO 2016 (DOI 10.1145/2908812.2908898). The evidence is for
  parent selection in population GP, not for MOSES exemplar selection (`BACKLOG.md` section 8, A4).
  The ε rule above is the "semi-dynamic" form named in later work by La Cava et al. (not checked
  again in the paper text).

### BASE-PS: the pole safeguard after LM

* `--const-opt-pole-check {none,box}`, default `none`.
* **Probe points:** 256 points, uniform in the box [minⱼ − 0.1·rⱼ, maxⱼ + 0.1·rⱼ] for each input j
  of the training rows (rⱼ = maxⱼ − minⱼ). Drawn once for each run from a separate RNG seeded by
  the run seed, so the other random streams do not change.
* **Check:** after LM improves a program, predict the old and the tuned program on the probes, each
  with its linear scaling fitted on the training rows. S(p) = the largest |prediction − mean of the
  training target| over the probes; R = the range of the training target. Reject the tuned program
  if a probe prediction is not finite (and all of the old program's are finite), or if
  S(tuned) > 10·R and S(tuned) > S(old). A rejected tuning keeps the old program, as a guard
  rejection does.
* The check runs before the validation guard. Each probe pass counts as one evaluation (two for
  each check).
* Telemetry: `lm_pole_checked`, `lm_pole_rejected`.
* **Agent choices:** 256 probes, the 10 % margin and the factor 10. The pilot reports the rejection
  share.

### Rules that the approved text did not cover (agent, after the approval)

These completions came during the implementation. None of them changes an arm, a contrast or a
gate.

* **Pole check, non-finite values:** a probe prediction that is not finite counts as +∞ in S. So
  when the old and the tuned program both have a non-finite prediction, the check accepts the
  tuning (+∞ is not greater than +∞).
* **Pole check, boolean inputs:** MOSES types a 0/1 column as boolean (Bos `x4`). Such an input
  gets false or true, each with probability 0.5, and the margin does not apply to it. Each input
  of each probe still takes one draw, so the probes of a table with numeric inputs only do not
  change. The smoke run found the gap: the two Bos runs of BASE-PS stopped with a `ValueError` at
  b49d050. The fix is b4ffc5b.
* **Pole check, seed:** the probe generator is seeded with the run seed (`-r`), as the text above
  says. Its first draws are therefore correlated with the first draws of the main stream. The main
  stream does not change.
* **Clipped NMSE:** a prediction of +∞ or −∞ becomes the bound; a `nan` prediction stays `nan`, so
  the NMSE is +∞. For TLS-G, the clip acts on the scaling that the guard keeps.
* **TLS-G:** a one-term program, or a per-term fit that falls back, is a tie, so the guard keeps
  plain.
* **Descriptive contrasts:** the report puts the 9 descriptive pooled contrasts (TLS-NP, TLS-G and
  TLS-LM against BASE; the 6 primary contrasts on the clipped NMSE) in their own Holm family. The
  findings report them as descriptive, with no verdict.

## Contrasts

* **Primary (6), pooled over 100 pairs, test NMSE, Holm across the 6:**
  1. BASE-TLS vs BASE (the confirmation);
  2. TLS-NP vs BASE-TLS;
  3. TLS-G vs BASE-TLS;
  4. TLS-LM vs BASE-TLS;
  5. BASE-LEX vs BASE;
  6. BASE-PS vs BASE.
* **The gap to HC-LM (7), its own Holm family:** BASE and the 6 other arms, each against HC-LM.
* **Size and CPU (12), one Holm family:** program size and CPU for the 6 primary contrasts.
* **Descriptive:** TLS-NP, TLS-G and TLS-LM each against BASE; the test NMSE with clipped
  predictions for every arm.
* **Per dataset, descriptive (60):** the 6 primary contrasts on each dataset, no verdicts.
* **Trade-off table:** for each arm, the NMSE ratio against BASE and against HC-LM, the size and
  CPU ratios against BASE, the demes, knobs per deme, total knobs, the LM share, the terms (TLS
  arms) and the guard, pole and lexicase counts.
* **Power:** with 100 pairs, the lowest p values are far below 0.025 / 6.

## Local pilot before the server (a gate, not a selection)

* Each new arm (TLS-NP, TLS-G, TLS-LM, BASE-LEX, BASE-PS) on Con and Dow, split 1, 20,000
  evaluations, on the clone's code. Do not read the NMSE.
* Measure: TLS-LM LM calls, LM share and guard rejections; the TLS-G share of candidates that keep
  plain scaling; BASE-LEX selections, cases used per selection, pool size, behavioural-score
  recomputes, CPU and peak memory; BASE-PS checks and rejections; peak memory of each arm.
* **Gate:** every run ends ok; the LM share of TLS-LM is at most 50 %; BASE-PS rejects at most
  50 % of the LM results that it checks; the peak memory of each arm is at most 50 % of its limit.
  If a check fails, the agent reports to the user before the server run.

## Memory

As in round 12: Dow 8,192 MB, all other datasets 5,120 MB. BASE-LEX keeps the behavioural scores of
the metapopulation (candidates × training rows; largest on Tow and WiW). The pilot measures it.

## Implementation (by subagents, in separate clones)

| Clone | Content |
|---|---|
| E | T3.1 and the pole safeguard (both change the LM path: `const_opt.py`, the optimizer's deme-end LM, `problem_params.py`) |
| F | ε-lexicase (`metapopulation.select_exemplar`, the merge's behavioural scores, `problem_params.py`) |
| G | Splits 51–60; the runner (TLS-G guard, plain-scaling fields, clipped NMSE); the report (TLS size with terms, the list of runs above 100, knob and term columns); `configs/gomea/phase1bh/`, `plan13.json`, the smoke plan, `test_plan13.py`, the SERVER-RUN section; after E and F merge |

* Each option: the identity reference committed first, then the code (`_PORT_OPTIONS` or
  `_GOMEA_OPTIONS`, `_OPTIONAL_DEFAULTS`, telemetry only when on, no random draws when off,
  refusals, help text, `USAGE.md`).
* E and F start from the same commit. The first merges with `--ff-only`, the other by cherry-pick.
* An idea whose code is not ready leaves the round; the others still run (user rule of round 12,
  2026-11-16).
* Then: merge, full test suite, smoke run of the whole chain in a clean clone, dry run, RV16
  review, and the push only with a fresh approval.

## Size and time

* CPU per 100 runs in round 12: BASE 16.3 h, HC-LM 22.4 h, BASE-TLS 18.3 h, BASE-NP 12.0 h.
  Estimates for the new arms: TLS-NP about 13 h, TLS-G about 18 h, TLS-LM 19–25 h, BASE-LEX
  17–25 h, BASE-PS 16–18 h. The phase: about 140–160 CPU-h, about 13–15 h at W = 11. S0: about
  1.3 h.
* **About 15–18 hours in total.** The pilot gives a better estimate.

## Limits

* One budget (20,000 evaluations) and one value for each new option (256 probes, factor 10, the
  semi-dynamic ε).
* TLS-G guards only the final choice, not the search.
* BASE-LEX and BASE-PS are tested against BASE, not on top of a TLS arm. A later round can combine
  them with the winner.
* No mux11 phase: ε-lexicase is tested on regression only.
* The pole check samples the input box; a pole outside the widened box stays unseen.
* The new code is first used on the server after local tests and the pilot only.
