# Server round 12: pre-registration draft (2026-11-16)

**Status:** a draft for the user's approval. The user decided the frame (2026-11-16, `APPROVALS.md`):
BASE stays the reference, BASE-R runs again as an arm, one fixed reference for all arms, fresh
splits 41–50, and the round includes T1.6 and T2.1–T2.4. Everything else in this file is
agent-written and not user-approved: the option rules, the arm details, the contrasts, the pilot
gate and the time estimate. Not implemented.

**Questions:**

1. Does LM on the top 5 programs of each deme (T1.6) make BASE more accurate at the same budget?
2. Do the four tier 1 ideas (`ROADMAP.md`, stage 2) make BASE more accurate, and at what cost in
   program size and CPU?
3. Does BASE-R confirm its round 11 result on new splits?

HC-LM is the accuracy baseline (user, 2026-11-11).

## Background

* Round 11 (`SERVER-ROUND11-FINDINGS.md`): BASE-R was lower than BASE in 62 of 82 pairs that
  differ (ratio 0.984), at the same CPU; on Air, Con, Dow and Tow it closed 42–76 % of the gap to
  HC-LM. One BASE-R run failed on a pole (T1.5 follow-up).
* A BASE deme hands 288–746 programs to the metapopulation, but LM tunes only the best one (local
  probe, 2026-11-16; `BACKLOG.md` section 10, item 5).
* The code survey of 2026-11-16 (agent) found the hook for each idea. T2.2 needs the most new
  code, then T2.3, T2.1 and T1.6. T2.4 needs no code for its first arm.

## Rules

* As in rounds 1 to 11: on the server, one phase at a time, W from `resources.py`, state in
  `server-run12/`. Round 12 never overlaps with another round.
* **Phase A (regression):** 10 datasets, **fresh splits 41–50**, 20,000 evaluations for each run
  (`-m 20000`), no time limit, a wall cap of 36,000 s as a guard only. LM passes count.
* **Phase B (mux11):** 10 fresh seeds 3101–3110 (not used in any earlier manifest), a wall-time
  budget of 3,600 s as in phases 1ay and 1az.
* Splits 41–50 need `SEEDS = range(1, 51)` in `tools/gomea/sr_datasets.py`, 400 split files in
  `setup.sh`, and the requirement `datasets:...:41-50`. Splits 1–40 stay byte-identical.
* **Statistics:** as in round 11 (the pooled test of round 8). Failed runs count as infinitely
  bad. The report lists every run with a test NMSE above 100, with its training and validation
  NMSE (from the T1.5 follow-up).
* **Results only (agent assumption, as in round 11):** no adoption rule. The verdict column (Holm
  p < 0.025 and a median pair difference of at least 10 %) is descriptive.
* **Reported (D1):** test NMSE, program size, train–test gap, CPU, evaluations, demes, knobs per
  deme, total knobs, LM calls and LM share, guard rejections; the extra values of each arm below.
* **S0:** full pytest and the reproducibility check.

## Phase A arms (800 runs: 8 arms × 10 datasets × 10 splits)

Option names are proposals. Each new option is opt-in: without it, the output of every existing
setting stays byte-identical (identity tests).

| Arm | Task | Flags |
|---|---|---|
| **BASE** | reference | the round 11 BASE flags (`configs/gomea/phase1be/`) |
| **HC-LM** | accuracy baseline | the round 11 HC-LM flags |
| **BASE-R** | T1.5 | BASE + `--const-opt-reserve 500` |
| **BASE-K5** | T1.6 | BASE + `--const-opt-top-k 5` (new) |
| **BASE-NP** | T2.1 | BASE with `--gomea-fos np` (new) in place of `lt-b` |
| **BASE-TLS** | T2.2 | BASE with `--linear-scaling terms` (new) and `--const-opt none` (no LM, no guard) |
| **BASE-LME** | T2.3 | BASE + `--const-opt-during elite` (new) |
| **BASE-DIV** | T2.4 | BASE + `--diversity-pressure P --diversity-autoscale 0` (existing options) |

### T1.6: `--const-opt-top-k K` (BASE-K5, K = 5)

* At deme end, after `oracle.finalize`, sort the deme entries by the MOSES score (ties: the entry
  order). Walk down the list and take the first K programs with distinct structures. The structure
  key is the reduced tree with every constant replaced by one placeholder.
* Run LM with the validation guard on each of the K programs, as `_tune_best` does for the best
  one. Write back each accepted program (Lamarckian, as now). The first of the K is the program
  that BASE tunes, so BASE-K5 adds K − 1 LM calls to each deme.
* Every LM pass counts against the run budget (`charge_tuning`), as now.
* A new method; `_tune_best` and HC-LM stay unchanged. The option is refused with `-a hc` and
  `-a hc-gomea`, and without `--const-opt lm`.
* Expected cost: about 4 × (34–115) extra evaluations for each deme (round 11 LM call medians),
  plus 3 guard and scoring passes for each call.

### T2.1: `--gomea-fos np` (BASE-NP)

* The FOS is a UPGMA linkage tree, as `lt-b` builds it, but on the similarity S(i, j) = −d(i, j).
  d is the number of edges between the exemplar nodes of knobs i and j (the path through their
  lowest common ancestor in the decorated exemplar). It needs no population data.
* Computed once for each deme. Ties use the existing tie rule of the linkage tree (`fos.py`).
* Source: Koch, Alderliesten and Bosman, GECCO 2026 (`BACKLOG.md` section 8, A1).

### T2.2: `--linear-scaling terms` (BASE-TLS)

* The scorer splits the program at a top-level `+` into terms (else one term). It fits
  y ≈ w0 + Σ wᵢ·tᵢ by least squares on the training rows, in place of the single a + b·f.
* If the term matrix is rank-deficient, or a term output is not finite, the scorer uses the plain
  a + b·f fit for that program.
* The runner refits the same weights on the training rows when it scores this arm's candidates
  (an opt-in flag of `sr_eval`), so the reported NMSE is that of the weighted model.
* **LM is off in this arm** (`--const-opt none`). B4a is also the candidate alternative to LM
  (`BACKLOG.md` section 8). LM with per-term weights is tier 2 (T3.1).
* `_scaled_bscore` (plain linear scaling) stays byte-identical.

### T2.3: `--const-opt-during elite` (BASE-LME)

* During the search, when an offspring becomes a new deme best (a strict improvement under the
  MOSES score), run LM with the validation guard on it.
* Write the tuned constants back into the offspring's genotype as free contin codes, as
  `--const-opt mutate` does (`register_contin_value`). Keep the tuned offspring only if it is not
  worse under the GOM comparator (ratio 0.125), and mark it with `keep`.
* These LM passes go through a charge function that obeys the deme cap and the stall rule.
* The LM call at deme end stays, as in BASE.
* If the genotype write-back cannot be built in time (it needs a map from the constants of the
  unreduced tree to the genes), this arm leaves the round (user rule, 2026-11-16).

### T2.4: `--diversity-pressure P` (BASE-DIV)

* No new code. `--diversity-autoscale 0`, because autoscale gives a pressure of 0 for regression
  (`contin_bscore` has no worst possible score).
* **P = 6 × training rows** (agent choice): one third of the exemplar selection window, which is
  0.3 × `-v` = 18 × training rows. The pilot measures the size of the penalty (see "Pilot").
* The diversity step recomputes behavioural scores in the merge. This CPU is not counted as
  evaluations; the report gives the CPU.
* ε-lexicase exemplar selection (the second A4 arm) needs new code and is not in this round.

## Phase B arms (mux11, 40 runs: 4 arms × 10 seeds)

The Boolean setting that the user chose in round 6 (dynamic population, F 0.05, no cap): `-a gomea --gomea-pop-size dynamic --hc-fraction-of-nn 0.05
--gomea-skip-inactive 0 --gomea-max-evals-per-deme 0 --gomea-genotype-memo 1
--knob-probe-baseline reduced`, with `--gomea-fos` = `np`, `lt-b`, `rt` or `uni`. Metric: CPU to
solve (an unsolved run is infinite), as in the Boolean phases. The backlog asks for NP against RT
and UNI first.

## Contrasts

* **Phase A primary (6), pooled over 100 pairs, test NMSE, Holm across the 6:** BASE-R, BASE-K5,
  BASE-NP, BASE-TLS, BASE-LME and BASE-DIV, each against BASE. Paired Wilcoxon signed-rank on
  log(NMSE_A / NMSE_B), two-sided, α = 0.025.
* **The gap to HC-LM (7), its own Holm family:** BASE and the 6 arms, each against HC-LM.
* **Size and CPU (12), one Holm family:** program size and CPU for the 6 primary contrasts.
* **Per dataset, descriptive (60):** the 6 primary contrasts on each dataset, no verdicts.
* **Phase B (3), Holm across the 3:** NP against RT, UNI and LT-B, CPU to solve.
* **Trade-off table:** for each arm, the NMSE ratio against BASE and against HC-LM, the size and
  CPU ratios against BASE, the demes, knobs per deme, total knobs and the LM share.

## Local pilot before the server (a gate, not a selection)

* Each new arm on Con and Dow, split 1, 20,000 evaluations, on the clone's code. Do not read the
  NMSE.
* Measure: BASE-K5 LM calls per deme and LM share; BASE-LME LM calls and LM share; BASE-TLS term
  count, fallback share and CPU per evaluation; BASE-NP FOS size; BASE-DIV penalty size against the
  selection window; peak memory of each arm.
* **Gate:** every run ends ok; the LM share of BASE-LME is at most 50 %; the median diversity
  penalty of BASE-DIV is at least 1 % of the selection window. If a check fails, the agent reports
  to the user before the server run.

## Memory

As in round 11: Dow 8,192 MB, all other datasets 5,120 MB. No arm of phase A shortens the demes,
so the knob growth of the round 11 D arms is not expected. The pilot measures the peak memory.
mux11: 3,072 MB, as in the reproducibility check of S0.

## Implementation (by subagents, in separate clones)

| Clone | Content |
|---|---|
| A | T1.6 and T2.3 (both change the LM code in `optimizer.py`) |
| B | T2.2 (`bscores.py`, the solver, `sr_eval.py` and the runner) |
| C | T2.1 (`fos.py`, engine and optimizer wiring) |
| D | Splits 41–50, `configs/gomea/phase1bf/` and the mux11 manifest, `plan12.json`, the smoke plan, `test_plan12.py`, the SERVER-RUN section, after A to C merge |

* Each option: the identity reference committed first, then the code (the checklist of the code
  survey: `_PORT_OPTIONS` or `_GOMEA_OPTIONS`, `_OPTIONAL_DEFAULTS`, telemetry only when on, no
  random draws when off, refusals, help text, USAGE.md).
* All three code clones start from the same commit. The first merges with `--ff-only`, the
  others by cherry-pick (`problem_params.py` will need small merges).
* An idea whose code is not ready leaves the round; the others still run (user, 2026-11-16).

## Size and time

* CPU per 100 runs in round 11: BASE 16.9 h, BASE-R 16.9 h, HC-LM 22.6 h. The new arms are not
  measured; the estimate assumes 17–25 h each. Phase A: about 160 CPU-h, about 15 h at W = 11.
  Phase B: at most 40 h CPU, likely 1–3 h. S0: about 1.3 h.
* **About 17–20 hours in total.** The pilot gives a better estimate.

## Limits

* One budget (20,000 evaluations) and one value for each new option (K = 5, P = 6 × rows).
* BASE-TLS changes two things against BASE: per-term weights in place of a + b·f, and no LM. It
  tests per-term least squares as an alternative to LM, not as an addition.
* BASE-K5 and BASE-LME make more LM calls, so they also spend less budget on the search.
* BASE-NP is tested at the dynamic population size (N about 100), not at the N of the source paper
  (1,024).
* The diversity pressure P is an agent choice from the score scale, not a tuned value.
* The arms are compared with BASE one at a time. A later round combines the ideas that win.
* New code in four places, first used in this round after local tests only.
