# Server round 11: pre-registration draft (2026-11-11)

**Status:** the user approved the direction and answered all four questions (2026-11-11, see
`APPROVALS.md`). Agent-written, not user-approved in detail: the BASE-R rule, the contrasts and
the Holm families, the pilot gate and the memory limits. Implemented in ab5047d, 3398504,
709b27f, e730598, 9e7cc39, 5ee8c83 and 102d79e; pilot result in 98c0a8c (the gate passed).
After the pilot, the user decided to run the D arms as built and to keep BASE-R with the text
corrected (2026-11-11, see `APPROVALS.md`). The "Arms", "Code change" and "Limits" sections now
state what the pilot showed.

**Question:** do shorter GOMEA demes (more demes in each run) make the base more accurate? How much
larger do the programs get? HC-LM is the accuracy baseline (user, 2026-11-11).

## Background

Round 10 (`SERVER-ROUND10-FINDINGS.md`, splits 1–10) and round 9 (`SERVER-ROUND9-FINDINGS.md`):

| Arm (round) | Demes in a run (median, by dataset) | Test NMSE against BASE | Program size against BASE |
|---|---|---|---|
| BASE (10) | 1–6.5 (1 on Dow and Tow) | reference | reference (pooled median 47) |
| HC-LM (10) | 9.5–25.5 | lower in 81 of 100 pairs | 1.6× (pooled median 76) |
| BASE-N16 (9, 1ba) | 3–25 | lower median on 8 of 8 datasets | 1.5–6.3× |
| BASE-C250 (9, 1bc) | 50.5–59.5 | lower median on 5 of 6 datasets | 30–220 % larger |

* **Observation, not tested:** in each case, more demes came with a lower test NMSE and larger
  programs. Each deme starts from a new exemplar, and LM runs at the end of each deme.
* BASE ends a deme after 5 generations without improvement (`--gomea-stall-gens`, default 5). It
  has 4–10 generations in a deme (round 10). One generation costs about N × 2 × knobs
  evaluations, with N about 100, so a deme uses much of the budget.
* **BASE often got no LM (round 10 findings, observation 2):** one deme used the whole budget, so
  the LM call at its end got 0 evaluations. This occurred in 17 of 100 BASE runs (Dow 10 of 10,
  Tow 7 of 10) and in 36 of 100 BASE-S runs. HC-LM-W had the same fault before the LM reserve.
  More demes also give more LM calls, so this round cannot separate the two effects.
* C250 is an extreme case: the cap of 250 evaluations is smaller than one generation, so each deme
  had 1 generation only. The user left C250 out because of the program size (2026-11-10).

## Code check (2026-11-11): one change under `pymoses` (BASE-R)

| Need | Existing option | Where |
|---|---|---|
| End a deme sooner when it stops improving | `--gomea-stall-gens K` (default 5) | `problem_params.py:345`, `engine.py:240` |
| Give a run about D demes | `--gomea-deme-share D`: cap = max(1, floor(-m / D)) | `problem_params.py:439`, `:880` |
| Other end conditions (not used) | `--gomea-max-gens`, `--gomea-stall-evals`, `--gomea-max-evals-per-deme` | `problem_params.py:338`, `:393`, `:352` |

* **Measured in round 10 (BASE, all demes):** a deme spends almost all of its evaluations while it
  improves. After the last improving generation, the median deme spent only 3.0 % more (per
  dataset 0–5.9 %, Yac 8.2 %). On Dow and Tow the one deme was still improving at the budget.
  Because of the genotype memo, a generation with no improvement costs almost no evaluations.
  Thus `--gomea-stall-gens 1` saves about 3 % of the budget and gives almost no extra demes.
  Script: `scratchpad/r10res/gen_cost.py` (session scratchpad, not in the repo).
* **LM in capped demes (round 9, 1bc):** BASE-C250 made 3,320 LM calls, and only 38 (1 %) had no
  budget. The cap limits GOMEA only; LM runs after the cap and uses the run budget. LM used
  375,107 evaluations in the 60 BASE-C250 runs, about 31 % of their budget.
* The stagnation test uses the penalized score (PEN125) when `--gomea-accept-complexity-ratio` is
  set, as in BASE.
* With the dynamic population, N comes from the deme budget (at least 100, at most the evaluations
  left). A deme cap changes N only when N would pass the cap. Round 10 BASE had N = 100 on all
  datasets.

## Rules

* As in rounds 1 to 10: on the server, one phase at a time, W from `resources.py`, state in
  `server-run11/`. Round 11 never overlaps with another round.
* **Budget:** 20,000 evaluations for each run (`-m 20000`), no time limit, a wall cap of 36,000 s
  as a guard only. LM passes count.
* **Splits 31–40 (fresh)** on all 10 datasets. Splits 1–30 were used for selection in earlier
  rounds. This needs a change in `tools/gomea/sr_datasets.py` (`SEEDS = range(1, 31)`).
* **Statistics:** as in round 10 (the pooled test of round 8). Failed runs count as infinitely bad.
* **Reported (D1):** test NMSE, program size, train–test gap, CPU, evaluations, demes in each run,
  generations in each deme, N, LM calls and guard rejections.
* **S0:** full pytest and the reproducibility check.

## Arms

| Arm | Flags |
|---|---|
| **BASE** | the round 10 BASE flags (`configs/gomea/phase1bd/`) |
| **BASE-R** | BASE + `--const-opt-reserve 500` (new, opt-in): see "Code change" |
| **BASE-D5** | BASE + `--gomea-deme-share 5`: each deme gets at most 4,000 evaluations |
| **BASE-D10** | BASE + `--gomea-deme-share 10`: each deme gets at most 2,000 evaluations |
| **BASE-D20** | BASE + `--gomea-deme-share 20`: each deme gets at most 1,000 evaluations |
| **HC-LM** | the round 10 HC-LM flags (the accuracy baseline, user 2026-11-11) |

* **BASE-R** gives LM a budget in each deme (user, 2026-11-11: "Yes let's go with that"). It does
  not change only the LM budget (pilot, corrected 2026-11-11). LM on the first deme's best program
  used only 13–28 of the 500 reserved evaluations. MOSES gave the rest to 4–5 tiny demes (0–1
  generations), each with an LM call, mostly on the same best program. Thus BASE-R = BASE + LM on
  the main deme's best program + a few tiny demes with repeated LM. The user kept the arm on these
  terms (2026-11-11: "Keep, fix the text (Recommended)").
* **BASE-D5, BASE-D10, BASE-D20** change only the deme cap (user, 2026-11-11: "All three"). Three
  caps show whether the effect grows with the number of demes. A run gets at least about 5, 10 or
  20 demes (pilot: 6, 10 and 17–18), which is in the HC-LM range of round 10 (9.5–25.5), not in
  the C250 range of round 9 (50–60).
* **Correction (pilot, 2026-11-11):** the draft said that the caps are "much longer than C250
  (250)". That is true in evaluations, but not in generations. After the first deme, the knob count
  grows (Air 14 → 56–157, Con 20 → 65–334). Then one generation costs more than the cap. So every
  D10 and D20 deme after the first stops inside generation 0 (the first GOM pass), as C250 did. In
  D5, the same occurred, except 1 deme on Con (8 generations). Thus an accuracy gain in the D arms
  comes mostly from restarts and LM calls, not from model-based search inside a deme. At 20,000
  evaluations with N = 100, a cap that allows full generations gives only 1–2 demes. The user kept
  the arms (2026-11-11: "Run as built (Recommended)").
* The D arms have no LM reserve. Round 9 showed that LM runs after a capped deme (99 % of the
  calls had budget). Only the last deme of a run can lose its LM call.
* **BASE-STALL1 is not in the round.** `--gomea-stall-gens 1` saves only about 3 % of the budget
  (see "Code check"), so it does not change the deme count.
* BASE-S and HC-LM-W are not in this round. Seeding is a separate question. Widen search was worse
  in round 10.

## Code change: LM reserve for GOMEA (`--const-opt-reserve R`)

* **What:** a new integer option, default 0 (off). With `-a gomea --const-opt lm` and R > 0, GOMEA
  in each deme stops at the deme budget minus min(R, deme budget // 2). LM on the deme's best
  program can then use the evaluations that are left. This is the rule of HC-LM-W (`hc_lm.py`).
* A deme that ends by stagnation before this point is not changed.
* **Opt-in:** without the option, GOMEA runs stay byte-identical. An identity test proves it
  (reference output captured before the change).
* **Tests:** a run where one deme would use the whole budget gets LM evaluations > 0 with R > 0;
  the total evaluations never pass `-m`; the option is refused without `--const-opt lm`. The code
  also refuses it with `-a hc` and with the hc-gomea hybrid (implementation, agent).
* The `const_opt_reserve` value in a deme's `end` event is that deme's reserve, min(R, deme
  budget // 2), not R.

## Local pilot before the server (a gate, not a selection)

* BASE and BASE-R on Dow and Tow, split 1. BASE-D5, BASE-D10 and BASE-D20 on Air and Con, split 1.
  All at 20,000 evaluations, on the clone's code.
* Measure only the LM evaluations (BASE-R) and the demes in each run (D arms). Do not look at the
  NMSE.
* **Gate:** BASE-R has LM evaluations > 0 on Dow and Tow; each D arm has at least D / 2 demes. If a
  check fails, the agent reports to the user before the server run.

## Pilot result (2026-11-11, agent): the gate passed

Local machine, the clone's code (709b27f), split 1, `-m 20000`, nice 19, 4 jobs, the datasets of
the local `phase1f/datasets`. The NMSE was not read. "Demes" counts the demes with evaluations.

| Dataset | Arm | Demes | LM evaluations | LM calls without budget | Gate |
|---|---|---|---|---|---|
| Dow | BASE | 1 | 0 | 1 of 1 | (reference) |
| Dow | BASE-R | 5 | 92 | 0 of 5 | LM > 0: pass |
| Tow | BASE | 2 | 13 | 1 of 2 | (reference) |
| Tow | BASE-R | 6 | 83 | 0 of 6 | LM > 0: pass |
| Air | BASE-D5 | 6 | 486 | 1 of 6 | ≥ 2.5 demes: pass |
| Con | BASE-D5 | 6 | 591 | 1 of 6 | ≥ 2.5 demes: pass |
| Air | BASE-D10 | 10 | 1,395 | 1 of 10 | ≥ 5 demes: pass |
| Con | BASE-D10 | 10 | 835 | 1 of 10 | ≥ 5 demes: pass |
| Air | BASE-D20 | 17 | 3,000 | 0 of 17 | ≥ 10 demes: pass |
| Con | BASE-D20 | 18 | 2,297 | 1 of 18 | ≥ 10 demes: pass |

Observations (not part of the gate, not tested):

* **BASE-R adds small demes.** In the first deme, LM used only 28 (Dow) and 13 (Tow) of the 500
  reserved evaluations. MOSES then started more demes with the rest of the budget, as for HC-LM-W in
  round 10: 4 on Dow and 5 on Tow, with 0 or 1 generation each and a reserve that halves each time.
  Each of these demes also got an LM call. Thus BASE-R also has more demes and more LM calls than
  BASE. On Tow the first deme ended by stagnation before the stop point and was the same as in BASE.
* **The D arms have very short demes.** Generations in a deme: D5 1–9, D10 1–2, D20 always 1. One
  generation with N = 100 costs about N × 2 × knobs evaluations, so a cap of 1,000 or 2,000 gives
  about 1 generation, as C250 did in round 9. (Reviewer check: more than 1 generation occurred
  only in the first deme of a run, and in 1 later deme on Con in D5.)
* **LM share of the evaluations:** D5 2–3 %, D10 4–7 %, D20 12–15 %; BASE and BASE-R below 1 %.
* **Code:** at 5da6731 the D arms stopped with a `ValueError` (`--gomea-deme-share` with
  `--gomea-pop-size dynamic`). Commit 709b27f corrected this without a change to N or the search.

## Contrasts

The user asked for results only: no adoption rule. The report gives the verdict column of round 10
(Holm p < 0.025 and a median pair difference of at least 10 %) as a description only.

* **Primary (4), pooled over 100 pairs (10 datasets × 10 splits), test NMSE, Holm across the 4:**
  1. BASE-R vs BASE;
  2. BASE-D5 vs BASE;
  3. BASE-D10 vs BASE;
  4. BASE-D20 vs BASE.
  * **Test:** paired Wilcoxon signed-rank on r = log(NMSE_A / NMSE_B), two-sided, α = 0.025. A
    failed run gets the largest rank against it.
* **The gap to HC-LM (5), pooled, its own Holm family:** BASE, BASE-R, BASE-D5, BASE-D10 and
  BASE-D20, each vs HC-LM.
* **Size and CPU (8), pooled, one Holm family:** program size and CPU, each for contrasts 1–4.
* **Per dataset, descriptive (40):** contrasts 1–4 on each dataset, no verdicts.
* **Trade-off table:** for each arm, the median pair ratio of test NMSE against BASE and against
  HC-LM, the median size ratio against BASE, the median demes in a run and the LM share of the
  evaluations.

## The user's answers (2026-11-11)

1. Verdicts: "Results only."
2. Values: the agent explained the values and measured that `--gomea-stall-gens 1` does not change
   the deme count. It proposed caps D5 and D10 and offered D20. The user answered: "All three."
3. Splits: "Yes let's do split this time." Fresh splits 31–40.
4. LM reserve for GOMEA: "Yes let's go with that." The arm BASE-R joins the round.

## Memory

| Manifest | `mem_limit_mb` | Reason |
|---|---|---|
| Dow | 8,192 | HC-LM peaked at 3.34 GB on Dow in round 10 |
| All other datasets | 5,120 | all round 10 arms except NOCAP stayed below 3.4 GB; round 9 BASE-C250 stayed below 1 GB |

There is no N = 16 arm, so the round 9 memory risk does not apply.

## Implementation (by a subagent, in a separate clone)

* `--const-opt-reserve` under `pymoses` (see "Code change"), in its own commit, with the identity
  test committed first.
* `tools/gomea/sr_datasets.py`: splits 31–40, and the matching dataset checks in `setup.sh`.
* `configs/gomea/phase1be/`: manifests and spec.
* `tools/server/plan11.json`, a smoke plan, tests and the SERVER-RUN section.
* The local pilot, with its result written into this file before the server run.

## Size and time

* 600 runs (6 arms × 10 datasets × 10 splits).
* CPU for each run: round 10 BASE median 430 s, HC-LM 674 s; round 9 BASE-C250 748 s (1.8× BASE).
  The D arms can cost up to about 2× BASE.
* About 12–18 hours at W = 11 (W = 7 for Dow), plus about 1.3 hours for S0. The local pilot takes
  about 1 hour.

## Limits

* One budget (20,000 evaluations).
* A cap can change N as well as the deme length, on a dataset where N would pass the cap.
* Three cap values show a direction, not the best value.
* The D arms also make more LM calls. BASE-R does not show the effect of LM budget alone: it also
  adds a few tiny demes with repeated LM (see "Arms").
* After the first deme, the D arms stop inside the first generation. They test restarts plus LM in
  the HC-LM deme range, with a short GOMEA search in each deme (see "Arms").
* Nobody measured the memory of the D arms at 20,000 evaluations. Later demes have up to 334 knobs
  (the tiny BASE-R demes up to 1,277). Round 9 C250 stayed below 1 GB.
* `--const-opt-reserve` is new code. Its first use is in this round, after local tests only.
