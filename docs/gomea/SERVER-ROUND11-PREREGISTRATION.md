# Server round 11: pre-registration draft (2026-11-11)

**Status:** draft. The user approved the direction and answered questions 1, 3 and 4 (2026-11-11,
see `APPROVALS.md`). Question 2 (the values) is open. Not implemented.

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

## Code check (2026-11-11)

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
| **BASE-STALL1** | BASE + `--gomea-stall-gens 1`: the deme ends after the first generation with no improvement |
| **BASE-D10** | BASE + `--gomea-deme-share 10`: each deme gets at most 2,000 evaluations, so a run gets about 10 or more demes |
| **BASE-R** | BASE + an LM reserve (new opt-in flag): a deme that has the rest of the run budget keeps min(500, budget // 2) evaluations for LM, as HC-LM-W does |
| **HC-LM** | the round 10 HC-LM flags (accuracy baseline) |

* BASE-STALL1 changes only the end rule. A deme still runs full generations.
* BASE-D10 sets a deme length that does not depend on the search. On a dataset with many knobs,
  2,000 evaluations can be less than one generation, as in C250.
* BASE-R changes only the LM budget of the last deme. It shows how much of the gap comes from LM
  that had no budget (user, 2026-11-11: "Yes let's go with that").
* BASE-S and HC-LM-W are not in this round. Seeding is a separate question. Widen search was worse
  in round 10.

## Local pilot before the server (a gate, not a selection)

* BASE, BASE-STALL1 and BASE-D10 on Air, Con and Tow, splits 1 and 2, at 20,000 evaluations.
* Measure only the demes in each run and the generations in each deme. Do not look at the NMSE.
* **Gate:** each variant must have at least 2× the median demes of BASE on at least 2 of the 3
  datasets. If a variant fails, the agent reports to the user before the server run.

## Contrasts

* **Primary (2), pooled over 100 pairs (10 datasets × 10 splits), test NMSE:**
  1. BASE-STALL1 vs BASE;
  2. BASE-D10 vs BASE.
  * **Test:** paired Wilcoxon signed-rank on r = log(NMSE_A / NMSE_B), two-sided, Holm across the
    2, α = 0.025. A failed run gets the largest rank against it.
* **The gap to HC-LM (3), pooled, its own Holm family:** BASE vs HC-LM, BASE-STALL1 vs HC-LM,
  BASE-D10 vs HC-LM.
* **Size and CPU (4), pooled, one Holm family:** program size and CPU, each for BASE-STALL1 vs BASE
  and BASE-D10 vs BASE.
* **Per dataset, descriptive (20):** contrasts 1 and 2 on each dataset, no verdicts.
* **Trade-off table:** for each arm, the median pair ratio of test NMSE against BASE and against
  HC-LM, and the median size ratio against BASE.

## The user's answers (2026-11-11)

1. Verdicts: "Results only." There is no adoption rule and no verdict rule beyond the pooled test.
2. Values: open. The user asked for an explanation.
3. Splits: "Yes let's do split this time." Fresh splits 31–40.
4. LM reserve for GOMEA: "Yes let's go with that." The arm BASE-R joins the round. It needs a code
   change under `pymoses`: opt-in, with an identity test that BASE without the flag is
   byte-identical.

## Open questions for the user (as first asked)

1. **Verdicts:**
   * (a) results only, as in round 10;
   * (b) the round 10 verdict (Holm p < 0.025 and a median pair difference of at least 10 %);
   * (c) a rule that includes the size.
   * Recommendation: (a). This round is the first test of the deme count, so the results guide
     the rule for later rounds.
2. **Values:** `--gomea-stall-gens 1` and `--gomea-deme-share 10`, or other values? A third arm,
   for example `--gomea-deme-share 20`, adds 100 runs and about 3 hours.
3. **Fresh splits 31–40:** recommended, because splits 1–10 are selection data.
4. **LM reserve for GOMEA:** add an arm BASE-R, BASE with an LM reserve as in HC-LM-W, so that the
   last deme always gets LM? It separates "more LM" from "more demes". It needs a code change
   under `pymoses` and adds 100 runs. Recommendation: yes, because LM did nothing in 17 BASE runs.

## Memory

| Manifest | `mem_limit_mb` | Reason |
|---|---|---|
| Dow | 8,192 | HC-LM peaked at 3.34 GB on Dow in round 10; more demes can need more memory |
| All other datasets | 5,120 | BASE and HC-LM stayed below 3.4 GB in round 10 |

There is no N = 16 arm, so the round 9 memory risk does not apply.

## Implementation (by a subagent, in a separate clone)

* `tools/gomea/sr_datasets.py`: splits 31–40, and the matching dataset checks in `setup.sh`.
* `configs/gomea/phase1be/`: manifests and spec.
* `tools/server/plan11.json`, a smoke plan, tests and the SERVER-RUN section.
* The local pilot, with its result written into this file before the server run.
* No change under `pymoses`.

## Size and time

* 500 runs (5 arms × 10 datasets × 10 splits), with BASE-R.
* CPU for each run in round 10: BASE median 430 s, HC-LM 674 s. C250 used 1.1–3.9× the CPU of
  BASE, so the variants can cost more.
* About 10–16 hours at W = 11, plus about 1.3 hours for S0. The local pilot takes about 1 hour.

## Limits

* One budget (20,000 evaluations).
* BASE-D10 can change N as well as the deme length, on a dataset where N would pass 2,000.
* Two values only. They show a direction, not the best value.
