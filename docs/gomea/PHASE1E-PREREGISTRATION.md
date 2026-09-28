# Phase 1e pre-registration: can embedded GOMEA beat MOSES hill climbing?

Written 2026-09-27 **before any Phase 1e run**. The commit that adds this file is the
timestamp.

**Requested by the user:** "What else do you think we do for gomea to beat hill climbing?
Shall we test it with more problems instead?" After the agent proposed this plan, the user
replied "Go ahead then what are you waiting for" (`APPROVALS.md` #9). The criteria are
agent-proposed.

## Background

In Phase 1d, capping GOMEA at 500 evaluations per deme tied hill climbing on maj5 and
halved GOMEA's mux6 cost. Four levers remained untested:

* other cap values;
* univariate GOMEA with a cap;
* the duplicate memo end-to-end;
* a stall-based stop.

A fifth idea is also untested: a hybrid that runs hill climbing first and hands over to
GOMEA only at a local optimum.

## Systems (code: the Phase 1e commit that adds `--gomea-stall-evals` and `-a hc-gomea`)

The common GOMEA flags are `--gomea-pop-size 16 --gomea-skip-inactive 0`. The shorthand
below is: `C<n>` means `--gomea-max-evals-per-deme <n>`, `M` means
`--gomea-genotype-memo 1`, and `S<n>` means `--gomea-stall-evals <n>`.

| Arm | Optimizer | FOS | Settings |
|---|---|---|---|
| HC | `-a hc` | — | stock |
| LTB-C250-M | `-a gomea` | lt-b | C250, M |
| LTB-C500-M | `-a gomea` | lt-b | C500, M |
| UNI-C250-M | `-a gomea` | uni | C250, M |
| UNI-C500-M | `-a gomea` | uni | C500, M |
| LTB-C500 | `-a gomea` | lt-b | C500, memo off (memo contrast) |
| UNI-C500 | `-a gomea` | uni | C500, memo off (memo contrast) |
| LTB-S200-M | `-a gomea` | lt-b | S200, M, no cap |
| UNI-S200-M | `-a gomea` | uni | S200, M, no cap |
| HYB-LTB | `-a hc-gomea` | lt-b | C500, M (the cap applies to the GOMEA phase) |
| HYB-UNI | `-a hc-gomea` | uni | C500, M |

## Stage 1: development round (choose the setup; not confirmatory)

**Setup.** maj5 and mux6 (`-H maj -k 5`, `-H mux -k 2`) and pa4 (`-H pa -k 4`),
`-m 20000`, **new seeds 2001–2010** (10 per task). 11 arms × 30 = 330 runs.

**Selection rule** among the GOMEA and hybrid arms:

1. Rank by the sum over the three tasks of the success rate.
2. Break ties by the lower sum over the tasks of median censored evaluations-to-success
   (a failure counts as 20,001).
3. If still tied, prefer the lower median final error summed over the tasks.

The top arm is the **winner**. The runner-up is recorded.

**Descriptive only** (no confirmatory claim):
* each arm vs HC, paired by seed, with Wilcoxon on final error and evaluations;
* the memo contrasts (LTB-C500-M vs LTB-C500, and UNI-C500-M vs UNI-C500);
* the stall arms vs the cap arms.

## Stage 2: larger problems (confirmatory)

**Setup.**
* **Tasks:** maj7 (`-H maj -k 7`), pa5 (`-H pa -k 5`) and mux11 (`-H mux -k 3`), where
  Phase 1c suggested GOMEA-style search has an edge on majority and parity.
* **Arms:** `HC` vs the stage 1 **winner**.
* **Seeds:** 3001–3020 (20 per task), disjoint from every earlier phase.
* **Budget:** `-m 40000`, since larger problems need more than 20k.
* **Limits:** wall cap 7,200 s, a 512 MB memory limit, and no MOSES time limit.

**Tests.** Paired by seed: the same MOSES seed gives the same initial exemplar, and the
trajectories then diverge. Per task:

* Wilcoxon signed-rank on final error, and on censored evaluations-to-success (a failure
  counts as 40,001);
* Holm correction within each task across these 2 metrics;
* success rates with Wilson CIs.

**Decision rule** (the same as Phases 1 and 1d). The winner is *better* on a task if a Holm
p < 0.05 **and** it has a success rate at least 10 percentage points higher or a median
censored evaluations-to-success at least 20 % lower. *Worse* is the mirror image;
otherwise it is a *tie*.

"**GOMEA beats hill climbing**" is claimed only for the tasks where the verdict is
*better*. If none are better, the report says so, and says where the gap remains.

## Deviation (recorded 2026-09-28, after stage 2 ran and before its analysis)

Stage 2 as registered did not give an equal-evaluation comparison:

* **maj7:** all 20 `LTB-C250-M` runs hit the 7,200 s wall cap before 40,000 evaluations.
  Pure-Python GOMEA costs more wall time per evaluation than HC on representations of
  about 470–840 knobs.
* **mux11:** three `LTB-C250-M` runs (seeds 3003, 3016, 3019) hit the 512 MB memory
  limit, and one (3005) hit the wall cap. One `HC` run (3016) crashed with a
  `RecursionError` in MOSES's recursive neighborhood counting, which nests once per knob.

Those 25 runs are re-run with the same code (`63023d9` changes only the harness; the MOSES
and GOMEA code is identical to `8224f3f`), the same seeds, budget and arguments, under
`configs/gomea/phase1e/stage2-rerun.json`. The only changes are:

* a wall cap of 36,000 s;
* a 2,048 MB memory limit;
* a per-run Python recursion limit of 100,000.

None of these limits changes a run's trajectory; they only decide whether it can finish.
The analysis replaces each of the 25 original runs with its re-run; the other 95 are kept.
The tests and decision rule are unchanged. Approved by the user (`APPROVALS.md` #10).
