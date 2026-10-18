# Phase 1r pre-registration: seeding the metapopulation with GP-GOMEA elites

Written 2026-10-18, before any Phase 1r run.

**Requested by the user:** backlog item 9 (the user's idea, 2026-10-14). Continue through the
backlog, reach conclusions faster, and compare GOMEA with hill climbing (HC) under the same
per-deme cap.

## Question

Does starting MOSES from GP-GOMEA's elite programs, instead of from the empty program alone,
lower test error at the same total budget? The question is asked for both MOSES optimizers
(capped HC and GOMEA).

## Code

Commit `a1c2ce7` (merged and identity-checked; unseeded arms are unchanged).
`tools/gomea/moses_sr_runner.py` runs seeding when an arm has a `seed` block:

1. **GP-GOMEA run:** on `train_scaled.csv` (the file MOSES reads), with 20 % of the run's
   10,000 evaluations. It stops after the generation that reaches that share, so it may
   overshoot slightly (2,042 evaluations in the Con pilot).
   * Settings: tree height 3, functions `+ - *`, top 5 distinct elites.
   * Division is excluded because seeds with it gave an infinite test error.
2. **MOSES run:** with the remaining evaluations, started with `-e SEED1 … -e SEED5 -e +`, the
   five seeds plus the empty program.
3. **Accounting:**
   * `n_evals_total` and `cpu_s` include GP-GOMEA's evaluations and CPU.
   * `lineage.json` records whether the reported program descends from a seed.

**Pilots (Con, split 1):**

| Arm | Seeds | Seeded test NMSE | Unseeded test NMSE |
|---|---|---|---|
| Capped HC | with `/` | 51.6 | 71.2 |
| GOMEA | `+ - *` | 46.4 (best seed alone: 46.1) | — |

* With `/`, seeded GOMEA's reported program had an infinite test error.
* In both pilots the reported program descended from a seed.

## Design (all arms: 10,000 evaluations in total, cap 250 per deme)

**Arms:**

| Arm | Settings |
|---|---|
| `HC-C250` | `-a hc-capped --gomea-max-evals-per-deme 250` |
| `HC-C250-S` | HC-C250, seeded |
| `LTB-C250-M` | current GOMEA (LT-B, N = 16, memo, `--gomea-contin copy`) |
| `LTB-C250-M-S` | LTB-C250-M, seeded |

**Problems and splits:**

* **Problems:** Con and EnH, from the Virgolin et al. 2021 datasets used since Phase 1f.
  These are two of Phase 1m's three regression problems; EnC is left out to keep the phase
  short.
* **Splits:** 1–10. Extension: 11–20, only for a problem with an "extend" primary contrast.

**Control:** the unseeded arms are the control. The empty program is also in every seeded
start, but MOSES's exemplar selection may never pick it, so it is not the control.

**Manifests:** `configs/gomea/phase1r/`. **Launcher:** `tools/gomea/phase1r_run.sh`, 18
workers. It starts automatically when Phase 1q logs "all done".

**Boolean problems are left out of this phase.**

* The tooling supports them: `gpgomea_seed.py --bool` writes the truth table and translates
  AND/OR/NOT.
* In a mux11 pilot, seeded capped HC (seed 3001) needed 16,623 evaluations to solve, about
  twice capped HC's unseeded median (7,978, Phase 1n).
* Leaving them out keeps the phase short. A Boolean phase can follow if regression seeding
  helps.

## Analysis (`tools/gomea/phase1r_report.py`)

**Metric:** test NMSE of the candidate the runner reports, paired by split. A failed run
counts as infinite.

**Contrasts:**

* **Primary,** with Holm correction over 4: per problem, `seed-hc` (HC-C250-S vs HC-C250)
  and `seed-gomea` (LTB-C250-M-S vs LTB-C250-M).
* **Secondary,** with Holm correction over 2: per problem, `seeded-gomea-vs-hc`
  (LTB-C250-M-S vs HC-C250-S).
* **Descriptive:**
  * the best seed's own test NMSE;
  * the number of reported programs that descend from a seed;
  * final program size;
  * CPU per run;
  * runs with an infinite test error.

**Rule:** two looks, each at α = 0.025.

* **Better / worse:** Holm p < 0.025 and a median at least 10 % lower / higher.
* **Extend:** a primary contrast at stage 1 with 0.025 ≤ Holm p < 0.25.
* Otherwise a **tie**.
* **Stage 2:** the Holm family stays complete. A problem that was not extended enters with
  its stage-1 data, and only contrasts that were "extend" at stage 1 take a stage-2 verdict.

**Reading:**

* **Seeding helps an optimizer:** its `seed-*` contrast is "better" on a problem.
* **MOSES adds value beyond the seeds:** the seeded arm's test NMSE is below the best seed's
  own (descriptive).
