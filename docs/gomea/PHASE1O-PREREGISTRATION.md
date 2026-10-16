# Phase 1o pre-registration: a complexity penalty inside GOMEA's acceptance only

Written 2026-10-16, before any Phase 1o run.

**Requested by the user:** run backlog item 13 ("Sure let's do that as well"). Per the user's
standing rule, GOMEA is compared with hill climbing under the same per-deme cap.

## Question

Can GOMEA's cost per evaluation be cut by curbing its exemplar growth, without losing
accuracy?

* **Phase 1n:** with the same 250-evaluation cap, GOMEA beat capped hill climbing (HC) per
  evaluation, but cost about 2.75× the CPU per evaluation on mux11.
* **The exemplar-growth analysis:**
  * GOMEA's exemplars grow about 2× faster, because a capped GOM deme accepts the first
    not-worse trial.
  * MOSES's default complexity penalty (`-z 3.5`, 0.29 per complexity unit) is negligible
    against these score scales.
  * Knob-building cost grows with exemplar size.
* **Pilot:** a global `-z 0.125` halved exemplar size but changed MOSES's exemplar selection
  and broke one seed.

## Code

Commit `14707df`; with both options off, outputs are identical (identity harness).

* **`--gomea-accept-complexity-ratio R`:** GOMEA's own comparisons use raw score − complexity/R.
  These are GOM acceptance, forced improvement, elite and stagnation. MOSES's scoring, merge
  and exemplar selection are unchanged.
* **`--gomea-deme-store accepted`:** only the initial population and accepted individuals are
  kept in the deme handed back to MOSES; every evaluation is still counted.

**One-seed pilot (mux11, seed 3001, 6,000 evaluations):**

| Setting | CPU vs default | Result |
|---|---|---|
| R = 0.125 | −31 % | solved at 5,500 evaluations (default 5,250) |
| R = 0.125 plus accepted-only storage | −51 % | not solved |

## Penalty strength, scaled per problem

The same relative strength on both problems: about 0.4 % of the score range per complexity
unit.

* **mux11:** the raw score is −(wrong rows) out of 2,048, so R = 0.125, i.e. 8 rows per unit.
* **Con:** the raw score is −SSE on the training split (516 rows, target scaled to sd 10, so a
  constant predictor scores about −51,600). R = 0.005, i.e. 200 per unit.

## Design (all arms capped at 250 evaluations per deme)

**Arms:**

| Arm | Settings |
|---|---|
| `LTB-C250-M` | current GOMEA: `-a gomea --gomea-fos lt-b --gomea-pop-size 16 --gomea-skip-inactive 0 --gomea-max-evals-per-deme 250 --gomea-genotype-memo 1`, plus `--gomea-contin copy` on Con. Rerun here so CPU is measured under the same conditions. |
| `PEN` | LTB-C250-M + `--gomea-accept-complexity-ratio R` |
| `PEN-ACC` | PEN + `--gomea-deme-store accepted` |
| `HC-C250` | `-a hc-capped --gomea-max-evals-per-deme 250`, reused from Phase 1n (accuracy only; same code path, unaffected by the new options) |

The Con extension runs HC-C250 for splits 11–20.

**Problems:**

* **mux11:** `-m 40000`, seeds 3001–3010, extension 3011–3020.
* **Con:** `-m 10000`, splits 1–10, extension 11–20.

**Manifests:** `configs/gomea/phase1o/`. 10 workers, nothing else running.

## Analysis (`tools/gomea/phase1o_report.py`)

**Metrics:**

* **Accuracy:** evaluations to solve on mux11 (unsolved = 40,000); test NMSE on Con.
* **Cost:** CPU seconds per run.

**Contrasts** (paired Wilcoxon):

* **Primary,** with Holm correction over 4: per problem, `cpu` and `accuracy` of PEN vs
  LTB-C250-M.
* **Secondary,** with Holm correction over 6: per problem,
  * `acc-store` and `cpu-store` (PEN-ACC vs PEN);
  * `vs-hc` (PEN accuracy vs HC-C250).

**Rule:** two looks at α = 0.025.

* **A better / worse:** Holm p < 0.025 and a median at least 10 % lower / higher.
* **Extend:** for primary contrasts at stage 1, 0.025 ≤ Holm p < 0.25.
* Otherwise a **tie**.

**Reading:**

* **The penalty works** if PEN's `cpu` is "better" and its `accuracy` is not "worse".
  Accuracy "better" is a bonus.
* **Equivalence:** a tie in accuracy does not prove equivalence. The medians and CIs are
  reported.

**Descriptive:** median final best complexity (mux11), solved counts, CPU.
