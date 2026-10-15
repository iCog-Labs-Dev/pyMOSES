# Phase 1m pre-registration: budget-derived demes in full MOSES runs

Written 2026-10-15, before any Phase 1m run.

**Requested by the user:** make the linkage data more informative, building on their design:
"generate 50 for example and then use that as our maximum population for this specific deme
through out the evaluation, the linkage should be built always using all of these
populations ... until the evaluation finishes", with each generation's model learned from the
current N programs. The user also wanted:

* the per-deme cap derived from the total budget (fewer, longer demes);
* a cheaper diverse start, compared at equal evaluations;
* not a cross-deme linkage model.

## Question

In full MOSES runs, does deriving each deme's population and cap from the budget improve GOMEA
and make learned linkage beat random linkage? Today's demes learn one linkage model from 16
unscored starting programs and rarely finish a generation. Phases 1h and 1i showed that
learned linkage beats random only with more data (N = 1,024) or a more diverse sample.

## Code

Commit `4e9b9ed` adds the options:

* `--gomea-deme-share D`: cap = `-m` / D.
* `--gomea-pop-fraction F`: N = max(2, floor(F × deme budget)).
* `--gomea-pop-max M`: clamps N.
* With the options off, results are identical to before (identity harness).

**Measured while building it:** at these knob counts (mux11 demes reach 500–1,600 knobs), no
generation completes under any tested setting. One generation costs about
0.25 × 2 × knobs × N evaluations. So in this design each deme still learns its linkage model
mainly from its starting population, but from N = 200 programs instead of 16. More programs
then get mixed within the larger cap.

## Design

**Arms:**

| Arm | Optimizer | Per-deme cap | Population | Linkage model | Start |
|---|---|---|---|---|---|
| `HC` | hill climbing | – | – | – | – |
| `G-C250-N16` | GOMEA (current tuned setup) | 250 | 16 | LT-B | default |
| `G-D10-LTB` | GOMEA | total / 10 | 0.2 × cap, at most 200 | LT-B | default |
| `G-D10-RT` | GOMEA | total / 10 | 0.2 × cap, at most 200 | random tree | default |
| `G-D10-LTB-WIDE` | GOMEA | total / 10 | 0.2 × cap, at most 200 | LT-B | distances 1, 2, 4, 8 knobs, equally weighted (`--gomea-radii 1:0.25,2:0.25,4:0.25,8:0.25`) |

* **All GOMEA arms:** `--gomea-genotype-memo 1 --gomea-skip-inactive 0`, plus
  `--gomea-contin copy` on regression.
* **The 0.2 population fraction follows the user's example** (cap 250 → 50).
* **The 200 clamp** keeps mux11 demes (cap 4,000) at the same N as regression demes
  (cap 1,000).

**Problems:**

* **mux11:** `benchmarks/gomea/runner.py`, `-m 40000`, seeds 3001–3010.
* **Con, EnH and EnC:** `tools/gomea/moses_sr_runner.py`, `-m 10000` (the Phase 1f part D
  budget), splits 1–10.
* **Extension, if borderline:** seeds 3011–3020 and splits 11–20.

**Manifests:** `configs/gomea/phase1m/{mux11,mux11-ext,sr,sr-ext}.json`, code `a42d5c8`.

**Run:** after Phase 1l, 12 workers, memory cap of 3 GB per run. The estimate is about
10–14 h for stage 1.

## Analysis (`tools/gomea/phase1m_report.py`)

**Metrics:**

* **mux11:** evaluations to a solution. An unsolved run counts as 40,000.
* **Regression:** test NMSE of the program chosen on validation.

**Contrasts** (paired Wilcoxon per problem):

* **Primary,** with Holm correction over all 8 (4 problems × 2):
  * `design`: G-D10-LTB vs G-C250-N16;
  * `linkage`: G-D10-LTB vs G-D10-RT.
* **Secondary,** with Holm correction over its 8:
  * `wide`: G-D10-LTB-WIDE vs G-D10-LTB;
  * `vs-hc`: G-D10-LTB vs HC.
* **Descriptive:** G-C250-N16 vs HC; per-arm medians, solved counts and CPU.

**Rule:** two looks, each at α = 0.025.

* **A better:** Holm p < 0.025 and a median at least 10 % lower.
* **A worse:** the mirror image.
* **Extend:** for primary contrasts only, 0.025 ≤ Holm p < 0.25 at stage 1.
* Otherwise a **tie**.

**Reading:**

* `design` better: budget-derived demes are the better use of the budget.
* `linkage` better: with N = 200, learned linkage finally beats random inside full runs.
* `wide` better: a moderately diverse start helps at no extra budget.
* `vs-hc`: whether the new design ties with or beats HC.
