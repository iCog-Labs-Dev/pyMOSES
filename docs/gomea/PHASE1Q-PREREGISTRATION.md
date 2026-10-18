# Phase 1q pre-registration: fixing knob collapse inside GOMEA

Written 2026-10-16, before any Phase 1q run.

**Requested by the user:** keep going through the backlog until it is finished, comparing
GOMEA with hill climbing (HC) under the same per-deme cap. This phase tests the knob-collapse
mitigation from `BACKLOG.md` section 4, item 1.

## Question

Does probing knobs against the *reduced* exemplar (`--knob-probe-baseline reduced`) help
GOMEA inside MOSES, in accuracy or in CPU?

* **Knob collapse:** classic MOSES behaviour, the same in C++. `disc_probe` assumes the
  exemplar is already a fixpoint of the reducer. Effort-2 Boolean reduction is not confluent,
  so once a flip wrapper lets the reducer find a simplification, almost every later knob is
  thrown away.
* **Frequency:** in Phase 1o's 10 current-GOMEA mux11 runs, 35 of 278 demes had fewer than
  30 % of the most search fields any earlier deme of the same run had. That is the measure
  used below.
* **Pilot (3 seeds, mux11):** the option removed the collapse. Accuracy did not change
  noticeably, and GOMEA's CPU rose 1.3–2×, because the demes that collapsed before are now
  full size.

## Code

The option ships with the merge of the four option patches, after Phase 1p. With the option
off, outputs are identical to the previous code (identity harness). The manifests record the
merge commit.

* **`--knob-probe-baseline reduced`:** this diverges from C++ and is opt-in. A setting is
  dropped only if reducing the decorated exemplar removes more complexity than reducing the
  undecorated one does. In symbols, it is dropped when
  `c(T_i) − c(R(T_i)) > c(T_0) − c(R(T_0))`. The classic rule drops it whenever
  `c(R(T_i)) < c(T_i)`.
* **Scope:** `disc_probe` is used only when building logical knobs. That is why both problems
  are Boolean; on regression problems the option changes nothing.

## Design (all arms capped at 250 evaluations per deme, `-m 40000`)

**Arms:**

| Arm | Settings |
|---|---|
| `LTB-C250-M` | current GOMEA: `-a gomea --gomea-fos lt-b --gomea-pop-size 16 --gomea-skip-inactive 0 --gomea-max-evals-per-deme 250 --gomea-genotype-memo 1` |
| `KR` | LTB-C250-M + `--knob-probe-baseline reduced` |
| `HC-C250` | `-a hc-capped --gomea-max-evals-per-deme 250` |

All three arms are rerun here, so CPU is measured under the same conditions.

**Problems:**

* **mux11** (`-H mux -k 3`).
* **pa5** (`-H pa -k 5`). Neither method solved it in Phase 1e at this budget, so the metric
  is final error. Its exemplars are large, so collapse is likely.

**Seeds:** 3001–3010. Extension: 3011–3020, only for a problem with an "extend" primary
contrast.

**Manifests:** `configs/gomea/phase1q/`. **Launcher:** `tools/gomea/phase1q_run.sh`, 18
workers on 22 logical cores. Runs use 60–625 MB each, and the arms interleave, so CPU
comparisons within the phase are fair.

## Analysis (`tools/gomea/phase1q_report.py`)

**Metrics, paired by seed:**

* **Accuracy:**
  * **mux11:** evaluations to a solution; an unsolved run counts as 40,000.
  * **pa5:** final best error (wrong rows).
* **Cost:** CPU seconds per run.

**Contrasts:**

* **Primary,** with Holm correction over 4: per problem, `cpu` and `accuracy` of KR vs
  LTB-C250-M.
* **Secondary,** with Holm correction over 2: per problem, `vs-hc`, KR accuracy vs HC-C250.
* **Descriptive:**
  * median search fields per GOMEA deme;
  * the number of collapsed demes, as defined above;
  * demes per run;
  * best complexity.

**Rule:** two looks, each at α = 0.025.

* **Better / worse:** Holm p < 0.025 and a median at least 10 % lower / higher.
* **Extend:** a primary contrast at stage 1 with 0.025 ≤ Holm p < 0.25.
* Otherwise a **tie**.
* **Stage 2:** the Holm family stays complete. A problem that was not extended enters with
  its stage-1 data. Only contrasts that were "extend" at stage 1 take a stage-2 verdict; all
  others keep their stage-1 verdict.

**Reading:**

* **Mitigation helps:** KR's accuracy is better on a problem, or its CPU is better with
  accuracy not worse.
* **Mitigation is a cost with no gain:** KR's CPU is worse and its accuracy is a tie. This
  is the outcome the pilot predicts.
