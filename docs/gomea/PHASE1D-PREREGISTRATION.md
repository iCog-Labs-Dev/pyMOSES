# Phase 1d pre-registration: why did MOSES hill climbing beat embedded GOMEA?

Written 2026-09-27 **before any Phase 1d run**. The commit that adds this file is the
timestamp.

**Requested by the user:** "The next experiment should be trying to figure out why this is
the case." After the agent proposed the design below, the user replied "yes go ahead with
the full design" (`APPROVALS.md` #8). The hypotheses and criteria are agent-proposed.

## Motivation (Phase 1 telemetry, maj5 LT-B, 532 demes)

* **Near-copy starting population.** A median deme has 127 knobs and a population of 16.
  87 % of knob values equal the exemplar's.
* **Mixing rarely helps.** 93 % of mixing trials change nothing. Of the evaluated trials,
  5 % improve.
* **Diversity collapses.** Per-knob entropy falls from 0.08 to 0.003 bits within a deme,
  and a deme ends with a median of 2 distinct genotypes out of 16.
* **Early stops.** About half the demes end by convergence.
* **Wasted evaluations.** 32–47 % of charged requests re-score a duplicate genotype.

## Hypotheses and manipulations

Each manipulation is applied to the Phase 1 LT-B configuration (N = 16, sampler
initialization, non-worse acceptance, no memo, no cap), one factor at a time.

| ID | Hypothesis | Arm | Manipulation |
|---|---|---|---|
| A1 | The initial population is too uniform | `LTB-WIDE` | `--gomea-radii 2:0.25,4:0.25,8:0.25,max:0.25` |
| A2 | Same, via an unconstrained initial population | `LTB-UNIF` | `--gomea-init uniform` |
| B1 | The population is too small, so it collapses | `LTB-N32` | `--gomea-pop-size 32` |
| B2 | Same | `LTB-N64` | `--gomea-pop-size 64` |
| C | Re-scoring duplicates wastes budget | `LTB-MEMO` | `--gomea-genotype-memo 1` (a GOMEA-only mechanism test) |
| D | GOMEA spends too long per deme, so MOSES gets fewer demes | `LTB-CAP` | `--gomea-max-evals-per-deme 500` (stage 2 only) |
| E | Accepting equal-score changes causes drift and collapse | `LTB-STRICT` | `--gomea-accept strict` |

Reference arms: `HC` (stock `-a hc`), `UNI` (`--gomea-fos uni`) and `LT-B`, all as in
Phase 1. The common GOMEA flags are `-a gomea --gomea-pop-size 16 --gomea-skip-inactive 0`
unless an arm overrides them.

## Stage 1: frozen representations (inner optimizer only)

**Setup.** The 11 frozen exemplars from `configs/gomea/phase1/frozen_fixtures.json`
(4 pa4, 4 maj5, 3 mux6), each run as `-e <exemplar> -g 1 -m 5000`. Seeds 1201–1210, so
there are 110 units per arm. The arms are HC, UNI, LT-B, LTB-WIDE, LTB-UNIF, LTB-N32,
LTB-N64, LTB-MEMO and LTB-STRICT: 990 runs.

**Primary metric.** The best error (wrong rows) reached **by 1,000 and by 5,000
evaluations**, taken from each run's scorer trace.

**Mechanism metrics.**
* distinct genotypes at the end of the deme;
* the share of demes ending by convergence;
* per-knob entropy after generation 1;
* the share of improving trials;
* the harness duplicate-call share, measured identically for HC;
* memo hits.

**Tests.** Each factor arm is compared with `LT-B` by a paired Wilcoxon test over the
(fixture, seed) units within a task family, with W/T/L. Holm correction runs across the
factor arms × the two checkpoints within each family.

**Carry-forward rule** (decides which factors enter stage 2). A factor is carried if, at
either checkpoint, it is significantly better than LT-B (Holm p < 0.05) in at least one
family and significantly worse in none. D is carried regardless, because it only acts
end-to-end. The combination arm `LTB-COMBO` applies every carried factor together. When
two variants of one hypothesis are both carried (A1/A2 or B1/B2), the combination uses the
one with the better median primary metric summed over families.

## Stage 2: end-to-end MOSES

**Setup.** maj5 and mux6 with seeds 1001–1030, and pa4 with seeds 1001–1020. `-m 20000`,
the Phase 1 tasks and budget.

**Arms.** `LTB-CAP` plus each carried factor arm, plus `LTB-COMBO`. They are compared with
the **Phase 1 `HC`, `LT-B` and `UNI` runs, reused**, which is valid only if the new code
reproduces Phase 1 exactly with default options. A regression test pins that, and for
stage 2 we also re-run and compare 3 Phase 1 LT-B runs.

**Tests.** The Phase 1 pre-registered rule: paired Wilcoxon on final error and on censored
evaluations-to-success, Holm within each task, and the practical thresholds (10 percentage
points of success, 20 % fewer evaluations). Contrasts are each new arm vs `LT-B` and each
new arm vs `HC`.

**Interpretation rule.**
* A factor **explains part of the gap** if its arm is *better* than LT-B on at least one
  task.
* It **closes the gap** on a task if its arm is *tie* or *better* against HC there.
* The fact-checked findings note will say which hypotheses the data supports, contradicts,
  or leaves undetermined.
