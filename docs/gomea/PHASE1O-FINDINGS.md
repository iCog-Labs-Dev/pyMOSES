# Phase 1o findings: a complexity penalty inside GOMEA's acceptance only

**Status:** complete, on 2026-10-16:

* **Stage 1:** 10:43 to 14:24 (Con 30 runs, mux11 30 runs).
* **Con extension:** to 16:12 (splits 11–20, 40 runs, including capped HC).

All runs ended `ok`. Protocol: `PHASE1O-PREREGISTRATION.md`. Code `14707df`. Results:
`phase1o-results/`, `../gomea-results/phase1o/` and `../gomea-results/1/phase1o-mux11-*/`.

**Answer:** the penalty cuts cost, but probably at some accuracy.

* **Con, registered results:**
  * PEN lowered CPU per run by 37 % (stage 2, **better**).
  * Its accuracy contrast was a **tie** at stage 1 (Holm p 0.39). It was not eligible for
    extension, so that is its registered verdict.
* **Con, descriptive:** the extension data (20 splits, collected for the CPU contrast) show
  PEN's test error worse in 15 of 20 splits (raw p 0.003). This is a likely accuracy cost,
  not a registered finding.
* **mux11:** nothing was significant at 10 seeds. The direction was cheaper but slower to
  solve.
* **Against capped HC:** at equal evaluations, PEN still beats it on Con, while using about
  3.6× its CPU.

## Setup

* **Cap:** 250 evaluations per deme in every arm.
* **Arms:**
  * `LTB-C250-M`: current GOMEA, rerun here for comparable CPU.
  * `PEN`: + `--gomea-accept-complexity-ratio R`. R = 0.125 on mux11 and 0.005 on Con, about
    0.4 % of the score range per complexity unit.
  * `PEN-ACC`: + `--gomea-deme-store accepted`.
  * `HC-C250`: capped hill climbing. Reused from Phase 1n for mux11 and Con splits 1–10; run
    here for Con splits 11–20.
* **Budgets:** mux11 `-m 40000`; Con `-m 10000`.

## Results

**Medians** (mux11: 10 seeds; Con: 20 splits):

| Problem (accuracy metric) | LTB-C250-M | PEN | PEN-ACC | HC-C250 |
|---|---|---|---|---|
| mux11: evaluations to solve (all solved) | **5,345** | 7,145 | 7,458 | 7,978 |
| mux11: CPU per run (s) | 1,880 | 1,345 | 1,448 | 1,118 |
| mux11: final best complexity | 37 | 29.5 | 29 | 27 |
| Con: test NMSE | **35.2** | 42.8 | 44.0 | 70.2 |
| Con: CPU per run (s) | 1,881 | 1,176 | 739 | 329 |

* mux11's HC-C250 column covers Phase 1n's 20 seeds; the pairs use only the 10 shared seeds
  (paired-seed median 7,306).
* The reused HC-C250 runs (mux11, and Con splits 1–10) come from Phase 1n's code (`8b1e728`);
  the other runs ran on `14707df`. The Con manifests record no commit.

**Primary contrasts** (PEN vs LTB-C250-M). Holm correction is over the registered family of
4; for stage 2, mux11's stage-1 p-values were used.

| Problem | Contrast | Stage | PEN lower / tie / higher | Holm p | Verdict |
|---|---|---|---|---|---|
| Con | cpu | 1 (10 splits) | 8 / 0 / 2 | 0.11 | extend |
| Con | cpu | 2 (20 splits) | 17 / 0 / 3 | 0.00084 | **PEN better** (−37 %) |
| Con | accuracy | 1 | 4 / 0 / 6 | 0.39 | **tie** (registered; not eligible for extension) |
| Con | accuracy | 20 splits, descriptive | 5 / 0 / 15 | (0.0095) | — |
| mux11 | cpu | 1 (10 seeds) | 7 / 0 / 3 | 0.39 | tie |
| mux11 | accuracy | 1 | 4 / 0 / 6 | 0.70 | tie |

**Corrections:**

* The stage-2 report script corrected over Con's contrasts only (giving 0.0004 and 0.0031).
  The values above use the registered family. No verdict changes.
* The pre-registration does not say how a contrast that was not extended is treated once its
  problem gets more data. Here, only the extended contrast gets a stage-2 verdict.

**Secondary contrasts:**

* **Registered verdicts:** all six were ties at stage 1, and secondary contrasts are not
  eligible for extension. The mux11 secondaries all have Holm p 1.0.
* **Descriptive, with the 20 Con splits** (Holm over the registered 6):

| Contrast | Lower / tie / higher | Holm p |
|---|---|---|
| PEN-ACC vs PEN, CPU | 16 / 0 / 4 | 0.0043 |
| PEN-ACC vs PEN, accuracy | 6 / 0 / 14 | 0.61 |
| PEN vs HC-C250, accuracy (equal evaluations; PEN used about 3.6× the CPU) | 18 / 0 / 2 | 0.00011 |

## What this shows

1. **The mechanism lowers cost.**
   * **Con:** CPU fell 37 % (registered, better).
   * **mux11:** median CPU fell about 28 %, and median final best complexity from 37 to 29.5.
     Neither is significant at 10 seeds.
   * **Exemplar size:** Con's summaries do not record complexity, so smaller exemplars are
     shown on mux11 only.
2. **On Con, the chosen strength probably costs accuracy.**
   * Registered: a tie at 10 splits.
   * Descriptive: with 20 splits, median test NMSE was 42.8 against 35.2, worse in 15 of 20.
   * PEN is still far better than capped HC at equal evaluations (70.2).
   * A weaker penalty might keep more accuracy; this phase tested one strength per problem.
3. **Keeping only accepted individuals saved more CPU on Con** (−37 % over PEN; descriptive,
   since it was a stage-1 tie).
   * Its accuracy was worse than PEN's in 14 of 20 splits, median +2.7 %, not significant.
   * On mux11 it used slightly *more* CPU than PEN (+7.6 %).
4. **On mux11, 10 seeds were not enough to separate the arms.** The directions match Con:
   cheaper, but needing more evaluations.

## Limits

* One penalty strength per problem, two problems, equal evaluations (not equal time).
* mux11 was not extended under the registered rule.
