# Phase 1p findings: GOMEA vs capped hill climbing at equal wall-clock time

**Status:** complete.

* **Stage 1:** 2026-10-16 17:22 to 2026-10-17 09:27 (Con 30 runs, then mux11 30 runs).
* **Extension:** both problems, to 12:02 (60 runs).

All 120 runs ended `ok`. Protocol: `PHASE1P-PREREGISTRATION.md`. Code `23c63b6` (no code
changed between it and the launch). Results: `phase1p-results/`, `../gomea-results/phase1p/`
and `../gomea-results/1/phase1p-mux11*/`.

**Answer:** with the same time and the same 250-evaluation cap per deme, GOMEA is **better**
than capped hill climbing (HC) on regression (Con) and **worse** on the Boolean problem
(mux11). The user's goal, "at least tie", is met on Con and not on mux11.

## Setup

* **Arms**, all `--max-time T -m 1000000000` with a cap of 250 evaluations per deme:
  * `HC-C250`: `-a hc-capped`.
  * `LTB-C250-M`: current GOMEA (LT-B, N = 16, memo; `--gomea-contin copy` on Con).
  * `PEN` (exploratory): LTB-C250-M + `--gomea-accept-complexity-ratio` (0.125 on mux11,
    0.005 on Con).
* **Problems:** mux11 with T = 3,600 s, seeds 3001–3020; Con with T = 1,000 s, splits 1–20.
* **Load:** 10 workers on 22 logical cores. At launch two agent pilots also used about 5
  cores (see the pre-registration's launch note).

## Results

**Medians over 20 pairs:**

| Problem (metric) | LTB-C250-M | PEN | HC-C250 |
|---|---|---|---|
| mux11: wall time to solve (s; unsolved = 3,600) | 2,568 | 1,725 | **1,311** |
| mux11: solved | 16 / 20 | 20 / 20 | 18 / 20 |
| mux11: evaluations done | 4,395 | 6,694 | 8,125 |
| mux11: demes per run | 18 | 27 | 32.5 |
| mux11: final best complexity | 41.5 | 30 | 27 |
| Con: test NMSE | **43.2** | 43.8 | 68.4 |
| Con: evaluations done in 1,000 s | 5,667 | 8,645 | 19,535 |

**Primary contrast** (`vs-hc`, LTB-C250-M vs HC-C250; Holm correction over the 2 problems):

| Problem | Stage | GOMEA lower / tie / HC lower | Holm p | Verdict |
|---|---|---|---|---|
| Con | 1 (10 splits) | 7 / 0 / 3 | 0.065 | extend |
| Con | 2 (20 splits) | 16 / 0 / 4 | 0.0003 | **GOMEA better** |
| mux11 | 1 (10 seeds) | 1 / 1 / 8 | 0.055 | extend |
| mux11 | 2 (20 seeds) | 3 / 2 / 15 | 0.0056 | **GOMEA worse** |

The mux11 ties are seeds that neither arm solved (3005, and at stage 2 also 3016), so both
arms count 3,600 s.

**Secondary contrasts** (Holm over 4). All four were ties at stage 1, and secondary contrasts
are not eligible for extension, so these are the registered verdicts:

| Problem | Contrast | Stage 1: lower / higher | Holm p | Verdict |
|---|---|---|---|---|
| Con | PEN vs HC-C250 | 7 / 3 | 0.20 | tie |
| Con | PEN vs current | 6 / 4 | 0.86 | tie |
| mux11 | PEN vs HC-C250 | 4 / 6 | 0.86 | tie |
| mux11 | PEN vs current | 8 / 2 | 0.32 | tie |

**Descriptive, with all 20 pairs:**

* **Con, PEN vs HC:** lower in 15 of 20 (Holm p 0.0028).
* **Con, PEN vs current:** 10 / 10 (p 0.90).
* **mux11, PEN vs HC:** 8 / 12 (p 0.90).
* **mux11, PEN vs current:** 15 / 5 (Holm p 0.098). PEN solved all 20 runs.

## What this shows

1. **On regression, GOMEA wins on time, not just on evaluations.**
   * GOMEA had 37 % lower median test error than capped HC on Con, at equal time.
   * It did so with about 3.4× fewer evaluations (5,667 against 19,535 in 1,000 s).
   * So each GOMEA evaluation costs much more, but buys more.
   * This agrees with Phase 1n (equal evaluations) and Phase 1j (equal time against
     uncapped HC).
2. **On mux11, GOMEA's cost per evaluation still decides it.**
   * At equal evaluations GOMEA needed about 0.58× HC's evaluations (Phase 1n).
   * At equal time it did about 0.54× as many evaluations and needed about 2× the wall time
     to solve (median 2,568 s against 1,311 s).
   * Here, capping HC does not stop it from winning on time.
3. **The complexity penalty (PEN) narrows the mux11 gap but does not close it.**
   * Its median time to solve was 1,725 s, against 2,568 s for current GOMEA and 1,311 s
     for HC.
   * It solved all 20 runs, against 16 for current GOMEA and 18 for HC.
   * Neither difference is significant (Holm p 0.098 and 0.90). These are descriptive
     directions from an exploratory arm.
   * On Con, PEN's error matched current GOMEA's.
4. **Where to go next for mux11:** lower GOMEA's per-evaluation cost. Most of it is knob
   building (about 68 % of CPU; see `BACKLOG.md` items 1.x), which grows with exemplar
   size. PEN's smaller exemplars (complexity 30 against 41.5) are consistent with that.

## Limits

* One Boolean and one regression problem, one cap (250) and one time limit per problem.
* Wall-clock results depend on the machine and its load. All arms ran interleaved on the same
  machine.
* The Con runner records no commit in its manifest. The mux11 manifest records `23c63b6`.
