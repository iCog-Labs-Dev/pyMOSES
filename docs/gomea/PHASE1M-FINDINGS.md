# Phase 1m findings: budget-derived demes in full MOSES runs

**Status:** complete. Stage 1 ran on 2026-10-15: regression from 11:43 to 18:28 (150 runs), then
mux11 until 20:53 (50 runs). All runs ended `ok`. No primary contrast was borderline, so no
extension ran. Protocol: `PHASE1M-PREREGISTRATION.md`. Code `a42d5c8`. Results:
`phase1m-results/`, `../gomea-results/phase1m/` and `../gomea-results/1/phase1m-mux11-*/`.
The regression manifest has no commit field (that runner does not check one); those runs
used the same working tree, `a42d5c8`.

**Answer:** no gain in search quality.

* **mux11:** deriving each deme's cap and population from the budget made GOMEA much
  **worse**.
* **Con, EnH, EnC:** it **tied** the current tuned GOMEA, at 37–64 % less CPU per run.
* **Learned vs random linkage:** no significant difference under this design. On mux11 the
  direction favoured LT-B: lower in 8 of 10 pairs, Holm p 0.34.
* **Wider start:** no help.

## Setup

* **Arms:**
  * `HC`: uncapped hill climbing. The phase was registered before the rule that HC must get
    GOMEA's cap; Phase 1n repeats the comparison with capped HC.
  * `G-C250-N16`: the current tuned GOMEA (cap 250, N = 16, LT-B, memo).
  * `G-D10-LTB` / `G-D10-RT`: cap = total budget / 10; N = 0.2 × the deme budget, at most 200.
  * `G-D10-LTB-WIDE`: as `G-D10-LTB`, with starting distances 1, 2, 4 and 8 equally likely.
* **Budgets:** mux11 `-m 40000`; Con, EnH and EnC `-m 10000`.
* **Pairs:** 10 per problem.

**Effective settings (mux11 telemetry):**

| Arm | Per-deme cap | Population | Demes per run (median) |
|---|---|---|---|
| G-D10 arms | 4,000 | 200 (61–200 for WIDE, where the remaining budget was smaller) | median 8–10 (range 7–10) |
| G-C250-N16 | 250 | 16 | median 22 |

**Generations:**

* **G-D10-LTB and G-D10-RT:** every deme that started mixing stopped inside its first
  generation, after at most 62 % (median about 11 %) of a full generation's trials.
* **G-D10-LTB-WIDE:** 6 of 104 demes (107–429 knobs) completed 3–7 generations and ended on
  stagnation or convergence.
* **G-C250-N16:** 1 of 278 demes (25 search fields) ran 6 generations.
* **Target reached at start:** three demes reached the target in their starting population.

## Results (medians)

| Problem (metric) | HC | G-C250-N16 | G-D10-LTB | G-D10-RT | G-D10-LTB-WIDE |
|---|---|---|---|---|---|
| mux11 (evaluations to solve; unsolved = 40,000) | 7,953 | **5,345** | 30,006 | 36,494 | 40,000 (0/10 solved) |
| Con (test NMSE) | 54.9 | **37.1** | 41.6 | 51.4 | 54.2 |
| EnC (test NMSE) | 27.1 | 20.4 | **19.0** | 20.1 | 21.5 |
| EnH (test NMSE) | 26.8 | **15.7** | 15.9 | 16.4 | 18.9 |
| CPU per run, median (s): mux11 / Con / EnC / EnH | 295 / 422 / 714 / 992 | 2,151 / 2,451 / 3,794 / 4,724 | 1,690 / 1,549 / 1,573 / 2,034 | 1,595 / 1,484 / 1,537 / 1,684 | 1,153 / 844 / 541 / 691 |

On mux11, G-D10-RT solved 8 of 10 runs; every other arm except WIDE solved 10 of 10.

### Primary contrasts

Holm correction over 8; "A lower / tie / B lower" counts pairs.

| Problem | `design` (G-D10-LTB vs G-C250-N16) | `linkage` (G-D10-LTB vs G-D10-RT) |
|---|---|---|
| mux11 | 0 / 0 / 10, Holm p 0.016: **worse** | 8 / 0 / 2, p 0.049, Holm p 0.34: tie |
| Con | 3 / 0 / 7, Holm p 1.0: tie | 6 / 0 / 4, Holm p 1.0: tie |
| EnC | 5 / 0 / 5, Holm p 1.0: tie | 5 / 0 / 5, Holm p 1.0: tie |
| EnH | 3 / 0 / 7, Holm p 1.0: tie | 5 / 0 / 5, Holm p 1.0: tie |

### Secondary contrasts

Holm correction over 8.

| Problem | `wide` (WIDE vs G-D10-LTB) | `vs-hc` (G-D10-LTB vs HC) |
|---|---|---|
| mux11 | 0 / 0 / 10: **worse** (Holm p 0.016) | 0 / 0 / 10: **worse** (Holm p 0.016) |
| Con | 3 / 0 / 7: tie | 7 / 0 / 3: tie |
| EnC | 6 / 0 / 4: tie | 7 / 0 / 3: tie |
| EnH | 3 / 0 / 7: tie | 9 / 0 / 1, Holm p 0.059: tie |

### Descriptive: current GOMEA vs uncapped HC

The current GOMEA (G-C250-N16) had lower error or needed fewer evaluations than uncapped HC in
7 of 10 pairs on mux11 (p 0.38) and in 10, 8 and 9 of 10 pairs on Con, EnC and EnH (p 0.002,
0.014 and 0.006, uncorrected). These two arms are deterministic reruns of Phase 1f part D's HC
and LT-B runs: they give exactly the same test NMSE on all 30 regression splits. So this
repeats Phase 1f exactly; it is not an independent replication.

## What this shows

1. **On mux11, long demes cost about 5.6× more evaluations to solve** (ratio of medians;
   5.8× per pair).
   * With a 4,000-evaluation cap, runs get 7–10 exemplar rebuilds instead of about 22.
   * Each deme spends most of its budget mixing inside its first generation.
   * **Hypothesis, not tested here:** the loss comes from fewer rebuilds around better
     exemplars. The cap, N and the number of demes changed together, so a larger, unselected
     population spending its budget on mixing could explain it equally well.
2. **On regression the budget-derived design matches the current one at much lower cost.**
   * There is no significant difference in test error.
   * CPU per run is 37–64 % lower. On mux11 this goes with fewer demes (about 8 against 22)
     and fewer knobs built per run (about 6,600 against 19,200). The regression runs have no
     deme telemetry, so less knob building is the likely but unmeasured cause there.
   * Con is the weakest case: median 41.6 against 37.1, 3 pairs better and 7 worse, not
     significant.
3. **A larger population (N = 200) did not make learned linkage beat random linkage.**
   * Each deme still learns its linkage model once, from its unselected starting sample;
     no generation completes.
   * This agrees with Phase 1h, where N had to reach 1,024 on a frozen problem.
   * mux11 leaned towards LT-B (8 of 10 pairs, 10/10 solved against 8/10), but not
     significantly.
4. **The wider start hurt or did nothing.** On mux11 it never solved the problem. Its CPU was
   the lowest. On mux11 its exemplars were smaller (best complexity median 27.5 against
   37.5), which likely contributed, though it built about as many knobs per run (6,410
   against 6,596).
5. **Practical reading:**
   * For Boolean problems, keep short demes (cap 250).
   * For regression, budget-derived demes are a way to cut cost without losing accuracy.
     Whether that holds at equal wall-clock time, and against capped HC, is open; Phase 1n
     tests capped HC.

## Limits

* 10 pairs per problem and one setting per option (D = 10, F = 0.2, N ≤ 200).
* HC was uncapped here.
* Under G-D10-LTB and G-D10-RT, no generation completes at these knob counts, so the
  design could not show what several generations per deme would do.
