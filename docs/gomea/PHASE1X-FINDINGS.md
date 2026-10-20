# Phase 1x findings: deme length (per-deme cap) by problem type

**Status:** complete. The run was on 2026-10-20 from 03:56 to 07:36 (mux11 30 runs, then
Con 30 runs). All runs ended `ok`. No primary contrast fell in the extension band, so there
was no extension. Protocol: `PHASE1U-1AA-PREREGISTRATION.md`. Code `311d8eb`. Results:
`phase1x-results/`, `../gomea-results/phase1x/` and `../gomea-results/1/phase1x-bool-*/`.

**Answer: keep the 250 cap.**

* Halving it to 125 makes mux11 **worse** (2.4× the CPU to solve).
* Doubling it to 500 was a tie on both problems: cheaper on mux11, less accurate on Con,
  neither significant.

## Results (medians, 10 pairs)

| Problem (metric) | Cap 125 | Cap 250 | Cap 500 |
|---|---|---|---|
| mux11: CPU to solve (s) | 3,490 | 1,464 | 1,000 |
| mux11: evaluations to solve | 5,166 | 5,538 | 6,934 |
| mux11: CPU per evaluation (s) | 0.63 | 0.29 | 0.12 |
| mux11: demes per run | 42.5 | 22.5 | 14.5 |
| Con: test NMSE | 48.5 | **37.1** | 41.8 |
| Con: CPU per run (s) | 2,517 | 2,052 | 1,963 |

**Primary contrasts** (each cap against 250; Holm over 4):

| Problem | Contrast | Lower / tie / higher | Holm p | Verdict |
|---|---|---|---|---|
| mux11 | cap 125 | 0 / 0 / 10 | 0.0078 | **worse** |
| mux11 | cap 500 | 6 / 0 / 4 | 0.252 | tie |
| Con | cap 125 | 3 / 0 / 7 | 0.252 | tie |
| Con | cap 500 | 2 / 0 / 8 | 0.252 | tie |

* **Extension band:** three Holm p-values sit just above the 0.25 extension threshold. Under
  the registered rule they are ties, and no extension was run.

## What this shows

1. **The cap sets how often MOSES rebuilds knobs, and knob building dominates CPU.**
   * Shorter demes mean more demes (42.5 at cap 125 against 22.5 at 250) and more knob
     rebuilding per evaluation, so CPU per evaluation doubles.
   * Longer demes halve it again (0.12 s at cap 500).
   * Evaluations to solve change much less (5,166 to 6,934).
2. **On mux11 a longer deme looks cheaper.** Cap 500 needed about 25 % more evaluations but
   about a third less CPU, with 6 of 10 seeds faster. It is not significant at 10 seeds.
   * Phase 1m found a 4,000 cap much worse (5.6× the evaluations), so the useful range is
     narrow.
3. **On Con, 250 is the best of the three by median.** Both shorter and longer demes had
   higher test error (7 and 8 of 10 splits), though not significantly.
4. **Practical reading:** the 250 cap stays. For Boolean problems, a cap of 500 is a cheap
   follow-up candidate if CPU matters more than evaluations.

## Limits

* Three caps, 10 pairs, two problems.
* The cap-500 contrasts ended right at the extension boundary. They may resolve with more
  seeds; see `BACKLOG.md`.
