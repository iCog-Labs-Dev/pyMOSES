# Phase 1c findings: how stand-alone GP-GOMEA scales

**Status: complete.** There were 300 confirmatory runs, all `ok`: 200 GP-GOMEA and 100
classic C++ MOSES. There were also 40 descriptive GP-GOMEA runs at height 9. After one
session interruption, 15 in-flight runs were re-run from scratch; see the pre-registration's
deviations.

## Implementation and comparison

* **GP-GOMEA:** the authors' C++ code (`750220c`, the Phase 1b build). IMS starting at
  population 64, LT FOS, AND/OR/NOT, and a per-size template height chosen in the pilot.
* **Reference:** classic C++ MOSES `-a hc` (`5a5bf75`), untuned, with 10 seeds per size.
* **Tasks:** truth tables built from MOSES's own target functions.
* **Budget:** 500k evaluations per run. MOSES was also limited to 25 minutes.
* **Protocol:** `PHASE1C-PREREGISTRATION.md`.

| Size | GP-GOMEA solved (of 20) | MOSES solved (of 10) | GP-GOMEA ERT | MOSES ERT | Median error, GPG / MOSES |
|---|---|---|---|---|---|
| pa3 | 20 | 9 | 10.2k | 19.9k | 0 / 0 |
| pa4 | 19 | 4 | 160k | 186k | 0 / 2 |
| pa5 | 2 | 0 † | 4.5M | — | 2 / 6 |
| pa6 | 0 | 0 † | — | — | 12.5 / 17 |
| pa7 | 0 | 0 † | — | — | 37 / 64 |
| maj5 | 20 | 7 | 26.5k | 9.3k | 0 / 0 |
| maj7 | 14 | 2 † | 416k | 833k | 0 / 1 |
| maj9 | 0 | 0 † | — | — | 27.5 / 15.5 |
| mux6 | 20 | 10 | 38.2k | 1.9k | 0 / 0 |
| mux11 | 16 | 10 | 359k | 8.8k | 0 / 0 |

† **Every unsolved MOSES run at these sizes stopped on its 25-minute limit.** The median
evaluations reached ran from 87k (maj9) to 395k (pa7), short of 500k
(`moses_time_limit_censoring.csv`). GP-GOMEA's runs at these sizes took a median of 2–5
minutes. **These rows therefore match neither evaluations nor time.**

## What the evidence supports

Verdicts under the pre-registered rule, read literally:

* **Majority: GP-GOMEA scales better.** Its ERT grows about 4.0× per added input, against
  9.4× for MOSES. It solves maj7 14/20 against 2/10, but that is not significant after
  Holm (p 0.11), and MOSES's maj7 runs were time-limited.
* **Multiplexer: GP-GOMEA scales worse.** ERT grows 1.6× against 1.4× per input. MOSES
  solves mux11 in every run at about 8.8k evaluations per success; GP-GOMEA needs about
  359k (p_holm 5e-5). That is 20–40× fewer evaluations for MOSES on this family.
* **Parity: GP-GOMEA scales worse by the growth criterion,** 21× against 9.3× per input.
  Its only significant success advantage is at the small size pa4 (19/20 vs 4/10, p_holm
  0.019). *Caveat (ours, not in the rule):* the two fits span different sizes. GP-GOMEA's
  includes pa5, which only it solved (2/20).

**Neither system solves pa6, pa7 or maj9.** On pa6 and pa7 GP-GOMEA's error is lower; on
maj9 MOSES's is lower.

**Height 9** (descriptive):
* larger templates helped on pa5 (4/10 vs 2/20);
* they lowered the median error on pa6 (12.5 → 8.5) and maj9 (27.5 → 16);
* they did worse on mux11 (7/10 vs 16/20), where fewer generations fit in the budget.

*Interpretation:* template size limits GP-GOMEA on parity and majority, while larger
templates cost generations. Population size was never varied here.

## Strengths, weaknesses and costs

**GP-GOMEA's strengths:**
* **Throughput.** It completes 500k evaluations in minutes.
* **Majority and small parity.** It solves more instances.

**GP-GOMEA's weaknesses:**
* **Multiplexer.** Much less evaluation-efficient.
* **Parity growth.** Steeper ERT growth.
* **Tuning.** Template height needs tuning per size, while MOSES ran untuned.

**MOSES:** very efficient on the multiplexer, but its classic C++ build slows sharply on
large parity and majority representations. *Interpretation:* knob-building and
reduction cost.

## Decision

GP-GOMEA's scaling advantage is **family-specific**: better on majority, worse on
multiplexer and parity by the pre-registered rule.

The next justified experiments:
* MOSES without the time limit, for equal evaluations at the largest sizes;
* GP-GOMEA with adaptive template height;
* more MOSES seeds.

Evidence: `phase1c-results/` and `../gomea-results/phase1c/`. Reproduce with
`configs/gomea/phase1c/`, the runners and `phase1c_report.py`.
