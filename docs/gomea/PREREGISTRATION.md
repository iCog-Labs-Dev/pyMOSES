# Phase 1 confirmatory pre-registration

Written 2026-09-24, after the pilot (`pilot-a`) and **before any confirmatory run**. It is
committed together with the confirmatory manifests; the commit that adds this file is the
pre-registration timestamp. Nothing below may be changed after the first confirmatory
result is seen. Any later change must be recorded as a deviation, with a reason, in the
findings note.

The criteria and statistics are **agent-proposed**. The user approved the compute envelope
(`APPROVALS.md` #3) and asked for the runs to be executed on their server, but did not set
a practical-effect threshold (see `APPROVALS.md`).

## Questions (from 03_EXPERIMENT_PROTOCOL.md §1)

* Q1 (primary, practical): does MOSES with the GOMEA inner optimizer (LT-B) solve Boolean
  synthesis tasks better than MOSES with its matched hill climber (HC), under an equal
  evaluation budget?
* Q2 (primary, mechanism): does the initialization-biased linkage tree (LT-B) beat ordinary
  MI (LT-MI) and the univariate FOS (UNI)?
* Q3 (secondary): does random hierarchical linkage (RT) differ from learned linkage? Does
  the forced-improvement reconstruction (LT-B+FI) change LT-B?
* Q4 (secondary, inner optimizer only): on identical frozen representations with an equal
  local envelope, which inner optimizer finds better programs?

## Frozen configuration (chosen from the pilot; see `pilot-a/`)

| Setting | Value | Why |
|---|---|---|
| Arms | HC (`-a hc`, stock), UNI, RT, LT-MI, LT-B, LT-B+FI | protocol §1 core five, plus the FI variant |
| GOMEA population N | **16** | best or tied-best on all 3 pilot tasks. N=32/64 were uniformly worse (fewer generations per budget) |
| Per-deme evaluation cap | **none** (MOSES passes the remaining global budget, as for HC) | the 2k cap was no better in the pilot (maj5: 2/5 vs 3/5 solved) |
| Initial radii | `1:0.4,2:0.3,4:0.2,max:0.1` (the default) | not varied in the pilot. Note: the `max` radius can exceed HC's `-D` search distance, which is a design difference |
| Dedup / coverage | on / on (defaults) | not varied |
| skip-inactive (phenotype-proof skip) | **off** | measured about 10 % of wall time for about 0 skips; neutral single-field changes are rare in MOSES representations |
| Stall generations / max generations | 5 / 100 (defaults) | not varied |
| Forced improvements | off, except in the LT-B+FI arm (`--gomea-fi 1`) | protocol §1: FI is tested as a separate arm |
| MOSES outer settings | all classic defaults (`-l WARN` only affects logging) | unchanged outer policy |
| Score cache | **off in both arms**. The demo problems build their scorer with cache size 0 | symmetric; every request is a real evaluation |

## Tasks, seeds and budgets

| Manifest | Tasks | Seeds | Budget |
|---|---|---|---|
| `confirm-e2e.json` | maj5 (`-H maj -k 5`), mux6 (`-H mux -k 2`) | 1001–1030 | `-m 20000` evaluations |
| `confirm-e2e-pa4.json` | pa4 (`-H pa -k 4`) | 1001–1020 | `-m 20000` |
| `confirm-frozen.json` | 11 frozen exemplars (`frozen_fixtures.json`): 4 pa4, 4 maj5, 3 mux6; each is `-e <exemplar> -g 1` | 1101–1110 | `-m 5000` (one deme; the local envelope) |

The seeds are disjoint from the pilot (501–505) and the fixture capture (901–902). The
wall cap is 3600 s per run (7200 s for pa4). A run that hits the cap is recorded as
*censored*, never dropped. The primary budget is evaluations: one call of the MOSES scorer,
identical in both arms. CPU seconds are reported as a secondary cost. Wall time is only
descriptive, because the host's hybrid cores make it unreliable.

## Metrics

* **Success:** best raw score = 0 (zero wrong rows on the full truth table) within the
  budget.
* **Final error:** −(best raw score). Error or no-best runs get the documented per-task
  worst sentinel.
* **Evaluations-to-success:** unsuccessful runs are right-censored at the budget, and at
  budget+1 for rank tests.
* **Secondary:** CPU seconds, peak RSS, and the complexity of the best program.
* **Frozen comparison:** best raw score after the one deme, plus evaluations used. HC stops
  by itself and its unused envelope is reported, not reassigned.

## Primary contrasts and tests

Primary contrasts, per task: **LT-B vs HC**, **LT-B vs LT-MI**, **LT-B vs UNI**. Secondary:
RT vs LT-B, and LT-B+FI vs LT-B.

* **Paired unit:** the seed. Each seed gives the same initial MOSES state for both arms;
  the trajectories then diverge.
* **Tests:** a two-sided Wilcoxon signed-rank test on final error, and one on
  evaluations-to-success (censored as above). Holm correction across all contrasts ×
  metrics within a task. We also report W/T/L, the paired median difference with a
  bootstrap 95 % CI (10 000 resamples, seed 12345), and success rates with Wilson 95 % CIs.
* **Practical threshold (agent-proposed):** a method is called *better* on a task only if
  the Holm-adjusted p < 0.05 **and** its success rate is at least 10 percentage points
  higher, or its median evaluations-to-success is at least 20 % lower. Otherwise the result
  is reported as a tie, or as "worse" by the mirrored rule.
* **Frozen comparison:** the unit is a fixture × seed. A paired Wilcoxon test is run per
  task family on best raw score. It is descriptive and secondary; local calls are not
  independent MOSES runs.

## What would count as a negative result

If LT-B is not *better* than HC on any task under the rule above, the finding is "no
benefit under matched evaluation budgets". The report must then say where GOMEA lost:
per-deme cost, representation turnover, or linkage quality. The frozen comparison helps
separate those. A negative result is a completed finding.

## Pilot evidence that motivated the choices (not confirmatory)

`pilot-a` ran 5 seeds, 105 runs, with commit `07ea018` code. Success counts (HC vs the best
LT-B setting, N=16):

* maj5: 4/5 vs 3/5
* mux6: 5/5 vs 5/5 (median evaluations to success 1.9k vs 3.2k)
* pa4: 1/5 vs 0/5

Full tables are in `docs/gomea/pilot-a/`.
