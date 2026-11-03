# Server round 4: pre-registration (2026-11-03)

**Status:** the design is approved by the user (see `APPROVALS.md`). The configs are not yet
written. The search code does not change.

**Question:** what happens if we remove the 250-evaluation deme cap, and use other knob caps
than K = 16?

**Why:** round 2 (1ak) kept the 250 cap in every arm, and only K = 16 went through the main
phases. The cap gave 3–5 generations per deme, but KC and KC-STATIC were worse than GOMEA on all
three problems. The report's interpretation was that 16 of 300–1,700 knobs is too few. It is not
tested. This round tests larger K, and it removes the evaluation cap. Round 2 also had no linear
scaling.

**What earlier phases predict:** nothing clear. Longer demes give GOM more generations, but a run
visits fewer exemplars. Round 2 and the linkage investigation found no gain from learned linkage.

## Rules

* As in rounds 1 to 3: on the server, one phase at a time, W from `resources.py`, state in
  `server-run4/`.
* **Statistics:** paired Wilcoxon signed-rank, Holm across the phase's primary contrasts,
  α = 0.025. "Better" also needs a 10 % lower median. Holm p in [0.025, 0.25) at stage 1 extends
  that problem (splits or seeds 11–20). Secondary contrasts never extend. Failed or unsolved runs
  count as infinitely bad.
* **Power at stage 1:** the family has 20 primary contrasts. With 10 pairs, the lowest possible
  raw p is 0.00195, so the lowest Holm p is 0.039. So no primary contrast can get a verdict of
  better or worse at stage 1. A large effect goes to "extend", and it gets its verdict at 20 pairs.
* **Reported (D1):** test NMSE, program size, train–test gap, CPU, and generations per deme, number
  of demes, how demes end.
* **S0:** full pytest and the reproducibility check.

## 1as. Knob caps with and without the evaluation cap

* **Base arguments of every arm:** `-a gomea --gomea-fos lt-b --gomea-pop-size 16
  --gomea-skip-inactive 0 --gomea-genotype-memo 1 --knob-probe-baseline reduced`. Regression also
  has `--gomea-contin copy --linear-scaling 1`. There is no constant tuning.
* **Arms:**

| Arm | `--gomea-knob-cap` | `--gomea-max-evals-per-deme` |
|---|---|---|
| BASE | off | 250 |
| NOCAP | off | 0 (none) |
| K8 | 8 | 0 |
| K16 | 16 | 0 |
| K32 | 32 | 0 |
| K64 | 64 | 0 |
| K16-C250 (the round-2 setting) | 16 | 250 |

* **Knob selection in all K arms:** `--gomea-knob-select explore --gomea-knob-rank0 contrast`, with
  the default explore count.
* **Problems and budgets:** Con, EnH, Air (`--max-time 1000`, no evaluation limit, splits 1–10, then
  11–20) and mux11 (`--max-time 3600`, seeds 3001–3010, then 3011–3020).
* **Metrics:** test NMSE for regression, and CPU to solve for mux11.
* **Primary (20):** NOCAP, K8, K16, K32 and K64, each vs BASE, on each problem.
* **Secondary (8):** K16-C250 vs BASE, and K16 vs K16-C250, on each problem. The first contrast
  connects to round 2. The second contrast is the direct effect of removing the cap at K = 16.
* **Adoption rule (agent-proposed):** a setting qualifies if it is better than BASE on at least 2
  of the 3 regression datasets, and it is not worse on any of the 4 problems. If several qualify,
  recommend the one with the smallest median size over the regression datasets. If none qualifies,
  keep BASE.

## Size and time

* Stage 1: 7 arms × 40 tasks = 280 runs. At most 128 CPU-hours (58 for regression, 70 if no mux11
  run solves early), about 12 hours at W = 11, plus about 1.3 hours for S0.
* With all extensions: 560 runs, about 24 hours. Plan for one weekend.
