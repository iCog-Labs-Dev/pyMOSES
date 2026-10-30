# Server round 3: pre-registration (2026-10-30)

**Status:** approved by the user (2026-10-30). Settings come from the literature review
`reports/Constant optimization in symbolic regression.md` (2026-10-30). Nothing is
implemented.

**Why:** in pyMoses GOMEA, `--gomea-contin copy` copies each numeric constant between
individuals and never changes it. Inside a deme, a constant can only take a value that the
initial population sampled. The user asked for a method for continuous domains with low
compute and acceptable accuracy. The proposal: keep GOM for the discrete knobs, keep linear
scaling, and test two cheap ways to change constants: a few Levenberg–Marquardt (LM) steps
on elite programs only, and coefficient mutation between GOM steps.

**What the literature predicts** (review, measured unless marked):
* LM and BFGS win per generation, but at equal time LM did not beat cheap coefficient
  mutation on any of 5 datasets (Harrison et al., GECCO 2023). So the equal-CPU phase is
  the decisive one.
* Constant tuning does not make models smaller (Harrison et al. 2023: models stay at the
  cap with more constants; dos Reis et al. 2024: 7.4 → 10.5 nodes).
* LM Jacobians are often ill-conditioned, more so for large trees (Kronberger 2022).
  pyMoses programs have 280–380 nodes, so LM can often fail or stall.
* Inside GP-GOMEA, coefficient mutation gave only a small test-accuracy change on noisy
  data (Virgolin and Bosman, GECCO 2022).
* Thus the expected result is a small gain or a tie. A tie means: keep linear scaling only. This round also answers an open backlog item: GOMEA-LS against HC-LS at equal
CPU time (`BACKLOG.md`, section 5). The user then added four backlog items (2026-10-30):
size control (1ap), the uniform start as the regression default (1aq), the four Virgolin
et al. datasets never run in pyMoses (Dow, Tow, WiR, WiW), and Yac (both in 1ar).

**What earlier phases predict:** linear scaling gave the largest single gain so far (1ag:
Con HC 60.4 → 38.6, GOMEA 37.1 → 25.9). GP-GOMEA with linear scaling and no constant tuning
reached about the same test NMSE as GOMEA-LS (1ai, ties) with 25–30-node models against
GOMEA's 280–380. No phase has tuned constants.

## Rules for the whole round

* As in rounds 1 and 2: on the server, one phase at a time, W from `resources.py`, state in
  `server-run3/`.
* **Statistics:** paired Wilcoxon signed-rank, Holm across a phase's primary contrasts,
  α = 0.025. "Better" also needs a 10 % lower median. Holm p in [0.025, 0.25) at stage 1
  extends that dataset (splits 11–20). Secondary contrasts never extend. Failed runs count
  as infinitely bad.
* **Datasets:** Con, EnH and Air, splits 1–10 (as 1ai). Metric: test NMSE. Also reported
  (D1): program size, train–test gap, CPU, and for LM arms the number of LM calls, the
  evaluations they used, and the share of calls that lowered the training error.
* **Base arms** (round-1 settings plus `--linear-scaling 1`):
  * GOMEA-LS: LT-B, N 16, deme cap 250, memo, `--gomea-contin copy`,
    `--knob-probe-baseline reduced`, `--gomea-skip-inactive 0`.
  * HC-LS: the 1ag HC-LS arm.
* **S0:** full pytest and the reproducibility check. With LM off, the new code must give
  byte-identical results to round 1.

## The constant-tuning code (new, off by default)

* **Flags:** `--const-opt {none|lm|mutate|polish}` (default none).
* **`lm` (elite-only LM):** at the end of each deme, LM on the deme's best program. All
  numeric constants, training rows only, training MSE after linear scaling, finite-difference
  Jacobian, Marquardt damping, at most 10 iterations, stop when the relative improvement
  is below 1e-6. Lamarckian: the tuned constants replace the old ones if the training error
  is lower. If the Jacobian is singular or LM fails, keep the old program and log it.
* **`mutate` (coefficient mutation):** after each GOM step on an individual, one Gaussian
  change to each constant of that individual in turn (σ = 0.1·|c|, at least 0.1), kept if
  the fitness is not worse (after Virgolin and Bosman 2022; their decaying temperature is
  left out for simplicity).
* **`polish` (end of run only):** LM as above, once, on the final best program. The tuned
  program replaces it only if the validation NMSE is not worse.
* **Cost accounting:** every scoring pass over the training rows counts as one evaluation
  against the run's budget (LM: one per constant plus one per Jacobian, plus trial steps;
  `mutate`: one per attempt). So no arm gets extra evaluations.
* **Logged per run:** LM calls, singular or failed calls, the share of calls that lowered
  the training error, and the share of the budget that tuning used.

## 1an. Does constant tuning help at equal evaluations?

* **Arms (10,000 evaluations):** GOMEA-LS, GOMEA-LS-LM, GOMEA-LS-MUT, GOMEA-LS-POLISH, HC-LS.
* **Primary (6):** GOMEA-LS-LM vs GOMEA-LS and GOMEA-LS-MUT vs GOMEA-LS, on each dataset.
* **Secondary:** GOMEA-LS-POLISH vs GOMEA-LS; GOMEA-LS-LM vs GOMEA-LS-MUT; GOMEA-LS vs HC-LS.
* **Rule for 1ao:** the tuning arm (LM or MUT) with the lower median test NMSE, pooled over
  the three datasets' medians, goes to 1ao. A script writes the choice and logs it.
* **Runs:** 150 (+ up to 150).

## 1ao. Equal CPU time

* **Arms (1,000 s CPU per run, as 1ah, no evaluation cap):** GOMEA-LS, GOMEA-LS-TUNE (the
  arm chosen in 1an), HC-LS.
* **Primary (6):** GOMEA-LS vs HC-LS (the open backlog item) and GOMEA-LS-TUNE vs GOMEA-LS,
  on each dataset.
* **Secondary:** GOMEA-LS-TUNE vs HC-LS.
* **Runs:** 90 (+ up to 90).

## 1ap. Size control: the acceptance penalty with linear scaling

* **Arms (10,000 evaluations):** GOMEA-LS, GOMEA-LS-PEN5 (`--gomea-accept-complexity-ratio
  0.005`, the Phase 1o value) and GOMEA-LS-PEN1 (`0.001`). Both penalty arms also use
  `--gomea-deme-store accepted` (BACKLOG bug 4.3: without it, the merge brings back the
  rejected, larger trials that the penalty rejected).
* **Datasets:** Con, EnH, Air.
* **Primary (6):** program size, each penalty arm vs GOMEA-LS, on each dataset ("better" =
  smaller, with a 10 % lower median).
* **Secondary:** test NMSE, each penalty arm vs GOMEA-LS, on each dataset.
* **Adoption rule (agent-proposed, after `DECISIONS.md`: a moderate accuracy loss is
  acceptable for smaller programs):** a strength qualifies if it is better on size on at
  least 2 of 3 datasets and its median test NMSE is at most 10 % higher on each. If both
  qualify, recommend the one with the smaller median size over the three datasets.
* **Runs:** 90 (+ up to 90).

## 1aq. The uniform start as the regression default

* **Arms (10,000 evaluations):** GOMEA-LS, GOMEA-LS-UNIF (`--gomea-radii uniform`).
* **Datasets:** Con, EnH, EnC (as 1v and 1ad, now with linear scaling).
* **Primary (3):** test NMSE, GOMEA-LS-UNIF vs GOMEA-LS, on each dataset.
* **Also reported:** size and CPU (1ad: about 1.8× both).
* **Runs:** 60 (+ up to 60).

## 1ar. Breadth: Yac and the four untested datasets at equal CPU time

* **Before the round (local, no server time):** a descriptive analysis of the round-1 Yac
  runs (exemplar growth, train–test gap, number of demes). It is reported, not tested.
* **Datasets:** Yac, Dow, Tow, WiR, WiW (splits 1–10). Dow, Tow, WiR and WiW are not in
  `data/gomea-sr/` yet. Their licences must be checked before they are committed. If one
  cannot be committed, the user copies its `full.csv` to the server, and `setup.sh` checks
  it against `SHA256SUMS`.
* **Arms (1,000 s CPU per run, no evaluation cap):** GOMEA-LS, GOMEA-LS-TUNE (the 1an
  choice), HC-LS.
* **Primary (5):** test NMSE, GOMEA-LS vs HC-LS, on each dataset. A separate Holm family
  from 1ao, so that 1ao keeps its power.
* **Secondary:** GOMEA-LS-TUNE vs GOMEA-LS, on each dataset.
* **Runs:** 150 (+ up to 150).

## Order

S0, 1an, 1ao, 1ap, 1aq, 1ar, one at a time. 1ao and 1ar need the 1an choice.

## Size and time

* At most 300 (1an) + 180 (1ao) + 180 (1ap) + 120 (1aq) + 300 (1ar) = 1,080 runs.
* At stage 1 only: 570 runs, about 150 CPU-hours, about 14 hours at W = 11, plus about
  1.3 hours for S0. With all extensions: about 300 CPU-hours, about 28 hours.
* Round 2 took 2.7× its estimate, so plan for up to 2–3 days. Tow (4,999 rows) and Dow
  (57 features) cost more per evaluation than the other datasets.

## Amendment (2026-10-30, before any implementation)

* Bug 4.3 is checked: it matters only with a penalty inside GOMEA's acceptance, and the
  existing flag `--gomea-deme-store accepted` (tested in Phase 1o, arm PEN-ACC) avoids it.
  The 1ap penalty arms use it.
* Phase 1o (no linear scaling) found that the 0.005 penalty raised Con's test NMSE from 35.2
  to 42.8. The user chose to add a weaker arm (0.001) to 1ap.
