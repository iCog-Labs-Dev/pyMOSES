# Server round 5: pre-registration (2026-11-03)

**Status:** the design is approved by the user (see `APPROVALS.md`). Nothing is implemented.

**Question:** three follow-ups to round 3: can a validation guard stop LM's overfitting, does a
weaker size penalty keep the accuracy, and does a size penalty work with LM?

## Rules

* As in rounds 1 to 4: on the server, one phase at a time, W from `resources.py`, state in
  `server-run5/`. Round 5 starts after round 4 ends, and it never overlaps with round 4.
* **Statistics:** paired Wilcoxon signed-rank, Holm across a phase's primary contrasts, α = 0.025.
  "Better" also needs a 10 % lower median. Holm p in [0.025, 0.25) at stage 1 extends that dataset
  (splits 11–20). Secondary contrasts never extend. Failed runs count as infinitely bad.
* **Reported (D1):** test NMSE, program size, train–test gap, CPU, LM statistics.
* **S0:** full pytest and the reproducibility check.
* **Base arms:** GOMEA-LS and LM (`--const-opt lm`) as in round 3.

## New code: `--const-opt lm` with a validation guard (off by default)

* **Flag:** `--const-opt-guard {none|val}` (default none), with `--const-opt-val FILE`.
* **Behaviour:** as `lm`, but the tuned constants replace the old ones only if the training error
  is lower and the validation NMSE is not worse. Linear scaling is fitted on the training rows.
* **Cost:** each validation pass counts as one evaluation, as in round 3.

## 1at. The validation guard (1,000 s)

* **Arms:** GOMEA-LS, LM, LM-VAL.
* **Datasets:** WiR, Con, EnH, Air, Yac, Tow.
* **Primary (6):** test NMSE, LM-VAL vs LM, on each dataset.
* **Secondary (6):** test NMSE, LM-VAL vs GOMEA-LS, on each dataset.
* **Also reported:** the gap, and the share of LM results that the guard rejected.
* **Adoption rule:** LM-VAL qualifies if its median gap on WiR is lower than LM's, and it is not
  worse than LM on any dataset.
* **Runs:** 180 (+ up to 180).

## 1au. A weaker size penalty (10,000 evaluations, as 1ap)

* **Arms:** GOMEA-LS, PEN25, PEN125, PEN625 (`--gomea-accept-complexity-ratio` 0.025, 0.125, 0.625,
  all with `--gomea-deme-store accepted`). The penalty is `complexity / R`, so a larger R is weaker.
* **Datasets:** Con, EnH, Air.
* **Primary (9):** program size, each penalty arm vs GOMEA-LS, on each dataset.
* **Secondary (9):** test NMSE, the same contrasts.
* **Adoption rule (as 1ap):** a strength qualifies if it is better on size on at least 2 of 3
  datasets and its median test NMSE is at most 10 % higher on each. If several qualify, recommend
  the one with the smallest median size over the three datasets.
* **Runs:** 120 (+ up to 120).

## 1av. LM with a size penalty (1,000 s)

* **Arms:** GOMEA-LS, LM, LM-PEN5, LM-PEN25 (R 0.005 and 0.025, with the accepted store).
* **Datasets:** Con, EnH, Air.
* **Primary (6):** program size, each LM-PEN arm vs LM, on each dataset.
* **Secondary (6):** test NMSE, the same contrasts.
* **Adoption rule:** as 1au, against LM.
* **Why fixed strengths:** no script picks a value from 1au, so this phase does not depend on it.
* **Runs:** 120 (+ up to 120).

## Size and time

* Stage 1: 420 runs, at most 117 CPU-hours, about 11 hours at W = 11, plus about 1.3 hours for S0.
* With all extensions: 840 runs, about 22 hours.
