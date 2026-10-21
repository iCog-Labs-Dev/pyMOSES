# Phases 1ac–1ag: pre-registration (remaining backlog, from 2026-10-21)

**Requested by the user (2026-10-21):** run the remaining backlog. The rules of `QUEUE.md`
apply unchanged:

* one experiment at a time, 16 workers;
* mux11 CPU to solve at `-m 40000`, Con / EnH / EnC test NMSE at 10,000 evaluations;
* two looks at α = 0.025 with Holm correction over each phase's primary family;
* extend (seeds 3011–3020 or splits 11–20) only when stage 1 gives 0.025 ≤ Holm p < 0.25;
* "better" also needs a median at least 10 % lower.

**Base settings:**

* GOMEA is LT-B, N = 16, cap 250, memo, with `--gomea-contin copy` on regression.
* HC is `hc-capped` at the same cap.
* Every arm passes `--knob-probe-baseline reduced`.

**Reporting (D1):** every regression result reports, next to test NMSE, the final program
size, the train–test gap (test − train and test / train) and the CPU per run. D1 governs
adoption, not the registered verdicts.

**Where baselines are not rerun:** regression runs are deterministic. 1y and 1ab reproduced
earlier arms exactly. So a phase that already has a deterministic baseline on the same splits
and code path does not rerun it, as stated per phase below. Boolean phases always rerun their
baselines, because CPU is measured.

## 1ac. Confirm PEN's CPU-to-solve gain on mux11 (backlog 16)

* **Question:** does the acceptance penalty (`--gomea-accept-complexity-ratio 0.125`) cut
  GOMEA's CPU to solve mux11?
* **Seeds:** fresh seeds 3031–3040, extended with 3041–3050. 3001–3030 were used in 1o and
  1s.
* **Arms:** GOMEA and PEN.
* **Primary contrast:** PEN vs GOMEA, CPU to solve.
* **Runs:** 20 (+20).

## 1ad. Uniform start on a second and third regression dataset, and its combinations (backlog 17)

* **Primary questions:**
  * Does `--gomea-radii uniform` (UNI) lower test NMSE on **EnH** and **EnC**, as it did on
    Con (1v)?
  * Arms: GOMEA and GOMEA-UNI on EnH and on EnC, splits 1–10.
  * Two primary contrasts: UNI vs GOMEA on each dataset.
* **Secondary question (Con, splits 1–10):** can UNI's larger programs be kept small?
  * Arms: GOMEA-UNI, GOMEA-UNI-S10 (seeding at a 10 % share, as in 1y) and GOMEA-UNI-PEN
    (acceptance ratio 0.005, as in 1o).
  * Secondary contrasts: UNI-S10 vs UNI and UNI-PEN vs UNI.
  * Reported against 1v's GOMEA and GOMEA-UNI and 1y's GOMEA-S10, all deterministic on the
    same splits.
* **Runs:** 70 (+ extensions of EnH / EnC only).

## 1ae. Per-deme caps 250, 500 and 1,000 (backlog 18)

* **Arms:** GOMEA at caps 250, 500 and 1,000, on mux11 (seeds 3001–3010) and Con
  (splits 1–10).
* **Primary contrasts:** 500 vs 250 and 1,000 vs 250, on each problem (4 contrasts).
* **Also reported:**
  * generations per deme, and the share of demes that start a second generation (is the
    linkage model relearned?);
  * on Con, size and gap.
* **Runs:** 60 (+ extensions).

## 1af. Long demes for promising exemplars (backlog 19; code first)

**New opt-in options (built by a subagent, merged after 1ae):**

* `--gomea-long-deme-cap C`: a deme selected by the rule gets cap C instead of the usual cap.
* `--gomea-long-deme-rule top`: a deme is long when its exemplar is the metapopulation's
  best (by the selection score) at the moment it is chosen, and that exemplar has not had a
  long deme before.
* `--gomea-long-deme-rule random --gomea-long-deme-prob p`: each deme is long with
  probability p, drawn from the run's RNG.

**Design:**

* **Long cap:** C = 2,000 and N = 16 on both problems. This gives about 8–9 full GOM
  generations at about 13·N evaluations per generation. On Con a single long deme uses 20 %
  of the budget.
* **Arms:**
  * GOMEA, the current setting;
  * LONG-TOP;
  * LONG-RAND;
  * LONG-TOP-RT, which is LONG-TOP with `--gomea-fos rt`.
* **Matching the random control:**
  * p is set per problem to the median share of long demes in a 2-seed LONG-TOP pilot
    (seeds 3001–3002 / splits 1–2).
  * The pilot runs before stage 1 and is not part of the data.
* **Primary contrast:** LONG-TOP vs GOMEA, on mux11 and Con.
* **Secondary contrasts:**
  * LONG-TOP vs LONG-RAND: does the choice of exemplar matter?
  * LONG-TOP vs LONG-TOP-RT: does learned linkage beat a random tree once it is relearned?
* **Telemetry:** generations per deme, and the share of long demes that complete 2 or more
  generations.
* **Runs:** 80 (+ extensions).

## 1ag. Linear scaling in the regression fitness (backlog 10; code first)

**Background:**

* The search currently scores raw predictions. Linear scaling (least-squares a + b·f on the
  training data) is applied only when the error is reported (`tools/gomea/sr_eval.py`).
* In the literature, scaling the fitness gave larger gains than building-block libraries.

**New opt-in option (built by a subagent, merged after 1ae):**

* `--linear-scaling 1` for regression scoring.
* The fitness is the error of the best a + b·f fit.
* The behavioural score, the complexity penalty and the reported metric are otherwise
  unchanged. The report already scales, so test NMSE stays comparable.

**Design:**

* **Problem and splits:** Con, splits 1–10.
* **Arms:** HC (capped 250), HC-LS, GOMEA and GOMEA-LS.
* **Primary contrasts:** HC-LS vs HC and GOMEA-LS vs GOMEA, on test NMSE.
* **Not rerun:** HC and GOMEA are deterministic reruns of 1n / 1v on the same splits. They are
  reused if the code check shows the scoring path unchanged; otherwise they are rerun.
* **Runs:** 20–40 (+ extensions).

## Also in this round, not an experiment

**Backlog 15:** make `--knob-probe-baseline reduced` the code default.

* It has run in every arm of 1u–1ab, and no contrast attributable to it was a loss.
* It is a disclosed DIVERGENCE, merged after 1ae.
* All arms already pass the flag explicitly, so no result changes.

## Not queued

* **Shared linkage model across demes (backlog 2.3):** it needs a way to map knobs between
  demes, which is a design decision for the user.
* **Equal-time phases (2.7):** excluded by the queue rules; CPU to solve is the time measure.
* **Engineering items 1.3–1.7:** no experiment is needed.

## Order and estimate

* **Order:** 1ac → 1ad → 1ae (no code needed), then the merge of the 1af / 1ag code and
  item 15, then 1af → 1ag.
* **Time:** about 9–11 h on a quiet machine without extensions.
