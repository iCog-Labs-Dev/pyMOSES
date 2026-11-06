# Server round 7: pre-registration (2026-11-06)

**Status:** the design is approved by the user (see `APPROVALS.md`). Implemented in `09fbaf0` to `aab55a6`. Not run yet.

**Question:** a confirmation round. On data that no earlier selection used, does the final
combined setting beat MOSES hill climbing, and does it beat plain GOMEA-LS?

## Rules

* As in rounds 1 to 6: on the server, one phase at a time, W from `resources.py`, state in
  `server-run7/`. Round 7 starts after round 6 ends (the chain, without a pause) and never
  overlaps with it.
* **Fresh splits:** splits 21–30 on all 10 datasets. Rounds 1 to 6 used splits 1–20 only. No later
  splits exist, so this round has **no extension**.
* **Statistics:** see "Contrasts". Failed runs count as infinitely bad.
* **Reported (D1):** test NMSE, program size, train–test gap, CPU, LM and guard statistics, deme
  count and generations per deme.
* **S0:** full pytest and the reproducibility check.
* **No new code under `pymoses`.** New tooling: the choice of FINAL and the pooled test.

## The choice of FINAL (from round 6)

A script reads the final 1aw report and writes the choice into the round 7 state.

| Round 6 result | FINAL |
|---|---|
| Rule 2 of 1aw passes: LMVAL-PEN125 is better than LMVAL on size on at least 4 of 6 datasets, and its median test NMSE is at most 10 % higher on each | the LMVAL-PEN125 arm of 1aw |
| Otherwise | the LMVAL arm of 1aw |

Both arms have no per-deme cap. Rules 1 and 3 of 1aw do not change FINAL (user, 2026-11-06: "keep
the chain without a pause").

## 1ax. Confirmation (1,000 s)

* **Arms:**
  * **FINAL:** as chosen above.
  * **NOCAP:** `-a gomea --gomea-fos lt-b --gomea-pop-size 16 --gomea-skip-inactive 0
    --gomea-genotype-memo 1 --gomea-contin copy --knob-probe-baseline reduced --linear-scaling 1`.
  * **HC-LS:** `-a hc --knob-probe-baseline reduced --linear-scaling 1` (no cap, the same cap as
    GOMEA).
* **Datasets:** Air, Bos, Con, Dow, EnC, EnH, Tow, WiR, WiW, Yac. Splits 21–30.

## Contrasts

* **Primary (2), pooled over the 100 pairs (10 datasets × 10 splits):**
  * test NMSE, FINAL vs HC-LS;
  * test NMSE, FINAL vs NOCAP.
  * **Test:** paired Wilcoxon signed-rank on r = log(NMSE_A / NMSE_B), two-sided, Holm across the
    2 contrasts, α = 0.025. A failed run gets the largest rank against it.
  * **Verdict:** "better" if Holm p < 0.025 and exp(median r) ≤ 0.9; "worse" if Holm p < 0.025 and
    exp(median r) ≥ 1/0.9; otherwise "tie".
* **Secondary (20), descriptive:** the same two contrasts on each dataset (paired Wilcoxon on test
  NMSE, Holm across the 20). With 10 pairs and 20 contrasts the lowest possible Holm p is 0.039, so
  these give no verdicts. They show where FINAL wins or loses.

## Size and time

* 300 runs, about 83 CPU-hours, about 7.6 hours at W = 11, plus about 1.3 hours for S0.
* **Risk:** Dow has 57 inputs. No run of Dow without the cap exists. The smoke test runs Dow.
