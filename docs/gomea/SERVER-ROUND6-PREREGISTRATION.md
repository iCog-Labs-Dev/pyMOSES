# Server round 6: pre-registration (2026-11-06)

**Status:** the design is approved by the user (see `APPROVALS.md`). Not implemented yet.

**Question:** two follow-ups to rounds 4 and 5 (`BACKLOG.md`, section 7): does LM-VAL still help
with no per-deme cap, and does PEN125 make LM-VAL's programs smaller?

## Rules

* As in rounds 1 to 5: on the server, one phase at a time, W from `resources.py`, state in
  `server-run6/`. Round 6 never overlaps with another round.
* **Statistics:** paired Wilcoxon signed-rank, Holm across the 12 primary contrasts, α = 0.025.
  "Better" also needs a 10 % lower median. Holm p in [0.025, 0.25) at stage 1 extends that dataset
  (splits 11–20). Secondary contrasts never extend. Failed runs count as infinitely bad.
* **Reported (D1):** test NMSE, program size, train–test gap, CPU, LM and guard statistics, deme
  count and generations per deme.
* **S0:** full pytest and the reproducibility check.
* **No new code.** The guard (`--const-opt-guard val`) and the penalty exist.

## 1aw. LM-VAL with no cap, and LM-VAL with PEN125 (1,000 s)

* **Base flags (all arms):** `-a gomea --gomea-fos lt-b --gomea-pop-size 16 --gomea-skip-inactive 0
  --gomea-genotype-memo 1 --gomea-contin copy --knob-probe-baseline reduced --linear-scaling 1`.
* **Arms:**
  * **NOCAP:** the base flags only (no `--gomea-max-evals-per-deme`), the base of later rounds.
  * **LMVAL:** NOCAP + `--const-opt lm --const-opt-guard val`.
  * **LMVAL-PEN125:** LMVAL + `--gomea-accept-complexity-ratio 0.125 --gomea-deme-store accepted`.
  * **LMVAL-C250:** LMVAL + `--gomea-max-evals-per-deme 250` (the LM-VAL arm of 1at).
* **Datasets:** WiR, Con, EnH, Air, Yac, Tow. Splits 1–10, extension 11–20.
* **Primary (12):**
  * test NMSE, LMVAL vs NOCAP, on each dataset (6);
  * program size, LMVAL-PEN125 vs LMVAL, on each dataset (6).
* **Secondary (12):**
  * test NMSE, LMVAL vs LMVAL-C250, on each dataset (6);
  * test NMSE, LMVAL-PEN125 vs LMVAL, on each dataset (6).
* **Adoption rules:**
  1. LM-VAL stays the preferred LM variant with no cap if LMVAL is better than NOCAP on at least 3
     of 6 datasets and worse on none.
  2. PEN125 joins LM-VAL if LMVAL-PEN125 is better on size on at least 4 of 6 datasets and its
     median test NMSE is at most 10 % higher on each.
  3. If LMVAL-C250 is better than LMVAL on at least 3 datasets, the report says that the no-cap
     base costs LM accuracy. The user decides if LM runs keep a cap.
* **Runs:** 240 (+ up to 240).

## Size and time

* Stage 1: 240 runs, about 67 CPU-hours, about 6 hours at W = 11, plus about 1.3 hours for S0.
* With all extensions: 480 runs, about 12 hours.
