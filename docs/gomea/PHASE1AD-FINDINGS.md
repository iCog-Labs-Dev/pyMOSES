# Phase 1ad findings: uniform start on EnH and EnC, and keeping its programs small on Con (backlog 17)

**Status:** complete.

* **Run:** 2026-10-21 12:19 (restart) to 2026-10-22 15:19, 110 runs, 16 workers, all `ok`.
  The machine rebooted overnight; the queue resumed on 2026-10-22 at 09:13. The runner kept
  47 finished runs and reran the rest, which is deterministic (see `QUEUE.md`).
* **Extensions:** EnH and EnC both had stage-1 Holm p 0.098, inside the extension band, so
  splits 11–20 were added for both.
* **Code:** `17a7bd0`, code check ok. Every arm used `--knob-probe-baseline reduced`.
* **Protocol:** `PHASE1AC-1AG-PREREGISTRATION.md`.
* **Results:** `phase1ad-results/` (reports, D1 extras, summary hashes) and
  `../gomea-results/phase1ad/`.

**Answer: not confirmed.** On EnH and EnC the uniform start (`--gomea-radii uniform`, UNI)
lowered median test NMSE by about 20 %, lower in 14 of 20 splits on each. Both primary
contrasts are registered **ties** (Holm p 0.053 and 0.031, against α 0.025). On Con, neither
seeding at a 10 % share nor the acceptance penalty kept UNI's accuracy: both are
**worse** (0/10 splits), though they make programs 3–10× smaller.

## Primary: UNI vs GOMEA (splits 1–20)

| Problem | Arm | Test NMSE | Train NMSE | Test − train | Size (nodes) | CPU per run |
|---|---|---|---|---|---|---|
| EnH | GOMEA | 15.6 | 14.3 | 1.5 | 409 | 3,192 s |
| EnH | GOMEA-UNI | **12.8** | 11.3 | 1.5 | 708 | 5,929 s |
| EnC | GOMEA | 19.2 | 17.1 | 2.2 | 371 | 3,255 s |
| EnC | GOMEA-UNI | **15.7** | 13.7 | 2.4 | 732 | 5,834 s |

| Contrast | UNI lower / higher | p | p (Holm) | Verdict |
|---|---|---|---|---|
| EnH: UNI vs GOMEA | 14 / 6 | 0.053 | 0.053 | **tie** |
| EnC: UNI vs GOMEA | 14 / 6 | 0.015 | 0.031 | **tie** |

## Secondary: keeping UNI small on Con (splits 1–10)

| Arm | Test NMSE | Train NMSE | Test − train | Size | CPU per run | vs UNI (lower / higher, Holm p) |
|---|---|---|---|---|---|---|
| GOMEA-UNI | **29.2** | 24.2 | 4.4 | 309 | 2,492 s | — |
| GOMEA-UNI-PEN (0.005) | 37.2 | 33.8 | 3.8 | 92 | 1,028 s | 0 / 10, 0.0039: **A worse** |
| GOMEA-UNI-S10 (10 % seeds) | 43.2 | 37.5 | 2.7 | 30 | 494 s | 0 / 10, 0.0039: **A worse** |

Reference, same splits (deterministic): 1v GOMEA 37.1 and GOMEA-UNI 29.2 (309 nodes, which
1ad reproduces exactly); 1y GOMEA-S10 40.6 (20 splits).

## What this shows

1. **UNI's gain carries over in direction but not at the registered strength.** It was lower
   on 14 of 20 splits on both datasets, with the same train–test gap, so the gain is not
   overfitting. But it is short of significance after Holm, unlike Con in 1v (10/10).
2. **The cost is the same on every dataset:** about 1.8–2× the program size and, on
   EnH/EnC, about 1.8× the CPU. (On Con in 1v the CPU was the same.)
3. **D1 trade-off on Con.** Neither size lever keeps UNI's accuracy:
   * UNI-PEN keeps the accuracy of plain GOMEA (37.2 against 37.1) at half its size (92
     against 187 nodes) and about 56 % of its CPU, with a smaller gap.
   * UNI-S10 gives very small programs (30 nodes, gap 2.7, 494 s) at a large accuracy cost.
     It is slightly worse than 1y's GOMEA-S10 (43.2 against 40.6), so UNI adds nothing on
     top of seeding.
4. **Practical reading:** UNI stays a candidate default for regression when accuracy matters
   more than size and CPU, supported on Con and leaning the same way on EnH/EnC. UNI-PEN is
   a D1 candidate (GOMEA accuracy at half the size); this is descriptive, not registered.

## Limits

* No per-generation telemetry: the 1ad regression runs predate the `--gomea-log` change, so
  generations per deme are not reported.
* Secondary contrasts never extend, so Con has 10 splits only.
* The UNI-PEN against GOMEA comparison is across phases (1v), on the same deterministic
  splits.
