# Server round: pre-registration (2026-10-25)

**Why:** the user asked for every experiment still needed to judge the GOMEA integration
to run unattended on a Debian Jupyter pod, with the results brought back for a decision.
This round has the unfinished phases of `PHASE1AC-1AG-PREREGISTRATION.md` (1ae, 1af, 1ag)
and three new confirmatory phases (1ah, 1ai, 1aj). The designs are the agent's, written at
the user's request; they were not reviewed one by one (see `APPROVALS.md`). How to run it:
`SERVER-RUN.md`.

## Rules for the whole round

* **Everything reruns on the server.** No result from the local machine is pooled with a
  server result. Contrasts are within one machine. Cross-machine comparisons are
  descriptive only.
* **Code:** one commit for the whole round (the merge of the long-deme and linear-scaling
  work, `514f602`, plus a packaging fix, `12718ec`, plus tools). `--knob-probe-baseline reduced` is the code default; every
  arm still passes it explicitly.
* **Statistics** as in every phase since 1u: paired Wilcoxon signed-rank, Holm across a
  phase's primary contrasts, α = 0.025. "Better" also needs a 10 % lower median. Holm
  p in [0.025, 0.25) at stage 1 extends that problem (seeds 3011–3020 or splits 11–20).
  Secondary contrasts never extend. Failed runs count as infinitely bad.
* **One phase at a time.** Phases never overlap (the user's standing rule). Each phase
  gets as many parallel runs as the pod can hold (see "Resources").
* **Order:** by decision value, so that the most important results exist first if the pod
  stops early: S0, 1ah, 1ai, 1af-pilot, 1af, 1aj, 1ag, 1ae.
* **The C++ dependency:** 1ai and 1aj need the authors' GP-GOMEA built on the pod. If the
  build fails, those phases are skipped and the skip is logged. Nothing else depends on it.

## Resources

* **Workers:** W = min(usable CPUs − 1, ⌊(memory − reserve) / 3 GB⌋), at least 1.
  * Usable CPUs: the cgroup CPU quota if there is one, else the physical cores in the
    process's affinity set (no hyper-thread oversubscription).
  * Memory: the cgroup limit, else MemTotal. The reserve is max(2 GB, 10 %).
  * 3 GB is each run's address-space limit (`RLIMIT_AS`, as in every phase), so W runs
    cannot exceed the memory. The local crash of 2026-10-22 was 16 workers × up to 3 GB on
    14 GB.
* **Why not more:** 1ah and 1aj give each run a wall-clock budget, and mux11 is scored on
  CPU time, so runs must not share cores.

## S0. Environment check (not an experiment)

1. The full test suite (`pytest`). A failure is logged and flagged, not fatal.
2. **Reproducibility check:** four runs whose local results are known, rerun with the
   round's code, compared exactly (test NMSE, train NMSE, evaluations, final program; for
   mux11, evaluations to solve):
   * 1ad EnH GOMEA split 1, 1ad EnC GOMEA-UNI split 1, 1ad Con GOMEA-UNI split 1;
   * 1ae mux11 GOMEA seed 3001.

   A mismatch means the pod's floating point differs from the local machine. It does not
   stop the round, because all contrasts are within the server, but it makes every
   cross-machine comparison invalid, and the report must say so.

## 1ah. GOMEA vs MOSES hill climbing at equal time, beyond Con (new)

* **Question:** does GOMEA's regression win at equal time (1p, Con only) hold on the other
  datasets?
* **Datasets:** Air, Bos, EnC, EnH, Yac (all of 1f's confirmatory datasets except Con),
  splits 1–10.
* **Arms:**
  * GOMEA: the base setting (LT-B, N = 16, cap 250, memo, `--gomea-contin copy`).
  * HC-C250: `-a hc-capped --gomea-max-evals-per-deme 250`.
  * HC: plain MOSES hill climbing (`-a hc`), MOSES's own default.
* **Budget:** 1,000 s wall time per run (as 1p).
* **Primary:** GOMEA vs HC-C250 on each dataset, test NMSE (5 contrasts).
* **Secondary:** GOMEA vs HC on each dataset.
* **Also reported:** size, train–test gap and evaluations used (D1).
* **Runs:** 150 (+ up to 100 in extensions).

## 1ai. Integrated GOMEA vs stand-alone GP-GOMEA at equal evaluations (new)

* **Question:** per evaluation, does GOMEA inside MOSES match or beat the authors' GP-GOMEA?
  This is the comparison the integration claim rests on. 1f compared the two only at very
  different budgets.
* **Datasets:** Air, Con, EnC, EnH, splits 1–10.
* **Arms:**
  * GPGOMEA-P16: the authors' C++ GP-GOMEA (`750220c`), 1f part B's setting (IMS g = 4,
    height 4, ERC, linear scaling) with population 16 (the integrated GOMEA's N) and
    `--evaluations 10000`. It trains on the same train rows as pyMoses.
  * GPGOMEA: the same with 1f's population of 500 (the authors' setting).
  * GOMEA: pyMoses GOMEA, base setting, 10,000 evaluations.
  * GOMEA-LS: GOMEA with `--linear-scaling 1`, the same fitness scaling as GP-GOMEA's.
* **Budget caveat (found while configuring):**
  * GP-GOMEA checks its budget only between generations, so it overshoots.
  * Measured locally on Con, EnH and Air, splits 1–2: population 16 used 11,050–12,983
    evaluations, and population 500 used 17,619–19,913. No setting gives exactly 10,000.
  * Evaluations used are reported per arm. The overshoot favours GP-GOMEA, so a tie or
    "A better" for GOMEA is conservative.
* **Primary:** GOMEA vs GPGOMEA-P16 on each dataset, test NMSE (4 contrasts).
* **Secondary:** GOMEA vs GPGOMEA, and GOMEA-LS vs GPGOMEA-P16, on each dataset.
* **Also reported:** size, gap and CPU.
  * CPU is not comparable across the languages and is descriptive only.
  * GP-GOMEA's size is its own node count, which excludes the scaling terms. pyMoses
    counts atoms plus parentheses.
* **Reading:** a tie or "A better" supports keeping the integration on accuracy; "A worse"
  means that per evaluation, plain GP-GOMEA is stronger.
* **Runs:** 160 (+ up to 80).

## 1af. Long demes for promising exemplars (as pre-registered)

As in `PHASE1AC-1AG-PREREGISTRATION.md`, including the 2-seed LONG-TOP pilot that sets
LONG-RAND's p before stage 1. The pilot's result is logged.

* **Correction:** the extension manifests now hold only the primary arms, GOMEA and
  LONG-TOP. Before, they also held LONG-RAND and LONG-TOP-RT, which would have given
  secondary contrasts 20 pairs, against the rule that secondary contrasts never extend.

## 1aj. Seeded GOMEA on mux11 at equal time (backlog 14, new)

* **Question:** does GP-GOMEA seeding (1u, GOMEA's best Boolean setting) close the Boolean
  time gap to MOSES hill climbing?
* **Arms:** GOMEA-SEED (1u's seeded GOMEA arm), HC-C250 and HC (uncapped).
* **Budget:** 3,600 s wall time per run (as 1p's mux11). Metric: CPU to solve, unsolved =
  infinite.
* **Seeding:**
  * The seeding step's GP-GOMEA CPU counts towards CPU to solve. Its wall time (about
    2.5 s in 1u) comes before MOSES's 3,600 s clock starts.
  * 1u sets the seeding budget as a 0.2 share of `-m`. Under a time budget `-m` is 10⁹,
    so the arm uses a fixed seeding budget of 8,000 evaluations instead: the value 1u
    actually ran with.
* **Primary:** GOMEA-SEED vs HC-C250. **Secondary:** GOMEA-SEED vs HC.
* **Seeds:** 3001–3010 (+ 3011–3020).
* **Runs:** 30 (+ 20).

## 1ag. Linear scaling (as pre-registered, one change)

* **The change:** the HC and GOMEA baselines are rerun on the server, not reused from 1n
  and 1v.

## 1ae. Per-deme caps (as pre-registered, rerun in full)

* **The change:** the locally interrupted run (stage 1 complete, Con extension 4/30)
  cannot be resumed on the server, so 1ae reruns from the start.
* **The local run:** its stage 1 stays on record as a separate result (mux11: cap 1,000
  better, 10/10). The server run's mux11 stage 1 is a replication on another machine.

## Logging (for the decision afterwards)

* `server-run/events.jsonl`: one line per event, appended as it happens. Events: round
  and phase starts and ends, every finished run with its full summary, every report and
  verdict, skips, errors and resource snapshots.
* Kept with it: the human-readable log, a scoreboard snapshot every 10 minutes, and a
  bundle of all results every hour and at the end.
