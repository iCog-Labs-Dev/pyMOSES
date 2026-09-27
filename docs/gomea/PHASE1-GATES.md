# Phase 1 gate packets

The handoff contract (`00_START_HERE.md`) asks for a hard stop and a review packet at every
increment. The user's instruction #1 (`APPROVALS.md`) told the agent to run Phase 1
without stopping, so these packets stand in for the stops. They let the user review each
increment after the fact.

Every task was implemented by a subagent and then passed a spec-and-quality review;
several also went through fix rounds and scoped re-reviews. There was a final whole-branch
review and a fix wave. The ledger of findings, fixes and deferred minors lives in the
(git-ignored) SDD workspace. Its conclusions are summarised here.

## P1.0: audit

* **Delivered:** `P1.0-AUDIT.md` (the real call graph, the seam, the sampler, scoring,
  single-deme behaviour, repairs, the reuse shortlist and the risk register).
* **Repairs needed:** none. The stock HC *is* the matched HC.
* **Post-audit correction:** the score cache is off for the demo problems.

## P1.1: baseline and contracts

* **Branch:** `experiment/moses-gomea-discrete` from `0410881`.
* **Stock suite:** 7898 passed and 5 skipped (log and `environment.json` are in
  `../gomea-results/phase1/baseline/`). The HC stdout captures used for byte-identity
  checks were taken at `0410881`.
* **Contracts:** `FrozenSchema` (77f0448) and `EvaluationOracle` (6ae8f30).
  * Contracts 1 and 4 are in `test_schema.py`; contracts 2, 3 and 5 are in
    `test_oracle.py` and `test_optimizer.py`.
  * Contract 5 also has an FI-boundary budget sweep (c6301cb).
  * Contract 6: probing is part of representation building, which is shared and
    unchanged.
  * Contract 7: HC stays byte-identical (`test_integration.py`).
* **Reuse strategy:** a narrow Python port of GP-GOMEA 750220c, not a native bridge. The
  reasons are in the audit §8 and `PROVENANCE.md`.

## P1.2: discrete engine

* **Linkage statistics and FOS (55cff00):** 15 golden cases from a g++ harness built from
  verbatim upstream code. A reviewer's 3000-case differential test found 0 mismatches.
* **Sampler-based population (25de80b).**
* **GOM and engine (1cefbc1, 6f207ce):** the live best-ever FI elite fixed a spec defect,
  so FI never returns something worse than its input.
* **Engine diagnostics:** on the concatenated trap (4 × 4 bits, N = 64), LT-B and LT-MI
  solved 10/10 seeds against UNI's 0/10. Every planted block appeared in ≥ 80 % of learned
  LT-B FOS (100 % observed).

## P1.3: lifecycle integration

* **`-a gomea` plus nine `--gomea-*` options (3272841, 07ea018).** Classic's `--help` and
  option set are unchanged, and the hidden `--g` alias preserves the classic abbreviation.
* **Integration tests:** at least 2 expansions, each with its own schema and calibration;
  determinism; HC byte-identity ×3; program, score and complexity agreement after the
  merge; contin rejection; distinct termination reasons.
* **Full suite after integration:** 8124 passed, 5 skipped.

## P1.4: pilot and pre-registration

* **Cost probes** (`../gomea-results/phase1/probe/`):
  * GOMEA at N = 64 spends about 2000 evaluations per generation on 32–48 fields, and its
    first deme takes about 10k evaluations.
  * HC spends 300–1000 evaluations per deme and moves on.
  * Phenotype-skip proofs cost about 10 % of time for about 0 skips, so they are off by
    default.
* **Pilot A** (105 runs, 5 seeds, `docs/gomea/pilot-a/`) selected N = 16 with no per-deme
  cap.
* **Final whole-branch review:** ready with fixes and no Critical findings. Fix wave F1–F10
  (c6301cb) added CPU time, shuffled run order, diversity telemetry, the missing contract
  tests, the censored-success fix, memory-limit safety, portable result directories and the
  code-commit check.
* **Pre-registration:** `PREREGISTRATION.md`, committed with the manifests before any
  confirmatory run.

## P1.5: confirmation (done)

* **Where:** the server attempt failed, so at the user's request (`APPROVALS.md` #4) the
  runs were made locally on 2026-09-26 with 14 workers. They took 4 h 22 min.
* **Code:** `c6301cb`, verified by the runner's code check for every manifest. The tag
  `gomea-phase1-confirmation` (= `4df9a47`) holds that code plus the manifests, driver and
  report tool.
* **Runs:** 1,140 confirmatory runs, all ok.
* **Findings:** `PHASE1-FINDINGS.md`. LT-B GOMEA is *worse* than HC on maj5, mux6 and pa4;
  UNI is the best GOMEA variant; LT-B vs LT-MI and FI on/off are ties.
* **Evidence:** `phase1-results/` (checksums, manifests, verdicts, `findings.json`,
  figures).
* **Post-confirmation polish** (it does not affect the results): help text for the
  `--gomea-*` options, and a default population of 16.

## Phase 1b: stand-alone GP-GOMEA vs MOSES hill climbing (user-requested follow-up)

* **Approvals:** `APPROVALS.md` #5 (the request) and #6 (the Armadillo download).
* **Pre-registration:** `PHASE1B-PREREGISTRATION.md`, committed (`595c992`) before any
  confirmatory run.
* **Build:** the authors' GP-GOMEA `750220c` was built out of tree
  (`tools/gomea/build_gpgomea.sh`). Only one evaluation-count log field was added.
* **Pilot:** 90 runs selected N = 64, h = 5. The authors' defaults (N = 500) solved
  0/15 at 20k.
* **Confirmation:** 160 runs, plus 60 descriptive 200k runs, all ok.
* **Analysis:** `tools/gomea/phase1b_report.py`, whose unpaired statistics are unit-tested
  (`tests/gomea/test_phase1b_stats.py`).
* **Findings:** `PHASE1B-FINDINGS.md`. GP-GOMEA is *worse* than MOSES HC on maj5, mux6
  and pa4 at equal evaluations. It solves maj5 and mux6 reliably at about 30k
  evaluations, and it is far faster per evaluation, which is largely a C++ vs Python
  effect.

## Phase 1c: GP-GOMEA scalability (user-requested follow-up)

* **Approval:** `APPROVALS.md` #7.
* **Pre-registration:** `PHASE1C-PREREGISTRATION.md`, committed (`d36a4d8`) before any
  confirmatory run. The report's input-loading fix (`5199bba`) is also from before the
  results.
* **Pilot:** 150 runs chose the GP-GOMEA template height per size.
* **Confirmation:** 300 runs and 40 descriptive height-9 runs, all ok. Run on 2026-09-26
  and 27; one interruption, after which 15 partial runs were re-run from scratch.
* **Findings:** `PHASE1C-FINDINGS.md`. Under the pre-registered rule, GP-GOMEA scales
  better on majority and worse on multiplexer and parity. At the largest parity and majority sizes classic
  MOSES hit its 25-minute limit before its evaluation budget. Template height is
  GP-GOMEA's scaling limit.

## Phase 1d: why HC beat embedded GOMEA (user-requested follow-up)

* **Approval:** `APPROVALS.md` #8.
* **Code:** opt-in options added in `ee1957e`. The defaults are byte-identical,
  confirmed by the review and by a 6-run reuse check.
* **Pre-registration:** `PHASE1D-PREREGISTRATION.md` (`d97a425`), before any run.
* **Stage 1:** 990 frozen runs. Only N = 64 was carried forward.
* **Stage 2:** 240 end-to-end runs.
* **Findings:** `PHASE1D-FINDINGS.md`. The main cause is per-deme budget use: a 500-eval
  cap closes the maj5 gap and halves mux6 cost. Population size was contradicted
  end-to-end; diversity and neutral drift were not supported; the memo is undetermined.

## Deferred minors (not blocking; kept for the record)

* The golden.cpp header should note that upstream's commented-out blocks were omitted.
* `learn()` does not re-validate code ranges.
* The Fisher–Yates helper is duplicated in 3 modules.
* `last_result` reads string keys from the population and engine records.
* A contin rejection prints a traceback.
* `_ClassicPrefixAlias` forwards only `type`.
* The runner-added `-f` overrides a user's `-f`.
* The +1.0 worst-error sentinel is not scaled.
* `cpu_s` for timed-out runs includes interpreter start-up.
