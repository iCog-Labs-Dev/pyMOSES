# GOMEA in pyMoses: experiment roadmap

**Written:** 2026-11-06. **Status:** a plan. Nothing in stages 1 to 4 is designed or approved.

This file turns the open ideas into a sequence of tasks, and it collects the working rules that the
earlier rounds taught us. The details of each idea are in `BACKLOG.md` (sections 7 and 8). The
results so far are in `FINDINGS-LOG.md`. The user's decisions are in `APPROVALS.md`.

Use it like this:

1. Take the next task whose prerequisites are done.
2. Follow the checklist in "How to run an experiment" for that task.
3. When the task ends, mark it here (date, verdict, findings file) and update the backlog.

## Where we are (2026-11-06)

| Setting | Status | Evidence |
|---|---|---|
| No per-deme evaluation cap | The base for all problems (user decision) | 1as, 1ae |
| Linear scaling (`--linear-scaling 1`) | Used in every regression arm | 1ag |
| `--knob-probe-baseline reduced` | Code default | 1q |
| LM with the validation guard (LM-VAL) | Preferred LM variant, not a default | 1at |
| PEN125 (`--gomea-accept-complexity-ratio 0.125 --gomea-deme-store accepted`) | Preferred penalty, not a default | 1au |
| Population N = 16 | Under test (round 6) | 1ay, 1az |
| Learned linkage (LT-B) | No gain over a random tree at our scale | 1g, 1m, 1al, round 2 check |
| Knob caps | Not recommended | 1ak, 1as |

## Stage 0: running now (rounds 6 to 8, one chain)

| Task | Content | Pre-registration |
|---|---|---|
| R6 | Population size and linkage (1ay); round 4 again with a dynamic population (1az) | `SERVER-ROUND6-PREREGISTRATION.md` |
| R7 | LM-VAL with no cap; LM-VAL + PEN125; control arm LMVAL-DYN (1aw) | `SERVER-ROUND7-PREREGISTRATION.md` |
| R8 | Confirmation on fresh splits 21–30, all 10 datasets; control arm FINAL-DYN (1ax) | `SERVER-ROUND8-PREREGISTRATION.md` |

**For each round when it ends:**

1. Bring its bundle to the MOSES folder and extract it into a new folder (`server-results-roundN`).
2. Analyse it (see "Analysis" below), and write `SERVER-ROUNDN-FINDINGS.md`.
3. Present the results phase by phase. The user decides on each setting.
4. Record the decisions word for word in `APPROVALS.md`, and update `FINDINGS-LOG.md`, `BACKLOG.md`,
   this file and the README.
5. Commit locally. Push only with a fresh approval.

**Stage 0 is done when** the user has decided on the population setting, LM-VAL with no cap, PEN125
with LM-VAL, and the confirmed final setting.

## Stage 1: settle the base setting

| Task | Goal | Prerequisites | Backlog |
|---|---|---|---|
| T1.1 | GP-GOMEA seeding with the current base: seeded against unseeded, seeding cost in the budget | Stage 0 | 7.5 |
| T1.2 | One GOMEA run with all good findings combined, with and without seeding, against HC and plain GOMEA | T1.1 | 7.4 |
| T1.3 | LT-B against RT with no cap | Only if 1ay leaves it open | 7.3 |

* **T1.1 questions:** does seeding still help when linear scaling, LM-VAL, no cap and PEN125 are in
  use? Does the Con loss of 1y and 1ab remain? Use the existing tools (`tools/gomea/gpgomea_seed.py`,
  the `seed` block of the runners). Report seed lineage.
* **T1.2:** the result of T1.1 decides if the final setting uses seeding. Use fresh data for any
  claim about the final setting.

**Stage 1 is done when** one base setting is chosen and recorded as the reference for stage 2.

## Stage 2: tier 1 ideas (small code, reuse what exists)

Run them in this order. Each task compares against the stage 1 base.

| Task | Idea | Code needed | Backlog |
|---|---|---|---|
| T2.1 | A1. Node-proximity linkage: a FOS from the tree distance between knob positions | A new FOS option | 8, A1 |
| T2.2 | B4a. Least squares weights for each top-level term in the scorer | Extend the linear-scaling code | 8, B4a |
| T2.3 | A2. LM during the search, on accepted improvements in the deme | Extend `const_opt` | 8, A2 |
| T2.4 | A4. Case-based exemplar selection: classic, `--diversity-pressure`, ε-lexicase | A small rule in `select_exemplar` | 8, A4 |

* **T2.1:** arms FOS = node proximity, RT, UNI, at the population setting of stage 1. Problems:
  mux11, Con, EnH. It needs no population data, so it can work at small N. The paper tested only
  large N and regression.
* **T2.2:** arms base against per-term least squares, with the validation guard. Report the number
  of terms, the size and the gap. It is also the candidate replacement for LM if LM does not
  scale, so report CPU per evaluation.
* **T2.3:** arms LM-VAL at deme end against LM-VAL on accepted improvements, at equal CPU.
* **T2.4:** start with the no-code arm (`--diversity-pressure`), so a part of the answer costs
  nothing. Boosting (`--boost`) needs `get_error` on the regression scorer first.

**Stage 2 is done when** each task has a registered verdict and a user decision.

## Stage 3: tier 2 ideas (new representation or deme code)

| Task | Idea | Prerequisites | Backlog |
|---|---|---|---|
| T3.1 | B4-split. LM on each term's inner constants, then least squares for the term weights | T2.2 | 8, B4-split |
| T3.2 | B1. GP-RV-GOMEA: real-valued GOMEA on the constants, together with the structure | Stage 2 | 8, B1 |
| T3.3 | B2. Semantic backpropagation with a subtree library: a new knob type | Stage 2 | 8, B2 |
| T3.4 | B4b. A full IT representation as a deme type (later TIR) | T2.2, T3.1 | 8, B4b |
| T3.5 | B6. Modular GP-GOMEA: hints for several demes in GP-GOMEA | **The user reads and validates the paper first. No experiment before that.** | 8, B6 |
| T3.6 | B7. Higher-arity GP-GOMEA | Stage 2 | 8, B7 |

* Each tier 2 task starts with a feasibility note: what code exists, what is new, what it breaks,
  and the estimated effort. The user decides on the note before any design.

## Control and low priority

| Task | Idea | When |
|---|---|---|
| C0 | `-a un` (univariate EDA) as a no-code control against GOMEA (agent suggestion, not user-approved) | Any time after stage 1. No code needed. |
| C1 | hBOA as a deme optimizer, as a control against GOMEA | Classic MOSES and pyMOSES have no BOA structure learning (a stub in C++). Needs a new BDe/BIC split policy on the ported `pymoses/moses/eda/` code. Feasibility note first. |
| L1 | A3. Intron-masked mutual information | Only if we run out of options, and only if T2.1 shows that a better linkage helps |

## Engineering tasks (no experiment)

| Task | Content | Backlog |
|---|---|---|
| E1 | Print `a` and `b` with the final program when `--linear-scaling 1` is on | 6.3 |
| E2 | A CPU profile of a regression run (only mux11 has one) | 6.5 |
| E3 | Speed-ups that do not change results: bit-parallel truth tables, a cache of reduced candidates, cheaper linkage statistics | 1.4–1.6 |

## How to run an experiment (checklist)

### 1. Before the design

* **Check what we already know.** Read `FINDINGS-LOG.md`, the related phase findings and
  `BACKLOG.md`. Round 4 repeated 1ae because we did not check. Say in the design which earlier
  phases touch the question.
* **Write the question and the expected result.** One question per phase. Say what each outcome
  would change.
* **Read the primary source of an idea from the literature.** Separate measured results from
  claims. Mark numbers that are not verified in the paper text.

### 2. The design (pre-registration)

* **Exact arms:** the full argv of every arm. One difference between arms per contrast.
* **Data:** datasets and splits (regression), problems and seeds (Boolean). Default: 10 splits or
  seeds, 10 more only by the extension rule.
* **Budget:** say what is equal across arms, CPU time or evaluations, and why. Report both
  evaluations and CPU in the results. Round 4: the cap saved evaluations but cost CPU.
* **Fair comparisons:** HC gets the same per-deme cap as GOMEA. Seeding counts against the same
  budget. A selection step and its confirmation use different data (fresh splits 21–30 are used
  by round 8; no later splits exist).
* **Statistics:** paired Wilcoxon signed-rank, Holm across the primary contrasts of a phase,
  α = 0.025. "Better" also needs a 10 % lower median. Holm p in [0.025, 0.25) at stage 1 extends.
  Secondary contrasts never extend. Failed runs count as infinitely bad.
* **Power check:** with n pairs, the lowest two-sided p is 2/2^n (n = 10: 0.00195). Holm needs that
  value × the number of primary contrasts to be below 0.025. With n = 10, use at most 12 primary
  contrasts, or plan for the extension, or pool (as in round 8).
* **Adoption rule:** write it before the run. Say what happens if no arm qualifies.
* **Reported (D1):** test NMSE or CPU to solve, program size, train–test gap, CPU, and the method's
  own counts (LM calls, guard checks, deme counts, generations per deme, population size).
* **Automatic choices** (a later phase uses an earlier result): write the rule, the tie rule and
  the fallback in the pre-registration.
* **Approval:** the user approves the design. Quote the user word for word in `APPROVALS.md`, and
  mark the parts that the agent wrote and the user did not approve in detail.

### 3. Implementation

* **Subagents build** in a separate clone (never `git worktree`), with the pre-registration as the
  specification. They stop and report if the design is unclear. The agent that talks to the user
  keeps the design, the verification and the findings.
* **New code is opt-in.** The default stays byte-identical. Prove it with an identity test.
* **Tests fail before and pass after.** Never weaken a test or change an expected value.
* **Choice scripts never write tracked files.** They write into the state directory. Round 3's
  choice script dirtied the server tree, and a test then failed in every later S0.
* **Boolean manifests name the last code commit** (`check_code_commit`). A code change moves it.
* **Smoke test** the whole chain in a clean rig, with the largest expected memory case.
* **Dry run** each real phase: run counts and the argv per arm.

### 4. Verification (before any push)

* Fast-forward the main repo from the clone (`git fetch <clone> <branch> && git merge --ff-only
  FETCH_HEAD`). Never rebase or force-push.
* Check authors (the user only) and that there are no attribution lines.
* Read the diff under `pymoses`, `benchmarks` and `pyproject.toml`.
* Run the new tests again. Check every arm, budget and contrast against the pre-registration with
  a script.
* Push only with a fresh approval from the user.

### 5. On the server

* One phase at a time. Rounds never overlap. A chain starts the next round after `DONE`.
* Do not `git pull` or change files on the server while a round runs.
* Stop only with `stop.sh`. Kill processes only by PID or PGID.
* Check S0 first: pytest and the reproducibility check. A failed S0 only sets a flag, so read it.
* The user does the server steps, one step at a time.

### 6. Analysis

* Extract each bundle into its own new folder. Read untrusted data with `python -I`.
* Check the run counts and failures. Recompute the key medians from the raw `summary.json` files.
* Separate the registered verdict from descriptive numbers. A tie with all medians on one side is
  still a tie; say the direction.
* Write the findings file: answer in short, S0, each phase with tables, interpretation,
  deviations, limits. Correct your own earlier errors in the file.
* Present the results phase by phase. The user decides.

### 7. Literature research

* Plan first: the families, queries, sources and inclusion criteria.
* Criteria for us: domain-agnostic, cheap, measured gains on real data against a fair baseline,
  well cited for its age, and feasible in MOSES.
* Give the citation source and date. Mark numbers that are not verified in the paper text.
* Put ideas whose only blocker is the integration into MOSES in a separate list.
* The user reads key papers before a design when the user asks for that (B6).

### 8. Records and communication

* Write everything in ASD-STE100 Simplified Technical English.
* Commits are by the user's identity only, with no attribution lines.
* Never commit `MOSES_GOMEA_LOCAL_AGENT_HANDOFF/`, `reports/`, `research_notes/`, or the Dow and
  Tow datasets.
* After each decision, update `APPROVALS.md`, `FINDINGS-LOG.md`, `BACKLOG.md`, this file and, for
  a status change, the README.
