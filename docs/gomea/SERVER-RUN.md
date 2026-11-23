# Running the server round

How to run the round pre-registered in `SERVER-ROUND-PREREGISTRATION.md` on a Jupyter pod,
unattended, and what to bring back. Everything is in `tools/server/`.

## What you need

* A Debian-based pod (any Linux with bash, git, curl, tar and coreutils works). No sudo.
* Internet access during set-up: GitHub (uv, micromamba, GP-GOMEA), the Python download
  that uv uses, conda-forge (only if the system has no complete C++ toolchain) and
  gitlab.com (Armadillo, for GP-GOMEA).
* At least 20 GB free disk next to the clone, and a home directory that survives restarts.
* Time: the round is about 300 CPU-hours. With W parallel runs it takes about 300 / W
  hours: 8 cores (W = 7) about 43 h, 16 cores (W = 15) about 20 h, 32 cores about 10 h.
  W also needs 3 GB of memory per run plus a reserve (see "Workers" below), so 16 cores
  with 32 GB give W = 9, about 33 h.

## Steps

1. Open a **Jupyter Terminal** (File > New > Terminal) and clone:

       git clone --branch experiment/moses-gomea-discrete https://github.com/iCog-Labs-Dev/pyMOSES.git pyMoses
       cd pyMoses

   Results go to `../gomea-results` (next to the clone). To put them elsewhere, set
   `export GOMEA_RESULTS=/big/disk/gomea-results` before step 2; set-up then makes
   `../gomea-results` a link to it.

2. Set up (10 to 40 minutes; safe to rerun):

       bash tools/server/setup.sh

3. Read the summary it ends with. `SETUP OK` means you can start. Per step:
   * `disk WARN`: less than 20 GB free. Free space first if you can.
   * `gpgomea FAILED`: the C++ GP-GOMEA could not be built. The round still runs, but
     phases 1ai and 1aj are skipped (logged). The reason is in
     `../gomea-results/server-run/gpgomea-build.log`.
   * any other `FAILED`: fix it (usually the network) and rerun step 2.

4. Start the round:

       bash tools/server/start.sh

   It returns at once and prints how to follow it. The round runs in the background and
   survives closing the terminal or the browser.

## Following it

* `tail -f ../gomea-results/server-run/run.log`: what is happening, with timestamps.
* `bash tools/server/status.sh`: running or not, each step's status, the current phase
  and its W, the last events, memory and load, and the latest scoreboard.
* `cat ../gomea-results/server-run/scores-latest.txt`: per phase, problem and arm, the
  finished runs and running medians (refreshed every 10 minutes).

The order is S0 (tests and a reproducibility check), 1ah, 1ai, 1af-pilot and 1af, 1aj,
1ag, 1ae. One phase at a time.

## Jupyter notes

* A notebook cell `!bash tools/server/start.sh` (run from the clone) also works, since
  the round detaches. Do not run the phases themselves in a cell.
* Idle culling, a pod restart or a crash kills the round. Nothing is lost: run
  `bash tools/server/start.sh` again. It skips finished steps and resumes the current
  phase (finished runs are kept; only the runs that were in progress start over).
* Optional: `bash tools/server/install_autostart.sh` makes the Jupyter server rerun
  `start.sh` each time it starts (it adds a marked block to
  `~/.jupyter/jupyter_server_config.py`; nothing happens once the round is DONE).
  `bash tools/server/uninstall_autostart.sh` removes it.

## Stopping

    bash tools/server/stop.sh

stops the whole round (the orchestrator and every run it started), waits up to 60 s,
then forces it, and writes a fresh `bundle-latest.tar.gz`. `start.sh` resumes later.
`RETRY_FAILED=1 bash tools/server/start.sh` also reruns steps that failed or were skipped.

## What to bring back

When the round ends, `../gomea-results/server-run/DONE` exists and `run.log` says
`DONE. Bring back ...`. Bring back

    ../gomea-results/server-run/bundle-final-<UTC time>.tar.gz

If it stopped early, bring `../gomea-results/server-run/bundle-latest.tar.gz` (rewritten
every hour and by `stop.sh`). The bundle holds the logs, `events.jsonl`, the scoreboard
history, every run summary and manifest, the reports, the per-run GOMEA event logs and the
environment; `round-summary.md` inside it is the overview. Download it from the Jupyter
file browser (right click > Download) or with `scp`.

## Workers

`python3 tools/server/resources.py` prints the decision. W = min(usable CPUs - 1,
(memory - reserve) / 3 GB), at least 1. Usable CPUs are the container's CPU quota if it has
one, else the physical cores (no hyper-threads). Memory is the container limit, else the
machine's. The reserve is max(2 GB, 10 %). `MAX_WORKERS=N bash tools/server/start.sh`
caps W (the cap is decided again at each phase start).

## Troubleshooting

* **1ai and 1aj skipped**: the GP-GOMEA build failed. Rerun set-up with
  `REBUILD_GPGOMEA=1 bash tools/server/setup.sh` (or `FORCE_CONDA=1` to skip the system
  compiler), then `RETRY_FAILED=1 bash tools/server/start.sh`, which reruns
  failed and skipped steps (done steps are never rerun). If the round is already DONE,
  add `--force`.
* **Reproducibility mismatch** (S0, `server-run/repro.json`, flagged in `run.log` and the
  summary): the pod's floating point (CPU, maths library) gives different results from
  the local machine for the same seeds. The round goes on and all its contrasts are
  still valid, because they compare runs on the same machine; only comparisons between
  local and server numbers become invalid. `server-run/gpgomea-build.json`
  (`smoke_matches_local`) is the same check for the C++ build.
* **pytest failures** (S0, `server-run/pytest.log`): flagged, not fatal. Report them.
* **"already running"**: an orchestrator is alive; use `status.sh`, or `stop.sh` first.
* **Out of disk**: stop, free space, start again.
* **A phase failed**: `phase_run.sh` was retried once; its log is
  `../gomea-results/<phase>.log`. The round continued with the next step.

## Smoke test

`bash tools/server/start.sh --plan smoke` runs a short version (pytest on
`tests/server` only, the mux11 reproducibility run, one tiny phase in
`configs/gomea/server-smoke/`, and a step that is skipped on purpose) with its own state in
`../gomea-results/server-run-smoke/`, so it never marks real steps done. The mux11 run
takes about 15 to 20 minutes, the rest about two. `status.sh --plan smoke` and
`stop.sh --plan smoke` work the same way. Before its phase it sets the `commit` of
`configs/gomea/server-smoke/bool.json` to the current HEAD (the Boolean runner's code
check), which leaves that file modified in the working tree; `git checkout` it afterwards.

## Round 2

The second round (`SERVER-ROUND2-PREREGISTRATION.md`: the knob cap, the linkage ablation
at that cap, and the generation-0 ranking) has its own plan, `tools/server/plan2.json`:
S0 (full pytest and the round-1 reproducibility check, now with the knob-cap code, which
must reproduce round 1 exactly), then 1ak-pilot, 1ak, 1al and 1am, one phase at a time.

1. Update the clone and set up as in steps 1 to 3 (`git pull`, then
   `bash tools/server/setup.sh`; safe to rerun on a pod that ran round 1).
2. Start it:

       bash tools/server/start.sh --plan round2

   (`PLAN=round2 bash tools/server/start.sh` or `--plan tools/server/plan2.json` work
   too). `status.sh --plan round2` and `stop.sh --plan round2` work the same way. A
   round-1 autostart block does not start round 2: run `uninstall_autostart.sh`, then
   `install_autostart.sh --plan round2`.
3. After 1ak-pilot, `tools/gomea/set_knob_cap.py` picks K per problem (the largest of 8, 16
   and 32 whose median generations per deme is at least 3, else 8) and writes it into
   every knob-cap arm of `configs/gomea/phase1ak`, `phase1al` and `phase1am` (this leaves
   those files modified in the working tree; that is expected). 1al and 1am do not start
   without that choice.

Outputs, all under `../gomea-results/`:

* `server-run2/`: the round's state and logs (`run.log`, `events.jsonl`, `state.json`,
  the scoreboards, `round-summary.md`, `bundle-latest.tar.gz` and at the end
  `bundle-final-<UTC time>.tar.gz`, which is what to bring back). Round 1's
  `server-run/` is not touched.
* `phase1ak-pilot/knob-cap.json`: the pilot's numbers and the chosen K.
* `phase1ak-pilot/`, `phase1ak/`, `phase1al/`, `phase1am/`: the regression runs and
  each phase's `report-stage{1,2}/`; the Boolean runs are in `1/phase1a?-*-bool-*/`.
* `phase1ak-pilot.log`, `phase1ak.log`, ...: each phase's log.

Smoke test: `bash tools/server/start.sh --plan smoke2` runs a miniature of the same plan
(pytest on `tests/server` and the round-2 tool tests, the mux11 reproducibility run, then
the pilot, `set_knob_cap.py` and the three phases in `configs/gomea/server-smoke2-*` on
mux6, Con and EnH with tiny budgets) with its state in `../gomea-results/server-run2-smoke/`.
It sets the `commit` of the smoke Boolean manifests to HEAD and writes K into the smoke
configs, so `git checkout configs/gomea/server-smoke2-*` afterwards.

## Round 3

The third round (`SERVER-ROUND3-PREREGISTRATION.md`: constant tuning at equal evaluations
and at equal CPU time, size control with linear scaling, the uniform start, and five more
datasets) has its own plan, `tools/server/plan3.json`: S0 (full pytest and the round-1
reproducibility check, which must still match: with `--const-opt none` the new code gives
byte-identical output), then 1an, 1ao, 1ap, 1aq and 1ar, one phase at a time.

1. Update the clone and set up as in steps 1 to 3 (`git pull`, then
   `bash tools/server/setup.sh`). Set-up now also builds WiR and WiW (committed) and
   downloads Dow and Tow from github.com/grantdick/gpzgd at a pinned commit (their data has
   no licence and is not committed; see `data/gomea-sr/README.md`). It checks every file
   against its SHA-256. If a download or a check fails, set-up prints `SETUP FAILED` and
   names the dataset; phase 1ar then does not start (its step is skipped with that reason).
2. Start it:

       bash tools/server/start.sh --plan round3

   `status.sh --plan round3` and `stop.sh --plan round3` work the same way. An autostart
   block of an earlier round does not start round 3: run `uninstall_autostart.sh`, then
   `install_autostart.sh --plan round3`.
3. 1an runs as the pilot of the 1ao step. After it, `tools/gomea/set_const_opt.py` picks
   the tuning arm (LM or MUT, the lower median test NMSE pooled over the three datasets'
   medians, from 1an's final results) and writes `--const-opt lm` or `mutate` into the
   GOMEA-LS-TUNE arms of `configs/gomea/phase1ao` and `phase1ar` (this leaves those files
   modified in the working tree; that is expected). 1ao and 1ar do not start without that
   choice.

Outputs, all under `../gomea-results/`:

* `server-run3/`: the round's state and logs, as in round 2; bring back
  `bundle-final-<UTC time>.tar.gz`.
* `phase1an/const-opt-choice.json`: the per-dataset and pooled medians and the choice.
* `phase1an/` to `phase1ar/`: the runs (each `summary.json` has `const_opt` for a tuning
  arm: LM calls and statuses, tuning evaluations and their share of the run, mutation
  attempts and acceptances, polish accepted or not) and each phase's `report-stage{1,2}/`.
* `phase1an.log`, ...: each phase's log.

Cost note (measured locally on three Con programs of 212-265 nodes, 45-69 constants, 516
training rows): one LM call used 140-712 evaluations and 10-71 s of CPU. So in 1an the LM
arm spends most of its 10,000 evaluations on LM; this is the pre-registered cost
accounting.

Smoke test: `bash tools/server/start.sh --plan smoke3` runs a miniature of the same plan
(pytest on `tests/server` and the round-3 tests, the mux11 reproducibility run, then 1an
and the choice, 1ao, 1ap, 1aq and 1ar in `configs/gomea/server-smoke3-*`, 2 splits, 400
evaluations or 20 s per run) with its state in `../gomea-results/server-run3-smoke/`. It
writes the choice into the smoke configs, so `git checkout configs/gomea/server-smoke3-*`
afterwards.

## Round 4

The fourth round (`SERVER-ROUND4-PREREGISTRATION.md`: knob caps with and without the 250-evaluation
deme cap) has its own plan, `tools/server/plan4.json`: S0 (full pytest and the round-1
reproducibility check), then phase 1as. There is no pilot and no value to set during the run.
The search code does not change.

1. Update the clone and set up as in steps 1 to 3 (`git pull`, then
   `bash tools/server/setup.sh`).
2. Start the round:

       bash tools/server/start.sh --plan round4

   `status.sh --plan round4` and `stop.sh --plan round4` work the same way. An autostart block
   of an earlier round does not start round 4. Run `uninstall_autostart.sh`, then
   `install_autostart.sh --plan round4`.
3. Stage 1 cannot give a primary verdict: with 10 pairs and 20 contrasts, the lowest Holm p is
   0.039. So a large effect gets "extend", and `phase_run.sh` runs the extension of that problem.
   This is expected.

Arms and files:

* `configs/gomea/phase1as/`: `spec.json` (20 primary and 8 secondary contrasts), `bool.json` (mux11,
  seeds 3001-3010), `bool-ext-mux11.json` (seeds 3011-3020), `sr.json` (Con, EnH, Air, splits 1-10)
  and `sr-ext-{Con,EnH,Air}.json` (splits 11-20).
* All manifests hold all seven arms. The extensions also need K16-C250, because the secondary
  contrasts use it. A problem that extends costs 70 more runs.
* Time budgets: regression 1,000 s, wall cap 2,400 s. mux11 3,600 s, wall cap 5,400 s.

Outputs, all under `../gomea-results/`:

* `server-run4/`: the state and logs of the round, as in round 2. Bring back
  `bundle-final-<UTC time>.tar.gz`.
* `phase1as/phase1as-sr*/` and `1/phase1as-bool*/`: the runs, and `phase1as/report-stage{1,2}/`:
  the reports.
* `phase1as.log`, `phase1as-bench.log`, `phase1as-sr.log`: the logs of the phase.

Regression summaries have no `deme_count`. The report now has these values (see "Round 5"). The method by hand is: Take the number
of demes, the generations per deme and how demes end from the `end` events in each run's
`events.jsonl`. The bundle holds these files. A deme that starts when the time is spent has 0
evaluations and ends with `time_limit`. Do not count it as a real deme.

Time check (local, one run each, `--max-time 60`, nice 19, on a loaded machine; not a result):

| Run | Wall s | Evaluations | Demes (with evaluations) | Genes per deme | Generations per deme | Deme ends |
|---|---|---|---|---|---|---|
| Con NOCAP | 61.9 | 1,616 | 2 | 20, 162 | 7, 2 | converged, time limit |
| Con K64 | 60.9 | 1,418 | 2 | 20, 128 | 9, 1 | stagnation, time limit |
| Con K8 | 60.8 | 1,491 | 3 | 20, 135, 162 | 13, 13, 3 | stagnation, stagnation, time limit |
| mux11 NOCAP | 94.1 | 3,011 | 3 | 88, 425, 952 | 11, 6, 1 | stagnation, converged, time limit |
| mux11 K64 | 79.9 | 3,785 | 3 | 88, 317, 637 | 10, 11, 2 | stagnation, stagnation, time limit |

The time limit stops a deme before the next evaluation, so a deme does not overshoot. A mux11 run
overshoots because the program spends time outside the deme. In these two mux11 runs, 20 to 34 s
were over the limit. The wall caps allow for more.

Smoke test: `bash tools/server/start.sh --plan smoke4` runs a miniature of the same plan (pytest on
`tests/server` and the knob-cap tests, the mux11 reproducibility run, then 1as in
`configs/gomea/server-smoke4-1as`: mux6, Con, EnH and Air, 2 seeds or splits, 15 s per run) with its
state in `../gomea-results/server-run4-smoke/`. With 2 pairs no problem extends, so the smoke plan does
not run the `*-ext-*` manifests. The smoke Boolean manifests name the same `commit` as the real ones
(see "Round 5"), and the smoke plan does not change them.

## Round 5 and the chain from round 4

The fifth round (`SERVER-ROUND5-PREREGISTRATION.md`: a validation guard for LM, a weaker size
penalty, LM with a size penalty) has its own plan, `tools/server/plan5.json`: S0, then phases 1at,
1au and 1av. Its state is in `../gomea-results/server-run5/`. There is no pilot and no value to set
during the run.

`plan4.json` has `"then": "round5"`, so one command runs both rounds, one after the other:

    bash tools/server/start.sh --plan round4

How the chain works:

* When round 4 ends, it writes its own `round-summary.md`, `bundle-final-<UTC time>.tar.gz` and
  `DONE`. Then it logs the `chain_start` event and starts `start.sh --plan round5` in a new session.
* Each round reports separately: `server-run4/` and `server-run5/` each have their own summary and
  final bundle. A bundle holds only the files of its round. Bring back both.
* Round 4 goes on to round 5 when it ends with a failed or skipped step. `run.log` of round 4 reports
  those steps, and so does its `round-summary.md`.
* `stop.sh` with either plan name stops the running orchestrator of the chain. A stopped round does
  not start the next round. It writes `server-run4/chain.stop`, and the next `start.sh` removes it.
* Never two at once: each orchestrator of the chain holds `server-run4/chain.lock`, whichever round
  it runs. `start.sh` refuses, and a direct `orchestrate.py` exits with 3.
* `RETRY_FAILED` and `REPRO_ONLY` apply only to the plan that the command names. They do not pass to
  the next plan.

Commands, with the state of the chain:

| Command | Round 4 not DONE | Round 4 DONE, round 5 not DONE | Both DONE |
|---|---|---|---|
| `start.sh --plan round4` | starts or resumes round 4 | starts or resumes round 5 | says where both bundles are |
| `start.sh --plan round4 --force` | as above, without the next round | reruns the end of round 4 only | reruns the end of round 4 only |
| `status.sh --plan round4` | all, and the `chain:` section | all, and the `chain:` section | all, and the `chain:` section |
| `stop.sh --plan round4` | stops round 4 | stops round 5 | nothing to stop |

The `chain:` section of `status.sh` gives the state of each plan and where its bundle is.

`start.sh --plan round5` also works when round 4 is not DONE and not running. It prints no error,
and it does not run round 4. Use it only to skip round 4.

**Autostart.** Run `uninstall_autostart.sh`, then `install_autostart.sh --plan round4`. The block
returns only when both rounds are DONE. In all other cases it runs `start.sh --plan round4`, so a pod
restart during round 4 resumes round 4, and a pod restart during round 5 resumes round 5. As in earlier
rounds, the block also resumes a round that `stop.sh` stopped. To prevent this, run
`uninstall_autostart.sh` after `stop.sh`.

**Time.** In the step records and in `round-summary.md`, `duration_s` is now the active time: the sum
over the orchestrator sessions, without the idle time between them. `elapsed_s` is the wall clock
since the step first started. A kill loses at most one poll interval (60 s) of active time.

**Report.** `aggregate.csv` now has, for each arm, `median_train_test_gap`,
`median_generations_per_deme` and `deme_end_share_<reason>`. For regression, `median_deme_count` now
has a value. All of them count only demes with evaluations (`ledger.requests > 0`), from each
run's `events.jsonl`. The note in "Round 4" about taking these values by hand no longer applies. The
arm LM-VAL also has `median_lm_guard_checked`, `median_lm_guard_rejected` and
`median_lm_guard_rejected_share` (rejected / checked).

Arms and files:

* `configs/gomea/phase1at/`: GOMEA-LS, LM, LM-VAL on WiR, Con, EnH, Air, Yac, Tow, 1,000 s, wall cap
  2,400 s.
* `configs/gomea/phase1au/`: GOMEA-LS, PEN25, PEN125, PEN625 on Con, EnH, Air, 10,000 evaluations, wall
  cap 36,000 s (as 1ap). Primary metric: `program_size`.
* `configs/gomea/phase1av/`: GOMEA-LS, LM, LM-PEN5, LM-PEN25 on Con, EnH, Air, 1,000 s, wall cap
  2,400 s. Primary metric: `program_size`.
* Each has `spec.json`, `sr.json` (splits 1-10) and `sr-ext-<dataset>.json` (splits 11-20, all arms).
* The phases have `datasets:` requirements. `setup.sh` builds Yac and WiR from `data/gomea-sr/`, and it
  downloads Tow. If a dataset is missing, the step is skipped, and the log gives the reason. So run
  `setup.sh` after `git pull`.

Smoke test: `bash tools/server/start.sh --plan smoke4` runs the miniature of round 4, then
`smoke5` by itself (`configs/gomea/server-smoke5-*`: 2 splits, 15 s or 400 evaluations per run). The
state is in `server-run4-smoke/` and `server-run5-smoke/`.

**Boolean manifests of round 4.** The guard changed `pymoses`, so the `commit` of the Boolean manifests
of 1as and of its smoke test moved to the commit with the guard. If `pymoses`, `benchmarks` or
`pyproject.toml` change again, `check_code_commit` refuses these runs until the `commit` moves too. The name of the result folder holds a hash of the manifest,
so runs under an older `commit` are not reused: `1/phase1as-bool-<new hash>/` starts empty.

The dynamic population of round 6 changed `pymoses` again, so the `commit` of these manifests moved
to `0e8dfa2`, the code commit of round 6. The 1ay and 1az manifests name the same commit.


## Round 6 and the chain 6, 7, 8

The sixth round (`SERVER-ROUND6-PREREGISTRATION.md`: is the fixed GOMEA population of 16 a
bottleneck?) has its own plan, `tools/server/plan6.json`: S0, then phase 1ay, then the choice of
the population, then phase 1az. Its state is in `../gomea-results/server-run6/`. Round 6 runs
first. `plan6.json` has `"then": "round7"` and `plan7.json` has `"then": "round8"`, so one command
runs the three rounds, one after the other, without a pause:

    bash tools/server/start.sh --plan round6

The chain works as in "Round 5 and the chain from round 4": three summaries, three final bundles,
one chain lock (`server-run6/chain.lock`). Bring back the three bundles. Rounds 7 and 8 read the
choice of round 6 from `server-run6/`, so do not start round 7 alone before round 6 is DONE.

**New code: `--gomea-pop-size dynamic`.** N for each deme is the number of new instances that
classic hill climbing takes at distance 1: `hill_climbing.estimate_neighborhood(1, fields)`, then
`n_new_instances(1, ...)` with `--hc-fraction-of-nn F` (default 2.0), `--hc-max-nn-evals`
(default 20,000), the classic minimum of 100 and the evaluations left of the deme. The population
builder draws N programs as before. With `--gomea-pop-size 16` (or no option) the output is
byte-identical to the code before (`tests/gomea/test_pop_size_identity.py`). The telemetry has
`deme_pop_size` and `deme_budget` on `invocation_start`, and `pop_size_dynamic` on `end`, only with
`dynamic`. `aggregate.csv` has `median_pop_size`, `min_pop_size` and `max_pop_size` for each arm
(from `pop_size_effective` of each deme with evaluations).

**The choice of the population.** The step `choose-population` runs
`tools/gomea/set_population.py choose`:

1. It reads the final 1ay report, with the stage rule of `set_final.py`.
2. POP (rounds 7 and 8): a fraction qualifies if its LT-B arm is "A better" than N16-LTB on at
   least 2 of the 3 problems and "A worse" on none. Of several, the best median rank. If none
   qualifies, POP is N = 16.
3. DYN (1az and the control arms): the fraction with the best median rank, qualified or not.
4. Median rank: on each problem, the 3 dynamic LT-B arms ranked by `median_metric` of the
   report's `aggregate.csv` (equal medians: the smaller F first). Then the median of the 3 ranks.
   Equal median ranks go to the smaller F.
5. The control population is DYN, or N = 16 if POP equals DYN.
6. It writes `server-run6/phase1az-config/`: the resolved 1az manifests, `spec.json` and
   `population-choice.json` (the rules, the numbers, POP, DYN and the control). The orchestrator
   logs it as the `choice` event.

**Templates.** A phase whose argv depends on the choice has `template.json` (resolved to `sr.json`)
and `template-<name>.json` (resolved to `<name>.json`). In an argv, the token `{POP}`, `{DYN}` or
`{CONTROL}` becomes `--gomea-pop-size 16` or `--gomea-pop-size dynamic --hc-fraction-of-nn F`.
The choices write only into the state directories, never into tracked files.

Arms and files:

* `configs/gomea/phase1ay/`: N16, DYN2 (F = 2.0), DYN025 (F = 0.25) and DYN005 (F = 0.05), each with
  `--gomea-fos lt-b` (`-LTB`) and `rt` (`-RT`). mux11 (`bool.json`, seeds 3001-3010, 3,600 s, wall cap
  5,400 s), Con and EnH (`sr.json`, splits 1-10, 1,000 s, wall cap 2,400 s). Extensions:
  `bool-ext-mux11.json` (seeds 3011-3020) and `sr-ext-{Con,EnH}.json` (splits 11-20). No arm has an
  evaluation cap. 240 runs (+ up to 240).
* `spec.json` of 1ay: 9 primary contrasts (each dynamic LT-B arm against N16-LTB, on each problem)
  and 12 secondary (LT-B against RT at each population, on each problem). With 10 pairs, Holm over
  9 can give a verdict at stage 1.
* `configs/gomea/phase1az/`: templates only. NOCAP-DYN, K8-DYN, K16-DYN, K32-DYN, K64-DYN (knob flags
  as in 1as) and NOCAP-N16, all LT-B. mux11, Con, EnH, Air, as in 1as. 16 primary and 4 secondary
  contrasts. With 16 contrasts, stage 1 gives no primary verdict, so a large effect extends (as in
  round 4). 240 runs (+ up to 240).
* The Boolean manifests of 1ay and 1az name the code commit of the dynamic population. If
  `pymoses`, `benchmarks` or `pyproject.toml` change again, move that `commit` (and the one of the
  round 4 manifests) too.

**Memory.** At F = 2.0, N on mux11 can be several thousand. The smoke test runs mux11 at F = 2.0
for 60 s. The real runs have the usual limit of 3,072 MB per run.

Smoke test: `bash tools/server/start.sh --plan smoke6` runs the miniature of round 6
(`configs/gomea/server-smoke6-1ay` and `server-smoke6-1az`: 2 splits or seeds, 15 s per regression
run, 60 s per mux11 run in 1ay and 20 s in 1az), then `smoke7` and `smoke8`. The state is in
`server-run6-smoke/`, `server-run7-smoke/` and `server-run8-smoke/`. With 2 pairs nothing is
"A better", so the smoke choice gives POP = N16.

## Round 7

The seventh round (`SERVER-ROUND7-PREREGISTRATION.md`: LM-VAL with no per-deme cap, and LM-VAL with
PEN125) has its own plan, `tools/server/plan7.json`: S0, then the step `apply-population`, then
phase 1aw. Its state is in `../gomea-results/server-run7/`. Round 6 starts it (see "Round 6 and the
chain 6, 7, 8"). When round 7 ends, it starts round 8 (see "Round 8 and the chain from round 7").

**The population (round 6).** Every GOMEA arm of 1aw uses POP. The step `apply-population` runs
`tools/gomea/set_population.py apply` with `server-run6/phase1az-config/population-choice.json` and
writes the resolved manifests into `server-run7/phase1aw-config/` (with
`population-applied.json`). The phase step reads them from there (`config_dir`). If round 6 has no
choice, the step fails and 1aw is skipped.

Arms and files:

* `configs/gomea/phase1aw/`: templates only (`template.json`, `template-sr-ext-<dataset>.json`).
  NOCAP, LMVAL, LMVAL-PEN125, LMVAL-C250 and the control arm LMVAL-DYN (LMVAL with the control
  population) on WiR, Con, EnH, Air, Yac, Tow, 1,000 s, wall cap 2,400 s. Only LMVAL-C250 has
  `--gomea-max-evals-per-deme 250`. With POP = N16 its argv is the LM-VAL argv of 1at. 300 runs
  (+ up to 300).
* `spec.json` has 12 primary contrasts with two metrics: `test_nmse` for LMVAL vs NOCAP and
  `program_size` for LMVAL-PEN125 vs LMVAL. Each contrast names its metric (the fifth element).
  Holm covers all 12. 18 secondary contrasts: the 12 of the pre-registration and test NMSE,
  LMVAL-DYN vs LMVAL, on each dataset.
* A dataset extends (`sr-ext-<dataset>.json`, splits 11-20, all five arms) when one of its two
  primary contrasts is "extend" at stage 1. Then both contrasts of that dataset have 20 pairs at
  stage 2. Only a contrast that was "extend" at stage 1 takes its verdict from stage 2.
* The runner gives the validation split (`--const-opt-val`) to each arm with
  `--const-opt-guard val`, so to LMVAL, LMVAL-PEN125 and LMVAL-C250.
* The phase has a `datasets:` requirement. Run `setup.sh` after `git pull`.

Smoke test: `configs/gomea/server-smoke7-1aw` (2 splits, 15 s per run), as part of
`bash tools/server/start.sh --plan smoke6`, because it reads the smoke choice of round 6. The state
is in `server-run7-smoke/`.

## Round 8 and the chain from round 7

The eighth round (`SERVER-ROUND8-PREREGISTRATION.md`: the confirmation round on splits 21-30) has
its own plan, `tools/server/plan8.json`: S0, then the choice of FINAL, then phase 1ax. Its state is in
`../gomea-results/server-run8/`. `plan7.json` has `"then": "round8"`, so one command runs both
rounds, one after the other, without a pause:

    bash tools/server/start.sh --plan round7

The chain works as in "Round 5 and the chain from round 4": two summaries, two final bundles, one
chain lock (`server-run7/chain.lock`). Bring back both bundles.

**The choice of FINAL.** The step `choose-final` (step type `choice`) runs `tools/gomea/set_final.py`:

1. It reads the final 1aw report: `report-stage2` if a primary contrast was "extend" at stage 1,
   else `report-stage1`. Only an "extend" contrast takes its verdict from stage 2.
2. Rule 2 of 1aw passes if LMVAL-PEN125 vs LMVAL on `program_size` is "A better" on at least 4 of
   6 datasets, and the median test NMSE of LMVAL-PEN125 / LMVAL is at most 1.10 on each dataset.
3. FINAL gets the exact argv of the LMVAL-PEN125 arm of 1aw if rule 2 passes, else of the LMVAL arm.
   It reads that argv from the resolved 1aw of round 7 (`--source-dir
   server-run7/phase1aw-config`), so FINAL has POP.
4. With `--population server-run6/phase1az-config/population-choice.json` it gives NOCAP the
   population POP, and the control arm FINAL-DYN the argv of FINAL with the control population.
5. It writes `server-run8/phase1ax-config/`: `sr.json` (the resolved manifest), `spec.json` (a copy)
   and `final-choice.json` (the rule, the numbers and the choice). The orchestrator logs
   `final-choice.json` as the `choice` event, and `round-summary.md` has a "Choice" section.

The choice never writes a tracked file. `configs/gomea/phase1ax/` has only `template.json` (FINAL has
`"argv": null`) and `spec.json`. The phase step has `"config_dir": "{state}/phase1ax-config"`, and the
orchestrator gives it to `phase_run.sh` as `PHASE_CONFIG_DIR`. So the tree stays clean, and no later S0
test fails because of the choice (round 3 had this problem with `set_const_opt.py`). If the choice
fails, the step is failed, and phase 1ax is skipped with the reason in the log. A second run of
`set_final.py` with another result refuses to replace `sr.json` (use `--force` only by hand).

Arms and files:

* `configs/gomea/phase1ax/`: FINAL, NOCAP, HC-LS and FINAL-DYN on Air, Bos, Con, Dow, EnC, EnH, Tow,
  WiR, WiW, Yac, splits 21-30, 1,000 s, wall cap 2,400 s, 3,072 MB (as 1ar). 400 runs. No arm has a
  per-deme cap: HC-LS is `-a hc`, not `hc-capped`, and keeps its flags. There is no extension.
* `spec.json` has `pooled_primary`: FINAL vs HC-LS and FINAL vs NOCAP, each pooled over the 100
  (dataset, split) pairs. `phase_report.py` tests r = log(NMSE_A / NMSE_B) with a paired Wilcoxon
  signed-rank test and Holm across the 2. A failed run gets the largest rank against it. Verdict:
  "A better" if Holm p < 0.025 and exp(median r) <= 0.9, "A worse" if exp(median r) >= 1/0.9. The rows
  have problem `pooled`, `median_log_ratio` and `ratio`.
* The 20 secondary contrasts (each contrast on each dataset) have Holm across the 20 but no verdict
  (`"secondary_verdicts": false`).
* `pooled_secondary`: FINAL-DYN vs FINAL, pooled over the 100 pairs with the test of the primary
  contrasts, its own Holm family (of 1) and a verdict. It changes no rule and not the choice of
  FINAL.
* The phase has a `datasets:` requirement for all 10 datasets. `setup.sh` downloads Dow and Tow.

Smoke test: `bash tools/server/start.sh --plan smoke6` runs the miniatures of rounds 6 and 7, then
`smoke8` (`configs/gomea/server-smoke8-1ax`: splits 21-22, 15 s per run; the choice reads the smoke
1aw report and the smoke population of round 6). The state is in `server-run8-smoke/`.

## Round 9

The ninth round (`SERVER-ROUND9-PREREGISTRATION.md`: the new base on other datasets, seeding with
the new base, and the cap of 250 with LM-VAL) has its own plan, `tools/server/plan9.json`: S0, then
phases 1ba, 1bb and 1bc. Its state is in `../gomea-results/server-run9/`. Start it with:

    bash tools/server/start.sh --plan round9

The three phases are independent. There is no choice step, and no plan chains to or from round 9.
Round 9 never overlaps with another round. One summary and one final bundle. Bring back the bundle.

**Budget.** Each run gets 20,000 evaluations (`"budget": {"evals": 20000}`, so `-m 20000`) and no
`--max-time`. The wall cap of 36,000 s per run is only a guard against a hung process. CPU time is
measured, not capped.

**The new base (BASE).** `-a gomea --gomea-fos lt-b --gomea-pop-size dynamic --hc-fraction-of-nn 0.05
--gomea-skip-inactive 0 --gomea-genotype-memo 1 --gomea-contin copy --knob-probe-baseline reduced
--linear-scaling 1 --const-opt lm --const-opt-guard val --gomea-accept-complexity-ratio 0.125
--gomea-deme-store accepted`. BASE-N16 has `--gomea-pop-size 16` in place of the two dynamic flags.
The manifests are resolved: no `{POP}` token and no template.

Arms and files:

* `configs/gomea/phase1ba/`: BASE and BASE-N16 on Air, Bos, Dow, EnC, Tow, WiR, WiW, Yac, splits
  1-10. `sr.json` has the 7 datasets without Dow. `sr-Dow.json` has Dow alone, so that its
  `mem_limit_mb` can change without a change to the other manifests. Extensions:
  `sr-ext-<dataset>.json` (splits 11-20), with `sr-ext-Dow.json` for Dow. 160 runs (+ up to 160).
* `spec.json` of 1ba: 8 primary contrasts (test NMSE, BASE vs BASE-N16, on each dataset) and 24
  secondary (program size, train-test gap and CPU, on each dataset).
* `configs/gomea/phase1bb/`: BASE and BASE-S on Con, EnH, Air, Yac, splits 1-10, extensions
  `sr-ext-<dataset>.json` (splits 11-20). BASE-S has the seed block of GOMEA-S in 1ab (`share` 0.2,
  so 4,000 of the 20,000 evaluations go to GP-GOMEA, height 3, k 5, functions `+_-_*`, the empty
  program kept). 80 runs (+ up to 80). The phase needs the GP-GOMEA build of `setup.sh`
  (requirement `gpgomea`). Without it, 1bb is skipped with the reason in the log.
* `spec.json` of 1bb: 4 primary contrasts (test NMSE, BASE-S vs BASE) and 8 secondary (program size
  and CPU; a seeded run's `cpu_s` includes GP-GOMEA's CPU).
* `configs/gomea/phase1bc/`: BASE, BASE-C250, N16 and N16-C250 (the C250 arms add
  `--gomea-max-evals-per-deme 250`). `sr.json`: Con, Tow, Yac, WiR on splits 11-20.
  `sr-BosEnC.json`: Bos and EnC on splits 1-10. This is data that 1aw did not use. There is no
  extension. 240 runs.
* `spec.json` of 1bc: 6 primary contrasts (test NMSE, BASE-C250 vs BASE) and 18 secondary in two
  Holm families. Family A (6): test NMSE, N16-C250 vs N16. Family B (12): program size and LM
  calls, BASE-C250 vs BASE. The key `"secondary_families": [["n16c250-vs-n16"]]` sets this. It has
  `"extend": false`, so a primary contrast with Holm p in [0.025, 0.25) is "tie", not "extend".

**New report metrics.** `phase_report.py` has three new contrast metrics for the secondary
contrasts: `train_test_gap` (test NMSE minus training NMSE), `cpu_s` (the run's CPU) and `lm_calls`
(`const_opt.lm_calls`). Lower is better for each, and a failed run is infinitely bad.
The spec key `secondary_families` (a list of lists of secondary contrast names) makes each list a
Holm family of its own. The other secondary contrasts form one more family. In `contrasts.csv`
the column `holm_family` shows the family. Without the key, all secondary contrasts are one
family, as before.

**Memory and workers.** The Dow memory check found that the use is real (the reduced-candidate
cache). So the memory limits (`mem_limit_mb`) are:

| Manifests | `mem_limit_mb` | W on the server (cgroup 65,536 MB, 12 CPUs) |
|---|---|---|
| `phase1ba/sr-Dow.json`, `phase1ba/sr-ext-Dow.json` | 8,192 | floor((65,536 - 6,553) / 8,192) = 7 |
| every other round 9 manifest (1ba, 1bb, 1bc) | 5,120 | min(12 - 1, floor(58,983 / 5,120) = 11) = 11 |

W comes from `resources.py`, as before. Two plan keys on a phase step set its `PER_RUN_MB`:

* `per_run_mb` (5120 on the steps of 1ba, 1bb and 1bc): the `PER_RUN_MB` of the decision for the
  phase. It replaces `PER_RUN_MB` from the environment.
* `manifest_per_run_mb` (1ba only: `{"sr-Dow.json": 8192, "sr-ext-Dow.json": 8192}`): the
  orchestrator makes one more decision for each named manifest. It gives these W values to
  `phase_run.sh` in `MANIFEST_WORKERS="sr-Dow.json=7 sr-ext-Dow.json=7"`.

`phase_run.sh` runs one manifest at a time, and only the Dow manifests have Dow runs. So never
more than the Dow W of Dow runs run at the same time, and Dow runs never run beside other runs.
The 1ba report and its Holm family still hold all 8 datasets, and the Dow extension works as
before. `events.jsonl` has one `workers_decision` for the phase and one for each Dow manifest
(field `manifest`, with `per_run_mb` 8192). `phase_attempt` has `manifest_workers`. Plans 2 to 8
have neither key, so their W comes from the environment, as before. If you change a limit,
change the manifests and the plan key together (`tests/server/test_plan9.py` checks this).
`MAX_WORKERS` caps each decision.

**Size and time.** Stage 1 is 480 runs, about 16 hours at W = 11 (the 20 Dow runs at W = 7), plus S0. With all extensions, up to
720 runs.

Smoke test: `bash tools/server/start.sh --plan smoke9` runs the limited S0 and the miniatures
`configs/gomea/server-smoke9-1ba`, `-1bb` and `-1bc`. They have the same manifests, arms and flags,
2 splits each, a wall cap of 300 s and no extension manifests. The budget is `-m 300`, and
`-m 1000` in 1bb, because the first GP-GOMEA generation alone costs about 600 evaluations. The smoke
includes the Dow run and the seeded arm. The state is in `server-run9-smoke/`. The smoke plan has the
same `per_run_mb` and `manifest_per_run_mb` keys, so the smoke Dow run also gets its own W.

## Round 10

The tenth round (`SERVER-ROUND10-PREREGISTRATION.md`, roadmap task T1.2: the new base against MOSES
hill climbing with LM-VAL and against plain GOMEA) has its own plan, `tools/server/plan10.json`:
S0, then phase 1bd. Its state is in `../gomea-results/server-run10/`. Start it with:

    bash tools/server/start.sh --plan round10

There is no choice step, no extension and no adoption rule: the round only reports the results.
No plan chains to or from round 10. Round 10 never overlaps with another round. Bring back the
final bundle.

**Budget.** Each run gets 20,000 evaluations (`"budget": {"evals": 20000}`, so `-m 20000`) and no
`--max-time`. LM passes, guard passes and GP-GOMEA seed evaluations count. The wall cap of 36,000 s
per run is only a guard against a hung process.

**Arms** (`configs/gomea/phase1bd/`, the same five arms in each manifest):

| Arm | Flags |
|---|---|
| BASE | the round 9 base (as in 1bb) |
| BASE-S | BASE with the seed block of 1bb (`share` 0.2, height 3, k 5, functions `+_-_*`) |
| NOCAP | `-a gomea --gomea-fos lt-b --gomea-pop-size 16 --gomea-skip-inactive 0 --gomea-genotype-memo 1 --gomea-contin copy --knob-probe-baseline reduced --linear-scaling 1` (round 8) |
| HC-LM | `-a hc --knob-probe-baseline reduced --linear-scaling 1 --const-opt lm --const-opt-guard val` |
| HC-LM-W | HC-LM + `--hc-widen-search 1` |

* HC-LM and HC-LM-W use the new LM for hill climbing (`pymoses/gomea/hc_lm.py`): at the end of each
  HC deme, LM on the deme's best program, with the same code and rule as GOMEA. The runner gives
  them `--const-opt-val` (the scaled validation split) and `--gomea-log RUN_DIR/events.jsonl`, as
  for the GOMEA arms. Each HC deme writes one "end" event, so the summary has `const_opt` (LM
  calls, guard rejections) and the report has the deme count and the steps per deme.
* HC-LM-W keeps an LM reserve: in each deme the climb gets the deme budget minus
  min(500, budget // 2) evaluations, and LM gets the rest (the "end" event has `lm_reserve`).
* Datasets: all 10 (Air, Bos, Con, Dow, EnC, EnH, Tow, WiR, WiW, Yac), splits 1-10. 500 runs.
* The phase needs the GP-GOMEA build of `setup.sh` (requirement `gpgomea`, for BASE-S). Without it,
  1bd is skipped with the reason in the log.

**Report** (`spec.json` of 1bd; the pooled test of round 8: paired Wilcoxon on
r = log(NMSE_A / NMSE_B) over the 100 pairs, a failed run gets the largest rank against it):

* Primary, pooled, one Holm family (3): BASE vs HC-LM, BASE vs HC-LM-W, BASE vs NOCAP
  (`pooled_primary`).
* Seeding control, pooled, its own Holm family (4): BASE-S vs BASE, HC-LM, HC-LM-W and NOCAP.
* Widen search, pooled, its own Holm family (1): HC-LM-W vs HC-LM.
* Secondary, pooled, one Holm family (10): program size and CPU for BASE vs HC-LM, BASE vs HC-LM-W,
  BASE vs NOCAP, BASE-S vs BASE and HC-LM-W vs HC-LM.
* Per dataset, descriptive (40): contrasts 1 to 4 on each dataset, Holm across the 40, no verdict
  (`secondary` with `"secondary_verdicts": false`).

The control, widen and size/CPU contrasts are `pooled_secondary` rows. The new spec key
`pooled_secondary_families` (`[[the 4 control names], ["hclmw-vs-hclm"]]`) makes each inner list a
Holm family of its own; the other pooled secondary contrasts form one more family. In
`contrasts.csv` the column `holm_family` shows the family of each pooled secondary row
(`pooled_secondary/<names>` or `pooled_secondary`). Without the key nothing changes for other
phases. The verdicts describe the data only.

**Memory and workers.** Round 9 showed that N = 16 needs much memory at 20,000 evaluations. Each of
Dow and Tow has a manifest of its own:

| Manifest | `mem_limit_mb` | W on the server (cgroup 65,536 MB, 12 CPUs, reserve 6,553 MB) |
|---|---|---|
| `phase1bd/sr-Dow.json` | 16,384 | floor(58,983 / 16,384) = 3 |
| `phase1bd/sr-Tow.json` | 8,192 | floor(58,983 / 8,192) = 7 |
| `phase1bd/sr.json` (the 8 other datasets) | 5,120 | min(11, 11) = 11 |

The plan step has `per_run_mb` 5120 and `manifest_per_run_mb`
`{"sr-Dow.json": 16384, "sr-Tow.json": 8192}` (the round 9 mechanism). `events.jsonl` has one
`workers_decision` for the phase and one for each of the two manifests. `phase_run.sh` runs one
manifest at a time (`sr.json`, then `sr-Dow.json`, then `sr-Tow.json`). If a NOCAP run still fails
on memory, the report counts it as a BASE win; the findings then give contrast 3 with and without
the failed pairs.

**Size and time.** 500 runs, about 12-17 hours at W = 11, plus about 1.3 hours for S0. The 50 Dow
runs at W = 3 add about 1-2 hours.

Smoke test: `bash tools/server/start.sh --plan smoke10` runs the limited S0 (with the new tests)
and `configs/gomea/server-smoke10-1bd`: the same three manifests, arms and flags, splits 1 and 2,
`-m 1000` (the first GP-GOMEA generation alone costs about 600 evaluations), a wall cap of 300 s
and no extension. 100 runs. The state is in `server-run10-smoke/`. The smoke plan has the same
`per_run_mb` and `manifest_per_run_mb` keys, so the smoke Dow and Tow runs also get their own W.

## Round 11

The eleventh round (`SERVER-ROUND11-PREREGISTRATION.md`: do shorter GOMEA demes make the base
more accurate, and how much larger do the programs get) has its own plan,
`tools/server/plan11.json`: S0, then phase 1be. Its state is in `../gomea-results/server-run11/`.

**Before the round, run `bash tools/server/setup.sh` again.** Round 11 uses the fresh splits
31-40. A dataset set from an earlier round has splits 1-30 only, so set-up rebuilds it (splits
1-30 do not change). Then start the round with:

    bash tools/server/start.sh --plan round11

There is no choice step, no extension and no adoption rule: the round only reports the results.
No plan chains to or from round 11. Round 11 never overlaps with another round. Bring back the
final bundle.

**Budget.** Each run gets 20,000 evaluations (`"budget": {"evals": 20000}`, so `-m 20000`) and no
`--max-time`. LM passes and guard passes count. The wall cap of 36,000 s per run is only a guard
against a hung process.

**Arms** (`configs/gomea/phase1be/`, the same six arms in each manifest):

| Arm | Flags |
|---|---|
| BASE | the round 10 BASE (as in 1bd) |
| BASE-R | BASE + `--const-opt-reserve 500` |
| BASE-D5 | BASE + `--gomea-deme-share 5` (a deme cap of 4,000 evaluations) |
| BASE-D10 | BASE + `--gomea-deme-share 10` (a deme cap of 2,000 evaluations) |
| BASE-D20 | BASE + `--gomea-deme-share 20` (a deme cap of 1,000 evaluations) |
| HC-LM | the round 10 HC-LM (as in 1bd) |

* `--const-opt-reserve R` is new (`pymoses/gomea/optimizer.py`, opt-in): with `-a gomea
  --const-opt lm`, the search of each deme stops at the deme budget minus min(R, deme budget
  // 2). LM on the deme's best program can then use the evaluations that are left. A deme
  that ends by stagnation before that point does not change. The "end" event of each deme has
  `const_opt_reserve` (the reserve of that deme). If LM does not use all of the reserve, MOSES
  starts the next deme with the evaluations that are left, as for HC-LM-W in round 10.
* `--gomea-deme-share D` with `--gomea-pop-size dynamic` stopped with a `ValueError` before
  this round (the telemetry asked for N without the deme's field set). Commit 709b27f
  corrected this; N and the search do not change.
* All six arms use `--const-opt-guard val`, so the runner gives each one `--const-opt-val`
  and `--gomea-log RUN_DIR/events.jsonl`.
* Datasets: all 10 (Air, Bos, Con, Dow, EnC, EnH, Tow, WiR, WiW, Yac), splits 31-40. 600 runs.
* The phase needs the datasets with splits 31-40 (requirement
  `datasets:Air,...,Yac:31-40`). Without them, 1be is skipped with the reason in the log.
  There is no seeded arm, so the phase does not need the GP-GOMEA build.

**Report** (`spec.json` of 1be; the pooled test of round 8: paired Wilcoxon on
r = log(NMSE_A / NMSE_B) over the 100 pairs, a failed run gets the largest rank against it):

* Primary, pooled, one Holm family (4): BASE-R, BASE-D5, BASE-D10 and BASE-D20, each vs BASE
  (`pooled_primary`).
* The gap to HC-LM, pooled, its own Holm family (5): BASE, BASE-R, BASE-D5, BASE-D10 and
  BASE-D20, each vs HC-LM (`pooled_secondary_families`).
* Size and CPU, pooled, one Holm family (8): program size and CPU for contrasts 1 to 4.
* Per dataset, descriptive (40): contrasts 1 to 4 on each dataset, Holm across the 40, no
  verdict (`secondary` with `"secondary_verdicts": false`).

The verdicts describe the data only. For the trade-off table of the pre-registration,
`contrasts.csv` has the pooled ratios (`ratio` = exp(median r)) of test NMSE against BASE and
against HC-LM and of program size against BASE. `aggregate.csv` has, for each dataset and arm,
the median demes in a run (`median_deme_count`) and the median LM share of the evaluations
(`median_tuning_share`).

**Memory and workers.** Dow has a manifest of its own:

| Manifest | `mem_limit_mb` | W on the server (cgroup 65,536 MB, 12 CPUs, reserve 6,553 MB) |
|---|---|---|
| `phase1be/sr-Dow.json` | 8,192 | floor(58,983 / 8,192) = 7 |
| `phase1be/sr.json` (the 9 other datasets) | 5,120 | min(11, 11) = 11 |

The plan step has `per_run_mb` 5120 and `manifest_per_run_mb` `{"sr-Dow.json": 8192}`.
`events.jsonl` has one `workers_decision` for the phase and one for the Dow manifest.
`phase_run.sh` runs one manifest at a time (`sr.json`, then `sr-Dow.json`).

**Size and time.** 600 runs, about 12-18 hours at W = 11 (W = 7 for Dow), plus about 1.3 hours
for S0.

Smoke test: `bash tools/server/start.sh --plan smoke11` runs the limited S0 (with the new tests)
and `configs/gomea/server-smoke11-1be`: the same two manifests, arms and flags, splits 31 and 32,
`-m 1000` (so that the reserve of BASE-R and the caps of the D arms apply), a wall cap of 300 s
and no extension. 120 runs. The state is in `server-run11-smoke/`. The smoke plan has the same
`requires`, `per_run_mb` and `manifest_per_run_mb` keys.

## Round 12

The twelfth round (`SERVER-ROUND12-PREREGISTRATION.md`: does LM on the top 5 programs of each
deme, or one of the four tier 1 ideas, make BASE more accurate at the same budget, at what cost
in program size and CPU, and does BASE-R confirm its round 11 result) has its own plan,
`tools/server/plan12.json`: S0, then phase 1bf (regression), then phase 1bg (mux11). Its state
is in `../gomea-results/server-run12/`.

**One-time server steps before the round:**

1. Make sure that no other round runs (`bash tools/server/status.sh --plan round11` shows
   "not running"). Round 12 never overlaps with another round.
2. Update the clone to the branch commit that has this section (`git pull --ff-only`).
3. Run `bash tools/server/setup.sh` again. Round 12 uses the fresh splits 41-50. A dataset set
   from an earlier round has splits 1-40 only, so set-up rebuilds it (splits 1-40 do not
   change; Dow and Tow are built again from their raw files). It must end with `SETUP OK`.
   Each dataset then has 400 split files (50 splits x 8 files).
4. Start the round:

       bash tools/server/start.sh --plan round12

There is no choice step, no extension and no adoption rule: the round only reports the results.
No plan chains to or from round 12. Bring back the final bundle.

**Phase 1bf: budget.** Each run gets 20,000 evaluations (`"budget": {"evals": 20000}`, so
`-m 20000`) and no `--max-time`. LM passes and guard passes count. The wall cap of 36,000 s per
run is only a guard against a hung process.

**Phase 1bf: arms** (`configs/gomea/phase1bf/`, the same eight arms in each manifest):

| Arm | Task | Flags |
|---|---|---|
| BASE | reference | the round 11 BASE (as in 1be) |
| HC-LM | accuracy baseline | the round 11 HC-LM (as in 1be) |
| BASE-R | T1.5 | BASE + `--const-opt-reserve 500` (as in 1be) |
| BASE-K5 | T1.6 | BASE + `--const-opt-reserve 500 --const-opt-top-k 5` |
| BASE-NP | T2.1 | BASE with `--gomea-fos np` in place of `lt-b` |
| BASE-TLS | T2.2 | BASE with `--linear-scaling terms` and `--const-opt none`, without `--const-opt-guard val` |
| BASE-LME | T2.3 | BASE + `--const-opt-during elite` |
| BASE-DIV | T2.4 | BASE + `--diversity-pressure {6*n_train} --diversity-autoscale 0` |

* The four new options are opt-in (`docs/gomea/USAGE.md`). Without them, the output of every
  existing setting does not change (the identity tests in `tests/gomea/`).
* BASE-TLS has no LM and no guard, so the runner gives it no `--const-opt-val` (the CLI
  refuses that table without the guard). It scores each candidate with the per-term weights
  refitted on the training split, and its summaries have a `terms` record. All other arms
  use `--const-opt-guard val`, so the runner gives them `--const-opt-val`. All eight arms get
  `--gomea-log RUN_DIR/events.jsonl`.
* BASE-DIV: P = 6 x the training rows of the run's dataset. The runner puts the integer in the
  place of `{6*n_train}` (`moses_sr_runner.resolve_argv`); no other arm has a placeholder.

  | Dataset | Air | Bos | Con | Dow | EnC | EnH | Tow | WiR | WiW | Yac |
  |---|---|---|---|---|---|---|---|---|---|---|
  | Training rows | 753 | 254 | 516 | 534 | 384 | 384 | 2,501 | 801 | 2,450 | 154 |
  | P | 4,518 | 1,524 | 3,096 | 3,204 | 2,304 | 2,304 | 15,006 | 4,806 | 14,700 | 924 |

* Datasets: all 10, splits 41-50. 800 runs. The phase needs the datasets with splits 41-50
  (requirement `datasets:Air,...,Yac:41-50`). Without them, 1bf is skipped with the reason in
  the log.

**Phase 1bf: report** (`spec.json` of 1bf; the pooled test of round 8: paired Wilcoxon on
r = log(NMSE_A / NMSE_B) over the 100 pairs, a failed run gets the largest rank against it):

* Primary, pooled, one Holm family (6): BASE-R, BASE-NP, BASE-TLS, BASE-LME and BASE-DIV, each
  vs BASE, and BASE-K5 vs BASE-R (`pooled_primary`).
* The gap to HC-LM, pooled, its own Holm family (7): BASE and the 6 arms, each vs HC-LM.
* Size and CPU, pooled, one Holm family (12): program size and CPU for the 6 primary
  contrasts (BASE-K5 against BASE-R).
* BASE-K5 vs BASE (test NMSE, pooled): **descriptive** (both changes together). It is a Holm
  family of one in `pooled_secondary_families`, so `contrasts.csv` gives it a p, a Holm p and a
  verdict. Read that verdict as a description only.
* Per dataset, descriptive (60): the 6 primary contrasts on each dataset, Holm across the 60, no
  verdict (`secondary` with `"secondary_verdicts": false`).

The verdicts describe the data only. For the trade-off table, `contrasts.csv` has the pooled
ratios (`ratio` = exp(median r)) of test NMSE against BASE and against HC-LM, and of program size
and CPU against BASE. `aggregate.csv` has, for each dataset and arm, the median demes in a run and
the median LM share of the evaluations.

**Phase 1bg (mux11)** (`configs/gomea/phase1bg/`): four arms with the Boolean setting of round 6
(`-a gomea --gomea-pop-size dynamic --hc-fraction-of-nn 0.05 --gomea-skip-inactive 0
--gomea-max-evals-per-deme 0 --gomea-genotype-memo 1 --knob-probe-baseline reduced`) and
`--gomea-fos` `np` (NP), `lt-b` (LTB), `rt` (RT) or `uni` (UNI). Fresh seeds 3101-3110, a wall-time
budget of 3,600 s per run (`--max-time 3600 -m 1000000000`), a wall cap of 5,400 s. 40 runs.
Metric: CPU to solve (an unsolved run is infinite). Primary, one Holm family (3): NP vs RT, NP vs
UNI and NP vs LTB. No extension.

* The manifest names the code commit defa8fe (`"commit"`), as the Boolean manifests of rounds 4
  and 6 do, and the plan has no `set_commit_in`. The Boolean runner refuses to start if code under
  `pymoses/`, `benchmarks/` or `pyproject.toml` changed after that commit, or if the clone has
  local changes there. Then 1bg fails with the reason in `../gomea-results/phase1bg-bench.log`.

**Memory and workers:**

| Manifest | `mem_limit_mb` | W on the server (cgroup 65,536 MB, 12 CPUs, reserve 6,553 MB) |
|---|---|---|
| `phase1bf/sr-Dow.json` | 8,192 | floor(58,983 / 8,192) = 7 |
| `phase1bf/sr.json` (the 9 other datasets) | 5,120 | min(11, 11) = 11 |
| `phase1bg/bool.json` (mux11) | 3,072 (`phase_run.sh`) | min(11, 19) = 11 |

The 1bf step has `per_run_mb` 5120 and `manifest_per_run_mb` `{"sr-Dow.json": 8192}`; the 1bg
step has `per_run_mb` 3072. `events.jsonl` has one `workers_decision` for each phase and one
for the Dow manifest. `phase_run.sh` runs one manifest at a time.

**Size and time.** 1bf: 800 runs, about 160 CPU-hours, about 15 hours at W = 11 (W = 7 for Dow).
1bg: at most 40 CPU-hours, at most 4 hours (4 waves of 3,600 s at W = 11). S0: about 1.3 hours.
About 17 to 20 hours in total.

Smoke test: `bash tools/server/start.sh --plan smoke12` runs the limited S0 (with the tests of
the round 12 options), then `configs/gomea/server-smoke12-1bf` (the same two manifests, arms and
flags, splits 41 and 42, `-m 1000`, so that the reserve of BASE-R and BASE-K5 applies, a wall cap
of 300 s; 160 runs) and `configs/gomea/server-smoke12-1bg` (the same four arms, seeds 3101 and
3102, `--max-time 120`; 8 runs). The state is in `server-run12-smoke/`. The smoke plan has the
same `requires`, `per_run_mb` and `manifest_per_run_mb` keys.

## Round 13

The thirteenth round (`SERVER-ROUND13-PREREGISTRATION.md`: does BASE-TLS beat BASE again on fresh
splits, do node-proximity linkage, a validation guard or LM on the inner constants improve
BASE-TLS, and do ε-lexicase exemplar selection and a pole safeguard improve BASE) has its own plan,
`tools/server/plan13.json`: S0, then phase 1bh (regression). There is no Boolean phase. Its state
is in `../gomea-results/server-run13/`.

**One-time server steps before the round:**

1. Make sure that no other round runs (`bash tools/server/status.sh --plan round12` shows
   "not running"). Round 13 never overlaps with another round.
2. Update the clone to the branch commit that has this section (`git pull --ff-only`).
3. Run `bash tools/server/setup.sh` again. Round 13 uses the fresh splits 51-60. A dataset set
   from an earlier round has splits 1-50 only, so set-up rebuilds it (splits 1-50 do not
   change; Dow and Tow are built again from their raw files). It must end with `SETUP OK`.
   Each dataset then has 480 split files (60 splits x 8 files).
4. Start the round:

       bash tools/server/start.sh --plan round13

There is no choice step, no extension and no adoption rule: the round only reports the results
(the confirmation rule for BASE-TLS is a proposal to the user after the round). No plan chains to
or from round 13. Bring back the final bundle.

**Phase 1bh: budget.** Each run gets 20,000 evaluations (`"budget": {"evals": 20000}`, so
`-m 20000`) and no `--max-time`. LM passes, guard passes and pole-check passes count. The wall cap
of 36,000 s per run is only a guard against a hung process.

**Phase 1bh: arms** (`configs/gomea/phase1bh/`, the same eight arms in each manifest):

| Arm | Task | Flags |
|---|---|---|
| BASE | reference | the round 12 BASE (as in 1bf) |
| HC-LM | accuracy baseline | the round 12 HC-LM (as in 1bf) |
| BASE-TLS | T2.2, confirmation | the round 12 BASE-TLS (as in 1bf): `--linear-scaling terms --const-opt none`, no guard |
| TLS-NP | T2.1 + T2.2 | BASE-TLS with `--gomea-fos np` in place of `lt-b` |
| TLS-G | TLS guard | the BASE-TLS argv, and the arm key `"terms_guard": "val"` (runner only) |
| TLS-LM | T3.1 | BASE-TLS with `--const-opt lm --const-opt-guard val` in place of `--const-opt none` |
| BASE-LEX | A4, ε-lexicase | BASE + `--exemplar-selection lexicase` |
| BASE-PS | pole safeguard | BASE + `--const-opt-pole-check box` |

* The new options are opt-in (`docs/gomea/USAGE.md`). Without them, the output of every existing
  setting does not change (the identity tests in `tests/gomea/`).
* BASE-TLS, TLS-NP and TLS-G have no LM and no guard, so the runner gives them no
  `--const-opt-val` (the CLI refuses that table without the guard). TLS-LM and the other arms use
  `--const-opt-guard val`, so the runner gives them `--const-opt-val`. All eight arms get
  `--gomea-log RUN_DIR/events.jsonl`.
* TLS-G: MOSES gets the same command line as BASE-TLS, so the search is the same. The runner
  (`moses_sr_runner.py`) compares, for each printed candidate, the validation NMSE of the
  per-term model and of the plain a + b·f (both fitted on the training rows), and keeps the lower
  one; a tie keeps plain. Then it selects the candidate with the lowest validation NMSE, as for
  every arm. The summary has `terms.guard` (`terms` or `plain`); `candidates.json` has the choice
  of each candidate. The runner refuses `terms_guard` on an arm without `--linear-scaling terms`.
* New fields in the summaries (no other value changes): `test_nmse_clipped` for every arm (the
  test NMSE with the predictions clipped to [min − 0.1·R, max + 0.1·R] of the training target;
  the runner writes it because the manifests of 1bh have `"clip_test_nmse": true`);
  `plain_val_nmse` and `plain_test_nmse` for the four TLS arms (the plain a + b·f, descriptive;
  the runner writes them because the manifests of 1bh have `"plain_nmse": true`).
  BASE-PS summaries have `const_opt.lm_pole_checked` and `lm_pole_rejected`; BASE-LEX summaries
  have the `lexicase` block.
* Datasets: all 10, splits 51-60. 800 runs. The phase needs the datasets with splits 51-60
  (requirement `datasets:Air,...,Yac:51-60`). Without them, 1bh is skipped with the reason in
  the log.

**Phase 1bh: report** (`spec.json` of 1bh; the pooled test of round 8: paired Wilcoxon on
r = log(A / B) over the 100 pairs, a failed run gets the largest rank against it):

* Primary, pooled, one Holm family (6), test NMSE: BASE-TLS vs BASE (the confirmation); TLS-NP,
  TLS-G and TLS-LM, each vs BASE-TLS; BASE-LEX and BASE-PS, each vs BASE (`pooled_primary`).
* The gap to HC-LM, pooled, its own Holm family (7): BASE and the 6 other arms, each vs HC-LM.
* Size and CPU, pooled, one Holm family (12): program size and CPU for the 6 primary contrasts.
  The program size of a TLS arm is its node count plus `n_terms` when the final model uses the
  per-term fit (2 or more terms, no fallback; for TLS-G only when `terms.guard` is `terms`); a
  final model with plain a + b·f adds 0 (spec key `"size_terms": true`). `aggregate.csv` also has
  the node count (`median_nodes`).
* **Descriptive**, pooled, a Holm family of its own (9): TLS-NP, TLS-G and TLS-LM, each vs BASE
  (test NMSE), and the 6 primary contrasts on the clipped test NMSE (`test_nmse_clipped`).
  `contrasts.csv` gives them a p, a Holm p and a verdict. Read those verdicts as a description
  only.
* Per dataset, descriptive (60): the 6 primary contrasts on each dataset, Holm across the 60, no
  verdict (`secondary` with `"secondary_verdicts": false`).
* `report-stage1/runs.csv` (spec key `"run_table": true`): one row for each run, with the status,
  test and clipped test NMSE, training and validation NMSE, size and node count, `n_terms`,
  fallback and guard (TLS arms), the TLS-G candidates that kept plain scaling, CPU, evaluations,
  demes, knobs per deme (median of the run's demes) and total knobs (from `events.jsonl`), LM
  calls and LM share, guard checks and rejections, pole checks and rejections, and the lexicase
  values. `report-stage1/nmse_above_100.csv`: the ok runs with a test NMSE above 100, with the
  training and validation NMSE.
* `report-stage1/same_search.csv` and `same_search.json` (spec key `"same_search"`): for each
  dataset and split, whether TLS-G and BASE-TLS printed the same candidates (the same programs in
  the same order). `same_search.json` gives the number of pairs that differ; it must be 0.

**Memory and workers:**

| Manifest | `mem_limit_mb` | W on the server (cgroup 65,536 MB, 12 CPUs, reserve 6,553 MB) |
|---|---|---|
| `phase1bh/sr-Dow.json` | 8,192 | floor(58,983 / 8,192) = 7 |
| `phase1bh/sr.json` (the 9 other datasets) | 5,120 | min(11, 11) = 11 |

The 1bh step has `per_run_mb` 5120 and `manifest_per_run_mb` `{"sr-Dow.json": 8192}`.
`events.jsonl` has one `workers_decision` for the phase and one for the Dow manifest.
`phase_run.sh` runs one manifest at a time.

**Size and time.** 1bh: 800 runs, about 140 to 160 CPU-hours, about 13 to 15 hours at W = 11
(W = 7 for Dow). S0: about 1.3 hours. About 15 to 18 hours in total.

Smoke test: `bash tools/server/start.sh --plan smoke13` runs the limited S0 (`tests/server`, the
split test, the tests of the round 13 options and of the round 13 runner and report changes; the
mux11 repro run), then `configs/gomea/server-smoke13-1bh` (the same two manifests, arms, flags and
spec, splits 51 and 52, `-m 1000`, a wall cap of 300 s; 160 runs). The state is in
`server-run13-smoke/`. The smoke plan has the same `requires`, `per_run_mb` and
`manifest_per_run_mb` keys.

## Round 14

The fourteenth round (`SERVER-ROUND14-PREREGISTRATION.md`: does LM help when it gets a budget
(the LM reserve), with per-term scaling and under node-proximity linkage; does a relative pole
rule keep the good LM tunings; does TLS-NP stay near BASE-TLS on new splits; and how much of the
BASE-TLS accuracy does a plain linear model reach) has its own plan, `tools/server/plan14.json`:
S0, then phase 1bi (regression). There is no Boolean phase. Its state is in
`../gomea-results/server-run14/`.

**One-time server steps before the round:**

1. Make sure that no other round runs (`bash tools/server/status.sh --plan round13` shows
   "not running"). Round 14 never overlaps with another round.
2. Update the clone to the branch commit that has this section:

       git pull --ff-only

3. Run the set-up again:

       bash tools/server/setup.sh

   It must end with `SETUP OK`. Round 14 uses the fresh splits 61-70. A dataset set from an
   earlier round has splits 1-60 only, so set-up rebuilds it (splits 1-60 do not change; Dow and
   Tow are built again from their raw files). Each dataset then has 560 split files (70 splits x
   8 files).
4. Start the round:

       bash tools/server/start.sh --plan round14

That is all. Do not run a separate smoke test on the server: the round runs everything in one
go. S0 (the full pytest and the reproducibility check) records a failure in the log and in
`round-summary.md`, but it does not stop the phase. There is no choice step, no extension and no
adoption rule: the round only reports the results (the reference rule for BASE-TLS is a proposal
to the user after the round). No plan chains to or from round 14. Bring back the final bundle.

**Phase 1bi: budget.** Each MOSES run gets 20,000 evaluations (`"budget": {"evals": 20000}`, so
`-m 20000`) and no `--max-time`. LM passes, guard passes and pole-check passes count. The wall cap
of 36,000 s per run is only a guard against a hung process. The linear fits have no budget.

**Phase 1bi: arms** (`configs/gomea/phase1bi/`, the same ten arms in each manifest):

| Arm | Question | Flags |
|---|---|---|
| BASE-TLS | reference | the round 13 BASE-TLS (as in 1bh) |
| TLS-NP | next in line (Q4) | the round 13 TLS-NP (as in 1bh) |
| HC-LM | accuracy baseline | the round 13 HC-LM (as in 1bh) |
| TLS-LMR | T3.1 with an LM budget (Q1) | BASE-TLS with `--const-opt lm --const-opt-guard val --const-opt-reserve 500` in place of `--const-opt none` |
| TLSNP-LMR | LM under NP, per-term (Q2) | TLS-NP with the same change |
| NP-LMR | LM under NP, plain (Q2) | the round 12 BASE-NP (as in 1bf) + `--const-opt-reserve 500` |
| NP-NOLM | LM under NP, plain (Q2) | NP-LMR with `--const-opt none`, no guard and no reserve |
| NP-LMR-PSR | relative pole rule (Q3) | NP-LMR + `--const-opt-pole-check box-rel` |
| LIN-OLS | linear baseline (Q5) | the arm key `"linear": "ols"` (runner only, no MOSES run) |
| LIN-RIDGE | linear baseline (Q5) | the arm key `"linear": "ridge"` (runner only, no MOSES run) |

* The new option `box-rel` is opt-in (`docs/gomea/USAGE.md`). Without it, the output of every
  existing setting does not change (the identity tests in `tests/gomea/`).
* BASE-TLS, TLS-NP and NP-NOLM have no LM and no guard, so the runner gives them no
  `--const-opt-val` (the CLI refuses that table without the guard). HC-LM and the four reserve
  arms use `--const-opt-guard val`, so the runner gives them `--const-opt-val`. All eight MOSES
  arms get `--gomea-log RUN_DIR/events.jsonl`.
* **LIN-OLS and LIN-RIDGE** (`moses_sr_runner.py` with `sr_linear.py`): the runner starts no
  process. It fits y = w_0 + sum w_i x_i over all inputs of the split (a boolean input, Bos `x4`,
  is 0 or 1) to the training target scaled as for the MOSES arms, and scores the model's own
  predictions with the NMSE functions of `sr_eval.py`. So `train_nmse`, `val_nmse`, `test_nmse`
  and `test_nmse_clipped` mean the same as for the MOSES arms.
  * OLS: least squares with an intercept (`least_squares.fit_terms`). If the inputs are
    collinear or constant (for example EnC and EnH, where x4 is a linear function of x2 and x3),
    the runner drops each aliased input (weight 0) and fits the others; the summary names them
    (`linear.fallback` "aliased", `linear.dropped_inputs`).
  * Ridge: inputs standardized with the training mean and sd, lambda from 10^-6 to 10^3, the
    intercept is not penalized; the lambda with the lowest validation NMSE is used (a tie keeps
    the larger lambda), fitted on the training rows. The summary has `linear.lambda` and the
    validation NMSE of each lambda.
  * The summary has `n_evals` null, `cpu_s` (the CPU time of the fit), the weights, and for the
    size rule `final_program` `+($x1 ... $xk)` and a `terms` record with `n_terms` k. The report
    counts "(" plus atoms, so the size is (k + 2) + k = 2k + 2, the size of the same TLS program
    from MOSES. A fit takes well under a second.
* Summaries of every arm have `test_nmse_clipped` (the manifests of 1bi have
  `"clip_test_nmse": true`). The four TLS arms have `plain_val_nmse` and `plain_test_nmse`
  (`"plain_nmse": true`). The reserve arms report `const_opt.lm_budget` (LM calls with no budget
  left); NP-LMR-PSR reports `const_opt.lm_pole_checked` and `lm_pole_rejected`.
* Datasets: all 10, splits 61-70. 1,000 jobs: 800 MOSES runs and 200 linear fits (100 per arm,
  100 per split, 100 per dataset). The phase needs the datasets with splits 61-70 (requirement
  `datasets:Air,...,Yac:61-70`). Without them, 1bi is skipped with the reason in the log.

**Phase 1bi: report** (`spec.json` of 1bi; the pooled test of round 8: paired Wilcoxon on
r = log(A / B) over the 100 pairs, a failed run gets the largest rank against it):

* Primary, pooled, one Holm family (5), test NMSE: TLS-LMR vs BASE-TLS; TLSNP-LMR vs TLS-NP;
  NP-LMR vs NP-NOLM; NP-LMR-PSR vs NP-LMR; TLS-NP vs BASE-TLS (`pooled_primary`).
* The gap to HC-LM, pooled, its own Holm family (7): each MOSES GOMEA arm vs HC-LM.
* Size and CPU, pooled, one Holm family (10): program size and CPU for the 5 primary contrasts
  (`"size_terms": true`: the size of a final model with the per-term fit is its node count plus
  `n_terms`; `aggregate.csv` also has `median_nodes`).
* **Descriptive**, pooled, a Holm family of its own (14), with the verdict "descriptive" (spec key
  `"pooled_descriptive"`; p, Holm p and ratio stay): BASE-TLS, TLS-NP and HC-LM, each vs LIN-OLS
  and vs LIN-RIDGE (6); NP-NOLM vs TLS-NP; TLS-LMR vs TLSNP-LMR; TLSNP-LMR vs BASE-TLS
  (amendment, the user's decision on RV16 finding 4); the 5 primary contrasts on the clipped test
  NMSE (`test_nmse_clipped`).
* Per dataset, descriptive (50): the 5 primary contrasts on each dataset, Holm across the 50, no
  verdict (`secondary` with `"secondary_verdicts": false`).
* `report-stage1/runs.csv` and `report-stage1/nmse_above_100.csv` (spec key `"run_table": true`),
  as in round 13. The linear rows have no evaluations, demes or LM values. There is no
  same-search check this round.

**Memory and workers:**

| Manifest | `mem_limit_mb` | W on the server (cgroup 65,536 MB, 12 CPUs, reserve 6,553 MB) |
|---|---|---|
| `phase1bi/sr-Dow.json` | 8,192 | floor(58,983 / 8,192) = 7 |
| `phase1bi/sr.json` (the 9 other datasets) | 5,120 | min(11, 11) = 11 |

The 1bi step has `per_run_mb` 5120 and `manifest_per_run_mb` `{"sr-Dow.json": 8192}`.
`events.jsonl` has one `workers_decision` for the phase and one for the Dow manifest.
`phase_run.sh` runs one manifest at a time. A linear fit runs in the runner's worker process (no
child process), so `mem_limit_mb` does not apply to it; it uses a few MB.

**Size and time.** 1bi: 800 MOSES runs, about 125 to 140 CPU-hours, about 12 to 14 hours at
W = 11 (W = 7 for Dow); the 200 linear fits take a few minutes. S0: about 1.3 hours. About 14 to 16
hours in total, in one round.

**Local rehearsal only:** `tools/server/plan14-smoke.json` (`--plan smoke14`) runs a limited S0
(`tests/server`, the split test, the tests of the round 14 runner change, of `box-rel`, of the
guard fix of backlog 12.1 and of the round 13 runner and report; the mux11 repro run), then
`configs/gomea/server-smoke14-1bi` (the same two manifests, arms, flags, keys and spec, splits 61
and 62, `-m 1000`, a wall cap of 300 s; 200 jobs). Its state is in `server-run14-smoke/`. The
smoke plan is for a local check before a push, not a server step.

## Round 15

The fifteenth round (`SERVER-ROUND15-PREREGISTRATION.md`, the IT linkage study, BACKLOG 13.9: does
learned linkage help more on the IT genome than on GP-GOMEA's template; which linkage model works
best with IT; does our engine reproduce the known template result) is one phase, 1bj
(regression), in two plans that chain, because the local pilot showed more than 20 h for one plan
(the user's rule; see "Size and time"):

* `tools/server/plan15.json` (`round15`): S0, then 1bj on the first half of the datasets, Air,
  Con, Dow, EnH and Tow (the D2 fallback set; `configs/gomea/phase1bj-a`, 500 jobs). Its state is in
  `../gomea-results/server-run15/`. It has `"then": "round15b"`.
* `tools/server/plan15b.json` (`round15b`): 1bj with the full manifest of the 10 datasets
  (`configs/gomea/phase1bj`). It starts by itself when round15 ends. The two manifests have the same
  name, so the runs of round15 are in the same result directory (`../gomea-results/phase1bj/`) and
  the runner keeps them; round15b runs the other 500 jobs (Bos, EnC, WiR, WiW, Yac), and its report
  covers all 1,000 runs. It has no S0 (round15 ran S0 on the same commit). Its state is in
  `../gomea-results/server-run15b/`.

There is no Boolean phase and no MOSES run. If round15b cannot run, the report of round15
(`report-stage1` of its phase step, an interim report on 5 datasets) is the D2 fallback design.

**One-time server steps before the round:**

1. Make sure that no other round runs (`bash tools/server/status.sh --plan round14` shows
   "not running"). Round 15 never overlaps with another round.
2. Update the clone to the branch commit that has this section:

       git pull --ff-only

3. Run the set-up again:

       bash tools/server/setup.sh

   It must end with `SETUP OK`. Round 15 uses the fresh splits 71-80. A dataset set from an
   earlier round has splits 1-70 only, so set-up rebuilds it (splits 1-70 do not change; Dow and
   Tow are built again from their raw files). Each dataset then has 640 split files (80 splits x
   8 files).
4. Start the round (one command starts the chain; round15b starts when round15 ends):

       bash tools/server/start.sh --plan round15

That is all. Do not run a separate smoke test on the server. S0 (the full pytest and the
reproducibility check) records a failure in the log and in `round-summary.md`, but it does not stop
the phase. There is no choice step, no extension and no adoption rule. Each plan writes its own
`round-summary.md` and final bundle (as for the chain of rounds 4 and 5): bring back both, from
`server-run15/` and `server-run15b/`. The report on all 1,000 runs is in the bundle of round15b.
`status.sh --plan round15` shows the chain.

**Warning (RV16 nit 3, approved by the user on 2026-11-20).** Both plans write their report to the
same folder, `gomea-results/phase1bj/report-stage1`, so the last plan that ends decides which report
stays there.

* Start `round15` only. Never start `round15b` by itself, and never start `round15` after
  `round15b`: `round15` would then write its 5-dataset interim report over the full report, and it
  does not start `round15b` again.
* If `round15b` stops part way, its report covers the 500 runs of `round15` plus a part of the
  rest. That is neither the fallback design nor the full design. Do not use it: use the report in
  the bundle of `round15` (the D2 fallback design), and tell the agent.

**S0 and the Boolean manifests:** the 14 Boolean manifests of rounds 4, 6 and 12 name the code
commit 3c5bc87 (the harness and its stop rule), so the 4 pin tests (`test_plan4.py`,
`test_plan6.py` and two in `test_plan12.py`) pass again.

**Phase 1bj: the runs.** Each job is one stand-alone GOMEA population on a fixed genome
(`python -m pymoses.gomea.standalone`, started by the runner arm key `"standalone"`;
`docs/gomea/IT-LINKAGE-STUDY-PLAN.md`, sections 5 and 6), with N = 1,000 and E = 500,000
evaluations. A score-cache hit is not an evaluation. A run ends at the first of: E evaluations
(end reason `global_budget`); all genomes equal (`converged`); 10 generations in a row with no new
evaluation (`stalled`, `--stall-generations 10`); 300 generations (`max_generations`,
`--max-generations 300`). There is no time limit; the wall cap of 36,000 s per run is only a guard
against a hung process. The seed of a run is its split number.

**Phase 1bj: arms** (`configs/gomea/phase1bj/sr.json`, one manifest for the 10 datasets):

| Arm | Genome | Linkage model (`--fos`) |
|---|---|---|
| T-UNI | template: full binary tree of height 4, 31 positions, `+ - * aq`, no constants, a + b·f | `uni` |
| T-RT | template | `rt` |
| T-LT | template | `lt-mi` |
| T-LTB | template | `lt-b` |
| T-NP | template | `np` (node proximity on the template tree) |
| IT-UNI | IT: 8 terms x 2 factors x (input box, power box), 32 positions, powers -2, -1, 1, 2, per-term least squares | `uni` |
| IT-RT | IT | `rt` |
| IT-LT | IT | `lt-mi` |
| IT-LTB | IT | `lt-b` |
| IT-NP | IT | `np` (proximity on the IT tree: box, factor, term) |

* The arm block gives `genome`, `fos`, `pop` 1000, `max_evals` 500000, `max_generations` 300,
  `stall_generations` 10 and the genome sizes (`template_height` 4; `it_terms` 8, `it_factors` 2,
  `it_powers` [-2, -1, 1, 2]). The IT column cache is on (the default).
* The summary is the harness's `summary.json`: status, training, validation, test and clipped
  test NMSE of the reported model (the best training score of the final population), size, nodes,
  `n_terms` (IT), `n_evals`, `cpu_s`, `ru_maxrss_kb`, `generations`, `termination_reason`,
  `cache_hits`, `anytime` (the best training NMSE at 1, 2, 5, 10, 20, 50 and 100 % of E),
  `crosscheck` (the `sr_eval` cross-check; a failed cross-check fails the run) and a `standalone`
  block. `events.jsonl` has one record per generation (the linkage telemetry of the plan, section
  6.3) and an "end" record with the best-so-far trace.
* Datasets: all 10, splits 71-80. 1,000 jobs (100 per arm, 100 per split, 100 per dataset): 500 in
  round15 and 500 in round15b. Each phase step needs its datasets with splits 71-80 (requirement
  `datasets:...:71-80`). Without them, the step is skipped with the reason in the log.

**Phase 1bj: report** (`spec.json` of 1bj; the round 15 spec keys of `tools/gomea/linkage_report.py`;
a failed run counts as infinitely bad; Holm, alpha = 0.025, and the 10 % band in each family):

* The metric of every dataset is the training NMSE (`train_nmse`, primary). The other metrics are
  `test_nmse` and `anytime_area` (exp of the trapezoid area under ln(best training NMSE) over
  log10 of the evaluation fraction, divided by its width of 2 decades).
* **P1**, primary, a Holm family of its own (4): the interaction contrasts, for f in UNI, LT,
  LT-B and NP, d = log(IT-f / IT-RT) - log(T-f / T-RT) per (dataset, split) pair, paired Wilcoxon
  on d, ratio = exp(median d). The verdict "A1 gains more" means that f gains more over RT on IT
  than on the template; "A2 gains more" means the opposite; else "tie".
* **P2**, primary, a Holm family of its own (5): IT-UNI, IT-LT, IT-LTB and IT-NP each against IT-RT;
  IT-NP against IT-LTB (the pooled log-ratio test of round 8; "A better", "A worse", "tie").
* **S1**, secondary (5): the same five contrasts on the template.
* **S2**, secondary (2), test NMSE: IT-LTB against T-LTB, IT-NP against T-NP.
* **S3**, descriptive (18; p and Holm p, the verdict "descriptive"): P1 and P2 on the test NMSE and
  on the anytime area.
* **D**, descriptive (90): P1 and P2 on the training NMSE of each dataset.
* `report-stage1/evals_to_reference.csv`: for each dataset and arm, the evaluations to reach the
  median final training NMSE of RT on the same genome and dataset (descriptive).
* `report-stage1/runs.csv` and `nmse_above_100.csv` (`"run_table": true`); the size of an IT model
  is its node count plus its number of terms (`"size_terms": true`).
* round15 writes the same report on its 5 datasets (`configs/gomea/phase1bj-a/spec.json`: the full
  spec with the metrics of those datasets). Both plans write into `../gomea-results/phase1bj/`, so
  the report of round15b replaces it with the report on all 10 datasets; the bundle of round15
  keeps the interim report.

**Memory and workers:**

| Manifest | `mem_limit_mb` | W on the server (cgroup 65,536 MB, 12 CPUs, reserve 6,553 MB) |
|---|---|---|
| `phase1bj/sr.json` (10 datasets) | 2,048 | min(11, floor(58,983 / 2,048) = 28) = 11 (the CPUs) |

Both 1bj steps have `per_run_mb` 2048. The limit is RLIMIT_AS; a run's virtual size is its
resident size plus about 11 MB. The pilot's highest peak was 371 MB (IT on Dow, the IT column cache;
18 % of the limit); the template runs peaked at 108 to 128 MB.

**Size and time (the local pilot, `IT-LINKAGE-STUDY-PLAN.md` section 9; 2026-11-20).** All 10 arms
on Con and Dow, split 71, N = 1,000, E = 500,000, through the runner, at nice 19 on the shared local
machine (load 14 to 45 on 22 logical CPUs, so the CPU times can be high). CPU per evaluation:
template 1.3 to 1.9 ms (Con 516 and Dow 534 training rows), IT 0.24 to 0.29 ms on Con and 0.51 to
0.75 ms on Dow (57 inputs). The template is 3.5 to 4 times the plan's benchmark; the IT column
cache makes IT 3 to 10 times faster than the plan's estimate.

**The stop rule (after the pilot).** In the first pilot, 3 of the 20 runs did not end. In Con T-LT
and Con IT-LT the evaluation count stopped growing after generation 21 and 19, and the generations
went on (to generation 393 and 343, where the agent stopped them) with no new evaluation: the
population has one phenotype, but accepted
neutral changes of inactive positions keep the genotypes different, so "all genomes equal" does
not happen, and a score-cache hit does not count against E. Dow T-NP added only about 27
evaluations per generation for 800 generations. The harness option `--stall-generations K` (end
reason `stalled`) and the generation limit `--max-generations G` solve this; 1bj uses K = 10 and
G = 300. The 3 runs again, with the rule:

| Run | End reason | Evaluations | Generations | CPU s | Cross-check | Peak MB |
|---|---|---|---|---|---|---|
| Con T-LT | `stalled` | 266,173 | 31 | 354 | passed | 77 |
| Con IT-LT | `stalled` | 249,829 | 29 | 71 | passed | 80 |
| Dow T-NP | `max_generations` | 345,971 | 300 | 535 | passed | 91 |

The other 17 pilot runs had at most 70 generations and at most 2 generations in a row with no new
evaluation, so the rule does not change them (15 ended at E, Dow T-LT and Dow IT-LT by
convergence). All 20 runs passed the cross-check (gate 3).

**The estimate.** The CPU of each pilot run with the stop rule, per training row, the mean of Con
and Dow for each arm, scaled by the training rows of each dataset:

| Design | CPU hours (this machine) | at W = 11 | server 2x slower: CPU hours | at W = 11 |
|---|---|---|---|---|
| 10 datasets | about 226 (template 174, IT 52) | about 20.5 h | about 451 | about 41 h |
| round15: Air, Con, Dow, EnH, Tow (the fallback set) | about 121 | about 11 h | about 242 | about 22 h |
| round15b: Bos, EnC, WiR, WiW, Yac | about 104 | about 9.5 h | about 209 | about 19 h |

S0 adds about 1.3 hours to round15 (about 2.6 hours on a server 2x slower). Round 14's S0 pytest
took 1,514 s on the server for 10,217 tests and the local full suite takes 20 to 35 minutes, so
the server is probably about as fast as this machine for one thread. With the higher of the two
pilot datasets for each arm, the 10 datasets need about 271 CPU hours (about 25 h at W = 11). The
stop rule saves CPU only where a run stops early; on a dataset where LT does not stall the run
uses all E.

**Local rehearsal only:** `tools/server/plan15-smoke.json` (`--plan smoke15`) runs a limited S0
(`tests/server`, the split test, the tests of the round 15 report and of the stand-alone harness;
the mux11 repro run), then `configs/gomea/server-smoke15-1bj-a` (the same manifest, arms and spec on
the first half of the datasets, splits 71 and 72, E = 3,000, a wall cap of 300 s; 100 jobs), then
chains to `plan15b-smoke.json` (`smoke15b`: `configs/gomea/server-smoke15-1bj`, the 10 datasets;
100 jobs kept, 100 new; the report on 200). Its state is in `server-run15-smoke/` and
`server-run15b-smoke/`. The smoke plans are for a local check before a push, not a server step.

## Round 16

The sixteenth round (`SERVER-ROUND16-PREREGISTRATION.md`, the budget round of T3.2: at 100,000
evaluations, is real-valued GOMEA on the constants better than BASE-TLS; does the ratio of an RV
arm to BASE-TLS fall when the budget grows from 20,000 to 100,000; how much does BASE-TLS itself
gain from 5 times the budget) is one phase, 1bk (regression), in two plans that chain. The local
pilot estimate is about 39 h, more than 20 h for one plan, so the round has two plans (the user's
rule; see "Size and time"). The two plans split the round by splits:

* `tools/server/plan16.json` (`round16`): S0, then 1bk on splits 81-85
  (`configs/gomea/phase1bk-a`: the spec and the three manifests of 1bk with splits 81-85 only;
  210 runs; an interim report). Its state is in `../gomea-results/server-run16/`. It has
  `"then": "round16b"`.
* `tools/server/plan16b.json` (`round16b`): 1bk with the full config (`configs/gomea/phase1bk`,
  splits 81-90). It starts by itself when round16 ends. The manifests of the two config folders
  have the same names, so the runs of round16 are in the same result directories
  (`../gomea-results/phase1bk/`), and the runner keeps them. round16b runs the other 210 runs
  (splits 86-90), and its report covers all 420 runs. It has no S0 (round16 ran S0 on the same
  commit). Its state is in `../gomea-results/server-run16b/`.

There is no Boolean phase.

**One-time server steps before the round:**

1. Make sure that no other round runs (`bash tools/server/status.sh --plan round15b` shows
   "not running"). Round 16 never overlaps with another round.
2. Update the clone to the branch commit that has this section:

       git pull --ff-only

3. Run the set-up again:

       bash tools/server/setup.sh

   It must end with `SETUP OK`. Round 16 uses the fresh splits 81-90. A dataset set from an
   earlier round has splits 1-80 only, so set-up rebuilds it (splits 1-80 do not change; Dow and
   Tow are built again from their raw files). Each dataset then has 720 split files (90 splits x
   8 files).
4. Start the round. One command starts the chain; round16b starts by itself when round16 ends:

       bash tools/server/start.sh --plan round16

That is all. Do not run a separate smoke test on the server: the round runs everything in one
go. S0 (the full pytest and the reproducibility check) records a failure in the log and in
`round-summary.md`, but it does not stop the phase. There is no choice step, no extension and no
adoption rule: the round only reports the results (the proposals after the round go to the
user). Each plan writes its own `round-summary.md` and final bundle (as for the chain of rounds
4 and 5). Bring back both, from `server-run16/` and `server-run16b/`. The report on all 420 runs
is in the bundle of round16b.

**Following and stopping the chain** (the chain rules of "Round 5 and the chain from round 4"):

* `bash tools/server/status.sh --plan round16` shows the round. Its `chain:` section gives the
  state of round16 and of round16b (not started, RUNNING with the pid and the process group,
  stopped, or DONE) and where the bundle of each plan is. `status.sh --plan round16b` shows the
  same chain.
* `tail -f ../gomea-results/server-run16/run.log` follows round16. When round16 ends, its log
  shows the `chain: started round16b` line; then follow `server-run16b/run.log`.
* `bash tools/server/stop.sh --plan round16` stops the running orchestrator of the chain,
  whichever plan it runs (`--plan round16b` does the same). It writes `server-run16/chain.stop`,
  so a stopped round16 does not start round16b.
* `bash tools/server/start.sh --plan round16` after a stop resumes the chain: it starts or
  resumes the first plan that is not DONE (round16, else round16b), and it removes the stop
  marker. When both plans are DONE, it only tells where the two bundles are.
* Never two at once: each orchestrator of the chain holds `server-run16/chain.lock`, whichever
  plan it runs. A second `start.sh` refuses.
* Autostart: run `uninstall_autostart.sh`, then `install_autostart.sh --plan round16`. The block
  does nothing only when both plans are DONE.

**Warning (as in round 15).** Both plans write their report to the same folder,
`gomea-results/phase1bk/report-stage1`, so the last plan that ends decides which report stays
there.

* Start `round16` only. Never start `round16b` by itself, and never start `round16` after
  `round16b`: `round16` would then write its interim report (splits 81-85) over the full report,
  and it does not start `round16b` again.
* If `round16b` stops part way, the report folder does not have the report on all 420 runs. Do
  not use that folder: use the report in the bundle of `round16` (the interim report on splits
  81-85), and tell the agent.

**S0 and the Boolean manifests:** before the push, the 14 Boolean manifests of rounds 4, 6 and 12
must name the newest code commit (the RV option of T3.2), so that the 4 pin tests
(`test_plan4.py`, `test_plan6.py` and two in `test_plan12.py`) pass.

**Phase 1bk: budget.** Each run gets 20,000 or 100,000 evaluations (`"budget": {"evals": N}`, so
`-m N`) and no `--max-time`. Every RV evaluation counts in the budget. The wall cap of 36,000 s
per run is only a guard against a hung process.

**Phase 1bk: arms** (`configs/gomea/phase1bk/`, three manifests: `sr.json` has the three 20,000
arms and `sr-100K.json` the three 100,000 arms, both on Air, Con, EnC, EnH, Tow and Yac;
`sr-Dow.json` has all six arms on Dow):

| Arm | Budget | Flags |
|---|---|---|
| BASE-TLS | 20,000 | the round 14 BASE-TLS (as in 1bi) + `--gomea-trace 1` |
| BASE-TLS-100K | 100,000 | the same |
| TLS-RV | 20,000 | BASE-TLS + `--gomea-rv-step rvg --gomea-rv-budget balance` (before `--gomea-trace 1`) |
| TLS-RV-100K | 100,000 | the same |
| TLS-RVL | 20,000 | BASE-TLS + `--gomea-rv-step rvg --gomea-rv-budget late` (before `--gomea-trace 1`) |
| TLS-RVL-100K | 100,000 | the same |

* `--gomea-trace 1` (the RV16 review, finding 1, option a; the user's decision) is on every arm
  and changes only the GOMEA log: the "end" event of each deme gets the exact best-so-far trace
  of the training score, and the "init" event the SHA-256 of the initial genotypes. The search,
  stdout and the MOSES log do not change (`tests/gomea/test_rv_step_identity.py`).

* The RV option is opt-in (`docs/gomea/USAGE.md`). Without it, the output of every existing
  setting does not change (the identity tests in `tests/gomea/`). With it, the RV stream gets its
  seed from the run seed and the deme index, so everything up to the first RV generation is the
  same as with the option off (the seed fix).
* No arm has LM or the guard, so the runner gives no `--const-opt-val`. All six arms get
  `--gomea-log RUN_DIR/events.jsonl`.
* The manifests have `"clip_test_nmse"`, `"plain_nmse"` and (new in round 16) `"anytime"` true.
  So each summary has `test_nmse_clipped`, `plain_val_nmse` and `plain_test_nmse`, and from the
  GOMEA log: `demes` (the demes, the run evaluation at which each starts), `anytime` (the
  best-so-far training NMSE at 1, 2, 5, 10, 20, 50 and 100 % of the budget, exact, from the trace
  of `--gomea-trace 1`: `anytime_source` "trace"; null only before the first evaluation),
  `anytime_check` (the conversion of the raw score to the training NMSE, checked against the
  first printed program, and the checks of the trace against the generation records) and
  `init_genotypes_sha256` (the digest of the first deme's initial genotypes: an RV arm and
  BASE-TLS of the same split must have the same value, gate item 2 of the pilot). The four RV arms also have `rv`
  (the RV totals: evaluations and share, RV generations, improving, kept by p_accept, AMS, forced
  improvements, skips).
* `sr-100K.json` has the name of `sr.json` (`phase1bk-sr`). So the 100,000 runs are in
  `phase1bk/phase1bk-sr/runs/`, beside the 20,000 runs, as before the move. The runner writes
  `phase1bk/phase1bk-sr/manifest.json` at the start of each manifest, so that file holds the
  manifest that ran last (`sr-100K.json`). The git commit in the bundle gives both manifests.
* Datasets: Air, Con, Dow, EnC, EnH, Tow, Yac (7), splits 81-90. 420 MOSES runs (70 per arm, 42
  per split, 60 per dataset): 210 in round16 (splits 81-85) and 210 in round16b (splits 86-90).
  Each phase step needs the 7 datasets with splits 81-90 (requirement
  `datasets:Air,Con,Dow,EnC,EnH,Tow,Yac:81-90`; set-up builds all 90 splits at once). Without
  them, the step is skipped with the reason in the log.

**Phase 1bk: report** (`spec.json` of 1bk; the round 15 key `"contrast_families"`; a failed run
counts as infinitely bad; paired Wilcoxon on log(A / B) over the 70 pairs, Holm in each family,
alpha = 0.025, and the 10 % band):

* **P**, primary (4), test NMSE: TLS-RV-100K vs BASE-TLS-100K; TLS-RVL-100K vs BASE-TLS-100K; the
  interaction contrasts [TLS-RV-100K, BASE-TLS-100K, TLS-RV, BASE-TLS] and [TLS-RVL-100K,
  BASE-TLS-100K, TLS-RVL, BASE-TLS] (d = log(A1 / B1) - log(A2 / B2); "A1 gains more" means that
  the ratio of the RV arm to BASE-TLS is at least 10 % lower at 100,000 than at 20,000).
* **S**, secondary (8): TLS-RV and TLS-RVL vs BASE-TLS; BASE-TLS-100K vs BASE-TLS; TLS-RV-100K vs
  TLS-RV (test NMSE); the 4 primary contrasts on the training NMSE.
* **SC**, secondary (8): program size and CPU for primary 1 and 2 and for the two 20,000
  contrasts (`"size_terms": true`).
* **D**, descriptive (28): the 4 primary contrasts on each dataset. **DP**, descriptive (3):
  primary 1 and 2 on the clipped test NMSE; BASE-TLS-100K vs BASE-TLS on the training NMSE.
* The descriptive tables (round 16 keys of `tools/gomea/budget_report.py`):
  `report-stage1/anytime_ratios.csv` (the median ratio of each arm's best-so-far training NMSE to
  BASE-TLS of the same budget at each checkpoint, per dataset and pooled), `demes.csv` (demes per
  run, the evaluation at which the second deme starts) and `rv.csv` (the RV counters).
* `report-stage1/runs.csv` and `nmse_above_100.csv` (`"run_table": true`); `aggregate.csv` has
  the median train-test gap.
* The report reads the result directories of all three manifests (`phase_run.sh` gives them) and
  pairs the runs by dataset and split. So an interaction contrast pairs a 100,000 run of
  `sr-100K.json` with a 20,000 run of `sr.json` (`tests/gomea/test_round16_report.py`).
* round16 writes the same report on splits 81-85 (the spec of `configs/gomea/phase1bk-a` is the
  spec of 1bk): 35 pairs in each pooled contrast and 5 for each dataset. round16b writes the
  report on all 420 runs into the same folder; the bundle of round16 keeps the interim report.

**Memory and workers:**

| Manifest | `mem_limit_mb` | W on the server (cgroup 65,536 MB, 12 CPUs, reserve 6,553 MB) |
|---|---|---|
| `phase1bk/sr.json` (the 20,000 arms, 6 datasets) | 5,120 | min(11, 11) = 11 |
| `phase1bk/sr-100K.json` (the 100,000 arms, 6 datasets) | 8,192 | floor(58,983 / 8,192) = 7 |
| `phase1bk/sr-Dow.json` (all 6 arms on Dow) | 8,192 | floor(58,983 / 8,192) = 7 |

Both 1bk steps (round16 and round16b) have `per_run_mb` 5120 and `manifest_per_run_mb`
`{"sr-100K.json": 8192, "sr-Dow.json": 8192}`. `phase_run.sh` runs the manifests one at a time:
`sr.json`, then `sr-100K.json`, then `sr-Dow.json`. The limit is RLIMIT_AS.

* The local pilot (gate item 3): the Tow BASE-TLS-100K run peaked at 3,316 MB RSS, 64.8 % of
  5,120 MB. So the 100,000 arms have a manifest of their own with 8,192 MB (the Tow peak is then
  40.5 %).
* The runner starts the jobs of a manifest in its order (dataset, then arm, then split). Tow is
  first in `sr-100K.json`, so the Tow 100,000 runs (the longest, about 3 to 5 h each) start
  first.

**Size and time.** The pre-registration's estimate: BASE-TLS about 13 CPU-h, TLS-RV and TLS-RVL
about 8 h each, BASE-TLS-100K about 65 h, TLS-RV-100K and TLS-RVL-100K about 39 h each: about 170
CPU-h, about 16 h at W = 11, plus S0 (about 1.3 h). The local pilot gives a new estimate of about
39 h for the round, more than 20 h. So the round has two plans. Each plan has half of the splits
of each arm and dataset; round16 also runs S0.

**Local rehearsal only:** `tools/server/plan16-smoke.json` (`--plan smoke16`) runs a limited S0
(`tests/server`, the split test, the tests of the round 13 to 16 runner and report, the
per-term `sr_eval` test and the stand-alone runner test; the mux11 repro run), then
`configs/gomea/server-smoke16-1bk-a` (the same three manifests, arms, flags, keys and spec,
split 81 only, 1,000 evaluations for the 20,000 arms and 2,000 for the 100,000 arms, a wall cap
of 300 s; 42 jobs), then it chains to `plan16b-smoke.json` (`smoke16b`:
`configs/gomea/server-smoke16-1bk`, splits 81 and 82; 42 runs kept, 42 new; the report on 84).
The state is in `server-run16-smoke/` and `server-run16b-smoke/`. The smoke plans are for a
local check before a push, not a server step.
