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

`bash tools/server/start.sh --plan smoke` runs a few-minute version (limited pytest, the
mux11 reproducibility run, one tiny phase in `configs/gomea/server-smoke/`) with its own
state in `../gomea-results/server-run-smoke/`, so it never marks real steps done.
`status.sh --plan smoke` and `stop.sh --plan smoke` work the same way. The mux11 run alone
takes about 20 minutes.
