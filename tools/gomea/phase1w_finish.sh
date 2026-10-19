#!/usr/bin/env bash
# Phase 1w deviation, step 2 (2026-10-19): all 16 slots are held by slow TOPK runs, so
# GOMEA s3009 and TOPK-L s3003 would not start for hours. When GOMEA s3003 has finished:
# pause phase_run.sh (so the Con part cannot start), interrupt the mux11 runner, run only
# GOMEA s3009 and TOPK-L s3003, let phase1w_earlystop.py stop that runner once every pair's
# sign is fixed, then resume phase_run.sh.
#   phase1w_finish.sh PHASE_RUN_PID RUNNER_PID
set -u
cd "$(dirname "$0")/../.."
PR=$1; RUN=$2
PY=.venv/bin/python
M=configs/gomea/phase1w/bool.json
D=$($PY -m benchmarks.gomea.runner $M --dry-run | sed -n 's/^result directory: //p')
until [ -f "$D/runs/mux11__GOMEA__s3003/summary.json" ]; do sleep 30; done
echo "GOMEA s3003 done $(date -Is); pausing phase_run $PR, interrupting runner $RUN"
kill -STOP "$PR"
kill -INT "$RUN"
while kill -0 "$RUN" 2>/dev/null; do sleep 5; done
echo "runner stopped $(date -Is)"
$PY -m benchmarks.gomea.runner $M --only '^mux11__(GOMEA__s3009|TOPK-L__s3003)$' --workers 2 \
    --recursion-limit 100000 --mem-limit-mb 3072 >> ../gomea-results/phase1w-bench.log 2>&1 &
R2=$!
echo "rerun runner $R2 $(date -Is)"
$PY tools/gomea/phase1w_earlystop.py "$D" "$R2"
wait "$R2"
echo "decided/finished $(date -Is); resuming phase_run $PR"
kill -CONT "$PR"
