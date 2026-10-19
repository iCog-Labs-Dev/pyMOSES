#!/usr/bin/env bash
# Phase 1w deviation, step 3 (replaces phase1w_finish.sh's tail, whose wait loop could not
# see the interrupted runner exit: a paused parent leaves it a zombie). Run only GOMEA s3009
# and TOPK-L s3003 until phase1w_earlystop.py has decided every pair, then resume
# phase_run.sh (which reaps the old runner and goes on to the Con part).
#   phase1w_finish2.sh PHASE_RUN_PID
set -u
cd "$(dirname "$0")/../.."
PR=$1
PY=.venv/bin/python
M=configs/gomea/phase1w/bool.json
D=$($PY -m benchmarks.gomea.runner $M --dry-run | sed -n 's/^result directory: //p')
$PY -m benchmarks.gomea.runner $M --only '^mux11__(GOMEA__s3009|TOPK-L__s3003)$' --workers 2 \
    --recursion-limit 100000 --mem-limit-mb 3072 >> ../gomea-results/phase1w-bench.log 2>&1 &
R2=$!
echo "rerun runner $R2 $(date -Is)"
$PY tools/gomea/phase1w_earlystop.py "$D" "$R2"
wait "$R2"
echo "decided/finished $(date -Is); resuming phase_run $PR"
kill -CONT "$PR"
