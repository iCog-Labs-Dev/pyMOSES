#!/usr/bin/env bash
# Phase 1k launcher: wait for the given PID (Phase 1i's launcher) to exit, then run stage 1;
# report; if a primary contrast says "extend", run the extension and report stage 2.
# Usage, from the repository root:
#   setsid nohup tools/gomea/phase1k_run.sh WAIT_PID > ../gomea-results/phase1k.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
R=../gomea-results
PY=.venv/bin/python
STEP1=$R/1/phase1h-step1-2e6f467ecb9c
WAIT_PID=${1:-}
while [ -n "$WAIT_PID" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
dir_of() { $PY -m benchmarks.gomea.runner "$1" --dry-run | sed -n 's/^result directory: //p'; }
run() { $PY -m benchmarks.gomea.runner "$1" --workers 14 --mem-limit-mb 2048 >> "$2" 2>&1; }
echo "start $(date -Is)"
run configs/gomea/phase1k/stage1.json $R/phase1k-stage1.log
echo "stage 1 done $(date -Is)"
S1=$(dir_of configs/gomea/phase1k/stage1.json)
$PY tools/gomea/phase1k_report.py --stage 1 $R/phase1k/report-stage1 "$S1" "$STEP1"
if grep -q '"extend"' $R/phase1k/report-stage1/verdict.json; then
    run configs/gomea/phase1k/ext.json $R/phase1k-ext.log
    echo "extension done $(date -Is)"
    $PY tools/gomea/phase1k_report.py --stage 2 $R/phase1k/report-stage2 "$S1" "$STEP1" \
        "$(dir_of configs/gomea/phase1k/ext.json)"
fi
echo "all done $(date -Is)"
