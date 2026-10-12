#!/usr/bin/env bash
# Phase 1j launcher: wait for the given PID (the previous experiment's runner) to exit,
# then run Con and mux11 one after the other, so wall-clock limits are not distorted by
# another experiment's load. Usage, from the repository root:
#   setsid nohup tools/gomea/phase1j_run.sh WAIT_PID > ../gomea-results/phase1j.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
WAIT_PID=${1:-}
while [ -n "$WAIT_PID" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "start $(date -Is)"
.venv/bin/python tools/gomea/moses_sr_runner.py configs/gomea/phase1j/con.json \
    ../gomea-results/phase1j --workers 10 > ../gomea-results/phase1j-con.log 2>&1
echo "con done $(date -Is) exit $?"
.venv/bin/python -m benchmarks.gomea.runner configs/gomea/phase1j/mux11.json --workers 10 \
    --recursion-limit 100000 --mem-limit-mb 3072 > ../gomea-results/phase1j-mux11.log 2>&1
echo "mux11 done $(date -Is) exit $?"
