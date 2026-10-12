#!/usr/bin/env bash
# Phase 1j launcher: wait for the given PID (the previous experiment's runner) to exit, so
# wall-clock limits are not distorted by another experiment's load. Then:
#   stage 1: Con (splits 1-10) and mux11 (seeds 3001-3010), one after the other;
#   report stage 1; for each problem marked "extend", run its extension manifest
#   (splits 11-20 / seeds 3011-3020); report stage 2 on the extended problems.
# Usage, from the repository root:
#   setsid nohup tools/gomea/phase1j_run.sh WAIT_PID > ../gomea-results/phase1j.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
R=../gomea-results
PY=.venv/bin/python
WAIT_PID=${1:-}
while [ -n "$WAIT_PID" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "start $(date -Is)"

run_con() {  # $1 = manifest
    $PY tools/gomea/moses_sr_runner.py "$1" $R/phase1j --workers 10 >> $R/phase1j-con.log 2>&1
}
run_mux() {
    $PY -m benchmarks.gomea.runner "$1" --workers 10 --recursion-limit 100000 \
        --mem-limit-mb 3072 >> $R/phase1j-mux11.log 2>&1
}
mux_dir() {  # result directory the runner uses for a manifest
    $PY -m benchmarks.gomea.runner "$1" --dry-run | sed -n 's/^result directory: //p'
}

run_con configs/gomea/phase1j/con.json;   echo "con stage 1 done $(date -Is)"
run_mux configs/gomea/phase1j/mux11.json; echo "mux11 stage 1 done $(date -Is)"
CON1=$R/phase1j/phase1j-con
MUX1=$(mux_dir configs/gomea/phase1j/mux11.json)
$PY tools/gomea/phase1j_report.py --stage 1 $R/phase1j/report-stage1 "$MUX1" "$CON1"

verdict() { $PY -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" \
    $R/phase1j/report-stage1/verdict.json "$1"; }
MUX2=-; CON2=-
if [ "$(verdict Con)" = extend ]; then
    run_con configs/gomea/phase1j/con-ext.json; echo "con stage 2 done $(date -Is)"
    CON2="$CON1,$R/phase1j/phase1j-con-ext"
fi
if [ "$(verdict mux11)" = extend ]; then
    run_mux configs/gomea/phase1j/mux11-ext.json; echo "mux11 stage 2 done $(date -Is)"
    MUX2="$MUX1,$(mux_dir configs/gomea/phase1j/mux11-ext.json)"
fi
if [ "$MUX2" != - ] || [ "$CON2" != - ]; then
    $PY tools/gomea/phase1j_report.py --stage 2 $R/phase1j/report-stage2 "$MUX2" "$CON2"
fi
echo "all done $(date -Is)"
