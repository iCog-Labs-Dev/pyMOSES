#!/usr/bin/env bash
# Phase 1n launcher: wait for WAIT_PID (the Phase 1l/1m launcher), then run mux11 and Con
# (capped HC vs capped GOMEA); report stage 1; extend each borderline problem; report stage 2.
#   setsid nohup tools/gomea/phase1n_run.sh WAIT_PID > ../gomea-results/phase1n.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
R=../gomea-results
PY=.venv/bin/python
WAIT_PID=${1:-}
while [ -n "$WAIT_PID" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
dir_of() { $PY -m benchmarks.gomea.runner "$1" --dry-run | sed -n 's/^result directory: //p'; }
bench() { $PY -m benchmarks.gomea.runner "$1" --workers 10 --recursion-limit 100000 \
            --mem-limit-mb 3072 >> $R/phase1n-mux11.log 2>&1; }
sr() { $PY tools/gomea/moses_sr_runner.py "$1" $R/phase1n --workers 10 >> $R/phase1n-con.log 2>&1; }
echo "start $(date -Is)"
sr configs/gomea/phase1n/con.json
bench configs/gomea/phase1n/mux11.json
echo "stage 1 done $(date -Is)"
M1=$(dir_of configs/gomea/phase1n/mux11.json); C1=$R/phase1n/phase1n-con
$PY tools/gomea/phase1n_report.py --stage 1 $R/phase1n/report-stage1 "$M1" "$C1"
V=$R/phase1n/report-stage1/verdict.json
M2=-; C2=-
if [ "$($PY -c "import json,sys;print(json.load(open(sys.argv[1])).get('Con'))" $V)" = extend ]; then
    sr configs/gomea/phase1n/con-ext.json; C2="$C1,$R/phase1n/phase1n-con-ext"; fi
if [ "$($PY -c "import json,sys;print(json.load(open(sys.argv[1])).get('mux11'))" $V)" = extend ]; then
    bench configs/gomea/phase1n/mux11-ext.json; M2="$M1,$(dir_of configs/gomea/phase1n/mux11-ext.json)"; fi
if [ "$M2" != - ] || [ "$C2" != - ]; then
    $PY tools/gomea/phase1n_report.py --stage 2 $R/phase1n/report-stage2 "$M2" "$C2"; fi
echo "all done $(date -Is)"
