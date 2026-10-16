#!/usr/bin/env bash
# Phase 1o launcher: Con then mux11 (3 GOMEA arms; capped HC reused from Phase 1n), report
# stage 1, extend a problem whose primary contrasts include "extend", report stage 2.
#   setsid nohup tools/gomea/phase1o_run.sh > ../gomea-results/phase1o.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
R=../gomea-results
PY=.venv/bin/python
dir_of() { $PY -m benchmarks.gomea.runner "$1" --dry-run | sed -n 's/^result directory: //p'; }
bench() { $PY -m benchmarks.gomea.runner "$1" --workers 10 --recursion-limit 100000 \
            --mem-limit-mb 3072 >> $R/phase1o-mux11.log 2>&1; }
sr() { $PY tools/gomea/moses_sr_runner.py "$1" $R/phase1o --workers 10 >> $R/phase1o-con.log 2>&1; }
HC="$(dir_of configs/gomea/phase1n/mux11.json),$(dir_of configs/gomea/phase1n/mux11-ext.json),$R/phase1n/phase1n-con"
echo "start $(date -Is)"
sr configs/gomea/phase1o/con.json
bench configs/gomea/phase1o/mux11.json
echo "stage 1 done $(date -Is)"
M1=$(dir_of configs/gomea/phase1o/mux11.json); C1=$R/phase1o/phase1o-con
$PY tools/gomea/phase1o_report.py --stage 1 $R/phase1o/report-stage1 "$M1" "$C1" "$HC"
V=$R/phase1o/report-stage1/verdict.json
ext() { $PY -c "import json,sys; v=json.load(open(sys.argv[1])); print(any(x=='extend' for k,x in v.items() if k.startswith(sys.argv[2]+':')))" $V "$1"; }
M2=-; C2=-
if [ "$(ext Con)" = True ]; then sr configs/gomea/phase1o/con-ext.json; C2="$C1,$R/phase1o/phase1o-con-ext"; HC="$HC,$R/phase1o/phase1o-con-ext"; fi
if [ "$(ext mux11)" = True ]; then bench configs/gomea/phase1o/mux11-ext.json; M2="$M1,$(dir_of configs/gomea/phase1o/mux11-ext.json)"; fi
if [ "$M2" != - ] || [ "$C2" != - ]; then
    $PY tools/gomea/phase1o_report.py --stage 2 $R/phase1o/report-stage2 "$M2" "$C2" "$HC"; fi
echo "all done $(date -Is)"
