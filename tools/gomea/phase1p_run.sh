#!/usr/bin/env bash
# Phase 1p launcher (equal wall-clock time; run on an otherwise idle machine).
#   setsid nohup tools/gomea/phase1p_run.sh > ../gomea-results/phase1p.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
R=../gomea-results
PY=.venv/bin/python
dir_of() { $PY -m benchmarks.gomea.runner "$1" --dry-run | sed -n 's/^result directory: //p'; }
bench() { $PY -m benchmarks.gomea.runner "$1" --workers 10 --recursion-limit 100000 \
            --mem-limit-mb 3072 >> $R/phase1p-mux11.log 2>&1; }
sr() { $PY tools/gomea/moses_sr_runner.py "$1" $R/phase1p --workers 10 >> $R/phase1p-con.log 2>&1; }
echo "start $(date -Is)"
sr configs/gomea/phase1p/con.json
bench configs/gomea/phase1p/mux11.json
echo "stage 1 done $(date -Is)"
M1=$(dir_of configs/gomea/phase1p/mux11.json); C1=$R/phase1p/phase1p-con
$PY tools/gomea/phase1p_report.py --stage 1 $R/phase1p/report-stage1 "$M1" "$C1" -
V=$R/phase1p/report-stage1/verdict.json
ext() { $PY -c "import json,sys; v=json.load(open(sys.argv[1])); print(v.get(sys.argv[2]+':vs-hc')=='extend')" $V "$1"; }
# Stage 2 keeps every problem in the Holm family (a non-extended problem with its stage-1
# data); only contrasts that were "extend" at stage 1 take their verdict from stage 2.
M2=$M1; C2=$C1; EXT=no
if [ "$(ext Con)" = True ]; then sr configs/gomea/phase1p/con-ext.json; C2="$C1,$R/phase1p/phase1p-con-ext"; EXT=yes; fi
if [ "$(ext mux11)" = True ]; then bench configs/gomea/phase1p/mux11-ext.json; M2="$M1,$(dir_of configs/gomea/phase1p/mux11-ext.json)"; EXT=yes; fi
if [ "$EXT" = yes ]; then
    $PY tools/gomea/phase1p_report.py --stage 2 $R/phase1p/report-stage2 "$M2" "$C2" -; fi
echo "all done $(date -Is)"
