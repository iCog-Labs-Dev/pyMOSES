#!/usr/bin/env bash
# Phase 1q launcher: mux11 and pa5 (3 arms, -m 40000, cap 250), report stage 1, extend a
# problem whose primary contrasts include "extend", report stage 2 over the full family.
#   setsid nohup tools/gomea/phase1q_run.sh > ../gomea-results/phase1q.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
R=../gomea-results
PY=.venv/bin/python
Q=configs/gomea/phase1q
dir_of() { $PY -m benchmarks.gomea.runner "$1" --dry-run | sed -n 's/^result directory: //p'; }
bench() { $PY -m benchmarks.gomea.runner "$1" --workers 18 --recursion-limit 100000 \
            --mem-limit-mb 3072 >> $R/phase1q-bench.log 2>&1; }
echo "start $(date -Is)"
bench $Q/stage1.json
echo "stage 1 done $(date -Is)"
D1=$(dir_of $Q/stage1.json)
$PY tools/gomea/phase1q_report.py --stage 1 $R/phase1q/report-stage1 "$D1"
V=$R/phase1q/report-stage1/verdict.json
ext() { $PY -c "import json,sys; v=json.load(open(sys.argv[1])); print(any(x=='extend' for k,x in v.items() if k.startswith(sys.argv[2]+':')))" $V "$1"; }
D2=$D1; EXT=no
for t in mux11 pa5; do
    if [ "$(ext $t)" = True ]; then bench $Q/ext-$t.json; D2="$D2,$(dir_of $Q/ext-$t.json)"; EXT=yes; fi
done
if [ "$EXT" = yes ]; then
    $PY tools/gomea/phase1q_report.py --stage 2 $R/phase1q/report-stage2 "$D2"; fi
echo "all done $(date -Is)"
