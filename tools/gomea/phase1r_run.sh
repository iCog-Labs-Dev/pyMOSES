#!/usr/bin/env bash
# Phase 1r launcher: waits until Phase 1q has logged "all done", then runs Con and EnH
# (4 arms), reports stage 1, extends a problem with an "extend" primary contrast, and
# reports stage 2 over the full family.
#   setsid nohup tools/gomea/phase1r_run.sh > ../gomea-results/phase1r.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
R=../gomea-results
PY=.venv/bin/python
Q=configs/gomea/phase1r
until grep -q "all done" $R/phase1q.log 2>/dev/null; do sleep 60; done
sr() { $PY tools/gomea/moses_sr_runner.py "$1" $R/phase1r --workers 18 >> $R/phase1r-sr.log 2>&1; }
echo "start $(date -Is)"
sr $Q/sr.json
echo "stage 1 done $(date -Is)"
D1=$R/phase1r/phase1r-sr
$PY tools/gomea/phase1r_report.py --stage 1 $R/phase1r/report-stage1 "$D1"
V=$R/phase1r/report-stage1/verdict.json
ext() { $PY -c "import json,sys; v=json.load(open(sys.argv[1])); print(any(x=='extend' for k,x in v.items() if k.startswith(sys.argv[2]+':')))" $V "$1"; }
D2=$D1; EXT=no
for t in Con EnH; do
    if [ "$(ext $t)" = True ]; then sr $Q/sr-ext-$t.json; D2="$D2,$R/phase1r/phase1r-sr-ext-$t"; EXT=yes; fi
done
if [ "$EXT" = yes ]; then
    $PY tools/gomea/phase1r_report.py --stage 2 $R/phase1r/report-stage2 "$D2"; fi
echo "all done $(date -Is)"
