#!/usr/bin/env bash
# Phases 1l and 1m, one after the other (1l is wall-clock based and needs an idle machine).
#   1l: equal-time mux11 (HC vs tuned GOMEA) on the sped-up code; 10 seeds, +10 if borderline.
#   1m: budget-derived demes; regression (Con, EnH, EnC) then mux11; 10 seeds/splits, +10 for
#       a family whose primary contrasts include "extend".
# Usage, from the repository root:
#   setsid nohup tools/gomea/phase1lm_run.sh > ../gomea-results/phase1lm.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
R=../gomea-results
PY=.venv/bin/python
dir_of() { $PY -m benchmarks.gomea.runner "$1" --dry-run | sed -n 's/^result directory: //p'; }
bench() { $PY -m benchmarks.gomea.runner "$1" --workers "$2" --recursion-limit 100000 \
            --mem-limit-mb 3072 >> "$3" 2>&1; }
sr() { $PY tools/gomea/moses_sr_runner.py "$1" $R/phase1m --workers 12 >> $R/phase1m-sr.log 2>&1; }

echo "1l start $(date -Is)"
bench configs/gomea/phase1l/mux11.json 10 $R/phase1l-mux11.log
L1=$(dir_of configs/gomea/phase1l/mux11.json)
$PY tools/gomea/phase1j_report.py --stage 1 $R/phase1l/report-stage1 "$L1" -
if grep -q '"extend"' $R/phase1l/report-stage1/verdict.json; then
    bench configs/gomea/phase1l/mux11-ext.json 10 $R/phase1l-mux11.log
    $PY tools/gomea/phase1j_report.py --stage 2 $R/phase1l/report-stage2 \
        "$L1,$(dir_of configs/gomea/phase1l/mux11-ext.json)" -
fi
echo "1l done $(date -Is)"

echo "1m start $(date -Is)"
sr configs/gomea/phase1m/sr.json
echo "1m sr stage 1 done $(date -Is)"
bench configs/gomea/phase1m/mux11.json 12 $R/phase1m-mux11.log
echo "1m mux11 stage 1 done $(date -Is)"
M1=$(dir_of configs/gomea/phase1m/mux11.json)
S1=$R/phase1m/phase1m-sr
$PY tools/gomea/phase1m_report.py --stage 1 $R/phase1m/report-stage1 "$M1" "$S1"
V=$R/phase1m/report-stage1/verdict.json
EXT_SR=$($PY -c "import json,sys; v=json.load(open(sys.argv[1])); print(any(x=='extend' for k,x in v.items() if not k.startswith('mux11')))" $V)
EXT_MUX=$($PY -c "import json,sys; v=json.load(open(sys.argv[1])); print(any(x=='extend' for k,x in v.items() if k.startswith('mux11')))" $V)
M2=-; S2=-
if [ "$EXT_SR" = True ]; then sr configs/gomea/phase1m/sr-ext.json; S2="$S1,$R/phase1m/phase1m-sr-ext"; fi
if [ "$EXT_MUX" = True ]; then
    bench configs/gomea/phase1m/mux11-ext.json 12 $R/phase1m-mux11.log
    M2="$M1,$(dir_of configs/gomea/phase1m/mux11-ext.json)"
fi
if [ "$M2" != - ] || [ "$S2" != - ]; then
    $PY tools/gomea/phase1m_report.py --stage 2 $R/phase1m/report-stage2 "$M2" "$S2"
fi
echo "all done $(date -Is)"
