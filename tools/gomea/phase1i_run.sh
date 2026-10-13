#!/usr/bin/env bash
# Phase 1i launcher: the identity check (1 run) alongside stage 1 (80 runs); report stage 1;
# if a primary contrast says "extend", run the extension (seeds 4011-4020, all four arms)
# and report stage 2. Usage, from the repository root:
#   setsid nohup tools/gomea/phase1i_run.sh > ../gomea-results/phase1i.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
R=../gomea-results
PY=.venv/bin/python
STEP1=$R/1/phase1h-step1-2e6f467ecb9c
dir_of() { $PY -m benchmarks.gomea.runner "$1" --dry-run | sed -n 's/^result directory: //p'; }
run() { $PY -m benchmarks.gomea.runner "$1" --workers "$2" --mem-limit-mb 3072 >> "$3" 2>&1; }

echo "start $(date -Is)"
run configs/gomea/phase1i/identity-check.json 1 $R/phase1i-identity.log &
run configs/gomea/phase1i/stage1.json 12 $R/phase1i-stage1.log
wait
echo "stage 1 done $(date -Is)"
S1=$(dir_of configs/gomea/phase1i/stage1.json)
$PY tools/gomea/phase1i_report.py --stage 1 $R/phase1i/report-stage1 "$S1" "$STEP1"
if grep -q '"extend"' $R/phase1i/report-stage1/verdict.json; then
    run configs/gomea/phase1i/ext.json 12 $R/phase1i-ext.log
    echo "extension done $(date -Is)"
    $PY tools/gomea/phase1i_report.py --stage 2 $R/phase1i/report-stage2 "$S1" "$STEP1" \
        "$(dir_of configs/gomea/phase1i/ext.json)"
fi
echo "all done $(date -Is)"
