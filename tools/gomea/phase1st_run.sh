#!/usr/bin/env bash
# Phase 1s/1t launcher: runs Phase 1s (PEN vs LTB-C250-M) to completion and reports it,
# THEN Phase 1t (HC-PEN vs HC-C250) and reports it; strictly sequential, never
# overlapping. mux11, 2 arms each, seeds 3021-3030, -m 40000, cap 250; Phase 1s pools
# Phase 1o's mux11 runs (seeds 3001-3010).
#   setsid nohup tools/gomea/phase1st_run.sh > ../gomea-results/phase1st.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
R=../gomea-results
PY=.venv/bin/python
S=configs/gomea/phase1s/mux11.json
T=configs/gomea/phase1t/mux11.json
dir_of() { $PY -m benchmarks.gomea.runner "$1" --dry-run | sed -n 's/^result directory: //p'; }
bench() { $PY -m benchmarks.gomea.runner "$1" --workers 16 --recursion-limit 100000 \
            --mem-limit-mb 3072 >> "$2" 2>&1; }
echo "1s start $(date -Is)"
bench $S $R/phase1s-bench.log
$PY tools/gomea/phase1s_report.py $R/phase1s/report "$(dir_of $S)" \
    "$(dir_of configs/gomea/phase1o/mux11.json)"
echo "1s done $(date -Is)"
echo "1t start $(date -Is)"
bench $T $R/phase1t-bench.log
$PY tools/gomea/phase1t_report.py $R/phase1t/report "$(dir_of $T)"
echo "1t done $(date -Is)"
echo "all done $(date -Is)"
