#!/usr/bin/env bash
# Run phases one at a time, in order: `queue_run.sh phase1u phase1v ...` (each via
# phase_run.sh, log ../gomea-results/<phase>.log). Experiments never overlap: before each
# phase it waits (polling every 60 s) until no other experiment runner
# (benchmarks.gomea.runner, moses_sr_runner.py) is alive. With WAIT_FOR_FILE and
# WAIT_FOR_LINE set, it first waits until that file contains that line.
#   [WORKERS=N] [WAIT_FOR_FILE=F WAIT_FOR_LINE=L] setsid nohup tools/gomea/queue_run.sh \
#       phase1u phase1v > ../gomea-results/queue.log 2>&1 &
set -u
[ $# -ge 1 ] || { echo "usage: $0 PHASE ..." >&2; exit 2; }
cd "$(dirname "$0")/../.."
R=../gomea-results
for p in "$@"; do
    [ -f "configs/gomea/$p/spec.json" ] || { echo "no configs/gomea/$p/spec.json" >&2; exit 2; }
done
if [ -n "${WAIT_FOR_FILE:-}" ] && [ -n "${WAIT_FOR_LINE:-}" ]; then
    echo "waiting for '$WAIT_FOR_LINE' in $WAIT_FOR_FILE $(date -Is)"
    until grep -qF -- "$WAIT_FOR_LINE" "$WAIT_FOR_FILE" 2>/dev/null; do sleep 60; done
fi
busy() { pgrep -f "benchmarks\.gomea\.runner|moses_sr_runner\.py" > /dev/null; }
for p in "$@"; do
    if busy; then
        echo "waiting for other experiment runners to exit $(date -Is)"
        while busy; do sleep 60; done
    fi
    echo "phase $p start $(date -Is)"
    tools/gomea/phase_run.sh "$p" > "$R/$p.log" 2>&1
    echo "phase $p end (rc=$?) $(date -Is)"
done
echo "queue done $(date -Is)"
