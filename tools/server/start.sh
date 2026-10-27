#!/usr/bin/env bash
# Start (or resume) the server round in the background and return at once.
#   bash tools/server/start.sh [--plan round|smoke|round2|smoke2|FILE] [--force]
# --force starts even if the round is DONE (it then only rewrites the summary and bundles).
# Refuses if an orchestrator of this plan is already running. Survives closing the
# terminal (setsid + nohup), not a pod restart: after one, run this again; it resumes.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
FORCE=no; for a in "${ARGS[@]}"; do [ "$a" = --force ] && FORCE=yes; done
cd "$REPO" || exit 1
[ -x .venv/bin/python ] || { echo "no .venv: run bash tools/server/setup.sh first" >&2; exit 2; }
mkdir -p "$SD"
if [ -f "$SD/DONE" ] && [ $FORCE = no ]; then
    echo "The round is DONE ($(cat "$SD/DONE")). Bring back: $(ls -1t "$SD"/bundle-final-*.tar.gz 2>/dev/null | head -1)"
    exit 0
fi
if L=$(live_orchestrator); then
    echo "already running (pid/pgid $L); follow: tail -f $SD/run.log" >&2; exit 1
fi
if command -v flock >/dev/null && ! flock -n "$SD/orchestrator.lock" true; then
    echo "$SD/orchestrator.lock is held by another process; not starting" >&2; exit 1
fi
[ $FORCE = yes ] && rm -f "$SD/DONE"
setsid nohup .venv/bin/python tools/server/orchestrate.py --plan "$PLAN" \
    >> "$SD/orchestrator.out" 2>&1 < /dev/null &
sleep 3
if L=$(live_orchestrator); then
    echo "started: pid/pgid $L, plan $PLAN"
    echo "follow:  tail -f $SD/run.log"
    echo "status:  bash tools/server/status.sh${PLAN_ARG:+ --plan $PLAN_ARG}"
    echo "stop:    bash tools/server/stop.sh${PLAN_ARG:+ --plan $PLAN_ARG}"
else
    echo "the orchestrator did not stay up; last lines of $SD/orchestrator.out:" >&2
    tail -20 "$SD/orchestrator.out" >&2; exit 1
fi
