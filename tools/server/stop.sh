#!/usr/bin/env bash
# Stop the round: SIGTERM to the orchestrator's process group (recorded in
# orchestrator.pid), SIGKILL after GRACE seconds (default 60), then write a fresh
# bundle-latest.tar.gz. No pattern kills. Restart with start.sh; it resumes.
# In a chain (round4, then round5) this stops the running orchestrator of the chain, whichever
# it is, and a stopped round does not start the next one.
#   bash tools/server/stop.sh [--plan round|smoke|round2|smoke2|round3|smoke3|round4|smoke4|round5|smoke5|round6|smoke6|FILE]
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
GRACE=${GRACE:-60}
cd "$REPO" || exit 1
# The marker stops a chained start that is on its way (start.sh and orchestrate.py read it).
# The next start that is not chained removes it.
MARKER=$(chain_cmd stop-marker "$PLAN")
mkdir -p "$(dirname "$MARKER")"
date -u +%Y-%m-%dT%H:%M:%SZ > "$MARKER"
if ! LIVE=$(chain_live); then echo "no orchestrator running for the chain of $PLAN"; exit 0; fi
while read -r PID PGID P; do
    echo "stopping orchestrator pid $PID, process group $PGID, plan $P"
    kill -TERM -- "-$PGID" 2>/dev/null
    for ((i = 0; i < GRACE; i++)); do
        kill -0 -- "-$PGID" 2>/dev/null || break
        sleep 1
    done
    if kill -0 -- "-$PGID" 2>/dev/null; then
        echo "still alive after $GRACE s: SIGKILL to process group $PGID"
        kill -KILL -- "-$PGID" 2>/dev/null; sleep 2
    fi
    kill -0 -- "-$PGID" 2>/dev/null && echo "WARNING: process group $PGID still has processes" >&2
    echo "stopped; writing bundle-latest.tar.gz"
    .venv/bin/python tools/server/orchestrate.py --plan "$P" --bundle
done <<< "$LIVE"
