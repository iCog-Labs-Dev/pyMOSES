#!/usr/bin/env bash
# Stop the round: SIGTERM to the orchestrator's process group (recorded in
# orchestrator.pid), SIGKILL after GRACE seconds (default 60), then write a fresh
# bundle-latest.tar.gz. No pattern kills. Restart with start.sh; it resumes.
#   bash tools/server/stop.sh [--plan round|smoke|round2|smoke2|round3|smoke3|FILE]
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
GRACE=${GRACE:-60}
cd "$REPO" || exit 1
if ! L=$(live_orchestrator); then echo "no orchestrator running for $PLAN"; exit 0; fi
read -r PID PGID <<< "$L"
echo "stopping orchestrator pid $PID, process group $PGID"
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
.venv/bin/python tools/server/orchestrate.py --plan "$PLAN" --bundle
