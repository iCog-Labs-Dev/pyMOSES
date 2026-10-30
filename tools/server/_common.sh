# Sourced by start.sh, stop.sh and status.sh: REPO, RES, PLAN, SD from
# "--plan round|smoke|round2|smoke2|round3|smoke3|FILE" (default round; env PLAN works too, e.g.
# PLAN=round2 or PLAN=tools/server/plan2.json).
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RES=$REPO/../gomea-results
PY3=$REPO/.venv/bin/python; [ -x "$PY3" ] || PY3=python3
PLAN_ARG=${PLAN:-round}
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --plan) PLAN_ARG=$2; shift 2 ;;
        --plan=*) PLAN_ARG=${1#--plan=}; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
case "$PLAN_ARG" in
    round) PLAN=$REPO/tools/server/plan.json ;;
    smoke) PLAN=$REPO/tools/server/plan-smoke.json ;;
    round2) PLAN=$REPO/tools/server/plan2.json ;;
    smoke2) PLAN=$REPO/tools/server/plan2-smoke.json ;;
    round3) PLAN=$REPO/tools/server/plan3.json ;;
    smoke3) PLAN=$REPO/tools/server/plan3-smoke.json ;;
    *) PLAN=$(realpath "$PLAN_ARG") ;;
esac
[ -f "$PLAN" ] || { echo "no plan file $PLAN" >&2; exit 2; }
SD=$RES/$("$PY3" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state_dir", "server-run"))' "$PLAN" 2>/dev/null || echo server-run)
PIDFILE=$SD/orchestrator.pid
# prints "PID PGID" of a live orchestrator of this plan, or nothing
live_orchestrator() {
    [ -f "$PIDFILE" ] || return 1
    local pid pgid
    read -r pid pgid < <("$PY3" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["pid"], d["pgid"])' "$PIDFILE" 2>/dev/null) || return 1
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "orchestrate.py" || return 1
    echo "$pid $pgid"
}
