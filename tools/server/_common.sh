# Sourced by start.sh, stop.sh and status.sh: REPO, RES, PLAN, SD from
# "--plan round|smoke|round2|smoke2|round3|smoke3|round4|smoke4|round5|smoke5|round6|smoke6|round7|smoke7|round8|smoke8|round9|smoke9|round10|smoke10|FILE" (default round; env PLAN
# works too, e.g. PLAN=round2 or PLAN=tools/server/plan2.json).
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
    round4) PLAN=$REPO/tools/server/plan4.json ;;
    smoke4) PLAN=$REPO/tools/server/plan4-smoke.json ;;
    round5) PLAN=$REPO/tools/server/plan5.json ;;
    smoke5) PLAN=$REPO/tools/server/plan5-smoke.json ;;
    round6) PLAN=$REPO/tools/server/plan6.json ;;
    smoke6) PLAN=$REPO/tools/server/plan6-smoke.json ;;
    round7) PLAN=$REPO/tools/server/plan7.json ;;
    smoke7) PLAN=$REPO/tools/server/plan7-smoke.json ;;
    round8) PLAN=$REPO/tools/server/plan8.json ;;
    smoke8) PLAN=$REPO/tools/server/plan8-smoke.json ;;
    round9) PLAN=$REPO/tools/server/plan9.json ;;
    smoke9) PLAN=$REPO/tools/server/plan9-smoke.json ;;
    round10) PLAN=$REPO/tools/server/plan10.json ;;
    smoke10) PLAN=$REPO/tools/server/plan10-smoke.json ;;
    *) PLAN=$(realpath "$PLAN_ARG") ;;
esac
[ -f "$PLAN" ] || { echo "no plan file $PLAN" >&2; exit 2; }
chain_cmd() { "$PY3" "$REPO/tools/server/chain.py" "$@"; }
# the state directory of a plan file
state_dir_of() {
    "$PY3" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state_dir", "server-run"))' "$1" 2>/dev/null || echo server-run
}
SD=$RES/$(state_dir_of "$PLAN")
# prints "PID PGID PLAN_FILE" for each live orchestrator of the chain of this plan, or nothing
chain_live() {
    local out
    out=$(chain_cmd live "$PLAN") && [ -n "$out" ] && echo "$out"
}
