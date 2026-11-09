#!/usr/bin/env bash
# Start (or resume) the server round in the background and return at once.
#   bash tools/server/start.sh [--plan round|smoke|round2|smoke2|round3|smoke3|round4|smoke4|round5|smoke5|round6|smoke6|round7|smoke7|round8|smoke8|round9|smoke9|FILE] [--force]
# A plan with "then" starts a chain (round4, then round5; docs/gomea/SERVER-RUN.md). If this plan
# is DONE, this script starts or resumes the first later plan that is not DONE.
# --force starts this plan even if it is DONE (it then only rewrites the summary and bundles),
# and that session does not start the next plan.
# Refuses if an orchestrator of this chain is already running. Survives closing the
# terminal (setsid + nohup), not a pod restart: after one, run this again; it resumes.
# (--chained: only the end of the plan before this one uses it.)
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
FORCE=no; CHAINED=no
for a in "${ARGS[@]}"; do
    [ "$a" = --force ] && FORCE=yes
    [ "$a" = --chained ] && CHAINED=yes
done
cd "$REPO" || exit 1
[ -x .venv/bin/python ] || { echo "no .venv: run bash tools/server/setup.sh first" >&2; exit 2; }
mkdir -p "$SD"
if L=$(chain_live); then
    echo "already running (pid pgid plan: $L); not starting" >&2; exit 1
fi
if [ $FORCE = no ]; then
    T=$(chain_cmd target "$PLAN") || { echo "cannot read the chain of $PLAN" >&2; exit 2; }
    if [ -z "$T" ]; then
        while read -r P; do
            D=$RES/$(state_dir_of "$P")
            echo "The round is DONE ($(cat "$D/DONE")). Bring back: $(ls -1t "$D"/bundle-final-*.tar.gz 2>/dev/null | head -1)"
        done < <(chain_cmd forward "$PLAN")
        exit 0
    fi
    if [ "$T" != "$(realpath "$PLAN")" ]; then
        echo "$PLAN is DONE; the next plan of the chain is $T"
        unset RETRY_FAILED REPRO_ONLY  # they belong to the plan that the command names
        PLAN=$T
        SD=$RES/$(state_dir_of "$PLAN")
        mkdir -p "$SD"
    fi
fi
LOCK=$(chain_cmd lock "$PLAN")
if [ $CHAINED = yes ]; then
    if [ -e "$(chain_cmd stop-marker "$PLAN")" ]; then
        echo "stop.sh ran; not starting $PLAN"; exit 0
    fi
    export ORCH_CHAINED=1
else
    unset ORCH_CHAINED
fi
unset ORCH_NO_CHAIN
mkdir -p "$(dirname "$LOCK")"
if command -v flock >/dev/null && ! flock -n "$LOCK" true; then
    echo "$LOCK is held by another process; not starting" >&2; exit 1
fi
if [ $FORCE = yes ]; then
    rm -f "$SD/DONE"
    export ORCH_NO_CHAIN=1
fi
setsid nohup .venv/bin/python tools/server/orchestrate.py --plan "$PLAN" \
    >> "$SD/orchestrator.out" 2>&1 < /dev/null &
UP=no
for _ in $(seq 1 30); do
    sleep 1
    if L=$(chain_live) || [ -f "$SD/DONE" ]; then UP=yes; break; fi
done
if [ $UP = yes ]; then
    echo "started: pid pgid plan: ${L:-finished at once}, plan $PLAN"
    echo "follow:  tail -f $SD/run.log"
    echo "status:  bash tools/server/status.sh${PLAN_ARG:+ --plan $PLAN_ARG}"
    echo "stop:    bash tools/server/stop.sh${PLAN_ARG:+ --plan $PLAN_ARG}"
else
    echo "the orchestrator did not stay up; last lines of $SD/orchestrator.out:" >&2
    tail -20 "$SD/orchestrator.out" >&2; exit 1
fi
