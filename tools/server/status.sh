#!/usr/bin/env bash
# State, the chain (each plan's state and bundle), current phase and workers, step table, last
# events, resources, scoreboard.
#   bash tools/server/status.sh [--plan round|smoke|round2|smoke2|round3|smoke3|round4|smoke4|round5|smoke5|round6|smoke6|round7|smoke7|FILE]
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$REPO" && exec .venv/bin/python tools/server/orchestrate.py --plan "$PLAN" --status
