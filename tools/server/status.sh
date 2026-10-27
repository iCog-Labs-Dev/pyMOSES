#!/usr/bin/env bash
# State, current phase and workers, step table, last events, resources, scoreboard.
#   bash tools/server/status.sh [--plan round|smoke|round2|smoke2|FILE]
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$REPO" && exec .venv/bin/python tools/server/orchestrate.py --plan "$PLAN" --status
