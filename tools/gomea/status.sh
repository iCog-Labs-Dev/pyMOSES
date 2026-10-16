#!/usr/bin/env bash
# One-screen status of running experiments: running runners, plus the last line of every
# experiment log changed in the last 24 h. Usage (any directory):
#   tools/gomea/status.sh            # once
#   watch -n 30 tools/gomea/status.sh  # refresh every 30 s
R="$(cd "$(dirname "$0")/../../../gomea-results" && pwd)"
echo "== $(date '+%F %T')  load: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "== running launchers / runners"
ps -eo pid=,args= | grep -E 'phase1[a-z]*_run.sh|benchmarks.gomea.runner|moses_sr_runner.py' \
    | grep -vE 'grep|zsh|/bin/sh' \
    | awk '{for (i = 2; i <= NF; i++) if ($i ~ /(_run\.sh|\.json)$/) { print "  " $1 "  " $i; break }}'
echo "== experiment logs (last line, changed in the last 24 h)"
find "$R" -maxdepth 1 -name 'phase1*.log' -mmin -1440 -printf '%T@ %p\n' | sort -n | cut -d' ' -f2- |
while read -r f; do
    printf '  %-28s %s\n' "$(basename "$f")" "$(grep -v '^{' "$f" | tail -n 1 | cut -c1-110)"
done
