#!/usr/bin/env bash
# Phase 1q deviation helper: while the stage-1 runner (PID $1) or any rerun is alive, start a
# 36,000 s-cap rerun (phase1q_recap.py) for every newly timed-out (task, arm, seed); when
# all are done, write the stage-1 report over the original and rerun directories.
#   setsid nohup tools/gomea/phase1q_autorecap.sh RUNNER_PID > ../gomea-results/phase1q-autorecap.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
R=../gomea-results
PY=.venv/bin/python
M=configs/gomea/phase1q/stage1.json
D1=$($PY -m benchmarks.gomea.runner $M --dry-run | sed -n 's/^result directory: //p')
launch_new() {
    local n; n=configs/gomea/phase1q/recap$(( $(ls -d configs/gomea/phase1q/recap* 2>/dev/null | wc -l) + 1 ))
    mkdir -p "$n"
    $PY tools/gomea/phase1q_recap.py $M "$D1" "$n" > /dev/null
    $PY - "$n" <<'P'
import glob, json, os, sys
new = sys.argv[1]
done = {}
for f in glob.glob("configs/gomea/phase1q/recap*/*.json"):
    if not f.startswith(new + "/"):
        done.setdefault(os.path.basename(f), set()).update(json.load(open(f))["seeds"])
for f in glob.glob(new + "/*.json"):
    m = json.load(open(f))
    m["seeds"] = [s for s in m["seeds"] if s not in done.get(os.path.basename(f), set())]
    if m["seeds"]:
        open(f, "w").write(json.dumps(m, indent=2) + "\n")
    else:
        os.remove(f)
P
    rmdir "$n" 2>/dev/null && return
    for m in "$n"/*.json; do
        echo "rerun $(date -Is) $m"
        $PY -m benchmarks.gomea.runner "$m" --workers 4 --recursion-limit 100000 \
            --mem-limit-mb 3072 >> $R/phase1q-recap.log 2>&1 &
    done
}
alive() { kill -0 "$1" 2>/dev/null || pgrep -f "benchmarks.gomea.runner configs/gomea/phase1q/recap" > /dev/null; }
while alive "$1"; do launch_new; sleep 120; done
launch_new
while pgrep -f "benchmarks.gomea.runner configs/gomea/phase1q/recap" > /dev/null; do sleep 120; launch_new; done
DIRS=$D1
for m in configs/gomea/phase1q/recap*/*.json; do
    DIRS="$DIRS,$($PY -m benchmarks.gomea.runner "$m" --dry-run | sed -n 's/^result directory: //p')"
done
echo "dirs $DIRS"
$PY tools/gomea/phase1q_report.py --stage 1 $R/phase1q/report-stage1 "$DIRS"
echo "recap done $(date -Is)"
