#!/usr/bin/env bash
# Generic phase launcher. configs/gomea/<PHASE>/ holds spec.json (tools/gomea/phase_report.py)
# and the stage-1 manifests bool.json (benchmarks.gomea.runner) and/or sr.json
# (moses_sr_runner.py); optional extra stage-1 regression manifests sr-<name>.json (for arm
# sets that differ by dataset); optional stage-1 gp.json (stand-alone GP-GOMEA,
# gpgomea_sr_runner.py); optional extensions bool-ext-<task>.json, sr-ext-<dataset>.json,
# gp-ext-<dataset>.json.
# Runs stage 1 (Boolean first, then regression, never concurrently), reports stage 1, runs
# the extension of every problem with an "extend" primary contrast (one at a time), and
# reports stage 2 over all result directories.
#   [WORKERS=N] setsid nohup tools/gomea/phase_run.sh PHASE > ../gomea-results/PHASE.log 2>&1 &
set -u
[ $# -eq 1 ] || { echo "usage: $0 PHASE" >&2; exit 2; }
P=$1
cd "$(dirname "$0")/../.."
R=../gomea-results
PY=.venv/bin/python
Q=configs/gomea/$P
W=${WORKERS:-16}
[ -f "$Q/spec.json" ] || { echo "no $Q/spec.json" >&2; exit 2; }
[ -f "$Q/bool.json" ] || [ -f "$Q/sr.json" ] || { echo "no $Q/bool.json or $Q/sr.json" >&2; exit 2; }
mkdir -p "$R/$P"
dir_of() {  # result directory of a manifest
    case "$(basename "$1")" in
        bool*) $PY -m benchmarks.gomea.runner "$1" --dry-run | sed -n 's/^result directory: //p' ;;
        sr*|gp*) echo "$R/$P/$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["name"])' "$1")" ;;
    esac
}
bench() { $PY -m benchmarks.gomea.runner "$1" --workers "$W" --recursion-limit 100000 \
            --mem-limit-mb 3072 >> "$R/$P-bench.log" 2>&1; }
sr() { $PY tools/gomea/moses_sr_runner.py "$1" "$R/$P" --workers "$W" >> "$R/$P-sr.log" 2>&1; }
gp() { $PY tools/gomea/gpgomea_sr_runner.py "$1" "$R/$P" --workers "$W" >> "$R/$P-gp.log" 2>&1; }
run() { case "$(basename "$1")" in bool*) bench "$1" ;; sr*) sr "$1" ;; gp*) gp "$1" ;; esac; }
echo "start $(date -Is)"
D1=""
for m in "$Q/bool.json" "$Q/sr.json" "$Q"/sr-*.json "$Q/gp.json"; do
    [ -f "$m" ] || continue
    case "$(basename "$m")" in sr-ext-*) continue ;; esac  # extensions run at stage 2
    echo "run $m $(date -Is)"
    run "$m"
    D1="${D1:+$D1,}$(dir_of "$m")"
done
echo "stage 1 done $(date -Is)"
$PY tools/gomea/phase_report.py --stage 1 "$Q/spec.json" "$R/$P/report-stage1" "$D1"
V=$R/$P/report-stage1/verdict.json
# problems with an "extend" primary contrast
EXTP=$($PY -c '
import json, sys
spec, v = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
print(" ".join(sorted({p for p, n, a, b in spec["primary"] if v.get(f"{p}:{n}") == "extend"})))
' "$Q/spec.json" "$V" 2>/dev/null)
D2=$D1; EXT=no
for t in $EXTP; do
    for m in "$Q/bool-ext-$t.json" "$Q/sr-ext-$t.json" "$Q/gp-ext-$t.json"; do
        [ -f "$m" ] || continue
        echo "extend $t: $m $(date -Is)"
        run "$m"
        D2="$D2,$(dir_of "$m")"; EXT=yes
    done
done
if [ "$EXT" = yes ]; then
    $PY tools/gomea/phase_report.py --stage 2 "$Q/spec.json" "$R/$P/report-stage2" "$D2"; fi
echo "all done $(date -Is)"
