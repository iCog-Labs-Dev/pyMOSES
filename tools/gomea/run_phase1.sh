#!/usr/bin/env bash
# Run the whole Phase 1 confirmation (smoke -> frozen -> end-to-end -> pa4), then analyse.
#
# Usage (from the repository root):
#   tools/gomea/run_phase1.sh OUTPUT_ROOT WORKERS [MEM_LIMIT_MB]
#
# It is resumable: re-running the same command skips every run that already has a
# summary.json. Runs are never retried automatically. To retry one, delete its
# runs/<run_id> directory. See docs/gomea/SERVER-RUNBOOK.md.
set -euo pipefail

OUT=${1:?output root, e.g. /data/gomea-results}
WORKERS=${2:?number of parallel runs, e.g. 16}
MEM=${3:-512}
PY=${PY:-.venv/bin/python}
CFG=configs/gomea/phase1

mkdir -p "$OUT"
LOG="$OUT/run_phase1.log"
echo "== $(date -Is) start: out=$OUT workers=$WORKERS mem=${MEM}MB head=$(git rev-parse HEAD)" | tee -a "$LOG"

for m in smoke confirm-frozen confirm-e2e confirm-e2e-pa4; do
  echo "== $(date -Is) $m" | tee -a "$LOG"
  # The runner exits non-zero when any run is not "ok" (a timeout or an error). That is
  # recorded evidence, not a harness failure, so keep going and let the analysis report it.
  "$PY" -m benchmarks.gomea.runner "$CFG/$m.json" \
      --output-root "$OUT" --workers "$WORKERS" --mem-limit-mb "$MEM" 2>&1 | tee -a "$LOG" || true
done

echo "== $(date -Is) analysis" | tee -a "$LOG"
for d in "$OUT"/1/confirm-e2e-[0-9a-f]* "$OUT"/1/confirm-e2e-pa4-* "$OUT"/1/confirm-frozen-*; do
  [ -d "$d" ] || continue
  "$PY" -m benchmarks.gomea.analyze "$d" --baseline HC \
      --contrasts "LT-B:HC,LT-B:LT-MI,LT-B:UNI,RT:LT-B,LT-B-FI:LT-B" 2>&1 | tee -a "$LOG"
done
echo "== $(date -Is) done" | tee -a "$LOG"
