# Running the Phase 1 confirmation on a server

> The confirmation has already been run (`PHASE1-FINDINGS.md`). This runbook now serves
> for **reproducing** it. Check out the tag `gomea-phase1-confirmation`, whose code is
> identical to the manifests' pinned `c6301cb`. Later commits change `pymoses/`, so the
> code check would refuse to run on them.

This runbook covers everything needed to run the pre-registered Phase 1 experiments
(`PREREGISTRATION.md`) on another machine and bring the results back. The code is
standard-library Python only. There is nothing to compile.

## 1. What it costs

These figures were measured on the development machine, one run at a time, with no other
load. Each run is a separate single-threaded process.

| Manifest | Runs | Typical CPU per run | Total CPU |
|---|---|---|---|
| `smoke` | 24 | ~5 s | ~2 min |
| `confirm-frozen` (11 frozen exemplars × 6 arms × 10 seeds, `-m 5000`) | 660 | ~1 min | ~11 h |
| `confirm-e2e` (maj5 + mux6 × 6 arms × 30 seeds, `-m 20000`) | 360 | HC 1–2 min; GOMEA arms ~4 min on maj5, ~1–2 min on mux6 | ~15 h |
| `confirm-e2e-pa4` (pa4 × 6 arms × 20 seeds, `-m 20000`) | 120 | HC ~8 min, GOMEA arms ~4 min | ~9 h |

The total is about **35 CPU-hours**. With 16 dedicated cores that is about 2.5 h of wall
time; with 8 cores, about 5 h.

Hyper-threads and slower efficiency cores stretch individual runs a lot: under an
oversubscribed laptop load the same runs took 3–5× longer. Use roughly one worker per
physical core.

**Memory.** The peak RSS measured per run is 29–40 MB. Pass `--mem-limit-mb 512` (the
driver's default), which caps each run's address space. Total memory is then at most
about `workers × 512 MB`, and in practice about `workers × 40 MB`.

If a run exceeds the cap, it is recorded as `status: error, error: memory_limit`. It never
takes down the batch and is never silently scored as a bad program. An ok run that comes
within 90 % of the cap is flagged `near_memory_limit`. Pick workers so that
`workers × 512 MB` fits the server's cap. Example: a 4 GiB cap allows up to 8 workers.

## 2. Get the exact code onto the server

The results are only valid for the committed code. The runner refuses to run if anything
under `pymoses/`, `benchmarks/` or `pyproject.toml` differs from the manifests' `commit`
(`c6301cb…`). Commits that touch only docs/configs/tools are allowed.

Branch: `experiment/moses-gomea-discrete`. It has **not** been pushed. Choose one of:

```bash
# (a) Push the branch from the dev machine, then clone it on the server
git push origin experiment/moses-gomea-discrete            # dev machine
git clone --branch experiment/moses-gomea-discrete https://github.com/iCog-Labs-Dev/pyMOSES.git
# (b) Or copy a bundle with no remote involved
git bundle create gomea.bundle experiment/moses-gomea-discrete   # dev machine
git clone --branch experiment/moses-gomea-discrete gomea.bundle pyMoses   # server
```

## 3. Python ≥ 3.14 and a venv

```bash
cd pyMoses
python3.14 --version || uv python install 3.14   # if the server lacks 3.14; uv is optional
python3.14 -m venv .venv
.venv/bin/pip install -e ".[dev]"                # dev extras only add pytest/ruff; runtime deps: none
```

Optional check (about 5 min): run `env -u FORCE_COLOR .venv/bin/python -m pytest tests/gomea -q`.
The expected result is 314 passed.

## 4. Pre-flight (seconds)

```bash
OUT=/path/with/space/gomea-results      # anywhere outside the repo
for m in smoke confirm-frozen confirm-e2e confirm-e2e-pa4; do
  .venv/bin/python -m benchmarks.gomea.runner configs/gomea/phase1/$m.json \
      --output-root "$OUT" --workers 16 --mem-limit-mb 512 --dry-run
done
```

Each manifest must print `code check: ok`. If it says the code differs, stop. Do not use
`--allow-dirty` for the confirmation.

## 5. Run everything (one command, resumable)

Run it inside `tmux` or `screen`, or under `nohup`:

```bash
tools/gomea/run_phase1.sh "$OUT" 16 512      # OUTPUT_ROOT WORKERS MEM_LIMIT_MB
```

The driver runs `smoke`, then `confirm-frozen`, then `confirm-e2e`, then
`confirm-e2e-pa4`, then the analysis. Its log goes to `$OUT/run_phase1.log`.

* **Progress:** `ls $OUT/1/*/runs/*/summary.json | wc -l`; the total is 1164.
* **Interrupted?** Re-run the same command. Completed runs (ok, timeout or error) are
  skipped, and each new run starts from scratch. Nothing needs to be repaired.
* **Retrying a failed run on purpose:** delete its `$OUT/1/<manifest>-<hash>/runs/<run_id>/`
  directory and re-run.
* **Stopping:** Ctrl-C (or `kill -INT` on the runner). Children are terminated, and
  completed summaries stay.
* The runner exits non-zero when some runs timed out or errored. That is recorded evidence
  for the analysis, and the driver continues past it.

The result directory names are the manifest name plus a hash of its content. The output
path and worker count are not part of that hash, so they are the same on every machine:
`smoke-2cac4e10bdb6`, `confirm-frozen-490ea023700c`, `confirm-e2e-bbfe0e929452`,
`confirm-e2e-pa4-b932c6bd8748`.

## 6. Bring the results back

The bulk results (per-run traces and GOMEA telemetry) are about a few hundred MB at most.
Compress them:

```bash
tar czf gomea-phase1-results.tgz -C "$OUT" 1 run_phase1.log
```

On the dev machine, extract the archive into
`/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/gomea-results/`, so the directories
land under `gomea-results/1/`. Then tell Claude that the results are there. It will:

1. check the per-run checksums;
2. rerun `benchmarks.gomea.analyze` (which works on a copied directory);
3. do the pre-registered analysis, including the frozen-fixture pooling per task family;
4. write `PHASE1-FINDINGS.md` (300–500 words) and the machine-readable bundle;
5. commit the manifests' checksums and reports.

If you want the tables right away on the server, the driver has already written
`analysis/` inside each result directory: `aggregate_results.csv`, `contrasts.csv`,
`failures.md` and `figures/*.svg`.

## 7. What not to do

(Known and disclosed: every GOMEA arm writes its per-generation telemetry, which includes a cheap diversity pass, and that time is included in GOMEA's `cpu_s`. HC has no equivalent. The effect is small and works against GOMEA.)


* Use a **full** clone, not `--depth`. The code check needs the manifest commit in the history, and on a shallow clone it refuses to run (which is safe, but blocks you).
* Do not edit code (`pymoses/`, `benchmarks/`, **`pyproject.toml`**) or manifests on the server. Note: the local-modification check does not look at `pyproject.toml`, so this rule is on you. If you must change a manifest, it gets a
  new hash and results directory, and the change is a deviation from the pre-registration
  that must be reported.
* Do not delete timed-out or errored runs to "clean up". They are censored evidence and
  are counted in the analysis.
* Do not run two copies of the driver on the same `OUT` at once.
