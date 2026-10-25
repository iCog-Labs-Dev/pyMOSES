"""S0 reproducibility check of the server round (SERVER-ROUND-PREREGISTRATION.md, "S0").

Four runs whose local results are known are rerun with the round's code, through the same
runners and the same per-run settings as the phases (one-run manifests in
`configs/gomea/server-repro/`), and compared EXACTLY with the local values recorded in
`tools/server/repro_expected.json`:

* regression (moses_sr_runner.py): test_nmse, train_nmse, n_evals, final_program;
* Boolean (benchmarks.gomea.runner): success, evals_to_target, the final program
  (`stdout_best_line`). CPU time is never compared.

    .venv/bin/python tools/server/repro_check.py [--out DIR] [--report FILE]
        [--workers N] [--only ID,ID]

`--out` (default `../gomea-results/server-run/repro`) holds the run directories; reruns
resume (a run with a summary.json is not run again). The report (default
`DIR/../repro.json`) has `ok` (all compared runs match), and per run every field's
expected and obtained value. Exit code 0 when every selected run matches, 1 otherwise.
`--only` (or env REPRO_ONLY) selects runs by id; the others are reported as not run.

    .venv/bin/python tools/server/repro_check.py --extract LOCAL_RESULTS

rewrites `repro_expected.json` from the local summaries (done once, on the local machine).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO.parent / "gomea-results"
EXPECTED = Path(__file__).resolve().parent / "repro_expected.json"
PY = REPO / ".venv" / "bin" / "python"

SR_FIELDS = ("test_nmse", "train_nmse", "n_evals", "final_program")
BOOL_FIELDS = ("success", "evals_to_target", "stdout_best_line")

# id: (kind, manifest, run id, local summary relative to the local results root)
RUNS = {
    "1ad-EnH-GOMEA-s1": ("sr", "configs/gomea/server-repro/sr-EnH-GOMEA.json",
                         "EnH__GOMEA__s1", "phase1ad/phase1ad-sr/runs/EnH__GOMEA__s1"),
    "1ad-EnC-GOMEA-UNI-s1": ("sr", "configs/gomea/server-repro/sr-EnC-GOMEA-UNI.json",
                             "EnC__GOMEA-UNI__s1", "phase1ad/phase1ad-sr/runs/EnC__GOMEA-UNI__s1"),
    "1ad-Con-GOMEA-UNI-s1": ("sr", "configs/gomea/server-repro/sr-Con-GOMEA-UNI.json",
                             "Con__GOMEA-UNI__s1",
                             "phase1ad/phase1ad-sr-Con/runs/Con__GOMEA-UNI__s1"),
    "1ae-mux11-GOMEA-s3001": ("bool", "configs/gomea/server-repro/bool-mux11-GOMEA.json",
                              "mux11__GOMEA__s3001",
                              "1/phase1ae-bool-de50f5d62591/runs/mux11__GOMEA__s3001"),
}


def extract(local_res: Path) -> dict:
    out = {
        "note": "Local results the S0 reproducibility check compares against "
                "(written by repro_check.py --extract from the local machine's results).",
        "runs": {},
    }
    for rid, (kind, manifest, run_id, rel) in RUNS.items():
        s = json.loads((local_res / rel / "summary.json").read_text())
        fields = SR_FIELDS if kind == "sr" else BOOL_FIELDS
        out["runs"][rid] = {
            "kind": kind,
            "manifest": manifest,
            "run_id": run_id,
            "local_summary": f"../gomea-results/{rel}/summary.json",
            "expected": {f: s.get(f) for f in fields},
            "local_cpu_s": s.get("cpu_s"),
            "local_wall_s": s.get("wall_s"),
            "local_status": s.get("status"),
        }
    return out


def _bool_result_dir(manifest: str, out: Path) -> Path:
    r = subprocess.run([str(PY), "-m", "benchmarks.gomea.runner", manifest, "--dry-run",
                        "--output-root", str(out)], cwd=REPO, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("result directory: "):
            return Path(line.split(": ", 1)[1])
    raise RuntimeError(f"dry run of {manifest} gave no result directory: {r.stdout}{r.stderr}")


def _summary_path(spec: dict, out: Path) -> Path:
    if spec["kind"] == "sr":
        name = json.loads((REPO / spec["manifest"]).read_text())["name"]
        return out / name / "runs" / spec["run_id"] / "summary.json"
    return _bool_result_dir(spec["manifest"], out) / "runs" / spec["run_id"] / "summary.json"


def _command(spec: dict, out: Path) -> list[str]:
    if spec["kind"] == "sr":
        return [str(PY), "tools/gomea/moses_sr_runner.py", spec["manifest"], str(out),
                "--workers", "1"]
    # the per-run settings of phase_run.sh's Boolean stage
    return [str(PY), "-m", "benchmarks.gomea.runner", spec["manifest"], "--workers", "1",
            "--recursion-limit", "100000", "--mem-limit-mb", "3072", "--output-root", str(out)]


def compare(spec: dict, got: dict | None) -> dict:
    fields = {}
    for f, want in spec["expected"].items():
        have = None if got is None else got.get(f)
        fields[f] = {"expected": want, "got": have, "match": got is not None and have == want}
    return {
        "match": got is not None and all(v["match"] for v in fields.values()),
        "status": None if got is None else got.get("status"),
        "cpu_s": None if got is None else got.get("cpu_s"),
        "wall_s": None if got is None else got.get("wall_s"),
        "local_cpu_s": spec.get("local_cpu_s"),
        "fields": fields,
    }


def run_all(out: Path, workers: int, only: list[str] | None) -> dict:
    expected = json.loads(EXPECTED.read_text())["runs"]
    selected = [r for r in expected if only is None or r in only]
    out.mkdir(parents=True, exist_ok=True)
    paths = {r: _summary_path(expected[r], out) for r in selected}
    todo = [r for r in selected if not paths[r].exists()]
    active: dict[str, tuple] = {}
    rcs: dict[str, int | None] = {}
    t0 = time.time()
    while todo or active:
        while todo and len(active) < max(1, workers):
            r = todo.pop(0)
            log = open(out / f"{r}.log", "a")
            log.write(f"=== {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(_command(expected[r], out))}\n")
            log.flush()
            p = subprocess.Popen(_command(expected[r], out), cwd=REPO, stdout=log,
                                 stderr=subprocess.STDOUT)
            active[r] = (p, log)
            print(f"started {r} (pid {p.pid})", flush=True)
        time.sleep(2)
        for r in list(active):
            p, log = active[r]
            if p.poll() is not None:
                log.close()
                rcs[r] = p.returncode
                del active[r]
                print(f"finished {r} rc={p.returncode}", flush=True)
    runs = {}
    for r in expected:
        if r not in selected:
            runs[r] = {"match": None, "skipped": "not selected (--only / REPRO_ONLY)"}
            continue
        got = json.loads(paths[r].read_text()) if paths[r].exists() else None
        runs[r] = {**compare(expected[r], got), "rc": rcs.get(r), "summary": str(paths[r])}
    compared = [v for v in runs.values() if v.get("match") is not None]
    return {
        "ok": bool(compared) and all(v["match"] for v in compared),
        "n_selected": len(selected),
        "n_match": sum(bool(v["match"]) for v in compared),
        "wall_s": time.time() - t0,
        "runs": runs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", metavar="LOCAL_RESULTS")
    ap.add_argument("--out", default=str(RES / "server-run" / "repro"))
    ap.add_argument("--report", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=os.environ.get("REPRO_ONLY") or None)
    a = ap.parse_args()
    if a.extract:
        EXPECTED.write_text(json.dumps(extract(Path(a.extract)), indent=1) + "\n")
        print(f"wrote {EXPECTED}")
        return 0
    only = [x.strip() for x in a.only.split(",") if x.strip()] if a.only else None
    out = Path(a.out).absolute()
    rep = run_all(out, a.workers, only)
    report = Path(a.report) if a.report else out.parent / "repro.json"
    tmp = report.with_name(report.name + ".tmp")
    tmp.write_text(json.dumps(rep, indent=1) + "\n")
    os.replace(tmp, report)
    for r, v in rep["runs"].items():
        bad = [f for f, x in v.get("fields", {}).items() if not x["match"]]
        print(f"{r}: {'skipped' if v['match'] is None else 'match' if v['match'] else 'MISMATCH ' + ','.join(bad)}")
    print(f"repro ok={rep['ok']} ({rep['n_match']}/{rep['n_selected']} match); report {report}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
