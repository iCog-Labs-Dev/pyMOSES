"""Identity harness for speed-up work: same inputs must give the same outputs.

Usage (from the repository root)::

    .venv/bin/python tools/gomea/identity_check.py record OUT_DIR [--jobs J] [--only REGEX]
    .venv/bin/python tools/gomea/identity_check.py compare BASE_DIR NEW_DIR

`record` runs a fixed set of short pyMoses runs (HC and GOMEA on mux11, maj5, a frozen maj5
fixture and the Con regression dataset), each in its own process, and saves per case:
stdout, the GOMEA event log with timing fields removed, and the CPU time and evaluation
count. `compare` reports, per case, whether stdout and the event log are byte-identical,
and the CPU-time ratio NEW / BASE.

A speed-up is accepted only if every case is identical. The cases are deliberately short
(a few minutes each) so the check can run after every change.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable  # run with the main .venv python; the child imports this checkout (cwd)
CON = REPO.parent / "gomea-results/phase1f/datasets/Con/splits/s1/train.csv"
MAJ5_X0 = "or(and(or(and($3 $5) $2) $4) and(or($2 $4) or($3 $5) $1))"
GOMEA_C250 = [
    "-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "16",
    "--gomea-skip-inactive", "0", "--gomea-max-evals-per-deme", "250",
    "--gomea-genotype-memo", "1",
]
REGRESSION = ["-i", str(CON), "-u", "y", "-n", "sin", "-n", "exp", "-n", "log",
              "-v", "30960", "-c", "10"]

CASES: dict[str, list[str]] = {
    "mux11-hc": ["-H", "mux", "-k", "3", "-a", "hc", "-m", "2000", "-r", "3001"],
    "mux11-gomea": ["-H", "mux", "-k", "3", *GOMEA_C250, "-m", "2000", "-r", "3001"],
    "maj5-hc": ["-H", "maj", "-k", "5", "-a", "hc", "-m", "3000", "-r", "1001"],
    "maj5-gomea": ["-H", "maj", "-k", "5", *GOMEA_C250, "-m", "3000", "-r", "1001"],
    "pa4-gomea-uncapped": ["-H", "pa", "-k", "4", "-a", "gomea", "-m", "3000", "-r", "1001"],
    "maj5x0-frozen-n64": [
        "-H", "maj", "-k", "5", "-e", MAJ5_X0, "-g", "1", "-a", "gomea",
        "--gomea-pop-size", "64", "--gomea-skip-inactive", "0", "--gomea-fos", "lt-b",
        "-m", "6000", "-r", "4001",
    ],
    "con-hc": [*REGRESSION, "-a", "hc", "-m", "800", "-r", "1"],
    "con-gomea": [*REGRESSION, *GOMEA_C250, "--gomea-contin", "copy", "-m", "800", "-r", "1"],
}

TIMING = re.compile(r"(^|_)(s|time|timings)$|_s$|^max_time$|^timings$")

CHILD = (
    "import sys, time, json; sys.setrecursionlimit(100000)\n"
    "from pymoses.moses.main.moses_exec import moses_exec\n"
    "t0 = time.process_time()\n"
    "try:\n    rc = moses_exec(['moses', *sys.argv[2:]])\n"
    "except SystemExit as e:\n    rc = e.code\n"
    "open(sys.argv[1], 'w').write(json.dumps({'cpu_s': time.process_time() - t0, 'rc': rc}))\n"
)


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if not TIMING.search(k)}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def _run(name: str, out: Path) -> str:
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    events = d / "events.raw.jsonl"
    argv = ["-l", "WARN", *CASES[name]]
    if any(a in ("gomea", "hc-gomea") for a in argv):
        argv += ["--gomea-log", str(events)]
    p = subprocess.run([PY, "-c", CHILD, str(d / "cpu.json"), *argv], cwd=REPO,
                       capture_output=True, text=True)
    (d / "stdout.txt").write_text(p.stdout)
    (d / "stderr.txt").write_text(p.stderr)
    if events.exists():
        lines = [json.dumps(_strip(json.loads(x)), sort_keys=True)
                 for x in events.read_text().splitlines() if x.strip()]
        (d / "events.jsonl").write_text("\n".join(lines) + "\n")
        evals = sum(json.loads(x).get("ledger", {}).get("requests", 0)
                    for x in events.read_text().splitlines()
                    if '"generations"' in x and '"init"' in x)
        (d / "evals.txt").write_text(str(evals))
        events.unlink()
    return f"{name}: rc={p.returncode}"


def record(out: Path, jobs: int, only: str | None) -> None:
    names = [n for n in CASES if not only or re.search(only, n)]
    with cf.ProcessPoolExecutor(jobs) as ex:
        for msg in ex.map(_run, names, [out] * len(names)):
            print(msg, flush=True)


def compare(base: Path, new: Path) -> int:
    bad = 0
    for name in CASES:
        b, n = base / name, new / name
        if not n.exists():
            continue
        same = True
        for f in ("stdout.txt", "events.jsonl"):
            if (b / f).exists() != (n / f).exists():
                same = False
            elif (b / f).exists() and (b / f).read_bytes() != (n / f).read_bytes():
                same = False
        cb = json.loads((b / "cpu.json").read_text())["cpu_s"]
        cn = json.loads((n / "cpu.json").read_text())["cpu_s"]
        print(f"{name:20s} {'IDENTICAL' if same else 'DIFFERENT'}  cpu {cb:7.1f}s -> {cn:7.1f}s"
              f"  ({cn / cb:.2f}x)")
        bad += not same
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("out")
    r.add_argument("--jobs", type=int, default=8)
    r.add_argument("--only")
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("new")
    a = ap.parse_args()
    if a.cmd == "record":
        record(Path(a.out).resolve(), a.jobs, a.only)
        return 0
    return compare(Path(a.base), Path(a.new))


if __name__ == "__main__":
    sys.exit(main())
