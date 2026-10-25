"""Smoke-run the GP-GOMEA binaries on a tiny regression problem and compare their output
with the local machine's binaries (tools/server/gpgomea_smoke_expected.json).

    python3 tools/server/gpgomea_smoke.py GPGOMEA GPGOMEA_SEED DATASETS_DIR [--out JSON]
    python3 tools/server/gpgomea_smoke.py GPGOMEA GPGOMEA_SEED DATASETS_DIR --write-expected

The problem: Yac split 1 (train.dat, test.dat), seed 7, 1f part B's GP-GOMEA setting
(linear scaling, +_-_*_aq, ERC, GOMEA, initial height 4, IMS 4_1) at population 64 and 5
generations (about 17,000 evaluations, under a second). The comparison drops the lines
that differ between machines for reasons other than arithmetic (the data paths and the
running time) and compares the rest exactly; gpgomea-seed prints every seed's fitness,
scaling terms and constants at 17 significant digits, so it is a strict floating-point
check. Prints a JSON report; exit code 0 when both binaries ran (match or not).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

EXPECTED = Path(__file__).resolve().parent / "gpgomea_smoke_expected.json"
ARGS = ["--seed", "7", "--parallel", "1", "--linearscaling", "--functions", "+_-_*_aq",
        "--erc", "--gomea", "--initmaxtreeheight", "4", "--ims", "4_1", "--popsize", "64",
        "--generations", "5"]


def run(binary: str, ddir: Path) -> dict:
    sdir = ddir / "Yac" / "splits" / "s1"
    argv = [str(Path(binary).absolute()), "--prob", "symbreg", "--train", str((sdir / "train.dat").absolute()),
            "--test", str((sdir / "test.dat").absolute()), *ARGS]
    with tempfile.TemporaryDirectory() as tmp:  # GP-GOMEA writes result.txt to its cwd
        try:
            p = subprocess.run(argv, cwd=tmp, capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    keep = [ln for ln in p.stdout.splitlines()
            if not ln.startswith(("# train:", "# test:", "Running time:"))]
    text = "\n".join(keep) + "\n"
    ok = p.returncode == 0 and any(ln.startswith("Evaluations:") for ln in keep)
    return {
        "ok": ok,
        "rc": p.returncode,
        "stderr_tail": p.stderr[-500:],
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "best": next((ln for ln in keep if ln.startswith("Best solution found:")), None),
        "evaluations": next((ln for ln in keep if ln.startswith("Evaluations:")), None),
        "n_seed_lines": sum(ln.startswith("SEED_") for ln in keep),
        "output": text,
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("gpgomea")
    ap.add_argument("gpgomea_seed")
    ap.add_argument("datasets_dir")
    ap.add_argument("--out")
    ap.add_argument("--write-expected", action="store_true")
    a = ap.parse_args()
    ddir = Path(a.datasets_dir)
    res = {"gpgomea": run(a.gpgomea, ddir), "gpgomea-seed": run(a.gpgomea_seed, ddir)}
    if a.write_expected:
        EXPECTED.write_text(json.dumps(
            {"note": "output of the local machine's binaries (gpgomea_smoke.py)", "args": ARGS,
             **{k: {f: v[f] for f in ("sha256", "best", "evaluations", "output")}
                for k, v in res.items()}}, indent=1) + "\n")
        print(f"wrote {EXPECTED}")
        return 0
    exp = json.loads(EXPECTED.read_text()) if EXPECTED.exists() else {}
    for k, v in res.items():
        v["matches_local"] = bool(exp) and v.get("sha256") == exp.get(k, {}).get("sha256")
        v["local_best"] = exp.get(k, {}).get("best")
    report = {
        "ok": all(v["ok"] for v in res.values()),
        "matches_local": all(v["matches_local"] for v in res.values()),
        **res,
    }
    text = json.dumps(report, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {f: x for f, x in v.items() if f != "output"})
                      for k, v in report.items()}, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
