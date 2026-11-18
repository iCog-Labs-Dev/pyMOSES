"""Server round 12 (T2.2): without ``--linear-scaling terms``, the output is byte-identical
to the code before per-term least squares existed.

The reference files in ``data/ls_terms_0117486/`` were captured at commit 0117486 (before
any code for ``--linear-scaling terms``) by running this module as a script there::

    .venv/bin/python tests/gomea/test_linear_scaling_terms_identity.py OUT_DIR

MOSES cases: table regression runs on ``data/const_opt_none/train.csv`` with the fixed
regression setup of ``tools/gomea/moses_sr_runner.py``: GOMEA (the round-1 arm) and
``-a hc``, each without and with ``--linear-scaling 1``; the round 11 BASE flags
(``configs/gomea/phase1be/``; the validation table of the guard is the training table) at
log level INFO; and BASE without LM (``--const-opt none``, no guard), the setting nearest to
BASE-TLS. Compared: the full MOSES stdout, the MOSES log (``-f``) and the ``--gomea-log``
file (absent = empty). The log comparison leaves out the time stamp of each line, the host
name, the path of the test data (replaced by DATA) and the wall-clock column (``elapsed``)
of the "Stats:" rows. The ``--gomea-log`` comparison leaves out the wall-clock fields
(``CLOCK_KEYS``).

Report cases: ``tools/gomea/sr_eval.py`` (``evaluate`` and the command line) on a fixed
candidate list, and ``tools/gomea/moses_sr_runner.py`` on one job whose binary is a shell
script that prints the same fixed list. The splits are made from the same training table
(``SPLIT_OF``). Compared: the ``evaluate`` results and the CLI output as JSON, the runner's
``candidates.json`` and its ``summary.json`` without the machine values (``RUN_CLOCK_KEYS``),
with the path of the run replaced by TMP.

Server round 13 adds one field to the summary of every arm, ``test_nmse_clipped``
(``SERVER-ROUND13-PREREGISTRATION.md``, "Descriptive for every arm"). The summary comparison
leaves out that field only (``ROUND13_NEW_KEYS``); every other field must still be identical,
and the test checks that the new field is present.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data" / "const_opt_none"
REF = Path(__file__).resolve().parent / "data" / "ls_terms_0117486"
TRAIN = DATA / "train.csv"

sys.path.insert(0, str(REPO / "tools" / "gomea"))
import moses_sr_runner  # noqa: E402
import sr_eval  # noqa: E402


def _reg(m: str) -> list[str]:
    return ["-i", str(TRAIN), "-u", "y", "-r", "1", "-n", "sin", "-n", "exp", "-n", "log",
            "-v", "2400", "-c", "10", "-V1", "-l", "WARN", "-m", m]


_GOMEA = [
    "-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "16",
    "--gomea-skip-inactive", "0", "--gomea-max-evals-per-deme", "250",
    "--gomea-genotype-memo", "1", "--gomea-contin", "copy", "--knob-probe-baseline", "reduced",
]  # fmt: skip
_HC = ["-a", "hc", "--knob-probe-baseline", "reduced"]
_LS = ["--linear-scaling", "1"]
# The round 11 BASE arm (configs/gomea/phase1be/), flag for flag, without its validation
# table; and the same without LM (the setting nearest to BASE-TLS).
_BASE_SEARCH = [
    "-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "dynamic",
    "--hc-fraction-of-nn", "0.05", "--gomea-skip-inactive", "0",
    "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
    "--knob-probe-baseline", "reduced", "--linear-scaling", "1",
]  # fmt: skip
_BASE_TAIL = ["--gomea-accept-complexity-ratio", "0.125", "--gomea-deme-store", "accepted"]
_BASE = _BASE_SEARCH + ["--const-opt", "lm", "--const-opt-guard", "val"] + _BASE_TAIL
_BASE_NOLM = _BASE_SEARCH + ["--const-opt", "none"] + _BASE_TAIL
_VAL = ["--const-opt-val", str(TRAIN)]

CASES = {
    "gomea": _reg("800") + _GOMEA,
    "gomea-ls": _reg("800") + _GOMEA + _LS,
    "hc": _reg("800") + _HC,
    "hc-ls": _reg("800") + _HC + _LS,
    "base-info": _reg("1500") + _BASE + _VAL + ["-l", "INFO"],
    "base-nolm": _reg("1500") + _BASE_NOLM,
}
# Cases without linear scaling: also run with an explicit `--linear-scaling 0`.
PLAIN_CASES = ("gomea", "hc")

CLOCK_KEYS = frozenset({"linkage_s", "variation_s", "timings", "max_time"})
_MAIN = ("import sys; from pymoses.moses.main.moses_exec import moses_exec; "
         "sys.exit(moses_exec(sys.argv[1:]))")
_STAMP = re.compile(r"^\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d:\d{3}\] ")

# The fixed candidate list (MOSES stdout form): a sum of terms, a sum with a constant
# term, a product (one term), collinear terms, a term that overflows on some rows, a bare
# argument, and a sum with a division.
CANDIDATES = (
    "-12.5 +(sin(*(1.3 $1)) *(0.5 $2)) \n"
    "-13.25 +(sin($1) $2 0.25) \n"
    "-14 *(2.5 sin(*(1.3 $1))) \n"
    "-15 +($1 *(2 $1)) \n"
    "-16 +(log($1) $2) \n"
    "-17 $2 \n"
    "-18 +(/(1 +(4 $1)) sin($2) exp(*(0.1 $1))) \n"
    "n_evals: 1234\n"
)
RUN_CLOCK_KEYS = frozenset({"wall_s", "cpu_s", "ru_maxrss_kb"})
# server round 13: the new summary field of every arm (not in the references of 0117486)
ROUND13_NEW_KEYS = frozenset({"test_nmse_clipped"})


def canonical_log(text: str) -> str:
    """The MOSES log without its clock and host values (see the module docstring)."""
    out = []
    for line in text.splitlines():
        line = _STAMP.sub("", line)
        if line.startswith("[INFO] hostname:"):
            continue
        line = line.replace(str(DATA), "DATA")
        head, sep, rest = line.partition("Stats: ")
        if sep and not rest.startswith("#"):
            f = rest.split("\t")
            if len(f) > 2:
                f[2] = "T"
            line = head + sep + "\t".join(f)
        out.append(line)
    return "\n".join(out) + "\n"


def canonical_events(text: str) -> str:
    """The ``--gomea-log`` events without their wall-clock fields."""
    lines = []
    for line in text.splitlines():
        ev = {k: v for k, v in json.loads(line).items() if k not in CLOCK_KEYS}
        lines.append(json.dumps(ev, sort_keys=True))
    return ("\n".join(lines) + "\n").replace(str(DATA), "DATA") if lines else ""


def run_case(argv: list[str], workdir: Path) -> tuple[str, str, str]:
    """(stdout, canonical MOSES log, canonical --gomea-log) of one run in ``workdir``. Each
    run is a new process: the MOSES logger opens its file once per process."""
    for f in ("moses.log", "events.jsonl"):
        (workdir / f).unlink(missing_ok=True)
    log = ["--gomea-log", "events.jsonl"] if "gomea" in argv else []
    r = subprocess.run(
        [sys.executable, "-c", _MAIN, "moses", *argv, *log, "-f", "moses.log"],
        cwd=workdir, env={**os.environ, "PYTHONPATH": str(REPO)}, capture_output=True,
        text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    ev = workdir / "events.jsonl"
    return (r.stdout, canonical_log((workdir / "moses.log").read_text()),
            canonical_events(ev.read_text()) if ev.exists() else "")


def make_splits(dst: Path) -> Path:
    """train.csv, val.csv and test.csv from ``data/const_opt_none/train.csv``: row i goes to
    train when i % 5 < 3, to val when i % 5 == 3 and to test when i % 5 == 4."""
    dst.mkdir(parents=True, exist_ok=True)
    lines = TRAIN.read_text().splitlines()
    head, rows = lines[0], lines[1:]
    for name, keep in (("train", (0, 1, 2)), ("val", (3,)), ("test", (4,))):
        body = [r for i, r in enumerate(rows) if i % 5 in keep]
        (dst / f"{name}.csv").write_text("\n".join([head, *body]) + "\n")
    return dst


def sr_eval_report(split_dir: Path) -> str:
    """``sr_eval.evaluate`` on each fixed candidate (train with val, train with test), and
    the CLI output with ``--all``, as one JSON text."""
    train = sr_eval.read_table(str(split_dir / "train.csv"))
    val = sr_eval.read_table(str(split_dir / "val.csv"))
    test = sr_eval.read_table(str(split_dir / "test.csv"))
    rows = []
    for _score, prog in sr_eval.parse_moses_output(CANDIDATES):
        rows.append({"val": sr_eval.evaluate(prog, train, val),
                     "test": sr_eval.evaluate(prog, train, test)})
    cand = split_dir / "candidates.txt"
    cand.write_text(CANDIDATES)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = sr_eval.main(["--train", str(split_dir / "train.csv"),
                           "--test", str(split_dir / "test.csv"),
                           "--moses-output", str(cand), "--all"])
    assert rc == 0
    return json.dumps({"evaluate": rows, "cli": json.loads(buf.getvalue())}, indent=1) + "\n"


def runner_report(work: Path, argv: list[str]) -> tuple[str, str]:
    """(candidates.json, canonical summary.json) of one runner job whose binary prints the
    fixed candidate list."""
    split_dir = make_splits(work / "splits" / "s1")
    fake = work / "fake_moses.sh"
    fake.write_text("#!/bin/sh\ncat <<'EOF'\n" + CANDIDATES + "EOF\n")
    fake.chmod(0o755)
    run_dir = work / "run"
    job = {
        "run_id": "T__ARM__s1", "task": "T", "arm": "ARM", "seed": 1,
        "binary": str(fake), "argv": argv, "budget": {"evals": 1234}, "seed_spec": None,
        "split_dir": str(split_dir), "wall_cap_s": 60, "mem_limit_mb": None,
        "run_dir": str(run_dir),
    }  # fmt: skip
    moses_sr_runner._run_one(job)
    summary = json.loads((run_dir / "summary.json").read_text())
    summary = {k: v for k, v in summary.items()
               if k not in RUN_CLOCK_KEYS and k not in ROUND13_NEW_KEYS}
    text = json.dumps(summary, indent=2, sort_keys=True).replace(str(work), "TMP")
    return (run_dir / "candidates.json").read_text(), text + "\n"


# The runner job of the report case: the BASE-TLS search flags with plain linear scaling.
RUNNER_ARGV = _BASE_NOLM


@pytest.mark.parametrize("case", sorted(CASES))
def test_moses_without_terms_is_byte_identical_to_0117486(case, tmp_path):
    stdout, log, events = run_case(CASES[case], tmp_path)
    assert stdout == (REF / f"{case}.out").read_text()
    assert log == (REF / f"{case}.log").read_text()
    assert events == (REF / f"{case}.events.jsonl").read_text()


@pytest.mark.parametrize("case", PLAIN_CASES)
def test_linear_scaling_0_is_byte_identical_to_0117486(case, tmp_path):
    """An explicit ``--linear-scaling 0`` (off) is the same as no option."""
    stdout, log, events = run_case(CASES[case] + ["--linear-scaling", "0"], tmp_path)
    assert stdout == (REF / f"{case}.out").read_text()
    assert log.replace(" --linear-scaling 0", "") == (REF / f"{case}.log").read_text()
    assert events == (REF / f"{case}.events.jsonl").read_text()


def test_sr_eval_default_is_byte_identical_to_0117486(tmp_path):
    assert sr_eval_report(make_splits(tmp_path)) == (REF / "sr_eval.json").read_text()


def test_runner_default_is_byte_identical_to_0117486(tmp_path):
    cands, summary = runner_report(tmp_path, RUNNER_ARGV)
    assert cands == (REF / "runner.candidates.json").read_text()
    assert summary == (REF / "runner.summary.json").read_text()
    # server round 13: the summary has the new field too (left out of the comparison above)
    full = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert set(full) - set(json.loads(summary)) == RUN_CLOCK_KEYS | ROUND13_NEW_KEYS


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name in sorted(CASES):
        work = out / f"work-{name}"
        work.mkdir(exist_ok=True)
        s, lg, e = run_case(CASES[name], work)
        (out / f"{name}.out").write_text(s)
        (out / f"{name}.log").write_text(lg)
        (out / f"{name}.events.jsonl").write_text(e)
        print(name, len(s.splitlines()), "stdout lines,", len(lg.splitlines()), "log lines,",
              len(e.splitlines()), "events")
    (out / "sr_eval.json").write_text(sr_eval_report(make_splits(out / "work-sr_eval")))
    c, sm = runner_report(out / "work-runner", RUNNER_ARGV)
    (out / "runner.candidates.json").write_text(c)
    (out / "runner.summary.json").write_text(sm)
    print("sr_eval and runner references written")
