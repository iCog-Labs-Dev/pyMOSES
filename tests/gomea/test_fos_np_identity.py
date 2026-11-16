"""Server round 12 (T2.1): the existing ``--gomea-fos`` modes give byte-identical output
after the code for ``--gomea-fos np`` (node-proximity linkage) was added.

The reference files in ``data/fos_np_0117486/`` were captured at commit 0117486 (before
any code for ``np``) by running this module as a script there::

    .venv/bin/python tests/gomea/test_fos_np_identity.py OUT_DIR

Cases: table regression runs on ``data/const_opt_none/train.csv`` with the fixed regression
setup of ``tools/gomea/moses_sr_runner.py`` and the BASE flags of round 12
(``configs/gomea/phase1be/``; the validation table of the guard is the training table):
at ``-m 1500`` and log level INFO (the INFO log holds the command line), at ``-m 600``, and
at ``-m 1500`` with ``--gomea-fos rt`` and with ``--gomea-fos uni`` in place of ``lt-b``.
Also two Boolean ``-a gomea`` runs on the multiplexer (``-H mux -k 2 -m 10000``) with the
Boolean setting of round 12 (phase B): one with the default FOS (no ``--gomea-fos``, so
lt-b) and one with ``--gomea-fos rt``. Compared: the full MOSES stdout, the MOSES log
(``-f``) and the ``--gomea-log`` file, as in ``test_const_opt_reserve_identity.py`` (the
same canonical forms). The ``params`` of each ``invocation_start`` event hold ``fos``, so
the events also prove that the value of ``fos`` did not change.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data" / "const_opt_none"
REF = Path(__file__).resolve().parent / "data" / "fos_np_0117486"
TRAIN = DATA / "train.csv"


def _reg(m: str) -> list[str]:
    return [
        "-i",
        str(TRAIN),
        "-u",
        "y",
        "-r",
        "1",
        "-n",
        "sin",
        "-n",
        "exp",
        "-n",
        "log",
        "-v",
        "2400",
        "-c",
        "10",
        "-V1",
        "-l",
        "WARN",
        "-m",
        m,
    ]


def _base(fos: str) -> list[str]:
    """The round 12 BASE flags (``configs/gomea/phase1be/``) with ``--gomea-fos fos``."""
    return [
        "-a", "gomea", "--gomea-fos", fos, "--gomea-pop-size", "dynamic",
        "--hc-fraction-of-nn", "0.05", "--gomea-skip-inactive", "0",
        "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
        "--knob-probe-baseline", "reduced", "--linear-scaling", "1",
        "--const-opt", "lm", "--const-opt-guard", "val",
        "--gomea-accept-complexity-ratio", "0.125", "--gomea-deme-store", "accepted",
        "--const-opt-val", str(TRAIN),
    ]  # fmt: skip


# The Boolean setting of round 12, phase B (no --gomea-fos: the default).
_BOOL = [
    "-H", "mux", "-k", "2", "-r", "5", "-m", "10000", "-a", "gomea",
    "--gomea-pop-size", "dynamic", "--hc-fraction-of-nn", "0.05",
    "--gomea-skip-inactive", "0", "--gomea-max-evals-per-deme", "0",
    "--gomea-genotype-memo", "1", "--knob-probe-baseline", "reduced",
]  # fmt: skip

CASES = {
    "base-info": _reg("1500") + _base("lt-b") + ["-l", "INFO"],
    "base-m600": _reg("600") + _base("lt-b"),
    "base-rt": _reg("1500") + _base("rt"),
    "base-uni": _reg("1500") + _base("uni"),
    "bool-mux": list(_BOOL),
    "bool-mux-rt": _BOOL + ["--gomea-fos", "rt"],
}


CLOCK_KEYS = frozenset({"linkage_s", "variation_s", "timings", "max_time"})
_MAIN = (
    "import sys; from pymoses.moses.main.moses_exec import moses_exec; "
    "sys.exit(moses_exec(sys.argv[1:]))"
)
_STAMP = re.compile(r"^\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d:\d{3}\] ")


def canonical_log(text: str) -> str:
    """The MOSES log without its clock and host values (as in
    ``test_const_opt_reserve_identity.py``)."""
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
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            _MAIN,
            "moses",
            *argv,
            "--gomea-log",
            "events.jsonl",
            "-f",
            "moses.log",
        ],
        cwd=workdir,
        env={**os.environ, "PYTHONPATH": str(REPO)},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    ev = workdir / "events.jsonl"
    return (
        r.stdout,
        canonical_log((workdir / "moses.log").read_text()),
        canonical_events(ev.read_text()) if ev.exists() else "",
    )


@pytest.mark.parametrize("case", sorted(CASES))
def test_existing_fos_modes_are_byte_identical_to_0117486(case, tmp_path):
    stdout, log, events = run_case(CASES[case], tmp_path)
    assert stdout == (REF / f"{case}.out").read_text()
    assert log == (REF / f"{case}.log").read_text()
    assert events == (REF / f"{case}.events.jsonl").read_text()


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
        print(
            name,
            len(s.splitlines()),
            "stdout lines,",
            len(lg.splitlines()),
            "log lines,",
            len(e.splitlines()),
            "events",
        )
