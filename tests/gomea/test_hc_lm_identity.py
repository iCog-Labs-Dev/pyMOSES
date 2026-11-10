"""Server round 10: without ``--const-opt``, ``-a hc`` is byte-identical to the code before
LM for hill climbing existed.

The reference files in ``data/hc_lm_ec02194/`` were captured at commit ec02194 (before any
code for ``-a hc --const-opt lm``) by running this module as a script there::

    .venv/bin/python tests/gomea/test_hc_lm_identity.py OUT_DIR

Cases: table regression runs of ``-a hc`` on ``data/const_opt_none/train.csv`` with the
fixed regression setup of ``tools/gomea/moses_sr_runner.py`` (plain, with
``--linear-scaling 1`` as the round-8 arm HC-LS, with ``--hc-widen-search 1``, at log level
INFO, and with a ``--gomea-log`` that ``-a hc`` must not write), and one Boolean ``-a hc``
run (``-H pa -k 3``). Compared: the full MOSES stdout, the MOSES log (``-f``) and the
``--gomea-log`` file (absent = empty). The log comparison leaves out the time stamp of each
line, the host name, the wall-clock columns (``elapsed`` of the "Stats:" rows, both
microsecond columns of the "DemesHill:" rows) and the RAM share of the "DemesHill:" rows,
which depends on the machine's memory.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DATA =Path(__file__).resolve().parent / "data" / "const_opt_none"
REF = Path(__file__).resolve().parent / "data" / "hc_lm_ec02194"
TRAIN = DATA / "train.csv"

_REG = [
    "-i", str(TRAIN), "-u", "y", "-r", "1", "-n", "sin", "-n", "exp", "-n", "log",
    "-v", "2400", "-c", "10", "-V1", "-l", "WARN", "-m", "800",
]  # fmt: skip
_HC = ["-a", "hc", "--knob-probe-baseline", "reduced"]
_LS = ["--linear-scaling", "1"]

CASES = {
    "hc": _REG + _HC,
    "hc-ls": _REG + _HC + _LS,
    "hc-ls-widen": _REG + _HC + _LS + ["--hc-widen-search", "1"],
    "hc-ls-info": _REG + _HC + _LS + ["-l", "INFO"],
    "hc-ls-log": _REG + _HC + _LS + ["--gomea-log", "events.jsonl"],
    "bool-hc": ["-H", "pa", "-k", "3", "-r", "5", "-m", "2000", "-a", "hc"],
}

_MAIN = ("import sys; from pymoses.moses.main.moses_exec import moses_exec; "
         "sys.exit(moses_exec(sys.argv[1:]))")
_STAMP = re.compile(r"^\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d:\d{3}\] ")


def canonical_log(text: str) -> str:
    """The MOSES log without its clock, host and RAM values (see the module docstring)."""
    out = []
    for line in text.splitlines():
        line = _STAMP.sub("", line)
        if line.startswith("[INFO] hostname:"):
            continue
        if line.startswith("[INFO] Command line:"):
            line = line.replace(str(DATA), "DATA")
        for tag, cols in (("Stats: ", (2,)), ("DemesHill: ", (4, 6, 9))):
            head, sep, rest = line.partition(tag)
            if sep and not rest.startswith("#"):
                f = rest.split("\t")
                for c in cols:
                    if c < len(f):
                        f[c] = "T"
                line = head + sep + "\t".join(f)
        out.append(line)
    return "\n".join(out) + "\n"


def run_case(argv: list[str], workdir: Path) -> tuple[str, str, str]:
    """(stdout, canonical MOSES log, --gomea-log text) of one run in ``workdir``. Each run
    is a new process: the MOSES logger opens its file once per process."""
    for f in ("moses.log", "events.jsonl"):
        (workdir / f).unlink(missing_ok=True)
    r = subprocess.run(
        [sys.executable, "-c", _MAIN, "moses", *argv, "-f", "moses.log"], cwd=workdir,
        env={**os.environ, "PYTHONPATH": str(REPO)}, capture_output=True, text=True,
        timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    ev = workdir / "events.jsonl"
    return (r.stdout, canonical_log((workdir / "moses.log").read_text()),
            ev.read_text() if ev.exists() else "")


@pytest.mark.parametrize("case", sorted(CASES))
def test_hc_without_const_opt_is_byte_identical_to_ec02194(case, tmp_path):
    stdout, log, events = run_case(CASES[case], tmp_path)
    assert stdout == (REF / f"{case}.out").read_text()
    assert log == (REF / f"{case}.log").read_text()
    assert events == ""


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name in sorted(CASES):
        work = out / f"work-{name}"
        work.mkdir(exist_ok=True)
        s, lg, e = run_case(CASES[name], work)
        assert e == "", name
        (out / f"{name}.out").write_text(s)
        (out / f"{name}.log").write_text(lg)
        print(name, len(s.splitlines()), "stdout lines,", len(lg.splitlines()), "log lines")
