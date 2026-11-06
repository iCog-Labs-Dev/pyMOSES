"""With a fixed GOMEA population (``--gomea-pop-size 16``, or the option not set), every
output is byte-identical to the code before ``--gomea-pop-size dynamic`` existed
(server round 6, ``docs/gomea/SERVER-ROUND6-PREREGISTRATION.md``, "Identity").

The reference files in ``data/pop_size_fixed/`` were captured at commit 85c734a (before any
dynamic-population code) by running this module as a script there::

    PYTHONPATH=<checkout of 85c734a> .venv/bin/python tests/gomea/test_pop_size_identity.py OUT_DIR

Cases: table regression on ``data/const_opt_none/train.csv`` with the round 6 regression base
flags (LT-B and RT, no deme cap), mux6 with the round 6 Boolean base flags, the hybrid
``-a hc-gomea`` (``run_seeded``) and ``--gomea-pop-fraction``. Compared: the full MOSES stdout
and every ``--gomea-log`` event, with only the wall-clock fields left out (``CLOCK_KEYS``) and
the log's own path replaced by ``LOG``.
"""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "pop_size_fixed"
TRAIN = HERE / "data" / "const_opt_none" / "train.csv"

_REG = [
    "-i", str(TRAIN), "-u", "y", "-r", "1", "-n", "sin", "-n", "exp", "-n", "log",
    "-v", "2400", "-c", "10", "-V1", "-l", "WARN", "-m", "800",
]  # fmt: skip
_REG_BASE = [
    "-a", "gomea", "--gomea-skip-inactive", "0", "--gomea-genotype-memo", "1",
    "--gomea-contin", "copy", "--knob-probe-baseline", "reduced", "--linear-scaling", "1",
]  # fmt: skip
_MUX = ["-H", "mux", "-k", "2", "-r", "3", "-m", "1500", "-l", "WARN"]
_MUX_BASE = [
    "-a", "gomea", "--gomea-fos", "lt-b", "--gomea-skip-inactive", "0",
    "--gomea-max-evals-per-deme", "0", "--gomea-genotype-memo", "1",
    "--knob-probe-baseline", "reduced",
]  # fmt: skip

CASES = {
    "reg-ltb": _REG + _REG_BASE + ["--gomea-fos", "lt-b"],
    "reg-rt": _REG + _REG_BASE + ["--gomea-fos", "rt"],
    "mux6": _MUX + _MUX_BASE,
    "hc-gomea": _REG
    + ["-a", "hc-gomea", "--gomea-max-evals-per-deme", "250", "--gomea-contin", "copy"],
    "pop-fraction": _REG
    + _REG_BASE
    + ["--gomea-max-evals-per-deme", "250", "--gomea-pop-fraction", "0.1"],
}

CLOCK_KEYS = frozenset({"linkage_s", "variation_s", "timings", "max_time"})


def run_case(argv: list[str], workdir: Path) -> tuple[str, str]:
    """(stdout, canonical telemetry) of one in-process ``moses`` run in ``workdir``."""
    from pymoses.moses.main.moses_exec import moses_exec

    log = workdir / "events.jsonl"
    if log.exists():
        log.unlink()
    old = os.getcwd()
    os.chdir(workdir)
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = moses_exec(["moses", *argv, "--gomea-log", str(log)])
    finally:
        os.chdir(old)
    assert rc == 0
    lines = []
    for line in log.read_text().splitlines():
        ev = {k: v for k, v in json.loads(line).items() if k not in CLOCK_KEYS}
        if "telemetry_path" in ev.get("params", {}):
            ev["params"]["telemetry_path"] = "LOG"
        lines.append(json.dumps(ev, sort_keys=True))
    return buf.getvalue(), "\n".join(lines) + "\n"


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("extra", [[], ["--gomea-pop-size", "16"]], ids=["absent", "16"])
def test_fixed_pop_size_is_byte_identical_to_85c734a(case, extra, tmp_path):
    stdout, events = run_case(CASES[case] + extra, tmp_path)
    assert stdout == (DATA / f"{case}.out").read_text()
    assert events == (DATA / f"{case}.events.jsonl").read_text()


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    import pymoses

    print("pymoses from", pymoses.__file__)
    for name in sorted(CASES):
        work = out / f"work-{name}"
        work.mkdir(exist_ok=True)
        s, e = run_case(CASES[name], work)
        (out / f"{name}.out").write_text(s)
        (out / f"{name}.events.jsonl").write_text(e)
        print(name, len(s.splitlines()), "stdout lines,", len(e.splitlines()), "events")
