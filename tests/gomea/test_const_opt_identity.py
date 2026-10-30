"""With constant tuning off (``--const-opt none``, the default), every output is
byte-identical to the code before constant tuning existed.

The reference files in ``data/const_opt_none/`` were captured at commit 2dfe388
(before any constant-tuning code) by running this module as a script there::

    .venv/bin/python tests/gomea/test_const_opt_identity.py OUT_DIR

Each case is a table regression run on ``data/const_opt_none/train.csv`` (40 rows,
y = 2.5 sin(1.3 x1) + 0.5 x2) with the fixed regression setup of
``tools/gomea/moses_sr_runner.py``, through the round-1 GOMEA arm (``--gomea-contin
copy``) and the capped hill climber (``-a hc-capped``), each with and without
``--linear-scaling 1``. Compared: the full MOSES stdout and every ``--gomea-log``
event (GOMEA only), with only the wall-clock fields left out (``CLOCK_KEYS``) and the
log's own path replaced by ``LOG``.
"""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "data" / "const_opt_none"
TRAIN = DATA / "train.csv"

_REG = [
    "-i", str(TRAIN), "-u", "y", "-r", "1", "-n", "sin", "-n", "exp", "-n", "log",
    "-v", "2400", "-c", "10", "-V1", "-l", "WARN", "-m", "800",
]  # fmt: skip
_GOMEA = [
    "-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "16",
    "--gomea-skip-inactive", "0", "--gomea-max-evals-per-deme", "250",
    "--gomea-genotype-memo", "1", "--gomea-contin", "copy", "--knob-probe-baseline", "reduced",
]  # fmt: skip
_HC = ["-a", "hc-capped", "--gomea-max-evals-per-deme", "250", "--knob-probe-baseline", "reduced"]
_LS = ["--linear-scaling", "1"]

CASES = {
    "gomea": _REG + _GOMEA,
    "gomea-ls": _REG + _GOMEA + _LS,
    "hc": _REG + _HC,
    "hc-ls": _REG + _HC + _LS,
}

CLOCK_KEYS = frozenset({"linkage_s", "variation_s", "timings", "max_time"})


def run_case(argv: list[str], workdir: Path) -> tuple[str, str]:
    """(stdout, canonical telemetry) of one in-process ``moses`` run in ``workdir``."""
    from pymoses.moses.main.moses_exec import moses_exec

    log = workdir / "events.jsonl"
    if log.exists():
        log.unlink()
    gomea = "gomea" in argv
    old = os.getcwd()
    os.chdir(workdir)
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = moses_exec(["moses", *argv, *(["--gomea-log", str(log)] if gomea else [])])
    finally:
        os.chdir(old)
    assert rc == 0
    lines = []
    if gomea:
        for line in log.read_text().splitlines():
            ev = {k: v for k, v in json.loads(line).items() if k not in CLOCK_KEYS}
            if "telemetry_path" in ev.get("params", {}):
                ev["params"]["telemetry_path"] = "LOG"
            lines.append(json.dumps(ev, sort_keys=True))
    return buf.getvalue(), "\n".join(lines) + "\n"


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("extra", [[], ["--const-opt", "none"]], ids=["absent", "none"])
def test_const_opt_none_is_byte_identical_to_2dfe388(case, extra, tmp_path):
    stdout, events = run_case(CASES[case] + extra, tmp_path)
    assert stdout == (DATA / f"{case}.out").read_text()
    assert events == (DATA / f"{case}.events.jsonl").read_text()


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name in sorted(CASES):
        work = out / f"work-{name}"
        work.mkdir(exist_ok=True)
        s, e = run_case(CASES[name], work)
        (out / f"{name}.out").write_text(s)
        (out / f"{name}.events.jsonl").write_text(e)
        print(name, len(s.splitlines()), "stdout lines,", len(e.splitlines()), "events")
