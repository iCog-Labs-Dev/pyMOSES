"""With the knob cap off (``--gomea-knob-cap 0``, the default), every output is
byte-identical to the code before the knob cap existed.

The reference files in ``data/knob_cap_k0/`` were captured at commit 04c2020
(before any knob-cap code) by running this module as a script there::

    .venv/bin/python tests/gomea/test_knob_cap_identity.py OUT_DIR

Each case is a Boolean multiplexer (mux6) or the polynomial regression demo (``-H sr``), through the
round-1 GOMEA arm (``--gomea-fos lt-b``, N 16, cap 250, memo, ``--gomea-contin
copy`` for regression, ``--knob-probe-baseline reduced``) and a second FOS
(``rt`` or ``uni``). Compared: the full MOSES stdout and every ``--gomea-log``
event, with only the wall-clock fields left out (``CLOCK_KEYS``) and the log's
own path replaced by ``LOG``.
"""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "data" / "knob_cap_k0"

# The round-1 GOMEA arm (SERVER-ROUND2-PREREGISTRATION.md, "Base setting").
_BASE = [
    "-a", "gomea", "--gomea-pop-size", "16", "--gomea-skip-inactive", "0",
    "--gomea-max-evals-per-deme", "250", "--gomea-genotype-memo", "1",
    "--knob-probe-baseline", "reduced",
]  # fmt: skip
_MUX = ["-H", "mux", "-k", "2", "-m", "3000", "-r", "3001"]
_REG = ["-H", "sr", "-k", "2", "--gomea-contin", "copy", "-m", "2000", "-r", "1"]

CASES = {
    "mux6-ltb": _MUX + _BASE + ["--gomea-fos", "lt-b"],
    "mux6-rt": _MUX + _BASE + ["--gomea-fos", "rt"],
    "sr-ltb": _REG + _BASE + ["--gomea-fos", "lt-b"],
    "sr-uni": _REG + _BASE + ["--gomea-fos", "uni"],
}

# Wall-clock fields (max_time is the time left, in whole seconds): the only telemetry
# that differs between two runs of the same code.
CLOCK_KEYS = frozenset({"linkage_s", "variation_s", "timings", "max_time"})


def _argv(case: str) -> list[str]:
    return list(CASES[case])


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
            ev["params"]["telemetry_path"] = "LOG"  # the run's own path
        lines.append(json.dumps(ev, sort_keys=True))
    return buf.getvalue(), "\n".join(lines) + "\n"


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("extra", [[], ["--gomea-knob-cap", "0"]], ids=["absent", "k0"])
def test_knob_cap_off_is_byte_identical_to_04c2020(case, extra, tmp_path):
    stdout, events = run_case(_argv(case) + extra, tmp_path)
    assert stdout == (DATA / f"{case}.out").read_text()
    assert events == (DATA / f"{case}.events.jsonl").read_text()


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name in sorted(CASES):
        work = out / f"work-{name}"
        work.mkdir(exist_ok=True)
        s, e = run_case(_argv(name), work)
        (out / f"{name}.out").write_text(s)
        (out / f"{name}.events.jsonl").write_text(e)
        print(name, len(s.splitlines()), "stdout lines,", len(e.splitlines()), "events")
