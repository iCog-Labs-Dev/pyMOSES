"""Server round 12 (T1.6 and T2.3): without ``--const-opt-top-k`` and
``--const-opt-during``, the arms are byte-identical to the code before these options
existed.

The reference files in ``data/const_opt_round12a_0117486/`` were captured at commit 0117486
(before any code for the two options) by running this module as a script there::

    .venv/bin/python tests/gomea/test_const_opt_topk_during_identity.py OUT_DIR

Cases: table regression runs on ``data/const_opt_none/train.csv`` with the fixed regression
setup of ``tools/gomea/moses_sr_runner.py`` and the round 12 flags of BASE (the round 11
BASE flags, ``configs/gomea/phase1be/``; the validation table of the guard is the training
table): at ``-m 1500`` and log level INFO, and at ``-m 600`` (one deme, which ends at the
budget). Also the round 11 flags of HC-LM, the hybrid ``-a hc-gomea --const-opt lm``, one
``-a gomea --const-opt mutate`` run with a deme cap (the engine hook next to the new one)
and one Boolean ``-a gomea`` run (``-H pa -k 3``). Compared: the full MOSES stdout, the
MOSES log (``-f``) and the ``--gomea-log`` file, with the clock values, the host name and
the path of the test data left out as in ``test_const_opt_reserve_identity.py``.
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
REF = Path(__file__).resolve().parent / "data" / "const_opt_round12a_0117486"
TRAIN = DATA / "train.csv"


def _reg(m: str) -> list[str]:
    return ["-i", str(TRAIN), "-u", "y", "-r", "1", "-n", "sin", "-n", "exp", "-n", "log",
            "-v", "2400", "-c", "10", "-V1", "-l", "WARN", "-m", m]


# The round 12 BASE arm (the round 11 BASE flags, configs/gomea/phase1be/), flag for flag.
_BASE = [
    "-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "dynamic",
    "--hc-fraction-of-nn", "0.05", "--gomea-skip-inactive", "0",
    "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
    "--knob-probe-baseline", "reduced", "--linear-scaling", "1",
    "--const-opt", "lm", "--const-opt-guard", "val",
    "--gomea-accept-complexity-ratio", "0.125", "--gomea-deme-store", "accepted",
]  # fmt: skip
_HC_LM = ["-a", "hc", "--knob-probe-baseline", "reduced", "--linear-scaling", "1",
          "--const-opt", "lm", "--const-opt-guard", "val"]
_VAL = ["--const-opt-val", str(TRAIN)]

CASES = {
    "base-info": _reg("1500") + _BASE + _VAL + ["-l", "INFO"],
    "base-m600": _reg("600") + _BASE + _VAL,
    "hc-lm": _reg("1500") + _HC_LM + _VAL,
    "hc-gomea-lm": _reg("1500") + [
        "-a", "hc-gomea", "--gomea-pop-size", "16", "--gomea-contin", "copy",
        "--linear-scaling", "1", "--const-opt", "lm", "--gomea-max-evals-per-deme", "400",
    ],  # fmt: skip
    "gomea-mutate": _reg("1500") + [
        "-a", "gomea", "--gomea-pop-size", "16", "--gomea-contin", "copy",
        "--linear-scaling", "1", "--const-opt", "mutate", "--gomea-max-evals-per-deme", "400",
    ],  # fmt: skip
    "bool-gomea": [
        "-H", "pa", "-k", "3", "-r", "5", "-m", "2000", "-a", "gomea",
        "--gomea-pop-size", "dynamic", "--gomea-genotype-memo", "1",
        "--gomea-deme-store", "accepted", "--gomea-accept-complexity-ratio", "0.125",
    ],  # fmt: skip
}

CLOCK_KEYS = frozenset({"linkage_s", "variation_s", "timings", "max_time"})
_MAIN = ("import sys; from pymoses.moses.main.moses_exec import moses_exec; "
         "sys.exit(moses_exec(sys.argv[1:]))")
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


# Each new option at its off value. Only the command line in the INFO log names it.
OFF = {
    "absent": [],
    "topk-0": ["--const-opt-top-k", "0"],
    "during-none": ["--const-opt-during", "none"],
}


def run_case(argv: list[str], workdir: Path) -> tuple[str, str, str]:
    """(stdout, canonical MOSES log, canonical --gomea-log) of one run in ``workdir``. Each
    run is a new process: the MOSES logger opens its file once per process."""
    for f in ("moses.log", "events.jsonl"):
        (workdir / f).unlink(missing_ok=True)
    r = subprocess.run(
        [sys.executable, "-c", _MAIN, "moses", *argv, "--gomea-log", "events.jsonl",
         "-f", "moses.log"],
        cwd=workdir, env={**os.environ, "PYTHONPATH": str(REPO)}, capture_output=True,
        text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    ev = workdir / "events.jsonl"
    return (r.stdout, canonical_log((workdir / "moses.log").read_text()),
            canonical_events(ev.read_text()) if ev.exists() else "")


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("off", sorted(OFF))
def test_round_12_arms_without_the_options_are_byte_identical_to_0117486(case, off, tmp_path):
    extra = OFF[off]
    stdout, log, events = run_case(CASES[case] + extra, tmp_path)
    assert stdout == (REF / f"{case}.out").read_text()
    if extra:
        log = log.replace(" " + " ".join(extra), "")
    assert log == (REF / f"{case}.log").read_text()
    assert events == (REF / f"{case}.events.jsonl").read_text()


if __name__ == "__main__":
    assert DATA.is_dir()
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
