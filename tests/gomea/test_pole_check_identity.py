"""Server round 14 (NP-LMR-PSR): the existing values of ``--const-opt-pole-check`` (``none``
and ``box``) give byte-identical output after the code for ``box-rel`` was added.

The reference files in ``data/pole_check_dee8101/`` were captured at commit dee8101 (before
any code for ``box-rel``) by running this module as a script there::

    .venv/bin/python tests/gomea/test_pole_check_identity.py OUT_DIR

Cases: table regression runs on ``data/const_opt_none/train.csv`` with the fixed regression
setup of ``tools/gomea/moses_sr_runner.py`` and the round 13 BASE-PS flags of
``test_const_opt_pole.py`` (the round 12 BASE flags, the validation table of the guard is
the training table) at ``-m 3000`` with ``--gomea-max-evals-per-deme 300``. With this deme
cap, LM improves programs in several demes, so the pole checks of ``box`` really run:

* ``none-r1``: ``--const-opt-pole-check none``, run seed 1;
* ``box-r1``: ``box``, run seed 1, log level INFO (the INFO log holds the command line);
  5 checks, 1 rejection;
* ``box-r2``: ``box``, run seed 2; 4 checks, no rejection.

Compared: the full MOSES stdout, the MOSES log (``-f``) and the ``--gomea-log`` file, in
the canonical forms of ``test_round13_lm_identity.py``. The log comparison leaves out the
time stamp of each line, the host name, the path of the test data (replaced by DATA) and
the wall-clock column (``elapsed``) of the "Stats:" rows. The ``--gomea-log`` comparison
leaves out the wall-clock fields (``CLOCK_KEYS``) and replaces the path of the test data by
DATA. The events hold the ``pole``, ``pole_s_old`` and ``pole_s_new`` of each check and
the counts ``lm_pole_checked`` and ``lm_pole_rejected``.
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
REF = Path(__file__).resolve().parent / "data" / "pole_check_dee8101"
TRAIN = DATA / "train.csv"


def _reg(m: str, seed: str, level: str = "WARN") -> list[str]:
    return [
        "-i",
        str(TRAIN),
        "-u",
        "y",
        "-r",
        seed,
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
        level,
        "-m",
        m,
    ]


# The round 12 BASE flags (configs/gomea/phase1bf/), flag for flag, as in
# test_const_opt_pole.py; the validation table of the guard is the training table.
_BASE = [
    "-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "dynamic",
    "--hc-fraction-of-nn", "0.05", "--gomea-skip-inactive", "0",
    "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
    "--knob-probe-baseline", "reduced", "--linear-scaling", "1",
    "--const-opt", "lm", "--const-opt-guard", "val",
    "--gomea-accept-complexity-ratio", "0.125", "--gomea-deme-store", "accepted",
    "--const-opt-val", str(TRAIN), "--gomea-max-evals-per-deme", "300",
]  # fmt: skip


def _pole(mode: str) -> list[str]:
    return ["--const-opt-pole-check", mode]


CASES = {
    "none-r1": _reg("3000", "1") + _BASE + _pole("none"),
    "box-r1": _reg("3000", "1", "INFO") + _BASE + _pole("box"),
    "box-r2": _reg("3000", "2") + _BASE + _pole("box"),
}

# The pole counts (checked, rejected) summed over the "end" events of each reference.
POLE_COUNTS = {"none-r1": None, "box-r1": (5, 1), "box-r2": (4, 0)}

CLOCK_KEYS = frozenset({"linkage_s", "variation_s", "timings", "max_time"})
_MAIN = (
    "import sys; from pymoses.moses.main.moses_exec import moses_exec; "
    "sys.exit(moses_exec(sys.argv[1:]))"
)
_STAMP = re.compile(r"^\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d:\d{3}\] ")


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
        timeout=900,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    ev = workdir / "events.jsonl"
    return (
        r.stdout,
        canonical_log((workdir / "moses.log").read_text()),
        canonical_events(ev.read_text()) if ev.exists() else "",
    )


def pole_counts(events: str) -> tuple[int, int] | None:
    """(checked, rejected) summed over the "end" events, or None without the counts."""
    ends = [e for e in map(json.loads, events.splitlines()) if e["type"] == "end"]
    if not any("lm_pole_checked" in e["const_opt"] for e in ends):
        return None
    return (
        sum(e["const_opt"]["lm_pole_checked"] for e in ends),
        sum(e["const_opt"]["lm_pole_rejected"] for e in ends),
    )


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_references_hold_real_pole_checks(case):
    """The ``box`` references hold checks (and ``box-r1`` a rejection), so the comparison
    covers the check, its verdicts and its write-back rule."""
    assert pole_counts((REF / f"{case}.events.jsonl").read_text()) == POLE_COUNTS[case]


@pytest.mark.parametrize("case", sorted(CASES))
def test_pole_check_none_and_box_are_byte_identical_to_dee8101(case, tmp_path):
    stdout, log, events = run_case(CASES[case], tmp_path)
    assert stdout == (REF / f"{case}.out").read_text()
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
        print(
            name,
            len(s.splitlines()),
            "stdout lines,",
            len(lg.splitlines()),
            "log lines,",
            len(e.splitlines()),
            "events,",
            "pole counts",
            pole_counts(e),
        )
