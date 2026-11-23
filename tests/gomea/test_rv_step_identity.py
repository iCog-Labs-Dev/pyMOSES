"""Server round 16 (T3.2): without ``--gomea-rv-step rvg``, the output is byte-identical to the
code before the RV option existed.

The reference files in ``data/rv_step_2d407b2/`` were captured at commit 2d407b2 (before any
code for ``--gomea-rv-step`` and ``--gomea-rv-budget``) by running this module as a script
there::

    PYTHONPATH=<checkout of 2d407b2> .venv/bin/python tests/gomea/test_rv_step_identity.py OUT_DIR

Cases: table regression runs on ``data/const_opt_none/train.csv`` with the fixed regression
setup of ``tools/gomea/moses_sr_runner.py``, and one Boolean run:

* ``base-tls-info``: the round 14 BASE-TLS flags (``configs/gomea/phase1bi/sr.json``) at
  ``-m 1500``, log level INFO (the INFO log holds the command line);
* ``base-tls-m3000-r2``: BASE-TLS at ``-m 3000``, run seed 2;
* ``tls-np``: the round 14 TLS-NP flags (``--gomea-fos np``) at ``-m 1500``;
* ``pop24-converged``: the flags of the T3.2 spike test (a fixed N of 24, ``-c 3``); its
  first deme ends by the genotype convergence test;
* ``hc-gomea``: the hybrid ``-a hc-gomea`` (``run_seeded``) with ``--gomea-contin copy``;
* ``bool-maj3-n4``: ``-H maj -k 3 -a gomea --gomea-pop-size 4`` (a converged deme).

Compared: the full MOSES stdout, the MOSES log (``-f``) and the ``--gomea-log`` file, in the
canonical forms of ``test_round13_lm_identity.py``. The log comparison leaves out the time
stamp of each line, the host name, the path of the test data (replaced by DATA) and the
wall-clock column (``elapsed``) of the "Stats:" rows. The ``--gomea-log`` comparison leaves out
the wall-clock fields (``CLOCK_KEYS``) and replaces the path of the test data by DATA.

Each case runs with the new options absent and at their off values (``--gomea-rv-step off``,
alone and with ``--gomea-rv-budget late``, which ``off`` ignores). Only the command line in
the INFO log names the options. Before the code exists, only the "absent" runs pass (the off
values are unknown flags).

The RV16 review (finding 1 and nit 8): ``--gomea-trace 0`` is one more off value. With
``--gomea-trace 1`` (``-a gomea`` only, so not the case ``hc-gomea``) the search does not
change: stdout and the MOSES log (without the command line) are the references, and the GOMEA
log is the reference after the added fields are removed (``trace`` of each "end" event,
``genotypes_sha256`` of each "init" event, ``trace`` of the parameters). The same holds for a
run with ``--gomea-rv-step rvg``, against the same run without the trace.
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
REF = Path(__file__).resolve().parent / "data" / "rv_step_2d407b2"
TRAIN = DATA / "train.csv"


def _reg(m: str, seed: str = "1", c: str = "10") -> list[str]:
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
        c,
        "-V1",
        "-l",
        "WARN",
        "-m",
        m,
    ]


def _arm(fos: str) -> list[str]:
    """The round 14 BASE-TLS argv (``configs/gomea/phase1bi/sr.json``) with FOS ``fos``."""
    return [
        "-a", "gomea", "--gomea-fos", fos, "--gomea-pop-size", "dynamic",
        "--hc-fraction-of-nn", "0.05", "--gomea-skip-inactive", "0",
        "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
        "--knob-probe-baseline", "reduced", "--linear-scaling", "terms",
        "--const-opt", "none", "--gomea-accept-complexity-ratio", "0.125",
        "--gomea-deme-store", "accepted",
    ]  # fmt: skip


_BASE_TLS = _arm("lt-b")
_TLS_NP = _arm("np")
# The flags of the T3.2 spike test (tests/gomea/test_rv_core.py of commit be83e08).
_POP24 = [
    "-a", "gomea", "--gomea-pop-size", "24", "--gomea-skip-inactive", "0",
    "--gomea-genotype-memo", "1", "--gomea-contin", "copy", "--linear-scaling", "terms",
    "--gomea-deme-store", "accepted", "--gomea-accept-complexity-ratio", "0.125",
]  # fmt: skip
_HC_GOMEA = [
    "-a", "hc-gomea", "--gomea-pop-size", "16", "--gomea-contin", "copy",
    "--linear-scaling", "terms", "--gomea-max-evals-per-deme", "400",
]  # fmt: skip

CASES = {
    "base-tls-info": _reg("1500") + _BASE_TLS + ["-l", "INFO"],
    "base-tls-m3000-r2": _reg("3000", "2") + _BASE_TLS,
    "tls-np": _reg("1500") + _TLS_NP,
    "pop24-converged": _reg("1500", c="3") + _POP24,
    "hc-gomea": _reg("1500") + _HC_GOMEA,
    "bool-maj3-n4": [
        "-H",
        "maj",
        "-k",
        "3",
        "-a",
        "gomea",
        "--gomea-pop-size",
        "4",
        "-m",
        "2000",
        "-r",
        "2",
    ],
}

# Each new option at its off value. Only the command line in the INFO log names it.
OFF = {
    "absent": [],
    "rv-off": ["--gomea-rv-step", "off"],
    "rv-off-late": ["--gomea-rv-step", "off", "--gomea-rv-budget", "late"],
    "trace-off": ["--gomea-trace", "0"],  # the RV16 review
}
TRACE = ["--gomea-trace", "1"]

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


def reasons(events: str) -> list[str]:
    """The termination reason of each deme ("end" events)."""
    return [
        e["termination_reason"] for e in map(json.loads, events.splitlines()) if e["type"] == "end"
    ]


def test_the_references_cover_the_engine_end_rules():
    """The references hold demes that end by the genotype convergence test (the test that the
    RV hook replaces), by stagnation and by the budget."""
    got = {r for case in CASES for r in reasons((REF / f"{case}.events.jsonl").read_text())}
    assert {"converged", "stagnation", "global_budget"} <= got
    assert "converged" in reasons((REF / "pop24-converged.events.jsonl").read_text())


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("off", sorted(OFF))
def test_settings_without_the_rv_option_are_byte_identical_to_2d407b2(case, off, tmp_path):
    extra = OFF[off]
    stdout, log, events = run_case(CASES[case] + extra, tmp_path)
    assert stdout == (REF / f"{case}.out").read_text()
    if extra:
        log = log.replace(" " + " ".join(extra), "")
    assert log == (REF / f"{case}.log").read_text()
    assert events == (REF / f"{case}.events.jsonl").read_text()


def strip_trace(events: str) -> str:
    """The canonical ``--gomea-log`` text of a run with ``--gomea-trace 1`` without the fields
    that the option adds. Each "end" and "init" event must have its field."""
    lines = []
    for line in events.splitlines():
        ev = json.loads(line)
        if ev["type"] == "end":
            trace = ev.pop("trace")
            assert isinstance(trace, list)
            if ev["ledger"]["requests"] > 0:
                assert trace and trace[0][0] >= 1 and trace[-1][0] <= ev["ledger"]["requests"]
        elif ev["type"] == "init":
            digest = ev.pop("genotypes_sha256")
            assert isinstance(digest, str) and len(digest) == 64
        elif ev["type"] == "invocation_start":
            assert ev["params"].pop("trace") is True
        lines.append(json.dumps(ev, sort_keys=True))
    return "\n".join(lines) + "\n" if lines else ""


@pytest.mark.parametrize("case", sorted(c for c in CASES if c != "hc-gomea"))
def test_the_trace_option_adds_only_its_fields(case, tmp_path):
    stdout, log, events = run_case(CASES[case] + TRACE, tmp_path)
    assert stdout == (REF / f"{case}.out").read_text()
    assert log.replace(" " + " ".join(TRACE), "") == (REF / f"{case}.log").read_text()
    assert strip_trace(events) == (REF / f"{case}.events.jsonl").read_text()


RV_CLOCK_KEYS = frozenset({"eval_s", "step_s", "analyse_s"})


def _without_rv_clocks(events: str) -> str:
    """The canonical log without the wall-clock fields of the "rv" records."""
    lines = []
    for line in events.splitlines():
        ev = json.loads(line)
        if isinstance(ev.get("rv"), dict):
            ev["rv"] = {k: v for k, v in ev["rv"].items() if k not in RV_CLOCK_KEYS}
        lines.append(json.dumps(ev, sort_keys=True))
    return "\n".join(lines) + "\n"


def test_the_trace_option_does_not_change_an_rv_run(tmp_path):
    rv = ["--gomea-rv-step", "rvg", "--gomea-rv-budget", "balance"]
    argv = CASES["base-tls-m3000-r2"] + rv
    (tmp_path / "off").mkdir()
    (tmp_path / "on").mkdir()
    off = run_case(argv, tmp_path / "off")
    on = run_case(argv + TRACE, tmp_path / "on")
    assert on[0] == off[0]
    assert on[1].replace(" " + " ".join(TRACE), "") == off[1]
    assert _without_rv_clocks(strip_trace(on[2])) == _without_rv_clocks(off[2])
    assert '"rv": {' in off[2]  # RV ran


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
            "reasons",
            reasons(e),
        )
