"""Tests for the differential CLI sweep in ``tools/flag_sweep.py``.

These are regression tests for the *sweep*, not for the port.  Each one pins one
of the ways the first-generation sweep lied about what it had measured:

* it reported ``classic-rejects-all-candidates`` quietly, so an option it never
  managed to run looked the same as an option that agreed;
* it compared two empty outputs and called that agreement;
* it passed gated options without their gate and called the resulting
  no-op agreement ``IDENTICAL``;
* it was assembled from a shell loop over an unquoted variable, which in zsh
  passes one argument instead of many and therefore compares two runs that were
  never actually configured.

None of these tests need the C++ binary; they exercise the harness itself.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _load(name: str):
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fs_spec = _load("flag_sweep_spec")
fs = _load("flag_sweep")


# --------------------------------------------------------------------------
# The spec table itself
# --------------------------------------------------------------------------
def test_every_option_has_at_least_one_value():
    """The old sweep's ladder produced zero legal values for six options.

    A spec entry with an empty value list would reintroduce exactly that: an
    option nothing ever runs.
    """
    empty = [s.flag for s in fs_spec.SPECS if not s.values]
    assert empty == []


def test_option_names_are_unique_and_long_form():
    flags = [s.flag for s in fs_spec.SPECS]
    assert len(flags) == len(set(flags)), "duplicate option in SPECS"
    assert all(f.startswith("--") for f in flags)


def test_census_matches_the_binary_help():
    """`moses --help` lists 139 options; the sweep must know all of them.

    The predecessor was built from a filtered list that dropped every option
    with a short form -- 49 of them -- and nothing noticed.
    """
    assert len(fs_spec.SPECS) == 139


def test_every_spec_names_a_dataset_the_runner_can_build():
    unknown = {s.dataset for s in fs_spec.SPECS} - set(fs.DATASETS)
    assert unknown == set()
    unknown_combo = {c.dataset for c in fs_spec.COMBOS} - set(fs.DATASETS)
    assert unknown_combo == set()


def test_feature_selection_options_run_on_a_table_wide_enough_to_bite():
    """--fs-* on a 2-input table selects both inputs and changes nothing.

    That is how the predecessor scored --fs-focus/--fs-seed/--fs-algo/--fs-scorer
    `IDENTICAL` at the literal value ``1`` while exercising none of them.
    """
    for s in fs_spec.SPECS:
        if s.flag.startswith("--fs-") or s.flag == "--enable-fs":
            assert s.dataset in ("wide", "ts_wide"), s.flag
    assert "--fs-target-size=3" in fs_spec.FS


# --------------------------------------------------------------------------
# Case construction
# --------------------------------------------------------------------------
def test_no_case_passes_an_option_twice():
    """boost rejects a repeated scalar option with SIGABRT.

    A `needs` context that names the option under test (FS carries
    --fs-target-size, PRE carries -q0.1) would manufacture that abort and the
    sweep would report it as a finding.
    """
    for case in fs.build_cases(None) + fs.build_combo_cases(None):
        longs = [t.split("=", 1)[0] for t in case.argv_tail if t.startswith("--")]
        assert len(longs) == len(set(longs)), case
        shorts = [t[:2] for t in case.argv_tail if t.startswith("-") and not t.startswith("--")]
        assert len(shorts) == len(set(shorts)), case


def test_gated_options_carry_their_gate():
    by_flag = {c.flag: c for c in fs.build_cases(None)}
    assert "--enable-fs=1" in by_flag["--fs-algo"].argv_tail
    assert "--boost=1" in by_flag["--boost-promote"].argv_tail
    assert "--timestamp-feature=when" in by_flag["--time-bscore"].argv_tail


def test_a_filtered_sweep_that_selects_nothing_is_an_error(tmp_path):
    rc = fs.main(
        [
            "--classic",
            str(TOOLS / "flag_sweep.py"),  # exists, never executed
            "--out",
            str(tmp_path),
            "--only",
            "no-such-option-anywhere",
        ]
    )
    assert rc == 2, "a sweep with zero cases must not look like a clean run"


# --------------------------------------------------------------------------
# The bound on every C++ run
# --------------------------------------------------------------------------
def test_every_child_is_memory_and_time_bounded():
    argv = fs.bounded(["/bin/echo", "a b", "c"], as_kb=2097152, timeout=300)
    assert argv[0] == "/bin/bash"
    assert "ulimit -v 2097152" in argv[2]
    assert "timeout 300" in argv[2]
    # The mandated `"$@"` array form.  An unquoted $@ or $* here would re-split
    # arguments that contain spaces, silently changing the command under test.
    assert '"$@"' in argv[2]
    assert argv[3:] == ["_", "/bin/echo", "a b", "c"]


def test_bounded_preserves_arguments_containing_spaces():
    import subprocess

    argv = fs.bounded(["/bin/echo", "one two", "three"], as_kb=4194304, timeout=10)
    out = subprocess.run(argv, capture_output=True, text=True, check=True).stdout
    assert out == "one two three\n"


# --------------------------------------------------------------------------
# Classification and the loud paths
# --------------------------------------------------------------------------
def _v(flag, value, kind, crc, prc, fp="x"):
    return fs.Verdict(
        flag=flag,
        value=value,
        context="",
        kind=kind,
        classic_rc=crc,
        port_rc=prc,
        detail="",
        reproducer=f"moses {flag}={value}",
        classic_fingerprint=fp,
    )


def test_no_legal_value_is_detected_and_is_loud(tmp_path, capsys):
    """The exact hole being fixed: classic accepted no value for an option."""
    verdicts = [
        _v("--output-format", "1", "BOTH-FAIL-SAME-RC", 134, 1),
        _v("--output-format", "2", "BOTH-FAIL-SAME-RC", 134, 1),
        _v("--balance", "0", "IDENTICAL", 0, 0, "a"),
        _v("--balance", "1", "IDENTICAL", 0, 0, "b"),
    ]
    dead = fs.no_legal_value(verdicts)
    assert dead == ["--output-format"]

    rc = fs.report(verdicts, dead, tmp_path)
    text = capsys.readouterr().out
    assert rc == 1, "a NO-LEGAL-VALUE option must fail the sweep, not pass quietly"
    assert "NO-LEGAL-VALUE" in text
    assert "--output-format" in text
    assert "!!" in text, "the banner is the whole point; this must not be a one-liner"


def test_an_option_with_one_working_value_is_not_flagged_dead():
    verdicts = [
        _v("--output-format", "combo", "IDENTICAL", 0, 0, "a"),
        _v("--output-format", "bogus", "BOTH-FAIL-SAME-RC", 134, 1, "b"),
    ]
    assert fs.no_legal_value(verdicts) == []


def test_inert_options_are_reported_so_identical_is_not_oversold(tmp_path, capsys):
    verdicts = [
        _v("--cap-coef", "1", "IDENTICAL", 0, 0, "same"),
        _v("--cap-coef", "50", "IDENTICAL", 0, 0, "same"),
        _v("--balance", "0", "IDENTICAL", 0, 0, "a"),
        _v("--balance", "1", "IDENTICAL", 0, 0, "b"),
    ]
    assert fs.inert_flags(verdicts) == ["--cap-coef"]
    fs.report(verdicts, [], tmp_path)
    text = capsys.readouterr().out
    assert "INERT" in text and "--cap-coef" in text
    assert "--balance" not in text.split("INERT", 1)[1]


def test_two_empty_outputs_are_not_counted_as_agreement(tmp_path, capsys):
    verdicts = [_v("--whatever", "0", "BOTH-EMPTY", 0, 0)]
    fs.report(verdicts, [], tmp_path)
    text = capsys.readouterr().out
    assert "BOTH-EMPTY" in text
    assert "IDENTICAL" not in text.splitlines()[3:6]


@pytest.mark.parametrize(
    ("kind", "expect_fail"),
    [
        ("BOTH-OK-DIFFER", True),
        ("CLASSIC-OK-PORT-FAILS", True),
        ("PORT-OK-CLASSIC-FAILS", True),
        ("IDENTICAL", False),
        ("BOTH-FAIL-SAME-RC", False),
    ],
)
def test_report_exit_status_tracks_findings(tmp_path, capsys, kind, expect_fail):
    rc = fs.report([_v("--x", "1", kind, 0, 0)], [], tmp_path)
    capsys.readouterr()
    assert rc == (1 if expect_fail else 0)


def test_report_writes_machine_readable_output(tmp_path, capsys):
    import json

    fs.report([_v("--x", "1", "IDENTICAL", 0, 0)], [], tmp_path)
    capsys.readouterr()
    rows = json.loads((tmp_path / "flag_sweep.json").read_text())
    assert rows[0]["flag"] == "--x"
    assert (tmp_path / "flag_sweep.txt").read_text().startswith("=")


def test_acknowledged_dead_options_are_listed_but_do_not_fail(tmp_path, capsys, monkeypatch):
    """The banner only works if it stays rare enough to be read.

    An option classic genuinely cannot run at any value (--help returns 1;
    every --fs-hc-* inherits the --fs-algo=hc segfault) has to be written down
    once, not rediscovered loudly on every run -- otherwise the banner becomes
    scenery and the next real one is skipped over.
    """
    monkeypatch.setitem(fs.ACKNOWLEDGED_DEAD, "--known", "classic returns 1 by design")
    verdicts = [
        _v("--known", "a", "BOTH-FAIL-SAME-RC", 1, 1),
        _v("--surprise", "a", "BOTH-FAIL-SAME-RC", 134, 1),
    ]
    rc = fs.report(verdicts, ["--known"], tmp_path)
    text = capsys.readouterr().out
    assert rc == 0, "an acknowledged dead option must not fail the sweep"
    assert "acknowledged" in text and "classic returns 1 by design" in text
    assert "!!   --known" not in text

    rc = fs.report(verdicts, ["--known", "--surprise"], tmp_path)
    text = capsys.readouterr().out
    assert rc == 1, "an UNacknowledged dead option must still fail"
    assert "!!   --surprise" in text


def test_every_acknowledged_dead_option_states_a_reason():
    for flag, reason in fs.ACKNOWLEDGED_DEAD.items():
        assert len(reason) > 20, flag
