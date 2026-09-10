"""The port's installed command names and `python -m` entry points.

`pymetta` had no `[project.scripts]` and no `__main__.py`, so the only way
to run it was to hand-write a wrapper that called `moses_exec(sys.argv)`.
These pin the three entry points that replaced that.

Names mirror classic's, where the CMake TARGET name differs from the
installed BINARY name (CLAUDE.md, "CMake target names do not match binary
names"): target `moses-exec` installs as `moses`, target `fs-main` as
`feature-selection`.

The property that actually matters here is that `sys.argv` is passed
WHOLE. `moses_exec` reads the program name from `argv[0]`, as every C++
overload does, and `-F`/`determine_log_name` names its log after it.
Dropping `argv[0]` shifts every option by one -- a mistake made twice
while driving this port from a wrapper, and the reason for the
`argv[0]`-preserving test below.
"""

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_DATASET = _REPO / "tests" / "fixtures" / "table_data" / "contin_regression_sample.csv"

_EXPECTED_SCRIPTS = {
    "moses": "pymetta.moses.main.moses_exec:_console_main",
    "feature-selection": "pymetta.feature_selection.main.fs_main:_console_main",
    "eval-features": "pymetta.feature_selection.main.eval_features:_console_main",
}


def test_pyproject_declares_the_three_console_scripts():
    data = tomllib.loads((_REPO / "pyproject.toml").read_text())
    assert data["project"]["scripts"] == _EXPECTED_SCRIPTS


@pytest.mark.parametrize("target", sorted(_EXPECTED_SCRIPTS.values()))
def test_every_declared_console_entry_point_is_importable(target):
    """A `[project.scripts]` target that does not resolve fails only at
    install time, which no test here would otherwise reach.
    """
    module_path, _, func = target.partition(":")
    module = __import__(module_path, fromlist=[func])
    assert callable(getattr(module, func))


def test_python_m_runs_moses_end_to_end():
    """`python -m pymetta.moses.main` on a contin table -- the same
    invocation shape a user would type, through the real module entry
    rather than an in-process call.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "ignore",
            "-m",
            "pymetta.moses.main",
            f"-i{_DATASET}",
            "-u",
            "y",
            "-m300",
            "-W1",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(_REPO),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.startswith("-0.122694917023181915 0 \n")


def test_console_main_passes_argv0_through(monkeypatch):
    """`_console_main` must hand `moses_exec` the WHOLE `sys.argv`.

    Asserted by capturing what `moses_exec` receives, because the failure
    mode is silent: drop `argv[0]` and every option binds one position
    early, which looks like a parsing bug anywhere but here.
    """
    import pymetta.moses.main.moses_exec as me

    seen = []
    monkeypatch.setattr(me, "moses_exec", lambda argv: seen.append(list(argv)) or 0)
    monkeypatch.setattr(sys, "argv", ["moses", "-i", "x.csv", "-m300"])

    assert me._console_main() == 0
    assert seen == [["moses", "-i", "x.csv", "-m300"]]
