"""Shared environment resolution for tools/capture_*.py scripts.

Closes the half of docs/CARRY-FORWARD.md Sec 5 ("`ci.sh` cannot rebuild the
probes") that is actually fixable without a compiler: before this, every
capture script that needed the classic MOSES checkout, the installed
moses-legacy prefix, or a raw CogUtil source checkout had that path baked
in as a literal, single-developer absolute path
(`/home/yab/Documents/...`). That is what made every one of them
un-runnable on any other machine (including a CI runner, if one ever grew
a C++ toolchain) regardless of whether the toolchain was present.

Every capture script now resolves those three locations through the
functions below instead. Each is REQUIRED via an environment variable --
there is no machine-specific fallback -- and raises
`MissingCaptureEnvironmentError` (a `RuntimeError` subclass) with an
actionable message (which variable, what it must point at, an example
`export`) if the variable is unset or does not name a real directory.

This module is imported only by developer-run `tools/capture_*.py`
scripts, never by `tools/ci.sh` or the test suite (see that script's own
docstring and docs/CARRY-FORWARD.md Sec 5: CI must not require a C++
toolchain) -- so requiring these variables here adds no dependency to CI.
Every function is a plain call, resolved when invoked, not a module-level
constant -- so merely importing this module (or a capture script that
imports it) never raises; only actually needing an unset path does.
"""

import os
from pathlib import Path


class MissingCaptureEnvironmentError(RuntimeError):
    """An environment variable a capture script needs is unset, empty, or
    does not name an existing directory."""


def _require_dir(env_var: str, purpose: str) -> Path:
    value = os.environ.get(env_var, "").strip()
    if not value:
        raise MissingCaptureEnvironmentError(
            f"{env_var} is not set -- {purpose}\n"
            f"Set it and re-run, e.g.: export {env_var}=/path/to/checkout"
        )
    path = Path(value)
    if not path.is_dir():
        raise MissingCaptureEnvironmentError(
            f"{env_var}={value!r} does not name an existing directory -- {purpose}"
        )
    return path


def classic_dir() -> Path:
    """The classic OpenCog MOSES checkout (`$CLASSIC` in CLAUDE.md)."""
    return _require_dir(
        "PYMOSES_CLASSIC_DIR",
        "it must point at the classic OpenCog MOSES checkout ($CLASSIC).",
    )


def moses_prefix() -> Path:
    """The installed moses-legacy prefix -- `$CLASSIC`'s own
    `scripts/rebuild-moses-local.sh` `CMAKE_INSTALL_PREFIX` (see
    CLAUDE.md's Build section); default there is `$HOME/opt/moses-legacy`,
    but this function does not assume that or any other default.
    """
    return _require_dir(
        "PYMOSES_MOSES_PREFIX",
        "it must point at the installed moses-legacy prefix produced by "
        "$CLASSIC/scripts/rebuild-moses-local.sh.",
    )


def cogutil_dir() -> Path:
    """A raw CogUtil source checkout (`$COGUTIL` in CLAUDE.md) -- needed
    only by probes that `#include <opencog/util/...>` straight from a
    header-only source tree rather than via the installed prefix (e.g.
    `tree_probe.cc`, which deliberately avoids linking libcogutil at all
    -- see that probe's own header comment).
    """
    return _require_dir(
        "PYMOSES_COGUTIL_DIR",
        "it must point at a raw CogUtil source checkout ($COGUTIL).",
    )
