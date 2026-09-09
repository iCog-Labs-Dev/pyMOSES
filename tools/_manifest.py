"""Shared manifest schema for tests/fixtures/*.json oracle fixtures.

Generalizes the `_manifest` object S9 introduced for
tests/fixtures/hillclimbing.json (docs/CARRY-FORWARD.md Sec 5: "No fixture
carries a checksum, version, or input manifest ... All six fixtures can rot
silently") so every tools/capture_*.py script builds one the same way, and
tools/check_fixture_manifests.py can validate any fixture's manifest
without caring which script produced it.

Storage: two shapes coexist, both read by tools/check_fixture_manifests.py.

* EMBEDDED -- `tests/fixtures/hillclimbing.json` and
  `tests/fixtures/deme_expander.json` (S9/S10) carry `_manifest` as a
  top-level sibling key inside the fixture itself. That is safe there
  because their own oracle tests already expect `_manifest` in
  `set(golden)`. It is NOT safe to add to most other fixtures: several
  (`tests/fixtures/{boolean_builder,neighborhood,logical_knob,
  field_set,representation}.json`) have oracle tests asserting an exact
  `set(golden) == {...}` WITHOUT `_manifest` (test_build_knobs.py,
  test_neighborhood_sampling.py, test_knobs.py, test_field_set.py,
  test_representation.py) -- adding a new top-level key there would break
  a test this module's own author does not own and must not edit.
* SIDECAR -- every other fixture gets a `<name>.manifest.json` file next
  to it (`write_sidecar` below), leaving the fixture's own bytes and
  shape completely untouched. This is the default for every NEW
  manifest this module writes; only hillclimbing.json/deme_expander.json
  keep the embedded shape they already have.

Fields (schema_version 1) -- see `build_manifest`'s parameters for the
authoritative list; summary:

  schema_version         int, currently always 1.
  capture_script         repo-relative tools/capture_*.py path.
  capture_script_sha256  sha256 of that script's own bytes.
  probe_source           repo-relative tools/cpp_probe/*.cc path, or None.
  probe_source_sha256    sha256 of that probe's bytes, or None to match.
  probe_kind             "cpp_probe" | "classic_cli" | "python_only".
  cli_tool               prebuilt classic binary used, when probe_kind is
                          "classic_cli" (e.g. "bin/combo-fmt-converter");
                          else None.
  repo_commit            git commit this repo was at when captured, or
                          None if unavailable.
  classic_dir/prefix_dir informational record of $CLASSIC/$PREFIX as
                          resolved when captured -- not re-checked later.
  generated_at_utc       ISO-8601 UTC timestamp of capture.
  inputs                 free-form dict: seeds/datasets/exemplars/corpus
                          actually used for this run.

Fields added in schema_version 2:

  toolchain              dict, or None when unknown. Records the host C++
                          toolchain that produced the fixture:
                          {"cxx": "g++ (GCC) 16.2.1 ...",
                           "libstdcxx_soname": "libstdc++.so.6.0.36",
                           "glibcxx_max_symbol": "GLIBCXX_3.4.36",
                           "boost_lib_version": "1_92"}.

Why `toolchain` exists (docs/CARRY-FORWARD.md Sec 5.1). Several fixtures
encode behaviour that is a property of the HOST C++ IMPLEMENTATION, not of
classic's source: `crossover.json` depends on libstdc++'s `std::sort`
permutation for equal elements, and `contin_enumeration.json` depends on
boost's `binomial_coefficient<double>` rounding and on GCC's
double->unsigned conversion. Re-capturing any of those on a different
toolchain can legitimately produce different bytes, and until schema 2
nothing recorded which toolchain a fixture came from -- so an upgrade could
leave a fixture silently stale, or make a genuine divergence look like a
port bug.

`toolchain` is NOT re-checked as an equality assertion by
tools/check_fixture_manifests.py, and deliberately so: a g++ patch bump
would otherwise fail every fixture at once, including the many that cannot
possibly depend on the toolchain, and a check that cries wolf gets ignored.
It is validated for SHAPE, reported by `--show-toolchain-drift`, and is
there so that when a fixture DOES start disagreeing, the first question --
"did the toolchain move under us?" -- is answerable from the repo instead
of unanswerable.

`toolchain: null` means genuinely unknown, and is what every fixture
captured before schema 2 carries. It is not backfilled with today's values:
that would be recording a measurement nobody took.

`capture_script_sha256`/`probe_source_sha256` are what
tools/check_fixture_manifests.py recomputes and compares against the
CURRENT tools/capture_*.py and tools/cpp_probe/*.cc bytes to detect rot --
see that script's own docstring.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 2
PROBE_KINDS = ("cpp_probe", "classic_cli", "python_only")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _first_line(command: list[str]) -> str | None:
    """Best-effort single-line command output, or None. Never raises: a
    manifest must still be writable on a host missing any of these tools.
    """
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=10, check=True
        )
    except (subprocess.SubprocessError, OSError):
        return None
    line = result.stdout.splitlines()[0].strip() if result.stdout.strip() else ""
    return line or None


def detect_toolchain() -> dict | None:
    """Record the host C++ toolchain -- see this module's docstring for why.

    Every field is independently best-effort and may be None; the whole
    dict is None only if nothing at all could be determined, which is the
    same "unknown" the pre-schema-2 fixtures carry.
    """
    cxx = _first_line(["g++", "--version"])

    # The libstdc++ the probes actually link against, by resolved soname
    # rather than by package version: it is the soname that changes when
    # the sort/algorithm implementations change.
    libstdcxx_soname = None
    glibcxx_max_symbol = None
    for candidate in (Path("/usr/lib/libstdc++.so.6"), Path("/usr/lib64/libstdc++.so.6")):
        if candidate.exists():
            try:
                libstdcxx_soname = candidate.resolve().name
            except OSError:
                libstdcxx_soname = None
            strings_out = _first_line(
                ["sh", "-c", f"strings {candidate} | grep -o 'GLIBCXX_[0-9.]*' | sort -V | tail -1"]
            )
            glibcxx_max_symbol = strings_out
            break

    boost_lib_version = None
    boost_version_hpp = Path("/usr/include/boost/version.hpp")
    if boost_version_hpp.is_file():
        for line in boost_version_hpp.read_text().splitlines():
            if line.startswith("#define BOOST_LIB_VERSION"):
                boost_lib_version = line.split('"')[1] if '"' in line else None
                break

    toolchain = {
        "cxx": cxx,
        "libstdcxx_soname": libstdcxx_soname,
        "glibcxx_max_symbol": glibcxx_max_symbol,
        "boost_lib_version": boost_lib_version,
    }
    if all(value is None for value in toolchain.values()):
        return None
    return toolchain


def git_commit(repo_root: Path) -> str | None:
    """Best-effort -- returns None (rather than raising) when `git` is
    unavailable or this checkout is not a git repo, so a manifest can
    still be written in either case. Mirrors capture_hillclimbing.py's
    original `_git_commit`.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return result.stdout.strip() or None


def build_manifest(
    *,
    repo_root: Path,
    capture_script: str,
    capture_script_path: Path,
    probe_source: str | None = None,
    probe_source_path: Path | None = None,
    probe_kind: str = "cpp_probe",
    cli_tool: str | None = None,
    classic_dir: Path | str | None = None,
    prefix_dir: Path | str | None = None,
    inputs: dict | None = None,
    generated_at_utc: str | None = None,
    repo_commit: str | None | object = "AUTO",
    toolchain: dict | None | object = "AUTO",
) -> dict:
    """Build a schema-1 fixture manifest dict.

    `repo_commit="AUTO"` (the default) computes it via
    `git_commit(repo_root)` as of *now*; pass an explicit value (including
    `None`) to override -- used when reconstructing a manifest for a
    fixture that was actually captured at a past commit, so the recorded
    commit reflects when the DATA was produced rather than when the
    manifest was backfilled.
    """
    if probe_kind not in PROBE_KINDS:
        raise ValueError(f"probe_kind must be one of {PROBE_KINDS}, got {probe_kind!r}")
    if (probe_source is None) != (probe_source_path is None):
        raise ValueError("probe_source and probe_source_path must both be set or both be None")
    if repo_commit == "AUTO":
        repo_commit = git_commit(repo_root)
    if toolchain == "AUTO":
        toolchain = detect_toolchain()
    return {
        "schema_version": SCHEMA_VERSION,
        "capture_script": capture_script,
        "capture_script_sha256": sha256_of(capture_script_path),
        "probe_source": probe_source,
        "probe_source_sha256": sha256_of(probe_source_path) if probe_source_path else None,
        "probe_kind": probe_kind,
        "cli_tool": cli_tool,
        "repo_commit": repo_commit,
        "classic_dir": str(classic_dir) if classic_dir is not None else None,
        "prefix_dir": str(prefix_dir) if prefix_dir is not None else None,
        "generated_at_utc": generated_at_utc or datetime.now(UTC).isoformat(timespec="seconds"),
        "toolchain": toolchain,
        "inputs": inputs if inputs is not None else {},
    }


def sidecar_path(fixture_path: Path) -> Path:
    """`tests/fixtures/containers.json` -> `tests/fixtures/containers.manifest.json`."""
    return fixture_path.with_suffix(".manifest.json")


def write_sidecar(fixture_path: Path, manifest: dict) -> Path:
    """Write `manifest` as `fixture_path`'s sidecar manifest file (see this
    module's docstring for when to use this vs. embedding `_manifest`
    directly in the fixture). Returns the sidecar path written.
    """
    path = sidecar_path(fixture_path)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
