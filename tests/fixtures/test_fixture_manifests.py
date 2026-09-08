"""Oracle-fixture manifest coverage: every tests/fixtures/*.json carries a
present, internally consistent manifest, and tools/check_fixture_manifests.py
actually detects the rot it exists to detect.

Context (docs/CARRY-FORWARD.md Sec 5): "No fixture carries a checksum,
version, or input manifest, and `ci.sh` cannot rebuild the probes. All six
fixtures can rot silently." S9 gave `tests/fixtures/hillclimbing.json` a
`_manifest`; S10 mirrored it for `tests/fixtures/deme_expander.json`. This
module is the rest of that debt closing: every OTHER fixture now carries
the same schema (tools/_manifest.py) as a `<name>.manifest.json` sidecar
(see that module's docstring for why sidecar rather than embedded, for
every fixture but those first two), and `tools/check_fixture_manifests.py`
validates all of them uniformly.

Two halves, deliberately kept apart:

* `Test*RealRepo*` -- integration tests against the ACTUAL committed
  fixtures. These prove today's debt is actually closed: every real
  fixture has a manifest, and it is consistent with the real probe/script
  bytes sitting next to it in this repo.
* `Test*CheckerDetects*` -- unit tests against SYNTHETIC files under
  `tmp_path`. These prove the checker itself has teeth: this project's
  own testing bar (CLAUDE.md-adjacent task brief) is "a rot-detector that
  cannot detect rot is exactly [a 'correct but nothing tests it']
  failure" -- a validator proven only in the passing direction is not
  proven at all. Every failure mode `validate_manifest`/`load_manifest`
  claims to catch gets its own test here that PROVES it fails before
  trusting that it passes elsewhere.

This module deliberately does not need a C++ toolchain, the classic MOSES
checkout, or the installed moses-legacy prefix -- see
tools/check_fixture_manifests.py's own docstring for why (mirrors
tools/ci.sh's own constraint).
"""

import hashlib
import json
from pathlib import Path

import pytest

from tools.check_fixture_manifests import (
    FIXTURE_DIR,
    check_all_fixtures,
    discover_fixtures,
    load_manifest,
    validate_manifest,
)

# ---------------------------------------------------------------------
# Real-repo integration tests
# ---------------------------------------------------------------------

_REAL_FIXTURES = discover_fixtures(FIXTURE_DIR)

# The debt this task closes: growing past "all six fixtures can rot
# silently" (CARRY-FORWARD.md Sec 5) means there must be MORE than six by
# now, and every single one of them must be covered below -- a
# `discover_fixtures` bug that silently found fewer fixtures than exist
# on disk would otherwise pass unnoticed.
_MIN_EXPECTED_FIXTURE_COUNT = 17


def test_discovers_at_least_the_expected_fixtures():
    assert len(_REAL_FIXTURES) >= _MIN_EXPECTED_FIXTURE_COUNT, (
        f"expected at least {_MIN_EXPECTED_FIXTURE_COUNT} fixtures under {FIXTURE_DIR}, "
        f"found {len(_REAL_FIXTURES)}: {[p.name for p in _REAL_FIXTURES]}"
    )


@pytest.mark.parametrize("fixture_path", _REAL_FIXTURES, ids=lambda p: p.name)
def test_every_real_fixture_has_a_present_consistent_manifest(fixture_path):
    manifest, load_error = load_manifest(fixture_path)
    assert load_error is None, f"{fixture_path.name}: {load_error}"
    errors = validate_manifest(manifest)
    assert errors == [], f"{fixture_path.name}: {errors}"


def test_check_all_fixtures_reports_zero_failures_on_the_real_repo():
    """The same aggregate check `tools/ci.sh` runs -- kept as its own
    test (rather than relying solely on the per-fixture parametrization
    above) so a bug in `check_all_fixtures` itself (e.g. silently
    swallowing an entry) cannot hide behind individually-passing
    per-fixture tests.
    """
    failures = check_all_fixtures()
    assert failures == {}


def test_embedded_and_sidecar_shapes_both_actually_occur():
    """Guards the two-shapes-coexist design itself (tools/_manifest.py's
    docstring): if every fixture ever migrated to one shape, the other
    branch of `load_manifest` would go untested by the parametrized test
    above (it would still be CALLED, just never on data that exercises
    that branch) -- pin down that both are live.
    """
    embedded, sidecar = [], []
    for fixture_path in _REAL_FIXTURES:
        data = json.loads(fixture_path.read_text())
        if isinstance(data, dict) and "_manifest" in data:
            embedded.append(fixture_path.name)
        else:
            sidecar.append(fixture_path.name)
    assert embedded, "expected at least one embedded-manifest fixture (hillclimbing/deme_expander)"
    assert sidecar, "expected at least one sidecar-manifest fixture"
    assert set(embedded) == {"hillclimbing.json", "deme_expander.json"}


# ---------------------------------------------------------------------
# Checker-detects-rot unit tests (synthetic files, tmp_path only)
# ---------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A minimal synthetic "repo" with a capture script and a probe file
    under it, mirroring the real layout (tools/capture_x.py,
    tools/cpp_probe/x_probe.cc) closely enough for path-existence checks
    to behave the same way. Returns (repo_root, script_path, probe_path).
    """
    repo_root = tmp_path / "repo"
    (repo_root / "tools" / "cpp_probe").mkdir(parents=True)
    script_path = repo_root / "tools" / "capture_x.py"
    probe_path = repo_root / "tools" / "cpp_probe" / "x_probe.cc"
    script_path.write_text("# a synthetic capture script\n")
    probe_path.write_text("// a synthetic probe\nint main() { return 0; }\n")
    return repo_root, script_path, probe_path


def _valid_manifest(script_path: Path, probe_path: Path) -> dict:
    return {
        "schema_version": 1,
        "capture_script": "tools/capture_x.py",
        "capture_script_sha256": _sha256(script_path),
        "probe_source": "tools/cpp_probe/x_probe.cc",
        "probe_source_sha256": _sha256(probe_path),
        "probe_kind": "cpp_probe",
        "cli_tool": None,
        "repo_commit": "a" * 40,
        "classic_dir": "/some/classic",
        "prefix_dir": "/some/prefix",
        "generated_at_utc": "2026-09-08T00:00:00+00:00",
        "inputs": {"seed": 42},
    }


def test_a_valid_manifest_passes(tmp_path):
    """Baseline: the happy path must actually pass, or every failing-path
    test below would be meaningless (a validator that rejects everything
    would "detect" every perturbation trivially).
    """
    repo_root, script_path, probe_path = _make_repo(tmp_path)
    manifest = _valid_manifest(script_path, probe_path)
    assert validate_manifest(manifest, repo_root=repo_root) == []


def test_missing_manifest_is_detected(tmp_path):
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    fixture_path = fixture_dir / "orphan.json"
    fixture_path.write_text(json.dumps({"some": "data"}))
    # no _manifest key, no orphan.manifest.json sidecar
    manifest, error = load_manifest(fixture_path)
    assert manifest is None
    assert "no manifest found" in error


def test_sidecar_manifest_is_found(tmp_path):
    repo_root, script_path, probe_path = _make_repo(tmp_path)
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    fixture_path = fixture_dir / "thing.json"
    fixture_path.write_text(json.dumps({"some": "data"}))
    manifest = _valid_manifest(script_path, probe_path)
    (fixture_dir / "thing.manifest.json").write_text(json.dumps(manifest))

    loaded, error = load_manifest(fixture_path)
    assert error is None
    assert loaded == manifest
    assert validate_manifest(loaded, repo_root=repo_root) == []


def test_embedded_manifest_is_found(tmp_path):
    repo_root, script_path, probe_path = _make_repo(tmp_path)
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    fixture_path = fixture_dir / "thing.json"
    manifest = _valid_manifest(script_path, probe_path)
    fixture_path.write_text(json.dumps({"data": [1, 2, 3], "_manifest": manifest}))

    loaded, error = load_manifest(fixture_path)
    assert error is None
    assert loaded == manifest


def test_probe_source_hash_mismatch_is_detected_then_a_revert_passes_again(tmp_path):
    """THE demonstration this task's testing bar asks for, made
    permanent: perturb the probe (simulating an edit that never re-ran
    the capture script), show the checker fails with a specific,
    attributable error; revert; show it passes again.
    """
    repo_root, script_path, probe_path = _make_repo(tmp_path)
    manifest = _valid_manifest(script_path, probe_path)
    assert validate_manifest(manifest, repo_root=repo_root) == []

    original_probe_bytes = probe_path.read_bytes()
    probe_path.write_text("// PERTURBED: this probe changed after the fixture was captured\n")

    errors = validate_manifest(manifest, repo_root=repo_root)
    assert len(errors) == 1
    assert "probe_source_sha256 mismatch" in errors[0]
    assert "rot" in errors[0]

    probe_path.write_bytes(original_probe_bytes)
    assert validate_manifest(manifest, repo_root=repo_root) == []


def test_capture_script_hash_mismatch_is_detected(tmp_path):
    repo_root, script_path, probe_path = _make_repo(tmp_path)
    manifest = _valid_manifest(script_path, probe_path)
    script_path.write_text("# PERTURBED: this script changed after the fixture was captured\n")

    errors = validate_manifest(manifest, repo_root=repo_root)
    assert any("capture_script_sha256 mismatch" in e for e in errors)


def test_recorded_hash_perturbed_directly_is_also_detected(tmp_path):
    """The mirror image of the above: the FILES are untouched but the
    MANIFEST's own recorded hash is wrong (e.g. a hand-edit, or a bug in
    a future capture script) -- must fail exactly the same way.
    """
    repo_root, script_path, probe_path = _make_repo(tmp_path)
    manifest = _valid_manifest(script_path, probe_path)
    manifest["probe_source_sha256"] = "0" * 64

    errors = validate_manifest(manifest, repo_root=repo_root)
    assert any("probe_source_sha256 mismatch" in e for e in errors)


def test_missing_capture_script_file_is_detected(tmp_path):
    repo_root, script_path, probe_path = _make_repo(tmp_path)
    manifest = _valid_manifest(script_path, probe_path)
    manifest["capture_script"] = "tools/does_not_exist.py"

    errors = validate_manifest(manifest, repo_root=repo_root)
    assert any("does not exist" in e for e in errors)


def test_missing_probe_source_file_is_detected(tmp_path):
    repo_root, script_path, probe_path = _make_repo(tmp_path)
    manifest = _valid_manifest(script_path, probe_path)
    manifest["probe_source"] = "tools/cpp_probe/does_not_exist.cc"

    errors = validate_manifest(manifest, repo_root=repo_root)
    assert any("does not exist" in e for e in errors)


def test_unsupported_schema_version_is_detected(tmp_path):
    repo_root, script_path, probe_path = _make_repo(tmp_path)
    manifest = _valid_manifest(script_path, probe_path)
    manifest["schema_version"] = 999

    errors = validate_manifest(manifest, repo_root=repo_root)
    assert any("schema_version" in e for e in errors)


@pytest.mark.parametrize(
    "field",
    ["capture_script_sha256", "probe_source_sha256"],
)
def test_malformed_sha256_is_detected(tmp_path, field):
    repo_root, script_path, probe_path = _make_repo(tmp_path)
    manifest = _valid_manifest(script_path, probe_path)
    manifest[field] = "not-a-valid-hash"

    errors = validate_manifest(manifest, repo_root=repo_root)
    assert any("sha256" in e for e in errors)


def test_probe_kind_cpp_probe_requires_probe_source(tmp_path):
    repo_root, script_path, probe_path = _make_repo(tmp_path)
    manifest = _valid_manifest(script_path, probe_path)
    manifest["probe_source"] = None
    manifest["probe_source_sha256"] = None

    errors = validate_manifest(manifest, repo_root=repo_root)
    assert any('probe_kind is "cpp_probe"' in e for e in errors)


def test_probe_kind_classic_cli_requires_cli_tool_and_forbids_probe_source(tmp_path):
    repo_root, script_path, probe_path = _make_repo(tmp_path)
    manifest = _valid_manifest(script_path, probe_path)
    manifest["probe_kind"] = "classic_cli"
    # probe_source is still set from _valid_manifest -- both problems at once.
    errors = validate_manifest(manifest, repo_root=repo_root)
    assert any("cli_tool" in e for e in errors)
    assert any('probe_source is set' in e for e in errors)


def test_bad_repo_commit_format_is_detected(tmp_path):
    repo_root, script_path, probe_path = _make_repo(tmp_path)
    manifest = _valid_manifest(script_path, probe_path)
    manifest["repo_commit"] = "too-short"

    errors = validate_manifest(manifest, repo_root=repo_root)
    assert any("repo_commit" in e for e in errors)


def test_bad_generated_at_utc_is_detected(tmp_path):
    repo_root, script_path, probe_path = _make_repo(tmp_path)
    manifest = _valid_manifest(script_path, probe_path)
    manifest["generated_at_utc"] = "not a timestamp"

    errors = validate_manifest(manifest, repo_root=repo_root)
    assert any("generated_at_utc" in e for e in errors)


def test_non_dict_inputs_is_detected(tmp_path):
    repo_root, script_path, probe_path = _make_repo(tmp_path)
    manifest = _valid_manifest(script_path, probe_path)
    manifest["inputs"] = "not a dict"

    errors = validate_manifest(manifest, repo_root=repo_root)
    assert any("inputs" in e for e in errors)


def test_check_all_fixtures_aggregates_a_missing_manifest_as_a_failure(tmp_path):
    """End-to-end through `check_all_fixtures` (not just `validate_manifest`
    in isolation): a fixture directory containing one good fixture and one
    with no manifest at all reports exactly the bad one.
    """
    repo_root, script_path, probe_path = _make_repo(tmp_path)
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()

    good = fixture_dir / "good.json"
    good.write_text(json.dumps({"_manifest": _valid_manifest(script_path, probe_path)}))

    bad = fixture_dir / "bad.json"
    bad.write_text(json.dumps({"no": "manifest here"}))

    failures = check_all_fixtures(fixture_dir=fixture_dir, repo_root=repo_root)
    assert set(failures) == {"bad.json"}


def test_discover_fixtures_excludes_sidecar_manifest_files(tmp_path):
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    (fixture_dir / "thing.json").write_text("{}")
    (fixture_dir / "thing.manifest.json").write_text("{}")

    found = discover_fixtures(fixture_dir)
    assert [p.name for p in found] == ["thing.json"]
