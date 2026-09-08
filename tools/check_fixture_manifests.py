"""Verify every tests/fixtures/*.json oracle fixture carries a manifest,
and that the manifest is internally consistent with the repo it ships in.

Closes the detection half of docs/CARRY-FORWARD.md Sec 5: "No fixture
carries a checksum, version, or input manifest, and `ci.sh` cannot rebuild
the probes. All six fixtures can rot silently." (The fixture count has
since grown well past six -- this check covers every one of them, found by
globbing `tests/fixtures/*.json` rather than a hardcoded list, so a
fixture added later is covered automatically instead of needing this
script edited.)

This is deliberately a PURE-PYTHON, READ-ONLY check over already-committed
files: it never invokes a compiler, never needs the classic MOSES/CogUtil
checkouts, and never writes anything (in particular, it never
recaptures/regenerates a fixture -- see tools/capture_*.py for that). That
is what lets it run unconditionally in `tools/ci.sh`, which has no C++
toolchain (see that script's own docstring).

Two manifest storage shapes are recognized (see tools/_manifest.py's
docstring for why both exist):

* EMBEDDED -- the fixture's own top-level JSON is a dict with a
  `_manifest` key (tests/fixtures/hillclimbing.json,
  tests/fixtures/deme_expander.json).
* SIDECAR -- a `<name>.manifest.json` file next to the fixture (every
  other fixture).

What "internally consistent" means here, concretely, per fixture:

1. Present: a manifest is found (embedded or sidecar). A fixture with
   neither is the #1 failure this script exists to catch -- silently
   growing the fixture count without ever giving the new fixture a
   manifest is exactly how CARRY-FORWARD.md Sec 5's debt grew past "six"
   in the first place.
2. Schema-valid: `schema_version` is a supported int; `capture_script`/
   `probe_source`/`cli_tool` are consistent with `probe_kind`; every
   sha256 field is a well-formed 64-hex-char string (or None, when the
   corresponding path field is also None); `generated_at_utc` parses as
   an ISO-8601 timestamp; `inputs` is a dict.
3. Paths resolve: `capture_script` and (when set) `probe_source` name
   real files in THIS repo.
4. Rot-detected: `capture_script_sha256`/`probe_source_sha256` match the
   CURRENT bytes of those files. A mismatch means the script or probe
   that produced this fixture has changed since capture without the
   fixture being recaptured -- precisely "a fixture rotted silently"
   (CARRY-FORWARD.md Sec 5), and precisely what
   tests/moses/moses/optimization/test_hill_climbing.py's own
   `test_fixture_manifest_matches_current_probe_source` already checks
   for hillclimbing.json alone; this generalizes that check to every
   fixture with a manifest, and (via #1) requires every fixture to HAVE
   one.

What this script does NOT do: decide whether a fixture's DATA is still
correct. A probe_source hash mismatch says the probe changed; it does not
say whether the fixture's captured values are now wrong (they might not
be -- see rng.json's own history: rng_probe.cc was rewritten to link the
real cogutil MT19937RandGen instead of a hand-copy, and a verification
rebuild during this task proved byte-for-byte identical output despite
the source-level rewrite). That judgment call belongs to a human
re-running the capture script and diffing, never to an automated
overwrite -- see tools/capture_*.py's own docstrings and
docs/CARRY-FORWARD.md.

Usage:

    python3 tools/check_fixture_manifests.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"

SUPPORTED_SCHEMA_VERSIONS = (1,)
PROBE_KINDS = ("cpp_probe", "classic_cli", "python_only")
_SHA256_LEN = 64


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LEN
        and all(c in "0123456789abcdef" for c in value)
    )


def discover_fixtures(fixture_dir: Path) -> list[Path]:
    """Every `tests/fixtures/*.json` fixture, excluding sidecar manifest
    files themselves. Sorted for stable, diffable output.
    """
    return sorted(
        p
        for p in fixture_dir.glob("*.json")
        if not p.name.endswith(".manifest.json")
    )


def load_manifest(fixture_path: Path) -> tuple[dict | None, str | None]:
    """Return (manifest, error). `manifest` is None iff no manifest was
    found at all (embedded or sidecar) -- everything else is reported as
    a manifest that WAS found but fails validation, not "missing".
    """
    try:
        raw = fixture_path.read_text()
    except OSError as error:
        return None, f"could not read fixture: {error}"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        return None, f"fixture is not valid JSON: {error}"

    if isinstance(data, dict) and "_manifest" in data:
        manifest = data["_manifest"]
        if not isinstance(manifest, dict):
            return None, "embedded _manifest is present but is not a JSON object"
        return manifest, None

    sidecar = fixture_path.with_suffix(".manifest.json")
    if not sidecar.is_file():
        try:
            sidecar_display = sidecar.relative_to(REPO_ROOT)
        except ValueError:
            sidecar_display = sidecar
        return None, (
            f"no manifest found -- neither an embedded top-level \"_manifest\" key "
            f"nor a sidecar file at {sidecar_display}"
        )
    try:
        manifest = json.loads(sidecar.read_text())
    except json.JSONDecodeError as error:
        return None, f"sidecar manifest {sidecar.name} is not valid JSON: {error}"
    if not isinstance(manifest, dict):
        return None, f"sidecar manifest {sidecar.name} does not contain a JSON object"
    return manifest, None


def validate_manifest(manifest: dict, repo_root: Path = REPO_ROOT) -> list[str]:
    """Return a list of validation error strings (empty if valid). Never
    raises -- a malformed field is reported, not a crash.

    `repo_root` defaults to this repo but is overridable so
    tests/fixtures/test_fixture_manifests.py can exercise every failure
    mode against synthetic files under `tmp_path`, without touching a
    single real committed fixture, script, or probe.
    """
    errors: list[str] = []

    schema_version = manifest.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        errors.append(
            f"schema_version={schema_version!r} is not one of {SUPPORTED_SCHEMA_VERSIONS}"
        )

    capture_script = manifest.get("capture_script")
    if not isinstance(capture_script, str) or not capture_script:
        errors.append("capture_script must be a non-empty string")
    else:
        script_path = repo_root / capture_script
        if not script_path.is_file():
            errors.append(f"capture_script {capture_script!r} does not exist in this repo")
        else:
            script_hash = manifest.get("capture_script_sha256")
            if not _is_sha256(script_hash):
                errors.append("capture_script_sha256 must be a 64-char hex sha256 string")
            else:
                actual = hashlib.sha256(script_path.read_bytes()).hexdigest()
                if actual != script_hash:
                    errors.append(
                        f"capture_script_sha256 mismatch for {capture_script}: "
                        f"manifest records {script_hash}, current file hashes to {actual} "
                        "-- the capture script has changed since this fixture was "
                        "captured (rot -- see this script's module docstring)"
                    )

    probe_kind = manifest.get("probe_kind")
    if probe_kind not in PROBE_KINDS:
        errors.append(f"probe_kind={probe_kind!r} is not one of {PROBE_KINDS}")

    probe_source = manifest.get("probe_source")
    probe_source_sha256 = manifest.get("probe_source_sha256")
    if (probe_source is None) != (probe_source_sha256 is None):
        errors.append(
            "probe_source and probe_source_sha256 must both be set or both be None "
            f"(got probe_source={probe_source!r}, probe_source_sha256={probe_source_sha256!r})"
        )
    elif probe_source is not None:
        if not isinstance(probe_source, str):
            errors.append("probe_source must be a string when set")
        else:
            probe_path = repo_root / probe_source
            if not probe_path.is_file():
                errors.append(f"probe_source {probe_source!r} does not exist in this repo")
            elif not _is_sha256(probe_source_sha256):
                errors.append("probe_source_sha256 must be a 64-char hex sha256 string")
            else:
                actual = hashlib.sha256(probe_path.read_bytes()).hexdigest()
                if actual != probe_source_sha256:
                    errors.append(
                        f"probe_source_sha256 mismatch for {probe_source}: "
                        f"manifest records {probe_source_sha256}, current file hashes to "
                        f"{actual} -- the probe has changed since this fixture was "
                        "captured (rot -- see this script's module docstring)"
                    )

    if probe_kind == "cpp_probe" and probe_source is None:
        errors.append('probe_kind is "cpp_probe" but probe_source is None')
    if probe_kind == "classic_cli":
        if probe_source is not None:
            errors.append('probe_kind is "classic_cli" but probe_source is set (should be None)')
        cli_tool = manifest.get("cli_tool")
        if not isinstance(cli_tool, str) or not cli_tool:
            errors.append('probe_kind is "classic_cli" but cli_tool is not a non-empty string')
    if probe_kind == "python_only" and manifest.get("cli_tool") is not None:
        errors.append('probe_kind is "python_only" but cli_tool is set (should be None)')

    repo_commit = manifest.get("repo_commit")
    if repo_commit is not None and not (isinstance(repo_commit, str) and len(repo_commit) == 40):
        errors.append("repo_commit must be a 40-char git commit hash, or None")

    generated_at_utc = manifest.get("generated_at_utc")
    if not isinstance(generated_at_utc, str) or not generated_at_utc:
        errors.append("generated_at_utc must be a non-empty string")
    else:
        try:
            datetime.fromisoformat(generated_at_utc)
        except ValueError:
            errors.append(
                f"generated_at_utc={generated_at_utc!r} is not a valid ISO-8601 timestamp"
            )

    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        errors.append("inputs must be a JSON object")

    return errors


def check_all_fixtures(
    fixture_dir: Path = FIXTURE_DIR, repo_root: Path = REPO_ROOT
) -> dict[str, list[str]]:
    """Check every fixture in `fixture_dir`. Returns {fixture_name:
    [error, ...]} for fixtures with at least one error -- an empty dict
    means everything passed. A fixture with no manifest at all appears
    with a single-element error list from `load_manifest`.
    """
    results: dict[str, list[str]] = {}
    for fixture_path in discover_fixtures(fixture_dir):
        manifest, load_error = load_manifest(fixture_path)
        if load_error is not None:
            results[fixture_path.name] = [load_error]
            continue
        errors = validate_manifest(manifest, repo_root=repo_root)
        if errors:
            results[fixture_path.name] = errors
    return results


def main(argv: list[str]) -> int:
    del argv  # no options -- always checks every tests/fixtures/*.json
    fixtures = discover_fixtures(FIXTURE_DIR)
    if not fixtures:
        print(f"no fixtures found under {FIXTURE_DIR}", file=sys.stderr)
        return 1

    failures = check_all_fixtures(FIXTURE_DIR)
    if failures:
        print(f"FAIL: {len(failures)}/{len(fixtures)} fixture(s) have manifest problems:\n")
        for name in sorted(failures):
            print(f"  {name}:")
            for error in failures[name]:
                print(f"    - {error}")
        return 1

    print(f"OK: {len(fixtures)} fixture(s) all have a present, internally consistent manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
