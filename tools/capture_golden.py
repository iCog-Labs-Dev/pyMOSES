"""Build tools/cpp_probe/*.cc oracle probes and write their JSON output to
tests/fixtures/.

The committed tests/fixtures/*.json is what the default test suite actually
reads (see tests/conftest.py's `fixture_dir` fixture), so the suite runs
without a C++ toolchain. This script exists to regenerate those fixtures
when a probe changes; it is a developer tool, not part of `tools/ci.sh`.

Each probe here needs nothing beyond the C++ standard library -- no MOSES or
cogutil headers or libraries -- unless a given probe's own comment says
otherwise.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_DIR = REPO_ROOT / "tools" / "cpp_probe"
BUILD_DIR = PROBE_DIR / "build"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"

# probe source (under tools/cpp_probe/) -> fixture file (under tests/fixtures/)
PROBES = {
    "rng_probe.cc": "rng.json",
}


def build_probe(source: Path, binary: Path) -> None:
    """Compile one probe with a plain, dependency-free g++ invocation."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["g++", "-O2", "-std=c++17", "-o", str(binary), str(source)],
        check=True,
    )


def capture_fixture(binary: Path, fixture_path: Path) -> None:
    """Run one built probe and save its stdout as the golden fixture."""
    result = subprocess.run([str(binary)], capture_output=True, check=True, text=True)
    fixture_path.write_text(result.stdout)


def main(argv: list[str]) -> int:
    del argv  # no CLI options yet -- one fixed table of probes to (re)run
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for probe_name, fixture_name in PROBES.items():
        source = PROBE_DIR / probe_name
        binary = BUILD_DIR / probe_name.removesuffix(".cc")
        build_probe(source, binary)
        capture_fixture(binary, FIXTURE_DIR / fixture_name)
        print(f"wrote {FIXTURE_DIR / fixture_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
