"""Build tools/cpp_probe/*.cc oracle probes and write their JSON output to
tests/fixtures/.

The committed tests/fixtures/*.json is what the default test suite actually
reads (see tests/conftest.py's `fixture_dir` fixture), so the suite runs
without a C++ toolchain. This script exists to regenerate those fixtures
when a probe changes; it is a developer tool, not part of `tools/ci.sh`.

Each probe here needs nothing beyond the C++ standard library -- no MOSES or
cogutil headers or libraries -- unless a given probe's own comment says
otherwise. `rng_probe.cc` is the one exception (see its own header comment):
it links against the installed cogutil so it calls the real
`opencog::MT19937RandGen`, not a hand-copy of its algorithm.
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

# cogutil install prefix -- same default tools/cpp_probe/score_probe.cc's own
# build recipe comment uses, and the same one CLAUDE.md's rebuild-moses-local.sh
# installs to.
COGUTIL_PREFIX = Path.home() / "opt" / "moses-legacy"

# Probes that need cogutil's headers/library linked in, beyond the plain
# dependency-free g++ invocation every other probe here gets -- see this
# module's own docstring and rng_probe.cc's header comment for why.
_NEEDS_COGUTIL = {"rng_probe.cc"}


def build_probe(source: Path, binary: Path) -> None:
    """Compile one probe. Every probe but `rng_probe.cc` gets a plain,
    dependency-free g++ invocation; `rng_probe.cc` additionally links
    against the installed cogutil (see this module's own docstring).
    """
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["g++", "-O2", "-std=c++17"]
    if source.name in _NEEDS_COGUTIL:
        include = COGUTIL_PREFIX / "include"
        lib = COGUTIL_PREFIX / "lib"
        lib_opencog = lib / "opencog"
        cmd += ["-I", str(include)]
        cmd += ["-o", str(binary), str(source)]
        cmd += [
            "-L",
            str(lib),
            "-L",
            str(lib_opencog),
            "-Wl,-rpath," + str(lib),
            "-Wl,-rpath," + str(lib_opencog),
            "-lcogutil",
        ]
    else:
        cmd += ["-o", str(binary), str(source)]
    subprocess.run(cmd, check=True)


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
