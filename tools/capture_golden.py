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
sys.path.insert(0, str(REPO_ROOT))
from tools._capture_env import moses_prefix  # noqa: E402
from tools._manifest import build_manifest, write_sidecar  # noqa: E402

PROBE_DIR = REPO_ROOT / "tools" / "cpp_probe"
BUILD_DIR = PROBE_DIR / "build"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"

# probe source (under tools/cpp_probe/) -> fixture file (under tests/fixtures/)
PROBES = {
    "rng_probe.cc": "rng.json",
}

# Probes that need cogutil's headers/library linked in, beyond the plain
# dependency-free g++ invocation every other probe here gets -- see this
# module's own docstring and rng_probe.cc's header comment for why.
_NEEDS_COGUTIL = {"rng_probe.cc"}


def build_probe(source: Path, binary: Path) -> None:
    """Compile one probe. Every probe but `rng_probe.cc` gets a plain,
    dependency-free g++ invocation; `rng_probe.cc` additionally links
    against the installed cogutil (see this module's own docstring) --
    resolved via PYMOSES_MOSES_PREFIX (tools/_capture_env.py), only when
    actually needed, so this function does not require that variable for
    every other probe in PROBES.
    """
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["g++", "-O2", "-std=c++17"]
    if source.name in _NEEDS_COGUTIL:
        cogutil_prefix = moses_prefix()
        include = cogutil_prefix / "include"
        lib = cogutil_prefix / "lib"
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


# Per-probe manifest `inputs` -- deliberately not derivable generically
# from PROBES, since each probe's actual parameterization lives in its own
# .cc source, not in this script.
_INPUTS = {
    "rng_probe.cc": {
        "seeds": [0, 1, 7, 42, 12345, 4294967295],
        "draws_per_seed": 2000,
        "series": ["raw", "randint", "randint_7", "randbool", "randdouble", "randfloat"],
        "biased_randbool_weights": 20,
        "biased_randbool_groups": ["shared generator", "fresh generator per weight"],
    },
}


def main(argv: list[str]) -> int:
    del argv  # no CLI options yet -- one fixed table of probes to (re)run
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for probe_name, fixture_name in PROBES.items():
        source = PROBE_DIR / probe_name
        binary = BUILD_DIR / probe_name.removesuffix(".cc")
        fixture_path = FIXTURE_DIR / fixture_name
        build_probe(source, binary)
        capture_fixture(binary, fixture_path)
        write_sidecar(
            fixture_path,
            build_manifest(
                repo_root=REPO_ROOT,
                capture_script="tools/capture_golden.py",
                capture_script_path=Path(__file__).resolve(),
                probe_source=f"tools/cpp_probe/{probe_name}",
                probe_source_path=source,
                probe_kind="cpp_probe",
                prefix_dir=moses_prefix() if probe_name in _NEEDS_COGUTIL else None,
                inputs=_INPUTS.get(probe_name, {}),
            ),
        )
        print(f"wrote {fixture_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
