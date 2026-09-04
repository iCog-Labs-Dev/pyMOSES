"""Capture the combo tree round-trip oracle fixture from classic MOSES's
`combo-fmt-converter` CLI into tests/fixtures/combo_roundtrip.json.

Unlike tools/capture_golden.py's probes (dependency-free g++ compiles run
against nothing but the C++ standard library), combo tree I/O lives inside
libcomboreduct.so -- there is no standalone-compilable probe for it. This
script instead shells out to the classic MOSES CLI tool that already wraps
that library, `combo-fmt-converter` (built via
scripts/rebuild-moses-local.sh in the classic checkout; see CLAUDE.md),
assumed already built at CONVERTER below.

Corpus (30 combo tree strings): the 29 entries of the `trees` array in
$CLASSIC/testTrees.sh, plus the one tree literal constructed in
$CLASSIC/tests/comboreduct/combo/iostream_comboUTest.cxxtest's
iostream_comboUTest() constructor (line 11). See task-3-brief.md.

For each corpus string, combo-fmt-converter parses it and re-prints it in
its default (combo) output format -- exactly the str_to_combo_tree /
combo_tree_to_str round trip this task ports. Passing every corpus string
as a single `-c` per tree in one process invocation (rather than one
subprocess per tree) keeps this to a single bounded run.

Developer tool, not part of tools/ci.sh -- the committed
tests/fixtures/combo_roundtrip.json is what the test suite actually reads,
so the suite runs without a C++ toolchain or the classic checkout at all.
"""

import json
import resource
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "combo_roundtrip.json"
CONVERTER = Path.home() / "opt" / "moses-legacy" / "bin" / "combo-fmt-converter"

# ulimit -v 1048576 (KiB) / timeout 20 -- see CLAUDE.md: "this machine has
# been OOM-killed twice." resource.RLIMIT_AS is the setrlimit(2) equivalent
# of `ulimit -v`, applied via preexec_fn rather than a `bash -c` wrapper so
# that no shell has to re-quote combo strings full of `$`, `!`, `(`, `)`.
_MEM_LIMIT_BYTES = 1048576 * 1024
_TIMEOUT_S = 20

# C++: $CLASSIC/testTrees.sh -- the `trees` array, verbatim and in order.
TEST_TREES_SH_CORPUS = [
    "or(and(or(!$1)) and())",
    "or(and(or($1)) and())",
    "or(and(or(!$2)) and())",
    "or(and(or($2)) and())",
    "or(and(and($1 !$3) or()) and())",
    "or(and(or(!$1 $3) or()) and())",
    "or(and(and(!$2 !$3) or()) and())",
    "or(and(or($2 $3) or()) and())",
    "or(and(and(!$2 !$1) or()) and())",
    "or(and(or($2 $1) or()) and())",
    "or(and(!$3 or()) and())",
    "or(and($3 or()) and())",
    "or(and(!$1 or()) and())",
    "or(and($1 or()) and())",
    "or(and(!$2 or()) and())",
    "or(and($2 or()) and())",
    "or(and(or()) or(!$1 !$3) and())",
    "or(and(or()) and($1 $3) and())",
    "or(and(or()) or($2 !$1) and())",
    "or(and(or()) and(!$2 $1) and())",
    "or(and(or()) or(!$2 !$3) and())",
    "or(and(or()) and($2 $3) and())",
    "or(and(or()) !$3 and())",
    "or(and(or()) $3 and())",
    "or(and(or()) !$1 and())",
    "or(and(or()) $1 and())",
    "or(and(or()) !$2 and())",
    "or(and(or()) $2 and())",
    "or(and(or()) and($2))",
]

# C++: $CLASSIC/tests/comboreduct/combo/iostream_comboUTest.cxxtest:11
IOSTREAM_COMBO_UTEST_CORPUS = [
    "and(or($1 $2) not($3))",
]

CORPUS = TEST_TREES_SH_CORPUS + IOSTREAM_COMBO_UTEST_CORPUS


def _bound_resources() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))


def capture(trees: list[str]) -> list[str]:
    """Run combo-fmt-converter once, with one `-c <tree>` per corpus entry
    (order-preserving: --help documents that -c "may be used several times
    to convert several programs"), and return one output line per tree with
    its trailing space preserved (ostream_combo_tree, iostream_combo.cc:472,
    appends exactly one trailing space per tree) and only the line's
    terminating newline stripped.
    """
    if not CONVERTER.is_file():
        raise FileNotFoundError(
            f"combo-fmt-converter not found at {CONVERTER} -- build classic "
            "MOSES first (see CLAUDE.md's scripts/rebuild-moses-local.sh)"
        )
    args = [str(CONVERTER)]
    for tree in trees:
        args += ["-c", tree]
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
        preexec_fn=_bound_resources,
        check=True,
    )
    lines = result.stdout.split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # trailing newline after the last tree's output line
    if len(lines) != len(trees):
        raise RuntimeError(
            f"expected {len(trees)} output lines from combo-fmt-converter, got "
            f"{len(lines)}\nstdout:\n{result.stdout!r}\nstderr:\n{result.stderr!r}"
        )
    return lines


def main(argv: list[str]) -> int:
    del argv  # no CLI options yet -- one fixed corpus to (re)capture
    outputs = capture(CORPUS)
    fixture = [
        {"input": tree, "expected": out} for tree, out in zip(CORPUS, outputs, strict=True)
    ]
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {FIXTURE_PATH} ({len(fixture)} trees)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
