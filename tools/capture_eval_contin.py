"""Capture the contin_interpreter/mixed_interpreter differential oracle
fixture from classic MOSES's `eval-table` CLI into
tests/fixtures/eval_contin.json.

Runs a fixed corpus of 9 contin/mixed combo trees against
`$CLASSIC/trainingRegression.dat` (8,000-row contin regression, headed
CSV despite the `.dat` extension, columns x1/x2/y -- CLAUDE.md) -- the
oracle S14 Task 1's own plan document names explicitly. Every row's
`x1`/`x2` lies in (0.01, 1.0) (measured directly: min 0.01001, max
0.99966 across both columns), so every tree below is evaluated over its
full natural domain with no div-by-zero/log-domain/overflow edge case
anywhere in the file -- those are covered separately by
tools/cpp_probe/contin_interpreter_probe.cc, which the eval-table CLI
cannot reach at all (a whole-table pass has no way to feed a single
contrived input).

Only a SAMPLE of rows (every 200th, 40 of 8,000) is kept in the committed
fixture, chosen before this script was ever run against real output --
not selected after the fact to hide a disagreement. eval-table itself
still evaluates every one of the 8,000 rows; only the SAVED fixture is
sampled, to keep it a similar order of magnitude to this project's other
oracle fixtures (eval_boolean.json: 308 entries).

## The `$N` -> `$<header-name>` substitution trap (CARRY-FORWARD Sec 3)

Same substitution capture_eval_boolean.py already documents in full:
eval-table's positional `$N` arguments always resolve to a zero column
regardless of `-L`/`--labels`, because `parse_combo_variables` extracts
variable NAMES from the raw combo string before any label handling ever
runs. Every tree below is written with `$1`/`$2` for readability and
substituted to `$x1`/`$x2` before being handed to eval-table, exactly as
capture_eval_boolean.py already does for the boolean corpus.

Developer tool, not part of tools/ci.sh -- the committed
tests/fixtures/eval_contin.json is what the test suite actually reads, so
the suite runs without a C++ toolchain or the classic checkout at all.
"""

import csv
import json
import re
import resource
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tools._capture_env import classic_dir, moses_prefix  # noqa: E402
from tools._manifest import build_manifest, write_sidecar  # noqa: E402

FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "eval_contin.json"
PREFIX = moses_prefix()
EVAL_TABLE = PREFIX / "bin" / "eval-table"

CLASSIC = classic_dir()
DATASET_PATH = CLASSIC / "trainingRegression.dat"
TARGET = "y"

# Every 200th row (0, 200, ..., 7800) of the 8,000-row file -- decided
# before this script was ever run, not chosen after seeing any output.
_ROW_STRIDE = 200

# Nine trees mixing pure-contin arithmetic (plus/times/div/log/exp/sin,
# a literal contin constant) with mixed contin-input/boolean-output
# expressions (greater_than_zero, impulse) -- chosen to exercise both
# contin_interpreter's own arithmetic AND the fact that eval-table's real
# path (a homogeneous-contin ITable row) always routes through
# mixed_interpreter (table.h:332-336's "the output might be non-contin"
# rule), never bare contin_interpreter -- see interpreter.py's own module
# docstring.
_TREES = [
    "+($1 $2)",
    "*($1 $2)",
    "/($1 $2)",
    "log($1)",
    "exp($2)",
    "sin($1)",
    "+(*(2.5 $1) *(-1.0 $2))",
    "0<(+($1 *(-1.0 $2)))",
    "impulse(0<(+($1 *(-1.0 $2))))",
]

# ulimit -v 1048576 (KiB) / timeout 30 -- see CLAUDE.md: "this machine has
# been OOM-killed twice." 8,000 rows x 9 trees is more work than the
# boolean capture's tables, so a slightly larger timeout.
_MEM_LIMIT_BYTES = 1048576 * 1024
_TIMEOUT_S = 30

_ARG_RE = re.compile(r"\$(\d+)")


def _bound_resources() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))


def load_dataset_rows(
    path: Path, target: str
) -> tuple[list[str], list[list[float]], list[str], list[list[str]]]:
    """Returns (input column names, parsed float input rows, full header
    incl. target, raw string rows incl. target) -- the raw form is kept
    so the small table_data/ sample CSV this script also writes preserves
    the ORIGINAL tokens byte-for-byte, rather than round-tripping them
    through float formatting.
    """
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        input_cols = [h for h in header if h != target]
        idxs = [header.index(c) for c in input_cols]
        raw_rows = list(reader)
        rows = [[float(row[i]) for i in idxs] for row in raw_rows]
    return input_cols, rows, header, raw_rows


def to_label_form(tree_str: str, input_cols: list[str]) -> str:
    return _ARG_RE.sub(lambda m: f"${input_cols[int(m.group(1)) - 1]}", tree_str)


def main(argv: list[str]) -> int:
    del argv  # no CLI options yet -- one fixed corpus/dataset/sample to (re)capture
    if not EVAL_TABLE.is_file():
        raise FileNotFoundError(
            f"eval-table not found at {EVAL_TABLE} -- build classic MOSES first "
            "(see CLAUDE.md's scripts/rebuild-moses-local.sh)"
        )
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"trainingRegression.dat not found: {DATASET_PATH}")

    input_cols, rows, full_header, raw_rows = load_dataset_rows(DATASET_PATH, TARGET)
    n_rows = len(rows)
    substituted = [to_label_form(t, input_cols) for t in _TREES]

    args = [str(EVAL_TABLE), "-i", str(DATASET_PATH), "-u", TARGET]
    for s in substituted:
        args += ["-c", s]
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
        lines.pop()

    block_size = 1 + n_rows  # one header line ("y") + one value per row
    expected_lines = block_size * len(_TREES)
    if len(lines) != expected_lines:
        raise RuntimeError(
            f"expected {expected_lines} output lines ({len(_TREES)} trees x "
            f"{block_size}), got {len(lines)}\nstdout:\n{result.stdout!r}\n"
            f"stderr:\n{result.stderr!r}"
        )

    sample_idxs = list(range(0, n_rows, _ROW_STRIDE))
    fixture: list[dict] = []
    for i, tree in enumerate(_TREES):
        block = lines[i * block_size : (i + 1) * block_size]
        header_line, values = block[0], block[1:]
        if header_line != TARGET:
            raise RuntimeError(f"tree {tree!r}: expected header {TARGET!r}, got {header_line!r}")
        for row_idx in sample_idxs:
            fixture.append(
                {
                    "tree": tree,
                    "row_index": row_idx,
                    "row": rows[row_idx],
                    "expected": values[row_idx],
                }
            )

    # Also write a small table_data/ CSV of just the sampled rows (header
    # + original raw tokens, unchanged) so a test can exercise the REAL
    # table_io.loadTable -> ITable -> interpreter_visitor pipeline without
    # needing the full 8,000-row $CLASSIC file or a C++ toolchain present
    # -- same division of labor as tools/capture_score.py's DATASETS.
    sample_csv_path = REPO_ROOT / "tests" / "fixtures" / "table_data" / "regression_sample.csv"
    sample_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(full_header)
        for row_idx in sample_idxs:
            writer.writerow(raw_rows[row_idx])

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    write_sidecar(
        FIXTURE_PATH,
        build_manifest(
            repo_root=REPO_ROOT,
            capture_script="tools/capture_eval_contin.py",
            capture_script_path=Path(__file__).resolve(),
            probe_source=None,
            probe_source_path=None,
            probe_kind="classic_cli",
            cli_tool="bin/eval-table",
            classic_dir=CLASSIC,
            prefix_dir=PREFIX,
            inputs={
                "dataset": "trainingRegression.dat",
                "n_rows_total": n_rows,
                "row_stride": _ROW_STRIDE,
                "n_rows_sampled": len(sample_idxs),
                "trees": _TREES,
                "sample_csv": "tests/fixtures/table_data/regression_sample.csv",
            },
        ),
    )
    print(
        f"wrote {FIXTURE_PATH} ({len(fixture)} comparisons: {len(_TREES)} trees x "
        f"{len(sample_idxs)} sampled rows of {n_rows})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
