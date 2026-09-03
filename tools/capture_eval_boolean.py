"""Capture the boolean_interpreter differential oracle fixture from classic
MOSES's `eval-table` CLI into tests/fixtures/eval_boolean.json.

Crosses the 30-tree combo_roundtrip corpus (tests/fixtures/combo_roundtrip.json,
captured by tools/capture_combo_roundtrip.py -- see task-3-brief.md) against
every row of $CLASSIC/trainingBoolean.dat (3 boolean inputs, target `y`) and
$CLASSIC/trainingBooleanXor.dat (2 inputs), for every corpus tree whose
highest-numbered argument fits the dataset's arity (a tree referencing `$3`
is skipped for the 2-input xor table -- there is no third column to feed
it). All 30 trees fit trainingBoolean.dat's arity of 3; 17 of the 30 fit
xor's arity of 2.

Developer tool, not part of tools/ci.sh -- the committed
tests/fixtures/eval_boolean.json is what the test suite actually reads, so
the suite runs without a C++ toolchain or the classic checkout at all.

## Why corpus trees are rewritten from `$N` to `$<column-name>` before
## being handed to eval-table

This is the single trickiest part of this capture, worth recording in
full. eval-table's `-L`/`--labels` option (default enabled, i.e. `-L 1`)
does NOT mean "resolve `$1`/`$2`/... by position" when disabled and
"by label" when enabled, despite the flag's name suggesting a toggle
between those two readings. Tracing eval-table.cc's
`read_eval_output_results`:

1. `parse_combo_variables(tr_str)` extracts every `$<token>` in the RAW
   combo string -- BEFORE any label/placeholder handling -- purely as a
   variable NAME string. For a tree written with positional arguments,
   e.g. `$1`, this extracts the literal string `"1"`, not "column 1".
2. Every CSV header column whose name is not in that extracted set is
   added to `ignore_variables` and dropped from the loaded table. Since
   the training files' real headers (`x1`, `x2`, `x3`) never equal `"1"`,
   `"2"`, `"3"`, using positional `$N` syntax gets EVERY real input column
   ignored.
3. Whatever variable names WERE extracted (`"1"`, `"2"`, `"3"`) but are
   missing from the header are then appended as synthetic "missing
   variable" columns, defaulted to a constant (empirically: always 0/false
   -- see task-4-report.md for the verification transcript).
4. Only AFTER all of the above does `-L`'s value matter: it selects
   whether `str2combo_tree` is called with the real header labels (so
   `$x1` is rewritten to positional `$1` via `l2ph`, matching against
   whatever `ignore`-filtered table remains) or with no labels at all (so
   `$1` is parsed as a literal `argument(1)` against that same,
   already-mangled table).

Net effect verified empirically against the real binary: a combo string
written with positional arguments (`$1`, `$2`, ...) evaluates to a
constant 0 for every row, regardless of `-L`, because step 2 has already
discarded the real data columns by the time either parse path runs.
Writing `$x1`, `$x2`, ... (the CSV's own header names) with the default
`-L 1` is the only invocation that actually reads the real column values.
This script therefore text-substitutes `$<N>` -> `$<column name>` before
every `-c`, and reports results keyed by the ORIGINAL `$N` tree string
(what the Python side and the rest of this port actually use).
"""

import csv
import json
import re
import resource
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "eval_boolean.json"
ROUNDTRIP_PATH = REPO_ROOT / "tests" / "fixtures" / "combo_roundtrip.json"
EVAL_TABLE = Path.home() / "opt" / "moses-legacy" / "bin" / "eval-table"

CLASSIC = Path("/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses")
DATASETS = [
    {"name": "trainingBoolean.dat", "path": CLASSIC / "trainingBoolean.dat", "target": "y"},
    {"name": "trainingBooleanXor.dat", "path": CLASSIC / "trainingBooleanXor.dat", "target": "y"},
]

# ulimit -v 1048576 (KiB) / timeout 20 -- see CLAUDE.md: "this machine has
# been OOM-killed twice." Same pattern as tools/capture_combo_roundtrip.py.
_MEM_LIMIT_BYTES = 1048576 * 1024
_TIMEOUT_S = 20

_ARG_RE = re.compile(r"\$(\d+)")


def _bound_resources() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))


def load_dataset_rows(path: Path, target: str) -> tuple[list[str], list[list[int]]]:
    """Read a headed CSV (`.dat` despite the extension -- see CLAUDE.md).
    Returns (input column names in header order, one int row per data row,
    target column excluded).
    """
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        input_cols = [h for h in header if h != target]
        idxs = [header.index(c) for c in input_cols]
        rows = [[int(row[i]) for i in idxs] for row in reader]
    return input_cols, rows


def max_arg_index(tree_str: str) -> int:
    """Highest positional argument (`$N` or `!$N`) referenced anywhere in
    the tree string; 0 if it references none.
    """
    return max((int(m) for m in _ARG_RE.findall(tree_str)), default=0)


def to_label_form(tree_str: str, input_cols: list[str]) -> str:
    """Rewrite every `$N` to `$<input_cols[N-1]>` -- see module docstring
    for why this rewrite is required at all.
    """
    return _ARG_RE.sub(lambda m: f"${input_cols[int(m.group(1)) - 1]}", tree_str)


def capture_for_dataset(dataset: dict, trees: list[str]) -> list[dict]:
    input_cols, rows = load_dataset_rows(dataset["path"], dataset["target"])
    arity = len(input_cols)
    included = [t for t in trees if max_arg_index(t) <= arity]
    substituted = [to_label_form(t, input_cols) for t in included]

    if not EVAL_TABLE.is_file():
        raise FileNotFoundError(
            f"eval-table not found at {EVAL_TABLE} -- build classic MOSES "
            "first (see CLAUDE.md's scripts/rebuild-moses-local.sh)"
        )
    args = [str(EVAL_TABLE), "-i", str(dataset["path"]), "-u", dataset["target"]]
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
        lines.pop()  # trailing newline after the last tree's block

    n_rows = len(rows)
    block_size = 1 + n_rows  # one header line ("y") + one value per row
    expected_lines = block_size * len(included)
    if len(lines) != expected_lines:
        raise RuntimeError(
            f"{dataset['name']}: expected {expected_lines} output lines "
            f"({len(included)} trees x {block_size}), got {len(lines)}\n"
            f"stdout:\n{result.stdout!r}\nstderr:\n{result.stderr!r}"
        )

    entries = []
    for i, tree in enumerate(included):
        block = lines[i * block_size : (i + 1) * block_size]
        header_line, values = block[0], block[1:]
        if header_line != dataset["target"]:
            raise RuntimeError(
                f"{dataset['name']}: expected header {dataset['target']!r} for "
                f"tree {tree!r}, got {header_line!r}"
            )
        for row, value in zip(rows, values, strict=True):
            entries.append(
                {
                    "dataset": dataset["name"],
                    "tree": tree,
                    "row": row,
                    "expected": value,
                }
            )
    return entries


def main(argv: list[str]) -> int:
    del argv  # no CLI options yet -- one fixed corpus x dataset cross to (re)capture
    trees = [e["input"] for e in json.loads(ROUNDTRIP_PATH.read_text())]
    fixture: list[dict] = []
    for dataset in DATASETS:
        entries = capture_for_dataset(dataset, trees)
        fixture += entries
        print(f"{dataset['name']}: {len(entries)} (tree, row) comparisons")
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {FIXTURE_PATH} ({len(fixture)} total comparisons)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
