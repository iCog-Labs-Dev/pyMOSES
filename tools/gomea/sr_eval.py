"""Score a MOSES regression program on train/test CSV splits (GP-GOMEA SR protocol).

Usage (from the repository root)::

    .venv/bin/python tools/gomea/sr_eval.py --train train.csv --test test.csv \\
        [--target y] [--ignore-feature NAME ...] (--program "*(2 $3)" | --moses-output FILE)

`--moses-output` takes a file holding MOSES stdout (`-` for stdin); the first candidate
line is scored (`--all` scores every candidate line). Candidate lines may carry the
leading score (`-S1`, the default), a trailing composite score `[score=...]` (`-x1`) or
bscore (`-t1`), and labelled arguments (`-W1`, e.g. `$x3`); `n_evals:` and other
non-candidate lines are skipped.

Variable mapping (as MOSES reads a table): `$k` is the k-th column of the CSV header
once the target column and every `-Y/--ignore-feature` column are removed, in file order.

Evaluation uses pyMoses's own combo parser (`str_to_combo_tree`) and mixed interpreter
(which also handles `impulse`, `0<` and the logical operators that MOSES regression
programs can contain), so arithmetic is exactly MOSES's: `/(x y)` returns `x` when
`x == 0` (even if `y == 0`),
and raises `OverflowException` when `x / y` is inf or nan. MOSES scores such a candidate
as the worst possible score; here the affected rows get a `nan` prediction and every
NMSE of that split is reported as `inf`.

Reported metrics, per split, with `var` the population variance of that split's `y`:

* `nmse_scaled` = 100 * MSE(y, a + b * p) / var(y), where (a, b) is the least-squares
  linear scaling of the predictions `p` fitted on TRAIN only (Keijzer 2003):
  b = cov(y, p) / var(p), a = mean(y) - b * mean(p); b = 0 when var(p) == 0.
* `nmse_raw` = 100 * MSE(y, p) / var(y), the unscaled program (what MOSES optimises).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymoses.comboreduct.combo.vertex import builtin
from pymoses.comboreduct.crutil.exception import OverflowException
from pymoses.comboreduct.interpreter.interpreter import mixed_interpreter
from pymoses.comboreduct.table.table_io import infer_type_from_token2
from pymoses.comboreduct.type_checker.type_signature import type_node

_LABEL_RE = re.compile(r"(!?)\$([^\s()$]+)")
_FLOAT_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$|^[-+]?(inf|nan)$", re.I)


@dataclass
class Table:
    """Input matrix (row-major, MOSES argument order) and target vector."""

    labels: list[str]
    X: list[list[float]]
    y: list[float]
    # Feature columns MOSES types as boolean (every token is 0/1/T/F/t/f), as indices
    # into `labels`. MOSES passes those as logical values, so `impulse($k)` works.
    bool_cols: frozenset[int] = frozenset()


def read_table(path: str, target: str | None = None, ignore: Sequence[str] = ()) -> Table:
    """Read a headed CSV the way MOSES does: target and ignored columns removed.

    `target` defaults to a column named `y`, else the last column.
    """
    with open(path, newline="") as fh:
        rows = [r for r in csv.reader(fh) if r and any(c.strip() for c in r)]
    header = [h.strip() for h in rows[0]]
    if target is None:
        target = "y" if "y" in header else header[-1]
    if target not in header:
        raise ValueError(f"target column {target!r} not in header of {path}")
    missing = [f for f in ignore if f not in header]
    if missing:
        raise ValueError(f"ignored features {missing} not in header of {path}")
    t_idx = header.index(target)
    keep = [i for i, h in enumerate(header) if i != t_idx and h not in ignore]
    X, y = [], []
    types = [type_node.unknown_type] * len(keep)
    for r in rows[1:]:
        X.append([float(r[i]) for i in keep])
        y.append(float(r[t_idx]))
        types = [infer_type_from_token2(t, r[i].strip()) for t, i in zip(types, keep, strict=True)]
    bools = frozenset(j for j, t in enumerate(types) if t == type_node.boolean_type)
    return Table([header[i] for i in keep], X, y, bools)


def _extract_expr(s: str, pos: int) -> tuple[str, int]:
    """Return the combo expression starting at `pos` (an atom or a balanced call)."""
    n = len(s)
    start = pos
    depth = 0
    while pos < n:
        c = s[pos]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return s[start : pos + 1], pos + 1
        elif c.isspace() and depth == 0:
            break
        pos += 1
    if depth != 0:
        raise ValueError(f"unbalanced parentheses in {s!r}")
    return s[start:pos], pos


def parse_candidate_line(line: str) -> tuple[float | None, str] | None:
    """Split one MOSES stdout line into (score or None, program); None if not a candidate."""
    s = line.strip()
    if not s or s.startswith(("n_evals", "#", "[")):
        return None
    score = None
    first, rest = (s.split(None, 1) + [""])[:2]
    if rest and _FLOAT_RE.match(first):
        score, s = float(first), rest.lstrip()
    prog, _ = _extract_expr(s, 0)
    if not prog:
        return None
    return score, prog


def parse_moses_output(text: str) -> list[tuple[float | None, str]]:
    """All candidate (score, program) pairs of a MOSES stdout, in printed order."""
    out = []
    for line in text.splitlines():
        try:
            c = parse_candidate_line(line)
        except ValueError:
            c = None
        if c is not None:
            out.append(c)
    return out


def relabel(program: str, labels: Sequence[str]) -> str:
    """Rewrite `-W1` labelled arguments (`$x3`) into positional ones (`$3`)."""
    index = {lab: i + 1 for i, lab in enumerate(labels)}

    def sub(m: re.Match) -> str:
        name = m.group(2)
        if name.isdigit():
            return m.group(0)
        if name not in index:
            raise ValueError(f"unknown argument label ${name}; labels are {list(labels)}")
        return f"{m.group(1)}${index[name]}"

    return _LABEL_RE.sub(sub, program)


def predict(
    program: str,
    X: Sequence[Sequence[float]],
    labels: Sequence[str] = (),
    bool_cols: frozenset[int] = frozenset(),
) -> list:
    """Evaluate `program` on every row with MOSES's semantics (nan on overflow).

    Columns in `bool_cols` are passed as logical values (non-zero is true), as MOSES does
    for a column it types as boolean."""
    tr = str_to_combo_tree(relabel(program, labels) if labels else program)
    it = tr.begin()
    # The mixed interpreter, as MOSES's own table scoring uses: regression programs can
    # hold boolean sub-terms such as `impulse(0<(...))`, which the pure contin
    # interpreter rejects.
    interp = mixed_interpreter.from_contin_inputs([])
    if bool_cols:
        interp = mixed_interpreter([])
    preds = []
    for row in X:
        if bool_cols:
            interp._mixed_inputs = [
                (builtin.logical_true if x != 0 else builtin.logical_false)
                if j in bool_cols
                else float(x)
                for j, x in enumerate(row)
            ]
        else:
            interp.contin_inputs = list(row)
        try:
            v = float(interp.contin_eval(it))
        except OverflowException:
            v = math.nan
        preds.append(v)
    return preds


def _mean(v: Sequence[float]) -> float:
    return math.fsum(v) / len(v)


def _var(v: Sequence[float]) -> float:
    m = _mean(v)
    return math.fsum((x - m) ** 2 for x in v) / len(v)


def _finite(v: Sequence[float]) -> bool:
    return all(math.isfinite(x) for x in v)


def linear_scaling(y: Sequence[float], p: Sequence[float]) -> tuple[float, float]:
    """Least-squares (a, b) minimising sum (y - (a + b p))^2 (Keijzer 2003)."""
    if not _finite(p):
        return math.nan, math.nan
    my, mp_ = _mean(y), _mean(p)
    vp = math.fsum((x - mp_) ** 2 for x in p)
    if vp == 0.0 or not math.isfinite(vp):
        return my, 0.0
    b = math.fsum((x - mp_) * (t - my) for x, t in zip(p, y, strict=True)) / vp
    return my - b * mp_, b


def nmse(y: Sequence[float], p: Sequence[float], a: float = 0.0, b: float = 1.0) -> float:
    """100 * MSE(y, a + b p) / var(y); inf if any scaled prediction is not finite."""
    q = [a + b * x for x in p]
    if not _finite(q):
        return math.inf
    mse = math.fsum((t - x) ** 2 for t, x in zip(y, q, strict=True)) / len(y)
    vy = _var(y)
    return 100.0 * mse / vy if vy > 0 else (0.0 if mse == 0 else math.inf)


def evaluate(program: str, train: Table, test: Table | None = None) -> dict:
    """Train-fitted linear scaling, then scaled and raw NMSE on each split."""
    p_tr = predict(program, train.X, train.labels, train.bool_cols)
    a, b = linear_scaling(train.y, p_tr)
    res = {
        "program": program,
        "a": a,
        "b": b,
        "train": {
            "nmse_scaled": nmse(train.y, p_tr, a, b),
            "nmse_raw": nmse(train.y, p_tr),
            "n_nonfinite": sum(not math.isfinite(x) for x in p_tr),
        },
    }
    if test is not None:
        if test.labels != train.labels:
            raise ValueError("train and test feature columns differ")
        # The model's input types are those of the table it was trained on.
        p_te = predict(program, test.X, test.labels, train.bool_cols)
        res["test"] = {
            "nmse_scaled": nmse(test.y, p_te, a, b),
            "nmse_raw": nmse(test.y, p_te),
            "n_nonfinite": sum(not math.isfinite(x) for x in p_te),
        }
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--train", required=True)
    ap.add_argument("--test")
    ap.add_argument("--target", help="target column (default: 'y', else the last column)")
    ap.add_argument("-Y", "--ignore-feature", action="append", default=[])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--program", help="combo program, optionally preceded by its score")
    src.add_argument("--moses-output", help="file with MOSES stdout ('-' for stdin)")
    ap.add_argument("--all", action="store_true", help="score every candidate line")
    args = ap.parse_args(argv)

    train = read_table(args.train, args.target, args.ignore_feature)
    test = read_table(args.test, args.target, args.ignore_feature) if args.test else None
    if args.program is not None:
        cands = [parse_candidate_line(args.program)]
    else:
        fh = sys.stdin if args.moses_output == "-" else open(args.moses_output)
        with fh:
            cands = parse_moses_output(fh.read())
    cands = [c for c in cands if c is not None]
    if not cands:
        print("no candidate program found", file=sys.stderr)
        return 2
    results = []
    for score, prog in cands if args.all else cands[:1]:
        r = evaluate(prog, train, test)
        r["moses_score"] = score
        results.append(r)
    json.dump(results if args.all else results[0], sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
