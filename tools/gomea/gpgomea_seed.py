"""Seed MOSES with small GP-GOMEA elites (backlog item 9).

Usage (from the repository root)::

    .venv/bin/python tools/gomea/gpgomea_seed.py OUT_JSON --binary GPGOMEA_SEED \\
        --budget B --height H --seed S --k K \\
        (--bool "-H mux -k 3" | --regression TRAIN_CSV [--target y]) \\
        [--popsize P] [--functions F] [--gpgomea-argv "..."] [--workdir DIR]

1. Write the problem as a GP-GOMEA data file: a Boolean demo problem's full truth table
   (built by the very functions MOSES scores against, as `make_truth_tables.py` does), or
   a regression CSV exactly as MOSES reads it (target column moved last, the other
   columns in file order, so MOSES's `$k` is GP-GOMEA's `x(k-1)`). For regression, give
   the file MOSES is run on: `moses_sr_runner.py` runs MOSES on `train_scaled.csv`
   (target rescaled to sd 10), and the seeds must fit that target.
2. Run GP-GOMEA (`build_gpgomea_seed.sh`'s `gpgomea-seed`, which dumps every IMS run's
   elitist and the active populations at termination) with `--evaluations B`, a fixed
   tree template of height H (`--initmaxtreeheight H`; at most 2**(H+1)-1 nodes with
   binary functions) and a function set MOSES can express: AND/OR/NOT for Boolean;
   +, -, *, p/ for regression by default (plog, exp, sin, ^2, cos, NAND, NOR also
   translate; aq, XOR, sqrt, alog do not).
3. Translate the dumped programs, best GP-GOMEA fitness first, into combo strings. The
   regression form writes GP-GOMEA's linear scaling into the tree: `+(a *(b f))`
   (`a` alone when b == 0). Constants are written with `repr` (17 significant digits).
4. Verify each translation with pyMoses's OWN scorer: the problem is set up through
   `moses_exec` with the same MOSES flags, and the `behave_cscore` it builds is captured
   just before the search starts (no search runs). The raw score must equal GP-GOMEA's
   fitness on the same rows (Boolean: -wrong rows = -round(error rate * rows);
   regression: -SSE = -MSE * rows) within `--tol` (relative; pyMoses scores are
   float32). Programs are deduplicated by their reduced form (the reduction MOSES
   applies to every exemplar, `si_ca`), and the first K distinct verified ones are kept.
   Programs that do not translate or do not match are listed under `rejected`.

`evaluations_used` is GP-GOMEA's own final `Evaluations:` count (GP-GOMEA checks its
limit only between generations, so it can exceed B by up to one generation); a shared
budget gives MOSES `-m (M - evaluations_used)`. The pyMoses verification scores are not
counted: they are checks, and MOSES scores the exemplars again when it starts.

Protected operators: GP-GOMEA's `p/` is sign(b) * a / (1e-6 + |b|) and `plog` is
log|x| (0 where not finite); MOSES's `/` and `log` are unprotected. They are translated
to `/` and `log`, which agree whenever the divisor is far from 0 and the log argument
is positive on every row; otherwise the score check rejects the seed.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import resource
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

EMPTY_EXEMPLAR = {"bool": "and", "regression": "+"}
DEFAULT_FUNCTIONS = {"bool": "AND_OR_NOT", "regression": "+_-_*_p/"}
DEFAULT_POPSIZE = {"bool": 64, "regression": 32}


class Untranslatable(ValueError):
    """A GP-GOMEA operator with no MOSES equivalent (e.g. `aq`, `XOR`)."""


# ---------------------------------------------------------------------------
# problem files
# ---------------------------------------------------------------------------


def bool_rows(flags: list[str]) -> tuple[list[tuple[bool, ...]], list[bool]]:
    """The full truth table of a MOSES Boolean demo problem (`-H pa|maj|mux -k K`)."""
    import itertools

    import make_truth_tables as mtt

    fam = flags[flags.index("-H") + 1]
    k = flags[flags.index("-k") + 1]
    _name, (_flags, func, arity) = mtt._spec(f"{fam}:{k}")
    xs = list(itertools.product([False, True], repeat=arity))
    return xs, [bool(func(bits)) for bits in xs]


def write_bool_dat(flags: list[str], path: Path) -> int:
    xs, ys = bool_rows(flags)
    path.write_text(
        "".join(
            " ".join(str(int(b)) for b in x) + f" {int(y)}\n" for x, y in zip(xs, ys, strict=True)
        )
    )
    return len(xs)


def write_regression_dat(csv_path: Path, path: Path, target: str = "y") -> int:
    """CSV (with header) -> GP-GOMEA's headerless, space-separated file, target last."""
    with open(csv_path, newline="") as fh:
        rows = [r for r in csv.reader(fh) if r and any(c.strip() for c in r)]
    header = [h.strip() for h in rows[0]]
    k = header.index(target)
    lines = []
    for r in rows[1:]:
        feats = [c.strip() for i, c in enumerate(r) if i != k]
        lines.append(" ".join(feats + [r[k].strip()]) + "\n")
    path.write_text("".join(lines))
    return len(lines)


# ---------------------------------------------------------------------------
# GP-GOMEA output
# ---------------------------------------------------------------------------


def parse_dump(text: str) -> list[dict]:
    """The `SEED_ELITE` / `SEED_POP` lines of `gpgomea-seed`'s stdout."""
    out = []
    for line in text.splitlines():
        if not line.startswith(("SEED_ELITE\t", "SEED_POP\t")):
            continue
        tag, run, fit, a, b, expr = line.split("\t", 5)
        out.append(
            {
                "source": "elite" if tag == "SEED_ELITE" else "population",
                "run": int(run),
                "fit": float(fit),
                "a": float(a),
                "b": float(b),
                "expr": expr.strip(),
            }
        )
    return out


def parse_evaluations(text: str) -> int | None:
    for line in text.splitlines():
        if line.startswith("Evaluations:"):
            return int(float(line.split()[-1]))
    return None


def parse_prefix(expr: str):
    """`(+ x0 (* 2.5 x3))` -> ('+', ['x0', ('*', ['2.5', 'x3'])]); leaves are strings."""
    toks = expr.replace("(", " ( ").replace(")", " ) ").split()
    pos = 0

    def rec():
        nonlocal pos
        tok = toks[pos]
        pos += 1
        if tok != "(":
            return tok
        op = toks[pos]
        pos += 1
        args = []
        while toks[pos] != ")":
            args.append(rec())
        pos += 1
        return (op, args)

    tree = rec()
    if pos != len(toks):
        raise ValueError(f"trailing tokens in {expr!r}")
    return tree


def count_nodes(tree) -> int:
    return 1 if isinstance(tree, str) else 1 + sum(count_nodes(c) for c in tree[1])


# ---------------------------------------------------------------------------
# translation to combo
# ---------------------------------------------------------------------------


def _var(tok: str) -> str:
    if not (tok.startswith("x") and tok[1:].isdigit()):
        raise Untranslatable(f"unknown terminal {tok!r}")
    return f"${int(tok[1:]) + 1}"


def _num(x: float) -> str:
    if not math.isfinite(x):
        raise Untranslatable(f"non-finite constant {x!r}")
    return repr(float(x))


def bool_to_combo(tree) -> str:
    if isinstance(tree, str):
        return _var(tree)
    op, args = tree
    sub = [bool_to_combo(a) for a in args]
    if op == "AND":
        return f"and({' '.join(sub)})"
    if op == "OR":
        return f"or({' '.join(sub)})"
    if op == "NOT":
        (s,) = sub
        return "!" + s if s.startswith("$") else f"not({s})"
    if op == "NAND":
        return f"not(and({' '.join(sub)}))"
    if op == "NOR":
        return f"not(or({' '.join(sub)}))"
    raise Untranslatable(f"Boolean operator {op!r} has no MOSES equivalent")


def regression_to_combo(tree) -> str:
    if isinstance(tree, str):
        if tree.startswith("x") and tree[1:].isdigit():
            return _var(tree)
        try:
            return _num(float(tree))
        except ValueError:
            raise Untranslatable(f"unknown terminal {tree!r}") from None
    op, args = tree
    sub = [regression_to_combo(a) for a in args]
    if op == "+":
        return f"+({sub[0]} {sub[1]})"
    if op == "-":
        return f"+({sub[0]} *(-1.0 {sub[1]}))"
    if op == "*":
        return f"*({sub[0]} {sub[1]})"
    if op == "p/":  # protected in GP-GOMEA, plain in MOSES: checked by the score match
        return f"/({sub[0]} {sub[1]})"
    if op == "plog":  # log|x| in GP-GOMEA, log x in MOSES: checked by the score match
        return f"log({sub[0]})"
    if op == "exp":
        return f"exp({sub[0]})"
    if op == "sin":
        return f"sin({sub[0]})"
    if op == "cos":
        return f"sin(+({sub[0]} {_num(math.pi / 2)}))"
    if op == "^2":
        return f"*({sub[0]} {sub[0]})"
    raise Untranslatable(f"regression operator {op!r} has no MOSES equivalent")


def scaled_regression_to_combo(tree, a: float, b: float) -> str:
    """GP-GOMEA's linear scaling a + b * f(x), written into the tree."""
    if b == 0.0:
        return _num(a)
    return f"+({_num(a)} *({_num(b)} {regression_to_combo(tree)}))"


def translate(kind: str, cand: dict) -> str:
    tree = parse_prefix(cand["expr"])
    if kind == "bool":
        return bool_to_combo(tree)
    return scaled_regression_to_combo(tree, cand["a"], cand["b"])


def _matches(raw: float | None, expected: float, tol: float) -> bool:
    return raw is not None and abs(raw - expected) <= tol * max(1.0, abs(expected))


def expected_raw_score(kind: str, fit: float, rows: int) -> float:
    """GP-GOMEA's fitness as the pyMoses raw score of the same program."""
    if kind == "bool":
        return -float(round(fit * rows))
    return -fit * rows


# ---------------------------------------------------------------------------
# pyMoses's own scorer
# ---------------------------------------------------------------------------


class _Captured(Exception):
    pass


def pymoses_scorer(moses_flags: list[str]):
    """Set the problem up through `moses_exec(moses_flags)` and return the
    `(behave_cscore, si_ca)` pair it would hand to `metapop_moses_results`;
    the search itself is never started."""
    import pymoses.moses.main.demo_problems as dp
    import pymoses.moses.main.table_problems as tp
    from pymoses.moses.main.moses_exec import moses_exec

    got: dict = {}

    def capture(bases, type_sig, si_ca, si_kb, c_scorer, *args, **kwargs):
        got.update(scorer=c_scorer, si_ca=si_ca)
        raise _Captured

    saved = (dp.metapop_moses_results, tp.metapop_moses_results)
    dp.metapop_moses_results = tp.metapop_moses_results = capture
    try:
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            try:
                moses_exec(["moses", *moses_flags, "-f", str(Path(tmp) / "moses.log")])
            except _Captured:
                pass
    finally:
        dp.metapop_moses_results, tp.metapop_moses_results = saved
    if "scorer" not in got:
        raise RuntimeError(f"moses_exec {moses_flags} never reached metapop_moses_results")
    return got["scorer"], got["si_ca"]


def score_programs(moses_flags: list[str], programs: list[str], scorer=None) -> list[dict]:
    """pyMoses raw score and reduced form of each combo string. `scorer` is a
    `(behave_cscore, si_ca)` pair to use instead of setting the problem up
    from `moses_flags` (re-seeding passes the running MOSES's own)."""
    from pymoses.comboreduct.combo.iostream_combo import combo_tree_to_str, str_to_combo_tree
    from pymoses.comboreduct.combo.vertex import combo_tree

    scorer, si_ca = scorer if scorer is not None else pymoses_scorer(moses_flags)
    out = []
    for p in programs:
        tr = str_to_combo_tree(p)
        try:
            raw = float(scorer.get_cscore(tr).get_score())
        except Exception as exc:  # e.g. OverflowException: MOSES scores it as the worst
            out.append({"raw": None, "reduced": None, "error": f"{type(exc).__name__}"})
            continue
        red = combo_tree(tr.begin())
        si_ca(red)
        out.append({"raw": raw, "reduced": combo_tree_to_str(red).strip()})
    return out


# ---------------------------------------------------------------------------
# the whole step
# ---------------------------------------------------------------------------


def gpgomea_argv(
    kind: str,
    binary: str,
    data: Path,
    budget: int,
    height: int,
    seed: int,
    popsize: int | None = None,
    functions: str | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    argv = [binary, "--prob", "accuracy" if kind == "bool" else "symbreg"]
    argv += ["--train", str(data), "--test", str(data)]
    argv += ["--functions", functions or DEFAULT_FUNCTIONS[kind]]
    if kind == "regression":
        argv += ["--linearscaling", "--erc"]
    argv += ["--gomea", "--initmaxtreeheight", str(height), "--maxtreeheight", str(height)]
    argv += ["--popsize", str(popsize or DEFAULT_POPSIZE[kind])]
    argv += ["--evaluations", str(budget), "--generations", "-1"]
    argv += ["--seed", str(seed), "--parallel", "1"]
    return argv + list(extra or [])


def make_seeds(
    kind: str,
    binary: str,
    budget: int,
    height: int,
    seed: int,
    k: int,
    workdir: Path,
    bool_flags: list[str] | None = None,
    train_csv: Path | None = None,
    target: str = "y",
    popsize: int | None = None,
    functions: str | None = None,
    extra: list[str] | None = None,
    tol: float = 1e-5,
    max_checks: int | None = None,
    scorer=None,
) -> dict:
    """Run GP-GOMEA, translate and verify its k best distinct elites. Returns the
    record written to the seeds JSON (see the module docstring). `scorer`: see
    `score_programs`."""
    workdir.mkdir(parents=True, exist_ok=True)
    if os.sep in binary:  # GP-GOMEA runs in `workdir`: pin a relative path first
        binary = str(Path(binary).resolve())
    data = workdir / "gpgomea_train.dat"
    if kind == "bool":
        rows = write_bool_dat(bool_flags, data)
        moses_flags = list(bool_flags)
    else:
        rows = write_regression_dat(Path(train_csv), data, target)
        moses_flags = ["-i", str(train_csv), "-u", target]
    argv = gpgomea_argv(kind, binary, data, budget, height, seed, popsize, functions, extra)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.monotonic()
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        cwd=workdir,  # GP-GOMEA writes result.txt / stats_generations.txt to its cwd
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    wall = time.monotonic() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    (workdir / "gpgomea_stdout.txt").write_text(proc.stdout)
    (workdir / "gpgomea_stderr.txt").write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"GP-GOMEA exited {proc.returncode}: {proc.stderr[-500:]}")
    evals = parse_evaluations(proc.stdout)
    dump = parse_dump(proc.stdout)
    if evals is None or not dump:
        raise RuntimeError(
            "GP-GOMEA output has no Evaluations line or no SEED_* dump "
            "(is the binary build_gpgomea_seed.sh's gpgomea-seed?)"
        )

    # best GP-GOMEA fitness first; smaller programs first among ties
    for c in dump:
        c["nodes"] = count_nodes(parse_prefix(c["expr"]))
    dump.sort(key=lambda c: (c["fit"] if math.isfinite(c["fit"]) else math.inf, c["nodes"]))
    seen_expr: set[str] = set()
    pending: list[dict] = []
    rejected: list[dict] = []
    for c in dump:
        if not math.isfinite(c["fit"]):
            continue
        key = f"{c['expr']}|{c['a']!r}|{c['b']!r}"
        if key in seen_expr:
            continue
        seen_expr.add(key)
        try:
            c["combo"] = translate(kind, c)
        except Untranslatable as exc:
            rejected.append({**c, "reason": f"untranslatable: {exc}"})
            continue
        pending.append(c)
    max_checks = max_checks if max_checks is not None else max(8 * k, 40)
    pending = pending[:max_checks]
    scored = score_programs(moses_flags, [c["combo"] for c in pending], scorer) if pending else []

    seeds: list[dict] = []
    seen_reduced: set[str] = set()
    for c, s in zip(pending, scored, strict=True):
        expected = expected_raw_score(kind, c["fit"], rows)
        ok = _matches(s["raw"], expected, tol)
        diff = None if s["raw"] is None else abs(s["raw"] - expected)
        rec = {
            "combo": c["combo"],
            "reduced": s["reduced"],
            "gpgomea_expr": c["expr"],
            "gpgomea_fit": c["fit"],
            "a": c["a"],
            "b": c["b"],
            "gpgomea_nodes": c["nodes"],
            "source": c["source"],
            "expected_raw": expected,
            "pymoses_raw": s["raw"],
            "abs_diff": diff,
        }
        if not ok:
            reason = "score mismatch" if s["raw"] is not None else f"pyMoses: {s['error']}"
            rejected.append({**rec, "reason": reason})
            continue
        if s["reduced"] in seen_reduced or s["reduced"] == EMPTY_EXEMPLAR[kind]:
            continue
        seen_reduced.add(s["reduced"])
        if len(seeds) < k:
            seeds.append(rec)
    return {
        "kind": kind,
        "problem": {"bool_flags": bool_flags}
        if kind == "bool"
        else {"train_csv": str(train_csv), "target": target},
        "rows": rows,
        "moses_flags": moses_flags,
        "k": k,
        "height": height,
        "seed": seed,
        "budget": budget,
        "tol": tol,
        "gpgomea": {
            "argv": argv,
            "evaluations_used": evals,
            "cpu_s": cpu,
            "wall_s": wall,
            "best_fit": dump[0]["fit"],
            "n_dumped": len(dump),
            "n_checked": len(pending),
        },
        "evaluations_used": evals,
        "seeds": seeds,
        "n_score_matches": sum(
            _matches(s["raw"], expected_raw_score(kind, c["fit"], rows), tol)
            for c, s in zip(pending, scored, strict=True)
        ),
        "rejected": rejected[:50],
        "n_rejected": len(rejected),
        "empty_exemplar": EMPTY_EXEMPLAR[kind],
        "exemplar_argv": exemplar_argv(kind, [s["combo"] for s in seeds]),
    }


def exemplar_argv(kind: str, combos: list[str], with_empty: bool = True) -> list[str]:
    """`-e SEED ... -e EMPTY`: the seeds in rank order, then the default exemplar."""
    out: list[str] = []
    for c in combos:
        out += ["-e", c]
    if with_empty:
        out += ["-e", EMPTY_EXEMPLAR[kind]]
    return out


def shared_budget(total: int, used: int) -> int:
    """MOSES's `-m` once GP-GOMEA's evaluations come out of the total."""
    if used >= total:
        raise ValueError(f"seeding used {used} evaluations, the whole budget of {total}")
    return total - used


def replace_max_evals(argv: list[str], value: int) -> list[str]:
    """`argv` with every `-m N` / `--max-evals N` set to `value` (exactly one must exist)."""
    out = list(argv)
    hits = [i for i, t in enumerate(out) if t in ("-m", "--max-evals")]
    if len(hits) != 1 or hits[0] + 1 >= len(out):
        raise ValueError(f"expected exactly one -m in {argv}")
    out[hits[0] + 1] = str(value)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_json")
    ap.add_argument("--binary", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--height", type=int, default=3)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--k", type=int, default=5)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bool", help='MOSES demo-problem flags, e.g. "-H mux -k 3"')
    g.add_argument("--regression", help="training CSV exactly as MOSES reads it")
    ap.add_argument("--target", default="y")
    ap.add_argument("--popsize", type=int, default=None)
    ap.add_argument("--functions", default=None)
    ap.add_argument("--gpgomea-argv", default="", help="extra GP-GOMEA flags (one string)")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args(argv)
    out = Path(args.out_json).resolve()
    workdir = Path(args.workdir) if args.workdir else out.parent / (out.stem + "_gpgomea")
    rec = make_seeds(
        "bool" if args.bool else "regression",
        str(Path(args.binary).resolve()),
        args.budget,
        args.height,
        args.seed,
        args.k,
        workdir.resolve(),
        bool_flags=shlex.split(args.bool) if args.bool else None,
        train_csv=Path(args.regression).resolve() if args.regression else None,
        target=args.target,
        popsize=args.popsize,
        functions=args.functions,
        extra=shlex.split(args.gpgomea_argv),
        tol=args.tol,
    )
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print(
        f"{len(rec['seeds'])} seeds, {rec['evaluations_used']} GP-GOMEA evaluations, "
        f"best fit {rec['gpgomea']['best_fit']}, {rec['n_rejected']} rejected -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
