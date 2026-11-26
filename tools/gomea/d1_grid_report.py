"""Diagnostic D1 (backlog 15.4): does the contin knob grid limit the constants?

Usage (from the repository root; the script puts the repository on ``sys.path`` itself)::

    .venv/bin/python -I tools/gomea/d1_grid_report.py RESULTS [RESULTS ...] --out OUT_DIR
        [--arm ARM] [--no-lm]

``RESULTS`` is a results directory of ``moses_sr_runner.py`` (``OUTPUT_ROOT/<manifest name>``,
or its ``runs/`` directory, or one run directory). The runs must have the option
``--gomea-log-best-genes 1`` (``pymoses/gomea/best_genes.py``): each "end" event of their GOMEA
log (``events.jsonl``) then has the field ``best_genes`` of the deme's best program. A deme
without the field (an empty deme, or a run without the option) is left out and counted.

**What the script reads.** For each run directory: ``config.json`` (the data set, the arm, the
split and the MOSES command line), ``summary.json`` (the status and ``n_train``),
``events.jsonl`` (the "end" events) and ``train_scaled.csv``, the training table that MOSES
read (``-i``; the runner writes it, with the target scaled to y' = 10 (y - mean) / sd).

**a. The grid statistics** (no new computation): for each live constant gene of a deme's best
program (the class of ``rv_inert.analyse`` under the run's linear scaling; free codes are left
out), whether it is at a grid edge (a full-depth path in one direction, value = mean +- 31 x
step size), whether it is at the center (code 0: the exemplar's constant), its path length and
|value - mean| in step-size units.

**b. The LM probe** (offline; it reads the runs and writes only into ``OUT_DIR``):

* The scorer of the run, rebuilt as MOSES builds it: ``loadTable(train_scaled.csv, "y")``,
  ``contin_bscore`` with the run's linear scaling (``best_genes.linear_scaling``), and
  ``const_opt.RegressionRows.of_bscore`` (the rows that ``--const-opt lm`` uses).
* The reducer of the run: ``contin_reduction(-E, ignore ops)``, from the MOSES command line of
  ``config.json`` and the project's own option parser (``problem_params.add_options``). The
  check ``reduce_match`` compares the reduced text of each logged unreduced program with the
  logged reduced program.
* **Gene LM** (``lm_genes``): ``const_opt.levenberg_marquardt`` on the unreduced program, with
  only the leaves of the live genes as LM parameters (``tune``). At each pass the program is
  reduced first and then scored by the scorer's rule (with ``--linear-scaling terms``: the
  per-term least squares for the term weights, ``bscores.fit_program_terms``), so the objective
  is the training MSE of the program that the scorer sees. The LM settings are those of
  ``--const-opt lm`` (at most 10 iterations). The passes are not charged to anything.
* **Program LM** (``lm_program``): the LM of ``--const-opt lm`` at the end of a deme
  (``optimizer._tune_best``): ``levenberg_marquardt`` on every constant of the reduced program
  with the scorer's rows. Its gain is comparable with the LM records of rounds 13 and 14.
* For each live gene: the LM value; whether it is inside the grid range [mean - 31 x step,
  mean + 31 x step]; its distance to the nearest grid value, and that distance divided by the
  local grid spacing (the gap between the two grid values around it; 0 to 0.5), or, outside the
  range, its distance to the range in step units; and how far LM moved it, in step units.
* For each deme best: the training MSE before LM (``mse0``) and after, and the relative gain
  (mse0 - mse) / mse0 of the gene LM and of the program LM. The check ``raw_rel_diff``
  compares ``mse0`` with the deme's ``best.raw`` (``-raw / n_train``, the raw score of
  MOSES).

A deme best can be the deme's exemplar (``exemplar``: the all-zero genotype; every gene is then
at the center); ``demes.csv`` has the flag and ``summary.csv`` counts these demes
(``demes_exemplar``).

**Output** (in ``OUT_DIR``): ``demes.csv`` (one row per deme best), ``genes.csv`` (one row per
contin gene of a deme best), ``summary.csv`` (per data set and ``ALL``),
``distributions.csv`` (counts per bin, per data set and ``ALL``) and ``summary.txt`` (a short
text). ``--no-lm`` skips part b.

NOT A PORT: backlog 15.4, diagnostic D1.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pymoses.comboreduct.combo.iostream_combo import (  # noqa: E402
    str_to_combo_tree,
    tree_to_stream_str,
)
from pymoses.comboreduct.combo.vertex import combo_tree, get_contin  # noqa: E402
from pymoses.comboreduct.reduct import contin_reduction as _cr  # noqa: E402
from pymoses.comboreduct.reduct.clean_reduction import clean_reduce  # noqa: E402
from pymoses.comboreduct.table.table_io import loadTable  # noqa: E402
from pymoses.gomea.best_genes import (  # noqa: E402
    LS_PLAIN,
    LS_TERMS,
    PROGRAM_PRECISION,
    grid_values,
)
from pymoses.gomea.const_opt import (  # noqa: E402
    RegressionRows,
    constant_nodes,
    levenberg_marquardt,
    mse,
)
from pymoses.moses.representation.field_set import contin_spec  # noqa: E402
from pymoses.moses.scoring.bscores import contin_bscore  # noqa: E402

DATASET_ALL = "ALL"

# The bins of the distributions (labels in order).
OFFSET_BINS = [(0.0, 0.0, "0"), (0.0, 0.5, "(0,0.5]"), (0.5, 1.0, "(0.5,1]"),
               (1.0, 2.0, "(1,2]"), (2.0, 4.0, "(2,4]"), (4.0, 8.0, "(4,8]"),
               (8.0, 16.0, "(8,16]"), (16.0, 31.0, "(16,31]"), (31.0, math.inf, ">31")]  # fmt: skip
GAIN_BINS = [(-math.inf, 0.0, "<=0"), (0.0, 0.01, "(0,1%]"), (0.01, 0.05, "(1%,5%]"),
             (0.05, 0.2, "(5%,20%]"), (0.2, math.inf, ">20%")]  # fmt: skip
DIST_BINS = [(-math.inf, 0.1, "[0,0.1]"), (0.1, 0.25, "(0.1,0.25]"), (0.25, math.inf, "(0.25,0.5]")]


# ---- reading the runs


def run_dirs(roots: list[Path]) -> list[Path]:
    """The run directories under each root: the root itself when it has ``config.json``, else
    the directories of ``ROOT/runs`` (or of ``ROOT``) that have one. Sorted by name."""
    out: list[Path] = []
    for root in roots:
        if (root / "config.json").is_file():
            out.append(root)
            continue
        base = root / "runs" if (root / "runs").is_dir() else root
        out += sorted(d for d in base.iterdir() if (d / "config.json").is_file())
    return out


def end_events(path: Path) -> list[dict]:
    """The "end" events of a GOMEA log, in file order. The other events are read one line at a
    time and dropped (a long log stays out of memory)."""
    out = []
    with path.open() as fh:
        for line in fh:
            if '"type": "end"' not in line:
                continue
            e = json.loads(line)
            if e.get("type") == "end":
                out.append(e)
    return out


def moses_argv(config: dict) -> list[str]:
    """The MOSES options of a run: ``resolved_argv`` of ``config.json`` from its ``-i`` (the
    elements before it are the interpreter and the boot code of the runner)."""
    argv = [str(x) for x in config.get("resolved_argv") or []]
    if "-i" not in argv:
        raise ValueError("config.json has no resolved_argv with -i")
    return argv[argv.index("-i") :]


def reduct_settings(argv: list[str]) -> tuple[int, frozenset]:
    """``(reduct_candidate_effort, ignore_ops)`` of a MOSES command line, by the project's own
    option parser and ``problem_params._resolve_ignore_ops``."""
    from pymoses.moses.main.problem_params import problem_params

    pp = problem_params()
    parser = argparse.ArgumentParser(add_help=False)
    pp.add_options(parser)
    ns, _rest = parser.parse_known_args(argv)
    return int(ns.reduct_candidate_effort), frozenset(pp._resolve_ignore_ops(ns))


_REDUCER: tuple | None = None


def reducer(effort: int, ignore_ops: frozenset):
    """The candidate reducer of a run (``representation.clean_combo_tree(tr, True)``:
    ``clean_reduce``, then ``contin_reduction(effort, ignore_ops)``). ``contin_reduction`` keeps
    the rule of its first call for the whole process, so all the runs of one report must have
    the same settings (``ValueError`` otherwise)."""
    global _REDUCER
    if _REDUCER is None:
        _cr._reset_cache_for_testing()
        _REDUCER = ((effort, ignore_ops), _cr.contin_reduction(effort, set(ignore_ops)))
    elif _REDUCER[0] != (effort, ignore_ops):
        raise ValueError(
            f"the runs have different reduct settings: {_REDUCER[0]} and {(effort, ignore_ops)}"
        )
    rule = _REDUCER[1]

    def reduce(tr: combo_tree) -> combo_tree:
        out = combo_tree(tr.begin()) if tr.begin() is not None else combo_tree()
        clean_reduce(out)
        rule(out)
        return out

    return reduce


def scorer_rows(train_csv: Path, linear_scaling: str | None) -> RegressionRows:
    """The rows and the fit of the run's scorer, as MOSES builds them from ``-i train_scaled.csv
    -u y`` (``table_problems.common_setup``, ``contin_bscore(table)``) and as ``--const-opt lm``
    reads them (``RegressionRows.of_bscore``)."""
    table = loadTable(str(train_csv), "y")
    b = contin_bscore(table)
    if linear_scaling == LS_TERMS:
        b.set_linear_scaling_terms(True)
    elif linear_scaling == LS_PLAIN:
        b.set_linear_scaling(True)
    return RegressionRows.of_bscore(b)


class ReducedRows(RegressionRows):
    """``RegressionRows`` that reduce each program first (the gene LM works on the unreduced
    program; the scorer sees the reduced one)."""

    def __init__(self, base: RegressionRows, reduce) -> None:
        self.inputs = base.inputs
        self.targets = base.targets
        self.scaled = base.scaled
        self.terms = base.terms
        self.spread = base.spread
        self._reduce = reduce

    def residuals(self, tr: combo_tree, scaling=None):
        return super().residuals(self._reduce(tr), scaling)


# ---- the grid


def grid_position(value: float, grid: list[float], step: float) -> dict:
    """Where ``value`` lies on the sorted ``grid`` (see the module docstring)."""
    lo, hi = grid[0], grid[-1]
    nearest = min(grid, key=lambda x: abs(x - value))
    out = {
        "in_range": lo <= value <= hi,
        "nearest": nearest,
        "dist": abs(value - nearest),
        "spacing": None,
        "dist_rel": None,
        "outside_steps": None,
    }
    if out["in_range"]:
        i = min(max(bisect.bisect_right(grid, value) - 1, 0), len(grid) - 2)
        gap = grid[i + 1] - grid[i]
        out["spacing"] = gap
        out["dist_rel"] = out["dist"] / gap if gap > 0 else 0.0
    else:
        out["outside_steps"] = (value - hi if value > hi else lo - value) / step
    return out


def _spec(g: dict) -> contin_spec:
    return contin_spec(g["mean"], g["step_size"], g["expansion"], g["depth"])


# ---- one deme best


def _gain(mse0: float, mse1: float) -> float | None:
    if not (math.isfinite(mse0) and math.isfinite(mse1)) or mse0 <= 0:
        return None
    return (mse0 - mse1) / mse0


def probe(bg: dict, rows: RegressionRows | None, reduce, n_train: int, raw: float | None):
    """``(deme fields, {gene: LM fields})`` of one ``best_genes`` record. ``rows`` ``None``
    skips the LM probe."""
    genes = {g["gene"]: g for g in bg["genes"]}
    tree = str_to_combo_tree(bg["program"])
    leaves = constant_nodes(tree)
    leaf_genes = bg["leaf_genes"]
    d: dict = {
        "leaves": len(leaves),
        "leaves_match": len(leaves) == len(leaf_genes)
        and all(
            p is None or float(get_contin(n.data)) == genes[p]["value"]
            for n, p in zip(leaves, leaf_genes, strict=False)
        ),
    }
    red = reduce(tree)
    d["reduce_match"] = tree_to_stream_str(red, PROGRAM_PRECISION) == bg["program_reduced"]
    lm_genes: dict[int, dict] = {}
    if rows is None or not d["leaves_match"]:
        return d, lm_genes
    live = {p for p, g in genes.items() if g["class"] == "live"}
    flags = [p is not None and p in live for p in leaf_genes]
    rr = ReducedRows(rows, reduce)
    t0 = time.perf_counter()
    if any(flags):
        res = levenberg_marquardt(tree, rr, lambda: None, tune=flags)
        mse0, mse1 = res.mse0, res.mse
        d.update(lm_genes_status=res.status, lm_genes_iterations=res.iterations,
                 lm_genes_evals=res.evals, lm_genes_inert=res.n_inert)  # fmt: skip
        start = [float(get_contin(n.data)) for n, f in zip(leaves, flags, strict=True) if f]
        values = res.values if res.status == "improved" else start
        tuned = [p for p, f in zip(leaf_genes, flags, strict=True) if f]
        for p, v in zip(tuned, values, strict=True):
            lm_genes[p] = {"lm_value": float(v)}
    else:
        mse0 = mse1 = mse(rr.residuals(tree)[0])
        d.update(lm_genes_status="no_live", lm_genes_iterations=0, lm_genes_evals=1,
                 lm_genes_inert=0)  # fmt: skip
    d["lm_genes_s"] = time.perf_counter() - t0
    d.update(mse0=mse0, mse_lm_genes=mse1, gain_lm_genes=_gain(mse0, mse1))
    d["mse_log"] = -raw / n_train if raw is not None and n_train else None
    d["raw_rel_diff"] = (
        abs(mse0 - d["mse_log"]) / abs(d["mse_log"])
        if d["mse_log"] not in (None, 0.0) and math.isfinite(mse0)
        else None
    )
    t0 = time.perf_counter()
    resp = levenberg_marquardt(str_to_combo_tree(bg["program_reduced"]), rows, lambda: None)
    d.update(lm_program_status=resp.status, lm_program_constants=resp.n_constants,
             mse0_program=resp.mse0, mse_lm_program=resp.mse,
             gain_lm_program=_gain(resp.mse0, resp.mse),
             lm_program_s=time.perf_counter() - t0)  # fmt: skip
    for p, rec in lm_genes.items():
        g = genes[p]
        if g["free"]:
            continue
        pos = grid_position(rec["lm_value"], grid_values(_spec(g)), g["step_size"])
        rec.update(
            lm_in_range=pos["in_range"],
            lm_nearest_grid=pos["nearest"],
            lm_dist_grid=pos["dist"],
            lm_local_spacing=pos["spacing"],
            lm_dist_rel=pos["dist_rel"],
            lm_outside_steps=pos["outside_steps"],
            lm_move_steps=(rec["lm_value"] - g["value"]) / g["step_size"],
            lm_offset_steps=(rec["lm_value"] - g["mean"]) / g["step_size"],
        )
    return d, lm_genes


# ---- all runs


DEME_FIELDS = [
    "dataset", "arm", "split", "run_id", "deme", "termination_reason", "requests", "raw",
    "exemplar", "n_genes", "n_live", "n_absorbed", "n_dead", "n_zero", "n_absent", "n_free",
    "n_live_edge", "n_live_center", "linear_scaling", "fit", "leaves", "leaves_match",
    "reduce_match", "mse_log", "mse0", "raw_rel_diff", "lm_genes_status",
    "lm_genes_iterations", "lm_genes_evals", "lm_genes_inert", "mse_lm_genes",
    "gain_lm_genes", "lm_program_status", "lm_program_constants", "mse0_program",
    "mse_lm_program", "gain_lm_program", "lm_genes_s", "lm_program_s", "program",
    "program_reduced",
]  # fmt: skip
GENE_FIELDS = [
    "dataset", "arm", "split", "deme", "gene", "class", "code", "value", "mean", "step_size",
    "expansion", "depth", "free", "path", "length", "edge", "center", "offset_steps",
    "lm_value", "lm_in_range", "lm_nearest_grid", "lm_dist_grid", "lm_local_spacing",
    "lm_dist_rel", "lm_outside_steps", "lm_move_steps", "lm_offset_steps",
]  # fmt: skip


def collect(dirs: list[Path], arm: str | None, lm: bool, log=print):
    """``(demes, genes, runs)``: the rows of ``demes.csv`` and ``genes.csv`` and one record per
    run (its status and the demes left out)."""
    demes: list[dict] = []
    genes: list[dict] = []
    runs: list[dict] = []
    for rd in dirs:
        config = json.loads((rd / "config.json").read_text())
        if arm is not None and config.get("arm") != arm:
            continue
        summary_path = rd / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        run = {"run_id": config.get("run_id", rd.name), "dataset": config.get("task"),
               "arm": config.get("arm"), "split": config.get("seed"),
               "status": summary.get("status"), "demes": 0, "demes_without_field": 0,
               "demes_with_error": 0}  # fmt: skip
        runs.append(run)
        events_path = rd / "events.jsonl"
        if not events_path.is_file():
            run["status"] = f"{run['status']}; no events.jsonl"
            continue
        ends = end_events(events_path)
        n_train = summary.get("n_train")
        rows = None
        if lm and any(isinstance(e.get("best_genes"), dict) for e in ends):
            ls = next(e["best_genes"].get("linear_scaling") for e in ends
                      if isinstance(e.get("best_genes"), dict))  # fmt: skip
            rows = scorer_rows(rd / "train_scaled.csv", ls)
            n_train = n_train or len(rows.targets)
        reduce = reducer(*reduct_settings(moses_argv(config)))
        t0 = time.perf_counter()
        for k, e in enumerate(ends):
            run["demes"] += 1
            bg = e.get("best_genes")
            if not isinstance(bg, dict):
                run["demes_without_field"] += 1
                continue
            if "error" in bg:
                run["demes_with_error"] += 1
                continue
            raw = (e.get("best") or {}).get("raw")
            d, lm_genes = probe(bg, rows, reduce, n_train, raw)
            classes = [g["class"] for g in bg["genes"]]
            live = [g for g in bg["genes"] if g["class"] == "live" and not g["free"]]
            demes.append({
                "dataset": run["dataset"], "arm": run["arm"], "split": run["split"],
                "run_id": run["run_id"], "deme": k,
                "termination_reason": e.get("termination_reason"),
                "requests": (e.get("ledger") or {}).get("requests"), "raw": raw,
                "exemplar": bg.get("exemplar"),
                "n_genes": len(classes), "n_live": classes.count("live"),
                "n_absorbed": classes.count("absorbed"), "n_dead": classes.count("dead"),
                "n_zero": classes.count("zero"), "n_absent": classes.count("absent"),
                "n_free": sum(g["free"] for g in bg["genes"]),
                "n_live_edge": sum(g["edge"] for g in live),
                "n_live_center": sum(g["length"] == 0 for g in live),
                "linear_scaling": bg.get("linear_scaling"), "fit": bg.get("fit"),
                "program": bg["program"], "program_reduced": bg["program_reduced"],
                **d,
            })  # fmt: skip
            for g in bg["genes"]:
                row = {
                    "dataset": run["dataset"], "arm": run["arm"], "split": run["split"],
                    "deme": k, **{f: g.get(f) for f in ("gene", "class", "code", "value",
                    "mean", "step_size", "expansion", "depth", "free", "path", "length",
                    "edge")},
                    "center": (not g["free"]) and g["length"] == 0,
                    "offset_steps": (g["value"] - g["mean"]) / g["step_size"],
                }  # fmt: skip
                row.update(lm_genes.get(g["gene"], {}))
                genes.append(row)
        log(f"{run['run_id']}: {run['demes']} demes, {time.perf_counter() - t0:.1f} s")
    return demes, genes, runs


# ---- summaries


def _q(xs: list[float], p: float) -> float | None:
    """The p-quantile (linear interpolation, ``statistics.quantiles`` method inclusive)."""
    xs = sorted(x for x in xs if x is not None and math.isfinite(x))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = p * (len(xs) - 1)
    i = int(math.floor(pos))
    j = min(i + 1, len(xs) - 1)
    return xs[i] + (xs[j] - xs[i]) * (pos - i)


def _share(n: int, d: int) -> float | None:
    return n / d if d else None


def _bin(x: float, bins) -> str:
    for lo, hi, label in bins:
        if lo == hi == x or lo < x <= hi or (lo == -math.inf and x <= hi):
            return label
    return bins[-1][2]


def summarize(demes: list[dict], genes: list[dict], runs: list[dict]):
    """``(summary rows, distribution rows)`` per data set and ``ALL``."""
    datasets = sorted({r["dataset"] for r in runs if r["dataset"] is not None})
    summary, dist = [], []
    for ds in datasets + [DATASET_ALL]:
        sel = lambda r, ds=ds: ds == DATASET_ALL or r["dataset"] == ds  # noqa: E731
        R = [r for r in runs if sel(r)]
        D = [d for d in demes if sel(d)]
        G = [g for g in genes if sel(g)]
        live = [g for g in G if g["class"] == "live" and not g["free"]]
        lm = [g for g in live if g.get("lm_value") is not None]
        inside = [g for g in lm if g.get("lm_in_range")]
        gains = [d.get("gain_lm_genes") for d in D if d.get("n_live")]
        gains = [x for x in gains if x is not None]
        pgains = [x for x in (d.get("gain_lm_program") for d in D) if x is not None]
        offs = [abs(g["offset_steps"]) for g in live]
        summary.append({
            "dataset": ds, "runs": len(R),
            "runs_ok": sum(r["status"] == "ok" for r in R),
            "demes": len(D),
            "demes_left_out": sum(r["demes_without_field"] + r["demes_with_error"] for r in R),
            "demes_with_live": sum(1 for d in D if d["n_live"]),
            "demes_exemplar": sum(1 for d in D if d.get("exemplar")),
            "genes": len(G), "live": len(live),
            "absorbed": sum(g["class"] == "absorbed" for g in G),
            "dead": sum(g["class"] == "dead" for g in G),
            "zero": sum(g["class"] == "zero" for g in G),
            "absent": sum(g["class"] == "absent" for g in G),
            "free": sum(bool(g["free"]) for g in G),
            "live_edge": sum(bool(g["edge"]) for g in live),
            "share_live_edge": _share(sum(bool(g["edge"]) for g in live), len(live)),
            "share_live_center": _share(sum(bool(g["center"]) for g in live), len(live)),
            "mean_live_length": statistics.fmean(g["length"] for g in live) if live else None,
            "median_live_abs_offset_steps": _q(offs, 0.5),
            "p90_live_abs_offset_steps": _q(offs, 0.9),
            "max_live_abs_offset_steps": max(offs) if offs else None,
            "live_lm": len(lm),
            "share_lm_in_range": _share(len(inside), len(lm)),
            "median_lm_dist_rel": _q([g["lm_dist_rel"] for g in inside], 0.5),
            "share_lm_dist_rel_gt_0.25": _share(
                sum(g["lm_dist_rel"] > 0.25 for g in inside), len(inside)),
            "median_abs_lm_move_steps": _q([abs(g["lm_move_steps"]) for g in lm], 0.5),
            "p90_abs_lm_move_steps": _q([abs(g["lm_move_steps"]) for g in lm], 0.9),
            "demes_gene_lm": len(gains),
            "median_gain_lm_genes": _q(gains, 0.5),
            "p90_gain_lm_genes": _q(gains, 0.9),
            "share_gain_lm_genes_gt_5pct": _share(sum(x > 0.05 for x in gains), len(gains)),
            "median_gain_lm_program": _q(pgains, 0.5),
            "p90_gain_lm_program": _q(pgains, 0.9),
            "share_gain_lm_program_gt_5pct": _share(sum(x > 0.05 for x in pgains), len(pgains)),
            "reduce_match": _share(sum(bool(d["reduce_match"]) for d in D), len(D)),
            "leaves_match": _share(sum(bool(d["leaves_match"]) for d in D), len(D)),
            "max_raw_rel_diff": max((d["raw_rel_diff"] for d in D
                                     if d.get("raw_rel_diff") is not None), default=None),
        })  # fmt: skip

        def add(quantity: str, labels: list[str], values: list[str], ds=ds) -> None:
            for label in labels:
                n = values.count(label)
                dist.append({"dataset": ds, "quantity": quantity, "bin": label, "count": n,
                             "share": _share(n, len(values))})  # fmt: skip

        depth = max((g["depth"] for g in live), default=5)
        add("live_path_length", [str(k) for k in range(depth + 1)],
            [str(g["length"]) for g in live])  # fmt: skip
        add("live_abs_offset_steps", [b[2] for b in OFFSET_BINS],
            [_bin(x, OFFSET_BINS) for x in offs])  # fmt: skip
        add("lm_dist_rel", [b[2] for b in DIST_BINS] + ["outside"],
            [_bin(g["lm_dist_rel"], DIST_BINS) if g.get("lm_in_range") else "outside"
             for g in lm])  # fmt: skip
        add("gain_lm_genes", [b[2] for b in GAIN_BINS], [_bin(x, GAIN_BINS) for x in gains])
        add("gain_lm_program", [b[2] for b in GAIN_BINS], [_bin(x, GAIN_BINS) for x in pgains])
    return summary, dist


def _fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.3g}"
    return str(x)


def summary_text(summary: list[dict], runs: list[dict], lm: bool) -> str:
    """A short text of the summary (descriptive numbers only)."""
    lines = [
        "D1 grid report (backlog 15.4). Live genes: constant genes of the deme bests that a",
        "change can move the training error (rv_inert classes); free codes left out.",
        f"Runs: {len(runs)}; LM probe: {'on' if lm else 'off'}.",
        "",
        "dataset  runs demes live  edge%  center%  |off|med  " + (
            "inRange%  distRel med  gainGenes med  gainProg med" if lm else ""),
    ]  # fmt: skip
    for s in summary:
        pct = lambda x: "-" if x is None else f"{100 * x:.1f}"  # noqa: E731
        row = (f"{s['dataset']:<8} {s['runs']:>4} {s['demes']:>5} {s['live']:>5} "
               f"{pct(s['share_live_edge']):>6} {pct(s['share_live_center']):>8} "
               f"{_fmt(s['median_live_abs_offset_steps']):>9}")  # fmt: skip
        if lm:
            row += (f"  {pct(s['share_lm_in_range']):>8} {_fmt(s['median_lm_dist_rel']):>12} "
                    f"{pct(s['median_gain_lm_genes']):>14} "
                    f"{pct(s['median_gain_lm_program']):>13}")  # fmt: skip
        lines.append(row)
    a = summary[-1] if summary else {}
    lines += [
        "",
        "Checks (ALL): reduce_match "
        + _fmt(a.get("reduce_match"))
        + ", leaves_match "
        + _fmt(a.get("leaves_match"))
        + ", max raw_rel_diff "
        + _fmt(a.get("max_raw_rel_diff"))
        + ", demes left out "
        + _fmt(a.get("demes_left_out"))
        + ".",
        "Gains are relative training-MSE gains (mse0 - mse) / mse0; percentages are shares.",
    ]
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", default=None, help="only the runs of this arm")
    ap.add_argument("--no-lm", action="store_true", help="skip the LM probe (part b)")
    args = ap.parse_args(argv)
    dirs = run_dirs(args.results)
    if not dirs:
        print("no run directory found", file=sys.stderr)
        return 1
    lm = not args.no_lm
    demes, genes, runs = collect(dirs, args.arm, lm)
    summary, dist = summarize(demes, genes, runs)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "demes.csv", demes, DEME_FIELDS)
    write_csv(args.out / "genes.csv", genes, GENE_FIELDS)
    write_csv(args.out / "summary.csv", summary)
    write_csv(args.out / "distributions.csv", dist,
              ["dataset", "quantity", "bin", "count", "share"])  # fmt: skip
    text = summary_text(summary, runs, lm)
    (args.out / "summary.txt").write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
