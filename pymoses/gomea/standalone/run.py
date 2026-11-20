"""NOT A PORT: the IT linkage study (BACKLOG 13.9). The stand-alone run: CLI, engine call and
outputs (plan section 6).

Usage (from the repository root)::

    python -m pymoses.gomea.standalone \\
        --genome {template,it} --fos {uni,rt,lt-mi,lt-b,np} \\
        --train train.csv --val val.csv --test test.csv [--target y] \\
        --pop 1000 --max-evals 500000 --seed SPLIT_SEED \\
        [--template-height 4] [--it-terms 8 --it-factors 2 --it-powers=-2,-1,1,2] \\
        --out RUN_DIR [--arm NAME --task NAME --run-id ID] [--no-column-cache] \\
        [--max-generations G] [--stall-generations K]

``--train``, ``--val`` and ``--test`` are the ORIGINAL split files. The search uses the
training rows with the target scaled as the runner scales it, y' = 10 (y - mean) / sd (the
training mean and sd; ``scaled_target``). The reported NMSE values come from
``sr_eval.evaluate`` on the original splits, as for every arm of the rounds.

**The run.** A master ``MT19937RandGen(seed)`` seeds three streams in this order: start
population, linkage, variation (``pymoses/gomea/optimizer.py``, ``_draw_streams``). The start
population (``init.py``) is evaluated, then ``GomeaEngine`` runs with ``forced_improvement``
off (GP-GOMEA at 750220c has no forced improvements: ``GOMVariator::GOM`` and
``GOMEAGenerationHandler::PerformGeneration`` only do GOM), ``skip_inactive`` on,
acceptance ``nonworse`` and no generation or stall limit, so only the budget E or convergence
ends the run.

**The generation limit (opt-in).** NOT A PORT: the IT linkage study (BACKLOG 13.9).
``--max-generations G`` sets ``EngineParams.max_generations`` to G. The engine then stops after
G GOM generations after the start population (end reason ``max_generations``), as GP-GOMEA's
``--generations G`` does (``IMSHandler.cpp``: the start population is not a generation; each
``macro_generation`` does one ``DoGeneration``). The budget E or convergence can still end the
run first. Without the option, the limit is ``BIG`` and the outputs do not change. With the
option, ``max_generations`` is also in the "start" record's ``params`` and in the
``standalone`` block of ``summary.json``.

**The stall rule (opt-in).** NOT A PORT: the IT linkage study (BACKLOG 13.9, server round 15).
``--stall-generations K`` ends the run with the end reason ``stalled`` when K generations in a
row made no new evaluation: the evaluation count did not grow during the generation (score-cache
hits, no-op skips and phenotype skips do not count). A population with one phenotype can keep
different genotypes (GOM accepts neutral changes of inactive positions), so "all genomes equal"
can fail to happen, and cache hits do not use the budget E; without a rule such a run goes on
until the wall cap. ``RegressionEvaluator.check`` applies the rule; the engine calls it once at
the start of each generation, after its own checks of the generation before (convergence and the
generation limit come first). Without the option, ``check`` is as before and the outputs do not
change. With the option, ``stall_generations`` is also in the "start" record's ``params`` and in
the ``standalone`` block of ``summary.json``, with ``stall_count`` (the count at the end).

**Outputs** in ``RUN_DIR``: ``summary.json`` (the runner's summary keys where they apply, plus a
``standalone`` block), ``candidates.json`` (the one reported model) and ``events.jsonl`` (a
"start" record, one "generation" record per generation, ``telemetry.py``, and an "end"
record with the best-so-far trace). The reported model is the best training score of the
final population (no selection on validation).

**Cross-check.** ``run.py`` also scores the reported model with its own copies of the
``sr_eval`` formulas (the genome's arithmetic on the original splits). The train, validation,
test and clipped test NMSE must agree with ``sr_eval.evaluate`` on ``to_moses`` to a relative
1e-9 (equal values, or both not finite, also agree). Otherwise the run fails with an error.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

from pymoses.gomea.engine import EngineParams, GomeaEngine
from pymoses.gomea.fos import LinkageModel
from pymoses.gomea.gom import Individual
from pymoses.gomea.oracle import BudgetExhausted
from pymoses.gomea.standalone import positions
from pymoses.gomea.standalone._tools import sr_eval
from pymoses.gomea.standalone.evaluator import (
    ITScorer,
    RegressionEvaluator,
    TemplateScorer,
    TrainData,
    clip_range,
    linear_scaling,
    nmse,
    nmse_clipped,
    variance,
)
from pymoses.gomea.standalone.genome_it import DEFAULT_POWERS, ITGenome, nodes_of
from pymoses.gomea.standalone.genome_template import TemplateGenome
from pymoses.gomea.standalone.init import diversity, it_population, template_population
from pymoses.gomea.standalone.telemetry import LinkageTelemetry
from pymoses.moses.scoring.least_squares import fit_terms, predict_terms
from pymoses.util.rng import MT19937RandGen

GENOMES = ("template", "it")
ANYTIME_FRACTIONS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
CROSSCHECK_RTOL = 1e-9
BIG = 10**9  # "no limit" for max_generations and stall_generations


class CrossCheckError(RuntimeError):
    """The genome's own NMSE and ``sr_eval``'s NMSE of the reported model differ."""


def scaled_target(ys) -> tuple[list[float], float, float]:
    """``(y', mean, sd)``: the runner's scaling (``moses_sr_runner._scale_train``, the same
    expressions), y' = 10 (y - mean) / sd."""
    mean = sum(ys) / len(ys)
    sd = math.sqrt(sum((y - mean) ** 2 for y in ys) / len(ys))
    if not sd > 0:
        raise ValueError("the training target is constant; it cannot be scaled")
    return [10.0 * (y - mean) / sd for y in ys], mean, sd


def table_columns(table, bool_inputs) -> list[list[float]]:
    """The input columns of a ``sr_eval.Table``. A boolean input (typed on the training table)
    is 1.0 where the value is not 0 and 0.0 elsewhere, as ``sr_eval`` passes it (a logical
    value through ``impulse``)."""
    cols = [list(c) for c in zip(*table.X, strict=True)] if table.X else []
    for j in bool_inputs:
        cols[j] = [1.0 if v != 0 else 0.0 for v in cols[j]]
    return cols


def _agree(a, b) -> bool:
    if a is None or b is None:
        return a is b
    if a == b:
        return True
    if not (math.isfinite(a) and math.isfinite(b)):
        return (math.isnan(a) and math.isnan(b)) or (math.isinf(a) and math.isinf(b))
    return abs(a - b) <= CROSSCHECK_RTOL * max(abs(a), abs(b))


def build_genome(args, train):
    n = len(train.labels)
    bool_inputs = frozenset(train.bool_cols)
    if args.genome == "template":
        return TemplateGenome(n, args.template_height, bool_inputs)
    cols = table_columns(train, ())
    zero = frozenset(j for j in range(n) if any(v == 0.0 for v in cols[j]))
    zero_one = frozenset(j for j in range(n) if all(v in (0.0, 1.0) for v in cols[j]))
    return ITGenome(n, args.it_terms, args.it_factors, args.it_powers, zero, zero_one,
                    bool_inputs)


# ---- the genome's own scoring of the reported model (the sr_eval formulas)


def _own_template(genome, program, tables, bool_inputs) -> dict:
    train, val, test = tables
    p = {k: genome.predict(program, table_columns(t, bool_inputs))
         for k, t in zip(("train", "val", "test"), tables, strict=True)}
    a, b = linear_scaling(train.y, p["train"])
    lo, hi = clip_range(train.y)
    return {
        "train": nmse(train.y, p["train"], a, b),
        "val": nmse(val.y, p["val"], a, b),
        "test": nmse(test.y, p["test"], a, b),
        "test_clipped": nmse_clipped(test.y, [a + b * x for x in p["test"]], lo, hi),
    }


def _it_program_output(genome, kept, cols, n) -> list[float]:
    """The output of the printed model (``+`` sums from 0, left to right)."""
    if not kept:
        return [0.0] * n
    tcols = [genome.monomial_column(m, cols) for m in kept]
    if len(tcols) == 1:
        return tcols[0]
    out = []
    for r in range(n):
        s = 0.0
        for c in tcols:
            s += c[r]
        out.append(s)
    return out


def _own_it(genome, kept, tables, bool_inputs) -> dict:
    train, val, test = tables
    cols = [table_columns(t, bool_inputs) for t in tables]
    p = [_it_program_output(genome, kept, c, len(t.y)) for c, t in zip(cols, tables, strict=True)]
    a, b = linear_scaling(train.y, p[0])
    lo, hi = clip_range(train.y)
    out = {
        "train": nmse(train.y, p[0], a, b),
        "val": nmse(val.y, p[1], a, b),
        "test": nmse(test.y, p[2], a, b),
    }
    q_test = [a + b * x for x in p[2]]
    if len(kept) >= 2:
        tc = [[genome.monomial_column(m, c) for m in kept] for c in cols]
        fit = fit_terms(train.y, tc[0])
        if fit.weights is not None:
            out["train"] = nmse(train.y, fit.fitted)
            out["val"] = nmse(val.y, predict_terms(fit.weights, tc[1]))
            q_test = predict_terms(fit.weights, tc[2])
            out["test"] = nmse(test.y, q_test)
    out["test_clipped"] = nmse_clipped(test.y, q_test, lo, hi)
    return out


# ---- the run


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pymoses.gomea.standalone",
                                 description="Stand-alone GOMEA run of the IT linkage study.")
    ap.add_argument("--genome", required=True, choices=GENOMES)
    ap.add_argument("--fos", required=True, choices=LinkageModel.MODES)
    ap.add_argument("--train", required=True, help="the original training split (CSV)")
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--target", default="y")
    ap.add_argument("--pop", type=int, default=1000)
    ap.add_argument("--max-evals", type=int, default=500000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--template-height", type=int, default=4)
    ap.add_argument("--it-terms", type=int, default=8)
    ap.add_argument("--it-factors", type=int, default=2)
    ap.add_argument("--it-powers", default=",".join(str(p) for p in DEFAULT_POWERS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--task", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-column-cache", action="store_true",
                    help="IT: compute every term column again for each model (same results)")
    ap.add_argument("--it-cache-max-columns", type=int, default=None)
    ap.add_argument("--it-cache-max-pairs", type=int, default=1_000_000)
    # NOT A PORT: the IT linkage study (BACKLOG 13.9). Opt-in; None keeps the limit BIG.
    ap.add_argument("--max-generations", type=int, default=None,
                    help="stop after G GOM generations after the start population "
                         "(GP-GOMEA --generations); default: no generation limit")
    # NOT A PORT: the IT linkage study (BACKLOG 13.9, server round 15). Opt-in; None: no rule.
    ap.add_argument("--stall-generations", type=int, default=None,
                    help="stop (end reason stalled) when K generations in a row made no new "
                         "evaluation; default: no stall rule")
    args = ap.parse_args(argv)
    args.it_powers = tuple(int(x) for x in str(args.it_powers).split(",") if x.strip())
    if args.pop < 2:
        ap.error("--pop must be at least 2")
    if args.max_evals < args.pop:
        ap.error("--max-evals must be at least --pop (the start population is evaluated)")
    if args.max_generations is not None and args.max_generations < 1:
        ap.error("--max-generations must be at least 1")
    if args.stall_generations is not None and args.stall_generations < 1:
        ap.error("--stall-generations must be at least 1")
    return args


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def run(args) -> dict:
    """One stand-alone run; returns the summary (also written to ``RUN_DIR/summary.json``)."""
    t0, c0 = time.monotonic(), time.process_time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    se = sr_eval()
    tables = tuple(se.read_table(p, args.target) for p in (args.train, args.val, args.test))
    train = tables[0]
    if not train.labels == tables[1].labels == tables[2].labels:
        raise ValueError("train, val and test feature columns differ")
    y_scaled, mean, sd = scaled_target(train.y)
    genome = build_genome(args, train)
    bool_inputs = frozenset(train.bool_cols)
    data = TrainData(table_columns(train, bool_inputs), y_scaled)
    is_it = genome.kind == "it"
    if is_it:
        scorer = ITScorer(genome, data, column_cache=not args.no_column_cache,
                          max_columns=args.it_cache_max_columns,
                          max_pairs=args.it_cache_max_pairs)
    else:
        scorer = TemplateScorer(genome, data)
    E = args.max_evals
    checkpoints = [max(1, round(f * E)) for f in ANYTIME_FRACTIONS]
    stall = getattr(args, "stall_generations", None)  # the opt-in stall rule (round 15)
    ev = (RegressionEvaluator(genome, scorer, E, checkpoints) if stall is None else
          RegressionEvaluator(genome, scorer, E, checkpoints, stall_generations=stall))
    var_y = variance(y_scaled)  # of the search target: the NMSE of a search score

    master = MT19937RandGen(args.seed)
    rng_init = MT19937RandGen(master.randint())
    rng_link = MT19937RandGen(master.randint())
    rng_var = MT19937RandGen(master.randint())
    make = it_population if is_it else template_population
    genotypes, init_rec = make(genome, args.pop, rng_init)
    start = {**diversity(genotypes, genome.domains), **init_rec}

    if is_it:
        parent, node_of = positions.it_tree(genome.terms, genome.factors)
        related = lambda p, q: positions.it_related(p, q, genome.factors)  # noqa: E731
    else:
        parent, node_of = positions.template_tree(genome.height)
        related = positions.template_related
    distance = (positions.distance_fn(positions.distance_matrix(parent, node_of))
                if args.fos == "np" else None)

    events = (out / "events.jsonl").open("w")
    try:
        tele = LinkageTelemetry(genome, ev, var_y, related, out=events)
        params = {"genome": genome.kind, "fos": args.fos, "pop": args.pop, "max_evals": E,
                  "seed": args.seed, "length": genome.length, "domains": list(genome.domains)}
        # The opt-in generation limit (BACKLOG 13.9); without it, no key and the limit BIG.
        max_gens = getattr(args, "max_generations", None)
        if max_gens is not None:
            params["max_generations"] = max_gens
        if stall is not None:
            params["stall_generations"] = stall
        tele.write({"type": "start", "params": params, "start": start})
        pop = []
        try:
            for g in genotypes:
                s = ev.evaluate(g)
                pop.append(Individual(g, s, ev.phenotype_key(g)))
        except BudgetExhausted as exc:  # cannot happen: max_evals >= pop is checked
            raise ValueError("the budget ran out in the start population") from exc
        start["init_evals"] = ev.n_evals
        start["init_cache_hits"] = ev.cache_hits
        eparams = EngineParams(fos_mode=args.fos, forced_improvement=False,
                               skip_inactive=True,
                               max_generations=BIG if max_gens is None else max_gens,
                               stall_generations=BIG, acceptance="nonworse")
        eng = GomeaEngine(genome.domains, eparams, ev, rng_var, rng_link,
                          on_generation=tele.on_generation, distance=distance,
                          on_trial=tele.on_trial)
        tele.engine = eng
        res = eng.run(pop)

        best = res.best
        if is_it:
            model = genome.decode(best.genotype)
            fit = scorer.fit(model.monomials)
            kept = fit.kept
            program = genome.model_to_moses(kept)
            own = _own_it(genome, kept, tables, bool_inputs)
        else:
            _, post = genome.decode(best.genotype)
            program = genome.program_to_moses(post)
            own = _own_template(genome, post, tables, bool_inputs)
        on_val = se.evaluate(program, train, tables[1], terms=is_it)
        on_test = se.evaluate(program, train, tables[2], terms=is_it, clip=True)
        ref = {"train": on_val["train"]["nmse_scaled"], "val": on_val["test"]["nmse_scaled"],
               "test": on_test["test"]["nmse_scaled"],
               "test_clipped": on_test["test"]["nmse_clipped"]}
        bad = {k: (own[k], ref[k]) for k in ref if not _agree(own[k], ref[k])}
        search_nmse = 100.0 * (-best.score) / var_y if math.isfinite(best.score) else math.inf
        cross = {
            "rtol": CROSSCHECK_RTOL,
            "own": own,
            "sr_eval": ref,
            "passed": not bad,
            "search_train_nmse": search_nmse,
            "search_rel_diff": (abs(search_nmse - ref["train"]) / abs(ref["train"])
                                if ref["train"] and math.isfinite(search_nmse)
                                and math.isfinite(ref["train"]) else None),
        }
        if bad:
            raise CrossCheckError(f"cross-check failed for {program!r}: own vs sr_eval {bad}")

        trace = [[n, 100.0 * (-s) / var_y if math.isfinite(s) else None] for n, s in ev.trace]
        anytime = [{"evals": c, "frac": f, "best_train_nmse": None}
                   for c, f in zip(checkpoints, ANYTIME_FRACTIONS, strict=True)]
        reached = dict(ev.anytime)
        last = 100.0 * (-ev.best) / var_y if math.isfinite(ev.best) else None
        for a in anytime:
            if a["evals"] in reached:
                s = reached[a["evals"]]
                a["best_train_nmse"] = 100.0 * (-s) / var_y if math.isfinite(s) else None
                a["reached"] = True
            else:  # the run ended before this checkpoint: the final best so far
                a["best_train_nmse"], a["reached"] = last, False
        tele.write({"type": "end", "n_evals": ev.n_evals, "cache_hits": ev.cache_hits,
                    "generations": res.generations,
                    "termination_reason": res.termination_reason,
                    "trace": trace, "anytime": anytime})
    finally:
        events.close()

    sizes = ({"template_height": genome.height} if not is_it else
             {"it_terms": genome.terms, "it_factors": genome.factors,
              "it_powers": list(genome.powers)})
    standalone = {
        "genome": genome.kind, "fos": args.fos, "pop": args.pop, "max_evals": E,
        "length": genome.length, **sizes,
        "forced_improvement": False, "acceptance": "nonworse", "skip_inactive": True,
        "start": start,
        "noop_skips": ev.noop_skips, "phenotype_skips": ev.phenotype_skips,
        "best_score": best.score,
    }
    if max_gens is not None:
        standalone["max_generations"] = max_gens
    if stall is not None:
        standalone["stall_generations"] = stall
        standalone["stall_count"] = ev.stall_count
    summary = {
        "run_id": args.run_id, "task": args.task or Path(args.train).parent.name,
        "arm": args.arm or f"{genome.kind}-{args.fos}", "seed": args.seed,
        "status": "ok", "error": None, "exit_code": 0,
        "n_train": len(train.y), "target_mean": mean, "target_sd": sd,
        "final_program": program,
        "train_nmse": ref["train"], "val_nmse": ref["val"], "test_nmse": ref["test"],
        "test_nmse_clipped": ref["test_clipped"],
        "first_train_nmse": ref["train"], "first_test_nmse": ref["test"],
        "n_candidates": 1, "n_candidates_unscorable": 0,
        "n_evals": ev.n_evals, "cache_hits": ev.cache_hits,
        "generations": res.generations, "termination_reason": res.termination_reason,
        "anytime": anytime, "crosscheck": cross, "standalone": standalone,
    }
    if is_it:
        # the per-term fit is used (the rule of run_table.terms_size)
        terms_fit = on_val["terms"]["n_terms"] >= 2 and on_val["terms"]["fallback"] is False
        nodes = nodes_of(program)
        summary.update(
            nodes=nodes,
            size=nodes + (len(kept) if terms_fit and len(kept) >= 2 else 0),
            n_terms=len(kept),
            n_weights=(len(kept) + 1) if terms_fit else (2 if kept else 1),
            terms=on_val["terms"],
        )
        standalone.update(
            fit_mode=fit.mode, n_terms_decoded=len(model.monomials),
            dropped_constant=fit.dropped_constant, dropped_aliased=fit.dropped_aliased,
            forced_negative_evaluated=ev.forced_negative, forced_negative_reported=model.forced,
            column_cache=not args.no_column_cache, scorer_stats=dict(scorer.stats),
        )
    else:
        n_active = len(post)
        summary.update(nodes=n_active, size=n_active, final_program_nodes=n_active,
                       final_program_infix=genome.program_to_infix(post))
    cand = {"program": program, "train_nmse": ref["train"], "val_nmse": ref["val"],
            "test_nmse": ref["test"]}
    if is_it:
        cand["terms"] = on_val["terms"]
    (out / "candidates.json").write_text(json.dumps([cand], indent=2))
    summary.update(wall_s=time.monotonic() - t0, cpu_s=time.process_time() - c0,
                   ru_maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    _write_json(out / "summary.json", summary)
    return summary


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "summary.json", {
            "run_id": args.run_id, "task": args.task, "arm": args.arm, "seed": args.seed,
            "status": "error", "error": f"{type(exc).__name__}: {exc}"[:300], "exit_code": 1,
            "train_nmse": None, "val_nmse": None, "test_nmse": None, "test_nmse_clipped": None,
            "final_program": None, "n_evals": None,
        })
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
