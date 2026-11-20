"""The IT linkage study (BACKLOG 13.9, plan sections 6.2 and 8.1): ``to_moses`` scored by
``sr_eval.evaluate`` equals the genome's own NMSE (template with ``aq``; IT with
``terms=True``), on train, validation and test, also with the clipped test NMSE. The data has
a boolean input (``impulse``), an input with zeros (the negative-power rule) and a two-valued
input (aliased IT columns)."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest
from standalone_data import write_split

from pymoses.gomea.standalone import run as srun
from pymoses.gomea.standalone.evaluator import ITScorer, TrainData

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "gomea"))
import sr_eval  # noqa: E402


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    d = write_split(tmp_path_factory.mktemp("cc") / "s1", 80, 30, 30, seed=4)
    return tuple(sr_eval.read_table(str(d / f"{p}.csv")) for p in ("train", "val", "test"))


def _args(genome):
    return type("A", (), {"genome": genome, "template_height": 4, "it_terms": 8,
                          "it_factors": 2, "it_powers": (-2, -1, 1, 2)})()


def _ref(program, tables, terms):
    train, val, test = tables
    on_val = sr_eval.evaluate(program, train, val, terms=terms)
    on_test = sr_eval.evaluate(program, train, test, terms=terms, clip=True)
    return {"train": on_val["train"]["nmse_scaled"], "val": on_val["test"]["nmse_scaled"],
            "test": on_test["test"]["nmse_scaled"],
            "test_clipped": on_test["test"]["nmse_clipped"]}


def test_the_data_has_the_special_inputs(tables):
    train = tables[0]
    assert train.bool_cols == frozenset({4})
    g = srun.build_genome(_args("it"), train)
    assert g.zero_inputs == frozenset({2, 4}) and g.zero_one_inputs == frozenset({4})


def test_template_own_nmse_equals_sr_eval(tables):
    genome = srun.build_genome(_args("template"), tables[0])
    r = random.Random(12)
    n_finite = 0
    for _ in range(60):
        g = tuple(r.randrange(k) for k in genome.domains)
        _, post = genome.decode(g)
        program = genome.program_to_moses(post)
        own = srun._own_template(genome, post, tables, tables[0].bool_cols)
        ref = _ref(program, tables, terms=False)
        for k in ref:
            assert srun._agree(own[k], ref[k]), (program, k, own[k], ref[k])
            assert own[k] == ref[k] or (math.isinf(own[k]) and math.isinf(ref[k]))
        n_finite += math.isfinite(ref["train"])
    assert n_finite > 40
    assert any("exp(*(0.5 log(" in genome.to_moses(tuple(r.randrange(k) for k in
                                                          genome.domains)) for _ in range(20))


def test_it_own_nmse_equals_sr_eval_with_terms(tables):
    train = tables[0]
    genome = srun.build_genome(_args("it"), train)
    y, _, _ = srun.scaled_target(train.y)
    scorer = ITScorer(genome, TrainData(srun.table_columns(train, train.bool_cols), y))
    r = random.Random(13)
    modes = set()
    for _ in range(80):
        m = genome.decode(tuple(r.randrange(k) for k in genome.domains)).monomials
        fit = scorer.fit(m)
        modes.add(fit.mode)
        program = genome.model_to_moses(fit.kept)
        own = srun._own_it(genome, fit.kept, tables, train.bool_cols)
        ref = _ref(program, tables, terms=True)
        for k in ref:
            assert own[k] == ref[k] or (math.isinf(own[k]) and math.isinf(ref[k])), (
                program, k, own[k], ref[k])
    assert "terms" in modes


def test_it_aliased_terms_are_not_printed(tables):
    """x1, x4, x4^2 and 1/x4 (x4 has two values): the printed model has the kept terms
    only, so sr_eval's per-term fit does not fall back."""
    train = tables[0]
    genome = srun.build_genome(_args("it"), train)
    y, _, _ = srun.scaled_target(train.y)
    scorer = ITScorer(genome, TrainData(srun.table_columns(train, train.bool_cols), y))
    m = (((0, 1),), ((3, -1),), ((3, 1),), ((3, 2),))
    fit = scorer.fit(m)
    assert fit.mode == "terms" and len(fit.kept) == 2 and fit.dropped_aliased == 2
    program = genome.model_to_moses(fit.kept)
    on_val = sr_eval.evaluate(program, train, tables[1], terms=True)
    assert on_val["terms"]["fallback"] is False and on_val["terms"]["n_terms"] == 2
    # the full model would fall back in sr_eval
    full = sr_eval.evaluate(genome.model_to_moses(m), train, tables[1], terms=True)
    assert full["terms"]["fallback"] is True


def test_the_agreement_rule():
    assert srun._agree(1.0, 1.0 + 1e-12) and not srun._agree(1.0, 1.0 + 1e-8)
    assert srun._agree(math.inf, math.inf) and srun._agree(0.0, 0.0)
    assert not srun._agree(math.inf, 1.0) and not srun._agree(None, 1.0)
