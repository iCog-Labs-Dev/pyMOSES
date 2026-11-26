"""NOT A PORT, backlog 15.4 (D1b): the option ``--gomea-contin-start step|wide``.

* ``step`` (the default) is the code before the option, bit for bit: the populations of
  ``build_population`` for fixed seeds have the digests of the code at commit ddace3a, with
  and without the argument, and the full runs of ``test_rv_step_identity.py`` with
  ``--gomea-contin-start step`` give the references of ``data/rv_step_2d407b2/``.
* ``wide``: each sampler draw goes through ``widen_contin_genes``, in order, before the
  duplicate check. The genes that differ from the exemplar are the sampler's genes; each
  changed contin gene gets a legal code that is not 0; the codes with a path of length 2 or
  more occur; the 62 non-center paths are drawn about uniformly. Every genotype decodes.
* A schema without contin genes: ``wide`` gives the ``step`` population and makes no draw.
  A stub schema with discrete genes: ``widen_contin_genes`` does not change them.
* The optimizer and the CLI: the "init" event and the parameters of "invocation_start" get
  ``contin_start`` with ``wide`` only; the refusals of ``gomea_parameters``, the hybrid and
  the CLI.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymoses.comboreduct.reduct.logical_reduction import logical_reduction
from pymoses.comboreduct.type_checker.type_signature import gen_signature, type_node
from pymoses.gomea import population as popmod
from pymoses.gomea.optimizer import gomea_optimization, gomea_parameters
from pymoses.gomea.population import (
    CONTIN_START_MODES,
    build_population,
    parse_radii,
    widen_contin_genes,
)
from pymoses.gomea.schema import FrozenSchema
from pymoses.moses.moses.types import composite_score
from pymoses.moses.representation.instance_set import instance_set
from pymoses.moses.representation.representation import representation
from pymoses.util.rng import MT19937RandGen, randGen

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import test_rv_step_identity as ident  # noqa: E402

MIX = "1:0.4,2:0.3,4:0.2,max:0.1"
WIDE = ["--gomea-contin-start", "wide"]
STEP = ["--gomea-contin-start", "step"]


def _contin_schema() -> FrozenSchema:
    """27 contin genes of depth 5 (``$1``, two contin inputs, linear contin knobs)."""
    randGen().seed(1)
    rep = representation(
        lambda _t: None,
        lambda _t: None,
        str_to_combo_tree("$1"),
        gen_signature(type_node.contin_type, 2),
        linear_contin=True,
    )
    return FrozenSchema(rep, contin="copy")


def _bool_schema() -> FrozenSchema:
    """7 discrete genes, no contin gene."""
    randGen().seed(1)
    rule = logical_reduction(frozenset())(2)
    rep = representation(
        rule,
        rule,
        str_to_combo_tree("or(and($1 $2) $3)"),
        gen_signature(type_node.boolean_type, 3),
        perm_ratio=-1.0,
    )
    return FrozenSchema(rep)


def _digest(init) -> str:
    text = json.dumps(
        {
            "g": [list(g) for g in init.genotypes],
            "r": init.radii,
            "a": init.attempts,
            "d": init.duplicates_rejected,
            "c": init.coverage_replacements,
        },
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode()).hexdigest()


def _path(code: int, depth: int) -> str:
    """The ``L``/``R`` steps of a contin code up to the first Stop; ``ValueError`` for a code
    that is not a legal path (a digit 3, or a step after a Stop)."""
    digits = [(code >> (2 * k)) & 3 for k in range(depth)]
    if code >> (2 * depth):
        raise ValueError(f"code {code} is out of range")
    steps = []
    for k, d in enumerate(digits):
        if d == 0:
            if any(digits[k:]):
                raise ValueError(f"code {code} has a step after a Stop")
            break
        if d == 3:
            raise ValueError(f"code {code} has an invalid digit")
        steps.append("L" if d == 1 else "R")
    return "".join(steps)


# ---- step: the code before the option, bit for bit

# The digests, the next draw of ``rng`` and the next draw of ``randGen()`` after
# ``build_population``, from the code at commit ddace3a (before the option).
GOLDEN = {
    ("contin", 1, 16): (
        "70eeaa09e002626fc8cc6643ff8032840f4ef740782fc0cd4762796a4146d72c",
        1978431015,
        1612356713,
    ),
    ("contin", 2, 100): (
        "21e8c32c48c25453b8a7ba094812da3f945d217716c493496d06530b84ed83fe",
        131286109,
        722627063,
    ),
    ("contin", 3, 40): (
        "c12aab32a515342098ca294748c9fce7ebb712a243465b5b2f5f3806bdd6c27b",
        1093632249,
        1925023640,
    ),
    ("bool", 1, 16): (
        "235ca7b45e41a772f7e6091e53711dd48b5694e2d5ac8efa67aa9fa2514d214e",
        233508934,
        1273598499,
    ),
    ("bool", 2, 100): (
        "5910c74694716280abf15b214f90ff339ef360f452084fd940fe876d93ae5ded",
        670624776,
        385887259,
    ),
    ("bool", 3, 40): (
        "732836fb01d01fcbe5372ffb9347af1c07f0509d07c2a15d31209030a2830428",
        1876500095,
        1569183752,
    ),
}


@pytest.mark.parametrize("key", sorted(GOLDEN))
@pytest.mark.parametrize("kw", [{}, {"contin_start": "step"}], ids=["absent", "step"])
def test_step_is_the_population_of_the_code_before_the_option(key, kw):
    name, seed, n = key
    schema = _contin_schema() if name == "contin" else _bool_schema()
    randGen().seed(seed)
    rng = MT19937RandGen(seed + 10)
    init = build_population(schema, n, parse_radii(MIX), rng, **kw)
    assert (_digest(init), rng.randint(), randGen().randint()) == GOLDEN[key]


@pytest.mark.parametrize("case", ["base-tls-info", "base-tls-m3000-r2"])
def test_step_on_the_cli_gives_the_references(case, tmp_path):
    """The round 14 BASE-TLS runs of ``test_rv_step_identity.py`` with ``--gomea-contin-start
    step``: stdout, the MOSES log (without the command line) and the GOMEA log are the
    references of commit 2d407b2."""
    stdout, log, events = ident.run_case(ident.CASES[case] + STEP, tmp_path)
    assert stdout == (ident.REF / f"{case}.out").read_text()
    assert log.replace(" " + " ".join(STEP), "") == (ident.REF / f"{case}.log").read_text()
    assert events == (ident.REF / f"{case}.events.jsonl").read_text()


# ---- wide: the sampler's genes, new values


def _wide_recorded(schema, n, seed, monkeypatch):
    """``build_population`` with ``wide``; each sampler draw (as a genotype) and each call of
    ``widen_contin_genes`` (input, output) are recorded."""
    draws: list[tuple[int, ...]] = []
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    real_sampler = popmod.sample_from_neighborhood
    real_widen = popmod.widen_contin_genes

    def sampler(fields, r, k, out, n_out, center):
        real_sampler(fields, r, k, out, n_out, center)
        draws.append(schema.to_genotype(out[-1]))

    def widen(s, g, rng):
        res = real_widen(s, g, rng)
        calls.append((g, res))
        return res

    monkeypatch.setattr(popmod, "sample_from_neighborhood", sampler)
    monkeypatch.setattr(popmod, "widen_contin_genes", widen)
    randGen().seed(seed)
    init = build_population(
        schema, n, parse_radii(MIX), MT19937RandGen(seed + 10), contin_start="wide"
    )
    monkeypatch.undo()
    return init, draws, calls


@pytest.mark.parametrize("seed, n", [(1, 16), (2, 100), (3, 40)])
def test_wide_keeps_the_sampler_genes_and_draws_new_values(seed, n, monkeypatch):
    schema = _contin_schema()
    init, draws, calls = _wide_recorded(schema, n, seed, monkeypatch)
    assert len(init.genotypes) == n
    assert len(draws) == len(calls) == init.attempts  # every draw, before any check
    assert [g for g, _ in calls] == draws  # in order, the sampler's own genotypes
    contin = set(schema.contin_positions)
    outputs = set()
    for before, after in calls:
        outputs.add(after)
        for p, (b, a) in enumerate(zip(before, after, strict=True)):
            if p not in contin or b == 0:
                assert a == b  # unchanged genes and discrete genes follow the sampler
            else:
                assert len(_path(b, 5)) == 1  # the sampler's one step
                assert a != 0 and 0 <= a < schema.domains[p]
                assert 1 <= len(_path(a, schema.contin_depths[p])) <= 5
    assert init.genotypes[0] == schema.default_genotype()
    assert set(init.genotypes[1:]) <= outputs
    assert len(set(init.genotypes)) == n  # the duplicate check sees the new codes
    for g in init.genotypes:
        schema.decode(g)  # every genotype decodes
        for p in schema.contin_positions:
            schema.contin_value(p, g[p])


def test_wide_codes_reach_long_paths():
    schema = _contin_schema()
    lengths = Counter()
    for seed in range(1, 6):
        randGen().seed(seed)
        init = build_population(
            schema, 100, parse_radii(MIX), MT19937RandGen(seed), contin_start="wide"
        )
        for g in init.genotypes:
            for p in schema.contin_positions:
                if g[p]:
                    lengths[len(_path(g[p], 5))] += 1
    total = sum(lengths.values())
    assert set(lengths) == {1, 2, 3, 4, 5}
    # Uniform over 62 paths: 2/62 of length 1, 4/62 of 2, 8/62 of 3, 16/62 of 4, 32/62 of 5.
    for length, count in lengths.items():
        expected = total * 2**length / 62
        assert 0.6 * expected <= count <= 1.4 * expected, (length, count, expected)


def test_the_non_center_draw_is_about_uniform_over_62_paths():
    schema = _contin_schema()
    rng = MT19937RandGen(5)
    per_path = 200
    counts = Counter(
        schema.sample_contin_code(0, rng, exclude_center=True) for _ in range(62 * per_path)
    )
    assert 0 not in counts
    assert len(counts) == 62
    assert all(_path(code, 5) for code in counts)
    assert min(counts.values()) >= 0.6 * per_path
    assert max(counts.values()) <= 1.4 * per_path
    # the 62 codes are the 63 codes of the default draw without code 0, each a distinct value
    rng = MT19937RandGen(5)
    full = {schema.sample_contin_code(0, rng) for _ in range(63 * 100)}
    assert full == set(counts) | {0}
    assert len({schema.contin_value(0, c) for c in full}) == 63


def test_the_default_draw_of_sample_contin_code_does_not_change():
    schema = _contin_schema()
    a, b = MT19937RandGen(9), MT19937RandGen(9)
    for _ in range(500):
        assert schema.sample_contin_code(3, a) == schema.sample_contin_code(3, b, False)


def test_wide_without_contin_genes_is_step_and_makes_no_draw():
    schema = _bool_schema()
    assert not schema.contin_positions
    out = []
    for mode in CONTIN_START_MODES:
        randGen().seed(4)
        rng = MT19937RandGen(14)
        init = build_population(schema, 40, parse_radii(MIX), rng, contin_start=mode)
        out.append((_digest(init), rng.randint(), randGen().randint()))
    assert out[0] == out[1]


class _StubSchema:
    """Two contin genes (0 and 1, depth 5) and two discrete genes (2 and 3)."""

    contin_positions = (0, 1)
    contin_depths = (5, 5)
    sample_contin_code = FrozenSchema.sample_contin_code


def test_widen_changes_only_the_changed_contin_genes():
    s = _StubSchema()
    rng = MT19937RandGen(3)
    g = (0, 2, 1, 4)  # gene 1 is at R (one step); genes 2 and 3 are discrete
    for _ in range(200):
        w = widen_contin_genes(s, g, rng)
        assert w[0] == 0 and w[1] != 0 and w[2:] == (1, 4)
        _path(w[1], 5)
    # no changed contin gene: the same tuple and no draw
    ref = MT19937RandGen(3)
    rng = MT19937RandGen(3)
    g = (0, 0, 1, 4)
    assert widen_contin_genes(s, g, rng) is g
    assert rng.randint() == ref.randint()


def test_build_population_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="contin_start must be one of"):
        build_population(_contin_schema(), 8, parse_radii(MIX), MT19937RandGen(1), contin_start="x")


# ---- the optimizer


class _SumScorer:
    """Scores an instance by -|sum of its constants - 4| - 1 (never the 0 target)."""

    def __init__(self, rep):
        self.rep = rep

    def get_rep(self):
        return self.rep

    def __call__(self, inst):
        fields = self.rep.fields()
        total = sum(fields.get_contin(inst, i) for i in range(len(fields.contin())))
        return composite_score(-abs(total - 4.0) - 1.0, 0, 0.0, 0.0)


def _run_optimizer(tmp_path, **kw):
    schema = _contin_schema()
    log = tmp_path / "events.jsonl"
    log.unlink(missing_ok=True)
    randGen().seed(7)
    deme = instance_set(schema.rep.fields())
    gp = gomea_parameters(pop_size=12, contin="copy", telemetry_path=str(log), trace=True, **kw)
    opt = gomea_optimization(gp=gp)
    opt(deme, _SumScorer(schema.rep), 300, 10**6)
    events = [json.loads(x) for x in log.read_text().splitlines()]
    return opt, deme, events


def test_the_optimizer_records_wide_and_leaves_step_unchanged(tmp_path):
    opt0, deme0, ev0 = _run_optimizer(tmp_path)
    opt1, deme1, ev1 = _run_optimizer(tmp_path, contin_start="step")
    clock = {"timings", "linkage_s", "variation_s", "max_time"}

    def canon(evs):
        return [{k: v for k, v in e.items() if k not in clock} for e in evs]

    assert canon(ev0) == canon(ev1)
    assert opt0.last_result["best"] == opt1.last_result["best"]
    init0 = next(e for e in ev0 if e["type"] == "init")
    assert "contin_start" not in init0
    assert "contin_start" not in ev0[0]["params"]

    opt2, deme2, ev2 = _run_optimizer(tmp_path, contin_start="wide")
    init2 = next(e for e in ev2 if e["type"] == "init")
    assert init2["contin_start"] == "wide"
    assert ev2[0]["params"]["contin_start"] == "wide"
    assert init2["genotypes_sha256"] != init0["genotypes_sha256"]
    observed = {c for f in init2["diagnostics"]["per_field"] for c in f["observed"]}
    assert any(len(_path(c, 5)) >= 2 for c in observed)
    assert 0 < deme2.n_evals <= 300 and deme2.n_evals == len(deme2)


def test_gomea_parameters_refusals():
    from pymoses.gomea.hybrid import hc_gomea_optimization

    assert gomea_parameters().contin_start == "step"
    assert "contin_start" not in gomea_parameters().as_dict()
    assert gomea_parameters(contin="copy", contin_start="wide").as_dict()["contin_start"] == "wide"
    with pytest.raises(ValueError, match="contin_start must be one of"):
        gomea_parameters(contin="copy", contin_start="x")
    with pytest.raises(ValueError, match="contin_start 'wide' needs init_mode 'sampler'"):
        gomea_parameters(contin="copy", contin_start="wide", init_mode="uniform")
    with pytest.raises(ValueError, match="contin_start 'wide' needs contin 'copy'"):
        gomea_parameters(contin_start="wide")
    with pytest.raises(ValueError, match="contin_start 'wide' needs -a gomea, not the hc-gomea"):
        hc_gomea_optimization(None, None, gomea_parameters(contin="copy", contin_start="wide"))


# ---- the CLI


def _cli_error(argv, tmp_path) -> str:
    import os

    from pymoses.moses.main.moses_exec import moses_exec

    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(ValueError) as exc:
            moses_exec(["moses", *argv])
    finally:
        os.chdir(old)
    return str(exc.value)


def test_the_cli_accepts_step_and_wide_and_refuses_the_rest(tmp_path):
    base = ident._reg("200") + ident._BASE_TLS
    assert _cli_error(base + ["--gomea-contin-start", "far"], tmp_path) == (
        "--gomea-contin-start far is not recognized. Supported: step, wide."
    )
    for algo in ("hc", "hc-capped", "hc-gomea"):
        argv = base + WIDE
        argv[argv.index("-a") + 1] = algo
        assert (
            _cli_error(argv, tmp_path) == f"--gomea-contin-start wide needs -a gomea, got -a {algo}"
        )
    assert _cli_error(base + WIDE + ["--gomea-init", "uniform"], tmp_path) == (
        "--gomea-contin-start wide needs --gomea-init sampler, got uniform"
    )
    argv = base + WIDE
    argv[argv.index("--gomea-contin") + 1] = "reject"
    assert _cli_error(argv, tmp_path) == (
        "--gomea-contin-start wide needs --gomea-contin copy, got reject"
    )


def test_the_cli_runs_wide(tmp_path):
    """BASE-TLS with ``--gomea-trace 1 --gomea-log-best-genes 1 --gomea-contin-start wide`` at
    600 evaluations: the run ends, each "init" event has ``contin_start`` and codes with long
    paths, and the deme bests decode to the logged programs (the best_genes record)."""
    argv = ident._reg("600") + ident._BASE_TLS + ["--gomea-trace", "1"]
    argv += ["--gomea-log-best-genes", "1"] + WIDE
    stdout, _log, events = ident.run_case(argv, tmp_path)
    evs = [json.loads(x) for x in events.splitlines()]
    inits = [e for e in evs if e["type"] == "init"]
    assert inits and all(e["contin_start"] == "wide" for e in inits)
    starts = [e for e in evs if e["type"] == "invocation_start"]
    assert all(e["params"]["contin_start"] == "wide" for e in starts)
    long_codes = 0
    for e in inits:
        for f in e["diagnostics"]["per_field"]:
            long_codes += sum(len(_path(c, 5)) >= 2 for c in f["observed"])
    assert long_codes > 0
    ends = [e for e in evs if e["type"] == "end"]
    assert ends and all("error" not in (e.get("best_genes") or {}) for e in ends)
    assert stdout.strip()
