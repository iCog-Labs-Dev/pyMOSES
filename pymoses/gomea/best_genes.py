"""NOT A PORT, backlog 15.4 (diagnostic D1): the genes of a deme's best program in the GOMEA log.

With ``--gomea-log-best-genes 1`` (``gomea_parameters.log_best_genes``), the "end" event of
each deme gets the field ``best_genes`` (``best_genes_record``). It describes the deme's best
individual, the one that the field ``best`` of the same event describes
(``EvaluationOracle.best_genotype``). D1 asks whether the contin knob grid limits the
constants: each constant gene is a contin knob, centered on the exemplar's constant
(``mean``), with a step size, an expansion and a depth (``build_knobs.py``, ``knobs.py``,
``field_set.contin_stepper``). A gene code is a path of up to ``depth`` Left or Right steps.

The field (keys sorted in the log):

* ``program``: the unreduced program (``schema.candidate(g, False)``), with 17 significant
  digits for each constant, so that the text gives each constant back exactly;
* ``program_reduced``: the reduced program (the program that the scorer sees), in the same
  format;
* ``leaf_genes``: for each constant leaf of the unreduced program, in pre-order (the order of
  ``const_opt.constant_nodes``), the contin gene whose knob put the leaf there, or ``None``
  (``schema.candidate_genes``);
* ``linear_scaling``: the fit of the run's scorer: ``terms``, ``plain`` (``--linear-scaling
  1``) or ``none``; ``None`` for a scorer that is not a ``contin_bscore``;
* ``fit``: the fit mode of ``rv_inert.analyse`` for this program (``terms`` or ``plain``);
  ``None`` when the schema has no contin genes;
* ``genes``: one record for each contin gene of the schema, in gene order (``gene_record``);
* ``exemplar``: whether the genotype is all zeros, so that the program is the deme's exemplar
  (then every gene is at the center because the deme found nothing better).

A gene record: ``gene``, ``code``, ``value`` (the constant that the code decodes to), the knob
spec (``mean``, ``step_size``, ``expansion``, ``depth``), ``free`` (a free code of
``--const-opt mutate`` or ``--const-opt-during elite``: no step path), ``path`` (the steps up to
the first Stop, ``"L"`` and ``"R"``; ``None`` for a free code), ``length`` (the number of
steps), ``edge`` (a full-depth path in one direction: the value is ``mean +- (2 ** depth - 1)
x step_size`` for the expansion 2), and ``class``: ``zero``, ``dead``, ``absorbed`` or ``live``
(``rv_inert.analyse`` under the run's linear scaling), or ``absent`` (the knob is not in the
unreduced program, for example below a discrete knob that removes its subtree).

The record only reads: it draws no random number and changes no state of the run. It decodes
the codes with its own ``contin_stepper`` walk (not ``FrozenSchema.contin_value``, which keeps a
memo), and it builds the programs with ``FrozenSchema.candidate`` and ``candidate_genes``
(``rep.clean_combo_tree``, not the LRU cache of ``rep.get_candidate``).
"""

from __future__ import annotations

from collections.abc import Sequence

from pymoses.comboreduct.combo.iostream_combo import tree_to_stream_str
from pymoses.gomea.rv_inert import PLAIN, TERMS, analyse
from pymoses.moses.representation.field_set import contin_spec, contin_stepper

# The digits of each constant in the program texts: 17 significant digits give a double back
# exactly (``float(text) == value``).
PROGRAM_PRECISION = 17

# The values of ``linear_scaling`` (the fit of the run's scorer).
LS_TERMS = "terms"
LS_PLAIN = "plain"
LS_NONE = "none"

# The class of a contin gene whose knob is not in the unreduced program.
ABSENT = "absent"


def step_path(code: int, depth: int) -> str:
    """The steps of a contin gene code: ``"L"`` or ``"R"`` for each raw field up to the first
    Stop (as ``field_set.get_contin`` walks them). ``ValueError`` for a code with an invalid
    step (a digit 3) before the first Stop."""
    out = []
    for k in range(depth):
        step = (code >> (2 * k)) & 3
        if step == contin_spec.Stop:
            break
        if step == contin_spec.Left:
            out.append("L")
        elif step == contin_spec.Right:
            out.append("R")
        else:
            raise ValueError(f"contin code {code} has an invalid step at raw field {k}")
    return "".join(out)


def path_value(spec: contin_spec, path: str) -> float:
    """The constant at the end of ``path`` (``"L"``/``"R"`` steps) from ``spec.mean``, by the
    walk of ``contin_stepper``."""
    stepper = contin_stepper(spec)
    for s in path:
        if s == "L":
            stepper.left()
        elif s == "R":
            stepper.right()
        else:
            raise ValueError(f"invalid step {s!r} in path {path!r}")
    return stepper.value


def all_paths(depth: int) -> list[str]:
    """Every legal step path of a contin knob of ``depth``: ``2 ** (depth + 1) - 1`` paths
    (the empty path first, then by length)."""
    out = [""]
    level = [""]
    for _ in range(depth):
        level = [p + s for p in level for s in "LR"]
        out += level
    return out


def grid_values(spec: contin_spec) -> list[float]:
    """The sorted values that a contin knob with ``spec`` can hold (one for each legal path;
    the analysis script of D1 uses them for the distance to the grid)."""
    return sorted(path_value(spec, p) for p in all_paths(spec.depth))


def is_edge(path: str, depth: int) -> bool:
    """Whether ``path`` is a full-depth path in one direction (the end of the grid)."""
    return depth > 0 and len(path) == depth and len(set(path)) == 1


def scorer_fit(iscorer) -> str | None:
    """The fit of the scorer behind ``iscorer``: ``terms`` (``--linear-scaling terms``),
    ``plain`` (``--linear-scaling 1``), ``none``, or ``None`` when there is no
    ``contin_bscore``. Only reads attributes."""
    from pymoses.moses.scoring.bscores import contin_bscore

    cscorer = getattr(iscorer, "_cscorer", None)
    get = getattr(cscorer, "get_bscorer", None)
    b = get() if callable(get) else None
    if not isinstance(b, contin_bscore):
        return None
    if getattr(b, "_linear_scaling_terms", False):
        return LS_TERMS
    if getattr(b, "_linear_scaling", False):
        return LS_PLAIN
    return LS_NONE


def gene_classes(schema, g: Sequence[int], linear_scaling: str | None, reduce: bool) -> dict:
    """``rv_inert.analyse`` of ``g`` under the run's linear scaling. ``terms``: the analysis as
    it is (per-term fit, or plain when the program has one term). ``plain``: the plain fit for
    every program. ``none`` (no fit): the plain analysis, with each ``absorbed`` gene ``live``
    (no fit absorbs a change). ``None`` (not a regression scorer): as ``none``."""
    if linear_scaling == LS_TERMS:
        return analyse(schema, g, reduce)
    a = analyse(schema, g, reduce, mode=PLAIN)
    if linear_scaling != LS_PLAIN:
        a["genes"] = {p: ("live" if c == "absorbed" else c) for p, c in a["genes"].items()}
    return a


def gene_record(schema, pos: int, code: int, cls: str) -> dict:
    """The record of contin gene ``pos`` with ``code`` (see the module docstring)."""
    spec = schema._contin_specs[pos]
    depth = schema.contin_depths[pos]
    free = code >= 4**depth
    if free:
        value = float(schema._free_values[pos][code - 4**depth])
        path = None
    else:
        path = step_path(code, depth)
        value = float(path_value(spec, path))
    return {
        "gene": pos,
        "code": code,
        "value": value,
        "mean": float(spec.mean),
        "step_size": float(spec.step_size),
        "expansion": float(spec.expansion),
        "depth": depth,
        "free": free,
        "path": path,
        "length": None if path is None else len(path),
        "edge": path is not None and is_edge(path, depth),
        "class": cls,
    }


def best_genes_record(schema, g: Sequence[int], linear_scaling: str | None, reduce: bool) -> dict:
    """The field ``best_genes`` for genotype ``g`` (see the module docstring). ``reduce`` is
    the scorer's reduction flag (``iscorer._reduce``)."""
    tree, leaf_genes = schema.candidate_genes(g)
    reduced = schema.candidate(g, reduce)
    if schema.contin_positions:
        a = gene_classes(schema, g, linear_scaling, reduce)
        classes, fit = a["genes"], a["mode"]
    else:
        classes, fit = {}, None
    genes = [
        gene_record(schema, p, int(g[p]), classes.get(p, ABSENT)) for p in schema.contin_positions
    ]
    return {
        "program": tree_to_stream_str(tree, PROGRAM_PRECISION),
        "program_reduced": tree_to_stream_str(reduced, PROGRAM_PRECISION),
        "leaf_genes": leaf_genes,
        "linear_scaling": linear_scaling,
        "fit": fit,
        "genes": genes,
        "exemplar": all(int(v) == 0 for v in g),
    }


__all__ = [
    "ABSENT",
    "LS_NONE",
    "LS_PLAIN",
    "LS_TERMS",
    "PLAIN",
    "PROGRAM_PRECISION",
    "TERMS",
    "all_paths",
    "best_genes_record",
    "gene_classes",
    "gene_record",
    "grid_values",
    "is_edge",
    "path_value",
    "scorer_fit",
    "step_path",
]
