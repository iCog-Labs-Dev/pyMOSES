"""C++: moses/moses/main/table-problems.{h,cc}

Ports the shared table-loading machinery (`table_problem_params`,
`table_problem_base::common_setup`/`common_type_setup`) plus **seven** of
the eleven problems `register_table_problems` (table-problems.cc:562-580)
registers: `it` (this task's own reachable target), and `pre`, `prerec`,
`recall`, `bep`, `f_one`, `select` -- six more that turned out to be
straightforward to wire given the P2 scorer track's own `precision_bscore`/
`discriminating_bscore`/`select_bscore`/`time_dispersion` (commits
`e998da6`, `9e918d7`, `4a2a67f`, `3fc6e61`) landed already fully ported.
The remaining four -- `ip`, `ann-it`, `pre-conj`, `cluster` -- stay exactly
the raising stubs `pymetta/moses/main/problem.py`'s own
`_unported_table_problem` already provides: `ip`/`ann-it` are excluded
domains (interesting-predicates scoring, the ANN vocabulary) per this
task's own brief; `pre-conj`/`cluster` were not named in that brief's
"if straightforward" list and need `precision_conj_bscore`/`cluster_bscore`
wiring this task did not attempt.

## Composing on top of `problem.py` without editing it

`problem.py`'s own module docstring left `"it"` deliberately unregistered
("this file leaves it a clean seam by simply not naming 'it' anywhere
below") and built `_unported_table_problem`/`_unported_table_problems`/
`_register_unported_table_problems` as stand-ins for THIS task, explicitly
saying the last "may supersede this function entirely once every problem
it covers is ported for real". This file does exactly that, without
editing `problem.py`: `register_table_problems` below registers the seven
real problems first, then calls the existing `_register_unported_table_problems`
for the rest. `problem_manager.register_problem` uses `dict.setdefault`
(problem.py) -- first registration under a name wins, silently dropping
any later one under the same name -- so the six real registrations above
silently absorb the stub attempts `_register_unported_table_problems` would
otherwise make for `pre`/`prerec`/`recall`/`bep`/`f_one`/`select`, and only
`ip`/`ann-it`/`pre-conj`/`cluster`'s stub registrations actually take
effect. This mirrors the same "extend a sibling module's seam without
touching it" discipline `demo_problems.py` used for `problem.py`'s
`alphabet_size` gap (its own module docstring, "Small dependencies this
task also had to supply, and why they live here").

## `table_problem_params`: the CLI surface `problem_params.py` deliberately
## does not carry

`grep`-confirmed absent from `problem_params.py`: `-i`/`--input-file`,
`-u`/`--target-feature`, `--score-weight`, `--timestamp-feature`,
`-Y`/`--ignore-feature`. That module's own docstring explains why --
these are the fields of a SEPARATE C++ `option_base` (`table_problem_params`,
table-problems.h:36-45), not of `problem_params` itself, and are
"deferred to whichever future task actually wires them (`table_problems.py`,
`moses_exec.py`)". `table_problem_params` below is that struct, argparse-based
exactly like `problem_params.add_options`/`parse_options` (`option_base`/
`option_manager` are not ported anywhere in this project -- `problem.py`'s
own module docstring), and `parse_table_args` (bottom of this file) composes
both option surfaces onto ONE shared `argparse.ArgumentParser`, mirroring
`option_manager.register_options`/`init_options`'s own "many `option_base`s,
one `options_description`" composition (problem.cc:43-56).

## `pms.printer` stands in for `pms.mmr_pa`

`common_setup` (table-problems.cc:164) does `pms.mmr_pa.ilabels = ilabels`
unconditionally -- `mmr_pa` is the `metapop_printer` instance, not yet
ported (S13 Task 3, `moses_main.py`'s own open row). This port uses
`pms.printer` instead, the exact duck-typed stand-in convention
`demo_problems.py`'s `bool_problem_base.run` already established for the
same gap. It works even when `.printer` is a plain function/lambda (every
test in this file uses one): ordinary Python functions carry a writable
`__dict__`, so assigning an arbitrary attribute onto one succeeds the same
way it would on any other object. (A *bound method* does NOT support
this -- do not pass one as `.printer`.)

## The multi-input-file quirk: read all, score only the first

`common_setup` (table-problems.cc:101-165) loops over every `-i` file,
validates that they all share the same arity, and only THEN uses
`_tables.front()`/`_ctables.front()` for scoring -- "XXX FIXME -- the
multiple tables should be merged into one" is classic's own comment,
never acted on. DO NOT FIX: ported exactly this way (`self._tables[0]`/
`self._ctables[0]`), not merged.

## Unreachable-but-ported branches

## `_alphabet_size`'s contin branch, and its two-stage integer wrap

Six of this file's seven wired problems (`pre`, `prerec`, `recall`,
`bep`, `f_one`, `select`) force `common_type_setup(pms, boolean_type)`,
so `_alphabet_size`'s boolean branch (`3 + arity`) is the only one they
reach. `it_table_problem` is the exception: it calls `common_type_setup`
with no forced type, and since 2026-09-09 its contin arm no longer
raises, so **`8 + arity - len(ignore_ops)` is now live** on any
`-H it` run against a contin-output table.

That subtraction wraps twice in C++, and the port reproduces both stages
literally:

1. `problem.cc:156` computes `as = 8 + arity - ignore_ops.size()` where
   `as` is `unsigned` (32-bit, `problem.cc:139`/`problem.h:73`) and
   `ignore_ops.size()` is `size_t` (64-bit unsigned). `8 + arity`
   promotes to `size_t`, so `len(ignore_ops) > 8 + arity` yields a
   near-2**64 value, which is then TRUNCATED to 32 bits on assignment.
2. `table-problems.cc:335` (the `REGRESSION` macro) and `:372`
   (`pre_table_problem`'s hand-inlined copy) then both do
   `int as = alphabet_size(...)`, converting that `unsigned` to `int`.
   GCC's two's-complement conversion turns 0xFFFFFFFF into -1.

MEASURED, against the real `alphabet_size` linked out of
`libmoses_exec.so` (probe: `tools/cpp_probe/alphabet_size_probe.cc`,
2026-09-09), for arity 1..3 and 0..12 ignored operators. At arity 1 the
`unsigned`/`int` pair goes `... 1 -> 1`, `0 -> 0`, then
`4294967295 -> -1`, `4294967294 -> -2`, `4294967293 -> -3`. So C++ does
reach negative `as` values, and `set_noise_or_ratio` hands them straight
to `set_complexity_coef`.

**The port's masking changes no reachable result.** Over that whole
measured range the two-stage wrap lands on exactly the value plain
Python arithmetic would produce, and it must: the wrap only becomes
observable once `8 + arity - len(ignore_ops)` passes -2**31, which needs
about two billion distinct ignored operators. The masking below is
therefore fidelity, not a bug fix -- it makes the agreement follow from
C++'s declared types instead of from a coincidence a reader would have
to re-derive. It is written down here rather than left implicit because
this project's rule is that every subtraction gets checked for the
wrap shape, and "checked, reproduced, and inert" is a different claim
from "checked and ignored".

This is reachable from the CLI: `--ignore-operator` inserts any operator
`builtin_str_to_vertex` recognizes, not only the seven contin ones
(`problem-params.cc:2136-2142`), and `--linear-regression` silently adds
four more (`:2146-2151`).

FOLLOW-UP, not this file's to fix: `demo-problems.cc:112`
(`ann_problem`) declares `unsigned as`, not `int as`, so it does NOT
apply stage 2. `demo_problems.py` carries its own copy of
`_alphabet_size` and needs the same treatment with that one difference
preserved.

## `f_one_table_problem`: wired, and faithfully unusable

`f_one_bscore.__init__` (`discriminating_bscore.py`, ported by the P2
scorer track) always raises `AssertionError` -- classic's own
`f_one_bscore` is unconstructible for the identical reason (that module's
own docstring: hardcoded `discriminating_bscore(ct, 0.0, 1.0, 1.0e-20)`
always fails its base class's threshold assert). `f_one_table_problem`
below is registered and reaches that constructor call exactly like
`f_one_table_problem::run()` does in classic (table-problems.cc:461-468)
-- "wiring `f_one`" means reproducing the crash faithfully, not avoiding
it. DO NOT FIX.

## `bep`/`prerec`/`recall`: the `hardness == 0` CLI fallback

Each of these three `run()`s reproduces a literal, unconditional
`if (0.0 == pms.hardness) { ... }` block from table-problems.cc that
overrides `pms.hardness`/`pms.min_rand_input`/`pms.max_rand_input` with a
per-problem hardcoded fallback triple BEFORE constructing its scorer --
ported verbatim, including the mutation of `pms` itself (classic passes
`problem_params&`, a mutable reference; this port's `pms` is the same
object the caller (or `parse_table_args`) constructed, so the mutation is
equally visible to a caller that inspects `pms.hardness` afterward).

## `pre_table_problem`: hand-inlined, not the `REGRESSION` macro

Classic's own comment: "Very nearly identical to the REGRESSION macro
above, except that some new experimental features are being tried"
(table-problems.cc:360-361). Ported by hand for the same reason, with the
extra `gen_best_tree`/`ensemble_params.experts` steps `_regression`
(this file's port of the `REGRESSION` macro, table-problems.cc:328-356)
does not have.
"""

from __future__ import annotations

import argparse

from pymetta.comboreduct.combo.vertex import builtin, combo_tree
from pymetta.comboreduct.table.table import CTable, Table, type_node
from pymetta.comboreduct.table.table_io import loadTable
from pymetta.comboreduct.type_checker.type_signature import TypeSignature, gen_signature
from pymetta.comboreduct.type_checker.type_tree import get_output_type_tree
from pymetta.moses.main.problem import (
    _register_unported_table_problems,
    problem_base,
    problem_manager,
)
from pymetta.moses.main.problem_params import problem_params
from pymetta.moses.moses.moses_main import metapop_moses_results
from pymetta.moses.scoring.behave_cscore import behave_cscore
from pymetta.moses.scoring.bscores import contin_bscore, ctruth_table_bscore
from pymetta.moses.scoring.discriminating_bscore import (
    bep_bscore,
    f_one_bscore,
    prerec_bscore,
    recall_bscore,
)
from pymetta.moses.scoring.precision_bscore import precision_bscore
from pymetta.moses.scoring.select_bscore import select_bscore
from pymetta.moses.scoring.time_dispersion import TemporalGranularity

__all__ = [
    "bep_table_problem",
    "f_one_table_problem",
    "it_table_problem",
    "parse_table_args",
    "pre_table_problem",
    "prerec_table_problem",
    "recall_table_problem",
    "register_table_problems",
    "select_table_problem",
    "table_problem_base",
    "table_problem_params",
]


# ---------------------------------------------------------------------------
# Small local helpers -- own copies, not imports from demo_problems.py,
# matching that file's own "own private copy per file, cited to the same
# C++ source" discipline for `alphabet_size`/`type_to_exemplar` rather than
# a cross-file import of an underscore-prefixed name.
# ---------------------------------------------------------------------------

_GRANULARITY_BY_STR = {
    "day": TemporalGranularity.day,
    "month": TemporalGranularity.month,
}


def _granularity_from_str(gran: str) -> TemporalGranularity:
    """Not a C++ symbol. Classic stores `time_bscore_granularity` as a real
    `TemporalGranularity` enum member on `problem_params`
    (problem-params.h:156), converted once from the CLI string at
    parse-time (problem-params.cc, after validating it is one of exactly
    `"day"`/`"month"`). `problem_params.py`'s own port stores only the
    STRING half (`self.time_bscore_granularity_str`, already validated
    against those same two values in its own `parse_options`), leaving the
    enum conversion to whichever consumer needs it -- `pre_table_problem`
    is the only one in this file, so the conversion happens here.
    """
    try:
        return _GRANULARITY_BY_STR[gran]
    except KeyError:
        raise ValueError(f"Granularity {gran} not implemented") from None


# C++ `unsigned` is 32-bit and `int` is 32-bit two's-complement on every
# platform this port targets; `_alphabet_size` returns the former and its
# callers immediately narrow it to the latter, mirroring `problem.cc:139`'s
# `unsigned as` and `table-problems.cc:335,372`'s `int as = alphabet_size(...)`.
_UINT32_MAX = 0xFFFFFFFF


def _to_int32(value: int) -> int:
    """C++ `unsigned` -> `int` conversion, as GCC performs it.

    Formally implementation-defined before C++20; GCC (and every compiler
    this port is measured against) reinterprets the two's-complement bit
    pattern, which is what `table-problems.cc:335,372`'s
    `int as = alphabet_size(...)` relies on.
    """
    return value - 0x100000000 if value > 0x7FFFFFFF else value


def _alphabet_size(sig: TypeSignature, ignore_ops) -> int:
    """C++: moses/moses/main/problem.cc:138 alphabet_size

    Own copy of `demo_problems.py`'s identical `_alphabet_size` (same
    citation, same reasoning for using `sig.otype` directly in place of
    `get_type_node(get_signature_output(tt))`): that file's own docstring
    explains `problem.py` is not owned by this task either and is a
    stable, already-committed S13 Task 0 deliverable, so a private local
    copy is used here too rather than a cross-file import. See this
    module's own docstring for the one difference from that copy: a
    DIVERGENCE marker on the `contin_type` branch's subtraction, added
    per this task's own "check every subtraction for the size_t-wrap
    shape" instruction (inert here -- see that marker for why).
    """
    arity = len(sig.itypes)
    output_type = sig.otype
    if output_type == type_node.boolean_type:
        # 3 operators: and, or, not
        return 3 + arity
    if output_type == type_node.contin_type:
        # C++ assigns this to `unsigned as` (problem.cc:139) after
        # computing it in `size_t`, so an `ignore_ops` larger than
        # `8 + arity` wraps near 2**64 and is then truncated to 32 bits.
        # See this module's docstring for the full two-stage wrap and why
        # it is reproduced rather than left to Python's arbitrary-precision
        # ints. Reachable since `it_table_problem.run`'s contin arm was
        # wired; the comment here used to say "unreachable from every
        # problem this file wires".
        return (8 + arity - len(ignore_ops)) & _UINT32_MAX
    if output_type == type_node.ann_type:
        return 2 + arity * arity
    if output_type == type_node.enum_type:
        raise NotImplementedError(
            "alphabet_size: enum_type needs enum_t.size()/contains_type, not "
            "wired for alphabet computation anywhere in this port; unreachable "
            "by any problem this file registers (problem.cc:157-171)"
        )
    raise ValueError(f"alphabet_size: type {output_type!r} currently not supported.")


def _type_to_exemplar(output_type: type_node) -> combo_tree:
    """C++: moses/moses/representation/representation.cc:75 type_to_exemplar

    Own copy of `demo_problems.py`'s identical `_type_to_exemplar` (same
    citation and reasoning -- see that function's own docstring for why
    only `boolean_type`/`contin_type` are ported). `it_table_problem` can,
    in principle, be pointed at an enum-output table with no `-e` exemplar
    given; classic would build a `cond`-based enum exemplar there
    (representation.cc's own enum branch, not ported anywhere in this
    project) and then fail moments later inside `it_table_problem::run`'s
    own enum branch (`enum_effective_bscore`, also not ported). This port
    fails one step earlier, inside `common_type_setup` itself, for the
    identical reason (no enum exemplar machinery) -- same ultimate
    outcome (enum tables are unsupported), only the raise POINT differs,
    which changes nothing observable about what a caller can or cannot do.
    """
    if output_type == type_node.boolean_type:
        return combo_tree(builtin.logical_and)
    if output_type == type_node.contin_type:
        return combo_tree(builtin.plus)
    raise NotImplementedError(
        f"type_to_exemplar: type {output_type!r} not supported by this port "
        "(only boolean_type/contin_type are ever requested by this file's "
        "in-scope problems; representation.cc:75's enum_type/default branches "
        "are not ported)"
    )


def _set_noise_or_ratio(scorer, alphabet_size: int, noise: float, ratio: float) -> None:
    """C++: moses/moses/main/table-problems.cc:41 set_noise_or_ratio

    Own copy of `demo_problems.py`'s identical `_set_noise_or_ratio`
    (there citing the same-shaped `demo-problems.cc:73` template function;
    `table-problems.cc:41-48`'s own `set_noise_or_ratio` template is
    byte-for-byte the same body). Collapses the C++ template's
    single-type-parameter dispatch into a plain function, since Python
    has no separate template-instantiation step.
    """
    if noise >= 0.0:
        scorer.set_complexity_coef(alphabet_size, noise)
    else:
        scorer.set_complexity_coef(ratio)


# ---------------------------------------------------------------------------
# table_problem_params -- table-problems.h:36 / table-problems.cc:58.
# ---------------------------------------------------------------------------


class table_problem_params:
    """C++: moses/moses/main/table-problems.h:36 table_problem_params

    The shared table-loading CLI surface every `*_table_problem`
    (including the four still-stubbed ones in `problem.py`) reads through
    `self._tpp`. See this module's own docstring's "`table_problem_params`"
    section for why this is a separate class from `problem_params` here,
    matching classic's own separate `option_base`.
    """

    def __init__(self) -> None:
        """No C++ constructor line to cite: `table_problem_params` declares
        no constructor of its own (table-problems.h:36-45), so every field
        gets boost::program_options' own "empty string"/"empty vector"
        default for an option with no `->default_value(...)` -- verified,
        none of the five options registered below carries one
        (table-problems.cc:64-94).
        """
        self.input_data_files: list[str] = []
        self.target_feature: str = ""
        self.timestamp_feature: str = ""
        self.weighting_feature: str = ""
        self.ignore_features_str: list[str] = []

    def add_options(self, parser: argparse.ArgumentParser) -> None:
        """C++: moses/moses/main/table-problems.cc:58 table_problem_params::add_options"""
        add = parser.add_argument
        add(
            "-i",
            "--input-file",
            dest="input_data_files",
            action="append",
            default=[],
        )
        add("-u", "--target-feature", dest="target_feature", type=str, default="")
        add("--score-weight", dest="weighting_feature", type=str, default="")
        add("--timestamp-feature", dest="timestamp_feature", type=str, default="")
        add(
            "-Y",
            "--ignore-feature",
            dest="ignore_features_str",
            action="append",
            default=[],
        )

    def parse_options(self, args: argparse.Namespace) -> None:
        """No direct C++ line: `table_problem_params` declares no
        `parse_options` override (table-problems.h:36-45), so C++ uses
        `option_base::parse_options`'s default empty-body implementation
        (problem.h:38) -- every field is actually populated by
        boost::program_options' own `po::value<T>(&member)` binding
        during `po::store`/`po::notify` (problem.cc:63,68), not by any
        method body. argparse has no equivalent auto-binding, so this
        method does by hand what that binding does automatically in
        C++ -- the observable result (fields populated from parsed argv)
        is identical, not a divergence.
        """
        self.input_data_files = list(args.input_data_files)
        self.target_feature = args.target_feature
        self.weighting_feature = args.weighting_feature
        self.timestamp_feature = args.timestamp_feature
        self.ignore_features_str = list(args.ignore_features_str)


# ---------------------------------------------------------------------------
# table_problem_base -- table-problems.h:48 / table-problems.cc:101,170.
# ---------------------------------------------------------------------------


class table_problem_base(problem_base):
    """C++: moses/moses/main/table-problems.h:48 table_problem_base"""

    def __init__(self, tpp: table_problem_params) -> None:
        """C++: moses/moses/main/table-problems.h:51 table_problem_base::table_problem_base"""
        self._tpp = tpp
        self._tables: list[Table] = []
        self._ctables: list[CTable] = []
        self.table: Table = Table()
        self.ctable: CTable = CTable()
        self.ilabels: list[str] = []
        self.arity: int = 0
        self.table_type_signature: TypeSignature | None = None
        self.cand_type_signature: TypeSignature | None = None
        self.output_type: type_node = type_node.unknown_type

    def common_setup(self, pms: problem_params) -> None:
        """C++: moses/moses/main/table-problems.cc:101 table_problem_base::common_setup"""
        if not self._tpp.input_data_files:
            raise ValueError("no input data file has been specified (option -i)")

        self._tables = []
        self._ctables = []
        for idf in self._tpp.input_data_files:
            table = loadTable(
                idf,
                self._tpp.target_feature,
                self._tpp.timestamp_feature,
                self._tpp.ignore_features_str,
            )

            # C++: if (pms.nsamples > 0) subsampleTable(...) --
            # table-problems.cc:121-122. `subsampleTable` is not ported
            # anywhere in this project (table.py/table_io.py, grep-confirmed).
            if pms.nsamples > 0:
                raise NotImplementedError(
                    "table_problem_base.common_setup: --nsamples > 0 needs "
                    "subsampleTable, not ported anywhere in this project "
                    "(table.py/table_io.py) -- table-problems.cc:121-122"
                )

            ctable = table.compressed(self._tpp.weighting_feature)
            if self._tpp.weighting_feature:
                table.itable.delete_column(self._tpp.weighting_feature)
            self._tables.append(table)
            self._ctables.append(ctable)

            # C++: if (pms.balance) _ctables.back().balance() --
            # table-problems.cc:133-134. `CTable.balance()` is not ported
            # anywhere in this project (table.py, grep-confirmed).
            if pms.balance:
                raise NotImplementedError(
                    "table_problem_base.common_setup: --balance needs "
                    "CTable.balance(), not ported anywhere in this project "
                    "(table.py) -- table-problems.cc:133-134"
                )

        # DO NOT FIX: see this module's own docstring, "The
        # multi-input-file quirk" -- classic validates arity across every
        # `-i` file (below) but only ever scores against the FIRST one.
        self.table = self._tables[0]
        self.ctable = self._ctables[0]

        self.ilabels = []
        if pms.output_with_labels:
            self.ilabels = self.table.itable.get_labels()

        self.arity = self.table.get_arity()

        if len(self._tables) > 1:
            for i in range(1, len(self._tables)):
                test_arity = self._tables[i].get_arity()
                if test_arity != self.arity:
                    raise ValueError(
                        f"File {self._tpp.input_data_files[0]} has arity "
                        f"{self.arity} while file "
                        f"{self._tpp.input_data_files[i]} has arity {test_arity}"
                    )

        # C++: pms.mmr_pa.ilabels = ilabels (table-problems.cc:164). See
        # this module's own docstring, "`pms.printer` stands in for
        # `pms.mmr_pa`".
        pms.printer.ilabels = self.ilabels

    def common_type_setup(
        self, pms: problem_params, out_type: type_node = type_node.unknown_type
    ) -> None:
        """C++: moses/moses/main/table-problems.cc:170 table_problem_base::common_type_setup"""
        self.table_type_signature = self.table.get_signature()

        if out_type == type_node.unknown_type:
            # C++: get_type_node(get_signature_output(table_type_signature))
            # (table-problems.cc:179-180) -- `.otype` is the direct
            # equivalent, for the identical reason `_alphabet_size` above
            # (and demo_problems.py's own twin) use `sig.otype` in place
            # of the same two-function call: `get_signature_output(sig).node`
            # IS `sig.otype` by construction (type_signature.py), not an
            # approximation.
            out_type = self.table_type_signature.otype

        if not pms.exemplars:
            pms.exemplars.append(_type_to_exemplar(out_type))

        root = pms.exemplars[0].begin()
        self.output_type = get_output_type_tree(root.data).node
        if self.output_type == type_node.unknown_type:
            self.output_type = out_type

        self.cand_type_signature = gen_signature(self.table_type_signature.itypes, self.output_type)

    def _regression(self, pms: problem_params, bscore) -> None:
        """C++: moses/moses/main/table-problems.cc:328 REGRESSION (macro)

        Shared by every `*_table_problem::run()` that uses the
        `REGRESSION` macro verbatim in classic and is wired for real
        here: `it` (boolean branch), `prerec`, `recall`, `bep`, `f_one`,
        `select`. NOT used by `pre_table_problem` (hand-inlined in
        classic too, see this module's own docstring) nor by
        `pre_conj`/`cluster` (not wired here at all).
        """
        # C++: table-problems.cc:330-334. `feature_selector`
        # (moses/deme/feature_selector.h) is not ported anywhere in this
        # project (P2, --enable-fs). A real CLI run already rejects
        # `--enable-fs=1` earlier, inside `problem_params.parse_options`
        # (that module's own docstring, phase-2 gate list) -- this check
        # is therefore unreachable via a real `parse_table_args` call and
        # is ported here only for direct-call fidelity (a caller that
        # hand-builds a `problem_params` bypassing `parse_options`).
        if pms.enable_feature_selection and pms.fs_params.target_size > 0:
            raise NotImplementedError(
                "table_problem_base._regression: feature_selector is not "
                "ported anywhere in this project (P2, --enable-fs) -- "
                "table-problems.cc:330-334"
            )

        # C++: table-problems.cc:335,372 `int as = alphabet_size(...)` --
        # the `unsigned` return narrowed to a signed 32-bit `int`. Only the
        # contin/enum branches can produce a value this changes; see this
        # module's docstring for the two-stage wrap.
        as_ = _to_int32(_alphabet_size(self.cand_type_signature, pms.ignore_ops))
        _set_noise_or_ratio(bscore, as_, pms.noise, pms.complexity_ratio)
        if pms.meta_params.do_boosting:
            bscore.use_weighted_scores()
        # When boosting, cache must not be used, as otherwise, stale
        # composite scores get cached and returned.
        if pms.meta_params.do_boosting:
            pms.cache_size = 0

        mbcscore = behave_cscore(bscore, pms.cache_size)

        reduct_cand = pms.bool_reduct
        reduct_rep = pms.bool_reduct_rep
        # Use the contin reductors for everything else.
        if self.output_type != type_node.boolean_type:
            # `problem_params.__init__` builds a real
            # `contin_reduction(reduct_candidate_effort, ignore_ops)` since
            # `d439711`, so this is a defensive check for a hand-built
            # `problem_params` that bypassed the constructor, not a gate on
            # an unported subsystem. It read "problem_params.contin_reduct
            # is still None ... S14 this port has not built" until
            # 2026-09-09; that claim was false by then.
            if pms.contin_reduct is None:
                raise ValueError(
                    "table_problem_base._regression: non-boolean output "
                    "needs pms.contin_reduct, which is None -- a "
                    "problem_params built without running its constructor's "
                    "reductor setup (table-problems.cc:346-350)"
                )
            reduct_cand = pms.contin_reduct
            reduct_rep = pms.contin_reduct

        metapop_moses_results(
            pms.exemplars,
            self.cand_type_signature,
            reduct_cand,
            reduct_rep,
            mbcscore,
            pms.opt_params,
            pms.hc_params,
            pms.deme_params,
            pms.filter_params,
            pms.meta_params,
            pms.moses_params,
            pms.printer,
        )


# ---------------------------------------------------------------------------
# it_table_problem -- table-problems.h:194 / table-problems.cc:471.
# ---------------------------------------------------------------------------


class it_table_problem(table_problem_base):
    """C++: moses/moses/main/table-problems.h:194 it_table_problem"""

    def name(self) -> str:
        return "it"

    def description(self) -> str:
        return "Maximize Accuracy"

    def run(self, ob: object) -> None:
        """C++: moses/moses/main/table-problems.cc:471 it_table_problem::run"""
        pms = ob
        self.common_setup(pms)
        self.common_type_setup(pms)

        if self.output_type == type_node.boolean_type:
            self._regression(pms, ctruth_table_bscore(self.ctable))
        elif self.output_type == type_node.enum_type:
            raise NotImplementedError(
                "it_table_problem.run: enum-output tables need "
                "enum_effective_bscore/partial_solver (the 'leave "
                "well-enough alone' algorithm), neither ported anywhere in "
                "this project (P2) -- table-problems.cc:483-509"
            )
        elif self.output_type == type_node.contin_type:
            # C++: moses/moses/main/table-problems.cc:511-525 -- the contin
            # branch picks `contin_bscore` when `discretize_thresholds` is
            # empty, with `abs_error` if `--it-abs-err` is set and
            # `squared_error` otherwise (cc:514-518).
            #
            # This raised unconditionally until 2026-09-09, on the stated
            # grounds that contin_bscore and reduct::contin_reduction were
            # "S14 territory this port has not built". Both had in fact
            # landed -- contin_bscore in `07027b0`, contin_reduction in
            # `d439711` -- so the message was false and the branch was
            # blocking work that already existed. Found by driving the CLI
            # against classic on a real contin table.
            if pms.discretize_thresholds:
                # DIVERGENCE(C++ moses/moses/main/table-problems.cc:520-524):
                # the `--discretize-threshold` arm needs
                # `discretize_contin_bscore`, which is genuinely NOT ported
                # (bscores.py ports `ctruth_table_bscore` and `contin_bscore`
                # only). Classic runs this arm; this port raises.
                raise NotImplementedError(
                    "it_table_problem.run: --discretize-threshold needs "
                    "discretize_contin_bscore (bscores.h), which is not "
                    "ported -- table-problems.cc:520-524"
                )
            eft = contin_bscore.abs_error if pms.it_abs_err else contin_bscore.squared_error
            self._regression(pms, contin_bscore(self.table, eft=eft))
        else:
            # C++: table-problems.cc:528-535 logs an error and exit(1)s --
            # ported as a hard-failure ValueError, this port's usual
            # substitute for a non-recoverable exit() on bad/unsupported
            # input (matching demo_problems.py's mux_problem.get_arity
            # convention).
            raise ValueError(
                f"it_table_problem.run: output type {self.output_type!r} currently not supported."
            )


# ---------------------------------------------------------------------------
# pre_table_problem -- table-problems.h:116 / table-problems.cc:358.
# ---------------------------------------------------------------------------


class pre_table_problem(table_problem_base):
    """C++: moses/moses/main/table-problems.h:116 pre_table_problem"""

    def name(self) -> str:
        return "pre"

    def description(self) -> str:
        return "Precision-Activation scoring"

    def run(self, ob: object) -> None:
        """C++: moses/moses/main/table-problems.cc:358 pre_table_problem::run

        Not the `REGRESSION` macro -- see this module's own docstring's
        "`pre_table_problem`" section.
        """
        pms = ob
        self.common_setup(pms)
        self.common_type_setup(pms, type_node.boolean_type)

        # C++: table-problems.cc:367-370 -- see `_regression`'s own
        # identical comment for why this is unreachable via a real CLI run.
        if pms.enable_feature_selection and pms.fs_params.target_size > 0:
            raise NotImplementedError(
                "pre_table_problem.run: feature_selector is not ported "
                "anywhere in this project (P2, --enable-fs) -- "
                "table-problems.cc:367-370"
            )

        # C++: table-problems.cc:335,372 `int as = alphabet_size(...)` --
        # the `unsigned` return narrowed to a signed 32-bit `int`. Only the
        # contin/enum branches can produce a value this changes; see this
        # module's docstring for the two-stage wrap.
        as_ = _to_int32(_alphabet_size(self.cand_type_signature, pms.ignore_ops))
        bscore = precision_bscore(
            self.ctable,
            pms.hardness,
            pms.min_rand_input,
            pms.max_rand_input,
            pms.pre_positive,
            pms.time_dispersion_pressure,
            pms.time_dispersion_exponent,
            pms.meta_params.ensemble_params.exact_experts,
            pms.meta_params.ensemble_params.bias_scale,
            pms.time_bscore,
            _granularity_from_str(pms.time_bscore_granularity_str),
        )
        _set_noise_or_ratio(bscore, as_, pms.noise, pms.complexity_ratio)
        if pms.meta_params.do_boosting:
            bscore.use_weighted_scores()

        if pms.gen_best_tree:
            # experimental: use some canonically generated candidate as
            # exemplar seed.
            tr = bscore.gen_canonical_best_candidate()
            pms.exemplars.append(tr)

        # We support boosting by generating an "ensemble of experts" model.
        pms.meta_params.ensemble_params.experts = True
        # When boosting, cache must not be used, as otherwise, stale
        # composite scores get cached and returned.
        if pms.meta_params.do_boosting:
            pms.cache_size = 0

        mbcscore = behave_cscore(bscore, pms.cache_size)
        metapop_moses_results(
            pms.exemplars,
            self.cand_type_signature,
            pms.bool_reduct,
            pms.bool_reduct_rep,
            mbcscore,
            pms.opt_params,
            pms.hc_params,
            pms.deme_params,
            pms.filter_params,
            pms.meta_params,
            pms.moses_params,
            pms.printer,
        )


# ---------------------------------------------------------------------------
# prerec / recall / bep / f_one / select -- all REGRESSION-macro shaped.
# ---------------------------------------------------------------------------


class prerec_table_problem(table_problem_base):
    """C++: moses/moses/main/table-problems.h:142 prerec_table_problem"""

    def name(self) -> str:
        return "prerec"

    def description(self) -> str:
        return "Precision Maximization (holding recall constant)"

    def run(self, ob: object) -> None:
        """C++: moses/moses/main/table-problems.cc:425 prerec_table_problem::run"""
        pms = ob
        self.common_setup(pms)
        self.common_type_setup(pms, type_node.boolean_type)

        if pms.hardness == 0.0:
            pms.hardness = 1.0
            pms.min_rand_input = 0.5
            pms.max_rand_input = 1.0
        bscore = prerec_bscore(self.ctable, pms.min_rand_input, pms.max_rand_input, pms.hardness)
        self._regression(pms, bscore)


class recall_table_problem(table_problem_base):
    """C++: moses/moses/main/table-problems.h:155 recall_table_problem"""

    def name(self) -> str:
        return "recall"

    def description(self) -> str:
        return "Recall Maximization (holding precision constant)"

    def run(self, ob: object) -> None:
        """C++: moses/moses/main/table-problems.cc:437 recall_table_problem::run"""
        pms = ob
        self.common_setup(pms)
        self.common_type_setup(pms, type_node.boolean_type)

        if pms.hardness == 0.0:
            pms.hardness = 1.0
            pms.min_rand_input = 0.8
            pms.max_rand_input = 1.0
        bscore = recall_bscore(self.ctable, pms.min_rand_input, pms.max_rand_input, pms.hardness)
        self._regression(pms, bscore)


class bep_table_problem(table_problem_base):
    """C++: moses/moses/main/table-problems.h:168 bep_table_problem"""

    def name(self) -> str:
        return "bep"

    def description(self) -> str:
        return "Maximize Break-even Point"

    def run(self, ob: object) -> None:
        """C++: moses/moses/main/table-problems.cc:449 bep_table_problem::run"""
        pms = ob
        self.common_setup(pms)
        self.common_type_setup(pms, type_node.boolean_type)

        if pms.hardness == 0.0:
            pms.hardness = 1.0
            pms.min_rand_input = 0.0
            pms.max_rand_input = 0.5
        bscore = bep_bscore(self.ctable, pms.min_rand_input, pms.max_rand_input, pms.hardness)
        self._regression(pms, bscore)


class f_one_table_problem(table_problem_base):
    """C++: moses/moses/main/table-problems.h:181 f_one_table_problem

    See this module's own docstring, "`f_one_table_problem`: wired, and
    faithfully unusable" -- `run()` always raises `AssertionError`,
    matching classic's own crash.
    """

    def name(self) -> str:
        return "f_one"

    def description(self) -> str:
        return "Maximize F_1 score"

    def run(self, ob: object) -> None:
        """C++: moses/moses/main/table-problems.cc:461 f_one_table_problem::run"""
        pms = ob
        self.common_setup(pms)
        self.common_type_setup(pms, type_node.boolean_type)

        bscore = f_one_bscore(self.ctable)  # DO NOT FIX: always raises.
        self._regression(pms, bscore)  # pragma: no cover -- unreachable


class select_table_problem(table_problem_base):
    """C++: moses/moses/main/table-problems.h:207 select_table_problem"""

    def name(self) -> str:
        return "select"

    def description(self) -> str:
        return "Select range of rows"

    def run(self, ob: object) -> None:
        """C++: moses/moses/main/table-problems.cc:538 select_table_problem::run"""
        pms = ob
        self.common_setup(pms)
        self.common_type_setup(pms, type_node.boolean_type)

        bscore = select_bscore(
            self.ctable, pms.min_rand_input, pms.max_rand_input, pms.hardness, pms.pre_positive
        )
        self._regression(pms, bscore)


# ---------------------------------------------------------------------------
# register_table_problems -- table-problems.cc:562-581.
# ---------------------------------------------------------------------------


def register_table_problems(pmr: problem_manager) -> table_problem_params:
    """C++: moses/moses/main/table-problems.cc:562 register_table_problems

    `void register_table_problems(problem_manager&, option_manager&)` in
    C++; the `option_manager&` parameter is dropped (nothing to register
    CLI options against centrally -- see this module's own docstring) and
    the freshly built `table_problem_params` is RETURNED instead (no C++
    equivalent, same convention `demo_problems.py`'s `register_demo_problems`
    already established for its own shared `demo_params`).

    Registers the seven problems this task wired for real, then defers to
    `problem.py`'s own `_register_unported_table_problems` for the
    remaining four (`ip`, `ann-it`, `pre-conj`, `cluster`) -- see this
    module's own docstring, "Composing on top of `problem.py` without
    editing it", for why calling it AFTER the seven real registrations is
    safe (`register_problem`'s `dict.setdefault` semantics).
    """
    tpp = table_problem_params()
    pmr.register_problem(it_table_problem(tpp))
    pmr.register_problem(pre_table_problem(tpp))
    pmr.register_problem(prerec_table_problem(tpp))
    pmr.register_problem(recall_table_problem(tpp))
    pmr.register_problem(bep_table_problem(tpp))
    pmr.register_problem(f_one_table_problem(tpp))
    pmr.register_problem(select_table_problem(tpp))
    _register_unported_table_problems(pmr)
    return tpp


def parse_table_args(argv: list[str], tpp: table_problem_params) -> problem_params:
    """Not itself a ported C++ symbol. Mirrors `problem_params.py`'s own
    `parse_args` convenience function exactly, extended to also register
    and parse `table_problem_params`'s options on the SAME shared
    `argparse.ArgumentParser` -- the same "many `option_base`s, one
    `options_description`" composition `option_manager.register_options`/
    `init_options`/`parse_options` perform in C++ (problem.cc:43-88),
    collapsed into one entry point for this module's own callers and
    tests. S13 Task 3's `moses_exec.py` (not yet built) is free to
    supersede this once it exists.

    `tpp` is taken as a parameter, not constructed here, because it must
    be the SAME instance `register_table_problems` already built and
    wired into every registered `table_problem_base` -- exactly as
    classic's own `moses_exec.cc` shares ONE `table_problem_params*`
    across `register_table_problems` (which wires it into every
    `table_problem_base` subclass) and `mgr.parse_options` (which later
    populates it from argv). Passing a second, freshly-constructed
    `table_problem_params` here would parse into an object no registered
    problem ever reads -- the exact bug this signature is shaped to
    prevent a caller from making.
    """
    pp = problem_params()
    parser = argparse.ArgumentParser(add_help=True)
    pp.add_options(parser)
    tpp.add_options(parser)
    ns = parser.parse_args(argv)
    pp.parse_options(ns)
    tpp.parse_options(ns)
    return pp
