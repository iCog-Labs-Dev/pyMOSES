"""C++: moses/moses/main/table-problems.{h,cc}

Ports the shared table-loading machinery (`table_problem_params`,
`table_problem_base::common_setup`/`common_type_setup`) plus **nine** of
the eleven problems `register_table_problems` (table-problems.cc:562-580)
registers: `it` (this file's original reachable target), and `pre`,
`prerec`, `recall`, `bep`, `f_one`, `select` -- six more that turned out
to be straightforward to wire given the P2 scorer track's own
`precision_bscore`/`discriminating_bscore`/`select_bscore`/
`time_dispersion` (commits `e998da6`, `9e918d7`, `4a2a67f`, `3fc6e61`)
landed already fully ported -- plus `cluster`, wired later once
`cluster_bscore` was ported (bscores.py), and `pre-conj`.

The remaining two -- `ip` and `ann-it` -- stay the raising stubs
`pymoses/moses/main/problem.py`'s own `_unported_table_problem` provides.
Each stub states its OWN reason (see that function): `ip` needs
`interesting_predicate_bscore`, which is not ported; `ann-it` needs the
ANN vocabulary, excluded project-wide.

This paragraph said "**seven** ... The remaining four -- `ip`, `ann-it`,
`pre-conj`, `cluster` ... `pre-conj`/`cluster` ... need
`precision_conj_bscore`/`cluster_bscore`" until `cluster_table_problem`
landed below, then "**eight** ... The remaining three -- `ip`, `ann-it`,
`pre-conj`" until `pre_conj_table_problem` landed below it. Every version
of it was wrong about `pre-conj` for the same reason: it said the scorer
was missing when `precision_conj_bscore` had already been ported, and
then that "was simply never in any task's scope" was a reason to keep
a stub whose own message admitted only wiring was left.

## Composing on top of `problem.py`

`problem.py` built `_unported_table_problem`/`_unported_table_problems`/
`_register_unported_table_problems` as stand-ins, explicitly saying the
last "may supersede this function entirely once every problem it covers is
ported for real". `register_table_problems` below registers the eight real
problems first, then calls `_register_unported_table_problems` for the
rest.

`problem_manager.register_problem` uses `dict.setdefault` (problem.py) --
first registration under a name wins, silently dropping any later one
under the same name -- so the ordering makes the composition safe
regardless of what `problem.py` still stubs. This section used to lean on
that AND to claim the composition was done "without editing `problem.py`":
the first half still holds, the second no longer does. `problem.py` was
edited when `cluster_table_problem` landed, to drop the seven stubs this
file's real registrations had already shadowed -- a shadowed stub is
invisible, so a wrong reason string inside it can never be seen to be
wrong, which is exactly how the shared "precision/discriminating/select
bscore" message survived long after it stopped being true of any of them.
The two stubs `problem.py` still carries (`ip`, `ann-it`) are the two
this file genuinely does not wire.

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

Six of this file's eight wired problems (`pre`, `prerec`, `recall`,
`bep`, `f_one`, `select`) force `common_type_setup(pms, boolean_type)`,
so `_alphabet_size`'s boolean branch (`3 + arity`) is the only one they
reach. Two are exceptions: `it_table_problem` calls `common_type_setup`
with no forced type, and since 2026-09-09 its contin arm no longer
raises; `cluster_table_problem` forces `contin_type` outright
(table-problems.cc:555), so it reaches the contin branch even against a
boolean-output TABLE -- the forced type is what `_type_to_exemplar`
builds the default exemplar from, and `self.output_type` is read back off
that exemplar, so only an explicit `-y` exemplar with a different output
type can put it anywhere else. So **`8 + arity - len(ignore_ops)` is now
live** on any `-H it` run against a contin-output table, and on any
`-H cluster` run without such a `-y`. Verified directly: classic's own
`-Hcluster` on the boolean `dataset.csv` prints the contin candidate `0`,
not a boolean one.

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

from pymoses.comboreduct.combo.vertex import builtin, combo_tree, get_contin
from pymoses.comboreduct.table.table import CTable, Table, type_node
from pymoses.comboreduct.table.table_io import loadTable
from pymoses.comboreduct.type_checker.type_signature import (
    TypeSignature,
    gen_signature,
    ostream_type_node,
    ostream_type_signature,
)
from pymoses.comboreduct.type_checker.type_tree import get_output_type_tree
from pymoses.moses.deme.feature_selector import feature_selector
from pymoses.moses.deme.numeric_simplify import exemplar_numeric_simplifier
from pymoses.moses.main.problem import (
    _register_unported_table_problems,
    problem_base,
    problem_manager,
)
from pymoses.moses.main.problem_params import problem_params
from pymoses.moses.moses.moses_main import gomea_kwargs, metapop_moses_results
from pymoses.moses.scoring.behave_cscore import behave_cscore
from pymoses.moses.scoring.bscores import (
    cluster_bscore,
    contin_bscore,
    ctruth_table_bscore,
    discretize_contin_bscore,
    interesting_predicate_bscore,
)
from pymoses.moses.scoring.discriminating_bscore import (
    bep_bscore,
    f_one_bscore,
    prerec_bscore,
    recall_bscore,
)
from pymoses.moses.scoring.precision_bscore import precision_bscore, precision_conj_bscore
from pymoses.moses.scoring.select_bscore import select_bscore
from pymoses.moses.scoring.time_dispersion import TemporalGranularity
from pymoses.util.logger import logger

__all__ = [
    "bep_table_problem",
    "cluster_table_problem",
    "f_one_table_problem",
    "ip_problem_params",
    "ip_table_problem",
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


def _log_alphabet_size(alphabet_size: int, output_type: type_node) -> int:
    """C++: moses/moses/main/problem.cc:138 alphabet_size

    Not a C++ symbol: `alphabet_size` has one `as` variable, one switch and
    one `logger().info() << "Alphabet size = " << as << " output = " <<
    output_type;` at problem.cc:198-200 before its single `return`. This
    port's `_alphabet_size` returns early from each arm instead, so the
    shared tail lives here. Returns its first argument so a call site reads
    `return _log_alphabet_size(<expr>, output_type)`.
    """
    logger().info(f"Alphabet size = {alphabet_size} output = {ostream_type_node(output_type)}")
    return alphabet_size


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
    # C++ assigns each branch to a single `as` and falls through to ONE
    # `logger().info()` at problem.cc:198-200, just before `return as`
    # (the unsupported-type arm never reaches it -- `log_output_error_exit`
    # exits). `_log_alphabet_size` below is that single exit point, applied
    # to every arm that returns.
    if output_type == type_node.boolean_type:
        # 3 operators: and, or, not
        return _log_alphabet_size(3 + arity, output_type)
    if output_type == type_node.contin_type:
        # C++ assigns this to `unsigned as` (problem.cc:139) after
        # computing it in `size_t`, so an `ignore_ops` larger than
        # `8 + arity` wraps near 2**64 and is then truncated to 32 bits.
        # See this module's docstring for the full two-stage wrap and why
        # it is reproduced rather than left to Python's arbitrary-precision
        # ints. Reachable since `it_table_problem.run`'s contin arm was
        # wired; the comment here used to say "unreachable from every
        # problem this file wires".
        return _log_alphabet_size((8 + arity - len(ignore_ops)) & _UINT32_MAX, output_type)
    if output_type == type_node.ann_type:
        return _log_alphabet_size(2 + arity * arity, output_type)
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
# ip_problem_params -- table-problems.h:76 / table-problems.cc:204.
# ---------------------------------------------------------------------------


class ip_problem_params:
    """C++: moses/moses/main/table-problems.h:76 ip_problem_params

    A SECOND `option_base` that `register_table_problems` constructs and
    registers alongside `table_problem_params` (table-problems.cc:
    565-566), carrying the four `-H ip` weights and nothing else.

    It was missing entirely until the S13-CLI option-surface slice, and
    with it the only four flags classic's `--help` lists that this port's
    parser did not accept: `-K/--ip_kld_weight`, `-J/--ip_skewness_weight`,
    `-U/--ip_stdU_weight`, `-X/--ip_skew_U_weight`. Classic exits 0 on all
    four (they are parsed and stored before any problem runs); this port
    exited 2 with argparse's "unrecognized arguments". With these four
    added the two option surfaces are exactly equal -- 139 options each,
    no port-only extras -- which `tests/moses/main/test_moses_exec.py`
    pins directly against the compiled binary's own `--help`.

    They are stored and NOT gated, deliberately. `ip_problem` is one of
    the two table problems `problem.py` still registers as an unported
    stub, and classic's own `-H ip` `OC_ASSERT`s on any table whose output
    is not contin with boolean inputs (table-problems.cc:236) -- so
    gating the WEIGHTS would reject command lines classic accepts, to
    protect a problem that fails for its own reasons the moment it is
    actually selected. Gate the problem, not its parameters.
    """

    def __init__(self) -> None:
        """No C++ constructor line to cite: `ip_problem_params` declares no
        constructor of its own (table-problems.h:76-83). Unlike
        `table_problem_params`, all four of its options DO carry
        `->default_value(1.0)` (table-problems.cc:211,215,219,223), so the
        field defaults below are those, not boost's empty-value defaults.
        """
        self.ip_kld_weight = 1.0
        self.ip_skewness_weight = 1.0
        self.ip_stdU_weight = 1.0
        self.ip_skew_U_weight = 1.0

    def add_options(self, parser: argparse.ArgumentParser) -> None:
        """C++: moses/moses/main/table-problems.cc:204 ip_problem_params::add_options

        `po::value<double>` -- `contin_t`-shaped, so `type=float`. The
        short forms are `,K`/`,J`/`,U`/`,X` in the boost option names and
        are part of the surface: classic accepts `-K2` and `--ip_kld_weight=2`
        alike. The long names keep their C++ spelling exactly, underscores
        included -- they are the only options in the whole CLI that use
        underscores rather than hyphens, and normalising them would invent
        a flag classic does not have while dropping one it does.
        """
        add = parser.add_argument
        add("-K", "--ip_kld_weight", dest="ip_kld_weight", type=float, default=1.0)
        add("-J", "--ip_skewness_weight", dest="ip_skewness_weight", type=float, default=1.0)
        add("-U", "--ip_stdU_weight", dest="ip_stdU_weight", type=float, default=1.0)
        add("-X", "--ip_skew_U_weight", dest="ip_skew_U_weight", type=float, default=1.0)

    def parse_options(self, args: argparse.Namespace) -> None:
        """No direct C++ line: `ip_problem_params` declares no
        `parse_options` override either (table-problems.h:76-83), so C++
        uses `option_base::parse_options`' empty body and the fields are
        populated by boost's own `po::value<double>(&member)` binding. Done
        by hand here for the same reason `table_problem_params.parse_options`
        does it.
        """
        self.ip_kld_weight = args.ip_kld_weight
        self.ip_skewness_weight = args.ip_skewness_weight
        self.ip_stdU_weight = args.ip_stdU_weight
        self.ip_skew_U_weight = args.ip_skew_U_weight


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
        # C++: table-problems.cc:111 `size_t num_rows = 0;` -- accumulated
        # across every `-i` file purely to feed the log line after the loop.
        num_rows = 0
        for idf in self._tpp.input_data_files:
            logger().info("Read data file %s", idf)
            table = loadTable(
                idf,
                self._tpp.target_feature,
                self._tpp.timestamp_feature,
                self._tpp.ignore_features_str,
            )
            num_rows += table.size()

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
            # table-problems.cc:133-134. `CTable.balance()` (table.py) is a
            # faithful port of a function that PROVABLY does nothing: both
            # of its C++ range-for loops bind by value, so the scaling it
            # computes is written to temporaries and dropped. See that
            # method's own docstring for the measurement (and for why
            # "fixing" it would put this port at odds with every classic
            # `--balance` run).
            if pms.balance:
                self._ctables[-1].balance()

        logger().info("Number of rows in tables = %d", num_rows)

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

        logger().info("Inferred arity = %d", self.arity)

        # C++: pms.mmr_pa.ilabels = ilabels (table-problems.cc:164). See
        # this module's own docstring, "`pms.printer` stands in for
        # `pms.mmr_pa`".
        pms.printer.ilabels = self.ilabels

    def common_type_setup(
        self, pms: problem_params, out_type: type_node = type_node.unknown_type
    ) -> None:
        """C++: moses/moses/main/table-problems.cc:170 table_problem_base::common_type_setup"""
        self.table_type_signature = self.table.get_signature()
        logger().info("Table signature " + ostream_type_signature(self.table_type_signature))

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

        logger().info("Inferred output type: " + ostream_type_node(self.output_type))

    def _regression(self, pms: problem_params, make_bscore, table_or_ctable=None) -> None:
        """C++: moses/moses/main/table-problems.cc:328 REGRESSION (macro)

        `make_bscore` is a zero-argument factory, not an already-built
        scorer. The macro's statement order is
        `int as = alphabet_size(...); SCORER bscore ARGS;` -- `as` FIRST --
        and that order is observable: `alphabet_size` logs
        "Alphabet size = ...", every `bscore_ctable_base` subclass' ctor
        logs something of its own (`ctruth_table_bscore`'s
        `set_best_possible_bscore`, for one), and classic's log shows the
        alphabet line first. Passing a constructed scorer in, as this method
        did until the logging surface landed, silently reversed the two.

        Shared by every `*_table_problem::run()` that uses the
        `REGRESSION` macro verbatim in classic and is wired for real
        here: `it` (both the boolean and the contin branch), `prerec`,
        `recall`, `bep`, `f_one`, `select` and `cluster`. NOT used by
        `pre_table_problem` alone (hand-inlined in classic too, see
        this module's own docstring); `pre-conj` DOES use it, as classic
        does (table-problems.cc:421-422).

        `table_or_ctable` IS the macro's FIRST argument. classic uses it
        for one thing only -- constructing the `feature_selector`
        (table-problems.cc:333) -- and which of `ctable`/`table` each call
        site passes is not cosmetic: `feature_selector`'s two C++
        constructors differ (feature_selector.cc:51 takes a `CTable`
        as-is; :55 takes a `Table` and compresses it), and the three
        `REGRESSION(table, ...)` sites (cc:519, :522, :557) really do pass
        the uncompressed one. Defaulting to `None` keeps the parameter
        optional for a caller that has already refused feature selection.
        """
        # C++: table-problems.cc:330-334 -- "Enable feature selection while
        # selecting exemplar". Ported for real since 2026-09-10; it raised
        # `NotImplementedError` here until then, on the (by then false)
        # grounds that `feature_selector` was "not ported anywhere in this
        # project". It is: `pymoses.moses.deme.feature_selector`.
        #
        # A CLI run still cannot reach this with `--enable-fs=1`, because
        # `problem_params.parse_options` refuses that flag earlier -- but
        # the blocker is now one layer DEEPER than this line, in
        # `deme_expander.optimize_demes`' own `if (_params.fstor)` arm
        # (deme_expander.cc:432-450), which needs `behave_cscore.
        # ignore_cols()`/`best_possible_score()`. See that gate's own
        # message. This construction site is exercised directly by
        # `tests/moses/main/test_table_problems.py`.
        if pms.enable_feature_selection and pms.fs_params.target_size > 0:
            pms.deme_params.fstor = feature_selector(table_or_ctable, pms.festor_params)

        # C++: table-problems.cc:335,372 `int as = alphabet_size(...)` --
        # the `unsigned` return narrowed to a signed 32-bit `int`. Only the
        # contin/enum branches can produce a value this changes; see this
        # module's docstring for the two-stage wrap.
        as_ = _to_int32(_alphabet_size(self.cand_type_signature, pms.ignore_ops))
        # C++: table-problems.cc:336 `SCORER bscore ARGS ;` -- constructed
        # here, after `as`, never before.
        bscore = make_bscore()
        _set_noise_or_ratio(bscore, as_, pms.noise, pms.complexity_ratio)
        # NOT A PORT (pyMoses extension, `--linear-scaling 1`; no C++
        # counterpart): regression (`contin_bscore`, squared error) scores
        # a + b*f, the least-squares scaling of each program. Set on the one
        # scorer every optimizer (hc, hc-capped, gomea, hc-gomea) shares.
        if getattr(pms, "linear_scaling", 0):
            if self.output_type != type_node.contin_type or not isinstance(bscore, contin_bscore):
                raise ValueError(
                    "--linear-scaling applies to regression only (a contin output "
                    "scored by contin_bscore); this problem has output type "
                    f"{self.output_type.name} and scorer {type(bscore).__name__}"
                )
            if pms.it_abs_err:
                raise ValueError(
                    "--linear-scaling is a least-squares fit and needs squared error; "
                    "it cannot be combined with --it-abs-err"
                )
            bscore.set_linear_scaling(True)
        if pms.meta_params.do_boosting:
            bscore.use_weighted_scores()
        # When boosting, cache must not be used, as otherwise, stale
        # composite scores get cached and returned.
        if pms.meta_params.do_boosting:
            pms.cache_size = 0

        mbcscore = behave_cscore(bscore, pms.cache_size)

        # NOT A PORT (pyMoses extension, `--simplify-exemplar-numeric`; no
        # C++ counterpart): numerical simplification of exemplars, for
        # regression (`contin_bscore`) only. See `numeric_simplify.py`.
        tol = getattr(pms, "simplify_exemplar_numeric", 0.0)
        if tol > 0:
            if self.output_type != type_node.contin_type or not isinstance(bscore, contin_bscore):
                raise ValueError(
                    "--simplify-exemplar-numeric applies to regression only "
                    "(a contin output scored by contin_bscore); this problem "
                    f"has output type {self.output_type.name} and scorer "
                    f"{type(bscore).__name__}"
                )
            pms.deme_params.numeric_simplifier = exemplar_numeric_simplifier(
                bscore._cti, [get_contin(v) for v in bscore._target], tol
            )

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
            pms.ps_params,
            pms.deme_params,
            pms.filter_params,
            pms.meta_params,
            pms.moses_params,
            pms.printer,
            # GOMEA extension (not in classic MOSES): `-a gomea` settings.
            **gomea_kwargs(pms),
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
            self._regression(pms, lambda: ctruth_table_bscore(self.ctable), self.ctable)
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
                # C++: table-problems.cc:520-524. This arm raised until
                # 2026-09-10, on the accurate grounds that
                # `discretize_contin_bscore` was not ported. It is now.
                self._regression(
                    pms,
                    lambda: discretize_contin_bscore(
                        self.table.otable,
                        self.table.itable,
                        pms.discretize_thresholds,
                        pms.weighted_accuracy,
                    ),
                    # C++: REGRESSION(table, ...) -- table-problems.cc:522
                    # passes the UNCOMPRESSED table here, not ctable.
                    self.table,
                )
                return
            eft = contin_bscore.abs_error if pms.it_abs_err else contin_bscore.squared_error
            # C++: REGRESSION(table, ...) -- table-problems.cc:519.
            self._regression(pms, lambda: contin_bscore(self.table, eft=eft), self.table)
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

        # C++: table-problems.cc:367-370. `pre_table_problem::run` is
        # "very nearly identical to the REGRESSION macro" (its own C++
        # comment) and hand-inlines this construction -- see
        # `_regression`'s own comment for the deeper blocker.
        if pms.enable_feature_selection and pms.fs_params.target_size > 0:
            pms.deme_params.fstor = feature_selector(self.ctable, pms.festor_params)

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
            pms.ps_params,
            pms.deme_params,
            pms.filter_params,
            pms.meta_params,
            pms.moses_params,
            pms.printer,
            # GOMEA extension (not in classic MOSES): `-a gomea` settings.
            **gomea_kwargs(pms),
        )


# ---------------------------------------------------------------------------
# pre_conj_table_problem -- table-problems.h:129 / table-problems.cc:410.
# ---------------------------------------------------------------------------


class pre_conj_table_problem(table_problem_base):
    """C++: moses/moses/main/table-problems.h:129 pre_conj_table_problem

    Registered under the name `pre-conj`, with a HYPHEN
    (`table-problems.h:134`, `return "pre-conj";`) -- classic has no
    `pre_conj` name at all, so `-Hpre_conj` prints `problem type
    "pre_conj" is currently unsupported.` and exits 1 there just as it
    does here (measured against the real binary, 2026-09-09; the port
    reaches the identical message through `moses_exec.py`'s own
    `find_problem` miss). The class NAME keeps C++'s underscore
    spelling; only `name()` carries the hyphen.
    """

    def name(self) -> str:
        return "pre-conj"

    def description(self) -> str:
        return "Precision-Conjunction-Maximization"

    def run(self, ob: object) -> None:
        """C++: moses/moses/main/table-problems.cc:410 pre_conj_table_problem::run

        The plain `REGRESSION` macro shape (unlike `pre_table_problem`,
        which hand-inlines a variant of it), so `_regression` does the
        rest.
        """
        pms = ob
        self.common_setup(pms)
        self.common_type_setup(pms, type_node.boolean_type)

        # C++: table-problems.cc:419-420 `OC_ASSERT(not pms.meta_params.
        # do_boosting, "Boosting not supported for the pre problem!")`,
        # with classic's own message text (it says "pre", not "pre-conj").
        # Mapped to `NotImplementedError` the same way `demo_problems.py`
        # maps its three identical `OC_ASSERT(not do_boosting)` sites;
        # classic throws `AssertionException` out of `main` and dumps
        # core, which is not reproducible as library behaviour.
        if pms.meta_params.do_boosting:
            raise NotImplementedError("Boosting not supported for the pre problem!")

        self._regression(
            pms,
            lambda: precision_conj_bscore(self.ctable, pms.hardness, pms.pre_positive),
            self.ctable,
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
        self._regression(
            pms,
            lambda: prerec_bscore(
                self.ctable, pms.min_rand_input, pms.max_rand_input, pms.hardness
            ),
            self.ctable,
        )


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
        self._regression(
            pms,
            lambda: recall_bscore(
                self.ctable, pms.min_rand_input, pms.max_rand_input, pms.hardness
            ),
            self.ctable,
        )


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
        self._regression(
            pms,
            lambda: bep_bscore(self.ctable, pms.min_rand_input, pms.max_rand_input, pms.hardness),
            self.ctable,
        )


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

        # DO NOT FIX: `f_one_bscore` always raises.
        self._regression(pms, lambda: f_one_bscore(self.ctable), self.ctable)  # pragma: no cover


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

        self._regression(
            pms,
            lambda: select_bscore(
                self.ctable,
                pms.min_rand_input,
                pms.max_rand_input,
                pms.hardness,
                pms.pre_positive,
            ),
            self.ctable,
        )


# ---------------------------------------------------------------------------
# cluster_table_problem -- table-problems.h:220 / table-problems.cc:551.
# ---------------------------------------------------------------------------


class cluster_table_problem(table_problem_base):
    """C++: moses/moses/main/table-problems.h:220 cluster_table_problem

    "Perform clusterering" (table-problems.h:219's own comment, typo
    included). The one table problem whose scorer looks at the INPUT
    table only: `cluster_bscore` (bscores.py) takes `table.itable` and
    never sees `table.otable`, so `-u <target>` still decides which
    column is EXCLUDED from the inputs, but the values in it are never
    compared against anything.
    """

    def name(self) -> str:
        return "cluster"

    def description(self) -> str:
        return "Discover clustering function"

    def run(self, ob: object) -> None:
        """C++: moses/moses/main/table-problems.cc:551 cluster_table_problem::run

        Four lines in classic, and all four are here: `common_setup`,
        `common_type_setup(pms, id::contin_type)` -- note contin, NOT the
        boolean every other non-`it` table problem forces -- and
        `REGRESSION(table, cluster_bscore, (table.itable))`, which is
        `_regression` (this module).

        `REGRESSION`'s first macro argument is used ONLY to build a
        `feature_selector` (table-problems.cc:330-334), which this port
        gates on rather than building; so `table` vs `ctable` there makes
        no difference to any run this port can perform, and `_regression`
        takes no such argument. The `(table.itable)` in the third
        argument is the real one -- the scorer's own constructor
        argument.

        `self.output_type` is `contin_type` after `common_type_setup`
        (barring an explicit `-y` exemplar with a different output type),
        so `_regression` takes its `pms.contin_reduct` branch, exactly as
        `it_table_problem`'s contin arm does.
        """
        pms = ob
        self.common_setup(pms)
        self.common_type_setup(pms, type_node.contin_type)

        # C++: REGRESSION(table, ...) -- table-problems.cc:557.
        self._regression(pms, lambda: cluster_bscore(self.table.itable), self.table)


# ---------------------------------------------------------------------------
# ip_table_problem -- table-problems.h:86 / table-problems.cc:229.
# ---------------------------------------------------------------------------


class ip_table_problem(table_problem_base):
    """C++: moses/moses/main/table-problems.h:86 ip_problem

    "Find interesting predicates" (table-problems.h:85's own comment).
    Named `ip_problem` in C++ and `ip_table_problem` here, matching the
    `*_table_problem` spelling every sibling in this module already uses
    for the classes `register_table_problems` registers (C++ itself is
    inconsistent: `ip_problem`/`ann_table_problem`/`pre_table_problem`).
    `name()` returns classic's own `"ip"`.

    Takes a SECOND params object, `ip_problem_params` -- the only table
    problem that does. C++ builds both inside `register_table_problems`
    and hands them to `new ip_problem(*tpp, *ippp)` (table-problems.cc:
    564-570); this port's `register_table_problems` does the same, taking
    the `ip_problem_params` as an optional argument so that the ONE
    instance the argument parser fills is the one the problem reads (the
    same shared-instance discipline `parse_table_args`' own docstring
    already spells out for `table_problem_params`).

    REACHABILITY, measured against classic's own binary rather than
    assumed. `-H ip` exits 0 on any table whose output column is contin
    and whose every input column is boolean, and aborts (SIGABRT, exit
    134) on anything else -- the `OC_ASSERT` at table-problems.cc:236,
    reproduced in `run` below. At classic's DEFAULT activation bounds it
    prints exactly one line, `nan true`, because
    `interesting_predicate_bscore`'s activation penalty is `0.0f/0.0f`
    in both arms; with `-q 0.2 -w 0.8` it prints a normal ranked
    candidate list. Both are covered by
    `tests/moses/main/test_ip_problem_cli.py`.
    """

    def __init__(self, tpp: table_problem_params, ippp: ip_problem_params) -> None:
        """C++: moses/moses/main/table-problems.h:89 ip_problem

        `ip_problem(table_problem_params& tp, ip_problem_params& ip) :
        table_problem_base(tp), _ippp(ip) {}` -- both are REFERENCES
        (table-problems.h:96 declares `ip_problem_params& _ippp;`), so
        neither is copied and both see whatever the option parser later
        writes into them. Python's plain attribute binding is that.
        """
        super().__init__(tpp)
        self._ippp = ippp

    def name(self) -> str:
        return "ip"

    def description(self) -> str:
        return "Find interesting patterns"

    def run(self, ob: object) -> None:
        """C++: moses/moses/main/table-problems.cc:229 ip_problem::run

        NOT the `REGRESSION` macro and NOT `common_type_setup` either --
        this is the only registered table problem that calls
        `common_setup` alone (cc:232) and then builds its own signatures
        by hand. Three consequences worth naming, because each is a place
        a copy-paste from `it_table_problem` would go wrong:

        1. `self.cand_type_signature` and `self.output_type` are NEVER
           SET here (they are `common_type_setup`'s job). `run` uses its
           own local `tt`, an ALL-BOOLEAN signature
           (`gen_signature(id::boolean_type, arity)`, cc:241), for both
           `alphabet_size` and `metapop_moses_results`. The candidate is
           a Boolean predicate over Boolean inputs even though the
           TABLE's output column is contin.
        2. The default exemplar is `type_to_exemplar(id::boolean_type)`
           (cc:244-245), chosen from that same Boolean output type rather
           than from the table's.
        3. `alphabet_size(tt, ...)` therefore always takes the BOOLEAN
           arm (`3 + arity`, problem.cc:141-144), so the contin arm's
           unsigned wrap cannot be reached from here.

        The `OC_ASSERT` at cc:236-239 is the reachability gate this whole
        problem lives behind: the table's signature must be exactly
        `->(boolean ... boolean) contin`. Measured against classic on
        three tables -- a Boolean-output one, a contin-input one and a
        Boolean-input/contin-output one -- the first two abort with
        SIGABRT and this exact message and the third exits 0. Ported as
        an `AssertionError` carrying classic's own text, this port's
        convention for a direct `OC_ASSERT(false, ...)`.

        `OC_ASSERT(not pms.meta_params.do_boosting, ...)` (cc:263-264) is
        the second gate, with classic's own reason above it: "In order to
        support boosting, the interesting_predicate_bscore would have to
        modified to always return a bscore that is the smae length, and
        so that the same item always refered to the same row in the
        ctable." (cc:259-262, typos and all.)

        Nothing here does feature selection (no `fstor` block), calls
        `use_weighted_scores`, or zeroes `pms.cache_size` -- all three
        belong to the `REGRESSION` macro and to `pre_table_problem`, and
        `ip_problem::run` has none of them.
        """
        pms = ob
        self.common_setup(pms)

        # ip assumes that the inputs are boolean and the output is contin
        # C++: table-problems.cc:234-239.
        ettt = gen_signature(type_node.boolean_type, type_node.contin_type, self.arity)
        if ettt != self.table.get_signature():
            raise AssertionError(
                "The input table doesn't have the right data types."
                " The output should be contin and the inputs should"
                " be boolean"
            )

        # signature of the functions to learn. C++: table-problems.cc:239-240.
        tt = gen_signature(type_node.boolean_type, self.arity)

        # determine the default exemplar to start with
        # C++: table-problems.cc:243-245.
        if not pms.exemplars:
            pms.exemplars.append(_type_to_exemplar(type_node.boolean_type))

        # C++: table-problems.cc:247 `int as = alphabet_size(tt, pms.ignore_ops);`
        # -- note `tt`, the BOOLEAN signature, not `cand_type_signature`.
        as_ = _to_int32(_alphabet_size(tt, pms.ignore_ops))

        bscore = interesting_predicate_bscore(
            self.ctable,
            self._ippp.ip_kld_weight,
            self._ippp.ip_skewness_weight,
            self._ippp.ip_stdU_weight,
            self._ippp.ip_skew_U_weight,
            pms.min_rand_input,
            pms.max_rand_input,
            pms.hardness,
            pms.pre_positive,
        )
        _set_noise_or_ratio(bscore, as_, pms.noise, pms.complexity_ratio)

        # In order to support boosting, the interesting_predicate_bscore
        # would have to modified to always return a bscore that is the
        # smae length, and so that the same item always refered to the
        # same row in the ctable. C++: table-problems.cc:259-264.
        if pms.meta_params.do_boosting:
            raise AssertionError("Boosting not supported for the ip problem!")

        mbcscore = behave_cscore(bscore, pms.cache_size)
        metapop_moses_results(
            pms.exemplars,
            tt,
            pms.bool_reduct,
            pms.bool_reduct_rep,
            mbcscore,
            pms.opt_params,
            pms.hc_params,
            pms.ps_params,
            pms.deme_params,
            pms.filter_params,
            pms.meta_params,
            pms.moses_params,
            pms.printer,
            # GOMEA extension (not in classic MOSES): `-a gomea` settings.
            **gomea_kwargs(pms),
        )


# ---------------------------------------------------------------------------
# register_table_problems -- table-problems.cc:562-581.
# ---------------------------------------------------------------------------


def register_table_problems(
    pmr: problem_manager, ippp: ip_problem_params | None = None
) -> table_problem_params:
    """C++: moses/moses/main/table-problems.cc:562 register_table_problems

    `void register_table_problems(problem_manager&, option_manager&)` in
    C++; the `option_manager&` parameter is dropped (nothing to register
    CLI options against centrally -- see this module's own docstring) and
    the freshly built `table_problem_params` is RETURNED instead (no C++
    equivalent, same convention `demo_problems.py`'s `register_demo_problems`
    already established for its own shared `demo_params`).

    Registers the TEN problems this port has wired for real, then
    defers to `problem.py`'s own `_register_unported_table_problems` for
    the remaining one (`ann-it`) -- see this module's own docstring,
    "Composing on top of `problem.py` without editing it", for why
    calling it AFTER the real registrations is safe
    (`register_problem`'s `dict.setdefault` semantics). It said "seven ...
    remaining four (..., `cluster`)" until `cluster_table_problem` landed
    below, "EIGHT ... remaining three (..., `pre-conj`)" until
    `pre_conj_table_problem` did, and "NINE ... remaining two (`ip`,
    `ann-it`)" until `ip_table_problem` did.

    `ippp` has no C++ counterpart as a PARAMETER: classic constructs its
    `ip_problem_params` here (table-problems.cc:567) and registers it with
    the `option_manager` on the next line, which is how the one instance
    reaches both `new ip_problem(*tpp, *ippp)` and the parser. This port
    has no `option_manager`, so the caller that owns the argparse parser
    (`moses_exec.py`) constructs the instance and passes it IN; omitting
    it builds a private one at classic's defaults, which is right for
    every caller that never parses `-K/-J/-U/-X`. Handing `ip_table_problem`
    a DIFFERENT instance from the one the parser fills would silently run
    `-H ip` at default weights no matter what the command line said --
    the exact bug this parameter exists to prevent, and the one the old
    "construct a throwaway at the parser" arrangement would have had the
    moment `-H ip` stopped being a stub.
    """
    tpp = table_problem_params()
    if ippp is None:
        ippp = ip_problem_params()
    pmr.register_problem(ip_table_problem(tpp, ippp))
    pmr.register_problem(it_table_problem(tpp))
    pmr.register_problem(pre_table_problem(tpp))
    pmr.register_problem(pre_conj_table_problem(tpp))
    pmr.register_problem(prerec_table_problem(tpp))
    pmr.register_problem(recall_table_problem(tpp))
    pmr.register_problem(bep_table_problem(tpp))
    pmr.register_problem(f_one_table_problem(tpp))
    pmr.register_problem(select_table_problem(tpp))
    pmr.register_problem(cluster_table_problem(tpp))
    _register_unported_table_problems(pmr)
    return tpp


def parse_table_args(
    argv: list[str], tpp: table_problem_params, ippp: ip_problem_params | None = None
) -> problem_params:
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
    # `ip_problem_params` is the third `option_base` classic registers into
    # the ONE shared `options_description` (problem.cc:50-56, via
    # table-problems.cc:565-566), so its four flags parse here too.
    #
    # CORRECTED: this used to build a throwaway instance and DISCARD the
    # parsed values, on the stated grounds that "the `-H ip` problem that
    # would read them is an unported stub". It no longer is
    # (`ip_table_problem` above), so a discarded instance would run `-H ip`
    # at default weights whatever `-K/-J/-U/-X` said. `ippp` is now taken
    # as a parameter for the same reason `tpp` is: it must be the SAME
    # instance `register_table_problems` wired into `ip_table_problem`.
    # Omitting it keeps the old behaviour -- the flags still parse without
    # error -- for callers that never select `-H ip`.
    if ippp is None:
        ippp = ip_problem_params()
    ippp.add_options(parser)
    ns = parser.parse_args(argv)
    pp.parse_options(ns)
    tpp.parse_options(ns)
    ippp.parse_options(ns)
    return pp
