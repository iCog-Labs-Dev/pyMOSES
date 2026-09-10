"""C++: moses/comboreduct/table/table.h

`ITable` (table.h:738), `OTable` (table.h:866), `CTable` (table.h:550) and
`Table` (table.h:918) -- the data structures `loadTable` (table_io.py)
populates and `Table.compressed()` groups into a `CTable`.

Scope narrowing versus classic, disclosed here once rather than repeated
on every method:

* No `multi_type_seq`. Classic represents a *row* as a
  `boost::variant<builtin_seq, contin_seq, string_seq, vertex_seq,
  combo_tree_seq>` purely to save RAM (table.h:79-99's own comment: "the
  below is sort of weird... vector<variant<T1,...,Tn>> takes a lot more
  RAM than variant<vector<T1>,...,vector<Tn>>"). That trade-off has no
  Python analogue -- a Python list already stores heterogeneous objects
  as boxed pointers regardless -- so every `ITable` row here is simply a
  `tuple[vertex, ...]`, one element per column, matching `multi_type_seq`
  only in its *observable* behaviour (`get_at`, `size`, iteration), never
  its internal representation.
* No general `type_checker` inference engine. A classic signature-shaped
  `type_tree` (`lambda_type(itype1, ..., itypeN, otype)`) is represented
  by the immutable `TypeSignature` in `type_checker/type_signature.py`.
  `Signature` remains a compatibility alias here so pre-S8 table and
  scoring callers keep the same import and constructor surface.
* `TTable`/`TimedValue`/`TimedCounter`/`CTableTime` ARE ported (they were
  not, until the timestamp slice). `loadTable` fills `Table.ttable` from a
  `--timestamp-feature` column instead of discarding it, and
  `Table.compressed` keys each observation on `TimedValue(output,
  timestamp)` the way classic does. Two structural choices are documented
  as DIVERGENCE markers where they live rather than repeated here:

  - `TimedCounter`'s MAPPING interface is classic's `untimedCounter()`
    view (vertex -> summed count), with the `TimedValue`-keyed observation
    list reachable via `timed_items()`. See its class docstring.
  - `TimedValue.__lt__` is transliterated verbatim including its bug, and
    is NOT used to order any container. Classic's
    `Counter<TimedValue, count_t>` is a `std::map` under a comparator that
    is not a strict weak ordering, so classic's container silently loses
    or duplicates observations; this port computes the intended result.
    Measured in full in `TimedCounter`'s class docstring.
"""

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import date
from functools import cmp_to_key

from pymetta.comboreduct.combo.vertex import (
    bool_to_vertex,
    builtin,
    builtin_to_bool,
    combo_tree,
    get_contin,
    vertex,
    vertex_equal,
    vertex_less,
    vertex_rank,
)
from pymetta.comboreduct.interpreter.eval import arity as _tree_arity
from pymetta.comboreduct.interpreter.interpreter import boolean_interpreter, interpreter_visitor
from pymetta.comboreduct.type_checker.type_signature import (
    TypeSignature,
    boolean_arity,
    contin_arity,
    type_node,
)
from pymetta.util.logger import logger
from pymetta.util.rng import randGen

# Compatibility re-export for existing table/scoring callers.  New code should
# import TypeSignature from type_checker.type_signature.
Signature = TypeSignature


def _vertex_type_node(v: vertex) -> type_node:
    """Map a concrete Python value back to the `type_node` it was built
    from -- substitutes for `get_type_node(get_type_tree(v))`
    (type_checker/type_tree.cc), used by `ITable::insert_col`
    (table.cc:197) to infer a freshly-inserted column's type from its own
    values. Not a ported symbol (see module docstring: `type_checker` is
    out of scope) -- just enough type-of-value dispatch to cover this
    port's narrowed `vertex = builtin | argument | contin_t | str`.
    """
    if isinstance(v, bool):
        raise TypeError("bare bool is not a valid table vertex")
    if isinstance(v, builtin):
        return type_node.boolean_type
    if isinstance(v, float):
        return type_node.contin_type
    if isinstance(v, str):
        return type_node.enum_type
    raise TypeError(f"cannot infer a column type for {v!r}")


# ---------------------------------------------------------------------------
# get_indices -- table.cc:563
# ---------------------------------------------------------------------------


def get_indices(labels: Sequence[str], header: Sequence[str]) -> list[int]:
    """C++: moses/comboreduct/table/table.cc:563 get_indices

    "Get indices (aka positions or offsets) of a list of labels given a
    header. The labels can be sequenced in any order, it will always
    return the order consistent with the header." (table.h:64-68's own
    doc comment, reproduced verbatim -- this is the definitive answer for
    how `ignore_features` names turn into column positions: always in
    ascending, header-relative order, regardless of the order the caller
    listed them in.)
    """
    return [i for i, h in enumerate(header) if h in labels]


# ---------------------------------------------------------------------------
# Contin-target discretization -- table.cc:573 / table.cc:583
#
# DUPLICATE ALERT, disclosed rather than silently left to drift:
# `pymetta/feature_selection/scorers/mutual_info.py` carries its own copy of
# `TARGET_DISCRETIZED_BINS_NUM`/`discretize_contin_feature`/
# `get_discrete_bin`, landed concurrently, with a PLACEMENT divergence
# marker saying they belong here and were put there only because this file
# was owned by another task. These, in `table.py`, are the ones whose C++
# home (`table.cc`) matches; `mutual_info.py`'s copies should import from
# here and be deleted. Until that happens there are two implementations of
# the same three symbols, and this comment is the only thing that says so.
# Both are pinned against independently captured C++ fixtures
# (`ctable_balance.json`'s `discretize` group here, `mi_contin.json`'s
# `discretize` group there), so a drift between them turns one of the two
# red rather than going unnoticed.
# ---------------------------------------------------------------------------

# C++: moses/comboreduct/table/table.h:56 TARGET_DISCRETIZED_BINS_NUM
# `#define TARGET_DISCRETIZED_BINS_NUM 5  // discretize contin type target`
TARGET_DISCRETIZED_BINS_NUM = 5


def discretize_contin_feature(min_: float, max_: float) -> list[float]:
    """C++: moses/comboreduct/table/table.cc:573 discretize_contin_feature

    Returns the `TARGET_DISCRETIZED_BINS_NUM` LEFT EDGES of an equal-width
    partition of `[min_, max_]`: `min_ + i*interval` for `i in 0..4`, with
    `interval = (max_ - min_)/5`. The last edge is `min_ + 4*interval`,
    never `max_` itself -- the bins are half-open and the top one is
    unbounded above (see `get_discrete_bin`).

    `contin_t` is `double` and `TARGET_DISCRETIZED_BINS_NUM` is an `int`
    macro promoted by the division, so the arithmetic is plain binary64 in
    both languages. `min_ + i*interval` is computed exactly as written
    (one multiply, one add per element), so its rounding is reproduced
    rather than avoided by an accumulating or compensated sum.
    """
    interval = (max_ - min_) / TARGET_DISCRETIZED_BINS_NUM
    return [min_ + i * interval for i in range(TARGET_DISCRETIZED_BINS_NUM)]


def get_discrete_bin(disc_intvs: Sequence[float], val: float) -> builtin:
    """C++: moses/comboreduct/table/table.cc:583 get_discrete_bin

    Returns the 1-based index of the bin `val` falls in, CAST TO
    `builtin` -- `return (builtin)i;`, with `i` in `1..5`. The loop scans
    `i = 1..4` and breaks on the first `val < disc_intvs[i]`; falling off
    the end leaves `i == 5`. `disc_intvs[0]` (the bottom edge) is
    therefore never consulted: a `val` below it still lands in bin 1, and
    bin 5 is unbounded above.

    The `(builtin)` cast genuinely reinterprets a bin ORDINAL as a combo
    builtin token, so bins 1..5 come out as `logical_true`(1),
    `logical_false`(2), `logical_and`(3), `logical_or`(4),
    `logical_not`(5) -- vertex.h:72-79's own enumerator order. That is not
    a transcription slip here: the values are used only as opaque distinct
    keys for the `ioc`/`ycount` counters inside
    `mutualInformation(CTable, FeatureSet)` (table.h:1220), never
    interpreted as logic. Reproduced exactly, because the identity of
    those keys is observable through the resulting entropy sums the moment
    a bin ordinal collides with a real output value.

    DO NOT FIX: bin 5 landing on `logical_not` (an operator, not a
    constant) is classic's own behaviour, and is what a differential run
    has to reproduce.
    """
    i = 1
    while i < TARGET_DISCRETIZED_BINS_NUM:
        if val < disc_intvs[i]:
            break
        i += 1
    return builtin(i)


# ---------------------------------------------------------------------------
# ITable -- table.h:738
# ---------------------------------------------------------------------------

_DEFAULT_INPUT_LABEL = "i"

# `COEF_SAMPLE_COUNT` -- table.h:52, "involved in the formula that counts
# the number of trials needed for a given arity".
_COEF_SAMPLE_COUNT = 20.0


def _pow2(x: int) -> int:
    """C++: opencog/util/numeric.h:272 pow2

    `inline unsigned int pow2(unsigned int x) { return pow2<unsigned int>(x); }`,
    i.e. `1u << x`. Only ever called here on a `boolean_arity`, which is a
    non-negative count.
    """
    return 1 << x


def _sample_count(contin_arity_: int) -> int:
    """C++: moses/comboreduct/table/table.h:849 ITable::sample_count

    "this function take an arity in input and returns in output the number
    of samples that would be appropriate to check the semantics of its
    associated tree" (table.h:842-845's own comment).
    `COEF_SAMPLE_COUNT*log(contin_arity + M_E)` -- a `double` expression
    returned as `unsigned`, so it TRUNCATES toward zero: carity 1 gives 26,
    not 26.265233750364456.
    """
    if contin_arity_ == 0:
        return 1
    return int(_COEF_SAMPLE_COUNT * math.log(contin_arity_ + math.e))


class ITable(list):
    """C++: moses/comboreduct/table/table.h:738 ITable

    `public std::vector<multi_type_seq>`, narrowed to
    `list[tuple[vertex, ...]]` -- see module docstring. Each element is
    one row; `row[i]` is column `i`'s value for that row, in the same
    order as `get_labels()`/`get_types()`.
    """

    def __init__(
        self,
        types: Sequence[type_node] | TypeSignature = (),
        labels: Sequence[str] | int = (),
        min_contin: float = -1.0,
        max_contin: float = 1.0,
    ) -> None:
        """C++: moses/comboreduct/table/table.cc:77 ITable
        and moses/comboreduct/table/table.cc:93 ITable(const type_tree&, int,
        contin_t, contin_t)

        Two C++ constructors merged into one Python `__init__`, dispatched
        on the FIRST argument's type (the same convention `OTable.__init__`
        and `Table.__init__` already use in this file):

        * `Sequence[type_node]` -> `ITable(const type_seq&, const
          string_seq&)` (table.cc:77-78): an initially empty table with
          known column types/labels, rows added afterwards via
          `push_back`.
        * `TypeSignature` -> `ITable(const type_tree& tt, int nsamples,
          contin_t min_contin, contin_t max_contin)` (table.cc:93-127),
          the RANDOM-SAMPLING constructor, in which case `labels` carries
          `nsamples` ("generate an input table according to the signature
          tt ... if negative then the sample size is automatically
          determined", table.h:749-757's own doc comment). Neither
          `types` nor `labels` is set by that C++ constructor, so the
          lazily-defaulted `get_types()`/`get_labels()` (all
          `unknown_type`, `i1..iN`) are what a randomly-sampled table
          reports -- matched here by leaving both empty.

        The default-constructed `ITable()` (table.cc:75) and the
        rows-plus-labels overload (table.cc:80-81, `ITable(const super&
        mat, const string_seq& il)`) are still not separately ported -- no
        caller in this port's scope needs to hand rows to the constructor
        rather than to `push_back` afterwards.

        ## The random constructor draws from the RNG, and the order matters

        This is the port's first RNG-CONSUMING table constructor, so the
        draw ORDER and COUNT are part of its bit-identical-RNG claim, not
        an implementation detail. Per row, in column order, it draws
        `randGen().randint(2)` for a `boolean_type` column and
        `(max_contin - min_contin) * randGen().randdouble() + min_contin`
        for a `contin_type` column -- and nothing at all for
        `unknown_type` (a `vertex()`, i.e. a default-constructed
        `boost::variant`, whose first alternative is `builtin`, hence
        `id::null_vertex`). An `unknown_type` column is reachable: an
        argument index a `-y` combo program never mentions really does
        land there (classic's own `infer_type_tree` gives `or($1 !$3)`
        the signature `->(boolean unknown boolean boolean)`).

        Verified element for element, not set-wise, against a compiled
        probe linking the real `ITable` and the real cogutil
        `MT19937RandGen` (`tests/comboreduct/table/test_table.py`'s
        `_ITABLE_RANDOM_*` oracles record the exact probe invocations and
        their `%.17g` output).

        `enum_type` columns (`enum_t::get_random_enum()`, table.cc:115)
        are NOT ported -- the enum domain is excluded port-wide (this
        module's own docstring) and `enum_t` has no `get_random_enum`
        here. Any other `type_node` hits classic's own `OC_ASSERT(false,
        "Not implemented yet")` (table.cc:118), ported as
        `NotImplementedError`.
        """
        super().__init__()
        self._types: list[type_node] = []
        self._labels: list[str] = []
        if isinstance(types, TypeSignature):
            self._sample(types, -1 if labels == () else int(labels), min_contin, max_contin)
            return
        self._types = list(types)
        self._labels = list(labels)  # type: ignore[arg-type]

    def _sample(
        self, tt: TypeSignature, nsamples: int, min_contin: float, max_contin: float
    ) -> None:
        """C++: moses/comboreduct/table/table.cc:93 ITable

        The body of the random-sampling constructor, split out only so
        `__init__` above stays a dispatcher; see that docstring for the
        contract. Cites the same C++ definition line, which is where this
        code actually comes from.

        `nsamples < 0` -> `std::max(pow2(barity), sample_count(carity))`
        (table.cc:96-97). Both operands are `unsigned int`
        (`pow2(unsigned)` is `1u << x`, numeric.h:266-272;
        `sample_count` returns `unsigned`, table.h:849), and
        `sample_count` is `COEF_SAMPLE_COUNT*log(contin_arity + M_E)` with
        `COEF_SAMPLE_COUNT == 20.0` (table.h:52), truncated to `unsigned`
        on return -- so carity 1 gives `int(20.0 * log(1 + e))` == 26, not
        26.265. UNREACHABLE from either demo problem: `polynomial_problem
        ::run` passes `pms.nsamples > 0 ? pms.nsamples :
        pms.default_nsamples` and `combo_problem::run` forces
        `pms.nsamples = pms.default_nsamples` when it is `<= 0`, and
        `default_nsamples` is a `const unsigned int` 20 -- so every real
        caller passes a positive count. Ported and directly unit-tested
        against the probe anyway, because `Table(const combo_tree&, int
        nsamples=-1, ...)` (table.h:936) would reach it.

        `comp_tt` (table.cc:100) selects ENUMERATED booleans
        (`i & (1 << bidx++)`, low bit first, `bidx` restarting at 0 each
        row) over random ones whenever `nsamples == (int)pow2(barity)`.
        Reachable and OBSERVABLE: `-Hcp -y'+(6 *(6 impulse($1)))'` has
        `barity == 1`, and `--nsamples 2` flips it on -- measured in
        classic, which scores `-5` with it and `-51` without. For a
        signature with NO boolean columns (every `-Hsr` run, whose
        signature is `->(contin contin)`) `comp_tt` can still be TRUE --
        `pow2(0)` is 1, so `--nsamples 1` sets it -- but it is then inert,
        because the flag is only ever read inside the `boolean_type`
        branch. Measured, not assumed: `moses -Hsr -k2 -m200 -W1
        --nsamples 1` runs to completion in classic.
        """
        barity = boolean_arity(tt)
        carity = contin_arity(tt)

        if nsamples < 0:
            nsamples = max(_pow2(barity), _sample_count(carity))

        # in that case the boolean inputs are not picked randomly but
        # instead are enumerated
        comp_tt = nsamples == _pow2(barity)

        rng = randGen()
        for i in range(nsamples):
            bidx = 0  # counter used to enumerate all booleans
            vs: list[vertex] = []
            for t in tt.itypes:
                if t == type_node.boolean_type:
                    if comp_tt:
                        b = bool(i & (1 << bidx))
                        bidx += 1
                    else:
                        b = bool(rng.randint_n(2))
                    vs.append(bool_to_vertex(b))
                elif t == type_node.contin_type:
                    vs.append((max_contin - min_contin) * rng.randdouble() + min_contin)
                elif t == type_node.enum_type:
                    raise NotImplementedError(
                        "ITable(type_tree, ...): enum_type columns need "
                        "enum_t::get_random_enum() (table.cc:115); the enum domain "
                        "is excluded port-wide"
                    )
                elif t == type_node.unknown_type:
                    # `vs.push_back(vertex()); // push default vertex`
                    # (table.cc:116-117). `vertex` is `boost::variant<
                    # contin_t, enum_t, builtin, ...>` (vertex.h:128-141)
                    # and a default-constructed variant holds its FIRST
                    # alternative -- so this is a CONTIN 0.0, not
                    # `id::null_vertex`. Measured against the probe, which
                    # reports `is_contin(v)` true for that cell; guessing
                    # `builtin.null_vertex` (the natural reading of
                    # "default vertex") gave a table that differed from
                    # classic in exactly this column.
                    vs.append(0.0)
                else:
                    raise NotImplementedError(
                        f"ITable(type_tree, ...): column type {t!r} -- "
                        'OC_ASSERT(false, "Not implemented yet") (table.cc:118)'
                    )
            # input vector
            self.push_back(vs)

    def push_back(self, row: Sequence[vertex]) -> None:
        """C++: moses/comboreduct/table/table.h:738 ITable

        `push_back` is `std::vector<multi_type_seq>::push_back`, inherited
        (not declared in table.h itself) -- cited at the enclosing class
        per the project's convention for inherited members (see
        argument.py's `__lt__`/`__eq__`). Coerces to `tuple` so rows stay
        hashable (`Table.compressed()` uses a row directly as a `CTable`
        key).
        """
        self.append(tuple(row))

    def get_arity(self) -> int:
        """C++: moses/comboreduct/table/table.h:764 get_arity

        `super::front().size()` -- like C++, requires at least one row
        (raises `IndexError` on an empty table, analogous to C++'s UB on
        an empty `std::vector::front()`).
        """
        return len(self[0])

    def __eq__(self, other: object) -> bool:
        """C++: moses/comboreduct/table/table.cc:131 ITable

        Cites the enclosing class per the project's convention for
        non-identifier symbols (see argument.py's `__lt__`); the real
        member is `ITable::operator==` (table.cc:131-137).
        """
        if not isinstance(other, ITable):
            return NotImplemented
        return (
            list(self) == list(other)
            and self.get_labels() == other.get_labels()
            and self.get_types() == other.get_types()
        )

    __hash__ = None  # type: ignore[assignment]

    # -- labels ---------------------------------------------------------

    def get_default_labels(self) -> list[str]:
        """C++: moses/comboreduct/table/table.cc:149 ITable::get_default_labels"""
        return [f"{_DEFAULT_INPUT_LABEL}{i}" for i in range(1, self.get_arity() + 1)]

    def set_labels(self, labels: Sequence[str]) -> None:
        """C++: moses/comboreduct/table/table.cc:142 ITable::set_labels"""
        self._labels = list(labels)

    def get_labels(self) -> list[str]:
        """C++: moses/comboreduct/table/table.cc:159 ITable::get_labels"""
        if not self._labels and len(self) > 0:
            self._labels = self.get_default_labels()
        return self._labels

    # -- types ------------------------------------------------------------

    def set_types(self, types: Sequence[type_node]) -> None:
        """C++: moses/comboreduct/table/table.cc:166 ITable::set_types"""
        self._types = list(types)

    def get_types(self) -> list[type_node]:
        """C++: moses/comboreduct/table/table.cc:171 ITable::get_types"""
        if not self._types and len(self) > 0:
            self._types = [type_node.unknown_type] * self.get_arity()
        return self._types

    def get_type(self, name: str) -> type_node:
        """C++: moses/comboreduct/table/table.cc:183 ITable::get_type"""
        if not self._types:
            return type_node.unknown_type
        off = self.get_column_offset(name)
        if off == -1:
            return type_node.unknown_type
        return self._types[off]

    # -- column access ----------------------------------------------------

    def get_column_offset(self, name: str) -> int:
        """C++: moses/comboreduct/table/table.cc:234 ITable::get_column_offset

        Three cases, checked in this exact order: empty name -> column 0;
        a name whose first character is a digit -> a 1-based column
        *position* (`"2"` means the second column, offset 1); otherwise a
        column *name*, looked up by exact match against the raw `labels`
        field (not `get_labels()` -- deliberately bypasses the
        lazily-generated default labels, exactly as C++ does by reading
        the protected `labels` member directly).

        This is the ONLY place in the whole `loadTable` pipeline where a
        numeric string is ever treated as a column position rather than a
        name -- see table_io.py's module docstring for why that makes
        "target by position" a sparse-table-only feature.
        """
        if name == "":
            return 0
        if name[0].isdigit():
            return int(name) - 1
        try:
            return self._labels.index(name)
        except ValueError:
            return -1

    def get_column_data(self, key: str | int) -> list[vertex]:
        """C++: moses/comboreduct/table/table.cc:252 ITable::get_column_data(int)
        and moses/comboreduct/table/table.cc:269 ITable::get_column_data(const string&)

        One method covering both overloads, dispatching on the Python
        type of `key` the way the two C++ overloads dispatch on static
        type.
        """
        offset = self.get_column_offset(key) if isinstance(key, str) else key
        if offset == -1:
            return []
        return [row[offset] for row in self]

    def delete_column(self, name: str) -> str:
        """C++: moses/comboreduct/table/table.cc:274 ITable::delete_column"""
        off = self.get_column_offset(name)
        if off == -1:
            raise IndexError(f"Can't delete, unknown column name: {name}")
        for i, row in enumerate(self):
            self[i] = row[:off] + row[off + 1 :]
        rv = ""
        if self._labels:
            rv = self._labels[off]
            del self._labels[off]
        if self._types:
            del self._types[off]
        return rv

    def delete_columns(self, ignore_features: Sequence[str]) -> None:
        """C++: moses/comboreduct/table/table.cc:299 ITable::delete_columns"""
        for feat in ignore_features:
            self.delete_column(feat)

    def insert_col(self, clab: str, col: Sequence[vertex], off: int = -1) -> None:
        """C++: moses/comboreduct/table/table.cc:197 ITable::insert_col

        `off < 0` means "append after the last column" (matches C++'s
        "If off is negative, then insert after the last column").

        FIX WAVE (final review, Important): both `OC_ASSERT`s below
        (table.cc:213,219) are active in release C++ builds; ported as
        explicit raises rather than bare Python `assert`s, which
        `python -O` strips -- see argument.py's `__init__` for the same
        fix against the same class of bug.
        """
        idx = 1 if len(col) > 1 else 0
        col_type = _vertex_type_node(col[idx])
        if off >= 0:
            self._types.insert(off, col_type)
            self._labels.insert(off, clab)
        else:
            self._types.append(col_type)
            self._labels.append(clab)

        if len(self) == 0:
            if off >= 0:
                raise AssertionError("off must be negative when the table is empty")
            for v in col:
                self.push_back((v,))
            return

        if len(col) != len(self):
            raise AssertionError("Incorrect column length!")
        for i in range(len(col)):
            row = list(self[i])
            row.insert(off if off >= 0 else len(row), col[i])
            self[i] = tuple(row)

    def filtered(self, filter_indices: Iterable[int]) -> ITable:
        """C++: moses/comboreduct/table/table.h:807 ITable::filtered

        `filter_indices` is a container of ITable column indices to
        *keep*, in the ORDER GIVEN (mirroring `seq_filtered`,
        opencog/util/algorithm.h:352, which does `for idx in indices:
        res.push_back(seq[idx])` -- output order follows the filter's own
        iteration order, not necessarily ascending).
        """
        idxs = list(filter_indices)
        res = ITable()
        res.set_labels([self.get_labels()[i] for i in idxs])
        res.set_types([self.get_types()[i] for i in idxs])
        for row in self:
            res.push_back(tuple(row[i] for i in idxs))
        return res

    def copy(self) -> ITable:
        """Not a ported symbol -- stands in for C++'s implicit copy
        constructor (`ITable trimmed(itable);`, table.cc:525), used by
        `Table.compressed()`'s weighted branch.
        """
        new = ITable(types=self._types, labels=self._labels)
        new.extend(self)
        return new


# ---------------------------------------------------------------------------
# OTable -- table.h:866
# ---------------------------------------------------------------------------

_DEFAULT_OUTPUT_LABEL = "output"


class OTable(list):
    """C++: moses/comboreduct/table/table.h:866 OTable

    `public vertex_seq` (i.e. `public std::vector<vertex>`), narrowed to
    `list[vertex]` -- a single output column. Three of the five C++
    constructors are ported: label-only, values-and-label
    (table.cc:309-312) and the ITable-evaluating one (table.cc:319).
    `OTable(const combo_tree&, const CTable&, ...)` (table.cc:350) has no
    caller in this port and is not ported.
    """

    def __init__(
        self,
        values: Iterable[vertex] | combo_tree = (),
        label: str | ITable = _DEFAULT_OUTPUT_LABEL,
        ol: str = _DEFAULT_OUTPUT_LABEL,
    ) -> None:
        """C++: moses/comboreduct/table/table.cc:309 OTable

        Also covers `OTable(const super&, const string&)` (table.cc:312)
        and `OTable(const combo_tree&, const ITable&, const string&)`
        (table.cc:319) -- the breadcrumb cites the first overload's
        definition line, and the other two are named here in the body,
        per this project's convention for symbols the matcher cannot
        parse (docs/CARRY-FORWARD.md 4.5). The trailing comma this line
        used to carry made the symbol `OTable,`, which check_alignment.py
        could not match.

        One constructor covers three C++ overloads, dispatched on the
        FIRST argument's type (the same convention `ITable.__init__` and
        `Table.__init__` use):

        * a `combo_tree` -> `OTable(const combo_tree& tr, const ITable&
          itable, const string& ol)` (table.cc:319-347): "Construct the
          OTable by evaluating the combo tree @tr for each row in the
          input ITable" (table.h:876-877's own comment). `label` then
          carries the `ITable`, and `ol` the output label.
        * anything else -> the label-only overload (table.cc:309-310,
          `type = id::unknown_type`) and the values-plus-label overload
          (table.cc:312-318), which infers the column type from the FIRST
          value: `type = get_type_node(get_type_tree((*this)[0]))`.
          `values` being empty is exactly what distinguishes those two --
          `(*this)[0]` is C++ UB on an empty vector, which the real
          overload set avoids simply by that overload never being called
          with no values; the `len(self) > 0` guard here is the runtime
          equivalent, needed only because one Python constructor stands
          in for both.

        The `combo_tree` overload's `is_ann_type(*tr.begin())` arm
        (table.cc:323-338 -- `tree_transform().decodify_tree`,
        `net.feedforward_depth()`, `net.propagate()`) is not ported, and
        it is unreachable BY CONSTRUCTION rather than merely omitted:
        `is_ann_type` is `boost::get<ann_type>(&v)` (vertex.cc:143-146),
        i.e. "does this variant currently hold its `ann_type`
        alternative", and this port's `vertex` union has no such
        alternative at all (`vertex.py`'s own module docstring:
        `builtin | argument | contin_t | str`). There is therefore no
        value that could make the test true, and no predicate to write --
        which is why the code below has no branch where C++ has one. The
        entry point that would build such a tree, `-H ann-cp`
        (`demo_problems.py`'s `ann_combo_problem`), raises before reaching
        here.

        `OC_ASSERT(not tr.empty())` (table.cc:322) is ported as the
        `IndexError` `tr.begin()` already raises on an empty tree, not as
        a separate check.
        """
        if isinstance(values, combo_tree):
            if not isinstance(label, ITable):
                raise TypeError("OTable(combo_tree, itable, ol): second argument must be an ITable")
            self._from_tree(values, label, ol)
            return
        super().__init__(values)
        self._label = label
        self._type: type_node = (
            _vertex_type_node(self[0]) if len(self) > 0 else type_node.unknown_type
        )

    def _from_tree(self, tr: combo_tree, itable: ITable, ol: str) -> None:
        """C++: moses/comboreduct/table/table.cc:319 OTable

        The body of the ITable-evaluating constructor, split out only so
        `__init__` above stays a dispatcher; see that docstring for the
        contract and for what the ANN arm does instead of being ported.

        Note the trailing `type = get_type_node(get_type_tree((*this)[0]))`
        (table.cc:346): unlike the TEMPLATED `OTable(const Func&, const
        ITable&, ...)` overload (table.h:886-893), which never assigns
        `type` at all and so leaves it indeterminate, this one really does
        set it from the first evaluated row.
        """
        super().__init__()
        self._label = ol
        tr.begin()  # OC_ASSERT(not tr.empty()) -- raises IndexError if it is
        iv = interpreter_visitor(tr)
        for row in itable:
            self.push_back(iv(list(row)))
        self._type = _vertex_type_node(self[0])

    def push_back(self, v: vertex) -> None:
        """C++: moses/comboreduct/table/table.h:866 OTable

        Inherited `std::vector<vertex>::push_back` -- see ITable.push_back's
        docstring for why this alias exists.
        """
        self.append(v)

    def copy(self) -> OTable:
        """Not a ported symbol -- stands in for C++'s implicit copy
        constructor, needed because `OTable` (like `ITable`) is a mutable
        Python object: a bare `res.otable = self.otable` would alias the
        same list rather than copy it the way `res.otable = otable;`
        (table.h:987, `Table::filtered`) copies a C++ value type. Used by
        `Table.filtered()`.
        """
        new = OTable(self, self._label)
        new._type = self._type
        return new

    def set_label(self, label: str) -> None:
        """C++: moses/comboreduct/table/table.cc:364 OTable::set_label"""
        self._label = label

    def get_label(self) -> str:
        """C++: moses/comboreduct/table/table.cc:372 OTable::get_label"""
        return self._label

    def set_type(self, t: type_node) -> None:
        """C++: moses/comboreduct/table/table.cc:377 OTable::set_type"""
        self._type = t

    def get_type(self) -> type_node:
        """C++: moses/comboreduct/table/table.cc:382 OTable::get_type"""
        return self._type

    def __eq__(self, other: object) -> bool:
        """C++: moses/comboreduct/table/table.cc:389 OTable

        Cites the enclosing class (see ITable.__eq__ above); the real
        member is `OTable::operator==` (table.cc:389-395), which ends
        with `return rhs.get_label() == label;` (table.cc:396) -- easy to
        miss since it comes after the element-comparison loop, but without
        it a mislabeled output column compares equal to a correctly
        labeled one. Included here.

        NOT ported: C++'s epsilon-tolerant contin comparison
        (`is_approx_eq(get_contin(*lit), get_contin(*rit), 1e-12)`,
        table.cc:392-393) -- every element, contin included, is compared
        with plain `==` here. This errs safe (a false inequality where
        C++ would call two near-identical contin values equal, never the
        reverse turning a real difference into a false positive), left as
        a disclosed gap rather than fixed in this round.
        """
        if not isinstance(other, OTable):
            return NotImplemented
        return list(self) == list(other) and self.get_label() == other.get_label()

    __hash__ = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# TTable / TimedValue / TimedCounter / CTableTime -- table.h:454 .. :518
#
# The timestamp layer. Read `TimedValue.__lt__` and `TimedCounter`'s class
# docstring before changing anything here: classic's own comparator is not
# a strict weak ordering, so classic's container is not a faithful target.
# ---------------------------------------------------------------------------

# C++: moses/comboreduct/table/table.h:449 default_timestamp_label
# `static const std::string default_timestamp_label("timestamp");`
_DEFAULT_TIMESTAMP_LABEL = "timestamp"

# `TTable::value_type` is `boost::gregorian::date` (table.h:458). A
# default-constructed one is the special "not-a-date-time" value, which is
# what `Table::compressed` gives every observation of an UNtimestamped
# table (table.cc:513/:538, `TimedValue(*out_it)`); `None` stands for it
# here. Every real date is a `datetime.date`.
timestamp_t = "date | None"

_MONTH_NAMES = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)

# boost::gregorian's own year bounds -- measured, not assumed:
# `TTable::from_string("1399-12-31")` throws `bad_year` with the message
# "Year is out of valid range: 1400..9999", and so does "10000-01-01".
_MIN_YEAR, _MAX_YEAR = 1400, 9999

_DATE_TOKEN_SPLIT = re.compile(r"[^0-9A-Za-z]+")


def timestamp_less(a, b) -> bool:
    """No direct C++ symbol: `boost::gregorian::date::operator<`, over the
    two-valued domain this port represents (`datetime.date`, or `None` for
    boost's default-constructed not-a-date-time).

    MEASURED (`tools/cpp_probe/timed_counter_probe.cc`, `ordering` mode),
    because a sentinel's sort position is exactly the kind of thing that
    gets guessed wrong: the not-a-date value compares GREATER than every
    real date (`real < default` is true, `default < real` is false), and
    equal to itself. boost stores it as the maximum of the internal day
    count, so it sorts LAST.

    Only `CTableTime` (`ordered_by_time`) and `get_timestamps` actually
    order timestamps, and both drop or never see the sentinel -- but
    `TimedValue.__lt__` calls this on mixed input, so the sentinel arm is
    live.
    """
    if a is None:
        return False  # not-a-date is the maximum: never less than anything
    if b is None:
        return True  # ... and every real date is less than it
    return a < b


class TTable(list):
    """C++: moses/comboreduct/table/table.h:454 TTable

    "Table containing timestamps. There is only one column: a single
    timestamp for each row." (table.h:451-453's own comment.)
    `struct TTable : public std::vector<boost::gregorian::date>` -- a
    `list` here, one entry per row of the `ITable`/`OTable` it parallels,
    plus a label.
    """

    def __init__(self, tt: Iterable = (), label: str = _DEFAULT_TIMESTAMP_LABEL) -> None:
        """C++: moses/comboreduct/table/table.cc:454 TTable
        and moses/comboreduct/table/table.cc:457 TTable(const super&, const string&)

        One Python signature for both C++ constructors: `TTable(label)` is
        `TTable(tt=(), label=label)` here.
        """
        super().__init__(tt)
        self.label = label

    def set_label(self, label: str) -> None:
        """C++: moses/comboreduct/table/table.cc:460 TTable::set_label"""
        self.label = label

    def get_label(self) -> str:
        """C++: moses/comboreduct/table/table.cc:465 TTable::get_label"""
        return self.label

    @staticmethod
    def from_string(timestamp_str: str):
        """C++: moses/comboreduct/table/table.cc:470 TTable::from_string

        `return boost::gregorian::from_string(timestamp_str);` -- the
        YEAR-MONTH-DAY parser `istreamDenseTable_noHeader` (table_io.cc:1144)
        runs on every raw token of a `--timestamp-feature` column.

        The accepted grammar is MEASURED against boost on this machine, not
        reconstructed from its documentation (probe: a throwaway
        `TTable::from_string` driver over 35 candidate strings; the results
        are pinned in `tests/fixtures/table_timestamps.json`'s
        `from_string` group):

        * Fields are split on any run of non-alphanumeric characters, so
          `2012-01-11`, `2012/01/11`, `2012.01.11`, `2012 01 11` and
          `2012--01--11` all parse; leading/trailing whitespace is ignored.
        * Exactly the FIRST THREE fields are used and the rest ignored:
          `2012-01-11-05` parses as 2012-01-11.
        * Month may be a number `1..12` or an English month name, either in
          full (`september`) or as its FIRST THREE LETTERS (`sep`), case
          insensitive. `sept` and `janu` are REJECTED ("Month number is out
          of range 1..12") -- boost accepts only the exact full name or the
          exact 3-letter abbreviation, nothing between.
        * Leading zeros are fine anywhere (`02012-Jan-011` parses).
        * A run-together `20120111` is REJECTED (only three fields, split on
          delimiters; a single field is not a date).
        * Year must be `1400..9999`; day must be valid for the month and
          year (`2012-Feb-29` parses, `2011-Feb-29` does not).

        # DIVERGENCE(C++ moses/comboreduct/table/table.cc:470): ERROR TYPE
        # ONLY. boost throws `boost::bad_lexical_cast`,
        # `boost::gregorian::bad_year`, `bad_month` or `bad_day_of_month`
        # depending on which field failed; this raises `ValueError`
        # carrying boost's own message text, so the WHICH is preserved and
        # only the Python exception class differs. Classic never catches
        # any of them -- an unparseable timestamp column terminates the
        # process -- so the only observable difference is the traceback.
        """
        text = timestamp_str.strip()
        fields = [f for f in _DATE_TOKEN_SPLIT.split(text) if f]
        if len(fields) < 3:
            # boost reaches its day parse with an empty field and reports
            # the day, not the field count -- reproduced verbatim.
            raise ValueError("Day of month value is out of range 1..31")

        year_s, month_s, day_s = fields[0], fields[1], fields[2]

        if not year_s.isdigit():
            raise ValueError(
                "bad lexical cast: source type value could not be interpreted as target"
            )
        year = int(year_s)

        if month_s.isdigit():
            month = int(month_s)
        else:
            lowered = month_s.lower()
            month = 0
            for i, name in enumerate(_MONTH_NAMES, start=1):
                if lowered == name or lowered == name[:3]:
                    month = i
                    break
            if month == 0:
                if lowered.isalpha():
                    # boost's month-name lookup fails, then its numeric
                    # fallback reports the range error -- measured on
                    # "2012-Sept-11" and "2012-Janu-11".
                    raise ValueError("Month number is out of range 1..12")
                raise ValueError(
                    "bad lexical cast: source type value could not be interpreted as target"
                )
        if not (1 <= month <= 12):
            raise ValueError("Month number is out of range 1..12")

        if not day_s.isdigit():
            raise ValueError(
                "bad lexical cast: source type value could not be interpreted as target"
            )
        day = int(day_s)
        if not (1 <= day <= 31):
            raise ValueError("Day of month value is out of range 1..31")

        if not (_MIN_YEAR <= year <= _MAX_YEAR):
            raise ValueError(f"Year is out of valid range: {_MIN_YEAR}..{_MAX_YEAR}")

        try:
            return date(year, month, day)
        except ValueError as exc:  # day out of range for this month/year
            raise ValueError("Day of month is not valid for year") from exc

    @staticmethod
    def to_string(timestamp) -> str:
        """C++: moses/comboreduct/table/table.cc:474 TTable::to_string

        `boost::gregorian::to_iso_extended_string`, i.e. `YYYY-MM-DD`.
        The not-a-date sentinel prints as `not-a-date-time`, which is what
        boost's own `to_iso_extended_string` produces for it.
        """
        if timestamp is None:
            return "not-a-date-time"
        return timestamp.isoformat()


class TimedValue:
    """C++: moses/comboreduct/table/table.h:472 TimedValue

    A `(vertex value, TTable::value_type timestamp)` pair -- the KEY type
    of every `CTable` cell (`TimedCounter`, table.h:496). `__slots__`
    because a real table builds one per observation.
    """

    __slots__ = ("timestamp", "value")

    def __init__(self, value: vertex, timestamp=None) -> None:
        """C++: moses/comboreduct/table/table.h:476 TimedValue

        `TimedValue(const vertex v, const TTable::value_type t =
        TTable::value_type())` -- the default second argument is the
        not-a-date sentinel, `None` here.
        """
        self.value = value
        self.timestamp = timestamp

    def __lt__(self, other: TimedValue) -> bool:
        """C++: moses/comboreduct/table/table.h:472 TimedValue

        Cites the enclosing struct: the real member is
        `TimedValue::operator<` (table.h:482-484).

        # DIVERGENCE(C++ moses/comboreduct/table/table.h:482): DO NOT FIX --
        # this operator is transliterated verbatim, INCLUDING its bug, and
        # is deliberately NOT used to order any container in this port. C++
        # reads
        #
        #     return (value < r.value) || (timestamp < r.timestamp);
        #
        # which is missing the `value == r.value &&` guard that would make
        # it lexicographic. It is therefore not a strict weak ordering:
        # for a = (logical_true, 2012-01-11) and b = (logical_false,
        # 2012-01-03), `a < b` AND `b < a` are both true while `a == b` is
        # false (`logical_true` is enumerator 1 and `logical_false` is 2,
        # vertex.h:74, so the value order and the date order disagree).
        # `std::map` requires asymmetry, so classic's `Counter<TimedValue,
        # count_t>` is undefined behaviour whenever a cell holds such a
        # pair. See `TimedCounter`'s class docstring for what this port
        # does instead, and for the measured consequences.
        """
        return vertex_less(self.value, other.value) or timestamp_less(
            self.timestamp, other.timestamp
        )

    def __eq__(self, other: object) -> bool:
        """C++: moses/comboreduct/table/table.h:472 TimedValue

        Cites the enclosing struct: the real member is
        `TimedValue::operator==` (table.h:486-488), `(value == r.value) &&
        (timestamp == r.timestamp)`.

        `vertex_equal`, not a bare `==`, on the value half -- `contin_t` is
        a `float` and `builtin` an `IntEnum`, so `1.0 == logical_true` is
        `True` in bare Python where `boost::variant`'s own `operator==`
        compares `which()` first and can never be (vertex.py's own
        docstring; docs/CARRY-FORWARD.md Sec 2.2/2.5).

        Note this IS the equivalence relation `std::map` derives from the
        broken `operator<`: `!(a<b) && !(b<a)` expands to `!(av<bv) &&
        !(bv<av) && !(at<bt) && !(bt<at)`, i.e. exactly member-wise
        equality. So the SET of keys a correctly-ordered map would hold is
        well defined even though the container classic actually builds is
        not -- which is what makes this port's choice a definite one rather
        than an arbitrary pick.
        """
        if not isinstance(other, TimedValue):
            return NotImplemented
        return vertex_equal(self.value, other.value) and self.timestamp == other.timestamp

    def __hash__(self) -> int:
        """No C++ counterpart (`TimedValue` is a `std::map` key, never
        hashed). Hashes the vertex's TYPE alongside its value, so the
        `contin_t`/`builtin` ordinal collision `__eq__` guards against
        cannot sneak back in through the hash: `hash(1.0) ==
        hash(builtin.logical_true)` in bare Python, and two objects that
        compare unequal but hash equal would still collide into one dict
        bucket -- harmless for correctness, but it would make a
        `TimedCounter` silently O(n) and, worse, invites a future `==`
        fix to be judged safe by tests that never saw a collision.
        """
        return hash((vertex_rank(self.value), self.value, self.timestamp))

    def __repr__(self) -> str:
        return f"TimedValue({self.value!r}, {TTable.to_string(self.timestamp)})"


class TimedCounter(dict):
    """C++: moses/comboreduct/table/table.h:496 TimedCounter

    `struct TimedCounter : public Counter<TimedValue, count_t>`, i.e.
    `std::map<TimedValue, double, std::less<TimedValue>>` -- one CTable
    cell: how many times each `(output value, timestamp)` observation
    occurred, `count_t` being a `double` so weighted rows work.

    ## The port keeps two views, and which one is the mapping matters

    This object IS a `dict[vertex, count_t]` -- classic's own
    `untimedCounter()` (table.cc:616), the sum of each vertex's counts
    across all timestamps -- and carries the full `TimedValue`-keyed
    observation list alongside it, reachable through `timed_items()` /
    `timed_keys()` / `timed_len()`.

    # DIVERGENCE(C++ moses/comboreduct/table/table.h:496): the mapping
    # interface (`__iter__`, `items()`, `keys()`, `len()`) exposes the
    # UNTIMED view where C++'s iteration yields `TimedValue` keys.
    # Deliberate, for two reasons. (1) Fifteen modules outside this file
    # read a CTable cell, and every one of them either looks a vertex up
    # (`counter.get(v)`, which is precisely `TimedCounter::get(const
    # vertex&)`, table.cc:608, "the sum of that counter across all
    # timestamps") or sums the counts -- both of which the untimed view
    # answers exactly. (2) Classic's own consumers that DO iterate all
    # write `tcv.first.value` and then group by it, so the untimed view is
    # not an approximation of what they compute: it is what they compute.
    # `timed_items()` is there for the two places that genuinely need
    # per-observation timestamps (`precision_bscore`'s `time_bscore` /
    # dispersion branches, `ss_bscore`).

    ## What this port does NOT reproduce: classic's container corruption

    # DIVERGENCE(C++ moses/comboreduct/table/table.h:482): this port's
    # observation list is an insertion-ordered dict keyed on `TimedValue`
    # equality. Classic's is a `std::map` under `TimedValue::operator<`,
    # which is not a strict weak ordering, so classic's container is
    # undefined behaviour -- and MEASURED, not merely theoretically,
    # broken. `tools/cpp_probe/timed_counter_probe.cc` replays every
    # insertion sequence of length 1..4 over a pool of six `TimedValue`s
    # (two boolean outputs x three dates) into a real `TimedCounter`:
    #
    #     1554 sequences
    #      533 iterate in a different ORDER than a lexicographic map
    #      190 LOSE observations: `map::size()` says 3 while iteration
    #          visits 2, so the count of the unreachable node is silently
    #          dropped from `total_count()`, `uncompressed_size()`,
    #          `untimedCounter()` and every bscorer
    #       30 hold the SAME key TWICE, splitting its count in two
    #
    # End to end on a 9-row table (`tests/fixtures/table_data/
    # timestamp_bool_dup.csv`): classic's `--timestamp-feature=when` builds
    # a CTable whose `uncompressed_size()` is 8. It is worse than
    # non-deterministic-looking: WHICH observation vanishes changes when
    # the CTable is COPIED, and `table-problems.cc` copies it twice before
    # scoring, so the table classic actually scores (5 true / 3 false)
    # differs from the one `Table::compressed` returned (6 true / 2 false).
    #
    # A port cannot faithfully reproduce that: it would mean emulating
    # libstdc++'s red-black tree, its rebalancing, AND its copy
    # constructor, under a comparator the standard says makes the result
    # undefined. This port therefore computes the INTENDED result -- every
    # distinct `(value, timestamp)` observation retained with its full
    # count, which is exactly the key set C++'s own equivalence relation
    # (`!(a<b) && !(b<a)`, member-wise equality) defines.
    #
    # What a user sees, concretely: on a `--timestamp-feature` table where
    # some compressed row holds observations whose value order and date
    # order disagree, classic drops or duplicates observations and this
    # port does not. Scores computed from that table differ accordingly
    # (measured: classic `-Hpre` reports 0.625 on timestamp_bool_dup.csv
    # where all nine rows would give 6/9). On a table with no such row --
    # which includes EVERY table with no timestamp column at all, since
    # every observation then carries the same sentinel date -- the two
    # agree exactly, and `tests/fixtures/table_data/
    # timestamp_bool_clean.csv` is the corpus member that proves it.
    """

    __slots__ = ("_timed",)

    def __init__(self) -> None:
        """C++: moses/comboreduct/table/table.h:496 TimedCounter

        Cites the enclosing struct: `Counter`'s default constructor
        (opencog/util/Counter.h:72) is inherited, not declared here.
        """
        super().__init__()
        self._timed: dict[TimedValue, float] = {}

    # -- construction --------------------------------------------------

    def add(self, value: vertex, timestamp, count: float) -> None:
        """No direct C++ symbol: `res[*in_it][TimedValue(*out_it,
        *time_it)] += weight;` (table.cc:548) -- `std::map::operator[]`
        followed by `+=`, i.e. insert-or-accumulate. Named rather than
        spelled as `__setitem__` because it takes the timestamp too.
        """
        key = TimedValue(value, timestamp)
        self._timed[key] = self._timed.get(key, 0.0) + count
        dict.__setitem__(self, value, dict.get(self, value, 0.0) + count)

    # -- the timed view ------------------------------------------------

    def timed_items(self):
        """No direct C++ symbol: C++'s own `begin()`/`end()` over
        `Counter<TimedValue, count_t>` -- named explicitly here because
        this class's OWN iteration is the untimed view (see the class
        docstring's first DIVERGENCE).
        """
        return self._timed.items()

    def timed_keys(self):
        """No direct C++ symbol: see `timed_items`."""
        return self._timed.keys()

    def timed_len(self) -> int:
        """No direct C++ symbol: `Counter::size()`, i.e. the number of
        distinct `(value, timestamp)` OBSERVATIONS -- what
        `bscore_ctable_time_dispersion`'s constructor loop
        (time_dispersion.cc:48-50) iterates. `len(self)` is the number of
        distinct VALUES instead; the two coincide only when every
        observation shares one timestamp.
        """
        return len(self._timed)

    # -- C++'s own members ---------------------------------------------

    def get(self, v: vertex, default: float = 0.0) -> float:
        """C++: moses/comboreduct/table/table.cc:608 TimedCounter::get

        "Overload get(const TimedValue&) to work with a vertex v, in that
        case it returns the sum of that counter across all timestamps with
        vertex equal to v." (table.h:497-499's own comment.) Declaring
        `get` in the derived struct HIDES `Counter::get(const T&, CT)`
        entirely, so in classic there is no way to look a cell up by
        `TimedValue` through `.get()` at all -- only by vertex.

        C++ has no default parameter here and returns 0 for an absent
        vertex; `default` exists so the dozen call sites written against a
        plain `Counter[vertex]` (`counter.get(builtin.logical_true, 0.0)`)
        keep working verbatim. Every one of them passes `0.0`, which is
        exactly C++'s behaviour.
        """
        return dict.get(self, v, default)

    def untimedCounter(self) -> dict:
        """C++: moses/comboreduct/table/table.cc:616 TimedCounter::untimedCounter

        "Return a counter without timestamps" (table.h:502's own comment)
        -- `vc[vtc.first.value] += vtc.second` over every observation.
        Returns a COPY, matching C++'s by-value `Counter<vertex, count_t>`
        return: this object's own mapping half already IS that counter (see
        the class docstring), so mutating the result must not touch it.
        """
        return dict(self)

    def total_count(self) -> float:
        """C++: opencog/util/Counter.h:104 Counter

        Cites the enclosing class: the real member is
        `Counter::total_count()` (Counter.h:104-107),
        `boost::accumulate(*this | map_values, (CT)0)`. Summing the untimed
        view gives the same total as summing the observations, since the
        untimed view is exactly their per-vertex sum.
        """
        return sum(self.values())

    def mode(self) -> vertex:
        """C++: moses/comboreduct/table/table.cc:623 TimedCounter::mode

        `return untimedCounter().mode();`, and `Counter::mode()`
        (Counter.h:110-118) is

            T key = begin()->first; CT cnt = begin()->second;
            for (const auto& v : *this) if (cnt < v.second) key = v.first;
            return key;

        DO NOT FIX: `cnt` is never reassigned inside the loop, so this is
        NOT an argmax. It returns the LAST key whose count exceeds the
        FIRST entry's count, or the first key if none does. Reproduced
        verbatim.

        The iteration order this walks is `std::map`'s -- ascending key --
        so the port sorts the untimed view by `vertex_less` first rather
        than relying on insertion order, which would make "the first entry"
        a different entry. (`feature_selector._timed_counter_mode` is a
        pre-existing standalone transliteration of the same C++ method,
        landed before this class existed; it takes the same approach.)
        """
        items = sorted(
            self.items(),
            key=cmp_to_key(lambda a, b: -1 if vertex_less(a[0], b[0]) else 1),
        )
        if not items:
            raise IndexError(
                "TimedCounter.mode: empty counter -- C++'s `super::begin()->first` "
                "dereferences end() (Counter.h:112), which is undefined behaviour"
            )
        key, cnt = items[0]
        for v, c in items:
            if cnt < c:
                key = v
        return key

    # -- timestamp-blind writes ----------------------------------------

    # # DIVERGENCE(C++ moses/comboreduct/table/table.h:496): a plain
    # `counter[v] = c` cannot say WHICH observation it means, so it is
    # defined here to mean "replace every observation of `v` with a single
    # untimestamped one of count `c`". C++ has no such operation: its cells
    # are keyed by `TimedValue` throughout, and code that rebuilds a cell
    # writes `dst[tcv.first] += tcv.second`, keeping the timestamp.
    #
    # Three places in this port rebuild a cell vertex by vertex, written
    # before this class existed and outside this file's ownership:
    # `feature_selector.build_fs_ctable` (feature_selector.cc:246 keeps the
    # `TimedValue`), `feature_selection/ctable_subsample.ctable_remove_rows`
    # and `scoring_base._copy_ctable` (which goes further and rebuilds the
    # cell as a plain `collections.Counter`). Through any of them a table's
    # timestamps are LOST -- `get_timestamps()` on the result returns the
    # empty set. That is the port's behaviour before this slice, so it is a
    # non-regression rather than a new gap, and it is why `--time-bscore`
    # and `--time-dispersion-pressure` (which read `_wrk_ctable`) stay
    # gated in `problem_params.py`. Raising here instead was tried and
    # rejected: it breaks those three callers outright, two of them in
    # trees this task does not own.

    def __setitem__(self, key: vertex, value: float) -> None:
        self._timed = {tv: c for tv, c in self._timed.items() if not vertex_equal(tv.value, key)}
        self._timed[TimedValue(key, None)] = value
        dict.__setitem__(self, key, value)

    def __delitem__(self, key: vertex) -> None:
        self._timed = {tv: c for tv, c in self._timed.items() if not vertex_equal(tv.value, key)}
        dict.__delitem__(self, key)

    def setdefault(self, key: vertex, default: float = 0.0) -> float:
        if key not in self:
            self[key] = default
        return dict.get(self, key)

    def pop(self, key: vertex, *default):
        if key not in self:
            if default:
                return default[0]
            raise KeyError(key)
        value = dict.get(self, key)
        del self[key]
        return value

    def popitem(self):
        key, value = dict.popitem(self)
        self._timed = {tv: c for tv, c in self._timed.items() if not vertex_equal(tv.value, key)}
        return key, value

    def update(self, other=(), /, **kwargs) -> None:
        items = other.items() if hasattr(other, "items") else other
        for k, v in items:
            self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def clear(self) -> None:
        self._timed.clear()
        dict.clear(self)

    def copy(self) -> TimedCounter:
        """No direct C++ symbol: `std::map`'s copy constructor, used by
        `CTable`'s own value copies. `dict.copy()` would return a bare
        `dict` and lose the observation list entirely.
        """
        new = TimedCounter()
        for tv, c in self._timed.items():
            new.add(tv.value, tv.timestamp, c)
        return new


def timed_keys_of(counter) -> Iterable[TimedValue]:
    """No direct C++ symbol: iterating a `TimedCounter`'s keys, tolerating a
    cell that is a plain `collections.Counter` instead.

    In C++ every CTable cell is a `TimedCounter` and there is nothing to
    tolerate. In this port a cell can also be a plain `Counter[vertex]`,
    from two sources: `scoring_base._copy_ctable` rebuilds `_wrk_ctable`'s
    cells that way, and hand-built test tables do the same. Such a cell is
    semantically a `TimedCounter` all of whose observations carry the
    not-a-date sentinel -- exactly what `Table::compressed` builds for an
    untimestamped table (table.cc:513) -- so it is presented as one here
    rather than crashing on a missing attribute or, worse, being skipped.
    """
    timed = getattr(counter, "timed_keys", None)
    if timed is not None:
        return timed()
    return [TimedValue(v) for v in counter]


# C++: moses/comboreduct/table/table.h:518 CTableTime
# `typedef std::map<TTable::value_type, Counter<vertex, count_t>> CTableTime;`
# -- "Like CTable but the keys are timestamps. Inputs are currently not
# supported." (table.h:510-513's own comment.) A plain dict here, built
# ONLY by `CTable.ordered_by_time()`, which sorts its keys on the way out
# so the `std::map` ordering callers depend on is preserved.
CTableTime = dict


# ---------------------------------------------------------------------------
# CTable -- table.h:550
# ---------------------------------------------------------------------------


class CTable(dict):
    """C++: moses/comboreduct/table/table.h:550 CTable

    `public std::map<multi_type_seq, TimedCounter>`, narrowed exactly as
    the brief specifies: `dict[tuple[vertex, ...], Counter[vertex]]`.
    `TimedCounter` (table.h:496, `Counter<TimedValue, count_t>`) itself
    narrows to a plain `Counter[vertex]` -- `TimedValue` pairs a vertex
    with a timestamp (table.h:472), and this port carries no timestamps
    (see module docstring), so every `TimedValue` key collapses to its
    bare `vertex`.

    `count_t` is `double` (table.h:494, "so that this will work with
    weighted features") -- every counter value here is a Python `float`,
    never `int`, so weighted rows (`weighted.csv`) compress correctly.

    SWEPT (S14 Task 0, CARRY-FORWARD.md section 2.5): this class's own
    `dict[tuple[vertex, ...], Counter[vertex]]` is a vertex collection used
    as a dict/tuple key with no explicit type discrimination -- the same
    category of risk `vertex_equal` (comboreduct/combo/vertex.py) exists to
    guard against, since Python's tuple/dict machinery hashes and compares
    element-wise with bare `==`/`hash()`, and `hash(1.0) == hash(builtin.
    logical_true)` is guaranteed whenever `1.0 == builtin.logical_true` is
    (both `True` in bare Python -- see `vertex_equal`'s own docstring).
    Checked, not assumed: this is safe FOR EVERY `Table`/`CTable` this
    package itself ever builds, because every such table is populated by
    `table_io.py`'s loaders, which infer exactly ONE `type_node` per column
    (`infer_type_from_token2`) and convert every row's entry in that column
    through `token_to_vertex(that_one_type, token)` -- so a given tuple
    POSITION holds values of only one vertex alternative for the table's
    entire lifetime. The `contin_t`/`builtin` collision requires the SAME
    tuple position to hold both a `contin_t` and a `builtin` across
    different rows, which never happens under that invariant: two rows can
    disagree in VALUE at a position, never in KIND. Not proof against a
    caller that hand-builds an `ITable`/`CTable` bypassing `table_io.py`
    (this class enforces nothing at the container level, matching C++'s own
    `std::vector<vertex_seq>`/`std::map<multi_type_seq, TimedCounter>`,
    which are equally unchecked containers) -- no such caller exists in
    this package today (`grep -rn "push_back" comboreduct/` and `grep -rn
    "insert_col" comboreduct/` show only `table_io.py`'s own loaders).
    """

    def __init__(
        self,
        olabel: str = _DEFAULT_OUTPUT_LABEL,
        ilabels: Sequence[str] = (),
        signature: Signature | None = None,
    ) -> None:
        """C++: moses/comboreduct/table/table.cc:631 CTable
        and moses/comboreduct/table/table.cc:640
        CTable(const string&, const string_seq&, const type_tree&)
        """
        super().__init__()
        self.olabel = olabel
        self.ilabels: list[str] = list(ilabels)
        self._signature = signature

    def get_arity(self) -> int:
        """C++: moses/comboreduct/table/table.h:571 get_arity"""
        return len(self.ilabels)

    def uncompressed_size(self) -> float:
        """C++: moses/comboreduct/table/table.cc:780 CTable::uncompressed_size

        Sum, over every compressed row, of that row's total observation
        count (`count_t`, a `float` -- see class docstring). Equals the
        number of rows in the original uncompressed table only when every
        row weighs exactly 1.0.
        """
        return sum(sum(counter.values()) for counter in self.values())

    def get_labels(self) -> list[str]:
        """C++: moses/comboreduct/table/table.cc:753 CTable::get_labels

        "return the output label + list of input labels" (table.h:706's
        doc comment) -- output first, then inputs, same order
        `Table.get_labels()` uses.
        """
        return [self.olabel, *self.ilabels]

    def set_labels(self, labels: Sequence[str]) -> None:
        """C++: moses/comboreduct/table/table.cc:746 CTable::set_labels"""
        labels = list(labels)
        self.olabel = labels[0]
        self.ilabels = labels[1:]

    def get_output_label(self) -> str:
        """C++: moses/comboreduct/table/table.cc:760 CTable::get_output_label"""
        return self.olabel

    def get_input_labels(self) -> list[str]:
        """C++: moses/comboreduct/table/table.cc:765 CTable::get_input_labels"""
        return self.ilabels

    def set_signature(self, signature: Signature) -> None:
        """C++: moses/comboreduct/table/table.cc:770 CTable::set_signature"""
        self._signature = signature

    def get_signature(self) -> Signature | None:
        """C++: moses/comboreduct/table/table.cc:775 CTable::get_signature"""
        return self._signature

    def get_output_type(self) -> type_node:
        """C++: moses/comboreduct/table/table.cc:789 CTable::get_output_type"""
        return self._signature.otype if self._signature is not None else type_node.unknown_type

    def balance(self) -> None:
        """C++: moses/comboreduct/table/table.cc:806 CTable::balance

        `--balance`'s implementation (`table-problems.cc:133-134`), whose
        help text promises to "balance the resulting ctable so all classes
        have the same weight" (problem-params.cc:235-238).

        **It does nothing.** Not a narrowing, not an approximation: the
        write-back loop mutates temporaries and is discarded. Both of its
        range-for declarations bind BY VALUE --

            for (auto iorow : *this)            // copy of the (key, cell) pair
                for (auto tvc : iorow.second)   // copy of the (value, count) pair
                    tvc.second *= (usize / n) / class_count[tvc.first.value];

        -- so `tvc.second` is a field of a copy of a copy, and `*=` writes
        it and then drops it at the end of the iteration. `auto&` on either
        line would have made this the real thing; neither has it.

        MEASURED, not inferred (`tools/cpp_probe/ctable_balance_probe.cc`,
        captured into `tests/fixtures/ctable_balance.json`): on an 8-row
        boolean table split 7 `true` / 1 `false` -- where a real balance
        would scale the `true` rows by `(8/2)/7 = 0.571...` and the
        `false` row by `(8/2)/1 = 4.0` -- every cell's counts are
        byte-identical before and after, and `uncompressed_size()` stays
        8.0. Confirmed end to end as well: classic's `moses --balance=1`
        and `--balance=0` produce byte-identical stdout.

        DO NOT FIX. Classic accepts `--balance` and exits 0 having changed
        nothing; a port that actually balanced would silently disagree with
        every classic run that passes the flag. The `class_count`/`usize`
        computation IS reproduced below, because it is the part of the
        function that genuinely executes, and the loop that throws its
        result away is written out rather than elided so that the shape of
        the bug stays legible to the next reader.

        The `else` branch really does run: a non-discrete (contin) output
        type logs `logger().warn()` and leaves the table alone -- which,
        given the above, is the same observable outcome by a different
        route.
        """
        otype = self.get_output_type()
        if otype in (type_node.boolean_type, type_node.enum_type):
            # C++: table.cc:811-813. `class_count += iorow.second.untimedCounter()`
            # -- `untimedCounter()` (table.cc:616) collapses a cell's
            # `TimedValue` keys onto their bare vertices, which is exactly
            # what iterating this port's own cell mapping already yields
            # (see `TimedCounter`'s class docstring). `class_count` is a
            # real local, so THIS accumulation is not lost.
            class_count: dict[vertex, float] = {}
            for _row, counter in self.items():
                for value, count in counter.items():
                    class_count[value] = class_count.get(value, 0.0) + count

            # C++: table.cc:815. `count_t usize = uncompressed_size(), n = class_count.size();`
            usize = self.uncompressed_size()
            n = float(len(class_count))

            # C++: table.cc:829-832 -- the loop that does nothing. Both
            # bindings are copies here for the same reason they are copies
            # in C++, so that `scaled` is provably a dead store in both.
            for _row, counter in self.items():
                cell_copy = dict(counter)  # C++'s `auto iorow` (by value)
                for value in cell_copy:
                    tvc_second = cell_copy[value]  # C++'s `auto tvc` (by value)
                    denominator = class_count[value]
                    if denominator == 0.0:
                        # C++ divides in IEEE-754 double, which yields
                        # +/-inf (or NaN for 0/0) and never raises, where
                        # Python's `/` raises ZeroDivisionError. Reachable
                        # only from a weighted table whose weights sum to
                        # zero for some class; the product is discarded
                        # either way, so the branch exists purely so this
                        # port cannot raise where classic cannot.
                        continue
                    tvc_second *= (usize / n) / denominator  # noqa: F841  (dead in C++ too)
        else:
            # C++: table.cc:834-836. C++ streams the `type_node` through
            # its own `operator<<` (type_tree_def.h), which prints the
            # SHORT name ("contin"), not the enumerator spelling -- hence
            # `otype.name.removesuffix("_type")` rather than `%s` of the
            # Python enum, which would print "type_node.contin_type".
            logger().warn(
                "CTable::balance() - cannot balance non discrete output type %s",
                otype.name.removesuffix("_type"),
            )

    def filtered(self, filter_indices: Iterable[int]) -> CTable:
        """C++: moses/comboreduct/table/table.h:585 CTable::filtered

        Same "container of column indices to keep, in the order given"
        contract as `ITable.filtered` -- see its docstring. Rows that
        collide after filtering (two distinct original rows mapping to
        the same filtered key) have their counters MERGED, not
        overwritten (`res[key] += v.second` in C++, table.h:619).
        """
        idxs = list(filter_indices)
        sig = self._signature
        new_sig = Signature([sig.itypes[i] for i in idxs], sig.otype) if sig is not None else None
        res = CTable(self.olabel, [self.ilabels[i] for i in idxs], new_sig)
        for row, counter in self.items():
            key = tuple(row[i] for i in idxs)
            dst = res.setdefault(key, TimedCounter())
            # Merge per OBSERVATION, not per vertex: `res[key] += v.second`
            # (table.h:619) merges two whole `TimedCounter`s, which keeps
            # each `(value, timestamp)` distinct.
            for tv, c in counter.timed_items():
                dst.add(tv.value, tv.timestamp, c)
        return res

    def get_timestamps(self) -> set:
        """C++: moses/comboreduct/table/table.cc:735 CTable::get_timestamps

        The set of distinct timestamps appearing anywhere in the table,
        EXCLUDING the default not-a-date sentinel -- `if (vtc.first.
        timestamp != boost::gregorian::date())` (table.cc:740). So a table
        loaded without `--timestamp-feature`, whose every observation
        carries that sentinel, correctly yields the empty set; it is not a
        stub.

        Returned as a `set`, matching `std::set<TTable::value_type>`.
        Callers that need C++'s ascending iteration order must sort --
        `ss_bscore._partition_by_time` draws from it randomly and does not.
        """
        return {tv.timestamp for counter in self.values() for tv in counter.timed_keys()} - {None}

    def remove_rows_at_time(self, timestamp) -> None:
        """C++: moses/comboreduct/table/table.cc:713 CTable::remove_rows_at_time

        Drops every observation made at `timestamp`, then drops any row
        whose cell became empty -- both in place, matching C++'s
        `outputs.erase(v_it)` / `erase(row_it)`.
        """
        for row in list(self.keys()):
            counter = self[row]
            kept = [(tv, c) for tv, c in counter.timed_items() if tv.timestamp != timestamp]
            if len(kept) == counter.timed_len():
                continue
            if not kept:
                del self[row]
                continue
            replacement = TimedCounter()
            for tv, c in kept:
                replacement.add(tv.value, tv.timestamp, c)
            dict.__setitem__(self, row, replacement)

    def remove_rows_at_times(self, timestamps: Iterable) -> None:
        """C++: moses/comboreduct/table/table.cc:707 CTable::remove_rows_at_times

        `for (const TTable::value_type& timestamp : timestamps)
        remove_rows_at_time(timestamp);` -- one full pass per timestamp,
        reproduced rather than fused into a single pass, because the
        per-timestamp pass is what decides when a row disappears.
        """
        for timestamp in timestamps:
            self.remove_rows_at_time(timestamp)

    def ordered_by_time(self) -> dict:
        """C++: moses/comboreduct/table/table.cc:794 CTable::ordered_by_time

        "Turn the input to timestamped output map into timestamp to output
        map" (table.cc:796-797's own comment): every observation in the
        table, regrouped by timestamp and summed per value, discarding the
        inputs entirely.

        The result stands in for `CTableTime`, a `std::map` keyed on the
        date -- and unlike `TimedValue`, `boost::gregorian::date`'s own
        `operator<` IS a total order, so that map's iteration order is well
        defined and observable: `get_time_dispersion_penalty` walks it to
        build `act_time_dist` (time_dispersion.cc:85-90), and
        `precision_bscore`'s `time_bscore` branch pushes one bscore
        component per entry IN ORDER (precision_bscore.cc:266-274). The
        entries are therefore sorted here on the way out, ascending, with
        the not-a-date sentinel LAST (`timestamp_less`), rather than left
        in encounter order.
        """
        res: dict = {}
        for counter in self.values():
            for tv, c in counter.timed_items():
                bucket = res.setdefault(tv.timestamp, Counter())
                bucket[tv.value] = bucket.get(tv.value, 0.0) + c
        return {
            k: res[k]
            for k in sorted(res, key=cmp_to_key(lambda a, b: -1 if timestamp_less(a, b) else 1))
        }


# ---------------------------------------------------------------------------
# Table -- table.h:918
# ---------------------------------------------------------------------------


class Table:
    """C++: moses/comboreduct/table/table.h:918 Table

    `itable` + `otable` + `ttable`, plus `target_pos`/`timestamp_pos`, the
    original column positions of the target and timestamp columns (used by
    `ostreamTable` to put them back where they were).

    `ttable` is empty unless `loadTable` was given a `timestamp_feature`;
    when it is non-empty it holds exactly one date per row, parallel to
    `itable`/`otable`.
    """

    def __init__(self, otable: OTable | None = None, itable: ITable | None = None) -> None:
        """C++: moses/comboreduct/table/table.cc:482 Table
        and moses/comboreduct/table/table.cc:484 Table(const OTable&, const ITable&)

        Copies both arguments. table.cc:484's member-initializer list
        (`itable(itable_), otable(otable_)`) copy-constructs each field
        from a `const&` -- `ITable`/`OTable` are C++ value types
        (`std::vector`-derived), so mutating the caller's argument object
        after this call can never be observed through the constructed
        `Table`. A prior version of this port aliased both (`self.itable
        = itable`), which `_load_table_old` (table_io.py) depended on to
        observe `ITable.delete_column` through the constructed `Table` --
        see that function's docstring for how the dependency was removed
        by mirroring `istreamTable_OLD` (table_io.cc:856), which never
        calls this constructor at all: it fills `tab.itable` IN PLACE.
        docs/CARRY-FORWARD.md section 1.1.
        """
        self.itable: ITable = itable.copy() if itable is not None else ITable()
        self.otable: OTable = otable.copy() if otable is not None else OTable()
        self.ttable: TTable = TTable()
        self.target_pos: int = 0
        self.timestamp_pos: int = 0

    def size(self) -> int:
        """C++: moses/comboreduct/table/table.h:936 size"""
        return len(self.itable)

    def get_arity(self) -> int:
        """C++: moses/comboreduct/table/table.h:938 get_arity"""
        return self.itable.get_arity()

    def get_signature(self) -> Signature:
        """C++: moses/comboreduct/table/table.h:944 get_signature

        Built directly from `itable.get_types()` + `otable.get_type()`.
        `Signature` is the compatibility name for S8's canonical immutable
        `TypeSignature`, the same fixed-signature shape produced by
        `type_checker.gen_signature` without introducing a general type tree.
        """
        return Signature(self.itable.get_types(), self.otable.get_type())

    def get_labels(self) -> list[str]:
        """C++: moses/comboreduct/table/table.cc:492 Table::get_labels"""
        return [self.otable.get_label(), *self.itable.get_labels()]

    def get_target(self) -> str:
        """C++: moses/comboreduct/table/table.h:955 get_target"""
        return self.otable.get_label()

    def update_pos(self, pos: int, f: Iterable[int]) -> int:
        """C++: moses/comboreduct/table/table.h:961 update_pos

        Given `pos` (a column index in the pre-filter coordinate space)
        and `f` (the same "indices to keep" filter `filtered()` uses),
        returns `pos`'s position after filtering. Walks `f` in ascending
        order accumulating the gaps between consecutive kept indices --
        each such gap is exactly the count of columns dropped in that
        span -- until either an element `>= pos` is reached (then only
        the portion of that last gap before `pos` counts) or `f` is
        exhausted.
        """
        filtered_out_count = 0
        last = 0
        for v in f:
            if v < pos:
                filtered_out_count += v - last
            else:
                filtered_out_count += pos - last
                break
            last = v
        return pos - filtered_out_count

    def filtered(self, f: Iterable[int]) -> Table:
        """C++: moses/comboreduct/table/table.h:978 filtered

        `timestamp_pos` is updated only when `ttable` is non-empty
        (table.h:996-997's own `if (!ttable.empty())`), so a table with no
        timestamp column keeps `timestamp_pos == 0` rather than having a
        meaningless index shifted.
        """
        idxs = list(f)
        res = Table()
        res.itable = self.itable.filtered(idxs)
        res.otable = self.otable.copy()
        res.ttable = TTable(self.ttable, self.ttable.get_label())
        res.target_pos = self.update_pos(self.target_pos, idxs)
        if self.ttable:
            res.timestamp_pos = self.update_pos(self.timestamp_pos, idxs)
        return res

    def compressed(self, weighting_feature: str = "") -> CTable:
        """C++: moses/comboreduct/table/table.cc:501 Table::compressed

        If `weighting_feature` is empty, every row counts for 1.0
        (`count_t`, a float -- see CTable's class docstring). Otherwise
        the named column is dropped from the compressed table's own
        inputs (it is not a feature) and its per-row value -- read from
        the ORIGINAL, un-trimmed `itable` -- is added as that row's
        weight instead of a flat 1.0.

        Faithfully reproduces a pre-existing quirk of the weighted
        branch: `res`'s signature (`get_signature()`, built from the
        UN-trimmed `itable`) disagrees in arity with `res`'s own
        `ilabels` (built from the trimmed one) -- table.cc:527 passes
        `get_signature()` (not a filtered signature) exactly the same
        way. Not fixed here; this is what classic actually does.

        Both branches split on whether `ttable` is empty, exactly as C++
        does (table.cc:509-517 and :534-550): a timestamped row is counted
        under `TimedValue(out, date)`, an untimestamped one under
        `TimedValue(out)` -- the not-a-date sentinel. That second form is
        why an untimestamped table's cells are `TimedCounter`s too, rather
        than a separate shape: the observation list simply has one
        `(value, None)` entry per distinct value, so its untimed view and
        the plain `Counter[vertex]` this method used to build are the same
        mapping, element for element.
        """
        # C++: table.cc:503. `%d` fed an unsigned `size()` is classic's own
        # format/type mismatch; on x86-64 it prints the same digits.
        logger().debug("Compress the dataset, current size is %d", len(self.itable))
        signature = self.get_signature()
        timestamps = list(self.ttable) if self.ttable else None
        if not weighting_feature:
            res = CTable(self.otable.get_label(), self.itable.get_labels(), signature)
            for i, (row, out) in enumerate(zip(self.itable, self.otable, strict=True)):
                counter = res.setdefault(tuple(row), TimedCounter())
                counter.add(out, timestamps[i] if timestamps is not None else None, 1.0)
            # C++: table.cc:519 uses `%u` here and `%d` at :551 for the same
            # quantity -- reproduced as written rather than harmonised.
            logger().debug("Size of the compressed dataset is %u", len(res))
            return res

        trimmed = self.itable.copy()
        trimmed.delete_column(weighting_feature)
        res = CTable(self.otable.get_label(), trimmed.get_labels(), signature)
        widx = self.itable.get_column_offset(weighting_feature)
        for i, (orig_row, row, out) in enumerate(
            zip(self.itable, trimmed, self.otable, strict=True)
        ):
            weight = get_contin(orig_row[widx])
            counter = res.setdefault(tuple(row), TimedCounter())
            counter.add(out, timestamps[i] if timestamps is not None else None, weight)
        logger().debug("Size of the compressed dataset is %d", len(res))
        return res

    def __eq__(self, other: object) -> bool:
        """C++: moses/comboreduct/table/table.cc:599 Table

        Cites the enclosing class (see ITable.__eq__ above); the real
        member is `Table::operator==` (table.cc:599-602), which compares
        `itable`, `otable`, `ttable`, `target_pos` AND `timestamp_pos`.

        # DIVERGENCE(C++ moses/comboreduct/table/table.cc:599): the two
        # position fields are NOT compared here. They are output-formatting
        # state (`ostreamTable` uses them to put the target and timestamp
        # columns back where the input file had them), not data, and
        # several places in this port build a `Table` by assigning
        # `itable`/`otable` in place without restoring `target_pos` -- so
        # comparing them would make equality depend on construction route
        # rather than content. `ttable` IS compared, since it is data.
        """
        if not isinstance(other, Table):
            return NotImplemented
        return (
            self.itable == other.itable
            and self.otable == other.otable
            and list(self.ttable) == list(other.ttable)
        )


# ---------------------------------------------------------------------------
# complete_truth_table -- table.h:1436. Scoped in from S7's own "Known gap,
# deliberate" list (table.py's module docstring) into Task 5 (comboreduct/
# reduct): `logical_rules.py`'s `reduce_remove_subtree_equal_tt`
# (logical_rules.cc:833) needs it, and needs only it -- not the rest of
# table.h's scoring/subsampling machinery.
# ---------------------------------------------------------------------------


class complete_truth_table(list):
    """C++: moses/comboreduct/table/table.h:1436 complete_truth_table

    Enumerates a boolean combo tree's output over every one of its
    `2**arity` input combinations, in the canonical bit order table.h's
    own doc comment gives: input `i`'s j-th bit (`(i >> j) % 2`) is
    `$(j+1)`'s value.

    `class complete_truth_table : public bool_seq` (`bool_seq` =
    `std::vector<bool>`) is a plain `list[bool]` subclass here -- the same
    convention `ITable(list)`/`OTable(list)`/`CTable(dict)` already use in
    this file for a C++ container subclass. `==` between two instances is
    therefore ordinary `list.__eq__` (matching `bool_seq`'s own, since
    C++ declares no custom `operator==` for this class either).

    NARROWLY SCOPED, matching this module's own established convention
    for a scoped-in dependency (see `_vertex_type_node`'s docstring): only
    the ONE constructor form `reduce_remove_subtree_equal_tt` actually
    needs (`complete_truth_table(tr)`, table.h:1451, with the 2-arg
    explicit-arity overload at table.h:1445 folded in via a keyword
    default -- both used by `tools/cpp_probe/reduct_effort_probe.cc`) plus
    `same_complete_truth_table` are ported. NOT ported: the `(It, It)`
    range constructor (table.h:1443), the `(Func, arity)` functional
    constructor (table.h:1459), `operator()(It, It)` (the index-lookup
    accessor, table.h:1477), and `hamming_distance` (table.h:1485) -- zero
    call sites in this project's reachable scope (`reduce_remove_subtree_
    equal_tt` only ever constructs from a tree and only ever calls
    `same_complete_truth_table`).

    CORRECTED (Task 5 fix round): every C++ line number in this class's
    own docstrings was off by a small, systematic amount (the un-ported
    `(It, It)` ctor at table.h:1443 had been cited as this class's own
    PORTED-constructor breadcrumb, among other misses) -- caught by
    `tools/check_alignment.py` validating only that a cited line EXISTS
    in the named file, not that it names the cited symbol (that script's
    own module docstring says so explicitly). Every citation below was
    re-checked directly against `grep -n` on the real
    `moses/comboreduct/table/table.h`.
    """

    def __init__(self, tr: combo_tree, arity: int | None = None) -> None:
        """C++: moses/comboreduct/table/table.h:1445 complete_truth_table

        Two C++ constructor overloads collapsed into one Python signature
        (this port's usual convention for an overloaded C++ constructor):
        `arity` given explicitly (table.h:1445, `super(pow2(arity)),
        _arity(arity)`) or omitted, in which case it is inferred from `tr`
        itself (table.h:1451-1455, `_arity = arity(tr)` -- `arity`, the
        free function at `interpreter/eval.h:72`, ported as `eval.arity`
        and imported here under a local name to avoid shadowing this
        constructor's own `arity` parameter, which mirrors the C++
        parameter name verbatim).
        """
        self._arity: int = _tree_arity(tr) if arity is None else arity
        super().__init__([False] * (2**self._arity))
        self._populate(tr)

    def _populate(self, tr: combo_tree) -> None:
        """C++: moses/comboreduct/table/table.h:1495 complete_truth_table::populate

        `mutable builtin_seq inputs;` (table.h:1506) is a scratch buffer
        reused across calls in C++ (also by `same_complete_truth_table`
        below, hence `mutable` despite that method being `const`); no
        Python equivalent needed, a fresh list per row costs nothing here.
        """
        for i in range(len(self)):
            inputs = [bool_to_vertex(bool((i >> j) % 2)) for j in range(self._arity)]
            self[i] = builtin_to_bool(boolean_interpreter(inputs)(tr))

    def same_complete_truth_table(self, tr: combo_tree) -> bool:
        """C++: moses/comboreduct/table/table.cc:855 complete_truth_table::same_complete_truth_table

        Evaluates `tr` at THIS truth table's own `_arity` (not `tr`'s own
        -- `tr` may reference fewer arguments, e.g. after amputating a
        subtree; the extra, unreferenced synthetic inputs are simply
        unused by `boolean_interpreter`). C++'s own "optimized ... detect
        [inequality] before calculating the entire table" (table.h:1488-
        1489) is a short-circuiting `for` loop with an early `return
        false` -- the same short-circuit a Python `for`/`return` gives for
        free; no separate optimization to reproduce.
        """
        for i in range(len(self)):
            inputs = [bool_to_vertex(bool((i >> j) % 2)) for j in range(self._arity)]
            if self[i] != builtin_to_bool(boolean_interpreter(inputs)(tr)):
                return False
        return True
