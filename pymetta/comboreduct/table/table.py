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
* No `type_checker`/`type_tree`. Per the plan's "Known gap, deliberate"
  note, `type_checker/` is out of scope for the whole S0-S6 cycle; a
  `type_tree` (`opencog::tree<type_node>`, shaped
  `lambda_type(itype1, ..., itypeN, otype)`) is replaced by the plain
  `Signature` value object below, carrying exactly the same information
  (`Table.get_signature()`/`CTable.get_signature()`) without the tree
  machinery.
* No `TTable`/`CTableTime`. Timestamped compression is out of scope (the
  brief's `Table` interface has no `ttable` field); `loadTable`
  (table_io.py) still recognises and *excludes* a `timestamp_feature`
  column from `itable`, matching classic's column-exclusion behaviour
  exactly, but the timestamp values themselves are discarded rather than
  captured in a `TTable`. See table_io.py's module docstring.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from enum import IntEnum

from pymetta.comboreduct.combo.vertex import (
    bool_to_vertex,
    builtin,
    builtin_to_bool,
    combo_tree,
    get_contin,
    vertex,
)
from pymetta.comboreduct.interpreter.eval import arity as _tree_arity
from pymetta.comboreduct.interpreter.interpreter import boolean_interpreter

# ---------------------------------------------------------------------------
# type_node -- narrow stand-in for type_checker/type_tree_def.h's id::type_node
# ---------------------------------------------------------------------------


class type_node(IntEnum):
    """C++: moses/comboreduct/type_checker/type_tree_def.h:32 type_node

    Transcribed verbatim from `namespace id { enum type_node {...} }`:
    same member names (the `id::` qualifier dropped, matching how
    `builtin` drops it -- see vertex.py), same implicit ordinals. Ported
    here rather than as its own `type_checker` module because the plan's
    "Known gap, deliberate" note scopes `type_checker/` out of the whole
    S0-S6 cycle; this enum is the one piece of it `ITable`/`OTable`
    column typing cannot do without. Every member is carried for ordinal
    fidelity even though this port's `token_to_vertex` (table_io.py) only
    ever produces `boolean_type`, `contin_type`, `enum_type`,
    `definite_object_type` and `ill_formed_type` values -- the rest exist
    only so a future task extending this enum finds the same ordinals
    classic uses.
    """

    lambda_type = 0
    application_type = 1
    union_type = 2
    arg_list_type = 3
    boolean_type = 4
    contin_type = 5
    enum_type = 6
    list_type = 7
    action_result_type = 8
    definite_object_type = 9
    action_definite_object_type = 10
    indefinite_object_type = 11
    message_type = 12
    action_symbol_type = 13
    wild_card_type = 14
    ann_type = 15
    unknown_type = 16
    ill_formed_type = 17
    argument_type = 18


class Signature:
    """Stand-in for the `type_tree` classic builds as
    `lambda_type(itype1, ..., itypeN, otype)` -- see `Table.get_signature`
    (table.h:944) and `CTable`'s `tsig` field (table.h:731). Not itself a
    ported C++ symbol (no `type_tree` class exists in this port -- see
    module docstring); carries exactly the same two pieces of information
    `get_signature()` callers actually use.
    """

    __slots__ = ("itypes", "otype")

    def __init__(self, itypes: Sequence[type_node], otype: type_node) -> None:
        self.itypes: tuple[type_node, ...] = tuple(itypes)
        self.otype: type_node = otype

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Signature):
            return NotImplemented
        return self.itypes == other.itypes and self.otype == other.otype

    def __repr__(self) -> str:
        return f"Signature(itypes={self.itypes!r}, otype={self.otype!r})"


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
# ITable -- table.h:738
# ---------------------------------------------------------------------------

_DEFAULT_INPUT_LABEL = "i"


class ITable(list):
    """C++: moses/comboreduct/table/table.h:738 ITable

    `public std::vector<multi_type_seq>`, narrowed to
    `list[tuple[vertex, ...]]` -- see module docstring. Each element is
    one row; `row[i]` is column `i`'s value for that row, in the same
    order as `get_labels()`/`get_types()`.
    """

    def __init__(
        self,
        types: Sequence[type_node] = (),
        labels: Sequence[str] = (),
    ) -> None:
        """C++: moses/comboreduct/table/table.cc:77 ITable

        Covers the `ITable(const type_seq&, const string_seq&)` overload
        (table.cc:77-78): construct an (initially empty) table with known
        column types/labels, rows added afterwards via `push_back`. The
        default-constructed `ITable()` (table.cc:75) and the
        rows-plus-labels overload (table.cc:80-81, `ITable(const super&
        mat, const string_seq& il)`) are not separately ported -- no
        caller in this port's scope needs to hand rows to the
        constructor rather than to `push_back` afterwards.
        """
        super().__init__()
        self._types: list[type_node] = list(types)
        self._labels: list[str] = list(labels)

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
    `list[vertex]` -- a single output column. Only the constructors this
    port's callers actually need are ported: label-only/values-and-label
    (table.cc:309-312). The combo_tree-evaluating constructors
    (table.cc:319, table.cc:350 -- "construct the OTable by evaluating
    the combo tree for each row") are out of this task's scope (that is
    Task 6's `behave_cscore`/scoring path, not table loading).
    """

    def __init__(
        self, values: Iterable[vertex] = (), label: str = _DEFAULT_OUTPUT_LABEL
    ) -> None:
        """C++: moses/comboreduct/table/table.cc:309 OTable
        and moses/comboreduct/table/table.cc:312 OTable(const super&, const string&)

        One constructor covers both C++ overloads (merged the same way
        `ITable.__init__` covers its own two overloads): the label-only
        overload (table.cc:309-310, `type = id::unknown_type`) and the
        values-plus-label overload (table.cc:312-318), which infers the
        column type from the FIRST value:
        `type = get_type_node(get_type_tree((*this)[0]))`. `values` being
        empty is exactly what distinguishes the two -- `(*this)[0]` is C++
        UB on an empty vector, which the real overload set avoids simply
        by that overload never being called with no values; the `len(self)
        > 0` guard here is the runtime equivalent, needed only because one
        Python constructor stands in for both.
        """
        super().__init__(values)
        self._label = label
        self._type: type_node = (
            _vertex_type_node(self[0]) if len(self) > 0 else type_node.unknown_type
        )

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
        new_sig = (
            Signature([sig.itypes[i] for i in idxs], sig.otype) if sig is not None else None
        )
        res = CTable(self.olabel, [self.ilabels[i] for i in idxs], new_sig)
        for row, counter in self.items():
            key = tuple(row[i] for i in idxs)
            dst = res.setdefault(key, Counter())
            for v, c in counter.items():
                dst[v] = dst.get(v, 0.0) + c
        return res


# ---------------------------------------------------------------------------
# Table -- table.h:918
# ---------------------------------------------------------------------------


class Table:
    """C++: moses/comboreduct/table/table.h:918 Table

    No `ttable`/`timestamp_pos` field -- out of scope, see module
    docstring. `timestamp_pos` is still computed internally by
    `table_io.py`'s dense loader (it costs nothing and keeps the
    ignore/target/timestamp index arithmetic visibly parallel to C++'s
    `istreamDenseTable_noHeader`), but it is not part of this class's
    public surface since nothing in this port's scope reads it.
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
        self.target_pos: int = 0

    def size(self) -> int:
        """C++: moses/comboreduct/table/table.h:936 size"""
        return len(self.itable)

    def get_arity(self) -> int:
        """C++: moses/comboreduct/table/table.h:938 get_arity"""
        return self.itable.get_arity()

    def get_signature(self) -> Signature:
        """C++: moses/comboreduct/table/table.h:944 get_signature

        Built directly from `itable.get_types()` + `otable.get_type()` --
        the plan's "constructed directly as gen_signature(boolean_type,
        arity)" note (see docs/superpowers/plans/2026-09-03-pymetta-s0-s6
        -foundation.md's "Known gap, deliberate" section): no
        `type_checker::gen_signature` call needed since `Signature` is
        already this shape.
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
        """C++: moses/comboreduct/table/table.h:978 filtered"""
        idxs = list(f)
        res = Table()
        res.itable = self.itable.filtered(idxs)
        res.otable = self.otable.copy()
        res.target_pos = self.update_pos(self.target_pos, idxs)
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
        """
        signature = self.get_signature()
        if not weighting_feature:
            res = CTable(self.otable.get_label(), self.itable.get_labels(), signature)
            for row, out in zip(self.itable, self.otable, strict=True):
                counter = res.setdefault(tuple(row), Counter())
                counter[out] = counter.get(out, 0.0) + 1.0
            return res

        trimmed = self.itable.copy()
        trimmed.delete_column(weighting_feature)
        res = CTable(self.otable.get_label(), trimmed.get_labels(), signature)
        widx = self.itable.get_column_offset(weighting_feature)
        for orig_row, row, out in zip(self.itable, trimmed, self.otable, strict=True):
            weight = get_contin(orig_row[widx])
            counter = res.setdefault(tuple(row), Counter())
            counter[out] = counter.get(out, 0.0) + weight
        return res

    def __eq__(self, other: object) -> bool:
        """C++: moses/comboreduct/table/table.cc:599 Table

        Cites the enclosing class (see ITable.__eq__ above); the real
        member is `Table::operator==` (table.cc:599-602).
        """
        if not isinstance(other, Table):
            return NotImplemented
        return self.itable == other.itable and self.otable == other.otable


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
    needs (`complete_truth_table(tr)`, table.h:1449, with the 2-arg
    explicit-arity overload at table.h:1443 folded in via a keyword
    default -- both used by `tools/cpp_probe/reduct_effort_probe.cc`) plus
    `same_complete_truth_table` are ported. NOT ported: the `(It, It)`
    range constructor (table.h:1441), the `(Func, arity)` functional
    constructor (table.h:1456), `operator()(It, It)` (the index-lookup
    accessor, table.h:1467), and `hamming_distance` (table.h:1481) -- zero
    call sites in this project's reachable scope (`reduce_remove_subtree_
    equal_tt` only ever constructs from a tree and only ever calls
    `same_complete_truth_table`).
    """

    def __init__(self, tr: combo_tree, arity: int | None = None) -> None:
        """C++: moses/comboreduct/table/table.h:1443 complete_truth_table

        Two C++ constructor overloads collapsed into one Python signature
        (this port's usual convention for an overloaded C++ constructor):
        `arity` given explicitly (table.h:1443, `super(pow2(arity)),
        _arity(arity)`) or omitted, in which case it is inferred from `tr`
        itself (table.h:1449-1453, `_arity = arity(tr)` -- `arity`, the
        free function at `interpreter/eval.h:72`, ported as `eval.arity`
        and imported here under a local name to avoid shadowing this
        constructor's own `arity` parameter, which mirrors the C++
        parameter name verbatim).
        """
        self._arity: int = _tree_arity(tr) if arity is None else arity
        super().__init__([False] * (2**self._arity))
        self._populate(tr)

    def _populate(self, tr: combo_tree) -> None:
        """C++: moses/comboreduct/table/table.h:1487 complete_truth_table::populate

        `mutable builtin_seq inputs;` (table.h:1494) is a scratch buffer
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
        [inequality] before calculating the entire table" (table.h:1483-
        1485) is a short-circuiting `for` loop with an early `return
        false` -- the same short-circuit a Python `for`/`return` gives for
        free; no separate optimization to reproduce.
        """
        for i in range(len(self)):
            inputs = [bool_to_vertex(bool((i >> j) % 2)) for j in range(self._arity)]
            if self[i] != builtin_to_bool(boolean_interpreter(inputs)(tr)):
                return False
        return True
