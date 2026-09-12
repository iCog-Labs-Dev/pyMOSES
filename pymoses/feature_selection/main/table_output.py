"""C++: moses/comboreduct/table/table_io.h (the ``ostreamTable`` half) plus
``table_fmt_vertex_to_str`` (moses/comboreduct/table/table.cc:63).

The CSV writer the standalone ``feature-selection`` binary prints its
result through (``write_results`` -> ``ostreamTable(cout, table_wff)`` /
``saveTable(file, table_wff)``, feature-selection.cc:98-106). Everything
here is a verbatim transliteration of table_io.h:172-229; nothing is
invented.

# DIVERGENCE(C++ moses/comboreduct/table/table_io.h:196): PLACEMENT ONLY.
# These four symbols' C++ home is ``moses/comboreduct/table/table_io.{h,cc}``,
# whose Python counterpart is ``pymoses.comboreduct.table.table_io`` -- a
# module this task's own partition puts off limits (``pymoses/comboreduct/**``
# is owned by another agent). They are therefore defined HERE, in the package
# that needs them, exactly as ``pymoses.feature_selection.scorers.mutual_info``
# already ports ``mutualInformation``/``OTEntropy`` out of
# ``comboreduct/table/table.{h,cc}`` for the same reason (see that module's
# own docstring). Behaviour is unchanged; only the import path differs. When
# ``comboreduct/table/table_io.py`` grows its own ``ostreamTable``, this
# module should become a re-export of it and nothing else needs to move.

Scope note that is a real narrowing, not an oversight: this port carries no
``TTable`` at all (``pymoses.comboreduct.table.table``'s own module
docstring), so ``ostreamTableHeader``'s timestamp block
(table_io.h:182-186) and ``ostreamTable``'s (table_io.h:213-217) are both
statically dead here -- ``table.ttable.empty()`` is unconditionally true.
They are omitted rather than stubbed, matching how ``CTable``'s three
timestamp members are ABSENT rather than vacuous (docs/CARRY-FORWARD.md
section 3's companion note on why a vacuous ``get_timestamps`` would be
numerically correct and still wrong).
"""

from __future__ import annotations

from pymoses.comboreduct.combo.iostream_combo import (
    DEFAULT_OSTREAM_PRECISION,
    _vertex_to_str,
)
from pymoses.comboreduct.combo.vertex import is_boolean, vertex, vertex_to_bool
from pymoses.comboreduct.table.table import Table

__all__ = [
    "ostreamTable",
    "ostreamTableHeader",
    "saveTable",
    "table_fmt_vertex_to_str",
]


def table_fmt_vertex_to_str(v: vertex) -> str:
    """C++: moses/comboreduct/table/table.cc:63 table_fmt_vertex_to_str

    ``if (is_boolean(v)) ss << vertex_to_bool(v); else ss << v;`` -- a
    genuine boolean prints as ``0``/``1`` (C++ streams a ``bool`` as an
    integer unless ``boolalpha`` is set, and nothing here sets it), which
    is the "Boolean values are output in binary form" convention
    ``ostreamTable``'s own doc comment names (table_io.h:192-193).
    Anything else goes through ``operator<<(ostream&, const vertex&)``
    == ``ostream_vertex`` (iostream_combo.cc:428), ported as
    ``_vertex_to_str``.

    The stream is a bare ``std::stringstream`` (table.cc:65), so it
    carries an ``ostream``'s DEFAULT precision of 6 -- NOT
    ``metapop_printer``'s 18. ``DEFAULT_OSTREAM_PRECISION`` is passed
    explicitly rather than relied on as a default so that a future change
    to that constant cannot silently change this function.
    """
    if is_boolean(v):
        return "1" if vertex_to_bool(v) else "0"
    return _vertex_to_str(v, precision=DEFAULT_OSTREAM_PRECISION)


def ostreamTableHeader(table: Table) -> str:
    """C++: moses/comboreduct/table/table_io.h:172 ostreamTableHeader

    The C++ symbol is the function template ``template<typename Out> Out&
    ostreamTableHeader(Out&, const Table&)``; the breadcrumb cites the bare
    name per docs/CARRY-FORWARD.md 4.5. Returns the header LINE (without
    its trailing newline) rather than writing to a stream, since Python
    has no ``ostream`` to thread through.

    The target label is inserted at ``min(table.target_pos, hsize)``, so a
    ``Table`` whose ``target_pos`` was pushed past the end by
    ``filtered()``/``update_pos`` still produces a valid header -- that
    ``std::min`` is load-bearing, not defensive, and is why the standalone
    binary prints ``x2,y,x3`` (not ``x2,x3,y``) for
    ``-i trainingBoolean.dat -u y -C2``: ``update_pos(3, {1,2})`` == 1.
    """
    header = list(table.itable.get_labels())
    hsize = len(header)
    ol = table.otable.get_label()
    header.insert(min(table.target_pos, hsize), ol)
    return ",".join(header)


def ostreamTable(table: Table) -> str:
    """C++: moses/comboreduct/table/table_io.h:196 ostreamTable

    The C++ symbol is the function template ``template<typename Out> Out&
    ostreamTable(Out&, const Table&)``; the breadcrumb cites the bare name
    per docs/CARRY-FORWARD.md 4.5.

    Returns the whole CSV document as one string, header included, with a
    trailing newline after EVERY row including the last (C++ writes
    ``std::endl`` after each ``ostream_container``, header and rows
    alike), so ``sys.stdout.write(ostreamTable(t))`` reproduces
    ``ostreamTable(cout, t)`` byte for byte.

    ``OC_ASSERT(table.itable.empty() || isize == osize)`` (table_io.h:203)
    is reproduced as a raise: it is active in release C++ builds, so a
    bare Python ``assert`` (which ``python -O`` strips) would not be
    faithful -- the same call ``ITable.insert_col`` already makes.
    """
    isize = len(table.itable)
    osize = len(table.otable)
    if len(table.itable) != 0 and isize != osize:
        raise AssertionError("table.itable.empty() || isize == osize")

    out = [ostreamTableHeader(table)]
    for row in range(osize):
        # C++: table_io.h:206-210. `content` is empty when the itable is,
        # and then `min(target_pos, 0)` puts the output column first.
        content: list[str] = []
        if len(table.itable) != 0:
            content = [table_fmt_vertex_to_str(v) for v in table.itable[row]]
        csize = len(content)
        oc = table_fmt_vertex_to_str(table.otable[row])
        content.insert(min(table.target_pos, csize), oc)
        out.append(",".join(content))
    return "\n".join(out) + "\n"


def saveTable(file_name: str, table: Table) -> None:
    """C++: moses/comboreduct/table/table_io.cc:1322 saveTable

    ``ofstream out(file_name.c_str()); if (!out) OC_ASSERT(false, "Could
    not open %s", ...); ostreamTable(out, table);`` -- the ``ofstream``
    is opened in truncating mode (no ``ios_base::app``), unlike
    ``eval-features``'s own ``output_results``, which appends.

    C++'s ``OC_ASSERT`` on a failed open becomes whatever ``open()``
    itself raises (``OSError``); reproducing the assertion text would
    hide the real errno, and both end the process the same way.
    """
    with open(file_name, "w") as out:
        out.write(ostreamTable(table))
