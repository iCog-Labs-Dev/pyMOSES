"""C++: moses/comboreduct/table/table_io.cc

`loadTable` (table_io.cc:1296) and the parsing pipeline it drives:
`inferTableAttributes` (table_io.cc:911) makes a first pass over (at most
20 lines of) the file to guess column types, whether there is a header
row, and whether the file is dense or sparse; `istreamDenseTable`
(table_io.cc:1170) / `istreamDenseTable_noHeader` (table_io.cc:1094) then
load the common case, and `istreamSparseITable` (table_io.cc:530) /
`istreamTable_OLD` (table_io.cc:856) load the `key : val` sparse format
(see `sparse.data`).

# DIVERGENCE(C++ moses/comboreduct/table/table_io.cc:382): classic parses
# data rows in parallel with `OMP_ALGO::for_each` (visible in
# `istreamRawITable` and again in `istreamDenseTable_noHeader`'s
# `parse_line` lambda, table_io.cc:1152). This port parses rows serially,
# in file order. Row parsing has no cross-row state (each row's tokens ->
# typed values conversion depends only on that row's own text and the
# already-inferred column types), so parallel vs. serial execution order
# cannot change the resulting ITable/OTable/Table -- only the wall-clock
# time to build it.

## The column-ordering question this task exists to settle

`ITable`'s columns are exactly the file's header columns, in their
ORIGINAL left-to-right order, with the target column, the timestamp
column (if any) and every `ignore_features` column removed -- nothing is
ever reordered. This falls out of two independent mechanisms that must
agree (and do): `inferTableAttributes` computes `itypes` as `[types[i]
for i in range(arity) if i not in ignore_idxs]` (ignore_idxs here =
ignore_features indices + target_idx + timestamp_idx, ascending); the
actual per-row split (`_tokenize_row_iot`, mirroring `tokenizeRowIOT`,
table_io.cc:785) walks each row's tokens left-to-right by index and
routes each one to "input" (appended, so also left-to-right), "output"
(the target) or "timestamp", skipping ignored indices outright. Both are
driven by the same ascending index order, so the two never disagree about
either WHICH columns survive or in WHAT ORDER.

`target_pos` (table.h:1004, "position of the target, useful for writing
the table") is a separate number: the target's column index in the
ORIGINAL header, minus however many `ignore_features` columns (not
counting the target or timestamp themselves) precede it
(`istreamDenseTable_noHeader`, table_io.cc:1158-1159). It is exposed on
`Table` (see table.py) but this port does not otherwise use it (no
`ostreamTable` -- out of scope, see table.py's module docstring).

## Target by name vs. by position

The DENSE path (`istreamDenseTable`, used by every headed CSV in this
port's fixture corpus) resolves `target_feature`/`timestamp_feature`
ONLY by exact string match against the header (`header.index(...)` here,
`std::find` in C++) -- a numeric string like `"2"` is looked up as a
literal column NAME, not a position, and fails with "not found" unless a
column is actually named `"2"`. When there is no header at all, the
target is implicitly column 0 and `target_feature` must be empty (a
non-empty `target_feature`, `ignore_features` or `timestamp_feature`
without a header raises -- table_io.cc:1172-1179's `OC_ASSERT`).

Position-based targeting exists only on the SPARSE path (`sparse.data`),
and only incidentally: `ITable.get_column_offset` (table.py, ported from
table.cc:234) special-cases a name whose first character is a digit as a
1-based column position. `loadTable`'s sparse branch reaches this through
`ITable.get_column_data(target_feature)`/`delete_column(target_feature)`
(mirroring `istreamTable_OLD`, table_io.cc:885), so `target_feature="2"`
against a sparse file means "the second column" -- but the SAME string
against a dense/headed file means "the column literally named `2`".
These are genuinely different code paths with genuinely different
target-resolution semantics; this is not a bug introduced by this port,
it is what classic MOSES does.
"""

from collections.abc import Iterable, Sequence

from pymetta.comboreduct.combo.vertex import builtin, enum_t, vertex
from pymetta.comboreduct.table.table import (
    ITable,
    OTable,
    Table,
    TTable,
    get_indices,
    type_node,
)
from pymetta.comboreduct.type_checker.type_signature import (
    gen_signature,
    ostream_type_node,
    ostream_type_signature,
)
from pymetta.util.logger import logger

# ---------------------------------------------------------------------------
# get_data_line and friends -- table_io.cc:56-121
# ---------------------------------------------------------------------------

# is_comment (table_io.cc:84) treats an empty line as a comment too: C++'s
# `line[0]` on an empty std::string is well-defined (returns the null
# terminator, '\0'), and '\0' is one of is_comment's own recognised chars.
_COMMENT_CHARS = frozenset("#;!\n\r\0")


def _is_comment(line: str) -> bool:
    """C++: moses/comboreduct/table/table_io.cc:84 is_comment"""
    return line == "" or line[0] in _COMMENT_CHARS


def _remove_carriage_return(s: str) -> str:
    """C++: moses/comboreduct/table/table_io.cc:66 removeCarriageReturn"""
    return s[:-1] if s.endswith("\r") else s


def _remove_non_ascii(s: str) -> str:
    """C++: moses/comboreduct/table/table_io.cc:74 removeNonASCII

    Strips non-ASCII bytes from the START of the string only (C++'s own
    comment: "at the beginning of the line, only").
    """
    i = 0
    while i < len(s) and ord(s[i]) > 127:
        i += 1
    return s[i:]


def _read_data_lines(file_name: str) -> list[str]:
    """Combines `get_data_line`'s per-line cleanup loop (table_io.cc:104)
    with reading a whole file: classic streams line-by-line via
    `istream::getline` and re-seeks (`inferTableAttributes` peeks at up to
    20 lines, then `istreamTable` rewinds to the start before the real
    load); this port reads every cleaned line once, up front, since the
    rewind means classic's own real load pass sees the whole file anyway.
    `newline=""` (no universal-newline translation) so a literal
    trailing '\\r' -- as C++'s `std::ifstream` would see it on a DOS-line-
    ended file on a system with no newline translation -- survives to be
    stripped by `_remove_carriage_return` exactly where C++ strips it,
    rather than Python's text-mode reader silently normalising it away
    first.
    """
    with open(file_name, newline="") as f:
        content = f.read()
    raw_lines = content.split("\n")
    if raw_lines and raw_lines[-1] == "":
        raw_lines.pop()  # trailing newline at EOF, not a real final line

    lines = []
    for raw in raw_lines:
        if _is_comment(raw):
            continue
        raw = _remove_non_ascii(raw)
        raw = _remove_carriage_return(raw)
        lines.append(raw)
    return lines


def get_header(file_name: str) -> list[str]:
    """C++: moses/comboreduct/table/table_io.cc:395 get_header"""
    lines = _read_data_lines(file_name)
    if not lines:
        return []
    return tokenize_row(lines[0])


# ---------------------------------------------------------------------------
# Tokenizing -- table_io.cc:156 get_row_tokenizer / table_io.h:84 tokenizeRow
# ---------------------------------------------------------------------------


# DIVERGENCE(C++ moses/comboreduct/table/table_io.cc:156): `get_row_tokenizer`
# is a `boost::escaped_list_separator(escape='\\\\', separators=",\\t ",
# quote='"')`. `_split_dense_row` below reimplements its state machine
# (backslash escapes the next character; a `"` toggles a quoted region in
# which comma/tab/blank are literal and the quote characters themselves are
# stripped from the token) rather than delegating to it, so it is verified
# against the fixture corpus and against the C++ probe directly (see
# task-5-report.md's fix-round-1 section) rather than assumed correct.
# Genuinely UNTESTED against boost's exact algorithm (no fixture and no
# probe run exercises these): a literal quote character appearing MID-field
# rather than only at a field's start/end, and doubled quotes (`""`) as an
# RFC4180-style escaped-quote convention -- boost's separator does not
# document that convention and this port does not special-case it either
# (each `"` simply toggles quoting), so the two should agree, but this is
# reasoned from the escape/quote semantics, not probe-verified.
#
# This closes what was PREVIOUSLY documented here as an accepted gap: an
# ordinary `QUOTE_ALL`-dialect CSV (`"1","0","1"`, the default pandas/R/
# Excel export format) was silently mis-parsed as `ill_formed_type` string
# tokens (`'"1"'`) instead of `boolean_type` `true`/`false` -- a loadable
# dataset silently turned into garbage that looked like data. That failure
# mode is what forced implementing this rather than rejecting quoted input.
def _split_dense_row(line: str) -> list[str]:
    """C++: moses/comboreduct/table/table_io.cc:156 get_row_tokenizer

    Character-level port of `boost::escaped_list_separator`'s state
    machine (escape char `\\`, separators `,`, tab, blank, quote char
    `"`) -- see the DIVERGENCE note above for exactly what is and is not
    verified against the real boost implementation.
    """
    tokens: list[str] = []
    cur: list[str] = []
    in_quotes = False
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c == "\\" and i + 1 < n:
            cur.append(line[i + 1])
            i += 2
            continue
        if c == '"':
            in_quotes = not in_quotes
            i += 1
            continue
        if not in_quotes and c in (",", "\t", " "):
            tokens.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    tokens.append("".join(cur))
    return tokens


def tokenize_row(line: str, ignored_indices: Iterable[int] = ()) -> list[str]:
    """C++: moses/comboreduct/table/table_io.h:84 tokenizeRow

    `tokenizeRow<std::string>` -- the checker's symbol matcher does not
    handle template arguments, so the bare function name is cited (the
    only instantiation this port ever needs, string tokens).

    Trims each raw token and DROPS it entirely (not counting it towards
    `ignored_indices`'s positional indexing either) if trimming leaves it
    empty -- table_io.h:93-95's own comment: "Sometimes the tokenizer
    returns pure whitespace :-(". Used for header lines and for the
    handful of `inferTableAttributes` samples; NOT used for per-row
    input/output/timestamp splitting during the real load, which uses
    `_tokenize_row_iot` below and does NOT drop empty tokens -- an
    inconsistency that exists in classic itself (`tokenizeRowIOT`,
    table_io.cc:785, has no trim/drop step), ported faithfully rather
    than smoothed over.
    """
    ignored = set(ignored_indices)
    res = []
    i = 0
    for raw in _split_dense_row(line):
        clean = raw.strip()
        if not clean:
            continue
        if i not in ignored:
            res.append(clean)
        i += 1
    return res


def _tokenize_row_iot(
    line: str,
    ignored_indices: Iterable[int],
    target_idx: int,
    timestamp_idx: int,
) -> tuple[list[str], str, str]:
    """C++: moses/comboreduct/table/table_io.cc:785 tokenizeRowIOT

    Returns `(input_tokens, output_token, timestamp_token)`.
    `output_token`/`timestamp_token` are `""` if `target_idx`/
    `timestamp_idx` is not reached (`timestamp_idx < 0` disables it, same
    as C++). A column index present in `ignored_indices` is dropped
    outright and never considered for target/timestamp routing either --
    same branch nesting as C++ (an ignored column that also happens to
    equal `target_idx` would silently lose its value; not exercised by
    any fixture in this port's corpus, ported as-is rather than
    special-cased away).
    """
    ignored = set(ignored_indices)
    inputs: list[str] = []
    output = ""
    timestamp = ""
    for i, tok in enumerate(_split_dense_row(line)):
        if i in ignored:
            continue
        if target_idx == i:
            output = tok
        elif timestamp_idx == i:
            timestamp = tok
        else:
            inputs.append(tok)
    return inputs, output, timestamp


# ---------------------------------------------------------------------------
# Type inference -- table_io.cc:198-332
# ---------------------------------------------------------------------------


def infer_type_from_token(token: str) -> type_node:
    """C++: moses/comboreduct/table/table_io.cc:198 infer_type_from_token

    C++'s `isalpha` is locale-dependent but, for the "C" locale every
    fixture in this port's corpus is read under, recognises exactly
    `[A-Za-z]`; matched here with an explicit range rather than Python's
    more permissive (Unicode-aware) `str.isalpha()`.
    """
    if token in ("0", "1", "T", "F", "t", "f"):
        return type_node.boolean_type
    if token[:1] and ("a" <= token[0].lower() <= "z"):
        return type_node.enum_type
    try:
        float(token)
        return type_node.contin_type
    except ValueError:
        return type_node.ill_formed_type


def infer_type_from_token2(curr_guess: type_node, token: str) -> type_node:
    """C++: moses/comboreduct/table/table_io.cc:235 infer_type_from_token2"""
    tokt = infer_type_from_token(token)
    if curr_guess == type_node.unknown_type:
        return tokt
    if tokt == curr_guess:
        return tokt
    if curr_guess == type_node.contin_type and tokt == type_node.boolean_type:
        return curr_guess
    if curr_guess == type_node.boolean_type and tokt == type_node.contin_type:
        return tokt
    return type_node.enum_type


def token_to_boolean(token: str) -> builtin:
    """C++: moses/comboreduct/table/table_io.cc:262 token_to_boolean"""
    if token in ("0", "F", "f"):
        return builtin.logical_false
    if token in ("1", "T", "t"):
        return builtin.logical_true
    raise ValueError(f"Expecting boolean value, got {token}")


def token_to_contin(token: str) -> float:
    """C++: moses/comboreduct/table/table_io.cc:273 token_to_contin"""
    try:
        return float(token)
    except ValueError as exc:
        raise ValueError(f"Could not cast {token} to contin") from exc


def token_to_vertex(tipe: type_node, token: str) -> vertex:
    """C++: moses/comboreduct/table/table_io.cc:282 token_to_vertex

    Both `enum_type` and `ill_formed_type` construct `enum_t(token)` in
    C++ (table_io.cc:295-296 and :300 respectively) -- only
    `definite_object_type` returns a bare string (table_io.cc:298,
    `definite_object` being a genuinely distinct `std::string`-typed
    variant alternative, not `enum_t`). Ported with the same distinction:
    see vertex.py's module docstring for why `enum_t` is kept as its own
    (bare `str` subclass) type rather than collapsed into plain `str`.
    """
    if tipe == type_node.boolean_type:
        return token_to_boolean(token)
    if tipe == type_node.contin_type:
        return token_to_contin(token)
    if tipe == type_node.enum_type:
        if token[:1] and ("a" <= token[0].lower() <= "z"):
            return enum_t(token)
        raise ValueError(f"Enum type must begin with alphabetic char, but {token} doesn't")
    if tipe == type_node.definite_object_type:
        return token
    if tipe == type_node.ill_formed_type:
        # C++'s own comment (table_io.cc:298-301): "Ugly hack... feature
        # selection has to read and propagate columns of unknown type
        # (typically, dates, times)." Preserved as an `enum_t` of the raw
        # token string, exactly matching C++'s `return enum_t(token);`.
        return enum_t(token)
    raise ValueError(f'Unable to convert token "{token}" to type={tipe!r}')


def is_header(tokens: Sequence[str], col_types: Sequence[type_node]) -> bool:
    """C++: moses/comboreduct/table/table_io.cc:678 is_header"""
    for i, tok in enumerate(tokens):
        flt = infer_type_from_token2(col_types[i], tok)
        if flt == type_node.enum_type and col_types[i] != type_node.enum_type:
            return True
    return False


# ---------------------------------------------------------------------------
# inferTableAttributes -- table_io.cc:911
# ---------------------------------------------------------------------------

_SPARSE_DELIM = " : "
_MAXLINE = 20


class _InferredAttributes:
    """Return value of `_infer_table_attributes` -- not a ported symbol,
    just the 4-tuple `inferTableAttributes` computes via out-parameters
    in C++ (`type_tree& tt, bool& has_header, bool& is_sparse`, plus the
    return-by-reference `itypes`/`otype` this port keeps separate rather
    than assembling into a `Signature` this early).
    """

    __slots__ = ("itypes", "otype", "has_header", "is_sparse")

    def __init__(
        self, itypes: list[type_node], otype: type_node, has_header: bool, is_sparse: bool
    ) -> None:
        self.itypes = itypes
        self.otype = otype
        self.has_header = has_header
        self.is_sparse = is_sparse


def _infer_table_attributes(
    lines: Sequence[str],
    target_feature: str,
    timestamp_feature: str,
    ignore_features: Sequence[str],
) -> _InferredAttributes:
    """C++: moses/comboreduct/table/table_io.cc:911 inferTableAttributes

    Only the first `_MAXLINE` (20) lines are ever consulted for type
    inference and sparse detection -- ported faithfully, including the
    limitation that a column whose true type only reveals itself after
    line 20 is misclassified exactly as classic would misclassify it.
    """
    sample = lines[:_MAXLINE]
    if any(_SPARSE_DELIM in ln for ln in sample):
        return _InferredAttributes([], type_node.unknown_type, False, True)

    if not sample:
        return _InferredAttributes([], type_node.unknown_type, False, False)

    maybe_header = tokenize_row(sample[0])
    arity = len(maybe_header)
    types = [type_node.unknown_type] * arity
    for i, ln in enumerate(sample[1:], start=2):
        tokens = tokenize_row(ln)
        if len(tokens) != arity:
            raise ValueError(
                f"ERROR: Input file inconsistent: the {i}th row has a different "
                "number of columns than the rest of the file. All rows should "
                "have the same number of columns."
            )
        types = [infer_type_from_token2(t, tok) for t, tok in zip(types, tokens, strict=True)]

    has_header = is_header(maybe_header, types)

    if has_header:
        target_idx = 0
        if target_feature:
            if target_feature not in maybe_header:
                raise ValueError(f"Target {target_feature} not found")
            target_idx = maybe_header.index(target_feature)
        ignore_idxs = set(get_indices(ignore_features, maybe_header))
        ignore_idxs.add(target_idx)
        if timestamp_feature:
            if timestamp_feature not in maybe_header:
                raise ValueError(f"Timestamp feature {timestamp_feature} not found")
            ignore_idxs.add(maybe_header.index(timestamp_feature))
        otype = types[target_idx]
        itypes = [t for i, t in enumerate(types) if i not in ignore_idxs]
    else:
        otype = types[0]
        itypes = types[1:]

    # C++: table_io.cc:1021 `logger().debug() << "Infered type tree: " << tt;`
    # -- classic's own spelling of "Inferred". `tt` is the `gen_signature(
    # itypes, otype)` built just above (table_io.cc:1013/:1017), so it is
    # rebuilt here rather than threaded through `_InferredAttributes`, which
    # deliberately keeps the two halves separate (see its own docstring).
    # Only the DENSE path reaches this: the sparse branch returns at
    # table_io.cc:929-933, before the log.
    logger().debug("Infered type tree: " + ostream_type_signature(gen_signature(itypes, otype)))
    return _InferredAttributes(itypes, otype, has_header, False)


# ---------------------------------------------------------------------------
# Dense loading -- table_io.cc:1094 istreamDenseTable_noHeader,
# table_io.cc:1170 istreamDenseTable
# ---------------------------------------------------------------------------


def _load_dense_table(
    lines: Sequence[str],
    target_feature: str,
    timestamp_feature: str,
    ignore_features: Sequence[str],
    itypes: Sequence[type_node],
    otype: type_node,
    has_header: bool,
) -> Table:
    """C++: moses/comboreduct/table/table_io.cc:1170 istreamDenseTable
    and moses/comboreduct/table/table_io.cc:1094 istreamDenseTable_noHeader
    """
    if not has_header and (target_feature or timestamp_feature or ignore_features):
        raise ValueError(
            "If the data file has no header, then a target feature, ignore "
            "features or timestamp_feature cannot be specified"
        )

    target_idx = 0
    timestamp_idx = -1
    ignore_idxs: list[int] = []
    data_lines: Sequence[str] = lines

    itable = ITable()
    otable = OTable()
    ttable = TTable()

    if has_header:
        header = tokenize_row(lines[0])
        data_lines = lines[1:]

        if target_feature:
            if target_feature not in header:
                raise ValueError(f"Target {target_feature} not found")
            target_idx = header.index(target_feature)

        if timestamp_feature:
            if timestamp_feature not in header:
                raise ValueError(f"Timestamp feature {timestamp_feature} not found")
            timestamp_idx = header.index(timestamp_feature)

        ignore_idxs = get_indices(ignore_features, header)

        in_labels, out_label, ts_label = _tokenize_row_iot(
            lines[0], ignore_idxs, target_idx, timestamp_idx
        )
        itable.set_labels(in_labels)
        otable.set_label(out_label)
        # C++: table_io.cc:1221, `tab.ttable.set_label(std::get<2>(iotlabels))`
        # -- set UNCONDITIONALLY there, even when no timestamp column was
        # requested (in which case `_ts_label` is ""), so the label is not
        # guarded on `timestamp_idx >= 0` here either.
        ttable.set_label(ts_label)

    itable.set_types(list(itypes))
    otable.set_type(otype)

    # C++ pre-sizes both itable and otable to len(data_lines) up front
    # (table_io.cc:1105-1106, `tab.itable.resize(...); tab.otable.resize(
    # ...);`) and only CONDITIONALLY overwrites an otable slot ("if there
    # is no valid target index, then there is no 'output' column!",
    # table_io.cc:1133-1136) -- itable's slot is always overwritten
    # unconditionally, so it never has a genuinely unfilled row in
    # practice, but otable's can. Pre-sizing otable here with the C++
    # default value (`vertex()`'s first boost::variant alternative,
    # `contin_t() == 0.0`) and only overwriting non-empty output tokens
    # keeps itable/otable/row-count in lockstep even when a target token
    # is legitimately empty -- e.g. a blank trailing field. Appending
    # conditionally instead (as an earlier version of this port did)
    # silently drops that row's otable slot entirely, desynchronising
    # itable and otable by one and surfacing far downstream as an opaque
    # `zip() argument 2 is shorter` from `Table.compressed()`.
    otable_values: list[vertex] = [0.0] * len(data_lines)
    # C++: table_io.cc:1109-1110, `if (timestamp_idx >= 0) tab.ttable.
    # resize(lines.size());` -- the ttable is pre-sized ONLY when a
    # timestamp column was requested, and each slot is then overwritten
    # only `if ("" != date_str)` (table_io.cc:1141-1144), so a row whose
    # timestamp token is empty keeps the default-constructed
    # (not-a-date) value rather than being dropped. `None` is that value
    # here (table.py's `timestamp_t`).
    ttable_values: list = [None] * len(data_lines) if timestamp_idx >= 0 else []
    for i, line in enumerate(data_lines):
        inputs, output_str, timestamp_str = _tokenize_row_iot(
            line, ignore_idxs, target_idx, timestamp_idx
        )
        row = tuple(token_to_vertex(t, tok) for t, tok in zip(itypes, inputs, strict=True))
        itable.push_back(row)
        if output_str != "":
            otable_values[i] = token_to_vertex(otype, output_str)
        if timestamp_idx >= 0 and timestamp_str != "":
            ttable_values[i] = TTable.from_string(timestamp_str)
    otable.extend(otable_values)
    ttable.extend(ttable_values)

    table = Table(otable, itable)
    # `Table.__init__` COPIES itable/otable (table.cc:484) but has no
    # ttable parameter at all -- C++ fills `tab.ttable` in place
    # (table_io.cc:1144), so this assignment mirrors that rather than
    # widening the constructor.
    table.ttable = ttable
    ignore_before_target = sum(1 for i in ignore_idxs if i < target_idx)
    table.target_pos = target_idx - ignore_before_target
    # C++: table_io.cc:1163-1165 -- guarded on `timestamp_idx >= 0`, so a
    # table with no timestamp column keeps `timestamp_pos == 0` rather
    # than the meaningless `-1 - <count>`.
    if timestamp_idx >= 0:
        table.timestamp_pos = timestamp_idx - sum(1 for i in ignore_idxs if i < timestamp_idx)
    return table


# ---------------------------------------------------------------------------
# Sparse loading -- table_io.cc:138 parse_key_val, :182 tokenizeSparseRow,
# :530 istreamSparseITable, :856 istreamTable_OLD
# ---------------------------------------------------------------------------


def _parse_key_val(chunk: str) -> tuple[str, str] | None:
    """C++: moses/comboreduct/table/table_io.cc:138 parse_key_val

    Returns `None` where C++ returns a default-constructed (empty, empty)
    pair -- i.e. `chunk` does not contain the `" : "` delimiter.
    """
    pos = chunk.find(_SPARSE_DELIM)
    if pos == -1:
        return None
    key = chunk[:pos].strip()
    val = chunk[pos + len(_SPARSE_DELIM) :].strip()
    return key, val


def _tokenize_sparse_row(line: str) -> list[str]:
    """C++: moses/comboreduct/table/table_io.cc:182 tokenizeSparseRow

    Comma-only splitting (`get_sparse_row_tokenizer`, table_io.cc:167,
    unlike the dense tokenizer does not also split on tab/blank -- sparse
    rows use `", "` after each comma, and the values themselves may
    contain blanks, e.g. `"12-06-2012T14:35:02-0500"`).
    """
    return [t.strip() for t in line.split(",")]


def _load_sparse_itable(lines: Sequence[str]) -> ITable:
    """C++: moses/comboreduct/table/table_io.cc:530 istreamSparseITable"""
    header = lines[0]
    labs: list[str] = []
    fixed_arity = 0
    data_lines: Sequence[str] = lines
    if _SPARSE_DELIM not in header:
        hdr_tokens = _tokenize_sparse_row(header)
        fixed_arity = len(hdr_tokens)
        labs = hdr_tokens
        data_lines = lines[1:]

    if fixed_arity == 0:
        fixy = _tokenize_sparse_row(data_lines[0])
        while _SPARSE_DELIM not in fixy[fixed_arity]:
            fixed_arity += 1
    logger().info(f"Sparse file fixed column count={fixed_arity}")

    feats: set[str] = set()
    feat_type = type_node.unknown_type
    types = [type_node.unknown_type] * fixed_arity

    parsed_rows: list[list[str]] = []
    for line in data_lines:
        chunks = _tokenize_sparse_row(line)
        parsed_rows.append(chunks)

        for i in range(fixed_arity):
            types[i] = infer_type_from_token2(types[i], chunks[i])

        for chunk in chunks[fixed_arity:]:
            kv = _parse_key_val(chunk)
            if kv is None:
                break
            key, val = kv
            feats.add(key)
            feat_type = infer_type_from_token2(feat_type, val)

    # C++: table_io.cc:591-593. `lines.size()` there is the SAME `lines`
    # this port calls `data_lines`: the C++ function strips the header off
    # its own `lines` vector before this point (table_io.cc:544-552), so the
    # count excludes it in both.
    logger().info(f"Sparse file unique features count={len(feats)}")
    logger().info("Sparse file feature type=" + ostream_type_node(feat_type))
    logger().info(f"Sparse file row count={len(data_lines)}")

    index: dict[str, int] = {}
    cnt = fixed_arity
    for key in sorted(feats):  # std::set<string> iterates lexicographically
        types.append(feat_type)
        labs.append(key)
        index[key] = cnt
        cnt += 1

    itable = ITable()
    itable.set_labels(labs)
    itable.set_types(types)

    all_boolean = all(t == type_node.boolean_type for t in types)
    all_contin = all(t == type_node.contin_type for t in types)

    for chunks in parsed_rows:
        if all_boolean:
            row: list[vertex] = [builtin.logical_false] * len(types)
            for i in range(fixed_arity):
                row[i] = token_to_boolean(chunks[i])
            for chunk in chunks[fixed_arity:]:
                kv = _parse_key_val(chunk)
                if kv is None:
                    continue
                key, val = kv
                row[index[key]] = token_to_boolean(val)
        elif all_contin:
            row = [0.0] * len(types)
            for i in range(fixed_arity):
                row[i] = token_to_contin(chunks[i])
            for chunk in chunks[fixed_arity:]:
                kv = _parse_key_val(chunk)
                if kv is None:
                    continue
                key, val = kv
                row[index[key]] = token_to_contin(val)
        else:
            # vertex_seq() default-constructs every slot to vertex()'s
            # first boost::variant alternative, contin_t() == 0.0 -- see
            # from_sparse_tokens_visitor's `else` branch, table_io.cc:466.
            row = [0.0] * len(types)
            for i in range(fixed_arity):
                row[i] = token_to_vertex(types[i], chunks[i])
            for chunk in chunks[fixed_arity:]:
                kv = _parse_key_val(chunk)
                if kv is None:
                    continue
                key, val = kv
                idx = index[key]
                row[idx] = token_to_vertex(types[idx], val)
        itable.push_back(tuple(row))

    return itable


def _load_table_old(
    lines: Sequence[str],
    target_feature: str,
    ignore_features: Sequence[str],
) -> Table:
    """C++: moses/comboreduct/table/table_io.cc:856 istreamTable_OLD

    `istreamTable_OLD` never calls the `Table(const OTable&, const
    ITable&)` constructor (table.cc:484, which copies -- see
    `Table.__init__`) -- it takes an already-constructed `Table& tab` and
    fills `tab.itable` IN PLACE (`istreamITable(in, tab.itable,
    ignore_features)`, table_io.cc:860, passes `tab.itable` by
    reference), then mutates that same `itable` further in place
    (`tab.itable.delete_column(target_feature)`, table_io.cc:871) before
    finally building `tab.otable` from it. Mirrored here the same way:
    construct an empty `Table`, assign `table.itable` directly (bypassing
    the copying constructor), and do every subsequent read/mutation
    through `table.itable` itself rather than through a separate local
    `itable` variable -- the shape a naive `Table(OTable(...), itable)`
    call at the end would need the constructor to alias to work, which it
    no longer does.
    """
    table = Table()
    table.itable = _load_sparse_itable(lines)
    table.itable.delete_columns(list(ignore_features))

    otable_values = table.itable.get_column_data(target_feature)
    if len(otable_values) == 0:
        raise ValueError(f'Fatal Error: target feature "{target_feature}" not found')

    table.otable = OTable(otable_values)
    table.target_pos = table.itable.get_column_offset(target_feature)
    targ_type = table.itable.get_type(target_feature)
    targ_label = table.itable.delete_column(target_feature)
    table.otable.set_label(targ_label)
    table.otable.set_type(targ_type)
    return table


# ---------------------------------------------------------------------------
# loadTable -- table_io.cc:1296
# ---------------------------------------------------------------------------


def loadTable(
    file_name: str,
    target_feature: str = "",
    timestamp_feature: str = "",
    ignore_features: Sequence[str] = (),
) -> Table:
    """C++: moses/comboreduct/table/table_io.cc:1296 loadTable

    See the module docstring for what this settles about column ordering
    with ignored columns, a non-last target, and target-by-name vs.
    target-by-position.
    """
    if not file_name:
        raise ValueError("the file name is empty")

    lines = _read_data_lines(file_name)
    attrs = _infer_table_attributes(lines, target_feature, timestamp_feature, ignore_features)

    if attrs.is_sparse:
        # C++: istreamTable, table_io.cc:1046-1050.
        if timestamp_feature:
            raise ValueError("Timestamp feature not implemented")
        return _load_table_old(lines, target_feature, ignore_features)

    return _load_dense_table(
        lines,
        target_feature,
        timestamp_feature,
        ignore_features,
        attrs.itypes,
        attrs.otype,
        attrs.has_header,
    )
