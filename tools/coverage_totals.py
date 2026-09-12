#!/usr/bin/env python3
"""Regenerate (or verify) the summary tables in ``docs/MODULE-COVERAGE.md``.

The inventory's per-file rows are hand-maintained -- they carry a reason and a
named gap, which no script can write. The two aggregate tables at the top are
not: they are a pure function of those rows, and hand-editing them is how they
drift. Every prior cycle left them stale, and the S7 branch review's single
unmet exit item was exactly that.

Hand-editing also fails silently in a specific way worth naming. When this
file's logic was first run ad hoc, its status list omitted four rows that had
just been re-slugged; the per-status counts still *looked* fine and the printed
Total was still 243, because the total was summed over all rows while the table
listed only the known ones. The table summed to 239. That is why
``_aggregate`` asserts the listed statuses partition the rows, rather than
trusting an enumeration written by hand:

    assert set(order) == set(agg)
    assert sum(counts) == total

Usage::

    python3 tools/coverage_totals.py            # rewrite the two tables in place
    python3 tools/coverage_totals.py --check    # exit 1 if they are stale

``--check`` is the CI-shaped form: it never writes, so it is safe to run on a
dirty tree, and it fails loudly when someone edits a row without regenerating.

Beyond the tables, this tool runs three ROW-vs-TREE staleness checks. They are
described in full on the functions themselves; the short version is:

* ``check_planned_rows_against_reality`` -- PLANNED row whose named module
  already exists (forward direction: a slice landed and closeout was skipped).
* ``check_named_modules_exist`` -- PORTED/PARTIAL/EXCLUDED row naming a module
  that is NOT on disk (first inverse direction: a rename or deletion).
* ``check_ported_rows_do_not_gate`` -- PORTED row whose own C++ file still has a
  ported symbol that raises ``NotImplementedError`` (second inverse direction:
  the "marked PORTED but actually a gate that raises" defect).

``check_partial_rows_whose_gap_closed`` does NOT exist, deliberately. See the
module-level note ``WHY_NO_PARTIAL_TO_PORTED_CHECK`` below.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import OrderedDict
from pathlib import Path

DOC = Path(__file__).resolve().parent.parent / "docs" / "MODULE-COVERAGE.md"

# The inventory's own invariant: every file under $CLASSIC/moses/ (240, by
# `git ls-files`) plus the CogUtil files pyMoses ports. That CogUtil set GROWS
# as slices reach new dependencies -- S9 added lazy_random_selector.{h,cc}
# (+2 files, +97 lines), taking the total from 243/62,047 to 245/62,144. S11
# Task 1 added selection.h (+1 file, +155 lines), taking the total to
# 246/62,299. The 2026-09-09 reconciliation added digraph.h (+1 file, +124
# lines) -- already ported, with its own module and tests, and never given a
# row -- taking the total to 247/62,423. Update these two constants, with the
# reason, whenever a CogUtil file is newly ported.
#
# Both halves were re-measured on 2026-09-09 and hold exactly:
#   $CLASSIC:  git ls-files 'moses/*' | grep -E '\.(cc|h|cpp|hpp)$' -> 240 files
#              ... | xargs wc -l                                     -> 58,449 lines
#   $COGUTIL:  wc -l on the 7 rowed util files                       -> 3,974 lines
#   240 + 7 = 247;  58,449 + 3,974 = 62,423.
# That measurement is NOT automated here: doing so would make `--check` depend
# on the two classic checkouts being present, and the aggregate assertion below
# only proves the ROWS sum to these constants, not that the constants match
# upstream. A file added upstream would go unnoticed. See MODULE-COVERAGE.md's
# CogUtil section for the eleven CogUtil files that have breadcrumbs but no row.
EXPECTED_FILES = 247
EXPECTED_LINES = 62423

# Display order. `_aggregate` proves this is exhaustive rather than assuming it.
STATUS_ORDER = [
    "PORTED",
    "PARTIAL",
    "PLANNED: P2",
    "EXCLUDED",
]

# Statuses allowed to have ZERO rows, and why each one is.
#
# `_aggregate` treats the two directions of a STATUS_ORDER/row mismatch very
# differently, because only one of them can corrupt the table:
#
# * A row status NOT in STATUS_ORDER stays fatal. Such a row is summed into the
#   printed Total while no table line accounts for it, so the table under-sums
#   in silence -- the exact failure this module's docstring opens with.
# * A STATUS_ORDER entry with no rows corrupts nothing. It renders as a `0 | 0`
#   line. Refusing it only forces the display list to be hand-edited whenever a
#   bucket empties, and hand-editing the display list is how a status gets
#   dropped in the first place.
#
# `PLANNED: P2` emptied on 2026-09-11: the last PLANNED row -- `type_tree_def.cc`,
# 41 lines, `is_argument_type` and `arg_to_idx` -- was found already ported at
# `pymoses/comboreduct/type_checker/type_signature.py:115,:126`. Rendering the
# bucket at zero states that milestone in the table instead of hiding it, and
# leaves the PLANNED machinery (AWAITING_RE / FOLDS_INTO_RE and
# `check_planned_rows_against_reality`) armed for the next PLANNED row anyone
# adds. Every other status must still be backed by at least one real row.
MAY_BE_EMPTY = {"PLANNED: P2"}

ROW_RE = re.compile(r"^\| `([^`]+)` \| *([\d,]+) \| ([A-Z][A-Za-z0-9: ]*?) \|", re.M)

# The same row, with the reason column captured. Every staleness check below
# parses rows with this; ROW_RE (no reason) stays as the aggregation regex so a
# reason containing a `|` can never silently drop a row from the totals.
FULL_ROW_RE = re.compile(
    r"^\| `(?P<cpp>[^`]+)` \| *(?P<lines>[\d,]+) \| "
    r"(?P<status>[A-Z][A-Za-z0-9: ]*?) \| (?P<reason>[^\n]*)$",
    re.M,
)

# A row names its Python module by path, in the reason column. Backticks are
# optional because ~40 rows predate the convention and write the path bare.
MODULE_RE = re.compile(r"`?(pymoses/[\w/]+\.py)`?")

# A PLANNED row must declare which module it is waiting for, with one of two
# explicit markers. Bare `pymoses/...py` mentions elsewhere in the reason are
# CONTEXT, not the declaration -- a PLANNED row routinely points at a
# neighbouring ported module ("see the mutual_info.py row"), and treating those
# as the awaited module made `eval-features.h` report itself stale against a
# module it merely cites.
#
#   AWAITING `pymoses/x.py`    -- a module that does not exist yet. Flagged the
#                                 day it appears: that is the closeout the
#                                 forward check exists to catch.
#   FOLDS INTO `pymoses/x.py`  -- the C++ file's symbols would land INSIDE a
#                                 module that already exists (e.g.
#                                 `type_tree_def.cc`'s two functions belong in
#                                 `type_tree.py`). Existence proves nothing
#                                 here, so the test inverts: the host module
#                                 must be there, and staleness has to be found
#                                 by reading.
AWAITING_RE = re.compile(r"AWAITING `?(pymoses/[\w/]+\.py)`?")
FOLDS_INTO_RE = re.compile(r"FOLDS INTO `?(pymoses/[\w/]+\.py)`?")

# The project-wide breadcrumb convention (see tools/check_alignment.py): the
# FIRST line of a ported symbol's docstring is
# `C++: <path under a classic checkout>:<line> <C++ qualified symbol>`.
# `check_ported_rows_do_not_gate` uses it to attribute a raise to one C++ file
# rather than to a whole Python module, which is what makes that check exact.
BREADCRUMB_RE = re.compile(r"^C\+\+: (?P<file>\S+):(?P<line>\d+) ")

REPO_ROOT = DOC.parent.parent

WHY_NO_PARTIAL_TO_PORTED_CHECK = """\
Why there is no automated "this PARTIAL row's gap has closed" check.

Asked for, attempted, and rejected as unimplementable WITHOUT FALSE POSITIVES
during the 2026-09-09 coverage reconciliation. Recording the reasoning so the
next person does not read the absence as "already covered".

The two checks that DO exist both rest on a fact that is decidable from the
tree: a file exists or it does not; a symbol raises or it does not. "The gap
named in this row has closed" is not that kind of fact, for three measured
reasons:

1. **Most gaps are absences, not raises.** A PARTIAL row's gap is usually a C++
   entity with NO Python counterpart at all -- `contin_interpreter` before S14,
   `interesting_predicate_bscore` today. Nothing in the tree changes shape when
   such a gap closes except the appearance of code that no rule can recognise as
   "the thing that row meant". Measured on the 2026-09-09 row set: of the
   PARTIAL rows, only a minority name a gap that manifests as a
   `raise NotImplementedError`; the rest name unported functions, unported
   overloads, or narrowed grammars.

2. **The inverse of check 3 is unsound, not merely weak.** "PARTIAL row whose
   module no longer raises for its C++ file => promote to PORTED" fires on every
   row in category 1 from the day it is written, because those rows never had a
   raise to lose. It would report ~50 promotions on a correct tree.

3. **Narrowing it to rows whose reason literally says "raises" does not fix
   it.** A gap can move from a breadcrumbed method into an un-breadcrumbed
   helper, or from `raise NotImplementedError` into a `ValueError` (this project
   maps one C++ construct, `OC_ASSERT(false)`, to six different Python exception
   types -- CARRY-FORWARD Sec 5), and the check would then claim the gap closed
   when only its spelling changed. A false "this is done" is the exact defect
   class this tool exists to prevent; a check that produces one is worse than no
   check.

What closes the loop instead is a human re-reading the row's named gap against
the tree, which is what the 2026-09-09 audit did for all 133 PLANNED/PARTIAL
rows. The three automated checks catch the mechanical half; the semantic half
is scheduled work, not tooling.
"""

SUMMARY_RE = re.compile(
    r"\| Status \| Files \| Lines \| % of files \| % of lines \|\n"
    r"\|---\|---:\|---:\|---:\|---:\|\n(?:\|[^\n]*\n)+"
)
MACRO_RE = re.compile(
    r"\| Macro status \| Files \| Lines \|\n\|---\|---:\|---:\|\n(?:\|[^\n]*\n)+"
)
# The "Reading this table" paragraph restates the same aggregates in prose. It
# was hand-written and therefore drifted exactly like the tables once did: at
# the 2026-09-09 reconciliation it still said "the completed S0-S14 footprint
# plus most of the P2 tail" over numbers from an earlier row set. Every NUMBER
# in it is now generated. What the buckets CONTAIN stays hand-written, in the
# separate paragraph below it -- that is a judgement, not an aggregate.
READING_RE = re.compile(r"^\*\*Reading this table\.\*\* [^\n]*$", re.M)


def _aggregate(text: str) -> tuple[OrderedDict[str, tuple[int, int]], int, int]:
    """Fold the per-file rows into (files, lines) per status.

    Asserts the result is a partition: `STATUS_ORDER` must name every status
    present and no others, and the per-status counts must sum to the totals.
    Without the first assertion a status absent from `STATUS_ORDER` is dropped
    from the rendered table while still counting toward the printed total, so
    the table under-sums and nothing complains.
    """
    agg: OrderedDict[str, tuple[int, int]] = OrderedDict()
    for _path, raw_lines, status in ROW_RE.findall(text):
        status = status.strip()
        files, lines = agg.get(status, (0, 0))
        agg[status] = (files + 1, lines + int(raw_lines.replace(",", "")))

    total_files = sum(v[0] for v in agg.values())
    total_lines = sum(v[1] for v in agg.values())

    unknown = set(agg) - set(STATUS_ORDER)
    missing = set(STATUS_ORDER) - set(agg) - MAY_BE_EMPTY
    if unknown or missing:
        raise SystemExit(
            f"status mismatch: rows use {sorted(unknown)} which STATUS_ORDER omits; "
            f"STATUS_ORDER names {sorted(missing)} which no row uses. "
            "Rendering would silently drop rows from the table."
        )
    for status in MAY_BE_EMPTY:
        agg.setdefault(status, (0, 0))
    agg = OrderedDict((s, agg[s]) for s in STATUS_ORDER)
    if sum(agg[k][0] for k in STATUS_ORDER) != total_files:
        raise SystemExit("per-status file counts do not sum to the total")
    if (total_files, total_lines) != (EXPECTED_FILES, EXPECTED_LINES):
        raise SystemExit(
            f"inventory drifted: {total_files} files / {total_lines} lines, "
            f"expected {EXPECTED_FILES} / {EXPECTED_LINES}"
        )
    return agg, total_files, total_lines


def _render(agg, total_files: int, total_lines: int) -> tuple[str, str]:
    rows = [
        "| Status | Files | Lines | % of files | % of lines |",
        "|---|---:|---:|---:|---:|",
    ]
    for status in STATUS_ORDER:
        files, lines = agg[status]
        rows.append(
            f"| {status} | {files} | {lines:,} | "
            f"{files / total_files * 100:.1f}% | {lines / total_lines * 100:.1f}% |"
        )
    rows.append(
        f"| **Total** | **{total_files}** | **{total_lines:,}** | 100.0% | 100.0% |"
    )
    # Percentages are rounded independently, so a column can sum to 99.9 or
    # 100.1. That is inherent to per-row rounding; the Total row states the
    # exact counts, which are what the invariant is checked against.

    named = [k for k in STATUS_ORDER if k.startswith("PLANNED: S")]
    named_files = sum(agg[k][0] for k in named)
    named_lines = sum(agg[k][1] for k in named)
    macro = "\n".join(
        [
            "| Macro status | Files | Lines |",
            "|---|---:|---:|",
            f"| PORTED | {agg['PORTED'][0]} | {agg['PORTED'][1]:,} |",
            f"| PARTIAL | {agg['PARTIAL'][0]} | {agg['PARTIAL'][1]:,} |",
            f"| PLANNED: S9-S14 (named backlog slices) | {named_files} | {named_lines:,} |",
            f"| PLANNED: P2 (deferred tail) | {agg['PLANNED: P2'][0]} "
            f"| {agg['PLANNED: P2'][1]:,} |",
            f"| EXCLUDED | {agg['EXCLUDED'][0]} | {agg['EXCLUDED'][1]:,} |",
        ]
    )
    return "\n".join(rows), macro


def _render_reading(agg, total_files: int, total_lines: int) -> str:
    """Regenerate the numeric sentence of the "Reading this table" paragraph.

    Same reason the tables are generated: it restates the aggregates, and a
    restatement maintained by hand is a restatement that goes stale. It did --
    see READING_RE. Only arithmetic lives here; the characterisation of what
    each bucket contains is a separate, hand-written paragraph in the document.
    """
    ported_f, ported_l = agg["PORTED"]
    partial_f, partial_l = agg["PARTIAL"]
    excl_f, excl_l = agg["EXCLUDED"]
    have_f, have_l = ported_f + partial_f, ported_l + partial_l
    rest_f = total_files - have_f - excl_f
    rest_l = total_lines - have_l - excl_l
    return (
        f"**Reading this table.** PORTED ({ported_f} files, {ported_l:,} lines) and "
        f"PARTIAL ({partial_f} files, {partial_l:,} lines) together are the C++ that has "
        f"Python behind it: **{have_f} of {total_files} files / {have_l:,} of "
        f"{total_lines:,} lines ({have_l / total_lines * 100:.1f}%)**, of which {ported_f} "
        f"files are complete with no named gap. The PARTIAL bucket holds most of the real "
        f"code, so its per-row named gaps -- not the headline percentage -- are the honest "
        f"reading. EXCLUDED is {excl_f} files / {excl_l:,} lines "
        f"({excl_l / total_lines * 100:.1f}%). What has no Python at all is {rest_f} files "
        f"/ {rest_l:,} lines ({rest_l / total_lines * 100:.1f}%). "
        f"This sentence is GENERATED by tools/coverage_totals.py -- do not hand-edit its "
        f"numbers; the paragraph below it is the hand-written half."
    )


def check_planned_rows_against_reality(text: str) -> list[str]:
    """Flag rows still marked PLANNED whose Python module already exists.

    `_aggregate` reconciles the summary tables against the per-file rows. It
    cannot notice that a ROW ITSELF is stale, because it never looks outside
    this document -- so a slice can land, its files can be ported, and every
    check here still passes while the inventory says PLANNED.

    That is not hypothetical: it happened for S9 and S10 (both merged to main
    with their closeout undone) and again for S11's `moses_main`, which the
    branch review caught. Three times, and `--check` was green each time.

    This closes the loop cheaply. A PLANNED row names its intended Python
    module in backticks (`pymoses/...py`); if that file exists on disk, the row
    is stale. The reverse directions are now covered too, by
    `check_named_modules_exist` and `check_ported_rows_do_not_gate`.

    Sharp edge, measured 2026-09-09: this check was VACUOUS for its whole first
    life. Not one PLANNED row named a `pymoses/...py` module -- PLANNED reasons
    were written as prose ("PLANNED: P2 -- combo-tree similarity metric ..."),
    so the loop below never had a path to test, and 20 rows whose modules had
    landed sailed through green. The fix is a convention, enforced by
    `check_planned_rows_name_a_module` below: a PLANNED row MUST name the
    `pymoses/...py` module it is waiting for. Without that, this function is a
    guard over an empty set.
    """
    problems: list[str] = []
    for m in FULL_ROW_RE.finditer(text):
        status = m.group("status").strip()
        if not status.startswith("PLANNED"):
            continue
        reason = m.group("reason")
        for mod in dict.fromkeys(FOLDS_INTO_RE.findall(reason)):
            # Inverted for this shape: the host module must ALREADY be there.
            if not (REPO_ROOT / mod).exists():
                problems.append(
                    f"{DOC.name}: `{m.group('cpp')}` is {status} and says it FOLDS INTO "
                    f"{mod}, which does not exist. Name a module that is really there, "
                    f"or use AWAITING for a module still to be created."
                )
        for mod in dict.fromkeys(AWAITING_RE.findall(reason)):
            if (REPO_ROOT / mod).exists():
                problems.append(
                    f"{DOC.name}: `{m.group('cpp')}` is {status} "
                    f"but AWAITING {mod}, which already exists -- the row is stale, or "
                    f"the port landed inside it and the row should say FOLDS INTO."
                )
    return problems


def check_planned_rows_name_a_module(text: str) -> list[str]:
    """Require every PLANNED row to name the `pymoses/...py` it is waiting for.

    Exists only to keep `check_planned_rows_against_reality` from being
    vacuous, which is what it was until 2026-09-09 (see that function). A
    PLANNED row that names no module is unfalsifiable: nothing on disk can ever
    contradict it, so the slice can land and the row stays green forever.

    The named path is a PREDICTION, not a claim -- it is the module the porting
    work is expected to create. That is exactly why it must be written down: the
    prediction is what makes the row checkable the day it comes true.
    """
    problems: list[str] = []
    for m in FULL_ROW_RE.finditer(text):
        if not m.group("status").strip().startswith("PLANNED"):
            continue
        reason = m.group("reason")
        if not (AWAITING_RE.search(reason) or FOLDS_INTO_RE.search(reason)):
            problems.append(
                f"{DOC.name}: `{m.group('cpp')}` is {m.group('status').strip()} but its "
                f"reason declares neither AWAITING `pymoses/...py` (a module still to be "
                f"created) nor FOLDS INTO `pymoses/...py` (an existing module the symbols "
                f"belong in). Without one, check_planned_rows_against_reality can never "
                f"fire for this row."
            )
    return problems


def check_named_modules_exist(text: str) -> list[str]:
    """Flag a non-PLANNED row naming a `pymoses/...py` that is NOT on disk.

    The first of the two INVERSE checks -- the direction
    `check_planned_rows_against_reality` explicitly declines to cover. Its
    docstring justifies that with "a row may legitimately describe a symbol
    folded into a differently-named module", which is true but is an argument
    for rows that name NO module. It is not an argument for a row that names a
    specific path which does not exist: that row makes a concrete, checkable,
    false claim about the tree, and a reader following the path finds nothing.

    Catches, concretely: a module renamed without updating the inventory, a
    module deleted, and a row promoted to PORTED against a module that was
    planned but never written.

    Zero heuristics, so zero false positives: it compares a literal path in the
    document against `Path.exists()`. Measured clean over all 107 module-naming
    rows at the time it was added.
    """
    problems: list[str] = []
    for m in FULL_ROW_RE.finditer(text):
        status = m.group("status").strip()
        if status.startswith("PLANNED"):
            continue  # the forward check owns this direction
        for mod in dict.fromkeys(MODULE_RE.findall(m.group("reason"))):
            if not (REPO_ROOT / mod).exists():
                problems.append(
                    f"{DOC.name}: `{m.group('cpp')}` is {status} and names {mod}, "
                    f"which does not exist. The module was renamed or removed, or the "
                    f"row names the wrong path."
                )
    return problems


def _gates_in(module: Path) -> list[tuple[int, str | None, bool, str]]:
    """Every `raise NotImplementedError` in a module, attributed to a C++ file.

    Returns `(lineno, cpp_file_or_None, is_abstractmethod, python_qualname)`.

    `cpp_file` is taken from the nearest ENCLOSING `def`/`class` (innermost
    outward) whose docstring opens with this project's standard breadcrumb.
    That attribution is what makes `check_ported_rows_do_not_gate` exact
    instead of a heuristic: several Python modules host code ported from more
    than one C++ file, and a module-level "does it contain NotImplementedError"
    test therefore blames the wrong row. Two measured examples from the tree
    this was written against:

    * `pymoses/moses/main/problem_params.py` is named by four rows -- two
      PORTED (`moses_exec_def.{cc,h}`) and two PARTIAL (`problem-params.{cc,h}`).
      All eleven of its raises breadcrumb to `problem-params.cc`, so the PORTED
      rows are correctly left alone.
    * `pymoses/moses/main/problem.py` is named only by PORTED rows and does
      contain a non-abstract raise -- but it sits in `_unported_table_problem`,
      a port-invented stand-in with no breadcrumb, so it attributes to nothing
      and blames no row.

    `is_abstractmethod` is reported so the caller can skip a raise that is the
    body of an `@abstractmethod`: C++'s `virtual ... = 0;` has no body at all,
    and a Python abstract method that raises is the faithful port of it, not a
    gap. Without this exemption `moses/moses/main/problem.h` -- a genuinely
    complete port -- is flagged three times over `problem_base`'s three pure
    virtuals.
    """
    try:
        tree = ast.parse(module.read_text())
    except (OSError, SyntaxError):
        return []  # check_named_modules_exist owns "missing"; ruff owns "broken"

    def _breadcrumb(node: ast.AST) -> str | None:
        doc = ast.get_docstring(node, clean=False)
        if not doc:
            return None
        found = BREADCRUMB_RE.match(doc.strip())
        return found.group("file") if found else None

    def _is_abstract(node: ast.AST) -> bool:
        for dec in getattr(node, "decorator_list", []):
            name = dec.attr if isinstance(dec, ast.Attribute) else getattr(dec, "id", None)
            if name == "abstractmethod":
                return True
        return False

    out: list[tuple[int, str | None, bool, str]] = []

    def walk(node: ast.AST, chain: list[ast.AST]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                walk(child, chain + [child])
                continue
            if isinstance(child, ast.Raise):
                exc = child.exc
                if isinstance(exc, ast.Call):
                    exc = exc.func
                if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                    cpp = next(
                        (b for b in (_breadcrumb(a) for a in reversed(chain)) if b), None
                    )
                    out.append(
                        (
                            child.lineno,
                            cpp,
                            bool(chain) and _is_abstract(chain[-1]),
                            ".".join(getattr(a, "name", "?") for a in chain),
                        )
                    )
            walk(child, chain)

    walk(tree, [])
    return out


def check_ported_rows_do_not_gate(text: str) -> list[str]:
    """Flag a PORTED row whose OWN C++ file still has a symbol that raises.

    The second INVERSE check, and the one aimed squarely at this project's
    most-repeated documentation defect: a row marked PORTED whose Python is
    really a gate that raises `NotImplementedError`. `_aggregate` cannot see it
    (it never leaves the document) and `check_planned_rows_against_reality`
    cannot see it (it only looks at PLANNED rows), so before this existed the
    only thing standing between the inventory and a false PORTED claim was
    someone remembering to grep.

    The rule: for a row `PORTED` on C++ file X naming module M, flag every
    `raise NotImplementedError` in M whose nearest enclosing breadcrumbed symbol
    cites X, unless that symbol is an `@abstractmethod`. See `_gates_in` for why
    the breadcrumb attribution and the abstract-method exemption are both
    load-bearing rather than defensive.

    Precision, measured on the tree it was written against: 3 flags, all true
    positives (`opencog/util/mt19937ar.cc` was marked PORTED while
    `MT19937RandGen.randfloat`, `.randdouble_one_excluded` and `.rand_discrete`
    all raise -- and `randfloat` had since become reachable from three live call
    sites in `eval.py`, `interpreter.py` and `feature_selector.py`). Zero false
    positives across the other 29 PORTED rows.

    Deliberately NOT suppressible. A raise that is genuinely faithful -- an
    excluded domain, an upstream `OC_ASSERT` -- makes the row PARTIAL-with-a-
    named-gap under this project's own rule ("a row is PORTED only if the whole
    C++ file is ported"), and PARTIAL is not flagged. There is therefore no case
    that needs a waiver, and adding a waiver list would reopen the exact hole
    this closes.
    """
    problems: list[str] = []
    cache: dict[str, list[tuple[int, str | None, bool, str]]] = {}
    for m in FULL_ROW_RE.finditer(text):
        if m.group("status").strip() != "PORTED":
            continue
        cpp = m.group("cpp")
        for mod in dict.fromkeys(MODULE_RE.findall(m.group("reason"))):
            if mod not in cache:
                cache[mod] = _gates_in(REPO_ROOT / mod)
            for lineno, attributed, abstract, qualname in cache[mod]:
                if abstract or attributed != cpp:
                    continue
                problems.append(
                    f"{DOC.name}: `{cpp}` is PORTED but {mod}:{lineno} "
                    f"({qualname}) raises NotImplementedError for it. A row is PORTED "
                    f"only if the whole C++ file is ported -- make it PARTIAL and name "
                    f"the gap."
                )
    return problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the tables match the rows; write nothing, exit 1 if stale",
    )
    args = parser.parse_args(argv)

    text = DOC.read_text()
    agg, total_files, total_lines = _aggregate(text)
    stale = (
        check_planned_rows_against_reality(text)
        + check_planned_rows_name_a_module(text)
        + check_named_modules_exist(text)
        + check_ported_rows_do_not_gate(text)
    )
    for problem in stale:
        print(problem)

    summary, macro = _render(agg, total_files, total_lines)
    reading = _render_reading(agg, total_files, total_lines)

    updated = SUMMARY_RE.sub(lambda _m: summary + "\n", text, count=1)
    updated = MACRO_RE.sub(lambda _m: macro + "\n", updated, count=1)
    updated, n = READING_RE.subn(lambda _m: reading, updated, count=1)
    if n != 1:
        raise SystemExit(
            "the '**Reading this table.**' paragraph is missing or has been split "
            "across lines; READING_RE can no longer find it, so its numbers would "
            "silently go unregenerated."
        )
    if updated == text:
        print(f"{DOC.name}: summary tables up to date ({total_files} files)")
        return 1 if stale else 0
    if args.check:
        print(f"{DOC.name}: summary tables are STALE -- run tools/coverage_totals.py")
        return 1
    DOC.write_text(updated)
    print(f"{DOC.name}: summary tables regenerated ({total_files} files)")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
