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
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path

DOC = Path(__file__).resolve().parent.parent / "docs" / "MODULE-COVERAGE.md"

# The inventory's own invariant: every file under $CLASSIC/moses/ (240, by
# `git ls-files`) plus the CogUtil files pyMeTTa ports. That CogUtil set GROWS
# as slices reach new dependencies -- S9 added lazy_random_selector.{h,cc}
# (+2 files, +97 lines), taking the total from 243/62,047 to 245/62,144. S11
# Task 1 added selection.h (+1 file, +155 lines), taking the total to
# 246/62,299. Update these two constants, with the reason, whenever a CogUtil
# file is newly ported.
EXPECTED_FILES = 246
EXPECTED_LINES = 62299

# Display order. `_aggregate` proves this is exhaustive rather than assuming it.
STATUS_ORDER = [
    "PORTED",
    "PARTIAL",
    "PLANNED: S13",
    "PLANNED: S14",
    "PLANNED: P2",
    "EXCLUDED",
]

ROW_RE = re.compile(r"^\| `([^`]+)` \| *([\d,]+) \| ([A-Z][A-Za-z0-9: ]*?) \|", re.M)
SUMMARY_RE = re.compile(
    r"\| Status \| Files \| Lines \| % of files \| % of lines \|\n"
    r"\|---\|---:\|---:\|---:\|---:\|\n(?:\|[^\n]*\n)+"
)
MACRO_RE = re.compile(
    r"\| Macro status \| Files \| Lines \|\n\|---\|---:\|---:\|\n(?:\|[^\n]*\n)+"
)


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
    missing = set(STATUS_ORDER) - set(agg)
    if unknown or missing:
        raise SystemExit(
            f"status mismatch: rows use {sorted(unknown)} which STATUS_ORDER omits; "
            f"STATUS_ORDER names {sorted(missing)} which no row uses. "
            "Rendering would silently drop rows from the table."
        )
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
    module in backticks (`pymetta/...py`); if that file exists on disk, the row
    is stale. It is deliberately one-directional: a PORTED/PARTIAL row whose
    module is missing is NOT flagged, since a row may legitimately describe a
    symbol folded into a differently-named module.
    """
    problems: list[str] = []
    repo_root = DOC.parent.parent
    row_re = re.compile(
        r"^\| `(?P<cpp>[^`]+)` \| *[\d,]+ \| (?P<status>PLANNED: [^|]+?) \| (?P<reason>[^\n]*)$",
        re.M,
    )
    for m in row_re.finditer(text):
        for mod in re.findall(r"`(pymetta/[\w/]+\.py)`", m.group("reason")):
            if (repo_root / mod).exists():
                problems.append(
                    f"{DOC.name}: `{m.group('cpp')}` is {m.group('status').strip()} "
                    f"but {mod} already exists -- the row is stale, or the reason "
                    f"names the wrong module."
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
    stale = check_planned_rows_against_reality(text)
    for problem in stale:
        print(problem)

    summary, macro = _render(agg, total_files, total_lines)

    updated = SUMMARY_RE.sub(lambda _m: summary + "\n", text, count=1)
    updated = MACRO_RE.sub(lambda _m: macro + "\n", updated, count=1)
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
