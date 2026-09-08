"""Alignment checker for pyMeTTa's C++ breadcrumbs and DIVERGENCE markers.

Every ported symbol carries a docstring breadcrumb of the form
`C++: <path relative to a classic repo root>:<line> <C++ qualified symbol>` as
the first line of its docstring. This tool parses those breadcrumbs (and the
`# DIVERGENCE(...)` markers), validates each breadcrumb against whichever of
the classic checkouts (Moses, CogUtil) actually contains its cited file, and
generates docs/ALIGNMENT.md.

A `# DIVERGENCE(...)` marker may span multiple physical `#`-comment lines;
`parse_divergences` merges a marker with its continuation lines into one
`reason` -- see that function's docstring for the exact continuation rule.
"""

import ast
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

DRIFT_TOLERANCE = 25

BREADCRUMB_RE = re.compile(r"^C\+\+: (?P<file>\S+):(?P<line>\d+) (?P<symbol>.+)$")
# The first-line reason is OPTIONAL: `# DIVERGENCE(C++ f.cc:1):` with the text
# entirely on continuation lines is a legitimate style, and requiring `): <text>`
# here made the whole marker fail to match -- so it was SILENTLY DROPPED and
# never reached docs/ALIGNMENT.md. A divergence that documents itself into
# invisibility is the failure this tool exists to prevent, so an empty reason is
# now reported by `validate` rather than skipped. The trailing space after `:`
# is optional for the same reason.
DIVERGENCE_RE = re.compile(r"# DIVERGENCE\((?P<ref>C\+\+ [^)]+)\):(?: (?P<reason>.*))?")
# The REFERENCE itself may wrap across comment lines, e.g.
#     # DIVERGENCE(C++ moses/.../hill-climbing.cc:69,271,278 and
#     # hill-climbing.h:267) / DO NOT FIX: ...
# DIVERGENCE_RE cannot match that -- it needs `)` and `:` on one line -- so
# such markers were ALSO silently dropped (6 of them, including a load-bearing
# DO NOT FIX at hill_climbing.py:35). `_read_marker` joins continuation lines
DIVERGENCE_START_RE = re.compile(r"# DIVERGENCE\(")

# Strips a continuation line's leading `#`-comment marker(s) (and any
# whitespace interleaved with them) down to its bare text, e.g. "    # foo"
# and "# # foo" (a marker written inside an already `#`-per-line-commented
# block) both become "foo". See `parse_divergences` for how this is used.
DIVERGENCE_CONTINUATION_STRIP_RE = re.compile(r"^[#\s]+")

# What a stripped cpp_symbol must look like to be worth searching for at all.
# Guards against `symbol.split("(", 1)[0]` collapsing to "" -- e.g. a cited
# symbol of "()" or "(x)" -- which would otherwise turn the match into
# `re.search(r"\b\b", window)` and pass against almost any non-empty C++ text.
IDENTIFIER_RE = re.compile(r"^[A-Za-z_]\w*$")

REPO_ROOT = Path(__file__).resolve().parent.parent
PYMETTA_ROOT = REPO_ROOT / "pymetta"
ALIGNMENT_DOC = REPO_ROOT / "docs" / "ALIGNMENT.md"

# The classic checkouts that every breadcrumb is validated against, tried in
# order. Hardcoded for the same reason tests/conftest.py hardcodes them: this
# is a single-machine fork used to build one specific port, not a
# general-purpose tool. pymetta/ ports two upstream repos -- Moses proper
# (breadcrumb paths starting `moses/`) and its CogUtil dependency (paths
# starting `opencog/`) -- so a breadcrumb resolves under whichever root
# actually contains its cited file. The two prefixes are disjoint, so there
# is no ambiguity about which root a given breadcrumb belongs to.
CLASSIC_ROOTS = [
    Path("/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses"),
    Path("/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/CogUtil"),
]


@dataclass(frozen=True)
class Breadcrumb:
    py_file: Path
    py_symbol: str
    cpp_file: str
    cpp_line: int
    cpp_symbol: str


@dataclass(frozen=True)
class Divergence:
    py_file: Path
    line: int
    ref: str
    reason: str


def parse_breadcrumbs(path: Path) -> list[Breadcrumb]:
    """Walk `path` with `ast` and collect a Breadcrumb for every
    FunctionDef/AsyncFunctionDef/ClassDef whose docstring's first line is a
    `C++: <file>:<line> <symbol>` breadcrumb. Nodes without one are skipped
    here; `coverage()` reports those separately.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    crumbs = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        doc = ast.get_docstring(node)
        if not doc:
            continue
        match = BREADCRUMB_RE.match(doc.splitlines()[0])
        if not match:
            continue
        crumbs.append(
            Breadcrumb(
                py_file=path,
                py_symbol=node.name,
                cpp_file=match.group("file"),
                cpp_line=int(match.group("line")),
                cpp_symbol=match.group("symbol"),
            )
        )
    return crumbs


def validate(crumbs: list[Breadcrumb], roots: list[Path]) -> list[str]:
    """Return a human-readable problem per breadcrumb that fails to check out
    against `roots`: either the cited C++ file does not exist under any of
    `roots` (tried in order), or -- at whichever root does contain it -- the
    last `::`-separated component of `cpp_symbol`, with any trailing
    parameter list stripped, is not present as a whole word within
    +/-DRIFT_TOLERANCE lines of `cpp_line`.
    """
    problems = []
    for crumb in crumbs:
        cpp_path = next(
            (root / crumb.cpp_file for root in roots if (root / crumb.cpp_file).is_file()),
            None,
        )
        if cpp_path is None:
            tried = ", ".join(str(root) for root in roots)
            problems.append(
                f"{crumb.py_file}: {crumb.py_symbol}: C++ file not found: {crumb.cpp_file} "
                f"(tried: {tried})"
            )
            continue

        # MOSES is full of overloads, so a breadcrumb's cited symbol commonly
        # carries a parameter list, e.g. `randint(int n)` or
        # `build_logical(pre_it, pre_it)`. `\b` does not anchor reliably right
        # after a `)` (it is only a boundary if the next character is a word
        # character), so strip everything from the first `(` onward and match
        # on the bare identifier instead.
        symbol = crumb.cpp_symbol.rsplit("::", 1)[-1]
        symbol = symbol.split("(", 1)[0].strip()
        if not IDENTIFIER_RE.match(symbol):
            # A symbol like "()" or "(x)" strips to "": an empty pattern would
            # otherwise make the \b-anchored search below (`\b\b`) match
            # almost any non-empty text -- a malformed breadcrumb must not
            # pass silently just because it happened to cite a real file.
            problems.append(
                f"{crumb.py_file}: {crumb.py_symbol}: cpp_symbol {crumb.cpp_symbol!r} has no "
                f"plausible identifier to match (stripped to {symbol!r})"
            )
            continue

        lines = cpp_path.read_text(errors="replace").splitlines()
        lo = max(0, crumb.cpp_line - 1 - DRIFT_TOLERANCE)
        hi = min(len(lines), crumb.cpp_line - 1 + DRIFT_TOLERANCE + 1)
        window = "\n".join(lines[lo:hi])
        if not re.search(rf"\b{re.escape(symbol)}\b", window):
            problems.append(
                f"{crumb.py_file}: {crumb.py_symbol}: symbol '{symbol}' not found within "
                f"+/-{DRIFT_TOLERANCE} lines of {crumb.cpp_file}:{crumb.cpp_line}"
            )
    return problems


def _leading_whitespace(line: str) -> str:
    """Return the run of spaces/tabs before `line`'s first non-whitespace char."""
    return line[: len(line) - len(line.lstrip(" \t"))]


def _read_marker(lines, i, indent):
    """Parse the marker starting at `lines[i]`, whose reference may wrap.

    Returns `(ref, first_reason, index_of_last_reference_line)`, or `None`
    when no closing `)` appears within the marker's own comment run. Joins
    continuation lines at `indent` beginning with `#`, so a reference split
    across physical lines is recovered whole.
    """
    joined = lines[i]
    j = i
    while "):" not in joined:
        j += 1
        if j >= len(lines):
            return None
        nxt = lines[j]
        if not nxt.strip() or _leading_whitespace(nxt) != indent:
            return None
        if not nxt.lstrip(" \t").startswith("#"):
            return None
        joined += " " + DIVERGENCE_CONTINUATION_STRIP_RE.sub("", nxt, count=1).strip()
    match = DIVERGENCE_RE.search(joined)
    if not match:
        return None
    return match.group("ref"), match.group("reason"), j


def parse_divergences(path: Path) -> list[Divergence]:
    """Scan `path`'s source lines for `# DIVERGENCE(C++ <ref>): <reason>` markers,
    merging each marker with its continuation lines into one `Divergence`.

    CONTINUATION RULE (deliberate -- authors should write to it): a line
    immediately following a marker line (or another already-accepted
    continuation line) is treated as part of that SAME marker's `reason`
    when, and only when, all of the following hold:

      1. it is not blank;
      2. its leading whitespace is IDENTICAL to the marker's own leading
         whitespace (no dedent, no extra indent);
      3. after that leading whitespace, it begins with `#`;
      4. it does not itself match `# DIVERGENCE(...)` -- two adjacent
         markers never merge, even if line 2's indentation would otherwise
         qualify it as a continuation of marker 1.

    The first following line that fails any of those ends the marker: a
    blank line, code, a dedent/indent change, or the start of a new marker.
    A continuation line's own text is whatever remains after stripping its
    leading run of `#`/whitespace characters (so both a plain "    # foo"
    and a doubly-commented "# # foo" -- a marker written inside a block
    that is already `#`-per-line-commented -- contribute "foo"); each
    contributed piece is appended to `reason` separated by a single space.

    Practical effect for authors: put a multi-line marker's continuation
    `#` lines at the marker's own indentation, with no blank line breaking
    the run, and the whole comment is captured as one `reason`. There is
    no need to front-load the "essential" point onto the first physical
    line -- unlike before this was fixed, the full explanation now reaches
    docs/ALIGNMENT.md regardless of which line it is on.
    """
    lines = path.read_text().splitlines()
    divergences: list[Divergence] = []
    i = 0
    n = len(lines)
    while i < n:
        if not DIVERGENCE_START_RE.search(lines[i]):
            i += 1
            continue

        marker_line = i + 1
        indent = _leading_whitespace(lines[i])
        parsed = _read_marker(lines, i, indent)
        if parsed is None:
            i += 1
            continue
        ref, first, ref_end = parsed
        reason_parts = [first.strip()] if first and first.strip() else []

        j = ref_end + 1
        while j < n:
            candidate = lines[j]
            if not candidate.strip():
                break
            if _leading_whitespace(candidate) != indent:
                break
            if not candidate.lstrip(" \t").startswith("#"):
                break
            if DIVERGENCE_START_RE.search(candidate):
                break
            content = DIVERGENCE_CONTINUATION_STRIP_RE.sub("", candidate, count=1).strip()
            if content:
                reason_parts.append(content)
            j += 1

        divergences.append(
            Divergence(
                py_file=path,
                line=marker_line,
                ref=ref,
                reason=" ".join(reason_parts),
            )
        )
        i = j

    return divergences


def _comment_texts(path: Path) -> dict[int, str]:
    """Return {1-based line number: comment text} for every REAL `#`-comment
    token `tokenize` finds in `path` -- so text that merely LOOKS like a
    comment because it sits inside a string or docstring (e.g. prose
    mentioning `# DIVERGENCE(...)`, interpreter.py:22) never appears here,
    while a marker written as a TRAILING comment after code on the same
    line does. A plain `line.lstrip().startswith("#")` check (the previous
    approach) gets this backwards for that second case: it demands the
    marker be the first thing on its line, which a real Python comment
    need not be.
    """
    with path.open("rb") as fh:
        return {
            tok.start[0]: tok.string
            for tok in tokenize.tokenize(fh.readline)
            if tok.type == tokenize.COMMENT
        }


def unparsed_divergence_markers(path: Path) -> list[int]:
    """Return the 1-based line numbers in `path` that OPEN a
    `# DIVERGENCE(` marker but that `parse_divergences` did not turn into a
    `Divergence` -- i.e. markers that would be silently dropped from
    docs/ALIGNMENT.md.

    Why this exists (S12): a marker is dropped whenever `_read_marker`
    cannot find `"):"` inside the marker's own comment run, which is
    exactly what happens to the natural-looking

        # DIVERGENCE(C++ some/file.cc:687) / DO NOT FIX: ...

    form -- the reference closes with `) /`, never `):`. Two markers in
    `hill_climbing.py` were written that way and had been missing from
    ALIGNMENT.md ever since, undetected, because nothing counted markers
    against parsed results. That is the same failure mode
    docs/CARRY-FORWARD.md Sec 5 records for the 49 markers lost earlier in
    this cycle: a dropped marker makes a file LOOK faithful, which is worse
    than a marker that is merely badly worded.

    `opened` is read off `_comment_texts`, i.e. real comment TOKENS, not
    raw lines -- so a marker written as a trailing comment after code
    (`x = 1  # DIVERGENCE(C++ a.cc:1) / DO NOT FIX: ...`) is counted too.
    An earlier version of this function required a marker's line to have
    `#` as its first non-whitespace character, which made exactly that
    broken, mid-line shape invisible to BOTH this function (excluded from
    `opened`, since the line does not start with `#`) and
    `parse_divergences` (which cannot find `"):"` either, for the same
    reason as the whole-line case above) -- reported by neither, the third
    such gap this parser has had. Prose that merely MENTIONS the marker
    syntax inside a docstring (`... not a `# DIVERGENCE(C++ ...)` marker`,
    interpreter.py:22) is still excluded, because it is never a COMMENT
    token in the first place -- string content, wherever it sits on the
    line, is never mistaken for a comment by `tokenize`.
    """
    opened = [
        line_no
        for line_no, text in _comment_texts(path).items()
        if DIVERGENCE_START_RE.search(text)
    ]
    parsed = {d.line for d in parse_divergences(path)}
    return [line_no for line_no in opened if line_no not in parsed]


def coverage(pkg_root: Path) -> dict[str, tuple[int, int]]:
    """Map each module under `pkg_root` to (symbols with breadcrumbs, total
    public symbols), where a symbol is a FunctionDef/AsyncFunctionDef/ClassDef
    (at any nesting level) whose name does not start with `_`.
    """
    result: dict[str, tuple[int, int]] = {}
    for py_file in sorted(pkg_root.rglob("*.py")):
        tree = ast.parse(py_file.read_text(), filename=str(py_file))
        total = 0
        with_breadcrumb = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            if node.name.startswith("_"):
                continue
            total += 1
            doc = ast.get_docstring(node)
            if doc and BREADCRUMB_RE.match(doc.splitlines()[0]):
                with_breadcrumb += 1
        module = py_file.relative_to(pkg_root.parent).as_posix()
        result[module] = (with_breadcrumb, total)
    return result


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _render_alignment_doc(
    crumbs: list[Breadcrumb],
    divergences: list[Divergence],
    cov: dict[str, tuple[int, int]],
) -> str:
    lines = [
        "# Alignment Report",
        "",
        "Generated by `tools/check_alignment.py`. Do not edit by hand.",
        "",
        "## Breadcrumbs",
        "",
    ]
    if crumbs:
        lines += [
            "| Python file | Python symbol | C++ file | C++ line | C++ symbol |",
            "|---|---|---|---|---|",
        ]
        for c in sorted(crumbs, key=lambda c: (_display_path(c.py_file), c.py_symbol)):
            lines.append(
                f"| {_display_path(c.py_file)} | {c.py_symbol} | {c.cpp_file} "
                f"| {c.cpp_line} | {c.cpp_symbol} |"
            )
    else:
        lines.append("_No breadcrumbs yet._")

    lines += ["", "## Divergences", ""]
    if divergences:
        lines += [
            "| Python file | Line | C++ reference | Reason |",
            "|---|---|---|---|",
        ]
        for d in sorted(divergences, key=lambda d: (_display_path(d.py_file), d.line)):
            lines.append(f"| {_display_path(d.py_file)} | {d.line} | {d.ref} | {d.reason} |")
    else:
        lines.append("_No divergences yet._")

    lines += ["", "## Coverage", ""]
    reportable = {module: counts for module, counts in cov.items() if counts[1] > 0}
    if reportable:
        lines += [
            "| Module | Breadcrumbs | Public symbols | Coverage |",
            "|---|---|---|---|",
        ]
        for module in sorted(reportable):
            with_breadcrumb, total = reportable[module]
            pct = round(100 * with_breadcrumb / total)
            lines.append(f"| {module} | {with_breadcrumb} | {total} | {pct}% |")
    else:
        lines.append("_No public symbols yet._")

    lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    """Walk pymetta/, validate every breadcrumb against CLASSIC_ROOTS, write
    docs/ALIGNMENT.md, print any problems, and return 1 if any were found.
    """
    del argv  # no CLI options yet; the classic checkout paths are fixed constants

    crumbs: list[Breadcrumb] = []
    divergences: list[Divergence] = []
    for py_file in sorted(PYMETTA_ROOT.rglob("*.py")):
        crumbs.extend(parse_breadcrumbs(py_file))
        divergences.extend(parse_divergences(py_file))

    problems = validate(crumbs, CLASSIC_ROOTS)
    for py_file in sorted(PYMETTA_ROOT.rglob("*.py")):
        for line_no in unparsed_divergence_markers(py_file):
            problems.append(
                f"{py_file}:{line_no}: DIVERGENCE marker is not parseable and would be "
                f"silently dropped from {ALIGNMENT_DOC.name}; write it as "
                f"`# DIVERGENCE(C++ <ref>): <reason>` (the reference must close with `):`, "
                f"so put a `DO NOT FIX` note AFTER the colon, not between `)` and `:`)"
            )
    for problem in problems:
        print(problem)

    ALIGNMENT_DOC.parent.mkdir(parents=True, exist_ok=True)
    ALIGNMENT_DOC.write_text(_render_alignment_doc(crumbs, divergences, coverage(PYMETTA_ROOT)))

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
