"""Alignment checker for pyMeTTa's C++ breadcrumbs and DIVERGENCE markers.

Every ported symbol carries a docstring breadcrumb of the form
`C++: <path relative to a classic repo root>:<line> <C++ qualified symbol>` as
the first line of its docstring. This tool parses those breadcrumbs (and the
`# DIVERGENCE(...)` markers), validates each breadcrumb against whichever of
the classic checkouts (Moses, CogUtil) actually contains its cited file, and
generates docs/ALIGNMENT.md.
"""

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

DRIFT_TOLERANCE = 25

BREADCRUMB_RE = re.compile(r"^C\+\+: (?P<file>\S+):(?P<line>\d+) (?P<symbol>.+)$")
DIVERGENCE_RE = re.compile(r"# DIVERGENCE\((?P<ref>C\+\+ [^)]+)\): (?P<reason>.+)")

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


def parse_divergences(path: Path) -> list[Divergence]:
    """Scan `path`'s source lines for `# DIVERGENCE(C++ <ref>): <reason>` markers."""
    divergences = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        match = DIVERGENCE_RE.search(line)
        if match:
            divergences.append(
                Divergence(
                    py_file=path,
                    line=lineno,
                    ref=match.group("ref"),
                    reason=match.group("reason").strip(),
                )
            )
    return divergences


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
    for problem in problems:
        print(problem)

    ALIGNMENT_DOC.parent.mkdir(parents=True, exist_ok=True)
    ALIGNMENT_DOC.write_text(_render_alignment_doc(crumbs, divergences, coverage(PYMETTA_ROOT)))

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
