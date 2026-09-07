import subprocess, sys, textwrap  # noqa: E401, F401, I001
from pathlib import Path  # noqa: F401
from tools.check_alignment import parse_breadcrumbs, parse_divergences, Breadcrumb, Divergence

def test_parses_a_wellformed_breadcrumb(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(textwrap.dedent('''
        def build_logical(self, it):
            """C++: moses/moses/representation/build_knobs.cc:602 build_knobs::build_logical

            Decorate a logical subtree with knobs.
            """
    '''))
    crumbs = parse_breadcrumbs(f)
    assert crumbs == [Breadcrumb(
        py_file=f, py_symbol="build_logical",
        cpp_file="moses/moses/representation/build_knobs.cc",
        cpp_line=602, cpp_symbol="build_knobs::build_logical",
    )]

def test_reports_symbol_missing_from_cpp(tmp_path, classic_root):
    f = tmp_path / "m.py"
    f.write_text('def f():\n    """C++: moses/comboreduct/combo/vertex.h:69 no_such_symbol_xyz"""\n')  # noqa: E501
    from tools.check_alignment import validate
    problems = validate(parse_breadcrumbs(f), [classic_root])
    assert len(problems) == 1
    assert "no_such_symbol_xyz" in problems[0]

def test_accepts_symbol_within_drift_tolerance(tmp_path, classic_root):
    # builtin enum really is at vertex.h:69; cite it 10 lines off and it must still pass
    f = tmp_path / "m.py"
    f.write_text('def f():\n    """C++: moses/comboreduct/combo/vertex.h:79 builtin"""\n')
    from tools.check_alignment import validate
    assert validate(parse_breadcrumbs(f), [classic_root]) == []

def test_validates_cogutil_rooted_breadcrumb(tmp_path, classic_root, cogutil_root):
    # Regression: pymetta/util/rng.py ports CogUtil (opencog/util/...), not Moses.
    # A single classic_root can never see this file; this is the exact gap that let
    # 11 accurate breadcrumbs in rng.py go unverified.
    f = tmp_path / "m.py"
    f.write_text(
        'def f():\n    """C++: opencog/util/mt19937ar.cc:44 MT19937RandGen::randint"""\n'
    )
    from tools.check_alignment import validate
    assert validate(parse_breadcrumbs(f), [classic_root, cogutil_root]) == []

def test_reports_every_root_tried_when_file_not_found(tmp_path, classic_root, cogutil_root):
    f = tmp_path / "m.py"
    f.write_text('def f():\n    """C++: no/such/file.cc:1 whatever"""\n')
    from tools.check_alignment import validate
    problems = validate(parse_breadcrumbs(f), [classic_root, cogutil_root])
    assert len(problems) == 1
    assert str(classic_root) in problems[0]
    assert str(cogutil_root) in problems[0]

def test_strips_parameter_list_before_matching_symbol(tmp_path, cogutil_root):
    # Regression: mt19937ar.cc:66 really is `int MT19937RandGen::randint(int n) {`,
    # but a bare \b-anchored match on "randint(int n)" can never succeed -- `\b`
    # is not a boundary right after `)` when the next character is a space. The
    # checker must strip the parameter list and match on the bare identifier.
    f = tmp_path / "m.py"
    f.write_text(
        'def f():\n    """C++: opencog/util/mt19937ar.cc:66 MT19937RandGen::randint(int n)"""\n'  # noqa: E501
    )
    from tools.check_alignment import validate
    assert validate(parse_breadcrumbs(f), [cogutil_root]) == []

def test_rejects_symbol_that_strips_to_empty(tmp_path, classic_root):
    # Regression: a cpp_symbol whose last "::"-component begins with "(" (e.g.
    # "()", "(x)") or is pure whitespace strips to "". An empty pattern would
    # make the \b-anchored search match almost any non-empty C++ text, so a
    # malformed breadcrumb must be reported as a problem, never pass silently.
    from tools.check_alignment import validate
    for cpp_symbol in ("()", "(x)", "   "):
        crumb = Breadcrumb(
            py_file=tmp_path / "m.py", py_symbol="f",
            cpp_file="moses/comboreduct/combo/vertex.h", cpp_line=69,
            cpp_symbol=cpp_symbol,
        )
        problems = validate([crumb], [classic_root])
        assert len(problems) == 1, f"cpp_symbol={cpp_symbol!r} should be one problem, not pass"
        assert repr(cpp_symbol) in problems[0], f"problem should name {cpp_symbol!r}: {problems[0]}"

def test_parses_a_single_line_divergence_marker(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(
        "def f():\n"
        "    x = 1\n"
        "    # DIVERGENCE(C++ foo.cc:10): a simple one-line reason.\n"
        "    return x\n"
    )
    assert parse_divergences(f) == [Divergence(
        py_file=f, line=3, ref="C++ foo.cc:10", reason="a simple one-line reason.",
    )]

def test_merges_multiline_divergence_continuation(tmp_path):
    # The bug: continuation lines were dropped entirely, so every multi-line
    # marker rendered in docs/ALIGNMENT.md as a meaningless first-line fragment.
    f = tmp_path / "m.py"
    f.write_text(
        "def f():\n"
        "    # DIVERGENCE(C++ foo.cc:10): first part of the\n"
        "    # reason continues here across two more\n"
        "    # physical lines before it stops.\n"
        "    return 1\n"
    )
    divergences = parse_divergences(f)
    assert divergences == [Divergence(
        py_file=f, line=2, ref="C++ foo.cc:10",
        reason=(
            "first part of the reason continues here across two more "
            "physical lines before it stops."
        ),
    )]

def test_divergence_marker_followed_immediately_by_code_stays_single_line(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(
        "def f():\n"
        "    # DIVERGENCE(C++ foo.cc:10): reason cut short by code.\n"
        "    return 1\n"
    )
    assert parse_divergences(f) == [Divergence(
        py_file=f, line=2, ref="C++ foo.cc:10", reason="reason cut short by code.",
    )]

def test_divergence_marker_at_end_of_file(tmp_path):
    f = tmp_path / "m.py"
    f.write_text(
        "def f():\n"
        "    pass\n"
        "\n"
        "\n"
        "# DIVERGENCE(C++ foo.cc:10): reason that runs to\n"
        "# the very last line of the file."
    )
    assert parse_divergences(f) == [Divergence(
        py_file=f, line=5, ref="C++ foo.cc:10",
        reason="reason that runs to the very last line of the file.",
    )]

def test_adjacent_divergence_markers_do_not_merge(tmp_path):
    # Two markers back-to-back at the same indentation, no blank line between
    # them: each must stay its own Divergence with only its own reason, never
    # merged into one.
    f = tmp_path / "m.py"
    f.write_text(
        "def f():\n"
        "    # DIVERGENCE(C++ foo.cc:10): first marker reason.\n"
        "    # DIVERGENCE(C++ foo.cc:20): second marker reason.\n"
        "    return 1\n"
    )
    assert parse_divergences(f) == [
        Divergence(py_file=f, line=2, ref="C++ foo.cc:10", reason="first marker reason."),
        Divergence(py_file=f, line=3, ref="C++ foo.cc:20", reason="second marker reason."),
    ]
