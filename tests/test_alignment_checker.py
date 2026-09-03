import subprocess, sys, textwrap  # noqa: E401, F401, I001
from pathlib import Path  # noqa: F401
from tools.check_alignment import parse_breadcrumbs, Breadcrumb

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
    problems = validate(parse_breadcrumbs(f), classic_root)
    assert len(problems) == 1
    assert "no_such_symbol_xyz" in problems[0]

def test_accepts_symbol_within_drift_tolerance(tmp_path, classic_root):
    # builtin enum really is at vertex.h:69; cite it 10 lines off and it must still pass
    f = tmp_path / "m.py"
    f.write_text('def f():\n    """C++: moses/comboreduct/combo/vertex.h:79 builtin"""\n')
    from tools.check_alignment import validate
    assert validate(parse_breadcrumbs(f), classic_root) == []
