"""Tests for tools/coverage_totals.py's row-vs-tree staleness checks.

Every check here is proved in BOTH directions: a case it must flag and a case
it must NOT flag. The project's standing rule is that a guard which has never
been seen to fire is not a guard -- `check_planned_rows_against_reality` spent
its whole first life vacuous (no PLANNED row named a module, so its loop never
had a path to test) and stayed green through it, which is exactly what these
tests exist to prevent recurring.
"""

import textwrap

import pytest

from tools import coverage_totals as ct


def _doc(rows: str) -> str:
    """A minimal document body. The row checks parse rows directly with
    FULL_ROW_RE and never touch the summary tables, so a table-free stub is a
    faithful input for them (`_aggregate` is exercised separately, against the
    real document)."""
    return "| File | Lines | Status | pyMoses path / Slice / Reason |\n|---|---:|---|---|\n" + rows


# ---------------------------------------------------------------------------
# check_named_modules_exist -- inverse direction 1


def test_named_modules_exist_flags_a_ported_row_naming_a_missing_module(monkeypatch, tmp_path):
    monkeypatch.setattr(ct, "REPO_ROOT", tmp_path)
    text = _doc("| `moses/x/y.cc` | 10 | PORTED | `pymoses/x/y.py` -- ported in S3. |\n")
    problems = ct.check_named_modules_exist(text)
    assert len(problems) == 1
    assert "pymoses/x/y.py" in problems[0]
    assert "does not exist" in problems[0]


def test_named_modules_exist_accepts_a_module_that_is_there(monkeypatch, tmp_path):
    monkeypatch.setattr(ct, "REPO_ROOT", tmp_path)
    (tmp_path / "pymoses" / "x").mkdir(parents=True)
    (tmp_path / "pymoses" / "x" / "y.py").write_text("")
    text = _doc("| `moses/x/y.cc` | 10 | PORTED | `pymoses/x/y.py` -- ported in S3. |\n")
    assert ct.check_named_modules_exist(text) == []


def test_named_modules_exist_leaves_planned_rows_to_the_forward_check(monkeypatch, tmp_path):
    # A PLANNED row's module is a PREDICTION -- it is SUPPOSED not to exist
    # yet. Flagging it here would make every correct PLANNED row red.
    monkeypatch.setattr(ct, "REPO_ROOT", tmp_path)
    text = _doc("| `moses/x/y.cc` | 10 | PLANNED: P2 | `pymoses/x/y.py` -- not started. |\n")
    assert ct.check_named_modules_exist(text) == []


def test_named_modules_exist_is_clean_on_the_real_document():
    assert ct.check_named_modules_exist(ct.DOC.read_text()) == []


# ---------------------------------------------------------------------------
# check_planned_rows_against_reality / check_planned_rows_name_a_module


def test_planned_row_whose_awaited_module_landed_is_flagged(monkeypatch, tmp_path):
    monkeypatch.setattr(ct, "REPO_ROOT", tmp_path)
    (tmp_path / "pymoses").mkdir()
    (tmp_path / "pymoses" / "z.py").write_text("")
    text = _doc(
        "| `moses/x/z.cc` | 10 | PLANNED: P2 | AWAITING `pymoses/z.py` -- deferred tail. |\n"
    )
    problems = ct.check_planned_rows_against_reality(text)
    assert len(problems) == 1
    assert "already exists" in problems[0]


def test_planned_row_awaiting_a_module_that_is_absent_is_correct(monkeypatch, tmp_path):
    monkeypatch.setattr(ct, "REPO_ROOT", tmp_path)
    text = _doc(
        "| `moses/x/z.cc` | 10 | PLANNED: P2 | AWAITING `pymoses/z.py` -- deferred tail. |\n"
    )
    assert ct.check_planned_rows_against_reality(text) == []


def test_a_bare_module_mention_in_a_planned_row_is_context_not_a_declaration(
    monkeypatch, tmp_path
):
    # The eval-features.h shape: a PLANNED row that POINTS AT a neighbouring
    # ported module ("see the mutual_info.py row"). Treating that as the awaited
    # module made the row report itself stale against a file it merely cites.
    monkeypatch.setattr(ct, "REPO_ROOT", tmp_path)
    (tmp_path / "pymoses").mkdir()
    (tmp_path / "pymoses" / "neighbour.py").write_text("")
    text = _doc(
        "| `moses/x/z.cc` | 10 | PLANNED: P2 | AWAITING `pymoses/z.py`; the only oracle for "
        "`pymoses/neighbour.py` is this binary. |\n"
    )
    assert ct.check_planned_rows_against_reality(text) == []


def test_folds_into_inverts_the_existence_test(monkeypatch, tmp_path):
    # type_tree_def.cc's shape: two functions that belong INSIDE an existing
    # module. "Does the module exist yet" is the wrong question, so the row
    # declares FOLDS INTO and the test flips: the host must be there.
    monkeypatch.setattr(ct, "REPO_ROOT", tmp_path)
    text = _doc(
        "| `moses/x/z.cc` | 10 | PLANNED: P2 | FOLDS INTO `pymoses/host.py` -- 11 lines. |\n"
    )
    problems = ct.check_planned_rows_against_reality(text)
    assert len(problems) == 1
    assert "which does not exist" in problems[0]

    (tmp_path / "pymoses").mkdir()
    (tmp_path / "pymoses" / "host.py").write_text("")
    assert ct.check_planned_rows_against_reality(text) == []


def test_planned_row_without_a_declaration_is_flagged_as_unfalsifiable():
    # The vacuity bug itself: a PLANNED row declaring no module can never be
    # contradicted by anything on disk.
    text = _doc("| `moses/x/z.cc` | 10 | PLANNED: P2 | Deferred tail; no module named. |\n")
    problems = ct.check_planned_rows_name_a_module(text)
    assert len(problems) == 1
    assert "declares neither AWAITING" in problems[0]


def test_a_bare_module_mention_does_not_satisfy_the_convention():
    # Without a marker the checker cannot tell "the module I am waiting for"
    # from "a module I am pointing at", so a bare path is not a declaration.
    text = _doc("| `moses/x/z.cc` | 10 | PLANNED: P2 | `pymoses/z.py` -- deferred tail. |\n")
    assert len(ct.check_planned_rows_name_a_module(text)) == 1


@pytest.mark.parametrize("marker", ["AWAITING", "FOLDS INTO"])
def test_either_declaration_satisfies_the_convention(marker):
    text = _doc(f"| `moses/x/z.cc` | 10 | PLANNED: P2 | {marker} `pymoses/z.py` -- tail. |\n")
    assert ct.check_planned_rows_name_a_module(text) == []


def test_non_planned_rows_are_not_required_to_name_a_module():
    text = _doc("| `moses/x/z.cc` | 10 | EXCLUDED | Legacy embodiment vocabulary. |\n")
    assert ct.check_planned_rows_name_a_module(text) == []


def test_planned_row_checks_are_clean_on_the_real_document():
    text = ct.DOC.read_text()
    assert ct.check_planned_rows_against_reality(text) == []
    assert ct.check_planned_rows_name_a_module(text) == []


# ---------------------------------------------------------------------------
# _gates_in -- the attribution that makes the gate check exact


def _module(tmp_path, body: str):
    path = tmp_path / "m.py"
    path.write_text(textwrap.dedent(body))
    return path


def test_gates_in_attributes_a_raise_to_its_enclosing_breadcrumb(tmp_path):
    mod = _module(
        tmp_path,
        '''
        class C:
            def f(self):
                """C++: opencog/util/mt19937ar.cc:50 MT19937RandGen::randfloat"""
                raise NotImplementedError("needs float32 truncation")
        ''',
    )
    assert ct._gates_in(mod) == [(5, "opencog/util/mt19937ar.cc", False, "C.f")]


def test_gates_in_marks_an_abstractmethod(tmp_path):
    mod = _module(
        tmp_path,
        '''
        class C:
            @abstractmethod
            def f(self):
                """C++: moses/moses/main/problem.h:56 problem_base::name"""
                raise NotImplementedError
        ''',
    )
    (lineno, cpp, abstract, qual) = ct._gates_in(mod)[0]
    assert (cpp, abstract, qual) == ("moses/moses/main/problem.h", True, "C.f")
    assert lineno == 6


def test_gates_in_recognises_a_dotted_abstractmethod_decorator(tmp_path):
    mod = _module(
        tmp_path,
        '''
        class C:
            @abc.abstractmethod
            def f(self):
                """C++: moses/x/y.h:1 base::f"""
                raise NotImplementedError
        ''',
    )
    assert ct._gates_in(mod)[0][2] is True


def test_gates_in_falls_back_to_an_outer_breadcrumb(tmp_path):
    # A helper nested inside a breadcrumbed symbol, with no breadcrumb of its
    # own, still belongs to that C++ file.
    mod = _module(
        tmp_path,
        '''
        def outer():
            """C++: moses/x/y.cc:1 outer"""
            def inner():
                raise NotImplementedError
        ''',
    )
    assert ct._gates_in(mod) == [(5, "moses/x/y.cc", False, "outer.inner")]


def test_gates_in_attributes_nothing_when_there_is_no_breadcrumb(tmp_path):
    # The `_unported_table_problem` shape: a port-invented stand-in class, so
    # its raise is evidence against no C++ row at all.
    mod = _module(
        tmp_path,
        '''
        class _stand_in:
            """Not one specific C++ class -- a stub for unported problems."""
            def run(self, ob):
                raise NotImplementedError("problem not ported")
        ''',
    )
    assert ct._gates_in(mod) == [(5, None, False, "_stand_in.run")]


def test_gates_in_ignores_other_exception_types(tmp_path):
    mod = _module(
        tmp_path,
        '''
        def f():
            """C++: moses/x/y.cc:1 f"""
            raise ValueError("column not found")
        ''',
    )
    assert ct._gates_in(mod) == []


def test_gates_in_survives_an_unparsable_or_absent_module(tmp_path):
    assert ct._gates_in(tmp_path / "nope.py") == []
    bad = tmp_path / "bad.py"
    bad.write_text("def f(:\n")
    assert ct._gates_in(bad) == []


# ---------------------------------------------------------------------------
# check_ported_rows_do_not_gate -- inverse direction 2


def test_ported_row_whose_own_symbol_raises_is_flagged(monkeypatch, tmp_path):
    monkeypatch.setattr(ct, "REPO_ROOT", tmp_path)
    (tmp_path / "pymoses").mkdir()
    (tmp_path / "pymoses" / "rng.py").write_text(
        'class R:\n'
        '    def randfloat(self):\n'
        '        """C++: opencog/util/mt19937ar.cc:50 R::randfloat"""\n'
        '        raise NotImplementedError\n'
    )
    text = _doc(
        "| `opencog/util/mt19937ar.cc` | 94 | PORTED | `pymoses/rng.py` -- bit-identical. |\n"
    )
    problems = ct.check_ported_rows_do_not_gate(text)
    assert len(problems) == 1
    assert "R.randfloat" in problems[0]
    assert "make it PARTIAL" in problems[0]


def test_partial_row_with_the_same_raise_is_not_flagged(monkeypatch, tmp_path):
    # PARTIAL is the correct home for a named gap; only PORTED is a false claim.
    monkeypatch.setattr(ct, "REPO_ROOT", tmp_path)
    (tmp_path / "pymoses").mkdir()
    (tmp_path / "pymoses" / "rng.py").write_text(
        'class R:\n'
        '    def randfloat(self):\n'
        '        """C++: opencog/util/mt19937ar.cc:50 R::randfloat"""\n'
        '        raise NotImplementedError\n'
    )
    text = _doc(
        "| `opencog/util/mt19937ar.cc` | 94 | PARTIAL | `pymoses/rng.py` -- GAP: randfloat. |\n"
    )
    assert ct.check_ported_rows_do_not_gate(text) == []


def test_ported_row_is_not_flagged_by_a_raise_belonging_to_another_cpp_file(monkeypatch, tmp_path):
    # The real `problem_params.py` shape: one module named by four rows, two
    # PORTED and two PARTIAL, with all eleven raises breadcrumbed to the
    # PARTIAL pair's C++ file. A module-level grep blames the wrong row here.
    monkeypatch.setattr(ct, "REPO_ROOT", tmp_path)
    (tmp_path / "pymoses").mkdir()
    (tmp_path / "pymoses" / "pp.py").write_text(
        'def parse_options():\n'
        '    """C++: moses/moses/main/problem-params.cc:300 problem_params::parse_options"""\n'
        '    raise NotImplementedError("--enable-fs is unported")\n'
    )
    text = _doc(
        "| `moses/moses/main/moses_exec_def.cc` | 20 | PORTED | `pymoses/pp.py` -- folded in. |\n"
        "| `moses/moses/main/problem-params.cc` | 2,407 | PARTIAL | `pymoses/pp.py` -- GAP: fs. |\n"
    )
    assert ct.check_ported_rows_do_not_gate(text) == []


def test_ported_row_is_not_flagged_by_an_abstractmethod(monkeypatch, tmp_path):
    # C++'s `virtual ... = 0;` has no body; a Python @abstractmethod that
    # raises is its faithful port, not a gap. Without this exemption
    # `moses/moses/main/problem.h` -- a complete port -- goes red.
    monkeypatch.setattr(ct, "REPO_ROOT", tmp_path)
    (tmp_path / "pymoses").mkdir()
    (tmp_path / "pymoses" / "problem.py").write_text(
        'class problem_base:\n'
        '    @abstractmethod\n'
        '    def name(self):\n'
        '        """C++: moses/moses/main/problem.h:56 problem_base::name"""\n'
        '        raise NotImplementedError\n'
    )
    text = _doc(
        "| `moses/moses/main/problem.h` | 80 | PORTED | `pymoses/problem.py` -- registry. |\n"
    )
    assert ct.check_ported_rows_do_not_gate(text) == []


def test_gate_check_is_clean_on_the_real_document():
    assert ct.check_ported_rows_do_not_gate(ct.DOC.read_text()) == []


def test_gate_check_fires_on_the_real_document_when_a_row_is_wrongly_promoted():
    # Proof the guard actually bites in situ, not only on synthetic input:
    # take the real document, demote-to-PORTED the one row the 2026-09-09
    # audit corrected, and require the check to catch it again.
    text = ct.DOC.read_text()
    row = "| `opencog/util/mt19937ar.cc` | 94 | PARTIAL |"
    assert row in text, "the mt19937ar.cc row was reworded; update this regression test"
    problems = ct.check_ported_rows_do_not_gate(
        text.replace(row, "| `opencog/util/mt19937ar.cc` | 94 | PORTED |")
    )
    # Was three until `randfloat` was implemented (it is reachable from the
    # CLI through --fs-enforce-features-filename / biased_randbool, and
    # through the combo `rand` builtin). docs/MODULE-COVERAGE.md's own
    # mt19937ar.cc / RandGen.h reason text still names randfloat as a gap
    # and is stale in that one respect; the row's STATUS (PARTIAL) is still
    # right, because the two below really do still raise, so
    # `check_ported_rows_do_not_gate` stays clean on the real document.
    assert len(problems) == 2
    assert {"randdouble_one_excluded", "rand_discrete"} == {
        p.split("(MT19937RandGen.")[1].split(")")[0] for p in problems
    }


# ---------------------------------------------------------------------------
# aggregates and the generated prose


def test_aggregate_matches_the_documents_declared_invariant():
    agg, files, lines = ct._aggregate(ct.DOC.read_text())
    assert (files, lines) == (ct.EXPECTED_FILES, ct.EXPECTED_LINES)
    assert sum(v[0] for v in agg.values()) == files
    assert sum(v[1] for v in agg.values()) == lines


def test_reading_paragraph_numbers_agree_with_the_aggregates():
    text = ct.DOC.read_text()
    agg, files, lines = ct._aggregate(text)
    generated = ct._render_reading(agg, files, lines)
    assert generated in text, (
        "the '**Reading this table.**' paragraph is stale -- "
        "run tools/coverage_totals.py"
    )


def test_reading_paragraph_regex_still_finds_its_paragraph():
    # If the paragraph is ever reflowed across lines, READING_RE stops
    # matching and its numbers would silently stop being regenerated. main()
    # raises in that case; this pins the same condition at test time.
    assert ct.READING_RE.search(ct.DOC.read_text()) is not None


def test_check_mode_is_green_end_to_end():
    assert ct.main(["--check"]) == 0


def test_partial_to_ported_gap_is_documented_rather_than_half_checked():
    # The one check that was asked for and deliberately NOT built. If someone
    # later builds it, this test is the place that should change first.
    assert not hasattr(ct, "check_partial_rows_whose_gap_closed")
    note = ct.WHY_NO_PARTIAL_TO_PORTED_CHECK
    assert "WITHOUT FALSE POSITIVES" in note
    assert "Most gaps are absences, not raises" in note


@pytest.mark.parametrize(
    "status", [s for s in ct.STATUS_ORDER if s not in ct.MAY_BE_EMPTY]
)
def test_every_status_in_status_order_is_used_by_a_real_row(status):
    # `_aggregate` already raises if STATUS_ORDER and the rows disagree, but it
    # does so as a SystemExit inside a rendering path; this states the same
    # partition invariant as a plain assertion, per status.
    agg, _files, _lines = ct._aggregate(ct.DOC.read_text())
    assert agg[status][0] > 0


def test_planned_is_the_only_status_allowed_to_be_empty():
    # Pins the exemption rather than letting it widen by accident. If a second
    # status is ever added to MAY_BE_EMPTY, that is a decision someone should
    # have to make here, in a test, with a reason -- not a one-word edit to a
    # set literal.
    assert ct.MAY_BE_EMPTY == {"PLANNED: P2"}
    assert ct.MAY_BE_EMPTY < set(ct.STATUS_ORDER)


def test_the_empty_planned_bucket_still_renders_as_a_zero_row():
    # The whole point of MAY_BE_EMPTY: an emptied bucket must stay VISIBLE at
    # zero, because silently dropping it is indistinguishable from the
    # under-summing bug this module exists to prevent. Asserted against the
    # real document, which has had no PLANNED row since 2026-09-11.
    agg, files, lines = ct._aggregate(ct.DOC.read_text())
    assert agg["PLANNED: P2"] == (0, 0)
    summary, macro = ct._render(agg, files, lines)
    assert "| PLANNED: P2 | 0 | 0 | 0.0% | 0.0% |" in summary
    assert "| PLANNED: P2 (deferred tail) | 0 | 0 |" in macro


def test_an_unknown_row_status_is_still_fatal():
    # The direction that CAN corrupt the table is unchanged by MAY_BE_EMPTY:
    # a row whose status no table line accounts for must still abort, or the
    # table under-sums while the printed Total does not.
    rows = _doc(
        "| `a.cc` | 10 | PORTED | pymoses/x.py |\n"
        "| `b.cc` | 20 | WIBBLE | pymoses/y.py |\n"
    )
    with pytest.raises(SystemExit) as exc:
        ct._aggregate(rows)
    assert "WIBBLE" in str(exc.value)
