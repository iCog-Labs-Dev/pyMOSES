"""End-to-end tests for ``-a gomea`` through the real MOSES CLI.

Every run goes through ``moses_exec(argv)`` in-process with stdout
captured, the same convention ``tests/moses/main/test_moses_exec.py``
uses. The ``-a hc`` expectations and ``data/help_base.out`` were captured
at commit 0410881, which predates ALL GOMEA code (no ``pymoses/gomea``
package, no ``--gomea-*`` option), and pin that adding GOMEA leaves the
default optimizer's output and classic's option surface byte-identical.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymoses.comboreduct.type_checker.type_signature import gen_signature, type_node
from pymoses.gomea.schema import UnsupportedRepresentation
from pymoses.moses.main.demo_problems import (
    _alphabet_size,
    _set_noise_or_ratio,
    even_parity,
    logical_bscore,
    register_demo_problems,
)
from pymoses.moses.main.moses_exec import _add_demo_options, moses_exec
from pymoses.moses.main.problem import problem_manager
from pymoses.moses.main.problem_params import parse_args, problem_params
from pymoses.moses.main.table_problems import ip_problem_params, register_table_problems
from pymoses.moses.scoring.behave_cscore import behave_cscore

REPO = Path(__file__).resolve().parents[2]
TABLE_DATA = REPO / "tests" / "fixtures" / "table_data"
HELP_BASE = Path(__file__).resolve().parent / "data" / "help_base.out"
# `-H pa -k 3 -a gomea --gomea-pop-size 16 -m 3000 -r 5` stdout, captured at
# commit ce698be, before the Phase 1d ablation options existed.
GOMEA_PHASE1_REF = Path(__file__).resolve().parent / "data" / "gomea_pa3_pop16_m3000_r5.out"
# `-H maj -k 5 -a gomea --gomea-pop-size 16 --gomea-skip-inactive 0 -m 2000
# -r 1002` stdout, captured at commit 9472b66, before the Phase 1e options.
GOMEA_MAJ5_REF = (
    Path(__file__).resolve().parent / "data" / "gomea_maj5_pop16_skip0_m2000_r1002.out"
)

# ---------------------------------------------------------------------------
# -a hc stdout captured at commit 0410881, before any GOMEA code existed.
# ---------------------------------------------------------------------------

# moses -H pa -k 3 -m 2000 -r 1
_HC_PA3_M2000_R1 = (
    "-2 and(or(!$1 !$2) or($1 $2) $3) \n"
    "-2 and(or(and(!$1 !$2) $3) or(!$1 !$2)) \n"
    "-2 and(or(and(!$1 $3) !$2) or(!$1 $3)) \n"
    "-2 and(or(and(!$2 $3) !$1) or(!$2 $3)) \n"
    "-2 or(and(or(!$1 !$2) $3) and(!$1 !$2)) \n"
    "-2 or(and(or(!$1 $3) !$2) and(!$1 $3)) \n"
    "-2 or(and(or(!$2 $3) !$1) and(!$2 $3)) \n"
    "-2 and(or(and(!$1 $3) !$2) or(!$1 $2 $3)) \n"
    "-2 and(or(and(!$2 $3) !$1) or($1 !$2 $3)) \n"
    "-2 and(or(!$1 !$2) or(!$1 $3) or(!$2 $3)) \n"
)

# moses -H maj -k 3 -m 1500 -r 2
_HC_MAJ3_M1500_R2 = (
    "0 or(and(or($2 $3) $1) and($2 $3)) \n"
    "-1 and(or($1 $2) $3) \n"
    "-1 and(or($1 $3) $2) \n"
    "-1 and(or($2 $3) $1) \n"
    "-1 or(and($1 $2) $3) \n"
    "-1 or(and($1 $3) $2) \n"
    "-1 or(and($2 $3) $1) \n"
    "-1 or(and(or($2 $3) $1) $2) \n"
    "-1 or(and(or($2 $3) $1) $3) \n"
    "-2 $1 \n"
)

# moses -i tests/fixtures/table_data/dataset.csv -m 1000
_HC_DATASET_M1000 = (
    "-1 $2 \n"
    "-1 and($1 $2) \n"
    "-2 $1 \n"
    "-2 and(!$1 $2) \n"
    "-2 or(!$1 $2) \n"
    "-2 or($1 $2) \n"
    "-3 true \n"
    "-3 !$1 \n"
    "-3 and(!$1 !$2) \n"
    "-3 and($1 !$2) \n"
)


def _run(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = moses_exec(["moses", *argv])
    return rc, buf.getvalue()


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Keep ``moses.log`` out of the repo and help output uncoloured at 80 columns."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("PYTHON_COLORS", "0")
    monkeypatch.setenv("NO_COLOR", "1")


# ---------------------------------------------------------------------------
# The full lifecycle: several expansions, one calibration per schema.
# ---------------------------------------------------------------------------

# With the default settings one pa -k 3 deme (24 fields) spends a 3000-eval
# budget on its own in three generations, so the per-deme cap is what leaves
# room for further expansions here.
_PA3_GOMEA = [
    "-H", "pa", "-k", "3", "-a", "gomea", "-m", "3000", "-r", "1",
    "--gomea-max-evals-per-deme", "1000",
]  # fmt: skip


@pytest.fixture(scope="module")
def pa3_gomea_runs(tmp_path_factory):
    """The same ``-a gomea`` command run twice, each with its own log."""
    out = []
    for i in range(2):
        d = tmp_path_factory.mktemp(f"pa3_gomea_{i}")
        log = d / "gomea.jsonl"
        with pytest.MonkeyPatch.context() as mp:
            mp.chdir(d)
            rc, stdout = _run([*_PA3_GOMEA, "--gomea-log", str(log)])
        out.append((rc, stdout, log))
    return out


def test_gomea_run_spans_several_expansions_with_independent_calibrations(pa3_gomea_runs):
    rc, stdout, log = pa3_gomea_runs[0]
    assert rc == 0
    assert len(stdout.splitlines()) == 10
    events = _events(log)
    starts = [e for e in events if e["type"] == "invocation_start"]
    assert len(starts) >= 2, "expected at least two deme expansions"
    assert len({e["schema_id"] for e in starts}) >= 2

    # Every invocation calibrates its own population: an "init" event follows
    # each "invocation_start", for that invocation's own schema, before any
    # other invocation starts.
    current = None
    inits_per_invocation = []
    for e in events:
        if e["type"] == "invocation_start":
            current = e["schema_id"]
            inits_per_invocation.append(0)
        elif e["type"] == "init":
            assert e["schema_id"] == current
            inits_per_invocation[-1] += 1
    assert inits_per_invocation == [1] * len(starts)
    assert sum(e["type"] == "end" for e in events) == len(starts)


def test_gomea_run_is_deterministic(pa3_gomea_runs):
    (rc1, out1, _), (rc2, out2, _) = pa3_gomea_runs
    assert rc1 == rc2 == 0
    assert out1 == out2


# ---------------------------------------------------------------------------
# -a hc is untouched.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["-H", "pa", "-k", "3", "-m", "2000", "-r", "1"], _HC_PA3_M2000_R1),
        (["-H", "maj", "-k", "3", "-m", "1500", "-r", "2"], _HC_MAJ3_M1500_R2),
        (["-i", str(TABLE_DATA / "dataset.csv"), "-m", "1000"], _HC_DATASET_M1000),
    ],
    ids=["pa3", "maj3", "dataset"],
)
def test_hc_output_is_byte_identical_to_base(argv, expected):
    rc, stdout = _run(argv)
    assert rc == 0
    assert stdout == expected


def _help_text() -> str:
    buf = io.StringIO()
    with redirect_stdout(buf), pytest.raises(SystemExit) as exc:
        moses_exec(["moses", "--help"])
    assert exc.value.code == 1
    return buf.getvalue()


def test_help_only_adds_gomea_lines():
    base = HELP_BASE.read_text().splitlines()
    new = _help_text().splitlines()

    # Every base line survives unchanged and in the same relative order.
    added = []
    j = 0
    for line in new:
        if j < len(base) and line == base[j]:
            j += 1
        else:
            added.append(line)
    assert j == len(base), f"base help line missing or reordered: {base[j]!r}"

    assert added, "no --gomea-* options appear in --help"
    # A wrapped continuation of a gomea option's help text (argparse puts
    # help on its own indented line when the invocation is long, and wraps
    # the text itself across several such lines) mentions neither "gomea"
    # nor any flag of its own; it is allowed only right after a line that
    # does mention gomea (a flag line or an earlier continuation of one).
    prev_is_gomea = False
    for line in added:
        is_gomea = "gomea" in line.lower()
        is_continuation = (
            not is_gomea
            and prev_is_gomea
            and line.startswith(" ")
            and not line.lstrip().startswith("-")
        )
        assert is_gomea or is_continuation, f"non-GOMEA line added to --help: {line!r}"
        prev_is_gomea = is_gomea or is_continuation
    for flag in (
        "--gomea-pop-size",
        "--gomea-fos",
        "--gomea-fi",
        "--gomea-radii",
        "--gomea-skip-inactive",
        "--gomea-max-gens",
        "--gomea-stall-gens",
        "--gomea-max-evals-per-deme",
        "--gomea-log",
        "--gomea-init",
        "--gomea-accept",
        "--gomea-genotype-memo",
        "--gomea-stall-evals",
    ):
        assert any(line.strip().startswith(flag + " ") for line in added), flag


# ---------------------------------------------------------------------------
# The printed metapopulation agrees with the scorer.
# ---------------------------------------------------------------------------

_X1_LINE = re.compile(
    r"^(?P<score>\S+) (?P<prog>.*?)  \[score=(?P<s>[^,]+), penalized score=(?P<p>[^,]+), "
    r"complexity=(?P<c>[^,]+), complexity penalty=[^,]+, uniformity penalty=[^\]]+\]$"
)


def test_printed_candidates_rescore_to_their_printed_scores():
    rc, stdout = _run(["-H", "pa", "-k", "3", "-a", "gomea", "-m", "1000", "-r", "1", "-x1"])
    assert rc == 0
    lines = stdout.splitlines()
    assert lines

    pms = parse_args(["-H", "pa"])
    bscore = logical_bscore(even_parity, 3)
    sig = gen_signature(type_node.boolean_type, 3)
    _set_noise_or_ratio(
        bscore, _alphabet_size(sig, pms.ignore_ops), pms.noise, pms.complexity_ratio
    )
    cscore = behave_cscore(bscore)

    for line in lines:
        m = _X1_LINE.match(line)
        assert m, line
        cs = cscore.get_cscore(str_to_combo_tree(m["prog"]))
        assert float(m["score"]) == float(m["s"]) == cs.get_score()
        assert float(m["p"]) == cs.get_penalized_score()
        assert int(m["c"]) == cs.get_complexity()


# ---------------------------------------------------------------------------
# Unsupported problems and targets.
# ---------------------------------------------------------------------------


def test_regression_problem_is_rejected_with_a_clear_message():
    with pytest.raises(UnsupportedRepresentation) as exc:
        _run(
            [
                "-i",
                str(TABLE_DATA / "regression_sample.csv"),
                "-u",
                "y",
                "-a",
                "gomea",
                "-m",
                "300",
            ]
        )
    msg = str(exc.value)
    assert "gomea" in msg
    assert "contin" in msg


def test_reaching_the_target_ends_with_target_reached(tmp_path):
    log = tmp_path / "gomea.jsonl"
    rc, stdout = _run(
        ["-H", "pa", "-k", "2", "-a", "gomea", "-m", "2000", "-r", "1", "--gomea-log", str(log)]
    )
    assert rc == 0
    assert stdout.splitlines()[0].startswith("0 ")
    reasons = [e["termination_reason"] for e in _events(log) if e["type"] == "end"]
    assert "target_reached" in reasons


# ---------------------------------------------------------------------------
# Abbreviated classic options resolve as they did before GOMEA.
# ---------------------------------------------------------------------------


def _cli_parser(with_gomea: bool) -> argparse.ArgumentParser:
    """The parser ``moses_exec`` builds, with or without the GOMEA options."""
    pmr = problem_manager()
    tpp = register_table_problems(pmr)
    register_demo_problems(pmr)
    parser = argparse.ArgumentParser(prog="moses", add_help=False)
    parser.add_argument("-h", "--help", dest="help", action="store_true", default=False)
    pp = problem_params()
    pp.add_options(parser)
    tpp.add_options(parser)
    ip_problem_params().add_options(parser)
    _add_demo_options(parser)
    if with_gomea:
        pp.add_gomea_options(parser)
    return parser


def _resolve(parser: argparse.ArgumentParser, token: str) -> dict | None:
    """What ``token`` (plus a value, if its option takes one) parses to, or
    None when argparse rejects it (ambiguous or unknown)."""
    for action in parser._actions:
        action.default = argparse.SUPPRESS
    argv = [token]
    matches = [a for o, a in parser._option_string_actions.items() if o.startswith(token)]
    if matches and matches[0].nargs != 0:
        argv.append("1")
    err = io.StringIO()
    try:
        with redirect_stderr(err):
            return vars(parser.parse_args(argv))
    except SystemExit:
        return None


def test_base_parser_is_the_captured_classic_surface():
    longs = {o for o in _cli_parser(False)._option_string_actions if o.startswith("--")}
    base = set(re.findall(r"(?<![\w-])--[a-zA-Z][\w-]*", HELP_BASE.read_text()))
    assert longs == base


def test_every_classic_abbreviation_still_resolves_to_the_same_option():
    base, new = _cli_parser(False), _cli_parser(True)
    longs = [o for o in base._option_string_actions if o.startswith("--")]
    checked = 0
    for opt in longs:
        for end in range(3, len(opt) + 1):
            prefix = opt[:end]
            before = _resolve(base, prefix)
            if before is None:
                continue
            checked += 1
            assert _resolve(new, prefix) == before, prefix
    assert checked > len(longs)


def test_g_still_abbreviates_gen_best_tree():
    rc, stdout = _run(["-H", "pa", "-k", "2", "-m", "50", "--g", "1"])
    assert rc == 0
    assert _resolve(_cli_parser(True), "--g") == {"gen_best_tree": True}
    ns = _cli_parser(True).parse_args(["--g", "1"])
    assert ns.gen_best_tree is True
    assert not any(k.startswith("_classic_prefix_alias") for k in vars(ns))


def test_hidden_aliases_do_not_show_in_help():
    assert "--g " not in _help_text() and "--g]" not in _help_text()


# ---------------------------------------------------------------------------
# Option validation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        ["--gomea-fos", "xx"],
        ["--gomea-fi", "2"],
        ["--gomea-skip-inactive", "-1"],
        ["--gomea-pop-size", "1"],
        ["--gomea-max-evals-per-deme", "-3"],
        ["--gomea-radii", "nonsense"],
        ["--gomea-init", "random"],
        ["--gomea-accept", "greedy"],
        ["--gomea-genotype-memo", "2"],
        ["--gomea-stall-evals", "-1"],
    ],
)
def test_bad_gomea_options_are_rejected(bad):
    with pytest.raises(ValueError):
        _run(["-H", "pa", "-k", "2", "-a", "gomea", "-m", "10", *bad])


# ---------------------------------------------------------------------------
# Phase 1d ablation options: defaults reproduce Phase 1 bit for bit.
# ---------------------------------------------------------------------------

_PA3_POP16 = ["-H", "pa", "-k", "3", "-a", "gomea", "--gomea-pop-size", "16", "-m", "3000",
              "-r", "5"]  # fmt: skip


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--gomea-init", "sampler", "--gomea-accept", "nonworse", "--gomea-genotype-memo", "0"],
    ],
    ids=["implicit-defaults", "explicit-defaults"],
)
def test_gomea_defaults_reproduce_phase1_stdout(extra):
    rc, stdout = _run([*_PA3_POP16, *extra])
    assert rc == 0
    assert stdout == GOMEA_PHASE1_REF.read_text()


def test_ablation_options_parse_and_reach_gomea_parameters():
    from pymoses.moses.main.problem_params import _gomea_params_from_args

    ns = _cli_parser(True).parse_args(
        ["--gomea-init", "uniform", "--gomea-accept", "strict", "--gomea-genotype-memo", "1"]
    )
    gp = _gomea_params_from_args(ns)
    assert (gp.init_mode, gp.acceptance, gp.genotype_memo) == ("uniform", "strict", True)
    gp0 = _gomea_params_from_args(_cli_parser(True).parse_args([]))
    assert (gp0.init_mode, gp0.acceptance, gp0.genotype_memo) == ("sampler", "nonworse", False)
    assert _gomea_params_from_args(argparse.Namespace()).init_mode == "sampler"


@pytest.mark.parametrize(
    "extra",
    [["--gomea-init", "uniform"], ["--gomea-accept", "strict"], ["--gomea-genotype-memo", "1"]],
)
def test_each_ablation_option_runs_end_to_end(extra, tmp_path):
    log = tmp_path / "g.jsonl"
    argv = ["-H", "pa", "-k", "3", "-a", "gomea", "-m", "600", "-r", "1", "--gomea-log", str(log)]
    rc, stdout = _run([*argv, *extra])
    assert rc == 0 and stdout
    events = _events(log)
    start = next(e for e in events if e["type"] == "invocation_start")
    assert {"init_mode", "acceptance", "genotype_memo"} <= set(start["params"])
    ends = [e for e in events if e["type"] == "end"]
    assert ends and all("memo_hits" in e["ledger"] for e in ends)


# ---------------------------------------------------------------------------
# Phase 1e: the stall rule and the -a hc-gomea hybrid.
# ---------------------------------------------------------------------------

_MAJ5_GOMEA = ["-H", "maj", "-k", "5", "-a", "gomea", "--gomea-pop-size", "16",
               "--gomea-skip-inactive", "0", "-m", "2000", "-r", "1002"]  # fmt: skip


@pytest.mark.parametrize("extra", [[], ["--gomea-stall-evals", "0"]], ids=["implicit", "explicit"])
def test_gomea_defaults_reproduce_the_phase1d_maj5_stdout(extra):
    rc, stdout = _run([*_MAJ5_GOMEA, *extra])
    assert rc == 0
    assert stdout == GOMEA_MAJ5_REF.read_text()


def test_explicit_hc_reproduces_the_base_stdout():
    rc, stdout = _run(["-H", "pa", "-k", "3", "-a", "hc", "-m", "2000", "-r", "1"])
    assert rc == 0
    assert stdout == _HC_PA3_M2000_R1


def test_stall_option_parses_and_reaches_gomea_parameters():
    from pymoses.moses.main.problem_params import _gomea_params_from_args

    ns = _cli_parser(True).parse_args(["--gomea-stall-evals", "250"])
    assert _gomea_params_from_args(ns).stall_evals == 250
    assert _gomea_params_from_args(_cli_parser(True).parse_args([])).stall_evals == 0
    assert _gomea_params_from_args(argparse.Namespace()).stall_evals == 0


def test_stall_option_runs_end_to_end(tmp_path):
    log = tmp_path / "g.jsonl"
    rc, stdout = _run(["-H", "pa", "-k", "3", "-a", "gomea", "-m", "1500", "-r", "1",
                       "--gomea-stall-evals", "60", "--gomea-log", str(log)])  # fmt: skip
    assert rc == 0 and stdout
    ends = [e for e in _events(log) if e["type"] == "end"]
    assert "stall_evals" in {e["termination_reason"] for e in ends}


def test_hc_gomea_equals_hc_when_hill_climbing_solves_the_problem():
    base = ["-H", "pa", "-k", "2", "-m", "2000", "-r", "1"]
    rc_hc, out_hc = _run([*base, "-a", "hc"])
    rc_hy, out_hy = _run([*base, "-a", "hc-gomea"])
    assert rc_hc == rc_hy == 0
    assert out_hy == out_hc


def test_hc_gomea_full_cli_run_with_telemetry(tmp_path):
    log = tmp_path / "g.jsonl"
    rc, stdout = _run(["-H", "maj", "-k", "5", "-a", "hc-gomea", "--gomea-fos", "uni",
                       "--gomea-max-evals-per-deme", "500", "-m", "3000", "-r", "3",
                       "--gomea-log", str(log)])  # fmt: skip
    assert rc == 0 and stdout.strip()
    events = _events(log)
    hybrid = [e for e in events if e["type"] == "hybrid"]
    assert hybrid
    for e in hybrid:
        assert {"hc_evals", "hc_stop", "gomea_ran", "gomea_evals",
                "gomea_termination_reason"} <= set(e)  # fmt: skip
        assert e["hc_evals"] + e["gomea_evals"] <= e["max_evals"]
        assert e["gomea_evals"] <= 500
    assert any(e["gomea_ran"] for e in hybrid)
    ends = [e for e in events if e["type"] == "end"]
    assert len(ends) == sum(e["gomea_ran"] for e in hybrid)
