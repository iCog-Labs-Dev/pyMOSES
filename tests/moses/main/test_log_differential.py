"""The port's `-l DEBUG` log, compared record for record against classic's.

This is the oracle for the logging surface. `tools/capture_moses_log.py`
captured classic's own log for each configuration into
`tests/fixtures/moses_log.json`, reduced to normalised `LEVEL<TAB>MESSAGE`
records (`tools/normalize_log.py` -- timestamp dropped, hostname/paths/log
name/argv[0]/microsecond/RAM/elapsed columns replaced). This test runs the
PORT with the same argv, normalises the same way, removes the enumerated
allow-list below from classic's side, and asserts the two sequences are
EQUAL -- same records, same order.

## Why an allow-list, and why it is enumerated line by line

Some records classic emits have no Python behind them yet. Tolerating them
with a "the port may emit a subsequence" comparison would also tolerate a
record silently disappearing, or appearing in the wrong place, which is
exactly the failure this test exists to catch. So each absent record is
listed individually, with the reason it is absent, and `apply_allow_list`
REPORTS any entry that no longer matches -- so the list can only shrink as
call sites land, never quietly grow to cover a regression.
`test_allow_list_has_no_stale_entries` fails on a stale entry.

## What this has already caught

The `prerec` configuration is not decoration. Comparing it against classic
found a real numeric divergence that the stdout differentials could not
see: classic logged `Score: -nan` for a candidate selecting no rows, where
the port logged `Score: -3.40282e+38`, because `prerec_bscore` divided with
Python's `/` (which raises on 0/0) instead of IEEE division, and the
exception was caught upstream and turned into the worst possible score. The
two runs then kept different numbers of candidates -- 25 vs 23 -- so this
was a search difference, not a cosmetic one.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.capture_moses_log import resolve_argv  # noqa: E402
from tools.normalize_log import apply_allow_list, normalize_log  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "moses_log.json"

# --- the reasons an allow-list entry can carry ------------------------------

BUILD_KNOBS = (
    "build_knobs.cc's own `logger().debug()` calls (cc:535, :584, :589, :590, "
    ":625, :626, :635, :642, :671, :683, :684, :686). Their enclosing "
    "functions -- `add_logical_knobs` and `build_logical` -- ARE ported, so "
    "these are portable; `pymetta/moses/representation/build_knobs.py` is "
    "owned by a different task and was not to be edited."
)
MOSES_MAIN_CC = (
    "moses_main.cc:82-94 `adjust_termination_criteria`. The function IS "
    "ported (`pymetta/moses/moses/moses_main.py`), so these are portable; "
    "that file is owned by a different task and was not to be edited."
)
MOSES_MAIN_H = (
    "moses_main.h:225, inside `metapop_printer::operator()`'s "
    "`is_info_enabled()` block. Same file-ownership reason as MOSES_MAIN_CC."
)
DEME_ID = (
    "metapopulation.cc:93 streams `xmplr.get_demeID()`, and this port's "
    "`scored_combo_tree` carries no `demeID_t` at all -- the same gap that "
    "gates `--output-deme-id` (problem_params.py:1116-1126). NOT portable "
    "without adding that field; see metapopulation.py's own DIVERGENCE "
    "marker on `log_selected_exemplar`."
)

REASONS = (BUILD_KNOBS, MOSES_MAIN_CC, MOSES_MAIN_H, DEME_ID)

# One entry per OCCURRENCE, in the order classic emits them, per config.
ALLOW_LIST: dict[str, list[tuple[str, str]]] = {
    "it_dataset_m300": [
        (MOSES_MAIN_CC, 'INFO\tInferred target score = -3'),
        (MOSES_MAIN_CC, 'INFO\tSubtract 0.5 (minimum significant improvement) from the target score to deal with float imprecision = -3.5'),  # noqa: E501
        (DEME_ID, 'DEBUG\tSelected the 1th exemplar, from deme 0, for the 1th time(s)'),
        (BUILD_KNOBS, 'DEBUG\tFirst call to add_logical_knobs'),
        (BUILD_KNOBS, 'DEBUG\tStarted knob building at: or(and) TopNode: or'),
        (BUILD_KNOBS, 'DEBUG\tAdding knobs to tree: or(and)'),
        (BUILD_KNOBS, 'DEBUG\tAdding 4 logical knobs'),
        (BUILD_KNOBS, 'DEBUG\tAdded knobs to tree: or(and null_vertex($1) null_vertex($2) null_vertex(and(!$1 $2)) null_vertex(and($1 $2)))'),  # noqa: E501
        (BUILD_KNOBS, 'DEBUG\tGetting out of add_logical_knbos'),
        (BUILD_KNOBS, 'DEBUG\tSibiling: and TopNode: and'),
        (BUILD_KNOBS, 'DEBUG\tFirst call to add_logical_knobs'),
        (BUILD_KNOBS, 'DEBUG\tStarted knob building at: and TopNode: and'),
        (BUILD_KNOBS, 'DEBUG\tAdding knobs to tree: and'),
        (BUILD_KNOBS, 'DEBUG\tAdding 4 logical knobs'),
        (BUILD_KNOBS, 'DEBUG\tAdded knobs to tree: and(null_vertex($1) null_vertex($2) null_vertex(or(!$1 $2)) null_vertex(or($1 $2)))'),  # noqa: E501
        (BUILD_KNOBS, 'DEBUG\tGetting out of add_logical_knbos'),
        (BUILD_KNOBS, 'DEBUG\tSibiling: null_vertex($1) TopNode: null_vertex'),
        (BUILD_KNOBS, 'DEBUG\tCall add_logical_knobs for flipped subtree'),
        (BUILD_KNOBS, 'DEBUG\tSubtree: or(and(null_vertex($1) null_vertex($2) null_vertex(or(!$1 $2)) null_vertex(or($1 $2))) null_vertex($1) null_vertex($2) null_vertex(and(!$1 $2)) null_vertex(and($1 $2)))'),  # noqa: E501
        (BUILD_KNOBS, 'DEBUG\tAdding knobs to tree: or'),
        (BUILD_KNOBS, 'DEBUG\tAdding 4 logical knobs'),
        (BUILD_KNOBS, 'DEBUG\tAdded knobs to tree: or(null_vertex($1) null_vertex($2) null_vertex(and($1 $2)) null_vertex(and(!$1 $2)))'),  # noqa: E501
        (BUILD_KNOBS, 'DEBUG\tGetting out of add_logical_knbos'),
        (BUILD_KNOBS, 'DEBUG\tSubtree: or(and(null_vertex($1) null_vertex($2) null_vertex(or(!$1 $2)) null_vertex(or($1 $2)) or(null_vertex($1) null_vertex($2) null_vertex(and($1 $2)) null_vertex(and(!$1 $2)))) null_vertex($1) null_vertex($2) null_vertex(and(!$1 $2)) null_vertex(and($1 $2)))'),  # noqa: E501
        (BUILD_KNOBS, 'DEBUG\tAdded logical knobs on: and(null_vertex($1) null_vertex($2) null_vertex(or(!$1 $2)) null_vertex(or($1 $2)) or(null_vertex($1) null_vertex($2) null_vertex(and($1 $2)) null_vertex(and(!$1 $2))))'),  # noqa: E501
        (BUILD_KNOBS, 'DEBUG\tSibiling: null_vertex($1) TopNode: null_vertex'),
        (BUILD_KNOBS, 'DEBUG\tCall add_logical_knobs for flipped subtree'),
        (BUILD_KNOBS, 'DEBUG\tSubtree: or(and(null_vertex($1) null_vertex($2) null_vertex(or(!$1 $2)) null_vertex(or($1 $2)) or(null_vertex($1) null_vertex($2) null_vertex(and($1 $2)) null_vertex(and(!$1 $2)))) null_vertex($1) null_vertex($2) null_vertex(and(!$1 $2)) null_vertex(and($1 $2)))'),  # noqa: E501
        (BUILD_KNOBS, 'DEBUG\tAdding knobs to tree: and'),
        (BUILD_KNOBS, 'DEBUG\tAdding 4 logical knobs'),
        (BUILD_KNOBS, 'DEBUG\tAdded knobs to tree: and(null_vertex($1) null_vertex($2) null_vertex(or($1 $2)) null_vertex(or(!$1 $2)))'),  # noqa: E501
        (BUILD_KNOBS, 'DEBUG\tGetting out of add_logical_knbos'),
        (BUILD_KNOBS, 'DEBUG\tSubtree: or(and(null_vertex($1) null_vertex($2) null_vertex(or(!$1 $2)) null_vertex(or($1 $2)) or(null_vertex($1) null_vertex($2) null_vertex(and($1 $2)) null_vertex(and(!$1 $2)))) null_vertex($1) null_vertex($2) null_vertex(and(!$1 $2)) null_vertex(and($1 $2)) and(null_vertex($1) null_vertex($2) null_vertex(or($1 $2)) null_vertex(or(!$1 $2))))'),  # noqa: E501
        (MOSES_MAIN_H, 'INFO\tBest candidate:\\n-3 and($1 $2)  [score=-3, penalized score=-3.57142853736877441, complexity=2, complexity penalty=0.571428596973419189, uniformity penalty=0] demeID: 1\\n'),  # noqa: E501
    ],
    "prerec_dataset_m300": [
        (MOSES_MAIN_CC, 'INFO\tInferred target score = 0.666667'),
        (MOSES_MAIN_CC, 'INFO\tSubtract 0.111111 (minimum significant improvement) from the target score to deal with float imprecision = 0.555556'),  # noqa: E501
        (DEME_ID, 'DEBUG\tSelected the 1th exemplar, from deme 0, for the 1th time(s)'),
        (BUILD_KNOBS, 'DEBUG\tFirst call to add_logical_knobs'),
        (BUILD_KNOBS, 'DEBUG\tStarted knob building at: or(and) TopNode: or'),
        (BUILD_KNOBS, 'DEBUG\tAdding knobs to tree: or(and)'),
        (BUILD_KNOBS, 'DEBUG\tAdding 4 logical knobs'),
        (BUILD_KNOBS, 'DEBUG\tAdded knobs to tree: or(and null_vertex($1) null_vertex($2) null_vertex(and(!$1 $2)) null_vertex(and($1 $2)))'),  # noqa: E501
        (BUILD_KNOBS, 'DEBUG\tGetting out of add_logical_knbos'),
        (BUILD_KNOBS, 'DEBUG\tSibiling: and TopNode: and'),
        (BUILD_KNOBS, 'DEBUG\tFirst call to add_logical_knobs'),
        (BUILD_KNOBS, 'DEBUG\tStarted knob building at: and TopNode: and'),
        (BUILD_KNOBS, 'DEBUG\tAdding knobs to tree: and'),
        (BUILD_KNOBS, 'DEBUG\tAdding 4 logical knobs'),
        (BUILD_KNOBS, 'DEBUG\tAdded knobs to tree: and(null_vertex($1) null_vertex($2) null_vertex(or(!$1 $2)) null_vertex(or($1 $2)))'),  # noqa: E501
        (BUILD_KNOBS, 'DEBUG\tGetting out of add_logical_knbos'),
        (BUILD_KNOBS, 'DEBUG\tSibiling: null_vertex($1) TopNode: null_vertex'),
        (BUILD_KNOBS, 'DEBUG\tCall add_logical_knobs for flipped subtree'),
        (BUILD_KNOBS, 'DEBUG\tSubtree: or(and(null_vertex($1) null_vertex($2) null_vertex(or(!$1 $2)) null_vertex(or($1 $2))) null_vertex($1) null_vertex($2) null_vertex(and(!$1 $2)) null_vertex(and($1 $2)))'),  # noqa: E501
        (BUILD_KNOBS, 'DEBUG\tAdding knobs to tree: or'),
        (BUILD_KNOBS, 'DEBUG\tAdding 4 logical knobs'),
        (BUILD_KNOBS, 'DEBUG\tAdded knobs to tree: or(null_vertex($1) null_vertex($2) null_vertex(and($1 $2)) null_vertex(and(!$1 $2)))'),  # noqa: E501
        (BUILD_KNOBS, 'DEBUG\tGetting out of add_logical_knbos'),
        (BUILD_KNOBS, 'DEBUG\tSubtree: or(and(null_vertex($1) null_vertex($2) null_vertex(or(!$1 $2)) null_vertex(or($1 $2)) or(null_vertex($1) null_vertex($2) null_vertex(and($1 $2)) null_vertex(and(!$1 $2)))) null_vertex($1) null_vertex($2) null_vertex(and(!$1 $2)) null_vertex(and($1 $2)))'),  # noqa: E501
        (BUILD_KNOBS, 'DEBUG\tAdded logical knobs on: and(null_vertex($1) null_vertex($2) null_vertex(or(!$1 $2)) null_vertex(or($1 $2)) or(null_vertex($1) null_vertex($2) null_vertex(and($1 $2)) null_vertex(and(!$1 $2))))'),  # noqa: E501
        (BUILD_KNOBS, 'DEBUG\tSibiling: null_vertex($1) TopNode: null_vertex'),
        (BUILD_KNOBS, 'DEBUG\tCall add_logical_knobs for flipped subtree'),
        (BUILD_KNOBS, 'DEBUG\tSubtree: or(and(null_vertex($1) null_vertex($2) null_vertex(or(!$1 $2)) null_vertex(or($1 $2)) or(null_vertex($1) null_vertex($2) null_vertex(and($1 $2)) null_vertex(and(!$1 $2)))) null_vertex($1) null_vertex($2) null_vertex(and(!$1 $2)) null_vertex(and($1 $2)))'),  # noqa: E501
        (BUILD_KNOBS, 'DEBUG\tAdding knobs to tree: and'),
        (BUILD_KNOBS, 'DEBUG\tAdding 4 logical knobs'),
        (BUILD_KNOBS, 'DEBUG\tAdded knobs to tree: and(null_vertex($1) null_vertex($2) null_vertex(or($1 $2)) null_vertex(or(!$1 $2)))'),  # noqa: E501
        (BUILD_KNOBS, 'DEBUG\tGetting out of add_logical_knbos'),
        (BUILD_KNOBS, 'DEBUG\tSubtree: or(and(null_vertex($1) null_vertex($2) null_vertex(or(!$1 $2)) null_vertex(or($1 $2)) or(null_vertex($1) null_vertex($2) null_vertex(and($1 $2)) null_vertex(and(!$1 $2)))) null_vertex($1) null_vertex($2) null_vertex(and(!$1 $2)) null_vertex(and($1 $2)) and(null_vertex($1) null_vertex($2) null_vertex(or($1 $2)) null_vertex(or(!$1 $2))))'),  # noqa: E501
        (MOSES_MAIN_H, 'INFO\tBest candidate:\\n0.666667 and($1 $2)  [score=0.666666686534881592, penalized score=0.60317462682723999, complexity=2, complexity penalty=0.0634920671582221985, uniformity penalty=0] demeID: 1\\n'),  # noqa: E501
    ],
}

# Measured 2026-09-09. Asserted, not merely printed, so a change in any of
# them has to be acknowledged rather than noticed.
EXPECTED_COUNTS = {
    "it_dataset_m300": (173, 34, 139),
    "prerec_dataset_m300": (183, 34, 149),
}

CONFIGS = sorted(EXPECTED_COUNTS)


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def port_records(tmp_path_factory, golden) -> dict[str, list[str]]:
    """Run the port's own CLI once per configuration and normalise its log.

    A SUBPROCESS, not an in-process call: `logger()` is a process-wide
    singleton holding an open file handle, exactly as in C++, so running the
    CLI inside the pytest process would leak its log destination into every
    other test in the session (and inherit whatever level a previous test
    left set).
    """
    tmp = tmp_path_factory.mktemp("logdiff")
    code = (
        "import sys\n"
        "from pymetta.moses.main.moses_exec import moses_exec\n"
        "sys.exit(moses_exec(['moses'] + sys.argv[1:]))\n"
    )
    out = {}
    for name in CONFIGS:
        log = tmp / f"{name}.log"
        argv = resolve_argv(golden[name]["argv"], REPO_ROOT)
        proc = subprocess.run(
            [sys.executable, "-c", code, *argv, "-l", "DEBUG", "-f", str(log)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr[-3000:]
        out[name] = normalize_log(log.read_text())
    return out


@pytest.mark.parametrize("config", CONFIGS)
def test_port_log_matches_classic_record_for_record(config, golden, port_records):
    """Every record classic emits, minus the allow-list, in classic's order."""
    expected, stale = apply_allow_list(
        golden[config]["records"], [rec for _, rec in ALLOW_LIST[config]]
    )
    assert not stale, f"stale allow-list entries: {stale}"
    assert port_records[config] == expected


@pytest.mark.parametrize("config", CONFIGS)
def test_allow_list_has_no_stale_entries(config, golden):
    """An allow-list entry that no longer matches is a bug in the list.

    This is what makes the list shrink visibly. If a call site lands and its
    record starts being emitted, the entry stops matching classic's side and
    is reported here rather than sitting inert.
    """
    _, stale = apply_allow_list(
        golden[config]["records"], [rec for _, rec in ALLOW_LIST[config]]
    )
    assert stale == []


@pytest.mark.parametrize("config", CONFIGS)
def test_measured_counts(config, golden, port_records):
    classic_n, allowed_n, port_n = EXPECTED_COUNTS[config]
    assert len(golden[config]["records"]) == classic_n
    assert len(ALLOW_LIST[config]) == allowed_n
    assert len(port_records[config]) == port_n
    assert port_n == classic_n - allowed_n


@pytest.mark.parametrize("config", CONFIGS)
def test_every_allowed_record_carries_a_reason(config):
    for reason, rec in ALLOW_LIST[config]:
        assert reason in REASONS, rec


@pytest.mark.parametrize("config", CONFIGS)
def test_no_port_record_is_unexplained(config, golden, port_records):
    """Zero records the port emits that classic does not.

    The equality test above already implies this; it is stated separately
    because "unexplained must be zero" is the exit criterion's own wording,
    and a separate assertion says which direction failed.
    """
    extra = [rec for rec in port_records[config] if rec not in golden[config]["records"]]
    assert extra == []


def test_the_configs_cover_more_than_one_scorer(golden):
    """A one-configuration corpus is how a scorer-specific divergence hides.

    `prerec` was added after the `it` corpus was already passing, and it
    immediately found one (see this module's docstring). This asserts the
    corpus keeps at least two distinct problem types rather than quietly
    collapsing back to one.
    """
    problems = set()
    for name in CONFIGS:
        argv = golden[name]["argv"]
        problems.add(next((a for a in argv if a.startswith("-H")), "-Hit"))
    assert len(problems) >= 2


# --- the normaliser's own guarantees, and the format they rest on -----------

_TIMESTAMP_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}:\d{3}\] \[[A-Z]+\] ")


def test_record_prefix_uses_a_colon_before_milliseconds(tmp_path):
    """`"[%s:%03ld] "` (Logger.cc:512) -- a COLON, not a dot.

    Reads the port's own raw log rather than a synthesised string, so the
    assertion covers the real `Logger.log` path.
    """
    from pymetta.util.logger import Logger

    log = tmp_path / "fmt.log"
    lg = Logger(str(log), Logger.Level.DEBUG)
    lg.info("hello")
    line = log.read_text().splitlines()[0]
    assert _TIMESTAMP_RE.match(line), line
    assert "." not in line.split("]")[0]


def test_level_filtering_drops_more_verbose_records(tmp_path):
    """`if (level > currentLevel) return;` (Logger.cc:496)."""
    from pymetta.util.logger import Logger

    log = tmp_path / "lvl.log"
    lg = Logger(str(log), Logger.Level.INFO)
    lg.error("e")
    lg.warn("w")
    lg.info("i")
    lg.debug("d")
    lg.fine("f")
    body = log.read_text()
    assert "] e" in body and "] w" in body and "] i" in body
    assert "] d" not in body and "] f" not in body


def test_a_run_that_logs_nothing_leaves_no_file(tmp_path):
    """The file is opened on the first record, not on `set_filename`.

    Logger.cc:284-286's own comment: "this allows us to set the main
    logger's filename without creating a useless log file with the default
    filename". `problem_params` depends on it, since the singleton is built
    pointing at `opencog.log` long before `-f` is parsed.
    """
    from pymetta.util.logger import Logger

    log = tmp_path / "never.log"
    lg = Logger(str(log), Logger.Level.NONE)
    lg.error("suppressed")
    assert not log.exists()


def test_a_negative_nan_prints_with_its_sign(tmp_path):
    """libstdc++ honours NaN's sign bit; Python's `format` does not.

    `0.0f/0.0f` is a NEGATIVE NaN on x86-64, so classic writes `-nan` --
    measured with a compiled probe, and visible in classic's own `-Hprerec`
    log. This is the port's own rendering of both signs.
    """
    from pymetta.util.logger import ostream_double

    assert ostream_double(-abs(float("nan"))) == "-nan"
    assert ostream_double(abs(float("nan"))) == "nan"
    assert ostream_double(float("-inf")) == "-inf"
