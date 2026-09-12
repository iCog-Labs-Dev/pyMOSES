"""`-H pre --boost=1` end to end, against classic's own binary.

This is the configuration that made `precision_bscore.do_score` raise:
`table-problems.cc:385` calls `bscore.use_weighted_scores()` for every
boosted `pre` run, and the `_return_weighted_score` branch
(`precision_bscore.cc:283-289`) was a `NotImplementedError` gate. The
gate is gone; these tests are what keeps it gone.

## Why the score and the member MULTISET are asserted, but not the line

Same reason `test_boost_cli.py` gives, and it bites harder here: the
printed tree is `ensemble::get_exact_tree` (`ensemble.cc:361`) walking
`_scored_trees`, a `scored_combo_tree_set` -- i.e. a
`std::unordered_set` (`types.h:370`), whose iteration order C++ does not
specify. Measured over the ten configurations pinned below, classic and
this port agree on the score and on the set of members in ten of ten,
and differ in print order in five of ten. Asserting the byte string
would be measuring libstdc++'s hash layout.

`precision_bscore::exact_selection` (`precision_bscore.cc:391`) folds the
ensemble into one `logical_or` in that same unspecified order, but
`logical_or` is commutative and the tree is only interpreted there, so
the SCORE is order-free -- which is exactly why it can be asserted.

## The corpus, and why the obvious dataset is useless

An exact expert is admitted only when its error is ~0
(`ensemble.cc:193`), and for this scorer `error == 1 - precision`. So a
tree is admitted only if every row it selects is a pure true-positive
row. On a table with a contradictory row -- the same inputs appearing
with both outputs -- no tree can ever reach precision 1, the ensemble
stays EMPTY, and classic prints `-inf or` no matter what any boosting
flag is set to. `bool.csv` below is exactly that table, and it is kept
as a pinned case precisely because it is the degenerate one: it proves
the empty-ensemble path, and proves nothing else.

`xor4.csv` is the discriminating corpus: `--boost-promote` genuinely
changes the number of members (1, 2 and 3 members at 1, 2 and 5).

Measured with:

    ( ulimit -v 2097152; ulimit -c 0; timeout 120 \
      $CLASSIC/build/moses/moses/main/moses -i<data>.csv -ures -W1 \
      -Hpre -q0.1 -w0.9 --boost=1 -f/dev/null <flags> )
"""

import io
import itertools
from contextlib import redirect_stdout

import pytest

from pymetta.moses.main.moses_exec import moses_exec


def _table(fn, arity: int) -> str:
    head = "res," + ",".join(f"a{i + 1}" for i in range(arity))
    rows = [head]
    for bits in itertools.product((0, 1), repeat=arity):
        rows.append(",".join(str(v) for v in (fn(bits), *bits)))
    return "\n".join(rows) + "\n"


# res = (a1 XOR a2) AND (a3 OR a4) -- four inputs, no contradictory row, and
# no single conjunction covers every true row, so the ensemble grows.
_XOR4 = _table(lambda b: (b[0] ^ b[1]) & (b[2] | b[3]), 4)
# res = a1 AND a2 -- one conjunction is already perfect.
_AND3 = _table(lambda b: b[0] & b[1], 3)
# res = (a1 AND a2) OR (a3 AND a4).
_OR2AND = _table(lambda b: (b[0] & b[1]) | (b[2] & b[3]), 4)
# The degenerate table: (1,0) and (1,1) each appear with BOTH outputs, so
# no tree can ever reach precision 1.0 and the ensemble stays empty.
_CONTRADICTORY = "res,arg1,arg2\n0,0,1\n0,1,0\n1,1,1\n0,0,1\n0,1,1\n1,1,1\n0,0,0\n1,1,0\n"

_DATASETS = {"xor4": _XOR4, "and3": _AND3, "or2and": _OR2AND, "bool": _CONTRADICTORY}


def _members(line: str) -> tuple[str, frozenset[str]]:
    """Split `<score> <tree>` into the score and the multiset of `or`
    children -- the two things classic's own output actually determines.
    """
    score, _, tree = line.strip().partition(" ")
    tree = tree.strip()
    if not (tree.startswith("or(") and tree.endswith(")")):
        return score, frozenset([tree])
    body, depth, cur, parts = tree[3:-1], 0, "", []
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == " " and depth == 0:
            if cur:
                parts.append(cur)
                cur = ""
            continue
        cur += ch
    if cur:
        parts.append(cur)
    return score, frozenset(parts)


def _run(tmp_path, dataset: str, argv: list[str]) -> str:
    csv = tmp_path / f"{dataset}.csv"
    csv.write_text(_DATASETS[dataset])
    log = tmp_path / "moses.log"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = moses_exec(
            [
                "moses",
                f"-i{csv}",
                "-ures",
                "-W1",
                "-Hpre",
                "-q0.1",
                "-w0.9",
                "--boost=1",
                f"-f{log}",
                *argv,
            ]
        )
    assert rc == 0
    out = buf.getvalue()
    assert out.strip(), "empty output compares equal to anything"
    return out.strip()


# (dataset, extra flags, classic's score, classic's member multiset)
_CASES = [
    ("xor4", ["-m2000", "-r1"], "1", {"and(!$1 $2 $3)"}),
    (
        "xor4",
        ["-m2000", "-r1", "--boost-promote=2"],
        "1",
        {"and(!$1 $2 $4)", "and(!$1 $2 $3)"},
    ),
    (
        "xor4",
        ["-m2000", "-r1", "--boost-promote=5"],
        "1",
        {"and($1 !$2 $4)", "and(!$1 $2 $4)", "and(!$1 $2 $3)"},
    ),
    (
        "xor4",
        ["-m2000", "-r3", "--boost-promote=2"],
        "1",
        {"and($1 !$2 $3)", "and($1 !$2 $4)"},
    ),
    (
        "xor4",
        ["-m2000", "-r3", "--boost-promote=5"],
        "1",
        {"and($1 !$2 $3)", "and($1 !$2 $4)"},
    ),
    (
        "xor4",
        ["-m2000", "-r1", "--boost-promote=5", "--boost-expalpha=3"],
        "1",
        {"and($1 !$2 $4)", "and(!$1 $2 $4)", "and(!$1 $2 $3)"},
    ),
    (
        "xor4",
        ["-m2000", "-r1", "--boost-promote=5", "--boost-exact=1"],
        "1",
        {"and($1 !$2 $4)", "and(!$1 $2 $4)", "and(!$1 $2 $3)"},
    ),
    ("or2and", ["-m2000", "-r1", "--boost-promote=5"], "1", {"and($3 $4)"}),
    ("and3", ["-m500", "-r1"], "1", {"and($1 $2)"}),
    # The degenerate one: no tree can reach precision 1, so nothing is ever
    # promoted and `get_exact_tree` prints the childless `or` itself --
    # classic's exact output is the two tokens `-inf or`.
    ("bool", ["-m200", "-r1", "--boost-bias=1"], "-inf", {"or"}),
]


@pytest.mark.oracle
@pytest.mark.parametrize(
    ("dataset", "argv", "score", "expected_members"),
    _CASES,
    ids=lambda v: str(v)[:28],
)
def test_boosted_pre_matches_classic(tmp_path, dataset, argv, score, expected_members):
    got_score, got_members = _members(_run(tmp_path, dataset, argv))
    assert got_score == score
    assert got_members == frozenset(expected_members)


@pytest.mark.oracle
def test_boost_promote_really_grows_the_ensemble(tmp_path):
    """Guards the thing that would regress silently. If `exact_selection`
    or `update_weights` stopped working, the run would still print one
    perfectly good expert with score 1 -- it just would not be an
    ensemble. Classic gives 1, 2 and 3 members here; a port that lost the
    weight update would repeat the same first expert forever and give 1.
    """
    sizes = []
    for promote in ("1", "2", "5"):
        _, members = _members(
            _run(tmp_path, "xor4", ["-m2000", "-r1", f"--boost-promote={promote}"])
        )
        sizes.append(len(members))
    assert sizes == [1, 2, 3]


@pytest.mark.oracle
def test_inexact_experts_fail_exactly_where_classic_aborts(tmp_path):
    """`--boost-exact=0` reaches `precision_bscore::get_error`, whose
    `OC_ASSERT(exact_experts, ...)` (`precision_bscore.cc:503`) fires
    BEFORE `ensemble::add_expert` ever branches on the flag. Classic dies
    with SIGABRT (measured: rc -6 on every one of 48 dataset x budget x
    seed combinations); the port must raise rather than invent a result.

    This is also why `bias_selection` cannot be reached from the CLI and
    is unit-tested directly instead.
    """
    with pytest.raises(AssertionError, match="inexact experts not yet supported"):
        _run(tmp_path, "xor4", ["-m2000", "-r1", "--boost-exact=0"])
