"""CLI oracle tests for `--enable-fs`, byte-for-byte against classic.

Every expected string below is classic's own stdout, captured with

    ( cd <scratch> && ulimit -v 4194304; timeout 300 \
        $CLASSIC/build/moses/moses/main/moses <the argv in the case> )

on 2026-09-10. `@` in an argv stands for `tests/fixtures/table_data/`.

`--enable-fs` was gated at `parse_options` until 2026-09-10. Un-gating it
needed three things that were genuinely missing -- the `--fs-focus`/
`--fs-seed` switch (problem-params.cc:2196-2229), the enforce-features file
parse (:2231-2259), and the real `feature_selection_parameters` in place of
a private stand-in -- plus lifting `build_fs_ctable`'s boolean-only gate.
Each of those is discriminated by at least one PAIR of cases below whose
outputs differ, because a corpus where the faithful and the broken version
agree is this project's most-repeated defect. The pairs:

* `simple_a5_r1` vs `focus_all_a5_r1` -- the `--fs-focus` arm.
* `seed_add_a6_r1` vs `seed_none_a6_r1` -- the `--fs-seed` arm.
* `seed_add_a6_r1_smd` vs `seed_init_a6_r1_smd` -- `init_xmplr_features`,
  which only an algorithm that reads `fs_params.initial_features` can see.
* `seed_add_a6_r1` vs `prune_a6_r1` -- `--fs-prune-exemplar`.
* `smd_a5_r1` -- fails outright without the real `fs_params`.
* `it_contin_a5_r3`, `cluster_a5_r3` -- the two `REGRESSION(table, ...)`
  call sites, which reach `feature_selector`'s `Table` constructor overload
  with a non-boolean output type.
* `weighted_contin_r1`, `weighted_contin_r7` -- the macro's FIRST argument,
  `table` vs `ctable`. That choice is invisible on an unweighted table (both
  compress to the same counts) and decides the answer on a weighted one;
  swapping the two at `it`'s contin site is killed by these and by nothing
  else in the suite. Measured: 2 of 144 grid cells discriminate, and both
  are here.

  The THIRD `REGRESSION(table, ...)` site, `-Hcluster`
  (table-problems.cc:557), has NO such case and cannot have one: classic
  itself prints `-inf 0 ` for every combination of table, evaluation
  budget, seed and `--fs-target-size` tried (contin_weighted, contin_a5,
  contin_a4, contin_regression_sample, select_regression, regression_sample
  at `-m600`/`-m2000`/`-m4000`). `cluster_bscore` is degenerate on every
  table this project has, so `-Hcluster`'s stdout carries no information
  about which features were selected. The swap at that site is a
  zero-kill mutation because there is nothing to observe, not because the
  suite is missing a case -- `cluster_a5_r3` below still pins that the
  command line RUNS and matches classic.

Two `--enable-fs` command lines have NO oracle, because classic itself dies
on them; both are pinned at the bottom of this file with the measurement.
`--fs-algo=hc` is a third, already recorded in docs/CARRY-FORWARD.md and
re-confirmed here through the `--enable-fs` path (not just the standalone
`feature-selection` binary): pinned at the bottom too.
"""

# ruff: noqa: E501 -- the expected strings are classic's stdout, captured
# verbatim. Wrapping one to satisfy the line limit would mean editing an
# oracle, which is the one thing these literals must not be.

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from pymoses.moses.main.moses_exec import moses_exec

_TABLE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "table_data"


def _run(argv) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = moses_exec(argv)
    return rc, buf.getvalue()


def _argv(args) -> list[str]:
    return ["moses"] + [a.replace("@", str(_TABLE_DIR) + "/") for a in args]


# (name, argv, classic's stdout, why this case is here)
_CASES = [
    (
        "simple_a5_r1",
        [
            "-i@bool_a5_maj.csv",
            "-uy",
            "-m1000",
            "-W1",
            "--enable-fs=1",
            "--fs-target-size=2",
            "-r1",
        ],
        (
            "-3 or(and(or($x1 $x3) $x4 $x5) and(or($x4 $x5) $x2) and($x1 $x3)) \n"
            "-3 or(and(or($x1 $x2 $x3) $x4 $x5) and(or($x4 $x5) $x2) and($x1 $x3)) \n"
            "-3 or(and(or(and($x1 !$x2) $x3) $x4 $x5) and(or($x4 $x5) $x2) and($x1 $x3)) \n"
            "-3 or(and(or(and($x1 $x4) $x5) $x2) and(or($x1 $x3) $x4 $x5) and($x1 $x3)) \n"
            "-3 or(and(or(and($x1 $x4) $x5) $x2) and(or($x2 $x5) $x3 $x4) and($x1 $x3)) \n"
            "-3 or(and(or(and($x1 $x5) $x4) $x2) and(or($x1 $x3) $x4 $x5) and($x1 $x3)) \n"
            "-3 or(and(or(and(!$x2 $x3) $x1) $x4 $x5) and(or($x4 $x5) $x2) and($x1 $x3)) \n"
            "-3 or(and(or(and($x3 $x4) $x5) $x2) and(or($x1 $x3) $x4 $x5) and($x1 $x3)) \n"
            "-3 or(and(or($x1 $x3) or($x2 $x4) $x5) and(or($x4 $x5) $x2) and($x1 $x3)) \n"
            "-3 or(and(or($x1 $x3) or($x2 $x5) $x4) and(or($x4 $x5) $x2) and($x1 $x3)) \n"
        ),
        """The flagship: 5-input majority, `--fs-algo simple` (the default),
        two selected features. This is the case that caught the missing
        fs-seed switch -- with `ignore_xmplr_features` left at the struct
        default the port printed the `focus_all` answer below instead.""",
    ),
    (
        "focus_all_a5_r1",
        [
            "-i@bool_a5_maj.csv",
            "-uy",
            "-m1000",
            "-W1",
            "--enable-fs=1",
            "--fs-target-size=2",
            "-r1",
            "--fs-focus=all",
        ],
        (
            "-2 or(and(or($x1 $x2) or($x4 $x5) $x3) and($x1 $x2) and($x4 $x5)) \n"
            "-2 and(or(and(or($x1 $x2) or($x4 $x5) $x3) and($x1 $x2) and($x4 $x5)) or($x1 $x3)) \n"
            "-2 or(and(or(and(or(!$x2 $x4) $x1) $x2) or($x4 $x5) $x3) and($x1 $x2) and($x4 $x5)) \n"
            "-2 or(and(or(and(or($x3 $x5) $x4) $x5) or($x1 $x2) $x3) and($x1 $x2) and($x4 $x5)) \n"
            "-2 or(and(or(and($x1 $x2) $x4) $x5) and(or($x1 $x2) or($x4 $x5) $x3) and($x1 $x2)) \n"
            "-2 or(and(or(and($x1 $x2) $x5) $x4) and(or($x1 $x2) or($x4 $x5) $x3) and($x1 $x2)) \n"
            "-2 or(and(or(and($x2 $x3) $x5) $x4) and(or($x1 $x2) or($x4 $x5) $x3) and($x1 $x2)) \n"
            "-3 or(and(or(and($x1 $x2) $x4) $x5) and(or($x1 $x2) or($x4 $x5) $x3)) \n"
            "-3 or(and(or(and($x1 $x2) $x5) $x4) and(or($x1 $x2) or($x4 $x5) $x3)) \n"
            "-2 or(and(or(and($x1 $x2) and($x2 $x3) $x5) $x4) and(or($x1 $x2) or($x4 $x5) $x3) and($x1 $x2)) \n"
        ),
        """Same table and seed, `--fs-focus=all`. DIFFERENT stdout from
        `simple_a5_r1`, which is what makes the pair a real test of the
        `--fs-focus` half of problem-params.cc:2196-2208 rather than a
        pair that would agree however the switch were wired.""",
    ),
    (
        "smd_a5_r1",
        [
            "-i@bool_a5_maj.csv",
            "-uy",
            "-m1000",
            "-W1",
            "--enable-fs=1",
            "--fs-target-size=2",
            "-r1",
            "--fs-algo=smd",
        ],
        (
            "-4 or(and(or(and(or($x4 $x5) $x1) $x3) $x2) and($x4 $x5)) \n"
            "-4 or(and(or(and(or(!$x2 $x4 $x5) $x1) $x3) $x2) and($x4 $x5)) \n"
            "-4 or(and(or(and(or($x4 $x5) $x1) and($x4 $x5) $x3) $x2) and($x1 $x4)) \n"
            "-4 or(and(or(and(or($x4 $x5) $x1) and($x4 $x5) $x3) $x2) and($x4 $x5)) \n"
            "-4 or(and(or(and(or($x1 $x2) $x3) and(or($x4 $x5) $x1)) $x2) and($x4 $x5)) \n"
            "-5 or(and(or(and($x1 $x4) $x3) $x2) and($x4 $x5)) \n"
            "-5 or(and(or(and($x1 $x5) $x3) $x2) and($x4 $x5)) \n"
            "-4 and(or(and(or($x4 $x5) $x1) and(!$x1 $x4 $x5) $x3) or(and($x4 $x5) $x2)) \n"
            "-4 and(or(and(or($x4 $x5) $x1) and($x1 $x4 $x5) $x3) or(and($x4 $x5) $x2)) \n"
            "-4 and(or(and(or($x3 $x4) $x5) and(or($x4 $x5) $x1) $x3) or(and($x4 $x5) $x2)) \n"
        ),
        """`--fs-algo=smd` reads `fs_params.initial_features`, a field the
        two private stand-in parameter classes did not have; before they
        were replaced by the real ported ones this died with
        AttributeError on every dataset.""",
    ),
    (
        "random_a5_r1",
        [
            "-i@bool_a5_maj.csv",
            "-uy",
            "-m1000",
            "-W1",
            "--enable-fs=1",
            "--fs-target-size=2",
            "-r1",
            "--fs-algo=random",
        ],
        (
            "-2 and(or(and(or($x3 $x5) $x4) and($x1 $x2) and($x3 $x5)) or($x1 $x2)) \n"
            "-2 and(or(and(or($x3 $x4) $x1 $x2) and(or($x3 $x5) $x4) and($x3 $x5)) or($x1 $x2)) \n"
            "-2 and(or(and(or($x3 $x5) $x4) and($x1 $x2) and($x3 $x5)) or($x1 $x2) or($x4 $x5)) \n"
            "-2 and(or(and(or(and($x4 $x5) $x2) $x1) and(or($x3 $x5) $x4) and($x3 $x5)) or($x1 $x2)) \n"
            "-2 and(or(and(or($x3 $x4) $x1) $x2) or(and(or($x3 $x5) $x4) and($x1 $x2) and($x3 $x5))) \n"
            "-2 and(or(and(or($x3 $x5) $x4) and($x1 $x2) and($x3 $x5)) or(and(or($x4 $x5) $x1) $x2)) \n"
            "-2 or(and(or(and(or($x3 $x5) $x4) and($x1 $x2) and($x3 $x5)) or($x1 $x2)) and($x4 $x5)) \n"
            "-3 and(or(and(or($x3 $x5) $x4) and(!$x1 $x3 $x5) and($x1 $x2)) or($x1 $x2)) \n"
            "-3 and(or(and(or($x3 $x5) $x4) and($x1 $x2) and($x3 $x5)) or($x1 $x2 !$x3)) \n"
            "-3 and(or(and(or($x3 $x5) $x4) and($x1 $x2) and($x3 $x5)) or($x1 $x2 $x4)) \n"
        ),
        """`--fs-algo=random` -- a different RNG draw pattern from `simple`,
        so it pins the interleaving of the selector's draws with the
        search's.""",
    ),
    (
        "seed_add_a6_r1",
        [
            "-i@bool_a6_mix.csv",
            "-uy",
            "-m1000",
            "-W1",
            "--enable-fs=1",
            "--fs-target-size=2",
            "-r1",
        ],
        (
            "-4 and(or(and($x2 !$x6) $x1) or($x2 $x4)) \n"
            "-4 and(or(and($x2 !$x6) $x1) or($x4 !$x6)) \n"
            "-4 and(or(and(or($x1 $x2) !$x6) $x1) or($x2 $x4)) \n"
            "-4 and(or(and(or($x1 !$x6) $x2) $x1) or($x2 $x4)) \n"
            "-4 and(or(and(!$x1 $x3) $x2 $x4) or(and($x2 !$x6) $x1)) \n"
            "-4 or(and(or(and($x2 !$x6) $x1) or($x2 $x4)) and($x1 $x2)) \n"
            "-4 or(and(or(and(!$x1 $x3) $x2 $x4) or(and($x2 !$x6) $x1)) and($x1 $x2)) \n"
            "-6 and(or(and(or($x3 $x6) $x4) $x2) or(and($x2 !$x6) $x1)) \n"
            "-8 and(or(and($x2 !$x6) $x1) or($x2 $x4 $x6)) \n"
            "-8 and(or(and($x1 !$x5) and($x2 !$x6)) or($x2 $x4)) \n"
        ),
        """6-input mix, `--fs-seed` at its default `add`.""",
    ),
    (
        "seed_none_a6_r1",
        [
            "-i@bool_a6_mix.csv",
            "-uy",
            "-m1000",
            "-W1",
            "--enable-fs=1",
            "--fs-target-size=2",
            "-r1",
            "--fs-seed=none",
        ],
        (
            "-4 and(or(and($x2 !$x6) $x1) or($x2 $x4)) \n"
            "-4 and(or(and(!$x1 $x2) $x4) or(and($x2 !$x6) $x1)) \n"
            "-4 and(or(and($x1 !$x2) and($x2 !$x6)) or($x2 $x4)) \n"
            "-4 or(and(or(and($x2 !$x6) $x1) or($x2 $x4)) and($x1 $x2)) \n"
            "-4 or(and(or(and($x1 !$x2) and($x2 !$x6)) or($x2 $x4)) and($x1 $x2)) \n"
            "-12 and($x1 $x4) \n"
            "-12 and(or($x1 $x2) $x4) \n"
            "-12 and(or($x1 !$x6) $x4) \n"
            "-12 and(or($x2 $x4) $x1) \n"
            "-12 and(or($x4 !$x6) $x1) \n"
        ),
        """Same table and seed, `--fs-seed=none`. DIFFERENT stdout from
        `seed_add_a6_r1` -- the pair that discriminates the `--fs-seed`
        half of the switch. (On `bool_a5_maj` the two agree at every seed
        1..5, which is exactly the kind of corpus that would have passed
        while the switch was missing.)""",
    ),
    (
        "seed_add_a6_r1_smd",
        [
            "-i@bool_a6_mix.csv",
            "-uy",
            "-m1000",
            "-W1",
            "--enable-fs=1",
            "--fs-target-size=2",
            "-r1",
            "--fs-algo=smd",
        ],
        (
            "0 or(and($x1 $x4) and($x2 !$x6)) \n"
            "-12 and($x2 !$x6) \n"
            "-12 and(or($x1 $x2) !$x6) \n"
            "-12 and(or($x1 !$x6) $x2) \n"
            "-12 and(or($x2 $x4) !$x6) \n"
            "-12 and(or($x4 !$x6) $x2) \n"
            "-12 or(and($x2 !$x6) $x1) \n"
            "-12 or(and($x2 !$x6) $x4) \n"
            "-12 and(or(and(!$x3 $x4) !$x6) $x2) \n"
            "-12 or(and(or($x2 $x4) !$x6) $x4) \n"
        ),
        """`--fs-seed=add` with `smd`, the pair-mate of the next case.""",
    ),
    (
        "seed_init_a6_r1_smd",
        [
            "-i@bool_a6_mix.csv",
            "-uy",
            "-m1000",
            "-W1",
            "--enable-fs=1",
            "--fs-target-size=2",
            "-r1",
            "--fs-seed=init",
            "--fs-algo=smd",
        ],
        (
            "0 or(and($x1 $x4) and($x2 !$x6)) \n"
            "-12 and($x2 !$x6) \n"
            "-12 and(or($x1 $x2) !$x6) \n"
            "-12 and(or($x1 !$x6) $x2) \n"
            "-12 and(or($x2 $x4) !$x6) \n"
            "-12 and(or($x4 !$x6) $x2) \n"
            "-12 or(and($x2 !$x6) $x1) \n"
            "-12 or(and($x2 !$x6) $x4) \n"
            "-12 and(or(and(!$x3 $x4) $x2) !$x6) \n"
            "-12 and(or(and(!$x3 $x4) !$x6) $x2) \n"
        ),
        """`--fs-seed=init` with `smd`. `init_xmplr_features` differs from
        `add` only in that it seeds `fs_params.initial_features` with the
        exemplar's own columns, so it takes an algorithm that READS that
        field to tell them apart -- `simple` and `random` ignore it. The
        two outputs differ in their last two lines.""",
    ),
    (
        "prune_a6_r1",
        [
            "-i@bool_a6_mix.csv",
            "-uy",
            "-m1000",
            "-W1",
            "--enable-fs=1",
            "--fs-target-size=2",
            "-r1",
            "--fs-prune-exemplar=1",
        ],
        (
            "-12 and($x1 $x4) \n"
            "-12 and(or($x1 $x2) $x4) \n"
            "-12 and(or($x1 !$x6) $x4) \n"
            "-12 and(or($x2 $x4) $x1) \n"
            "-12 and(or($x4 !$x6) $x1) \n"
            "-12 or(and($x1 $x4) $x2) \n"
            "-12 or(and($x1 $x4) !$x6) \n"
            "-12 and(or(and($x2 $x3) $x1) $x4) \n"
            "-12 and(or(and($x2 $x3) $x4) $x1) \n"
            "-12 or(and(or($x1 $x2) $x4) $x2) \n"
        ),
        """`--fs-prune-exemplar=1` takes `create_representations`' other arm
        (deme_expander.cc:216-229): the exemplar is pruned of non-selected
        features and, unlike the default arm, its own features are NOT
        unioned back into the selected set.""",
    ),
    (
        "it_contin_a5_r3",
        [
            "-i@contin_a5.csv",
            "-uy",
            "-Hit",
            "-m600",
            "-W1",
            "-r3",
            "--enable-fs=1",
            "--fs-target-size=2",
        ],
        (
            "-7.58199977874755859 1.5 \n"
            "-7.99311113357543945 1.41666666666666674 \n"
            "-8.00699996948242188 1.625 \n"
            "-9.28824996948242188 1.3125 \n"
            "-9.68200016021728516 1.75 \n"
            "-10.4820003509521484 1.25 \n"
            "-24.207000732421875 0.875 \n"
            "-26.4264450073242188 0.83333333333333337 \n"
            "-31.2819995880126953 0.75 \n"
            "-42.6597785949707031 0.58333333333333337 \n"
        ),
        """A CONTIN table. `it_table_problem.run`'s contin branch passes
        `self.table` (the uncompressed one) into `_regression`, i.e.
        `REGRESSION(table, ...)` at table-problems.cc:519 -- the
        `feature_selector(const Table&)` constructor overload, which
        compresses it itself. `build_fs_ctable` rejected every
        non-boolean output type until 2026-09-10, so this command line
        raised where classic ran.""",
    ),
    (
        "cluster_a5_r3",
        [
            "-i@contin_a5.csv",
            "-uy",
            "-Hcluster",
            "-m2000",
            "-W1",
            "-r3",
            "--enable-fs=1",
            "--fs-target-size=2",
        ],
        "-inf 0 \n",
        """`-Hcluster`, the third `REGRESSION(table, ...)` site
        (table-problems.cc:557). Same `Table` overload, same previously
        rejected non-boolean output type.""",
    ),
    (
        "weighted_contin_r1",
        [
            "-i@contin_weighted.csv",
            "-uy",
            "-Hit",
            "-m600",
            "-W1",
            "-r1",
            "--enable-fs=1",
            "--fs-target-size=2",
            "--score-weight=wgt",
        ],
        (
            "-47.625 +(1 *(+(-1 $x1) +($x3 $x5))) \n"
            "-47.625 +(1 *(-1 +(*(+(1 *(-1 $x1)) $x3) $x5))) \n"
            "-62 +(1 *(-1 $x3)) \n"
            "-62 +(1 *(-1 $x3) $x5) \n"
            "-62 +(1 *(-1 +($x3 $x5))) \n"
            "-62 +(1 /(*(-1 $x3) +(1 $x5))) \n"
            "-62 +(1 /(*(-1 $x3) +(1 *(-1 $x5)))) \n"
            "-102.75 0 \n"
        ),
        """`--score-weight` is what makes the `REGRESSION` macro's FIRST argument
        observable at all. `ctable` is `table.compressed(weighting_feature)`
        (table-problems.cc:125) while `feature_selector`'s `Table` overload
        computes `table.compressed()` itself, unweighted
        (feature_selector.cc:56) -- so on a weighted table the selector sees
        DIFFERENT per-row counts depending on which one the call site hands
        it, and mutual information ranks the features differently. Swapping
        `self.table` for `self.ctable` at `it`'s contin site changes this
        case's output (239 -> 251 bytes); on every unweighted table in this
        file it changes nothing, which is exactly why the distinction needed
        a case of its own.""",
    ),
    (
        "weighted_contin_r7",
        [
            "-i@contin_weighted.csv",
            "-uy",
            "-Hit",
            "-m600",
            "-W1",
            "-r7",
            "--enable-fs=1",
            "--fs-target-size=2",
            "--score-weight=wgt",
        ],
        (
            "-30.5 +(1 *(-2 $x3) $x1) \n"
            "-30.5 +(1 *(-2 $x3) $x1 $x5) \n"
            "-30.5 +(1 /(+(*(-2 $x3) $x1) +(1 $x5))) \n"
            "-31.125 +(1 *(+(-2 $x4) $x3) $x1) \n"
            "-30.5 +(1 /(+(*(-2 $x3) $x1) +(1 *(-1 $x5)))) \n"
            "-32 +(1 *(-1 +($x3 $x4)) $x1) \n"
            "-62 +(1 *(-1 $x3)) \n"
            "-62 +(1 *(-1 $x3) $x5) \n"
            "-62 +(1 *(-1 +($x3 $x5))) \n"
            "-62 +(1 /(*(-1 $x3) +(1 $x5))) \n"
        ),
        """The second seed at which the same swap shows (313 -> 191 bytes). Two,
        not one, because a single seed agreeing or disagreeing is one
        sample.""",
    ),
]


@pytest.mark.parametrize("name,args,expected,why", _CASES, ids=[c[0] for c in _CASES])
def test_enable_fs_matches_classic(name, args, expected, why):
    del name, why
    rc, out = _run(_argv(args))
    assert rc == 0
    assert out == expected


def test_the_discriminating_pairs_really_do_differ():
    """Guards the corpus itself, not the port: if a future edit made any of
    these four pairs agree, the tests above would still pass while having
    stopped testing the thing they name. Compares the captured ORACLE
    strings, so it fails on a bad corpus even if the port is perfect."""
    by_name = {c[0]: c[2] for c in _CASES}
    pairs = [
        ("simple_a5_r1", "focus_all_a5_r1"),
        ("seed_add_a6_r1", "seed_none_a6_r1"),
        ("seed_add_a6_r1_smd", "seed_init_a6_r1_smd"),
        ("seed_add_a6_r1", "prune_a6_r1"),
    ]
    for a, b in pairs:
        assert by_name[a] != by_name[b], f"{a} and {b} no longer discriminate"


# ---------------------------------------------------------------------------
# Command lines classic cannot run. No oracle exists for either; the port
# fails loudly rather than answering where classic cannot.
# ---------------------------------------------------------------------------


def test_fs_demes_above_one_raises_where_classic_aborts():
    """MEASURED 2026-09-10: `moses -i@bool_a5_maj.csv -u y -m1000 -W1
    --enable-fs=1 --fs-target-size=2 -r1 --fs-demes=2` exits 134 --

        terminate called after throwing an instance of
        'opencog::AssertionException'
          what():  The code is probably not robust enough
          (.../moses/moses/deme/deme_expander.cc:63)

    -- for `--fs-algo=simple` and `--fs-algo=random` alike. That
    `OC_ASSERT` compares `sf_pop.size()` against `_demeIDs.size()`, and
    every selection algorithm returns exactly one feature set while
    `create_demeIDs` builds `--fs-demes` of them, so it can never hold
    above 1. This port ran the same command line to completion and printed
    a result until `log_selected_feature_sets` was ported -- accepting
    input classic refuses.
    """
    with pytest.raises(RuntimeError, match="probably not robust enough"):
        _run(
            _argv(
                [
                    "-i@bool_a5_maj.csv",
                    "-uy",
                    "-m1000",
                    "-W1",
                    "--enable-fs=1",
                    "--fs-target-size=2",
                    "-r1",
                    "--fs-demes=2",
                ]
            )
        )


def test_fs_seed_xmplr_raises_where_classic_segvs():
    """MEASURED under gdb 2026-09-10: the same command line with
    `--fs-seed=xmplr` SIGSEGVs classic (rc 139), faulting in
    `std::char_traits<char>::copy` under `deme_expander::fs_to_names`
    (deme_expander.cc:53) <- `log_selected_feature_sets` (cc:78) <-
    `create_representations` (cc:200).

    Cause: `xmplr_as_feature` makes `build_fs_ctable` append a synthetic
    `__exemplar_feature__` column at index `_ctable.get_arity()`, feature
    selection can select it, and nothing removes it --
    `feature_selector::remove_useless_features` (feature_selector.cc:295),
    the function written to do exactly that, has ZERO call sites in the
    whole classic tree. `fs_to_names` then indexes the ORIGINAL label
    vector with that one-past-the-end index through an unchecked
    `operator[]`.

    Undefined behaviour cannot be ported, and answering where classic
    cannot would be worse than failing, so `fs_to_names` bounds-checks and
    raises -- the same rule `interpreter.py` applies to an argument index
    beyond the table's arity.
    """
    with pytest.raises(IndexError, match="outside the 5 input column"):
        _run(
            _argv(
                [
                    "-i@bool_a5_maj.csv",
                    "-uy",
                    "-m1000",
                    "-W1",
                    "--enable-fs=1",
                    "--fs-target-size=2",
                    "-r1",
                    "--fs-seed=xmplr",
                ]
            )
        )


def test_fs_algo_hc_runs_here_and_segvs_classic():
    """RE-CONFIRMED 2026-09-10, and through a NEW path. CARRY-FORWARD.md 3
    records `feature-selection --algo hc` core-dumping via the standalone
    binary; the same crash is reachable from `moses --enable-fs=1
    --fs-algo=hc`, measured 16/16 over
    {bool_a5_maj, bool_a6_mix, bool_a4_par, classic's dataset.csv} x
    {--fs-target-size 2, 3} x {-r1, -r5}: rc 139, zero bytes of stdout.

    gdb, same command line as the other cases plus `--fs-algo=hc`:

        #0 representation::get_candidate(...) (this=0x0)
             representation.cc:319
        #1 hill_climbing::operator() hill-climbing.cc:69
        #2 optimize_deme_select_feature_sets<hill_climbing, ...>
             algo/deme_optimize.h:52
        ...
        #6 feature_selector::operator() feature_selector.cc:328
        #7 deme_expander::create_representations deme_expander.cc:187

    `this=0x0` is `iscorer_base::get_rep()` returning null into an
    unconditional debug-log line -- the exact deref that also core-dumps
    classic's own `optimizationUTest`. This port drops that log line, so
    its `hc` selector runs to completion. THERE IS NO ORACLE FOR `hc`:
    what this test pins is that the port produces something and does not
    crash, NOT that the something is right.
    """
    rc, out = _run(
        _argv(
            [
                "-i@bool_a5_maj.csv",
                "-uy",
                "-m1000",
                "-W1",
                "--enable-fs=1",
                "--fs-target-size=2",
                "-r1",
                "--fs-algo=hc",
            ]
        )
    )
    assert rc == 0
    assert out.strip() != ""
