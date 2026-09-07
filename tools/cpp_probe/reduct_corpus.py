"""Task 6 corpus assembly: the fixed list of combo trees the reduct oracle
(`reduct_effort_probe.cc` via `tools/capture_reduct.py`) is run over, from
the five upstream files in the three source groups below.

1. The 29 combo trees in `$CLASSIC/testTrees.sh`'s `trees` array, verbatim
   and in order (same corpus `tools/capture_combo_roundtrip.py` already
   uses for a different fixture -- combo_roundtrip.json -- copied here
   rather than imported, so this module has no import-time dependency on
   that script's own module-level state).

2. Every tree literal in `$CLASSIC/tests/comboreduct/reduct/
   BooleanReductUTest.cxxtest` (6 trees, `test_simplify1..6`) and
   `BooleanFullUTest.cxxtest` (20 trees, `test_simplify1..9`, `a`..`d`,
   `14`, `15..20` -- NOT `test_simplify1..24`; a previous miscount said 24
   test methods, there are 20). Every BooleanFullUTest tree contains a
   `0<(...)` contin predicate (see e.g. its own test_simplify1 docstring:
   `or(not(0<(+($1 $2))) not(0<($1)) not(0<($2)))`) and exercises
   `logical_reduction`'s effort>=2 "recurse into predicates" path
   (reduct.h: "For effort==2 and greater, this will also recurse down
   into predicates... only the greater-than-zero predicate is
   supported"), which needs contin_reduction -- out of reach of S7's
   boolean-only reduction path. These 20 are included in CORPUS with
   `runnable=False` so Task 6 can skip them explicitly rather than the
   corpus silently omitting 20 real upstream test cases.

   `FullReductUTest.cxxtest` has 14 literals: ten duplicate the BooleanFull
   inputs exactly and four use distinct bare-predicate wrapper shapes. The
   four unique inputs are also retained as explicit non-runnable S14 cases.

3. The 44 *unique* boolean expressions in elegant-normal-forms-python's
   `Tests/GeneralTestCases.py` `EXPRESSIONS` list. That list has 46 raw
   entries but only 44 distinct expressions once whitespace differences
   are normalized away: index 6/7 are an exact duplicate string, and
   index 38/39 differ only by a space inside `&(b,c)` vs `&(b, c)` --
   same expression. This module keeps the FIRST occurrence of each and
   drops the later duplicate, landing on exactly 44, matching the task
   brief's count. Two are annotated in that file as Holman paper (page
   19) examples, two as Moshe Looks's paper examples (one the "before",
   one the paper's own claimed reduced form -- ENF_TRANSLATED_CASES
   below reduces both to the SAME combo tree at effort 3, confirming the
   translation: see reduct_corpus's own smoke test in the task-6 report).

## Translating elegant-normal-forms-python's grammar to combo

elegant-normal-forms-python (`Utilities/BuildTree.py`) parses a
different, LISP-adjacent boolean grammar: `&(x,y)` / `|(x,y)` are binary
(exactly two arguments, never n-ary), `!(x)` is unary not, and leaves are
BARE NAMED LITERALS (`a`, `b`, `c`, ...) rather than combo's positional
`$1..$n` arguments. Combo instead uses n-ary `and(...)`/`or(...)`,
unary `not(...)`, and positional arguments.

The translation here (`enf_to_combo`) is a direct, non-flattening
syntactic rewrite: `&`->`and`, `|`->`or`, `!`->`not`, preserving the
EXACT binary-nesting shape of the original (never collapsing e.g.
`and(and($1 $2) $3)` into a flat `and($1 $2 $3)` -- that flattening is
`general_rules.level()`'s job during actual reduction, not this
translation's). Named literals are mapped to `$N` via LETTER_TO_ARG, a
fixed alphabetical assignment over the corpus's actual literal alphabet
(`a`..`g`, plus `k` and `x` used by two entries) -- `a`->`$1`, ...,
`g`->`$7`, `k`->`$8`, `x`->`$9`. This mapping is arbitrary (the ENF
grammar's literal names carry no positional meaning) but fixed and
recorded here so it is reproducible.

One thing worth flagging as lossy, found while writing this translator:
elegant-normal-forms-python's own `BuildTree.py` leaf case
(`case _: tree.value = first`) takes ONLY the first character of
whatever string remains as the literal name -- a real bug/limitation in
that upstream parser, not something this translator reproduces or needs
to, because every literal appearing anywhere in EXPRESSIONS already
happens to be exactly one character. If that corpus ever grows a
multi-character literal name, this mapping (and the fact that
elegant-normal-forms-python itself would silently truncate it) would
need revisiting.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------
# Source 1: $CLASSIC/testTrees.sh's `trees` array, verbatim and in order.
# ---------------------------------------------------------------------

TEST_TREES_SH = [
    "or(and(or(!$1)) and())",
    "or(and(or($1)) and())",
    "or(and(or(!$2)) and())",
    "or(and(or($2)) and())",
    "or(and(and($1 !$3) or()) and())",
    "or(and(or(!$1 $3) or()) and())",
    "or(and(and(!$2 !$3) or()) and())",
    "or(and(or($2 $3) or()) and())",
    "or(and(and(!$2 !$1) or()) and())",
    "or(and(or($2 $1) or()) and())",
    "or(and(!$3 or()) and())",
    "or(and($3 or()) and())",
    "or(and(!$1 or()) and())",
    "or(and($1 or()) and())",
    "or(and(!$2 or()) and())",
    "or(and($2 or()) and())",
    "or(and(or()) or(!$1 !$3) and())",
    "or(and(or()) and($1 $3) and())",
    "or(and(or()) or($2 !$1) and())",
    "or(and(or()) and(!$2 $1) and())",
    "or(and(or()) or(!$2 !$3) and())",
    "or(and(or()) and($2 $3) and())",
    "or(and(or()) !$3 and())",
    "or(and(or()) $3 and())",
    "or(and(or()) !$1 and())",
    "or(and(or()) $1 and())",
    "or(and(or()) !$2 and())",
    "or(and(or()) $2 and())",
    "or(and(or()) and($2))",
]
assert len(TEST_TREES_SH) == 29

# 13 of these 29 (the ones shaped `and(or())` -- an `and` wrapping a
# single, EMPTY `or()` -- as one sibling of the outer `or`, alongside a
# trailing empty `and()`) do NOT preserve their truth table under
# classic's own `logical_reduce` at ANY of efforts 1/2/3. This was
# verified directly against the real, unmodified
# `$CLASSIC/build/tests/comboreduct/reduct/test_reduct` binary, e.g.:
#
#   $ test_reduct 'or(and(or()) or(!$1 !$3) and())' 3
#   ...
#   WARNING: Truth tables do not match after reduction!
#
# i.e. this is upstream's own oracle disagreeing with itself, not a
# capture-tooling bug -- test_reduct.cc's `complete_truth_table tt1(tr)`
# (before reduction, arity inferred from the tree) does not match
# `complete_truth_table tt2(tr, integer_log2(tt1.size()))` (after, same
# fixed arity) for exactly these 13 inputs, at every effort. They stay in
# CORPUS as `runnable=True` (test_reduct's own reduction pipeline DOES
# run to completion and produce a tree -- there is a byte-identical
# reduced form for Task 6 to match against), but each captured case in
# reduct.json carries the oracle's own `tt_preserved` bit; Task 6 should
# treat `tt_preserved: false` here as "matches the known-divergent
# oracle", not as grounds to suspect a pyMeTTa bug.
TEST_TREES_SH_KNOWN_TT_DIVERGENT = {
    "or(and(or()) or(!$1 !$3) and())",
    "or(and(or()) and($1 $3) and())",
    "or(and(or()) or($2 !$1) and())",
    "or(and(or()) and(!$2 $1) and())",
    "or(and(or()) or(!$2 !$3) and())",
    "or(and(or()) and($2 $3) and())",
    "or(and(or()) !$3 and())",
    "or(and(or()) $3 and())",
    "or(and(or()) !$1 and())",
    "or(and(or()) $1 and())",
    "or(and(or()) !$2 and())",
    "or(and(or()) $2 and())",
    "or(and(or()) and($2))",
}
assert TEST_TREES_SH_KNOWN_TT_DIVERGENT <= set(TEST_TREES_SH)
assert len(TEST_TREES_SH_KNOWN_TT_DIVERGENT) == 13

# ---------------------------------------------------------------------
# Source 2a: $CLASSIC/tests/comboreduct/reduct/BooleanReductUTest.cxxtest
# -- all 6 `test_simplify1..6` tree literals, verbatim. All runnable:
# pure and/or/not/$n booleans, no `0<(...)` predicates. The upstream test
# itself only exercises effort=3 (`logical_reduce(3, tr, ignore_ops)`),
# but Task 6 captures all of 1/2/3 for these like every other runnable
# entry.
# ---------------------------------------------------------------------

BOOLEAN_REDUCT_UTEST = [
    "and(and(not(not(and($4 $3))) or(not(or($5 $4)) or(and($1 $5) and($1 $5)))) "
    "and(and(and(not($1) not($2)) not(not($3))) or(and(not($4) not($5)) or(or($2 $1) not($5)))))",
    "not(or(and(and(not($5) or($5 $3)) not(or($5 $2))) "
    "and(or(and($5 $3) or($5 $3)) not(not($1)))))",
    "and(or(not(and(or($5 $3) not($2))) or(or(or($1 $2) and($2 $3)) or(and($2 $4) not($5)))) "
    "not(not(and(or($3 $3) or($3 $4)))))",
    "or(and(not($1) $3 or($3 not($1)) or(not($1) and($2 $1) and($3 $2))) and($3 $1) "
    "and(not($3) or($3 $1) not(or($3 not($1))) not(or($2 $1))))",
    "or(and(or(not($1) not($2)) $3) and(or(not($1) $3) not($2)))",
    "and(or(and(or(not($1) $3) not($2)) and(not($1) $3)) or(not($2) $3))",
]
assert len(BOOLEAN_REDUCT_UTEST) == 6

# ---------------------------------------------------------------------
# Source 2b: $CLASSIC/tests/comboreduct/reduct/BooleanFullUTest.cxxtest
# -- all 20 `base_test(tree_str)` calls' tree literals, verbatim, in
# method-declaration order. NONE are runnable via `logical_reduce`
# directly: every one contains a `0<(...)` greater-than-zero predicate
# over contin/`+`/`*`/`impulse`/`cond` subexpressions, and
# BooleanFullUTest's own `base_test` drives them through
# `logical_reduction(ignore_ops)(reduct_effort)` for reduct_effort in
# 0..3 plus an OTable-equality check via ITable-fed contin evaluation --
# not the simple `complete_truth_table` before/after comparison
# BooleanReductUTest and test_reduct.cc use. Reduction levels ==2 recurse
# into these predicates (reduct.h's own comment: "Currently, only the
# greater-than-zero predicate is supported"), which needs
# contin_reduction and contin table evaluation -- both out of scope for
# S7's boolean-only reduction path. Kept in CORPUS with `runnable=False`
# so Task 6 can skip them explicitly (rather than the corpus silently
# omitting 20 real upstream regression cases, several of which document
# known-failing reductions in their own comments -- test_simplify15
# through 20 are literally titled "failing reduction of cond").
# ---------------------------------------------------------------------

BOOLEAN_FULL_UTEST = [
    "or(not(0<(+($1 $2))) not(0<($1)) not(0<($2)))",
    "or(0<(+($1 0.3)) 0<(+($1 -0.3)))",
    "and(0<(sin($1)) 0<($1) 0<(+(*(-1 $1) 3)))",
    "and(0<(sin($1)) not(0<(+(1 *(-1 $1)))) 0<(+(*(-1 $1) 3)))",
    "and(not(0<(*($1 -1))) 0<($1))",
    "and(not(0<(*($1 -1))) 0<(+($1 2)))",
    "and(0<(+( *($1 -1) *($2 0) *(0 $3))))",
    "or(0<(+( *($1 -1) *($2 0) *(0 $3) $1 1)))",
    "and(0<(+( *(0 $1) *($2 0) 1)))",
    "or( 0<(+( *(0 $1) *($2 0) 1)) 0<(/($3 1)) 0<( *(1 $4)))",
    "and( 0<(+( *(0 $1) *($2 0) 1)) 0<(/($3 1)) 0<( *(1 $4)))",
    "and( or($1) and(or($2 and($2))) 0<(/($3 1)) 0<( *(1 $4)))",
    "or(0<( +(1 impulse(and(or($1 and($1)))))))",
    "and(or(not(0<($2)) not(0<(+($1 0)))))",
    "cond(0<($1) POX 0<($2) ME2 CYT)",
    "cond(or(and(or(0<(+(/(*(+(*(/(+(1 $7) +(*(-1 $1) $6)) 1 1) $8 $1 $2 $3 $4 $5 $6 $7) 1) +"
    "(2))))))) POX or(and(or(not(0<(+(/(*(+(*($4 1 1) $8 $1 $2 $3 $4 $5 $6 $7) 1) 1))))))) CY"
    "T or(and(or(not(0<(+(/(*(+(*(*(+(1 $7) $8) +(-1) 1) *($7 1 1) $8 $1 $2 $3 $4 $5 $6 $7) 1"
    ") 1))))))) NUC or(and(or(not(0<(+(/(*(+(*(+(-1 *($8 $8 1 1) *($4 1 1) $1 $2 $3 $4 $5 $6 "
    "$7 $8) $8 $8 1 1) *($4 1 1) $8 $1 $2 $3 $4 $5 $6 $7) 1) 1))))))) ME2 CYT)",
    "cond(or(and(or(0<(+(/(*(+(*(/(+(1 $7) +(*(-1 $1) $6)) 1 1) $8 $1 $2 $3 $4 $5 $6 $7 0) 1)"
    " +(2)) 0))))) POX or(and(or(not(0<(+(/(*(+(*($4 1 1) $8 $1 $2 $3 $4 $5 $6 $7 0) 1) 1) 0)"
    "))))) CYT or(and(or(not(0<(+(/(*(+(*(*(+(1 $7) $8) +(-1) 1) *($7 1 1) $8 $1 $2 $3 $4 $5 "
    "$6 $7 0) 1) 1) 0)))))) NUC or(and(or(not(0<(+(/(*(+(*(+(-1 *($8 $8 1 1) *($4 1 1) $1 $2 "
    "$3 $4 $5 $6 $7 $8) $8 $8 1 1) *($4 1 1) $8 $1 $2 $3 $4 $5 $6 $7 0) 1) 1) 0)))))) ME2 CYT"
    ")",
    "cond(or(and(or(0<(+(/(*(+(*(/(+(1 $7) +(*(-1 $1) $6)) +(1) +(1)) *($8) *($1) *($2) *($3)"
    " *($4) *($5) *($6) *($7) 0) 1) +(2)) 0))))) POX or(and(or(not(0<(+(/(*(+(*($4 +(1) +(1))"
    " *($8) *($1) *($2) *($3) *($4) *($5) *($6) *($7) 0) 1) +(1)) 0)))))) CYT or(and(or(not(0"
    "<(+(/(*(+(*(*(+(1 $7) $8) +(-1) +(1)) *($7 +(1) +(1)) *($8) *($1) *($2) *($3) *($4) *($5"
    ") *($6) *($7) 0) 1) +(1)) 0)))))) NUC or(and(or(not(0<(+(/(*(+(*(+(-1 *($8 $8 +(1) +(1))"
    " *($4 +(1) +(1)) *($1) *($2) *($3) *($4) *($5) *($6) *($7) *($8)) $8 $8 +(1) +(1)) *($4 "
    "+(1) +(1)) *($8) *($1) *($2) *($3) *($4) *($5) *($6) *($7) 0) 1) +(1)) 0)))))) ME2 CYT)",
    "cond(or(and(or(0<(+(/(*(+(*(/(+(1 $7) +(*(-1 $1) $6)) +(1) +(1)) *($8 +(0)) *($1 +(0)) *"
    "($2 +(0)) *($3 +(0)) *($4 +(0)) *($5 +(0)) *($6 +(0)) *($7 +(0)) 0) 1) +(2)) 0))))) POX "
    "or(and(or(not(0<(+(/(*(+(*($4 +(1) +(1)) *($8 +(0)) *($1 +(0)) *($2 +(0)) *($3 +(0)) *($"
    "4 +(0)) *($5 +(0)) *($6 +(0)) *($7 +(0)) 0) 1) +(1)) 0)))))) CYT or(and(or(not(0<(+(/(*("
    "+(*(*(+(1 $7) $8) +(-1) +(1)) *($7 +(1) +(1)) *($8 +(0)) *($1 +(0)) *($2 +(0)) *($3 +(0)"
    ") *($4 +(0)) *($5 +(0)) *($6 +(0)) *($7 +(0)) 0) 1) +(1)) 0)))))) NUC or(and(or(not(0<(+"
    "(/(*(+(*(+(-1 *($8 $8 +(1) +(1)) *($4 +(1) +(1)) *($1 +(0)) *($2 +(0)) *($3 +(0)) *($4 +"
    "(0)) *($5 +(0)) *($6 +(0)) *($7 +(0)) *($8 +(0))) $8 $8 +(1) +(1)) *($4 +(1) +(1)) *($8 "
    "+(0)) *($1 +(0)) *($2 +(0)) *($3 +(0)) *($4 +(0)) *($5 +(0)) *($6 +(0)) *($7 +(0)) 0) 1)"
    " +(1)) 0)))))) ME2 CYT)",
    "cond(or(and(or(0<(+(/(*(+(*(/(+(1 $7) +(*(-1 $1) $6)) +(1 *(0 $1) *(0 $2) *(0 $3) *(0 $4"
    ") *(0 $5) *(0 $6) *(0 $7) *(0 $8)) +(1 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *"
    "(0 $7) *(0 $8))) *($8 +(0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8"
    "))) *($1 +(0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *($2 +(0 "
    "*(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *($3 +(0 *(0 $1) *(0 $"
    "2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *($4 +(0 *(0 $1) *(0 $2) *(0 $3) *("
    "0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *($5 +(0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5)"
    " *(0 $6) *(0 $7) *(0 $8))) *($6 +(0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 "
    "$7) *(0 $8))) *($7 +(0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8)))"
    " 0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8)) 1) +(2 *(0 $1) *(0 $"
    "2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) 0))))) POX or(and(or(not(0<(+(/(*(+"
    "(*($4 +(1 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8)) +(1 *(0 $1) *"
    "(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *($8 +(0 *(0 $1) *(0 $2) *(0 $3"
    ") *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *($1 +(0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0"
    " $5) *(0 $6) *(0 $7) *(0 $8))) *($2 +(0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) "
    "*(0 $7) *(0 $8))) *($3 +(0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $"
    "8))) *($4 +(0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *($5 +(0"
    " *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *($6 +(0 *(0 $1) *(0 "
    "$2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *($7 +(0 *(0 $1) *(0 $2) *(0 $3) *"
    "(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) 0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $"
    "6) *(0 $7) *(0 $8)) 1) +(1 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $"
    "8))) 0)))))) CYT or(and(or(not(0<(+(/(*(+(*(*(+(1 $7) $8) +(-1 *(0 $1) *(0 $2) *(0 $3) *"
    "(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8)) +(1 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 "
    "$6) *(0 $7) *(0 $8))) *($7 +(1 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *"
    "(0 $8)) +(1 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *($8 +(0 *"
    "(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *($1 +(0 *(0 $1) *(0 $2"
    ") *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *($2 +(0 *(0 $1) *(0 $2) *(0 $3) *(0"
    " $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *($3 +(0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) "
    "*(0 $6) *(0 $7) *(0 $8))) *($4 +(0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $"
    "7) *(0 $8))) *($5 +(0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) "
    "*($6 +(0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *($7 +(0 *(0 "
    "$1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) 0 *(0 $1) *(0 $2) *(0 $3) "
    "*(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8)) 1) +(1 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) "
    "*(0 $6) *(0 $7) *(0 $8))) 0)))))) NUC or(and(or(not(0<(+(/(*(+(*(+(-1 *($8 $8 +(1 *(0 $1"
    ") *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8)) +(1 *(0 $1) *(0 $2) *(0 $3) *"
    "(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *($4 +(1 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5"
    ") *(0 $6) *(0 $7) *(0 $8)) +(1 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *"
    "(0 $8))) *($1 +(0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *($2"
    " +(0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *($3 +(0 *(0 $1) "
    "*(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *($4 +(0 *(0 $1) *(0 $2) *(0 $"
    "3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *($5 +(0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *("
    "0 $5) *(0 $6) *(0 $7) *(0 $8))) *($6 +(0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6)"
    " *(0 $7) *(0 $8))) *($7 +(0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 "
    "$8))) *($8 +(0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *(0 $1)"
    " *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8)) $8 $8 +(1 *(0 $1) *(0 $2) *(0 "
    "$3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8)) +(1 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5)"
    " *(0 $6) *(0 $7) *(0 $8))) *($4 +(1 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 "
    "$7) *(0 $8)) +(1 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *($8 "
    "+(0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *($1 +(0 *(0 $1) *"
    "(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *($2 +(0 *(0 $1) *(0 $2) *(0 $3"
    ") *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *($3 +(0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0"
    " $5) *(0 $6) *(0 $7) *(0 $8))) *($4 +(0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) "
    "*(0 $7) *(0 $8))) *($5 +(0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $"
    "8))) *($6 +(0 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) *($7 +(0"
    " *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8))) 0 *(0 $1) *(0 $2) *(0"
    " $3) *(0 $4) *(0 $5) *(0 $6) *(0 $7) *(0 $8)) 1) +(1 *(0 $1) *(0 $2) *(0 $3) *(0 $4) *(0"
    " $5) *(0 $6) *(0 $7) *(0 $8))) 0)))))) ME2 CYT)",
]
assert len(BOOLEAN_FULL_UTEST) == 20

FULL_REDUCT_UTEST_UNIQUE = [
    ("test_simplify7", "0<(+( *($1 -1) *($2 0) *(0 $3)))"),
    ("test_simplify8", "0<(+( *($1 -1) *($2 0) *(0 $3) $1 1))"),
    ("test_simplify9", "0<(+( *(0 $1) *($2 0) 1))"),
    ("test_simplifyd", "0<( +(1 impulse(and(or($1 and($1))))))"),
]
assert len(FULL_REDUCT_UTEST_UNIQUE) == 4

# ---------------------------------------------------------------------
# Source 3: elegant-normal-forms-python's Tests/GeneralTestCases.py
# EXPRESSIONS list (44 unique entries after dedup -- see module
# docstring), translated to combo syntax.
# ---------------------------------------------------------------------

# Alphabetical assignment over the corpus's actual literal alphabet.
# Arbitrary but fixed: these named literals carry no positional meaning
# of their own in the source grammar.
LETTER_TO_ARG = {
    "a": 1,
    "b": 2,
    "c": 3,
    "d": 4,
    "e": 5,
    "f": 6,
    "g": 7,
    "k": 8,
    "x": 9,
}

_WS_RE = re.compile(r"\s+")


def _strip_ws(s: str) -> str:
    return _WS_RE.sub("", s)


def _split_top_level_comma(s: str) -> tuple[str, str]:
    """Split at the first comma with zero paren nesting -- mirrors
    elegant-normal-forms-python's own `Utilities/BuildTree.splitArgs`.
    """
    depth = 0
    for i, c in enumerate(s):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            return s[:i], s[i + 1 :]
    raise ValueError(f"no top-level comma in {s!r}")


def enf_to_combo(expr: str) -> str:
    """Translate one elegant-normal-forms-python EXPRESSIONS entry into
    combo syntax. See this module's docstring for the grammar mapping and
    what is lossy about it (nothing, for this specific corpus -- every
    literal here is one character, the one case where
    elegant-normal-forms-python's own parser would itself lose
    information).
    """
    s = _strip_ws(expr)
    first = s[0]
    if first in ("&", "|"):
        if s[1] != "(" or s[-1] != ")":
            raise ValueError(f"malformed ENF expression: {expr!r}")
        left, right = _split_top_level_comma(s[2:-1])
        op = "and" if first == "&" else "or"
        return f"{op}({enf_to_combo(left)} {enf_to_combo(right)})"
    if first == "!":
        if s[1] != "(" or s[-1] != ")":
            raise ValueError(f"malformed ENF expression: {expr!r}")
        return f"not({enf_to_combo(s[2:-1])})"
    if first in ("(", ")"):
        raise ValueError(f"invalid ENF expression: {expr!r}")
    if first not in LETTER_TO_ARG:
        raise KeyError(
            f"no combo argument mapping for literal {first!r} in {expr!r} -- "
            "add it to LETTER_TO_ARG"
        )
    return f"${LETTER_TO_ARG[first]}"


# The EXPRESSIONS list from elegant-normal-forms-python's
# Tests/GeneralTestCases.py, copied verbatim (46 raw entries, in their
# original order) so this module has no import-time dependency on that
# sibling checkout. Comments mark the four annotated examples from that
# file (two Holman paper p.19, two Moshe Looks paper).
_GENERAL_TEST_CASES_RAW = [
    "!(a)",
    "!(!(a))",
    "!(&(a, b))",
    "!(|(a, b))",
    "!(!(&(a, b)))",
    "!(!(|(a, b)))",
    "!(&(|(a, b), &(c,d)))",
    "!(&(|(a, b), &(c,d)))",  # exact duplicate of the previous entry
    "&(a, a)",
    "&(a, &(a, c))",
    "&(&(a,b),|(c,d))",
    "&(a, &(b, &(c, &(d,c))))",
    "&(a, |(b, &(c, |(d, |(e, a)))))",
    "&(!(|(|(a, &(!(b), a)), c)), b)",  # Holman paper (page 19)
    "&(a, |(b, &(c, |(d,&(e, !(a))))))",
    "&(a, &(b, &(a, &(a, a))))",
    "&(a, &(b, &(c, &(d, !(b)))))",
    "&(b, !(|(c, |(a, &(!(b), a)))))",
    "&(b, !(|(c, |(a, &(!(b), |(k,x))))))",
    "&(&(&(!(a), |(!(b), !(a))), !(c)), b)",  # Holman paper (page 19)
    "&(b, !(|(c, |(a, &(&(b, &(c, d)), a)))))",
    "&(a, &(b, &(c, &(|(a, |(b, |(c, a))), &(b, &(&(a, a), !(a)))))))",
    "|(a, b)",
    "|(a, a)",
    "|(&(a, b), |(a, c))",
    "|(&(a, b), &(a, b))",
    "|(&(a, b), |(c, d))",
    "|(!(a), |(!(c), !(d)))",
    "|(&(a, |(!(b), !(c))), d)",
    "|(a, |(b, |(c, |(d, c))))",
    "|(a, &(b, &(c, &(d, c))))",
    "|(&(!(a), b), |(!(c), !(d)))",
    "|(a, |(b, |(c, |(d, !(c)))))",
    "|(&(a, b), |(&(a, c), &(a, d)))",
    "|(&(a, b), &(a, &(c, |(d, &(e, a)))))",
    "|(&(a, b), &(a, &(c, |(d, |(e, a)))))",
    "|(&(a, b), &(a, &(c, |(d, |(e, !(a))))))",
    "|(&(a, b), &(a, &(c, |(d, &(e, !(a))))))",
    "|(|(!(a), &(a, &(b,c))), &(b, &(c, !(b))))",
    "|(|(!(a), &(a, &(b, c))), &(b, &(c, !(b))))",  # dup of prev (whitespace only)
    "|(|(!(a), &(a, &(b, c))), &(c, &(b, !(b))))",
    "|(&(&(c, a), d), |(&(c, a), &(&(c, a), e)))",
    "|(|(&(a,b), &(a, &(c, d))), &(a, &(c, &(e, !(a)))))",
    "|(&(a, b), |(&(a, &(c, d)), |(&(a, &(c, e)), &(c, a))))",
    "|(g, &(a, &(b, &(|(!(c), |(!(d), e)), |(c, &(c, f))))))",  # Moshe Looks paper
    "|(g,&(a, &(b, &(c, |(!(d), e)))))",  # Moshe Looks paper's own reduced form of the above
]
assert len(_GENERAL_TEST_CASES_RAW) == 46


def _dedup_enf(raw: list[str]) -> list[str]:
    """Keep the first occurrence of each expression, comparing with all
    whitespace stripped (elegant-normal-forms-python's own BuildTree.py
    does the same normalization before parsing, so two entries differing
    only in whitespace are the SAME test case, not two). Lands on exactly
    44 for `_GENERAL_TEST_CASES_RAW` -- see module docstring.
    """
    seen: set[str] = set()
    out: list[str] = []
    for e in raw:
        key = _strip_ws(e)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


GENERAL_TEST_CASES_ENF = _dedup_enf(_GENERAL_TEST_CASES_RAW)
assert len(GENERAL_TEST_CASES_ENF) == 44

GENERAL_TEST_CASES_COMBO = [enf_to_combo(e) for e in GENERAL_TEST_CASES_ENF]

# ---------------------------------------------------------------------
# CORPUS: the unified list every consumer (capture_reduct.py, and later
# Task 6's own tests) should iterate.
# ---------------------------------------------------------------------


def _make_corpus() -> list[dict]:
    corpus: list[dict] = []

    for i, tree in enumerate(TEST_TREES_SH):
        corpus.append(
            {
                "id": f"testTrees.sh#{i}",
                "source": "testTrees.sh",
                "combo": tree,
                "runnable": True,
                "known_tt_divergent": tree in TEST_TREES_SH_KNOWN_TT_DIVERGENT,
            }
        )

    for i, tree in enumerate(BOOLEAN_REDUCT_UTEST):
        corpus.append(
            {
                "id": f"BooleanReductUTest.cxxtest#test_simplify{i + 1}",
                "source": "BooleanReductUTest.cxxtest",
                "combo": tree,
                "runnable": True,
                "known_tt_divergent": False,
            }
        )

    for i, tree in enumerate(BOOLEAN_FULL_UTEST):
        corpus.append(
            {
                "id": f"BooleanFullUTest.cxxtest#{i}",
                "source": "BooleanFullUTest.cxxtest",
                "combo": tree,
                "runnable": False,
                "reason": (
                    "contains a 0<(...) contin predicate; logical_reduce's "
                    "effort>=2 predicate-recursion path needs contin_reduction "
                    "and contin table evaluation, out of reach of S7's "
                    "boolean-only reduction path"
                ),
            }
        )

    for test_name, tree in FULL_REDUCT_UTEST_UNIQUE:
        corpus.append(
            {
                "id": f"FullReductUTest.cxxtest#{test_name}",
                "source": "FullReductUTest.cxxtest",
                "combo": tree,
                "runnable": False,
                "reason": (
                    "contains a 0<(...) contin predicate and full_reduce requires "
                    "the S14 contin/mixed reduction pipeline"
                ),
            }
        )

    zipped = zip(GENERAL_TEST_CASES_ENF, GENERAL_TEST_CASES_COMBO, strict=True)
    for i, (enf, tree) in enumerate(zipped):
        corpus.append(
            {
                "id": f"GeneralTestCases.py#{i}",
                "source": "GeneralTestCases.py",
                "combo": tree,
                "runnable": True,
                "known_tt_divergent": False,
                "enf_original": enf,
            }
        )

    return corpus


CORPUS = _make_corpus()
assert len(CORPUS) == 29 + 6 + 20 + 4 + 44 == 103
assert sum(1 for c in CORPUS if c["runnable"]) == 29 + 6 + 44 == 79
assert sum(1 for c in CORPUS if not c["runnable"]) == 20 + 4 == 24
