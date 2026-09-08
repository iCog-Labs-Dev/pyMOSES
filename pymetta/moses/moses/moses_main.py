"""C++: moses/moses/moses/moses_main.{h,cc}

The entry point that wires an exemplar list, a scorer and a parameter
bundle into a running search: `run_moses` (dispatch to `local_moses`),
`adjust_termination_criteria`/`autoscale_diversity` (pre-search parameter
tweaks), and `metapop_moses_results_b`/`metapop_moses_results` (build the
optimizer/`deme_expander`/`metapopulation` and drive `run_moses`), per
S11 Task 2's own brief: "the entry point, with hill-climbing injected
directly."

NOT ported, all confirmed by `git grep` to have zero references from
`moses_main.cc`/`local_moses.cc` (the plan's own "Verified: ... zero
references" note):

* `version_string` (moses_main.h:49, moses_main.cc:39-52) -- a
  compile-time git-describe/version string with no runtime effect on the
  search; nothing in this port has an equivalent build step to source it
  from.
* `metapop_printer` (moses_main.h:60-265) -- output formatting (`-W`,
  `-o`, python-format output, ensemble printing): S13, per
  docs/CARRY-FORWARD.md's own classification of `test_output_labels`/
  `test_python_output`. `metapop_moses_results_b`/`metapop_moses_results`
  below still take a `printer` argument, matching the C++ template
  parameter exactly (`printer(metapop, dex, stats)`, called the same way)
  -- callers (this module's own tests, and eventually S13) supply
  whatever callable they need; this file does not implement one.
* `ps_parameters` is dropped from every signature below entirely (not
  merely unused) -- `particle_swarm`/`ps_bscore` are P2 and no
  `ps_parameters` class exists anywhere in this port to pass one of. The
  only C++ use of `ps_params` is constructing `particle_swarm`
  (moses_main.h:298), a branch `metapop_moses_results_b` below raises
  before ever reaching (see its own docstring).

Which `MOSESUTest` cases this module's own test suite can actually
exercise, corrected mid-task from the plan's original (broader) list of
nine: `demo-problems.cc` (`combo_problem`/`parity_problem`/
`multiplex_problem`, registering `-Hpa`/`-Hmux`/`-Hcp`) is itself PLANNED:
P2 (`docs/MODULE-COVERAGE.md`), and every one of those three problems
scores through `logical_bscore` (bscores.h:52) -- `pymetta/moses/scoring/
bscores.py` ports only `ctruth_table_bscore` (its own module docstring
says so explicitly), so `logical_bscore` does not exist in this port
either. That makes SEVEN of the plan's nine "S11-usable" cases
unreachable without porting a whole extra P2 problem-registry file plus
a whole extra scorer class -- scope creep this task does not take:
`test_2_parity`, `test_2_parity_with_labels`, `test_3_parity_multi_thread`
(`-Hpa`), `test_1_mux`, `test_2_mux` (`-Hmux`), `test_cp_simple`, `test_cp`
(`-Hcp`). Only `test_it`/`test_it_target_ignore` go through the OTHER
problem-registry file, `table-problems.cc`'s `it_table_problem`
(`REGRESSION(ctable, ctruth_table_bscore, (ctable))`, table-problems.cc:
479) -- already-ported machinery end to end -- so those two are this
module's actual measured oracle (`test_moses_main.py`'s own "Live oracle"
section). A later slice that ports `demo-problems.cc` + `logical_bscore`
can pick the other seven back up; nothing here forecloses that.

Gated to raise `NotImplementedError` rather than silently no-opping, per
the plan's own instruction:

* `run_moses`'s `mpi`/`distributed` branches (`mpi_moses` compiles to
  nothing -- `HAVE_MPI` is never defined, CLAUDE.md's own "Known sharp
  edges"; `distributed_moses` is a different, out-of-scope `-j host:N`
  mechanism that shells out to `ssh`).
* `metapop_moses_results_b`'s non-`"hc"` `opt_algo` branches (`sa`/`un`/
  `ps` construct `simulated_annealing`/`univariate_optimization`/
  `particle_swarm` in C++, moses_main.h:288-307 -- none are ported; S11
  wires hill-climbing in directly, per the plan's "Optimizer selection is
  dependency-injected" note).
* `metapop_moses_results`'s SS-fitness branch (`ss_bscore.{h,cc}`, P2 --
  see the function's own docstring for how the deliberately-unset
  `subsample_deme_filter_parameters` fields this check reads are
  handled).

# DIVERGENCE(C++ moses/moses/scoring/behave_cscore.h:60-69,127): classic's
# `behave_cscore` exposes `min_improv()` (header-inline, `return
# _bscorer.min_improv();`) and `get_bscorer()` (`return _bscorer;`) as
# genuine public one-line delegations/accessors to `_bscorer` -- but
# `best_possible_score()`/`worst_possible_score()` are NOT one-line
# delegations: only DECLARED in the header, they are DEFINED out-of-line
# in behave_cscore.cc:123-131 as `boost::accumulate(_bscorer.best_
# possible_bscore(), 0.0)` / `boost::accumulate(_bscorer.worst_possible_
# bscore(), 0.0)` -- a SUM over the bscorer's own vector-valued result,
# not a bare forward of a same-named call. This port's
# `behave_cscore` (behave_cscore.py) does not port any of the four --
# its own module docstring says so explicitly and names the workaround:
# "Call `.get_bscorer()` (i.e. the `behave_cscore` instance's own
# `_bscorer` attribute) directly instead." `adjust_termination_criteria`
# and `autoscale_diversity` below are the first callers in this port that
# actually need `best_possible_score`/`worst_possible_score`/
# `min_improv` (moses_main.cc:79,90,100,107), so they follow that
# instruction literally: `sc._bscorer.best_possible_bscore()` etc.,
# reached through the "private" attribute rather than a public delegating
# method that does not exist here. `best_possible_score()`/
# `worst_possible_score()` themselves are two-line C++ functions
# (`boost::accumulate(_bscorer.best_possible_bscore(), 0.0)`,
# behave_cscore.cc:123-131) reproduced inline via `bscore_base.sum_bscore`
# (scoring_base.py), which is exactly `boost::accumulate(bs, 0.0)`
# narrowed once at return -- see that method's own docstring.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

from pymetta.comboreduct.combo.vertex import combo_tree
from pymetta.comboreduct.type_checker.type_signature import TypeSignature
from pymetta.moses.deme.deme_expander import Simplifier, deme_expander
from pymetta.moses.deme.deme_params import deme_parameters
from pymetta.moses.metapopulation.metapop_params import (
    metapop_parameters,
    subsample_deme_filter_parameters,
)
from pymetta.moses.metapopulation.metapopulation import metapopulation
from pymetta.moses.moses.moses_params import moses_parameters, moses_statistics
from pymetta.moses.moses.optimization.hill_climbing import hc_parameters, hill_climbing
from pymetta.moses.moses.optimization.optimization import optim_parameters
from pymetta.moses.moses.types import _to_score_t, score_t, very_best_score
from pymetta.moses.scoring.behave_cscore import behave_cscore

Printer = Callable[[metapopulation, deme_expander, moses_statistics], None]

__all__ = [
    "Printer",
    "adjust_termination_criteria",
    "autoscale_diversity",
    "metapop_moses_results",
    "metapop_moses_results_b",
    "run_moses",
]


def run_moses(
    metapop: metapopulation,
    dex: deme_expander,
    pa: moses_parameters,
    stats: moses_statistics,
) -> None:
    """C++: moses/moses/moses/moses_main.cc:54 run_moses

    "Run moses, either on localhost, or distributed." (the function's own
    comment.) Only the `pa.local` branch is ported -- see the module
    docstring for why `pa.mpi`/the plain `else` (distributed) branches
    raise instead.

    The real import of `local_moses` (S11 Task 1, a sibling module owned
    by a concurrently-implemented task) is deferred to here, INSIDE the
    function body, rather than hoisted to this module's top-level
    imports: that keeps every other symbol in this file -- and every
    test of them -- importable and runnable independently of whether
    `local_moses.py` has landed yet, exactly as this task's own brief
    instructs ("write your module and tests against it anyway ... do not
    reimplement it"). Purely a Python import-organization choice with no
    C++ analogue (an `#include` is a compile-time dependency either way in
    C++, so classic has nothing to stagger here) -- not a `# DIVERGENCE`.
    """
    if pa.local:
        from pymetta.moses.moses.local_moses import local_moses

        local_moses(metapop, dex, pa, stats)
    elif pa.mpi:
        raise NotImplementedError(
            "run_moses: mpi_moses is dead code in any build this project "
            "produces (FIND_PACKAGE(MPI) is commented out of CMake, so "
            "HAVE_MPI is never defined and mpi_moses.cc's entire body is "
            "#ifdef'd out -- CLAUDE.md's own 'Known sharp edges'). "
            "Construct moses_parameters(local=True) (the default) and "
            "leave .mpi at its default (False)."
        )
    else:
        raise NotImplementedError(
            "run_moses: distributed_moses (-j host:N, shells out via "
            "ssh/system()/pgrep) is a different, out-of-scope mechanism, "
            "not ported by this slice. Construct moses_parameters("
            "local=True) (the default)."
        )


def adjust_termination_criteria(
    c_scorer: behave_cscore,
    opt_params: optim_parameters,
    moses_params: moses_parameters,
) -> None:
    """C++: moses/moses/moses/moses_main.cc:73 adjust_termination_criteria

    "This twiddles the termination criteria for deciding when to finish
    the search." (the function's own comment, typo included.) "An
    explicit user-specified max score always over-rides the inferred
    score" (moses_main.cc:78's own comment): the CLI's own `--max-score`
    default is `very_best_score` (`problem-params.cc:253`,
    `default_value(very_best_score)`), NOT `moses_parameters`'s own bare
    constructor default of `0` (`moses_params.py`'s own docstring already
    flags this CLI-vs-struct-default gap) -- a caller reproducing a real
    CLI run must construct `moses_parameters(max_score=very_best_score)`
    to get the "infer automatically" behaviour this function implements.

    Every intermediate here is `score_t` (narrowed via `_to_score_t`),
    matching moses_main.cc:79,90,96-97,100's own `score_t`-typed locals
    and assignments exactly -- `target_score`/`actual_min_improv` are each
    re-narrowed at every arithmetic step, not computed once in double and
    narrowed only at the end.
    """
    target_score: score_t = c_scorer._bscorer.sum_bscore(c_scorer._bscorer.best_possible_bscore())
    if very_best_score != moses_params.max_score:
        target_score = moses_params.max_score
    # else: keep the inferred `target_score` (moses_main.cc:80-85).

    # "negative min_improv is interpreted as percentage of improvement,
    # if so then don't subtract anything" (moses_main.cc:87-89's own
    # comment).
    actual_min_improv: score_t = max(c_scorer._bscorer.min_improv(), _to_score_t(0))
    target_score = _to_score_t(target_score - actual_min_improv)

    opt_params.terminate_if_gte = target_score
    moses_params.max_score = target_score
    opt_params.set_min_score_improv(c_scorer._bscorer.min_improv())


def autoscale_diversity(sc: behave_cscore, meta_params: metapop_parameters) -> None:
    """C++: moses/moses/moses/moses_main.cc:103 autoscale_diversity

    "Autoscale diversity pressure to match the magnitude of the scorer"
    (the function's own doc comment). A no-op whenever diversity is
    disabled (`meta_params.diversity.enabled()` is `False` by default,
    `pressure == 0.0` -- `metapop_params.py`'s own module docstring) --
    the short-circuiting `and` below means `.autoscale` -- left unset by
    `diversity_parameters.__init__` UNCONDITIONALLY, not merely "when
    diversity is off": `metapop_params.cc:27-34`'s constructor never
    touches this field at all, regardless of what `pressure`/`enabled()`
    end up being (that class's own DIVERGENCE note) -- is never even read
    on this short-circuited path, matching C++'s own `&&` exactly (C++
    itself never evaluates `.autoscale` there either, so its own
    genuinely-indeterminate default never gets read on this path; it is
    this port's explicit `AttributeError`-on-read behavior, not a
    diversity-off-specific one, that this short-circuit avoids
    triggering).

    `magnitude`/`meta_params.diversity.pressure` are each `score_t`,
    narrowed at every step, matching moses_main.cc:107-108's float32
    arithmetic (`sc.best_possible_score() - sc.worst_possible_score()`
    and `meta_params.diversity.pressure *= magnitude` are each a single
    float32 operation in C++, not a double one narrowed once at the end).
    """
    if meta_params.diversity.enabled() and meta_params.diversity.autoscale:
        best = sc._bscorer.sum_bscore(sc._bscorer.best_possible_bscore())
        worst = sc._bscorer.sum_bscore(sc._bscorer.worst_possible_bscore())
        magnitude: score_t = _to_score_t(best - worst)
        meta_params.diversity.pressure = _to_score_t(meta_params.diversity.pressure * magnitude)
        meta_params.diversity.set_dst2dp(meta_params.diversity.dst2dp_type)


def metapop_moses_results_b(
    bases: list[combo_tree],
    tt: TypeSignature,
    si_ca: Simplifier,
    si_kb: Simplifier,
    sc: behave_cscore,
    opt_params: optim_parameters,
    hc_params: hc_parameters,
    deme_params: deme_parameters,
    filter_params: subsample_deme_filter_parameters,
    meta_params: metapop_parameters,
    moses_params: moses_parameters,
    printer: Printer,
) -> None:
    """C++: moses/moses/moses/moses_main.h:270 metapop_moses_results_b

    "Create metapopulation, run moses, print results." (the template's
    own doc comment.) `ps_params` is dropped from this signature entirely
    -- see the module docstring. Only the `opt_params.opt_algo == "hc"`
    branch (moses_main.h:288-290) is ported; every other branch
    (moses_main.h:291-307, `sa`/`un`/`ps`) raises instead of constructing
    an optimizer this port does not have.

    `optimizer` is not explicitly `delete`d at the end (moses_main.h:323)
    -- nothing to free in Python; ordinary garbage collection reclaims it
    once `dex` (the only other holder of a reference to it) itself goes
    out of scope.

    "This seems kind of cheesy ... shouldn't the exemplars already be
    simplified, by now?" (moses_main.h:309-310's own comment, reproduced
    verbatim): `combo_tree siba(xmplr); si_ca(siba);` (moses_main.h:313-
    314) copy-constructs each base exemplar before simplifying it in
    place, leaving the caller's own `bases` list/trees untouched -- ported
    via `combo_tree(xmplr.begin())`, the deep-copying constructor overload
    (`util/tree.py`'s own docstring for why passing a `tree_node` root
    triggers that path), matching C++'s copy constructor exactly rather
    than aliasing the caller's tree.
    """
    stats = moses_statistics()

    if opt_params.opt_algo != "hc":
        raise NotImplementedError(
            f"metapop_moses_results_b: optimizer {opt_params.opt_algo!r} "
            "is P2 -- star-anneal/univariate/particle-swarm are not "
            "ported. moses_main.h:37-39's own #includes for them are NOT "
            "the only references: moses_main.h:292/295/298 construct all "
            "three (`new simulated_annealing`/`new univariate_optimization`/"
            "`new particle_swarm`) inside this very function's C++ "
            "original, `metapop_moses_results_b` -- neither moses_main.cc "
            "nor local_moses.cc dispatches to any of the three by name, "
            "but the header this function is ported from does construct "
            "them. For 'sa'/'un'/'ps' specifically, this matches one of "
            "C++'s own recognized branches (moses_main.h:291-299) -- just "
            "one this port does not implement. For any OTHER string, C++ "
            "does something different again: it falls through to the "
            "`else` (moses_main.h:300-306), prints 'Unknown optimization "
            "algo ...' to stderr, and calls exit(1) -- a hard process "
            "exit, not a raised/catchable error in either language. This "
            "port raises NotImplementedError uniformly for every "
            "non-'hc' value, recognized-but-unported or genuinely unknown "
            "alike, which is itself a divergence from that exit(1) path, "
            "not a port of it. Construct optim_parameters(opt_algo='hc') "
            "(the default) so S9's hill_climbing is wired in directly, "
            "per the plan's own 'Optimizer selection is dependency-"
            "injected' note."
        )
    optimizer = hill_climbing(opt_params, hc_params)

    simple_bases: list[combo_tree] = []
    for xmplr in bases:
        siba = combo_tree(xmplr.begin())
        si_ca(siba)
        simple_bases.append(siba)

    dex = deme_expander(tt, si_ca, si_kb, sc, optimizer, deme_params, filter_params)
    metapop = metapopulation(simple_bases, sc, meta_params, filter_params)

    run_moses(metapop, dex, moses_params, stats)
    printer(metapop, dex, stats)


def metapop_moses_results(
    bases: list[combo_tree],
    type_sig: TypeSignature,
    si_ca: Simplifier,
    si_kb: Simplifier,
    c_scorer: behave_cscore,
    opt_params: optim_parameters,
    hc_params: hc_parameters,
    deme_params: deme_parameters,
    filter_params: subsample_deme_filter_parameters,
    meta_params: metapop_parameters,
    moses_params: moses_parameters,
    printer: Printer,
) -> None:
    """C++: moses/moses/moses/moses_main.h:342 metapop_moses_results

    "like [metapop_moses_results_b] above, but assumes that the score is
    bscore based" (moses_main.h:340-341's own comment). `ps_params` is
    dropped -- see the module docstring.

    "Parameters that might get tweaked are copied" (moses_main.h:357's
    own comment): C++'s copy constructors are compiler-generated
    memberwise copies, which -- for `metapop_parameters`/`moses_parameters`
    -- recursively value-copy their own class-typed members too
    (`diversity`/`ensemble_params`, `jobs`), not merely alias them. A bare
    `copy.copy(meta_params)` would ALIAS `.diversity`/`.ensemble_params`
    (a shallow copy only copies the reference), so `autoscale_diversity`
    mutating `twk_meta_params.diversity.pressure` below would silently
    corrupt the caller's own `meta_params.diversity` too -- exactly the
    aliasing trap `metapop_parameters.__init__`'s own docstring already
    documents and avoids for its OWN `diversity` constructor parameter.
    Each nested value member is therefore re-copied explicitly here, the
    same way, to reproduce C++'s recursive memberwise copy. `optim_parameters`
    has no class-typed members, so a bare `copy.copy` reproduces its copy
    constructor exactly with no further work needed.

    # DIVERGENCE(C++ moses/moses/moses/moses_main.h:364-365,
    # moses/moses/metapopulation/metapop_params.h:138-155): classic reads
    # `filter_params.n_subsample_fitnesses`/`.low_dev_pressure` here
    # completely unconditionally (there is no gate on `n_subsample_demes`
    # or anything else guarding this specific read) -- and, per this
    # port's OWN `subsample_deme_filter_parameters` (metapop_params.py),
    # both fields are DELIBERATELY left unset on a default-constructed
    # instance (poisoned: reading either raises `AttributeError`), because
    # nothing else in the S10/S11 slices was expected to read them before
    # S13's CLI wiring populates them. This function is the first reader
    # in this port that actually needs to. Reading them unconditionally
    # here would make `metapop_moses_results` unusable by every ordinary
    # caller that passes a plain default `subsample_deme_filter_parameters
    # ()` -- including every S11-usable `MOSESUTest` case -- which is
    # strictly worse than classic's own behaviour (there, the two fields
    # hold genuinely indeterminate stack garbage, but in practice a fresh
    # stack frame's bytes are far more often zero/small than they are a
    # huge value that would spuriously satisfy `> 1`, so the SS-fitness
    # branch is not normally taken by accident either). `getattr(...,
    # 0)`/`getattr(..., 0.0)` below therefore treats "field never set" as
    # "not configured" (`0`/`0.0`, both fail their own `>` check) --
    # always well-defined here, unlike C++'s UB, and behaviourally
    # equivalent to the overwhelmingly common real case. A caller that
    # deliberately wants SS-fitness must set BOTH fields explicitly (the
    # same obligation `subsample_deme_filter_parameters`'s own DIVERGENCE
    # note already places on whoever ports the S13 `ss-*` CLI options) --
    # once set, `getattr`'s fallback is never reached and the check below
    # behaves exactly like C++'s.
    """
    twk_opt_params = copy.copy(opt_params)
    twk_meta_params = copy.copy(meta_params)
    twk_meta_params.diversity = copy.copy(meta_params.diversity)
    twk_meta_params.ensemble_params = copy.copy(meta_params.ensemble_params)
    twk_moses_params = copy.copy(moses_params)
    twk_moses_params.jobs = dict(moses_params.jobs)

    adjust_termination_criteria(c_scorer, twk_opt_params, twk_moses_params)
    autoscale_diversity(c_scorer, twk_meta_params)

    n_subsample_fitnesses = getattr(filter_params, "n_subsample_fitnesses", 0)
    low_dev_pressure = getattr(filter_params, "low_dev_pressure", 0.0)
    if n_subsample_fitnesses > 1 and low_dev_pressure > 0.0:
        raise NotImplementedError(
            "metapop_moses_results: SS-fitness (moses/moses/scoring/"
            "ss_bscore.{h,cc}) is P2, not ported. Leave "
            "subsample_deme_filter_parameters.n_subsample_fitnesses/"
            ".low_dev_pressure unset (the default) so this branch is "
            "never taken."
        )

    metapop_moses_results_b(
        bases,
        type_sig,
        si_ca,
        si_kb,
        c_scorer,
        twk_opt_params,
        hc_params,
        deme_params,
        filter_params,
        twk_meta_params,
        twk_moses_params,
        printer,
    )
