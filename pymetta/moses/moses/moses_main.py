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
* `ps_parameters` is dropped from the two signatures that carry it in
  C++ (`metapop_moses_results_b`, moses_main.h:278; `metapop_moses_
  results`, moses_main.h:350) -- but NOT because particle swarm is
  unported. CORRECTED (this bullet previously said "`particle_swarm`/
  `ps_bscore` are P2 and no `ps_parameters` class exists anywhere in this
  port", and that "a branch `metapop_moses_results_b` below raises before
  ever reaching" it; both were true when written and are false now):
  `ps_parameters` and `particle_swarm` are BOTH ported
  (`optimization/particle_swarm.py`, imported at the top of this module),
  and the `ps` branch of `metapop_moses_results_b` constructs a real
  `particle_swarm` rather than raising.

  What survives is only the parameter-THREADING: `ps_params` is the one
  optimizer-parameter bundle these two functions do not accept from their
  caller, so the `ps` branch constructs `ps_parameters()` with classic's
  own defaults instead. The two other C++ mentions of `ps_params`
  (moses_main.h:376,382) are pure pass-throughs from
  `metapop_moses_results` to `metapop_moses_results_b`; moses_main.h:298
  is the only place its VALUE is read. A run that customised `ps_params`
  in C++ would therefore not be reproduced -- see the DIVERGENCE marker
  on that branch.

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

## `metapop_printer` (S13 Task 3 addition)

`moses_main.h:60-265 struct metapop_printer` -- output formatting (`-S`/
`-x`/`-t`/`-C`/`-V`/`-W`/`--output-deme-id`/`-o`/`--output-format`,
python-format output, ensemble printing). Was explicitly NOT ported by
this module's S11 lineage (docs/CARRY-FORWARD.md's own classification of
`test_output_labels`/`test_python_output` as S13 work); lands here now,
since `moses_main.h:60` -- not any `main/` file -- is its real C++ home
(this task's own plan doc, "Two measured scoping facts"). String
rendering it depends on (`ostream_scored_combo_tree`, `ostream_behavioral_
score`, `composite_score.__str__`, `combo_tree_to_str` with `labels`/
`output_format`) lives in `types.py`/`iostream_combo.py` respectively --
see those modules' own docstrings for what each one ports.

`GATHER_STATS` block (moses_main.h:229-246) is `#ifdef`-gated debug-only
code with no corresponding CMake define anywhere in classic's build
(grepped) -- dead in every real build, not ported, matching this whole
codebase's convention for `#ifdef`-excluded C++ (CLAUDE.md's own "Known
sharp edges" section names the same pattern for `HAVE_MPI`). The
`logger().is_info_enabled()` block (moses_main.h:204-227) is likewise not
ported -- no `Logger` surface exists anywhere in this port
(`scoring_base.py`'s own module docstring notes the identical omission
for a `logger().info()` call it does not reach).

`metapop_printer`'s zero-argument constructor (moses_main.h:62,
`metapop_printer() {}`) is not ported: in C++ it leaves every POD member
genuinely uninitialized (read-before-write UB) and exists only so
`problem_params::mmr_pa` (a plain struct member with no initializer,
`problem-params.h:227`) has SOME default-constructed value before
`problem_params::parse_options` overwrites it with the real 13-argument
constructor's result later in the same call. This port's `moses_exec.py`
builds a `metapop_printer` in one step, fully populated, with no
intervening "uninitialized member" phase for any code to observe --
so there is no uninitialized state to reproduce, and the 13-argument
constructor below has no defaults (every field is mandatory).

Gated to raise `NotImplementedError` rather than silently no-opping, per
the plan's own instruction:

* `run_moses`'s `mpi`/`distributed` branches (`mpi_moses` compiles to
  nothing -- `HAVE_MPI` is never defined, CLAUDE.md's own "Known sharp
  edges"; `distributed_moses` is a different, out-of-scope `-j host:N`
  mechanism that shells out to `ssh`).
* `metapop_moses_results_b`'s UNRECOGNISED-`opt_algo` branch
  (moses_main.h:300-306) -- and that one raises `ValueError`, not
  `NotImplementedError`, because C++ `exit(1)`s there rather than
  throwing; see the DIVERGENCE marker at the dispatch itself.
  CORRECTED: this bullet used to claim the `sa`/`un`/`ps` branches were
  gated too ("none are ported; S11 wires hill-climbing in directly").
  All four of classic's named algorithms -- `hc`, `sa`, `un`, `ps`
  (moses_main.h:288-299) -- now construct a real optimizer here
  (`hill_climbing`, `simulated_annealing`, `univariate_optimization`,
  `particle_swarm`), and nothing in the dispatch raises for a recognised
  name.
* `metapop_moses_results`'s SS-fitness branch (`ss_bscore.{h,cc}`, P2 --
  see the function's own docstring for how the deliberately-unset
  `subsample_deme_filter_parameters` fields this check reads are
  handled).
* `metapop_printer.__call__`'s `output_demeID=True` branch (this port's
  `scored_combo_tree` carries no `demeID_t`, see `types.py`'s own
  docstring). That is the only gate `metapop_printer` raises itself.

  CORRECTED: this bullet used to list `output_ensemble=True` alongside
  it, "(`ensemble.py`'s `get_weight`/`get_weighted_tree`/`flat_score` are
  themselves unported stubs that raise)". Both halves are now false.
  `ensemble.py` is fully ported -- `get_weighted_tree` (ensemble.cc:280)
  and `flat_score` (ensemble.cc:427) are real implementations, not stubs
  -- and `metapop_printer.__call__` never gated `output_ensemble` at all:
  that whole branch (moses_main.h:111-136) is ported inline with no
  raise anywhere in it. What actually stops it is one layer down, and
  which thing stops it depends on the output format:

  - combo (and every non-python) format prints `ensm.flat_score()`,
    which calls `bscore_base.call_ensemble` -- a RAISING base body in
    both languages (`OC_ASSERT(false, "Ensemble scoring not implemented
    for bscorer %s", ...)`, scoring_base.cc:74-75), with no concrete
    override anywhere in this port yet (`ensemble.py`'s own module
    docstring, "Two REAL prerequisite gaps found", item 3).
  - `python`/`python3` format prints `sct.get_weight()` once per
    ensemble member (moses_main.h:124). That accessor exists now
    (`types.py`), but `ensemble.py` keeps its real boosting weights in a
    private identity-keyed side dict rather than on the
    `scored_combo_tree`, so every member would print a flat `1.0`.
    Explicitly gated with a `NotImplementedError` at that loop -- see
    the comment there. Before `get_weight` was ported this line died with
    `AttributeError`; the gate exists so that adding the accessor could
    not silently turn that failure into wrong output.

  Reaching `output_ensemble=True` through a real CLI run is separately
  blocked one layer UP, and that part of the old bullet was right:
  `output_ensemble` is wired straight from `--boost` (`moses_exec.py`;
  classic passes `boosting` into the same constructor slot,
  problem-params.cc:2400-2403), and `problem_params.py` gates `--boost`.

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
import sys
from collections.abc import Callable, Sequence

from pymetta.comboreduct.combo.iostream_combo import combo_tree_to_str, output_format
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
from pymetta.moses.moses.optimization.optimization import (
    optim_parameters,
    optimizer_base,
)
from pymetta.moses.moses.optimization.particle_swarm import (
    particle_swarm,
    ps_parameters,
)
from pymetta.moses.moses.optimization.star_anneal import simulated_annealing
from pymetta.moses.moses.optimization.univariate import univariate_optimization
from pymetta.moses.moses.types import (
    _format_score_t,
    _to_score_t,
    ostream_scored_combo_tree,
    score_t,
    very_best_score,
)
from pymetta.moses.scoring.behave_cscore import behave_cscore

Printer = Callable[[metapopulation, deme_expander, moses_statistics], None]

__all__ = [
    "Printer",
    "adjust_termination_criteria",
    "autoscale_diversity",
    "metapop_moses_results",
    "metapop_moses_results_b",
    "metapop_printer",
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


class metapop_printer:
    """C++: moses/moses/moses/moses_main.h:60 metapop_printer

    "Print metapopulation results to stdout, logfile, etc." (the struct's
    own doc comment.) See the module docstring's "`metapop_printer` (S13
    Task 3 addition)" section for what is and is not ported, and why the
    zero-argument C++ constructor has no equivalent here.
    """

    def __init__(
        self,
        result_count: int,
        output_score: bool,
        output_cscore: bool,
        output_bscore: bool,
        output_only_best: bool,
        output_ensemble: bool,
        output_eval_number: bool,
        output_with_labels: bool,
        output_demeID: bool,
        ilabels: Sequence[str],
        output_file: str,
        fmt: output_format,
        is_mpi: bool,
    ) -> None:
        """C++: moses/moses/moses/moses_main.h:63 metapop_printer::metapop_printer

        The 13-argument constructor (moses_main.h:63-88), a plain
        memberwise assignment with no logic of its own. `ilabels` is
        `public` in C++ (moses_main.h:260, the ONE field not under
        `private:`) -- no other field is ever read from outside this
        class in classic either, so that visibility split has no
        observable Python equivalent to reproduce; every field is a
        plain instance attribute here.
        """
        self.result_count = result_count
        self.output_score = output_score
        self.output_cscore = output_cscore
        self.output_bscore = output_bscore
        self.output_only_best = output_only_best
        self.output_ensemble = output_ensemble
        self.output_eval_number = output_eval_number
        self.output_with_labels = output_with_labels
        self.output_demeID = output_demeID
        self.ilabels = list(ilabels)
        self.output_file = output_file
        self.fmt = fmt
        self.is_mpi = is_mpi

    def _labels(self) -> list[str]:
        """Not a C++ member -- factors out `output_with_labels? ilabels :
        std::vector<std::string>()` (moses_main.h:126-127,178-179,186-187),
        the ternary every combo-tree-printing call site in `__call__`
        below repeats verbatim: labels are substituted only when `-W`
        (`output_with_labels`) is set, otherwise printing falls back to
        bare `$1`/`$2`-style indices.
        """
        return self.ilabels if self.output_with_labels else []

    def __call__(
        self,
        metapop: metapopulation,
        dex: deme_expander,
        stats: moses_statistics,
    ) -> None:
        """C++: moses/moses/moses/moses_main.h:93 metapop_printer::operator()

        Ported line-for-line off moses_main.h:93-247 (quoted in full in
        this task's own research notes); see the module docstring's
        "Gated to raise" section for the one branch this class gates
        itself (`output_demeID`), for what actually stops
        `output_ensemble=True` one layer DOWN (it is not gated here), and
        for the `GATHER_STATS`/`logger().is_info_enabled()` blocks that
        are skipped entirely rather than ported. `dex` is accepted, unused,
        purely to match the real 3-argument call signature
        `printer(metapop, dex, stats)` (moses_main.h:322,
        `metapop_moses_results_b`) -- the only thing classic's own
        `operator()` ever does with it is the `#ifdef GATHER_STATS` block.
        """
        del dex
        if self.is_mpi and metapop.size() == 0:
            return

        parts: list[str] = []
        if self.output_ensemble:
            tree_set = metapop.get_ensemble().get_ensemble()
            if self.fmt in (output_format.python, output_format.python3):
                parts.append(
                    "#!/usr/bin/env python\n"
                    if self.fmt == output_format.python
                    else "#!/usr/bin/env python3\n"
                )
                parts.append(f"#score: {_format_score_t(metapop.best_score())}\n")
                parts.append("def moses_eval(i):\n")
                parts.append("    sum = 0.0 \\\n")
                # GATE, and it is the whole reason it exists: the loop
                # below prints `sct.get_weight()` per ensemble member
                # (moses_main.h:124). `scored_combo_tree.get_weight()`
                # exists now (types.py) and returns the constructor's own
                # 1.0 default -- but this port's `ensemble.py` does not
                # store its AdaBoost/experts weights THERE. It keeps them
                # in a private `_sct_weights` dict keyed by object
                # identity (its own module docstring, "Two REAL
                # prerequisite gaps found", item 1), so every member would
                # print a flat `1.0` where classic prints its real alpha.
                # Before `get_weight` was ported this line died with
                # `AttributeError`; adding the accessor would otherwise
                # have turned that accidental failure into SILENTLY WRONG
                # output, which is the one outcome this project's gating
                # rule exists to prevent. Raise instead, until `ensemble.py`
                # is rewired onto the real field.
                if tree_set:
                    raise NotImplementedError(
                        "metapop_printer: --output-format=python with an "
                        "ensemble prints scored_combo_tree.get_weight() per "
                        "member (moses_main.h:124), but ensemble.py keeps its "
                        "boosting weights in a private _sct_weights dict keyed "
                        "by object identity, not on the scored_combo_tree -- so "
                        "every member would print 1.0 instead of its real "
                        "AdaBoost weight. Rewire ensemble.py onto "
                        "scored_combo_tree.set_weight() first."
                    )
                for sct in tree_set:
                    parts.append(f"      + {_format_score_t(sct.get_weight())} * ")
                    parts.append(combo_tree_to_str(sct.get_tree(), self._labels(), self.fmt))
                    parts.append("\\\n")
                # "return (0.0 < val)" -- NOT "sum": `val` is never defined
                # anywhere in this generated program. A real bug in
                # classic itself (moses_main.h:129), reproduced verbatim,
                # not "fixed".
                parts.append("\n    return (0.0 < val)\n")
            else:
                ensm = metapop.get_ensemble()
                # `ensm.get_weighted_tree()` is printed via a bare
                # `operator<<(ostream&, const combo_tree&)` in C++
                # (moses_main.h:135) -- NOT `ostream_combo_tree` with this
                # printer's own `labels`/`fmt`. That default is combo
                # format with no labels, exactly `combo_tree_to_str`'s own
                # defaults -- so labels/output-format are IGNORED here
                # regardless of `-W`/`--output-format`, a real, confirmed
                # asymmetry with the non-ensemble branch below (which
                # threads both through). Reproduced faithfully, not
                # "fixed" to match the other branch.
                parts.append(f"{_format_score_t(ensm.flat_score())} ")
                parts.append(combo_tree_to_str(ensm.get_weighted_tree()))
                parts.append("\n")
        else:
            if self.output_only_best:
                tree_set = list(metapop.best_candidates())
            else:
                tree_set = list(metapop.get_trees())

            cnt = 0
            for sct in tree_set:
                # `if (result_count == cnt++) break;` -- compares the
                # PRE-increment value, then increments; a negative
                # `result_count` can never equal a non-negative `cnt`, so
                # every candidate prints (matching `-c`'s own help text:
                # "If negative, then return all results.").
                if self.result_count == cnt:
                    break
                cnt += 1
                if self.fmt in (output_format.python, output_format.python3):
                    parts.append(
                        "#!/usr/bin/env python\n"
                        if self.fmt == output_format.python
                        else "#!/usr/bin/env python3\n"
                    )
                    parts.append(f"#score: {_format_score_t(sct.get_score())}\n")
                    parts.append("import operator as op\n")
                    parts.append("from functools import reduce\n")
                    parts.append("from math import log, exp, sin\n")
                    parts.append("def l0(i): return 0 < i\n")
                    parts.append("def adds(*args): return sum(args)\n")
                    parts.append("def muls(*args): return reduce(op.mul, args)\n")
                    parts.append("def pdiv(a, b): return a / (b + 0.000001)\n")
                    parts.append("def impulse(a): return 1.0 if a else 0.0\n")
                    parts.append("def moses_eval(i):\n")
                    parts.append("    return ")
                    parts.append(combo_tree_to_str(sct.get_tree(), self._labels(), self.fmt))
                    parts.append("\n")
                else:
                    parts.append(
                        ostream_scored_combo_tree(
                            sct,
                            self.output_score,
                            self.output_cscore,
                            self.output_demeID,
                            self.output_bscore,
                            self._labels(),
                            self.fmt,
                        )
                    )

        if self.output_eval_number:
            parts.append(f"n_evals: {stats.n_evals}\n")

        text = "".join(parts)
        if not self.output_file:
            sys.stdout.write(text)
        else:
            with open(self.output_file, "w") as f:
                f.write(text)

        # `logger().is_info_enabled()` block (moses_main.h:204-227) and the
        # `#ifdef GATHER_STATS` block (moses_main.h:229-246) are not
        # ported -- see module docstring.


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
    -- see the module docstring and the DIVERGENCE at the `ps` branch.
    All four of classic's optimizer branches (moses_main.h:288-299) are now
    ported: `hc`, `sa`, `un` and `ps`. The unrecognised-name branch
    (:300-306) diverges deliberately -- C++ exits the process, this port
    raises -- and carries its own marker.

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

    # C++: moses/moses/moses/moses_main.h:288-306 -- the optimizer dispatch.
    # Classic recognises exactly four names, declared at optimization.h:66-69
    # as `un`/`sa`/`hc`/`ps`, and constructs one of four types; anything else
    # prints "Unknown optimization algo ..." to stderr and calls exit(1).
    #
    # DIVERGENCE(C++ moses/moses/moses/moses_main.h:300-306): for an
    # UNRECOGNISED algo name C++ performs a hard `exit(1)` after writing to
    # stderr -- not a throw, and not catchable in either language. This port
    # raises ValueError instead, carrying classic's own message text. A
    # process exit is not reproducible as library behaviour, and swallowing
    # the name silently would be worse than either.
    if opt_params.opt_algo == "hc":
        optimizer: optimizer_base = hill_climbing(opt_params, hc_params)
    elif opt_params.opt_algo == "sa":
        optimizer = simulated_annealing(opt_params)
    elif opt_params.opt_algo == "un":
        optimizer = univariate_optimization(opt_params)
    elif opt_params.opt_algo == "ps":
        # DIVERGENCE(C++ moses/moses/moses/moses_main.h:298): C++ threads a
        # caller-supplied `ps_params` through this signature; this port
        # dropped that parameter (see the docstring below), so the defaults
        # are used. Those defaults are ported verbatim from
        # `particle-swarm.h`, so a default-configured run matches; a run
        # that customised ps_params in C++ would not.
        optimizer = particle_swarm(opt_params, ps_parameters())
    else:
        raise ValueError(
            f"Unknown optimization algo {opt_params.opt_algo}. Supported "
            "algorithms are un (for univariate), sa (for star-shaped "
            "search), hc (for local search) and ps (for particle swarm)"
        )

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
