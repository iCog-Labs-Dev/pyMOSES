"""Estimation-of-distribution machinery for `univariate_optimization`.

C++: `moses/moses/eda/`. This port carries only what `univariate.py` actually
reaches through `univariate.cc`'s own include list (`termination.h`,
`replacement.h`, `logging.h`, `local_structure.h`, `optimize.h`,
`../moses/neighborhood_sampling.h` -- the last one already ported):

* `termination.py` -- `terminate_if_gte_or_no_improv` (used) and
  `terminate_if_gte` (not used by `univariate.cc`, ported alongside it since
  the whole header is 56 lines and the two structs are inseparable in C++).
* `replacement.py` -- `replace_the_worst` only. `rtr_replacement` is NOT
  ported: `univariate.cc:83-84` calls it only inside a COMMENTED-OUT line
  (`// rtr_replacement(deme.fields(), ...)`), so it has zero live call sites
  anywhere this whole port's scope touches.
* `logging.py` -- `cout_log_best_and_gen`, kept as a documented no-op (see
  that module's own docstring) matching `optimization.py`'s and
  `hill_climbing.py`'s own "pure logging, no observable effect" convention.
* `local_structure.py` -- `local_structure_model`, `univariate` (the
  StructureLearningPolicy), and `local_structure_probs_learning`.
* `optimize.py` -- the generic `optimize()` EDA loop.
* `initialization.py` -- `occam_randomize_contin` (both overloads),
  `uniform_randomize_bit`, `uniform_randomize_disc`, and `randomize` (the
  combining wrapper). `occam_randomize_term` is not ported: term fields do
  not exist anywhere in this port (see that module's own docstring).

CORRECTION (this entry previously said `initialization.{h,cc}` was
"genuinely unreachable from this task's scope, not merely deferred",
reasoning that `univariate.cc` does not include `initialization.h` and that
`git grep -rn 'occam_randomize\\|uniform_randomize' $CLASSIC/moses` finds no
external call site). That grep was accurate for those two names but never
checked `randomize(` itself, which DOES have an external call site:
`neighborhood_sampling.h:58`'s `generate_initial_sample` -- the very
function `univariate.cc` uses for its initial population, per this same
paragraph's own next sentence. The reachability claim was simply wrong;
what was true is narrower: that call site had already been served,
independently, by a private duplicate (`neighborhood_sampling.py`'s
`_randomize`/`_uniform_randomize_disc`/`_uniform_randomize_bit`), so no
NEW behaviour was missing at the time. See `initialization.py`'s own module
docstring for the full account and the follow-up this leaves open.
"""
