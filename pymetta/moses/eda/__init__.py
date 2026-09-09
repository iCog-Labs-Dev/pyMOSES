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

NOT ported: `initialization.{h,cc}` (`occam_randomize_contin`/
`occam_randomize_term`/`uniform_randomize_bit`/`uniform_randomize_disc`/
`randomize`). `univariate.cc` does not include `initialization.h` at all --
it draws its initial population via `../moses/neighborhood_sampling.h`'s
`generate_initial_sample` instead (already ported,
`pymetta/moses/moses/neighborhood_sampling.py`) -- and
`git grep -rn 'occam_randomize\\|uniform_randomize' $CLASSIC/moses` confirms
every call site of `initialization.h`'s own functions lives inside
`initialization.cc` itself; nothing else in `moses/` calls them. Genuinely
unreachable from this task's scope, not merely deferred.
"""
