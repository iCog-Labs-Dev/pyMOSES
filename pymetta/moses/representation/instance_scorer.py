"""C++: moses/moses/representation/instance_scorer.h:36 iscorer_base

`iscorer_base` (an abstract functor over `instance -> composite_score`),
`distance_based_scorer` ("[m]ostly for testing the optimization algos ...
[r]eturns minus the hamming distance of the candidate to a given target
instance and constant null complexity", the header's own doc comment), and
`complexity_based_scorer` (wraps a `representation` + `behave_cscore` pair:
rebuild a combo tree from the instance, then score that tree).

NOT ported: `iscorer_base::argument_type`/`result_type` (instance_scorer.h:
37-38) -- legacy STL `unary_function`-style functor-adapter typedefs with no
behavioural role and no Python analogue (nothing in this port adapts a
functor through them).
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod

from pymetta.comboreduct.combo.vertex import combo_tree
from pymetta.moses.moses.types import _to_score_t, composite_score, worst_composite_score
from pymetta.moses.representation.field_set import field_set
from pymetta.moses.representation.instance import instance
from pymetta.moses.representation.representation import representation
from pymetta.moses.scoring.behave_cscore import behave_cscore


class iscorer_base(ABC):
    """C++: moses/moses/representation/instance_scorer.h:36 iscorer_base"""

    @abstractmethod
    def __call__(self, inst: instance) -> composite_score:
        """C++: moses/moses/representation/instance_scorer.h:39 iscorer_base

        Non-identifier member per the project's convention (see
        `scored_instance.py`'s `__eq__` docstring for the same pattern):
        the real member is the pure-virtual `operator()`
        (instance_scorer.h:39), a callable functor in both languages.
        """
        raise NotImplementedError

    def get_rep(self) -> representation | None:
        """C++: moses/moses/representation/instance_scorer.h:40 iscorer_base::get_rep

        Returns `None` (`nullptr`) by default; `complexity_based_scorer`
        overrides it below.
        """
        return None


class distance_based_scorer(iscorer_base):
    """C++: moses/moses/representation/instance_scorer.h:49 distance_based_scorer"""

    def __init__(self, fs: field_set, target_inst: instance) -> None:
        """C++: moses/moses/representation/instance_scorer.h:50 distance_based_scorer"""
        if not isinstance(fs, field_set):
            raise TypeError("distance_based_scorer requires a field_set")
        if not isinstance(target_inst, instance):
            raise TypeError("distance_based_scorer requires an instance target")
        self.fs = fs
        self.target_inst = target_inst

    def __call__(self, inst: instance) -> composite_score:
        """C++: moses/moses/representation/instance_scorer.h:53 distance_based_scorer::operator()

        `-fs.hamming_distance(target_inst, inst)` (instance_scorer.h:54):
        `hamming_distance` returns a signed C++ `int`, so this is a plain
        signed negation, not an unsigned wraparound. `score_t sc = ...`
        narrows the negated distance to `score_t` at the point of
        assignment -- a narrowing site in the same textual shape as the
        fourteen `types.py`'s module docstring traces, but UNLIKE those,
        not one with any demonstrated (or demonstrable) observable effect:
        `score_t` (float32) represents every integer up to 2**24 exactly,
        so this narrowing would only ever lose precision for a Hamming
        distance of at least 2**24 -- i.e. a `field_set` with over 16
        million raw fields, wildly outside anything this port's fidelity
        suite (or classic's own test suite) ever constructs. Ported for
        the same reason the other thirteen sites are -- matching C++'s own
        `score_t` assignment, not because a difference has been found or
        is expected here.
        """
        # DIVERGENCE(C++ moses/moses/representation/instance_scorer.h:55-60):
        # classic logs the evaluated instance and score at FINE level
        # through `logger()`. This port has no logging surface (see
        # `scoring_base.py`'s module docstring for the same omission) --
        # dropped, not a behavioural gap.
        sc = _to_score_t(-self.fs.hamming_distance(self.target_inst, inst))
        return composite_score(sc, 0, 0, 0)


class complexity_based_scorer(iscorer_base):
    """C++: moses/moses/representation/instance_scorer.h:70 complexity_based_scorer"""

    def __init__(self, cs: behave_cscore, rep: representation, reduce: bool) -> None:
        """C++: moses/moses/representation/instance_scorer.h:71 complexity_based_scorer

        `cs`/`rep` are not isinstance-checked, matching `behave_cscore.
        __init__`'s own treatment of its `bscorer` collaborator
        (behave_cscore.py) rather than `distance_based_scorer`'s treatment
        of its plain-data `fs`/`target_inst` arguments above: both are
        strategy-style collaborators here (only `get_cscore`/
        `get_candidate` are ever called on them), not data this class
        inspects, so duck typing is the established convention.
        """
        self._cscorer = cs
        self._rep = rep
        self._reduce = reduce

    def get_rep(self) -> representation:
        """C++: moses/moses/representation/instance_scorer.h:74 complexity_based_scorer::get_rep"""
        return self._rep

    def __call__(self, inst: instance) -> composite_score:
        """C++: moses/moses/representation/instance_scorer.h:75 complexity_based_scorer::operator()

        `combo_tree tr = _rep.get_candidate(inst, _reduce); return _cscorer.
        get_cscore(tr);` (instance_scorer.h:82-83).
        """
        # DIVERGENCE(C++ moses/moses/representation/instance_scorer.h:84-95):
        # classic's `catch (...)` -- a true catch-all, matching literally
        # any thrown type -- re-derives an UNREDUCED and a REDUCED tree a
        # second time (calling `get_candidate` twice more) purely to log
        # them ("The following instance could not be evaluated: ...",
        # logger().warn()) before falling through to `return
        # worst_composite_score`. Those two extra calls have no effect on
        # the return value or any state; this port keeps the one
        # observable behaviour -- swallow any exception from evaluating
        # `inst` and return `worst_composite_score` -- and drops the
        # logging-only recomputation, the same scoping call
        # `behave_cscore.get_cscore_nocache`'s own DIVERGENCE note makes
        # for a structurally identical try/except. `except Exception`
        # (not bare `except:`) is the closest Python analogue to `catch
        # (...)`, deliberately still letting `KeyboardInterrupt`/
        # `SystemExit`/`GeneratorExit` (which do not subclass `Exception`)
        # propagate rather than treating a process-level control signal as
        # "this instance could not be evaluated".
        try:
            tr: combo_tree = self._rep.get_candidate(inst, self._reduce)
            return self._cscorer.get_cscore(tr)
        except Exception:
            # DIVERGENCE(C++ moses/moses/representation/instance_scorer.h:
            # 96, and moses/moses/moses/types.cc:112): `return
            # worst_composite_score;` returns `composite_score` BY VALUE
            # from a `const` global (`virtual composite_score
            # operator()(const instance&) const`) -- every caller gets its
            # own independent copy. `worst_composite_score` here is a
            # plain Python module-level object (types.py:535); returning
            # it bare would hand back the SAME shared singleton to every
            # caller. That was inert while every caller only rebound the
            # result, but `composite_score.set_score`/
            # `set_uniformity_penalty` mutate a `composite_score` in
            # place, and S10's metapopulation does exactly that -- so a
            # bare return would let one caller's mutation corrupt every
            # other caller's "worst score" AND the singleton itself,
            # silently breaking `composite_score() ==
            # worst_composite_score` everywhere else in the process.
            # `copy.copy` is a full value copy here: `composite_score`
            # holds only plain float/int fields (the same reasoning
            # `scored_instance.py`'s module docstring gives for copying a
            # `composite_score` via `copy.copy`).
            return copy.copy(worst_composite_score)
