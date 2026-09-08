"""C++: moses/moses/metapopulation/ensemble.{h,cc}

A MINIMAL `ensemble`: enough to be held as `metapopulation`'s `_ensemble`
member and constructed unconditionally the way `metapopulation.cc:41`
constructs it (`_ensemble(sc, pa.ensemble_params)`, regardless of whether
boosting is on), and enough to answer `get_ensemble()` -- the only method
`metapopulation.best_composite_score()` (`metapopulation.cc:207`) ever
calls on it, and only when `_params.do_boosting` (a metapop_parameters
field, NOT `ensemble_parameters.do_boosting` -- these are two independently
defaulted `False` flags, see `metapop_params.py`) is true:

    if (not _params.do_boosting)
      return _best_cscore;
    return _cscorer.get_cscore(_ensemble.get_ensemble());   // only reached
                                                              // when boosting

Full ensembles/boosting (`add_adaboost`, `add_expert`, `get_adaboost_tree`,
`get_exact_tree`, `get_expert_tree`, `flat_score`) are P2
(`docs/MODULE-COVERAGE.md`'s `ensemble.*`/`ensemble_params.h` row) and are
NOT ported here -- deliberately, per this task's brief: "port an ensemble
that can be held and constructed, and which raises (NotImplementedError)
if actually driven." Every method below that would require the unported
boosting machinery raises `NotImplementedError` instead of silently
no-opping or returning a plausible-looking empty answer, so a caller that
actually tries to drive boosting fails loudly rather than getting a
metapopulation that silently never boosts.

`get_ensemble()` itself does NOT raise: it is the one trivial accessor
`metapopulation` calls unconditionally once boosting is on, and it is
correct to return "no candidates yet" (an empty container) for an ensemble
that has never had one accepted into it -- which, given `add_candidates`
below always raises before mutating anything, is every ensemble this port
can ever construct.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymetta.moses.metapopulation.ensemble_params import ensemble_parameters

if TYPE_CHECKING:
    from pymetta.moses.moses.types import scored_combo_tree
    from pymetta.moses.scoring.behave_cscore import behave_cscore

__all__ = ["ensemble"]


class ensemble:
    """C++: moses/moses/metapopulation/ensemble.h:60 ensemble

    "The ensemble holds a collection of the fittest combo trees ... In
    particular (depending on parameters), the ensemble is designed to
    hold a boosted collection of combo trees" (ensemble.h:35-58's own doc
    comment, condensed) -- see the module docstring for what "designed to"
    means here: the boosting design itself is not ported, only the shape
    that lets it be held.
    """

    def __init__(self, cs: behave_cscore, ep: ensemble_parameters | None = None) -> None:
        """C++: moses/moses/metapopulation/ensemble.cc:39 ensemble::ensemble

        Member-initializer-list order preserved: `_params`/`_bscorer` are
        bound FIRST (`_bscorer(cs.get_bscorer())` runs even when boosting
        is off), exactly as C++ does, before the constructor BODY's
        `if (not ep.do_boosting) return;` guard (ensemble.cc:42-43,
        "Don't mess with the scorer weights if not doing boosting"). Only
        past that guard does real C++ initialize `_tolerance`/`_bias`/
        `_row_bias` and log -- all boosting setup, none of it ported (see
        module docstring); reaching that point here raises instead.

        `cs.get_bscorer()` (ensemble.cc:39) is spelled `cs._bscorer` here:
        `behave_cscore.get_bscorer()` was deliberately not given its own
        method in `behave_cscore.py` ("a thin one-line delegation ... not
        named in [its own] interface line ... call `.get_bscorer()` (i.e.
        the `behave_cscore` instance's own `_bscorer` attribute) directly
        instead", that module's own docstring) -- so this is that same
        module's own documented replacement for the call, not a new
        convention invented here.

        # DIVERGENCE(C++ moses/moses/metapopulation/ensemble.h:83):
        # `_scored_trees` is a `scored_combo_tree_set` in C++ (types.h:
        # 370-373: a `std::unordered_set` with its own HASH and EQUALITY
        # functors, `scored_combo_tree_hash`/`scored_combo_tree_equal` --
        # NOT an ordered set; `types.py:1068`'s own `scored_combo_tree_set`
        # class already describes it correctly, only this docstring
        # previously mischaracterized it). This port uses a plain, empty
        # `list` instead: it
        # is sufficient because `add_candidates` (the ONLY mutator of
        # `_scored_trees` in either language) always raises before this
        # port ever appends to it, so every `ensemble` this port can
        # construct has -- and permanently keeps -- an empty
        # `_scored_trees`, for which "ordered set" vs "plain list" has no
        # observable difference.
        """
        self._params: ensemble_parameters = ep if ep is not None else ensemble_parameters()
        self._bscorer = cs._bscorer
        self._scored_trees: list[scored_combo_tree] = []
        if not self._params.do_boosting:
            return
        raise NotImplementedError(
            "ensemble boosting setup (ensemble.cc:44-64) is not ported; "
            "construct with ensemble_parameters(do_boosting=False) "
            "(the default) -- boosting/ensembles are out of scope for this port"
        )

    def add_candidates(self, cands: list[scored_combo_tree]) -> None:
        """C++: moses/moses/metapopulation/ensemble.h:66 ensemble::add_candidates

        The method that actually "drives" an ensemble -- `add_adaboost`/
        `add_expert` (ensemble.cc, not ported, see module docstring). Only
        ever called from `merging.cc`'s boosting branch, itself gated on
        `_params.do_boosting` (a metapop_parameters field -- S10 Task 3,
        not yet ported). Unconditionally not implemented here: there is no
        correct partial behaviour for "add these candidates to a boosted
        ensemble" without the boosting algorithm itself.
        """
        raise NotImplementedError(
            "ensemble::add_candidates (ensemble.cc, add_adaboost/add_expert) "
            "is not ported; boosting/ensembles are out of scope for this port"
        )

    def get_ensemble(self) -> list[scored_combo_tree]:
        """C++: moses/moses/metapopulation/ensemble.h:68 ensemble::get_ensemble

        `const scored_combo_tree_set& get_ensemble() const { return
        _scored_trees; }` -- a trivial accessor, not a "driving" method
        (see module and class docstrings for why this one does not raise).
        """
        return self._scored_trees

    def get_weighted_tree(self) -> None:
        """C++: moses/moses/metapopulation/ensemble.h:72 ensemble::get_weighted_tree

        Dispatches to `get_adaboost_tree`/`get_exact_tree`/
        `get_expert_tree` in C++ (ensemble.cc) -- none ported (module
        docstring). Meaningless on an ensemble that can never hold a
        candidate (`add_candidates` always raises), so this raises too
        rather than fabricating a placeholder tree.
        """
        raise NotImplementedError(
            "ensemble::get_weighted_tree (ensemble.cc) is not ported; "
            "boosting/ensembles are out of scope for this port"
        )

    def flat_score(self) -> None:
        """C++: moses/moses/metapopulation/ensemble.h:74 ensemble::flat_score

        Not ported (module docstring) -- same reasoning as
        `get_weighted_tree` above.
        """
        raise NotImplementedError(
            "ensemble::flat_score (ensemble.cc) is not ported; "
            "boosting/ensembles are out of scope for this port"
        )
