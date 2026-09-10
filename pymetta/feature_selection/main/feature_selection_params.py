"""C++: moses/feature-selection/main/feature-selection.h

``feature_selection_parameters`` (feature-selection.h:44) -- parameters
of the ALGORITHM (``"simple"``/``"inc"``/``"random"``/``"smd"``/``"hc"``),
shared by the standalone ``feature-selection`` binary and the in-loop
``feature_selector`` (which embeds one as ``feature_selector_parameters.
fs_params`` -- ``pymetta.moses.deme.feature_selector``).

Every field mirrors the C++ struct's own name and default EXACTLY,
including the two independently observed ways C++'s default and the CLI's
default disagree for the SAME field -- neither is "the" default; both are
real and are cited separately below (S13's own perm_ratio incident,
CARRY-FORWARD.md Sec 5.4, is exactly this shape: a scope simplification
that quietly became a correctness bug because only one of several real
defaults was captured):

* ``target_size``: struct ctor default ``1`` (feature-selection.h:50,
  pinned here) vs. the STANDALONE binary's own CLI default ``0``
  (fs-main.cc:208, ``disables this option``) vs. the IN-LOOP CLI's ``20``
  (problem-params.cc:1590, wired through ``feature_selector_parameters.
  fs_params.target_size`` -- not ported here, belongs to
  ``problem_params.py``, another agent's file).
* ``hc_crossover``: struct ctor default ``true`` (feature-selection.h:67)
  vs. BOTH CLIs' own default ``false`` (fs-main.cc and
  problem-params.cc:1891) -- the asymmetry this task's own brief names
  explicitly; pinned here as ``True``, matching the struct.
* ``hc_crossover_pop_size``: struct ctor default ``300``
  (feature-selection.h:68) vs. fs-main.cc's own CLI default (not
  separately overridden there -- fs-main.cc has no ``--fs-hc-crossover-
  pop-size`` option at all; only problem-params.cc's in-loop CLI exposes
  one, also defaulting to ``300``, i.e. NOT one of this field's
  divergent cases -- named here only to disclose that the brief's own
  "pop size 300 vs 120" example refers to a DIFFERENT struct entirely,
  ``hc_parameters.pop_size`` in ``moses/moses/optimization/hill-climbing.h``,
  not this one; not re-verified further since ``hill-climbing.py`` is
  out of this task's scope).
* ``hc_crossover_min_neighbors``: struct ctor default ``1000``
  (feature-selection.h:69) vs. problem-params.cc's own CLI default
  ``400`` (the brief's own "min neighbors 1000 vs 400" example) -- pinned
  here as ``1000``, matching the struct.

None of the above CLI defaults are wired by this port (``problem_params.py``
is out of scope here); they are recorded so a future CLI-wiring change
does not have to re-derive them.

Two fields are declared in the C++ struct but OMITTED from its own
constructor's member-initializer list (feature-selection.h:47-68) and are
primitive (``bool``/``double``, not a class type with its own default
constructor) -- genuinely INDETERMINATE on a fresh
``feature_selection_parameters()`` in C++, the same "port cannot leave a
field uninitialized" situation ``deme_params.py`` documents for
``max_candidates_per_deme``/``perm_ratio``. Unlike those two, every real
CLI caller (fs-main.cc, problem-params.cc) DOES set these two via
``po::value<...>()->default_value(...)`` before ``notify()`` returns, so
this port pins each to that CLI default rather than leaving it unset --
DIVERGENCE, not a struct-constructor fact:

* ``pre_penalty``: uninitialized in the struct; CLI default ``1.0``
  (fs-main.cc / problem-params.cc:1844).
* ``pre_min_activation``: uninitialized; CLI default ``0.5``
  (problem-params.cc:1848).
* ``pre_max_activation``: uninitialized; CLI default ``1.0``
  (problem-params.cc:1853).
* ``pre_positive``: uninitialized; CLI default ``True``
  (problem-params.cc:1859).

None of the four are read by the ``"mi"`` scorer (this port's only ported
scorer, ``scorers/mutual_info.py``) -- only ``fs_scorer<FeatureSet>``'s
``pre_scorer`` branch (feature-selection.h:181-186, not ported) reads
them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pymetta.util.logger import logger

if TYPE_CHECKING:
    from pymetta.comboreduct.table.table import CTable

# C++: <climits> INT_MAX, used verbatim as feature_selection_parameters's
# own `max_time(INT_MAX)` default (a `time_t`, but initialized from the
# 32-bit int constant regardless of time_t's own width).
_INT_MAX = 2_147_483_647

# feature-selection.h:39-41 -- module-level scorer name constants.
MI_SCORER = "mi"
PRE_SCORER = "pre"

# feature-selection.h:117 -- `typedef std::set<arity_t> feature_set;`
feature_set = set[int]

# feature-selection.h:120 -- `typedef std::multimap<double, feature_set,
# std::greater<double>> feature_set_pop;`. A `list[tuple[score, fs]]`
# reproduces a multimap's two load-bearing properties for every caller in
# this port's scope: iteration in descending-score order, and INSERTION
# order preserved among equal-score entries (C++11 multimap guarantees
# this for equivalent keys) -- achieved by building the list via a stable
# sort (Python's `sorted`/`list.sort` are stable) rather than an
# unordered structure.
feature_set_pop = list[tuple[float, feature_set]]


@dataclass
class feature_selection_parameters:
    """C++: moses/feature-selection/main/feature-selection.h:44 feature_selection_parameters"""

    algorithm: str = "simple"
    scorer: str = MI_SCORER
    input_file: str = ""
    target_feature_str: str = ""
    timestamp_feature_str: str = ""
    ignore_features_str: list[str] = field(default_factory=list)
    force_features_str: list[str] = field(default_factory=list)
    initial_features: list[str] = field(default_factory=list)
    output_file: str = ""
    target_size: int = 1
    exp_distrib: bool = False
    threshold: float = 0.0
    jobs: int = 1

    subsampling_ratio: float = 1.0

    # incremental selection parameters
    inc_target_size_epsilon: float = 1.0e-10
    inc_red_intensity: float = -1.0
    inc_interaction_terms: int = 1

    # stochastic max-dependency selection parameters
    smd_top_size: int = 100

    # hill-climbing selection parameters
    hc_max_evals: int = 10000
    max_time: int = _INT_MAX
    hc_max_score: float = 1.0e50
    hc_cache_size: int = 1000
    hc_fraction_of_remaining: float = 1.0
    hc_crossover: bool = True
    hc_crossover_pop_size: int = 300
    hc_crossover_min_neighbors: int = 1000
    hc_widen_search: bool = True
    hc_fraction_of_nn: float = 2.0

    # MI scorer parameters
    mi_confi: float = 50.0

    # precision scorer parameters -- see module docstring: uninitialized
    # in C++'s own struct ctor, pinned here to the CLI's real default.
    pre_penalty: float = 1.0
    pre_min_activation: float = 0.5
    pre_max_activation: float = 1.0
    pre_positive: bool = True


def fs_scorer(ctable: CTable, fs_params: feature_selection_parameters):
    """C++: moses/feature-selection/main/feature-selection.h:160 fs_scorer

    The C++ symbol is the class template `fs_scorer<FeatureSet>`
    (declared `template<typename FeatureSet> struct fs_scorer` at
    feature-selection.h:159-160). The breadcrumb names the bare class, per
    docs/CARRY-FORWARD.md 4.5's rule for symbols `check_alignment.py`
    cannot parse -- carrying the `<FeatureSet>` in the breadcrumb left it
    unmatchable and kept `tools/ci.sh` red.

    "A wrapper, simply so that optimizer gets the iscorer_base base
    class" -- no, that is ``iscorer_cache`` (feature-selection.h:143-156,
    not ported: zero callers once ``algo/deme_optimize.py`` uses a plain
    ``dict``-based memoization instead, see that module's own docstring).
    THIS wrapper (feature-selection.h:159-200) is the ``mi``/``pre``
    scorer union C++ builds with two nullable pointers and a runtime
    ``OC_ASSERT`` for anything else -- collapsed here into a single
    factory function returning the concrete scorer object directly
    (Python has no need for the pointer-pair-plus-destructor dance
    C++'s own lack of a tagged-union type forces).

    Placed in THIS module (feature-selection.h's own params file), not
    ``main/feature_selection.py`` (feature-selection.cc's own dispatch
    file), because ``algo/stochastic_max_dependency.py`` and
    ``algo/deme_optimize.py`` both need it too (stochastic_max_
    dependency.h:40, deme_optimize.h:124 both instantiate ``fs_scorer
    <std::set<arity_t>>`` directly) -- exactly mirroring feature-
    selection.h being a header ``algo/*.h`` files `#include`, while
    feature-selection.cc (this port's ``main/feature_selection.py``) is
    the one file that includes the algo headers, not the reverse.
    """
    # Local imports: mutual_info/moses_matrix both import CTable and (via
    # fs_scorer_base) sit one layer below this module in the dependency
    # graph; importing at call time (rather than module level) avoids
    # nothing here structurally but keeps this module's own import list
    # minimal for readers who never touch scorer="pre".
    from pymetta.feature_selection.scorers.moses_matrix import pre_scorer
    from pymetta.feature_selection.scorers.mutual_info import MICScorerCTable

    if fs_params.scorer == MI_SCORER:
        return MICScorerCTable(ctable, fs_params.mi_confi)
    if fs_params.scorer == PRE_SCORER:
        return pre_scorer(
            ctable,
            fs_params.mi_confi,
            fs_params.pre_penalty,
            fs_params.pre_min_activation,
            fs_params.pre_max_activation,
            fs_params.pre_positive,
        )
    raise NotImplementedError(
        f"fs_scorer: unknown feature selection scorer {fs_params.scorer!r} "
        "(feature-selection.h:178's own OC_ASSERT(false, ...))."
    )


def initial_features(
    ilabels: Sequence[str], fs_params: feature_selection_parameters
) -> feature_set:
    """C++: moses/feature-selection/main/feature-selection.cc:108 initial_features

    Converts ``fs_params.initial_features`` (label strings) into a
    ``feature_set`` of column indices, silently ignoring (with a would-be
    ``logger().warn`` -- no logging surface, see ``feature_selection.py``'s
    module docstring for the convention) any name not present in
    ``ilabels``.

    Placed in THIS module, not ``main/feature_selection.py`` (feature-
    selection.cc's own dispatch/orchestration file), for the same reason
    ``fs_scorer`` above is: ``algo/stochastic_max_dependency.py``
    (stochastic_max_dependency.cc:48) and ``algo/deme_optimize.py``
    (deme_optimize.cc:35, via ``initial_instance``) both call it directly,
    and ``main/feature_selection.py`` itself calls INTO those algo
    modules for ``select_feature_sets``'s own dispatch -- defining
    ``initial_features`` there would make the two modules import each
    other. C++ has no such problem: ``initial_features`` is DECLARED in
    feature-selection.h (:135, this module's own header) and DEFINED in
    feature-selection.cc, and every algo ``.cc`` file links against that
    one definition through the header declaration alone -- Python's
    import-time module graph has no equivalent of a header/impl split, so
    the definition moves to the header-shaped module instead.
    ``main/feature_selection.py`` re-exports this name unchanged so
    existing callers of ``pymetta.feature_selection.main.feature_selection.
    initial_features`` are unaffected.
    """
    # C++: feature-selection.cc:111 `vector<string> vif;` -- "valid initial
    # features, used for logging" (its own comment).
    vif: list[str] = []
    res: feature_set = feature_set()
    for name in fs_params.initial_features:
        try:
            idx = ilabels.index(name)
        except ValueError:
            # C++: feature-selection.cc:122
            logger().warn(
                "No such a feature %s in file %s. It will be ignored as initial feature.",
                name,
                fs_params.input_file,
            )
            continue
        res.add(idx)
        vif.append(name)
    # C++: feature-selection.cc:125-132. `ostream_container(ss, vif, ",")`
    # uses a "," delimiter and no left/right brackets.
    if not vif:
        logger().info("The search will start with the empty feature set")
    else:
        logger().info("The search will start with the following feature set: " + ",".join(vif))
    return res
