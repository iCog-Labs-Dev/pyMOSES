"""C++: moses/moses/deme/deme_params.h

``deme_parameters`` -- "parameters for deme management" (the header's own
comment). ``empty_ignore_ops`` (deme_params.h:32, a module-level
``static const operator_set``) is exactly what this port's ``ignore_ops:
Collection[vertex] = ()`` default already gives, so it is not ported as a
separate name.

``perceptions``/``actions`` (deme_params.h:41-42,46-47,63-65) are the
Petbrain/embodiment interactive-agent hooks the project ``CLAUDE.md``'s
"Known sharp edges" names as legacy (``action_result_type``/
``ant_combo_vocabulary``). They are stored verbatim, as opaque values
(matching the raw ``const combo_tree_ns_set*`` pointer C++ carries, default
``None`` for C++'s ``NULL``), but ``deme_expander.create_representations``
never forwards them to ``representation()`` -- see that method's own
DIVERGENCE note for why (S8's ``representation`` is Boolean-only and has no
parameter for either).

``fstor`` (deme_params.h:67) is the dynamic feature-selection hook
(``deme/feature_selector.{h,cc}``), ``--enable-fs``. Stored verbatim
(``None`` by default, matching C++'s ``NULL``, a pointer in C++ --
``const feature_selector* fstor`` -- represented here as an optional
``pymoses.moses.deme.feature_selector.feature_selector`` instance rather
than a pointer type, since Python has no pointer/reference distinction to
preserve). ``deme_expander.create_demeIDs`` and
``deme_expander.create_representations`` both honour a real ``fstor`` now
(the P2 feature-selection port) -- see ``deme_expander.py``'s own module
docstring for exactly what is and is not ported behind it (in short:
feature-subset selection and representation-building are fully ported;
``prune_xmplr`` and driving a real ``fstor`` through
``deme_expander.optimize_demes``'s own search loop are both still gated,
narrower than before this port existed).
"""

# DIVERGENCE(C++ moses/moses/deme/deme_params.h:54,82): ``max_candidates_per_
# deme`` and ``perm_ratio`` are declared as plain ``int``/``float`` members
# with NO default-member-initializer, and the constructor's own initializer
# list (deme_params.h:44-50) never touches either -- so a freshly-constructed
# C++ ``deme_parameters`` genuinely leaves both fields holding INDETERMINATE
# (uninitialized stack) values. This is the same shape ``metapop_params.py``
# already documents twice (``subsample_deme_filter_parameters``'s ten fields,
# ``diversity_parameters.autoscale``): Python has no equivalent of an
# uninitialized primitive, so rather than invent a plausible default (which
# would silently mask "this was never set" the way a stray ``0.0``/``-1``
# could) this port simply never sets these two attributes -- reading one
# before an explicit assignment raises ``AttributeError`` instead of
# returning C++'s indeterminate garbage.
#
# UNLIKE those two precedents, though, ``perm_ratio`` is NOT behind a P2
# gate: ``deme_expander.create_representations`` reads ``_params.perm_ratio``
# UNCONDITIONALLY on every deme built (deme_expander.cc:294, passed straight
# into every ``representation(...)`` call) -- reachable on the DEFAULT path,
# not merely a dormant field guarded by some other flag. This is a caller
# obligation in C++ too, not a gap this port introduces: every real classic
# caller sets it explicitly before constructing a ``deme_expander`` --
# ``problem-params.cc:2265``, ``deme_params.perm_ratio = perm_ratio;``
# (CLI default value ``0.0``), unconditionally, on every run. Whoever drives
# this port's ``deme_expander`` (a test, or the S13 CLI wiring once it
# exists) MUST set ``.perm_ratio`` before calling ``create_demes``/
# ``create_representations`` -- and per ``build_knobs.py``'s own
# ``_validate_request``, S8's ``representation``/``build_knobs`` accept
# ONLY exactly ``-1.0`` (the classic early-return "no pair sampling" mode),
# so ``-1.0`` is also the only value this port's ``deme_expander`` can
# actually drive to completion without raising ``NotImplementedError``
# itself (build_knobs.py's own gate, not this file's).
#
# CORRECTED (S10 branch review, Minor): this comment used to claim that
# whoever reads a ``deme_parameters.max_candidates_per_deme`` would be
# ``metapopulation``/``merging`` -- MEASURED FALSE. ``deme_params.h:54``'s
# field is never read ANYWHERE in classic (confirmed: zero references
# beyond its own declaration and the equally-dead ``deme_expander.cc``,
# which also never reads it -- zero references in that 533-line file
# either). ``merging.cc:536``'s ``_params.max_candidates_per_deme`` reads
# a DIFFERENT field on a DIFFERENT struct: ``metapopulation``'s own
# ``_params``, of type ``metapop_parameters`` (declared separately, at
# ``metapop_params.h:183``, and ported here as ``metapop_params.py``'s
# own ``metapop_parameters.max_candidates_per_deme``) -- the two fields
# merely share a name and a constructor default (``-1``) by coincidence
# of both authors solving the same "cap candidates per deme" problem
# independently; they are not the same storage, and nothing in classic
# ever copies one into the other. ``deme_params.h:54`` is therefore
# genuinely dead in classic, not merely dead in this port's own scope.
#
# Observed by inspection, NOT measured (no probe was run to confirm this is
# live UB rather than a technically-legal-but-surprising lifetime): both
# ``_params``/``_filter_params`` on ``deme_expander`` (deme_expander.h:150,
# 152) are ``const&`` members bound, via the constructor's initializer list,
# from constructor PARAMETERS (``pa``, ``fp``) that default to temporaries --
# ``deme_parameters()``, ``subsample_deme_filter_parameters()``
# (deme_expander.h:45-46) -- built at the call site. A reference data member
# initialized from a reference parameter does not itself extend a
# temporary's lifetime past the end of the constructor call's full
# expression, so a caller that constructs a ``deme_expander`` WITHOUT
# supplying either argument binds ``_params``/``_filter_params`` to
# already-destroyed temporaries for the rest of the object's life. Whether
# this is actually reachable is not established here: the only real
# construction site found in classic, ``moses_main.h:318`` (``deme_expander
# dex(tt, si_ca, si_kb, sc, *optimizer, deme_params, filter_params);``),
# supplies both arguments explicitly, every time -- so if this is live UB,
# it is dormant in every production code path this search covered. Python
# objects carry no such reference-lifetime hazard: this port's own
# ``deme_expander.__init__`` default (a freshly-constructed
# ``deme_parameters()``/``subsample_deme_filter_parameters()`` per call,
# ordinary garbage-collected objects) is actually SAFE where C++'s
# equivalent default argument may not be -- kept only for constructor-
# signature parity with C++, not because any caller here (or, per the
# above, in classic) is ever observed relying on it.
class deme_parameters:
    """C++: moses/moses/deme/deme_params.h:37 deme_parameters"""

    def __init__(
        self,
        reduce_all: bool = True,
        ignore_ops: object = (),
        perceptions: object = None,
        actions: object = None,
        fstor: object = None,
    ) -> None:
        """C++: moses/moses/deme/deme_params.h:39 deme_parameters

        Only the five real constructor parameters (deme_params.h:39-43) are
        accepted here; ``linear_contin`` (deme_params.h:49, initializer-list
        default ``true``) is set as a plain assignment below, matching every
        other C++ member-initializer-only field this port has already
        treated the same way (e.g. ``hc_parameters``'s non-constructor
        fields, hill_climbing.py). See the module docstring for why
        ``max_candidates_per_deme``/``perm_ratio`` are deliberately left
        unset rather than defaulted.
        """
        self.reduce_all = reduce_all
        self.ignore_ops = tuple(ignore_ops)
        self.perceptions = perceptions
        self.actions = actions
        self.fstor = fstor
        self.linear_contin = True
        # No C++ counterpart: `--knob-probe-baseline` (see
        # `build_knobs._disc_probe_reduced_baseline`). "classic" is classic
        # MOSES's knob probing.
        # DIVERGENCE(C++ moses/moses/representation/build_knobs.cc:311-341):
        # the default is "reduced" (backlog item 15), as on the command line.
        self.knob_probe_baseline = "reduced"
        # NOT A PORT (pyMoses extension, no C++ counterpart): the
        # `--simplify-exemplar-numeric` pass
        # (`numeric_simplify.exemplar_numeric_simplifier`), set by the
        # regression problems when the option is on; `None` = off.
        self.numeric_simplifier = None
