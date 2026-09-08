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
(``deme/feature_selector.{h,cc}``, 710 lines) -- P2, gated behind
``--enable-fs``, exactly the scope boundary this task's own brief names.
Stored verbatim (``None`` by default, matching C++'s ``NULL``) but never
dereferenced by anything in this port: ``deme_expander.create_demeIDs`` and
``deme_expander.create_representations`` both raise ``NotImplementedError``
the instant ``fstor is not None``, mirroring classic's own
``if (_params.fstor)`` branch condition -- "port the gate faithfully, raise
inside the gated body", the same discipline the S10 plan applies to
boosting/domination/diversity in ``metapop_params.py``.
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
# ``max_candidates_per_deme`` is never read anywhere in ``deme_expander.cc``
# itself (confirmed: zero references in the whole 533-line file) -- if
# anything in this port's scope reads a ``deme_parameters.
# max_candidates_per_deme``, it is ``metapopulation``/``merging`` (a
# concurrent S10 task, out of this task's two owned directories), and
# whoever wires the two together must populate it first, exactly the
# obligation the CLI layer already carries in C++.
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
