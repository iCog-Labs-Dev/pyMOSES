"""Shared test helpers for pymoses.gomea tests."""

from collections.abc import Callable

import pytest

from pymoses.comboreduct.combo.iostream_combo import str_to_combo_tree
from pymoses.comboreduct.reduct.logical_reduction import logical_reduction
from pymoses.comboreduct.type_checker.type_signature import gen_signature, type_node
from pymoses.moses.representation.representation import representation


@pytest.fixture
def make_boolean_rep() -> Callable[[str, int], representation]:
    """Factory fixture building a Boolean ``representation`` the same way
    the ``pa``/``maj`` demo problems do: the medium-effort logical
    simplifier (``logical_reduction(frozenset())(2)``, classic's own
    ``--reduct-candidate-effort``/``--reduct-knob-building-effort``
    default of 2 -- ``problem_params.py``) for both candidate reduction
    and knob-building reduction, and a Boolean type signature of the given
    arity. Mirrors ``_boolean_rep`` in
    ``tests/moses/representation/test_instance_scorer.py`` and
    ``_build_dex``'s own construction in
    ``tests/moses/deme/test_deme_expander.py`` -- ``perm_ratio=-1.0`` (no
    random logical pairs) keeps the build deterministic without seeding
    the process-wide RNG, matching both of those.
    """

    def _make(exemplar: str, arity: int) -> representation:
        rule = logical_reduction(frozenset())(2)
        return representation(
            rule,
            rule,
            str_to_combo_tree(exemplar),
            gen_signature(type_node.boolean_type, arity),
            perm_ratio=-1.0,
        )

    return _make
