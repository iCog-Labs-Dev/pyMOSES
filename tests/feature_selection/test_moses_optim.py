"""Guard tests for pymoses.feature_selection.scorers.moses_optim
(get_feature_set, deme_based_scorer).

No dedicated C++ oracle: both functions are pure, small bridging code
between a `field_set`/`instance` pair and a plain `set[int]` feature set --
covered indirectly, end to end, by `test_deme_optimize.py`'s own oracle-
adjacent tests (deme_optimize is this module's only real caller in
classic too -- `git grep get_feature_set` over `moses/` finds exactly one
call site, `deme_optimize.h:58`, plus its own declaration/definition).
Verified here directly against `field_set`/`instance`, independent of the
rest of the "hc" algorithm.
"""

import pytest

from pymoses.feature_selection.scorers.moses_optim import deme_based_scorer, get_feature_set
from pymoses.moses.moses.types import composite_score
from pymoses.moses.representation.field_set import contin_spec, field_set
from pymoses.moses.representation.instance import instance


@pytest.fixture()
def bool_fields() -> field_set:
    return field_set(field_set.disc_spec(2), 4)


def test_get_feature_set_all_zero_instance_is_empty(bool_fields):
    inst = instance(bool_fields.packed_width())
    assert get_feature_set(bool_fields, inst) == set()


def test_get_feature_set_reads_raw_bits_at_the_right_feature_index(bool_fields):
    inst = instance(bool_fields.packed_width())
    bool_fields.set_raw(inst, bool_fields.begin_bit_raw_idx() + 1, 1)
    bool_fields.set_raw(inst, bool_fields.begin_bit_raw_idx() + 3, 1)
    assert get_feature_set(bool_fields, inst) == {1, 3}


def test_get_feature_set_all_ones_instance_is_every_index(bool_fields):
    inst = instance(bool_fields.packed_width())
    for i in range(bool_fields.end_bit_raw_idx() - bool_fields.begin_bit_raw_idx()):
        bool_fields.set_raw(inst, bool_fields.begin_bit_raw_idx() + i, 1)
    assert get_feature_set(bool_fields, inst) == {0, 1, 2, 3}


def test_get_feature_set_offsets_by_begin_bit_raw_idx_when_nonzero():
    """`algo/deme_optimize.py`'s own field_set (`field_set(disc_spec(2),
    arity)`) always has `begin_bit_raw_idx() == 0` -- every OTHER test in
    this module is blind to a wrong/missing offset add for exactly that
    reason (CARRY-FORWARD.md Sec 5.4's own warning: "an oracle configured
    by the port cannot detect the port's configuration being wrong").
    This test builds a field_set with a CONTIN field ahead of the bit
    region (raw indices 0-1) so `begin_bit_raw_idx() == 2`, and proves
    `get_feature_set` maps raw index `2+i` back to feature index `i`, NOT
    raw index `i` itself.
    """
    specs = [contin_spec(0.0, 1.0, 2.0, 2), field_set.disc_spec(2), field_set.disc_spec(2)]
    fields = field_set(specs)
    assert fields.begin_bit_raw_idx() == 2
    assert fields.end_bit_raw_idx() == 4

    inst = instance(fields.packed_width())
    fields.set_raw(inst, 3, 1)  # raw index 3 == bit-region feature index 1

    assert get_feature_set(fields, inst) == {1}


def test_deme_based_scorer_wraps_a_plain_feature_set_scorer(bool_fields):
    calls = []

    def fs_scorer(fs):
        calls.append(frozenset(fs))
        return 3.5

    scorer = deme_based_scorer(fs_scorer, bool_fields)
    inst = instance(bool_fields.packed_width())
    bool_fields.set_raw(inst, bool_fields.begin_bit_raw_idx(), 1)

    result = scorer(inst)
    assert isinstance(result, composite_score)
    assert result.get_score() == 3.5
    assert result.complexity == 1  # fs.size()
    assert calls == [frozenset({0})]


def test_deme_based_scorer_complexity_penalty_is_zero(bool_fields):
    """C++: `composite_score csc(_fs_scorer(fs), fs.size(), 0);` -- third
    positional argument (complexity_penalty) is the literal `0`, not
    something derived from the feature set.
    """
    scorer = deme_based_scorer(lambda fs: 0.0, bool_fields)
    inst = instance(bool_fields.packed_width())
    result = scorer(inst)
    assert result.complexity_penalty == 0.0
