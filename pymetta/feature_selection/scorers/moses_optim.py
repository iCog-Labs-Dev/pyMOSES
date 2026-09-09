"""C++: moses/feature-selection/scorers/moses_optim.{h,cc}

Bridges a plain feature-set scorer to MOSES's own optimization algorithms
(``pymetta.moses.moses.optimization`` -- read-only import, out of this
package's edit scope): a deme is a binary string, one bit per feature,
and ``deme_based_scorer`` wraps a feature-set scorer so an
``optimizer_base`` (``hill_climbing``, in this port's scope) can search it
directly. Used by ``algo/deme_optimize.py`` (the ``"hc"`` algorithm).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymetta.moses.moses.types import composite_score
from pymetta.moses.representation.instance_scorer import iscorer_base

if TYPE_CHECKING:
    from collections.abc import Callable

    from pymetta.moses.representation.field_set import field_set
    from pymetta.moses.representation.instance import instance


def get_feature_set(fields: field_set, inst: instance) -> set[int]:
    """C++: moses/feature-selection/scorers/moses_optim.cc:27 get_feature_set

    "Translate an instance into a feature set. Each feature is represented
    by its index (the left most one is 0)" (moses_optim.h:40-41's own doc
    comment). C++ walks ``field_set::const_bit_iterator`` from
    ``begin_bit(inst)`` to ``end_bit(inst)`` -- the raw-index RANGE
    ``[begin_bit_raw_idx(), end_bit_raw_idx())`` is the exact same walk,
    since ``begin_bit``/``end_bit`` are themselves defined in terms of
    that same raw-index range (field_set.h:1188-1213) -- with a
    feature-index counter ``i`` starting at 0 for the FIRST bit field,
    not the raw index itself (the two coincide only when no contin/multi-
    bit-disc fields precede the bit region, which is always true for the
    ``field_set(disc_spec(2), arity)`` this module's only caller builds --
    see ``algo/deme_optimize.py``).
    """
    fs: set[int] = set()
    start = fields.begin_bit_raw_idx()
    end = fields.end_bit_raw_idx()
    for i in range(end - start):
        if fields.get_raw(inst, start + i):
            fs.add(i)
    return fs


class deme_based_scorer(iscorer_base):
    """C++: moses/feature-selection/scorers/moses_optim.h:52 deme_based_scorer

    "Wrapper to use a feature set scorer with MOSES's optimization
    algorithms operating on a deme. Each deme is a binary string where
    each bit represents whether a feature is selected or not" (the C++
    class's own doc comment).
    """

    def __init__(self, fs_scorer: Callable[[set[int]], float], fields: field_set) -> None:
        """C++: moses/feature-selection/scorers/moses_optim.h:54 deme_based_scorer"""
        self._fs_scorer = fs_scorer
        self._fields = fields

    def __call__(self, inst: instance) -> composite_score:
        """C++: moses/feature-selection/scorers/moses_optim.h:62 operator()

        "The feature set is represented by an instance encoding a field of
        booleans. Each boolean represents whether its corresponding
        feature is in the feature set of not" (the C++ method's own doc
        comment, typo included -- not reproduced in prose here, only in
        the citation). ``composite_score(_fs_scorer(fs), fs.size(), 0)``:
        third positional argument is ``complexity_penalty=0``.
        """
        fs = get_feature_set(self._fields, inst)
        return composite_score(self._fs_scorer(fs), len(fs), 0)
