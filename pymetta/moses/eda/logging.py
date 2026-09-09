"""C++: moses/moses/eda/logging.h

`cout_log_best_and_gen` -- the `LoggingPolicy` `optimize.py`'s generic EDA
loop calls once per generation. NOT ported as anything but a documented
no-op, matching `optimization.py`'s and `hill_climbing.py`'s own "pure
logging, no observable effect" convention (see each of those modules' own
docstrings for the same class of omission): `operator()`'s very first line
is `if (!logger().is_debug_enabled()) return;` (logging.h:42-43), and this
port has no logging surface at all -- `is_debug_enabled()` has no
equivalent that could ever be true here, so the guard would unconditionally
return before doing anything observable on every real call. Everything past
that guard (`std::max_element`, a `stringstream`, `logger().debug(...)`)
is therefore dead code from this port's point of view.
"""

from __future__ import annotations

from collections.abc import Sequence

from pymetta.moses.representation.field_set import field_set
from pymetta.moses.representation.scored_instance import scored_instance

__all__ = ["cout_log_best_and_gen"]


class cout_log_best_and_gen:
    """C++: moses/moses/eda/logging.h:37 cout_log_best_and_gen"""

    def __call__(
        self, items: Sequence[scored_instance], fields: field_set, gen: int
    ) -> None:
        """C++: moses/moses/eda/logging.h:39 operator()

        See module docstring: always a no-op in this port.
        """
        return
