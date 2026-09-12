"""C++: moses/moses/metapopulation/ensemble_params.h

`ensemble_parameters` -- the small POD-style parameter struct that both
`metapop_parameters` (`metapop_params.py`) and `ensemble` (`ensemble.py`)
hold a copy of / reference to. Ported verbatim, all six fields and their
defaults: `do_boosting`/`experts` default `false`, `exact_experts` defaults
`true`, `expalpha`/`bias_scale` default `2.0`/`1.0`, `num_to_promote`
defaults `1`.

None of these six fields are `score_t`-typed in C++ (`expalpha`/
`bias_scale` are plain `double`, matching `metapop_parameters.cap_coef`'s
own un-narrowed treatment in `metapop_params.py` -- see that module's
docstring for the general score_t-vs-double narrowing rule this port
follows), so nothing here goes through `_to_score_t`.
"""

from __future__ import annotations

__all__ = ["ensemble_parameters"]


class ensemble_parameters:
    """C++: moses/moses/metapopulation/ensemble_params.h:30 ensemble_parameters"""

    def __init__(self) -> None:
        """C++: moses/moses/metapopulation/ensemble_params.h:32 ensemble_parameters"""
        self.do_boosting = False  # boosting is enabled ...
        self.experts = False  # do "ensemble-of-experts" boosting.
        self.exact_experts = True  # experts must be perfectly correct.
        self.expalpha = 2.0  # Ad-hoc exact-expert boosting coefficient.
        self.bias_scale = 1.0  # Ad-hoc inexact-expert bias scale.
        self.num_to_promote = 1  # max demes accepted into the ensemble per iteration.
