"""C++: moses/comboreduct/crutil/exception.{h,cc}

Not in task-1-brief.md's own file list (`interpreter/eval.py` and
`combo/assumption.py`), but a direct, small dependency `eval.cc` needs
verbatim: `eval_throws_vertex`/`eval_throws_tree` throw `OverflowException`
and `EvalException`, and `eval_binding` specifically catches both by type
(eval.cc:290-299) to recover the offending vertex rather than letting the
exception propagate. Mirrored into its own module -- rather than folded
into `eval.py` -- to keep the file-for-file mirroring convention this
whole port (and the S7 plan's own "Architecture" section) uses throughout;
ported in full (all four declared exception types) for fidelity to the
header being mirrored, not just the two `eval.cc` happens to raise.

`ComboReductException`'s own C++ doc comment (exception.h:32-36) draws the
line this module does NOT cross: "Non-user errors, such as internal bugs,
use ComboException (defined in util/exceptions.h)" -- i.e. CogUtil's
`OC_ASSERT`/`ComboException` machinery, not this file's. Every `OC_ASSERT`
and bare `throw ComboException(TRACE_INFO, ...)` eval.cc itself contains is
therefore ported as a plain `AssertionError`/`ValueError` at its own call
site in eval.py, per this project's established OC_ASSERT convention
(argument.py's `__init__`) -- not as a class from this module.
"""

from pymetta.comboreduct.combo.vertex import vertex


class ComboReductException(Exception):
    """C++: moses/comboreduct/crutil/exception.h:37 ComboReductException

    Base class for all combo user-error exceptions (the C++ doc comment at
    exception.h:32-36). `_message`/`get_message()` are ported as a plain
    public `message` attribute plus a `get_message()` accessor, matching
    the C++ pair exactly rather than relying solely on `str(exc)`
    (Exception's own `__str__` already prints `message` via `super().
    __init__(message)`, but `get_message()` is kept for callers porting
    C++ call sites verbatim).
    """

    def __init__(self, message: str = "") -> None:
        """C++: moses/comboreduct/crutil/exception.h:43 ComboReductException"""
        super().__init__(message)
        self.message = message

    def get_message(self) -> str:
        """C++: moses/comboreduct/crutil/exception.h:45 get_message"""
        return self.message


class OverflowException(ComboReductException):
    """C++: moses/comboreduct/crutil/exception.h:49 OverflowException

    Overflow/divide-by-zero during evaluation -- eval.cc raises this for a
    `div`/`log`/`exp` result that is `nan`/`inf` (eval.cc:218-221, 230-233,
    240-241), carrying the offending vertex so `eval_binding` (eval.cc:
    290-299) can recover it instead of propagating the exception.
    """

    def __init__(self, v: vertex) -> None:
        """C++: moses/comboreduct/crutil/exception.h:54 OverflowException

        C++ constructs the message via the base's own default
        (exception.cc:29-30: `ComboReductException("Overflow Exception")`)
        -- ported verbatim rather than leaving it blank.
        """
        super().__init__("Overflow Exception")
        self.vertex = v

    def get_vertex(self) -> vertex:
        """C++: moses/comboreduct/crutil/exception.h:56 get_vertex"""
        return self.vertex


class EvalException(ComboReductException):
    """C++: moses/comboreduct/crutil/exception.h:60 EvalException

    Any other (non-overflow) evaluation exception -- eval.cc raises this
    for a vertex `eval_throws_vertex` does not recognise at all
    (eval.cc:280, "unrecognized expression").
    """

    def __init__(self, v: vertex, message: str = "") -> None:
        """C++: moses/comboreduct/crutil/exception.h:65 EvalException"""
        super().__init__(message)
        self.vertex = v

    def get_vertex(self) -> vertex:
        """C++: moses/comboreduct/crutil/exception.h:67 get_vertex"""
        return self.vertex


class TypeCheckException(ComboReductException):
    """C++: moses/comboreduct/crutil/exception.h:71 TypeCheckException

    Typing problem -- ported for completeness against the header (see
    module docstring); no call site in this port's scope raises or catches
    it.
    """

    def __init__(self, arg: int = 0) -> None:
        """C++: moses/comboreduct/crutil/exception.h:76 TypeCheckException"""
        super().__init__("Type check Exception")
        self.arg = arg
