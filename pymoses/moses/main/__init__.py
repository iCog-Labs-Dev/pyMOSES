"""The CLI layer: `moses/moses/main/` -- option parsing, the problem
registry, and the table-based problems, assembled by `moses_exec`.

``problem.py`` ports ``problem.{h,cc}`` (S13 Task 0): ``problem_base``,
the interface every ``-H <name>`` problem implements, and
``problem_manager``, the name -> ``problem_base`` registry
``moses_exec.cc`` builds once and queries once.
"""

from pymoses.moses.main.problem import problem_base, problem_manager

__all__ = ["problem_base", "problem_manager"]
