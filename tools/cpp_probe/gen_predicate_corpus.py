"""Generate a large corpus of predicate-bearing combo trees, one per line.

Used two ways: (a) fed to the C++ probe as extra `logical_reduce` rows,
(b) fed to a port-vs-port equivalence check for the surviving
`simplify_predicates` mutants.
"""

import random
import sys

random.seed(20260909)

CONSTS = ["0", "1", "-1", "2", "0.5", "-0.5", "3"]
ARGS = ["$1", "$2", "$3"]


def contin_expr(depth):
    if depth <= 0:
        return random.choice(ARGS + CONSTS)
    # `log` is deliberately ABSENT from this list. `logical_reduce` never
    # terminates on `0<(log(<negative constant>))` -- measured on classic
    # itself, not just on this port: `0<(log(-1))`, `0<(log(*(-1 1)))` and
    # `0<(+(log(-1) 1))` all run past 60s with zero output, while the same
    # `log(-1)` NOT under a predicate reduces in about a second. See
    # `simplify_predicates`' own docstring in
    # pymetta/comboreduct/reduct/logical_rules.py. A generator that can emit
    # `log` of a negative constant therefore cannot be used to build a
    # bounded oracle corpus; the two `log`-over-an-argument cases in the
    # curated block cover the operator instead.
    kind = random.choice(
        ["plus", "times", "div", "sin", "exp", "leaf", "leaf", "impulse"]
    )
    if kind == "leaf":
        return random.choice(ARGS + CONSTS)
    if kind in ("sin", "exp"):
        return f"{kind}({contin_expr(depth - 1)})"
    if kind == "impulse":
        return f"impulse({bool_expr(depth - 1)})"
    if kind == "div":
        return f"/({contin_expr(depth - 1)} {contin_expr(depth - 1)})"
    op = "+" if kind == "plus" else "*"
    n = random.choice([2, 2, 3])
    return f"{op}({' '.join(contin_expr(depth - 1) for _ in range(n))})"


def predicate(depth):
    body = f"0<({contin_expr(depth)})"
    return f"not({body})" if random.random() < 0.3 else body


def bool_expr(depth):
    if depth <= 0:
        return predicate(1)
    kind = random.choice(["and", "or", "pred", "pred", "arg"])
    if kind == "pred":
        return predicate(depth)
    if kind == "arg":
        return random.choice(["$1", "$2", "$3", "!$1", "!$2"])
    n = random.choice([2, 2, 3])
    return f"{kind}({' '.join(bool_expr(depth - 1) for _ in range(n))})"


def main(count):
    seen = set()
    out = []
    while len(out) < count:
        tree = bool_expr(random.choice([0, 1, 1, 2, 2, 3]))
        if "0<" not in tree or tree in seen:
            continue
        seen.add(tree)
        out.append(tree)
    print("\n".join(out))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
