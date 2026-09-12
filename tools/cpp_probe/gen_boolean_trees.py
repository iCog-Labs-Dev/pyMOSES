"""Random boolean combo-tree generator for logical_rules.py's differential
oracle (Task 4 review, Important: "land the differential harness in-repo").

Generates a deterministic (seeded), reproducible corpus of `and`/`or`/`not`/
`$n`/`!$n` combo-tree text-format strings -- the same grammar
`pymoses.comboreduct.combo.iostream_combo.str_to_combo_tree` and $CLASSIC's
`str2combo_tree` both parse -- one per line, written to a file both
`run_differential.py` (the Python side) and
`logical_rules_differential_probe.cc` (the C++ side, reading the SAME file)
consume, so both sides run over byte-identical inputs.

Usage:
    python3 tools/cpp_probe/gen_boolean_trees.py [--count N] [--seed S]
        [--max-depth D] [--arity A] > corpus.txt

Run standalone to regenerate a fixed corpus; `run_differential.py` invokes
this itself so a bare `run_differential.py` needs no separate step.
"""

from __future__ import annotations

import argparse
import random


def gen_tree(rng: random.Random, depth: int, arity: int, leaf_prob: float = 0.35) -> str:
    """One random boolean combo-tree, as text.

    A leaf is a bare argument (`$n`) or its negation (`!$n`) -- both are
    real `argument` vertices (positive/negative `idx`), never a `not(...)`
    wrapper at leaf level (that's what internal nodes are for). An
    internal node is `and`/`or` with 2-3 children, matching the shapes
    `reduce_to_enf` actually branches on -- unary and >3-ary nodes are
    already covered by the hand-picked corpus in `test_logical_rules.py`.
    """
    if depth <= 0 or rng.random() < leaf_prob:
        idx = rng.randint(1, arity)
        return f"!${idx}" if rng.random() < 0.5 else f"${idx}"
    op = rng.choice(["and", "or"])
    n_children = rng.randint(2, 3)
    children = [gen_tree(rng, depth - 1, arity, leaf_prob) for _ in range(n_children)]
    return f"{op}(" + " ".join(children) + ")"


def generate_corpus(count: int, seed: int, max_depth: int, arity: int) -> list[str]:
    rng = random.Random(seed)
    return [gen_tree(rng, max_depth, arity) for _ in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--arity", type=int, default=4)
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    trees = generate_corpus(args.count, args.seed, args.max_depth, args.arity)
    text = "\n".join(trees) + "\n"
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
