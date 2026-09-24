"""Capture tests/fixtures/gomea_golden.json from the GP-GOMEA golden harness.

``golden.cpp`` (next to this file) holds verbatim copies of GP-GOMEA's
``DetermineNearestNeighbour``, ``BuildLinkageTreeFromSimilarityMatrix`` and
the entropy / ``MI_distribution_adjustments`` loops of
``GenerateLinkageTreeFOS`` (GP-GOMEA@750220c, ``src/GOMEA/GOMEAFOS.cpp``,
Apache-2.0); its header comment lists every substitution. This script:

1. compiles it with ``g++ -O2 -std=c++17`` into ``tools/gomea_golden/build/``
   (git-ignored);
2. generates a fixed set of deterministic cases (``random.Random`` with fixed
   seeds, used for case generation only -- nothing here touches pyMoses's
   own MT19937 stream);
3. feeds them to the harness on stdin and writes its JSON output, together
   with the inputs, the upstream commit, the compiler version and the harness
   sha256, plus a sidecar ``gomea_golden.manifest.json`` (``tools/_manifest.py``).

Developer-run only (needs a C++ compiler); CI and the test suite only read
the committed fixture. Usage::

    .venv/bin/python tools/gomea_golden/capture.py
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SOURCE = HERE / "golden.cpp"
BUILD_DIR = HERE / "build"
BINARY = BUILD_DIR / "golden"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "gomea_golden.json"

UPSTREAM_REPO = "https://github.com/marcovirgolin/GP-GOMEA"
UPSTREAM_COMMIT = "750220c23d599e515ca337dcca617ae5faab8b4d"
UPSTREAM_FILE = "src/GOMEA/GOMEAFOS.cpp"

sys.path.insert(0, str(REPO_ROOT))
from tools._manifest import build_manifest, write_sidecar  # noqa: E402


def compile_harness() -> str:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["g++", "-O2", "-std=c++17", "-o", str(BINARY), str(SOURCE)],
        check=True,
    )
    version = subprocess.run(
        ["g++", "--version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    return version


# ---------------------------------------------------------------- case builders


def _uniform(rng: random.Random, n: int, alphabets: list[int]) -> list[list[int]]:
    return [[rng.randrange(k) for k in alphabets] for _ in range(n)]


def _ensure_varied(pop: list[list[int]], alphabets: list[int]) -> None:
    """Force every column to take at least two values (H0 > 0 everywhere)."""
    for j, k in enumerate(alphabets):
        if len({row[j] for row in pop}) < 2:
            pop[0][j] = (pop[1][j] + 1) % k


def _order(rng: random.Random, d: int) -> list[int]:
    order = list(range(d))
    rng.shuffle(order)
    return order


def _noisy_copy(rng: random.Random, src: int, k: int, p: float) -> int:
    return src % k if rng.random() >= p else rng.randrange(k)


def build_cases() -> list[dict]:
    cases: list[dict] = []

    def add(name: str, pop0, pop1, order, tags) -> None:
        cases.append({"name": name, "pop0": pop0, "pop1": pop1, "order": order, "tags": tags})

    # 1. d = 2, binary
    r = random.Random(1)
    p = _uniform(r, 16, [2, 2])
    _ensure_varied(p, [2, 2])
    add("d2_binary", p, p, _order(r, 2), ["d2"])

    # 2. d = 3, unequal alphabets, awkward N
    r = random.Random(2)
    a = [2, 3, 5]
    p = _uniform(r, 7, a)
    _ensure_varied(p, a)
    add("d3_unequal_alphabets", p, p, _order(r, 3), ["d3", "unequal_alphabets"])

    # 3. d = 3, identity order, dependent pair
    r = random.Random(3)
    p = []
    for _ in range(40):
        x = r.randrange(3)
        p.append([x, _noisy_copy(r, x, 3, 0.1), r.randrange(2)])
    _ensure_varied(p, [3, 3, 2])
    add("d3_dependent_pair", p, p, [0, 1, 2], ["d3", "dependent"])

    # 4. d = 12, unequal alphabets, independent
    r = random.Random(4)
    a = [r.choice([2, 3, 4, 7]) for _ in range(12)]
    p = _uniform(r, 64, a)
    _ensure_varied(p, a)
    add("d12_unequal_alphabets", p, p, _order(r, 12), ["d12", "unequal_alphabets"])

    # 5. d = 12, strongly dependent pairs / triples
    r = random.Random(5)
    a = [2, 2, 3, 3, 4, 4, 2, 2, 2, 5, 5, 3]
    p = []
    for _ in range(96):
        x, y, z, w = r.randrange(2), r.randrange(3), r.randrange(4), r.randrange(5)
        p.append([
            x, 1 - x,
            y, (y + 1) % 3,
            z, _noisy_copy(r, z, 4, 0.05),
            r.randrange(2), (x ^ r.randrange(2)) if r.random() < 0.3 else x, r.randrange(2),
            w, _noisy_copy(r, w, 5, 0.2), r.randrange(3),
        ])
    _ensure_varied(p, a)
    add("d12_dependent_pairs", p, p, _order(r, 12), ["d12", "dependent", "unequal_alphabets"])

    # 6. d = 30, random alphabets, independent
    r = random.Random(6)
    a = [r.choice([2, 2, 3, 4]) for _ in range(30)]
    p = _uniform(r, 128, a)
    _ensure_varied(p, a)
    add("d30_independent", p, p, _order(r, 30), ["d30", "unequal_alphabets"])

    # 7. d = 30, block structure (blocks of 5 noisy copies)
    r = random.Random(7)
    p = []
    for _ in range(160):
        row = []
        for _b in range(6):
            base = r.randrange(3)
            row.extend(_noisy_copy(r, base, 3, 0.15) for _ in range(5))
        p.append(row)
    _ensure_varied(p, [3] * 30)
    add("d30_blocks", p, p, _order(r, 30), ["d30", "dependent"])

    # 8. second generation: H0 from a uniform first population; the current
    #    population has converged fields and copied fields.
    r = random.Random(8)
    a = [r.choice([2, 3, 4]) for _ in range(12)]
    p0 = _uniform(r, 64, a)
    _ensure_varied(p0, a)
    p1 = []
    for _ in range(64):
        row = [r.randrange(k) for k in a]
        row[0] = 0 if r.random() < 0.9 else row[0]          # nearly converged
        row[1] = 0                                           # converged
        row[3] = row[2] % a[3]                               # linked
        row[5] = row[4] % a[5] if r.random() < 0.8 else row[5]
        p1.append(row)
    add("d12_second_generation", p0, p1, _order(r, 12), ["d12", "second_generation"])

    # 9. negative biased similarity: identical columns in H0, independent now
    r = random.Random(9)
    p0 = []
    for _ in range(50):
        x, y = r.randrange(2), r.randrange(3)
        p0.append([x, x, y, y, r.randrange(2)])
    _ensure_varied(p0, [2, 2, 3, 3, 2])
    p1 = _uniform(r, 50, [2, 2, 3, 3, 2])
    add("d5_negative_biased", p0, p1, _order(r, 5), ["negative_biased", "second_generation"])

    # 10. exact ties: full factorial over 3 bits, each bit duplicated
    rows = [[(m >> b) & 1 for b in range(3)] for m in range(8)]
    p = [[row[0], row[0], row[1], row[1], row[2], row[2]] for row in rows] * 3
    add("d6_factorial_ties", p, p, [0, 1, 2, 3, 4, 5], ["ties"])

    # 11. exact ties, same data, a different injected order
    add("d6_factorial_ties_shuffled", p, p, [3, 5, 0, 4, 1, 2], ["ties"])

    # 12. exact ties: all columns identical
    r = random.Random(12)
    col = [r.randrange(3) for _ in range(30)]
    col[0], col[1] = 0, 1
    p = [[v] * 5 for v in col]
    add("d5_all_identical", p, p, _order(r, 5), ["ties"])

    # 13. d = 12 ties: factorial over 4 bits with XOR / copies
    rows = [[(m >> b) & 1 for b in range(4)] for m in range(16)]
    p = [
        [*q, *q, q[0] ^ q[1], q[2] ^ q[3], q[0] ^ q[2], q[1] ^ q[3]]
        for q in rows
    ] * 2
    add("d12_factorial_ties", p, p, _order(random.Random(13), 12), ["d12", "ties"])

    # 14. d = 30, second generation with a shrinking alphabet use, H0 uniform
    r = random.Random(14)
    a = [r.choice([2, 3, 5]) for _ in range(30)]
    p0 = _uniform(r, 128, a)
    _ensure_varied(p0, a)
    p1 = []
    for _ in range(128):
        row = [r.randrange(k) for k in a]
        for j in range(0, 30, 3):
            if r.random() < 0.7:
                row[j + 1] = row[j] % a[j + 1]
        p1.append(row)
    add("d30_second_generation", p0, p1, _order(r, 30), ["d30", "second_generation"])

    # 15. d = 2, negative biased similarity
    p0 = [[0, 0], [1, 1], [0, 0], [1, 1]]
    p1 = [[0, 0], [0, 1], [1, 0], [1, 1]]
    add("d2_negative_biased", p0, p1, [1, 0], ["d2", "negative_biased", "second_generation"])

    return cases


def harness_input(cases: list[dict]) -> str:
    lines = [str(len(cases))]
    for c in cases:
        d = len(c["order"])
        lines.append(f"{d} {len(c['pop0'])} {len(c['pop1'])}")
        lines.extend(" ".join(map(str, row)) for row in c["pop0"])
        lines.extend(" ".join(map(str, row)) for row in c["pop1"])
        lines.append(" ".join(map(str, c["order"])))
    return "\n".join(lines) + "\n"


def main() -> None:
    compiler = compile_harness()
    cases = build_cases()
    for c in cases:
        d = len(c["order"])
        assert sorted(c["order"]) == list(range(d))
        assert all(len(row) == d for row in c["pop0"] + c["pop1"])
    out = subprocess.run(
        [str(BINARY)], input=harness_input(cases), check=True, capture_output=True, text=True
    ).stdout
    results = json.loads(out)["cases"]
    assert len(results) == len(cases)
    for c, res in zip(cases, results, strict=True):
        c.update(res)

    harness_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    manifest = build_manifest(
        repo_root=REPO_ROOT,
        capture_script="tools/gomea_golden/capture.py",
        capture_script_path=Path(__file__).resolve(),
        probe_source="tools/gomea_golden/golden.cpp",
        probe_source_path=SOURCE,
        probe_kind="cpp_probe",
        inputs={"case_names": [c["name"] for c in cases]},
    )
    fixture = {
        "upstream": {
            "repo": UPSTREAM_REPO,
            "commit": UPSTREAM_COMMIT,
            "file": UPSTREAM_FILE,
            "licence": "Apache-2.0",
        },
        "compiler": compiler,
        "compile_flags": "-O2 -std=c++17",
        "harness_sha256": harness_sha,
        "cases": cases,
    }
    # One compact line per case keeps the file small and diffs per case.
    head = {k: v for k, v in fixture.items() if k != "cases"}
    body = ",\n".join(json.dumps(c, separators=(",", ":")) for c in cases)
    text = json.dumps(head, indent=1)[:-2] + ',\n "cases": [\n' + body + "\n ]\n}\n"
    json.loads(text)  # round-trip guard
    FIXTURE.write_text(text)
    # Sidecar manifest: tests/fixtures/test_fixture_manifests.py pins the set
    # of embedded-manifest fixtures, so new fixtures use the sidecar shape.
    write_sidecar(FIXTURE, manifest)
    print(f"wrote {FIXTURE.relative_to(REPO_ROOT)}: {len(cases)} cases")


if __name__ == "__main__":
    main()
