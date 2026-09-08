"""One-time archaeological backfill: write `<name>.manifest.json` sidecars
for the fixtures that predate tools/_manifest.py, without touching a
single byte of any fixture's own content.

Context (docs/CARRY-FORWARD.md Sec 5, and the S11 fixture-integrity task
that closes it): S9 added a `_manifest` to tests/fixtures/hillclimbing.json
alone; S10 mirrored it for tests/fixtures/deme_expander.json. Every OTHER
fixture predates that idea entirely, so simply re-running each
tools/capture_*.py script to backfill a manifest would also re-invoke a
C++ compiler and (per this task's explicit constraint) risks silently
"fixing" a fixture whose data has quietly drifted from what its probe
would produce today -- exactly the failure mode CARRY-FORWARD.md Sec 5
warns about, and exactly what must be a reported FINDING, not a silent
overwrite.

So instead, this script reconstructs each manifest from **git history**:
for every fixture below, `repo_commit`/`generated_at_utc` are the commit
that last touched that exact fixture file and that commit's own date (a
faithful record of when the DATA was produced), and every
`*_sha256` is verified (see `_ARCHAEOLOGY_NOTE` per entry, and this task's
own report) to be byte-identical between that historical commit and HEAD
-- i.e. the probe/script that produced this fixture has not changed since,
so recording the CURRENT file's hash is provably the same as recording
the historical one, without needing to check out the old commit at
runtime. `rng.json` is the sole exception: `rng_probe.cc`/
`tools/capture_golden.py` WERE rewritten after rng.json was last captured
(cogutil-linked MT19937RandGen instead of a hand-copy) -- a live,
resource-bounded rebuild-and-diff during this task proved the rewritten
probe's output is byte-for-byte identical to the committed fixture, so
its manifest instead records repo_commit=HEAD (at backfill time) and
`generated_at_utc`=the verification's own timestamp, honestly describing
"verified consistent as of now" rather than misattributing a stale commit
to hashes that postdate it. See this task's own
.superpowers/sdd/2026-09-08-pymetta-s11-local-moses/fixture-integrity-report.md
for the full per-fixture ledger (git commands run, hashes compared).

This script is idempotent and safe to re-run; it only ever writes
`<fixture>.manifest.json` files, never the fixtures themselves. It is a
one-time tool, not part of tools/ci.sh -- once every current fixture has a
sidecar, new fixtures get theirs from their OWN capture script (see
tools/_manifest.py), not from here.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tools._manifest import build_manifest, write_sidecar  # noqa: E402

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
CPP_PROBE = REPO_ROOT / "tools" / "cpp_probe"

# The exact CLASSIC/PREFIX literal values every affected capture script
# hardcoded at the time these fixtures were actually captured (see git
# blame on each tools/capture_*.py before this task's _capture_env.py
# refactor) -- recorded here as plain informational strings, matching
# tools/_manifest.py's own "classic_dir/prefix_dir" field semantics
# ("as resolved when captured", not re-checked). Deliberately NOT read
# from PYMETTA_CLASSIC_DIR/PYMETTA_MOSES_PREFIX: this script must stay
# runnable with no environment variables and no checkouts present at all.
_CLASSIC_DIR = "/home/yab/Documents/Projects/iCog/HyperonRelated/MOSES/OpenCog/Moses"
_PREFIX_DIR = "/home/yab/opt/moses-legacy"

# One entry per fixture needing a backfilled sidecar (everything except
# hillclimbing.json/deme_expander.json, which already embed `_manifest`
# via their own capture scripts). Each entry: (fixture stem, commit that
# last touched tests/fixtures/<stem>.json, that commit's ISO-8601 date,
# kwargs for build_manifest beyond repo_root/capture_script/
# capture_script_path/repo_commit/generated_at_utc).
_ENTRIES: list[tuple[str, str, str, dict]] = [
    (
        "boolean_builder",
        "d0b409930174d21a48cf82a443389b3a526eb4e7",
        "2026-09-07T13:10:48+03:00",
        {
            "capture_script": "tools/capture_boolean_builder.py",
            "probe_source": "tools/cpp_probe/representation_probe.cc",
            "probe_source_path": CPP_PROBE / "representation_probe.cc",
            "probe_kind": "cpp_probe",
            "classic_dir": _CLASSIC_DIR,
            "prefix_dir": _PREFIX_DIR,
            "inputs": {
                "description": (
                    "fixed deterministic Boolean-builder exemplars/arities hardcoded "
                    "in representation_probe.cc; no external dataset or seed"
                )
            },
        },
    ),
    (
        "combo_roundtrip",
        "b5b0dcbc3fe41daa62526f8f30b40de80c0a675b",
        "2026-09-03T15:13:04+03:00",
        {
            "capture_script": "tools/capture_combo_roundtrip.py",
            "probe_source": None,
            "probe_source_path": None,
            "probe_kind": "classic_cli",
            "cli_tool": "bin/combo-fmt-converter",
            "prefix_dir": _PREFIX_DIR,
            "inputs": {
                "corpus_size": 30,
                "corpus_sources": [
                    "$CLASSIC/testTrees.sh (29 trees, TEST_TREES_SH_CORPUS)",
                    "$CLASSIC/tests/comboreduct/combo/iostream_comboUTest.cxxtest:11 "
                    "(1 tree, IOSTREAM_COMBO_UTEST_CORPUS)",
                ],
            },
        },
    ),
    (
        "containers",
        "4e209d50937b09d0faaa46d035f75e835c6a8c70",
        "2026-09-07T15:29:36+03:00",
        {
            "capture_script": "tools/capture_containers.py",
            "probe_source": "tools/cpp_probe/containers_probe.cc",
            "probe_source_path": CPP_PROBE / "containers_probe.cc",
            "probe_kind": "cpp_probe",
            "classic_dir": _CLASSIC_DIR,
            "prefix_dir": _PREFIX_DIR,
            "inputs": {
                "description": (
                    "fixed instance_set/scored_instance test cases hardcoded in "
                    "containers_probe.cc; no external dataset or seed"
                )
            },
        },
    ),
    (
        "containers_sct",
        "e88cd293f1eca33a7b642b0405275d32ab81fa6d",
        "2026-09-08T09:36:47+03:00",
        {
            "capture_script": "tools/capture_containers_sct.py",
            "probe_source": "tools/cpp_probe/containers_sct_probe.cc",
            "probe_source_path": CPP_PROBE / "containers_sct_probe.cc",
            "probe_kind": "cpp_probe",
            "classic_dir": _CLASSIC_DIR,
            "prefix_dir": _PREFIX_DIR,
            "inputs": {
                "description": (
                    "fixed scored_combo_tree ordering/dedup test cases hardcoded "
                    "in containers_sct_probe.cc; no external dataset or seed"
                )
            },
        },
    ),
    (
        "eval_boolean",
        "ffa52f3e4f65d970391f605ce49d4e149bd32c65",
        "2026-09-03T15:50:17+03:00",
        {
            "capture_script": "tools/capture_eval_boolean.py",
            "probe_source": None,
            "probe_source_path": None,
            "probe_kind": "classic_cli",
            "cli_tool": "bin/eval-table",
            "classic_dir": _CLASSIC_DIR,
            "prefix_dir": _PREFIX_DIR,
            "inputs": {
                "datasets": ["trainingBoolean.dat", "trainingBooleanXor.dat"],
                "combo_corpus_fixture": "tests/fixtures/combo_roundtrip.json",
            },
        },
    ),
    (
        "field_set",
        "6324ccf43ac1b1027a5d96be259969df59993fad",
        "2026-09-07T11:49:33+03:00",
        {
            "capture_script": "tools/capture_field_set.py",
            "probe_source": "tools/cpp_probe/field_set_probe.cc",
            "probe_source_path": CPP_PROBE / "field_set_probe.cc",
            "probe_kind": "cpp_probe",
            "classic_dir": _CLASSIC_DIR,
            "prefix_dir": _PREFIX_DIR,
            "inputs": {
                "description": (
                    "fixed field_set layout/pack/ABI test cases hardcoded in "
                    "field_set_probe.cc; no external dataset or seed"
                )
            },
        },
    ),
    (
        "logical_knob",
        "2453afbc68994c7ab870cd21f9b97b3be6053d04",
        "2026-09-07T12:56:55+03:00",
        {
            "capture_script": "tools/capture_logical_knob.py",
            "probe_source": "tools/cpp_probe/logical_knob_probe.cc",
            "probe_source_path": CPP_PROBE / "logical_knob_probe.cc",
            "probe_kind": "cpp_probe",
            "classic_dir": _CLASSIC_DIR,
            "prefix_dir": _PREFIX_DIR,
            "inputs": {
                "description": (
                    "fixed logical-knob active/transitions/append/disallow/copy/"
                    "clear test cases hardcoded in logical_knob_probe.cc; no "
                    "external dataset or seed"
                )
            },
        },
    ),
    (
        "neighborhood",
        "7ade2cc4019804906e514dfe469d63ca318d3a99",
        "2026-09-07T15:29:53+03:00",
        {
            "capture_script": "tools/capture_neighborhood.py",
            "probe_source": "tools/cpp_probe/neighborhood_probe.cc",
            "probe_source_path": CPP_PROBE / "neighborhood_probe.cc",
            "probe_kind": "cpp_probe",
            "classic_dir": _CLASSIC_DIR,
            "prefix_dir": _PREFIX_DIR,
            "inputs": {
                "description": (
                    "fixed neighborhood_sampling test cases hardcoded in "
                    "neighborhood_probe.cc; no external dataset or seed"
                )
            },
        },
    ),
    (
        "reduct",
        "b816f05500e9993353b017bbbff993177b6b3bd7",
        "2026-09-07T10:20:58+03:00",
        {
            "capture_script": "tools/capture_reduct.py",
            "probe_source": "tools/cpp_probe/reduct_effort_probe.cc",
            "probe_source_path": CPP_PROBE / "reduct_effort_probe.cc",
            "probe_kind": "cpp_probe",
            "classic_dir": _CLASSIC_DIR,
            "prefix_dir": _PREFIX_DIR,
            "inputs": {
                "efforts": [1, 2, 3],
                "corpus_module": "tools/cpp_probe/reduct_corpus.py",
                "sources": [
                    "testTrees.sh",
                    "BooleanReductUTest.cxxtest",
                    "BooleanFullUTest.cxxtest",
                    "FullReductUTest.cxxtest",
                    "GeneralTestCases.py",
                ],
            },
        },
    ),
    (
        "representation",
        "6e6d0bba3df1b5f330eefc371da79ef33cd519f5",
        "2026-09-07T14:44:34+03:00",
        {
            "capture_script": "tools/capture_representation.py",
            "probe_source": "tools/cpp_probe/candidate_roundtrip_probe.cc",
            "probe_source_path": CPP_PROBE / "candidate_roundtrip_probe.cc",
            "probe_kind": "cpp_probe",
            "classic_dir": _CLASSIC_DIR,
            "prefix_dir": _PREFIX_DIR,
            "inputs": {
                "description": (
                    "fixed Boolean candidate/representation round-trip cases "
                    "hardcoded in candidate_roundtrip_probe.cc; no external "
                    "dataset or seed"
                )
            },
        },
    ),
    (
        "score",
        "5fc5e659ec8aa87102a8e36c938fe7b84baf1c6a",
        "2026-09-04T10:21:16+03:00",
        {
            "capture_script": "tools/capture_score.py",
            "probe_source": "tools/cpp_probe/score_probe.cc",
            "probe_source_path": CPP_PROBE / "score_probe.cc",
            "probe_kind": "cpp_probe",
            "classic_dir": _CLASSIC_DIR,
            "prefix_dir": _PREFIX_DIR,
            "inputs": {
                "datasets": [
                    "trainingBoolean.dat",
                    "trainingBooleanXor.dat",
                    "weighted.csv",
                    "precision.csv",
                ],
                "combo_corpus_fixture": "tests/fixtures/combo_roundtrip.json",
                "complexity_ratio_cases": 10,
            },
        },
    ),
    (
        "table",
        "a54f39b0c6095b6f011cbedceda43bff85297584",
        "2026-09-03T16:43:01+03:00",
        {
            "capture_script": "tools/capture_table.py",
            "probe_source": "tools/cpp_probe/table_probe.cc",
            "probe_source_path": CPP_PROBE / "table_probe.cc",
            "probe_kind": "cpp_probe",
            "classic_dir": _CLASSIC_DIR,
            "prefix_dir": _PREFIX_DIR,
            "inputs": {
                "cases": [
                    "dataset.csv",
                    "dataset_force_feature.csv",
                    "sparse.data:target1",
                    "sparse.data:target2",
                    "weighted.csv",
                    "timestamp_table.csv",
                    "timestamp_table2.csv",
                    "timestamp_table3.csv",
                    "trainingBoolean.dat",
                    "trainingBooleanXor.dat",
                ]
            },
        },
    ),
    (
        "tree",
        "dfd2f13cae826aa668c859feafffc6490b1732e1",
        "2026-09-03T11:00:33+03:00",
        {
            "capture_script": "tools/capture_tree.py",
            "probe_source": "tools/cpp_probe/tree_probe.cc",
            "probe_source_path": CPP_PROBE / "tree_probe.cc",
            "probe_kind": "cpp_probe",
            "inputs": {
                "description": (
                    "fixed splice-operation and traversal-order test cases "
                    "hardcoded in tree_probe.cc; no external dataset or seed. "
                    "Needs only a raw CogUtil source checkout ($COGUTIL) -- no "
                    "libcogutil link, no moses-legacy prefix, no classic MOSES "
                    "checkout. tools/capture_tree.py itself did not exist until "
                    "this task (CARRY-FORWARD.md Sec 5's named gap: 'tree.json "
                    "has no capture script in-repo')."
                )
            },
        },
    ),
    (
        "type_signature",
        "f3b46529d7ba64687f4c893a60e8e9596fe1ecbe",
        "2026-09-07T10:51:09+03:00",
        {
            "capture_script": "tools/capture_type_signature.py",
            "probe_source": "tools/cpp_probe/type_signature_probe.cc",
            "probe_source_path": CPP_PROBE / "type_signature_probe.cc",
            "probe_kind": "cpp_probe",
            "classic_dir": _CLASSIC_DIR,
            "prefix_dir": _PREFIX_DIR,
            "inputs": {
                "description": (
                    "fixed type-signature test cases hardcoded in "
                    "type_signature_probe.cc; no external dataset or seed"
                )
            },
        },
    ),
]

# rng.json: the sole non-archaeological entry -- see module docstring.
_RNG_ENTRY = (
    "rng",
    {
        "capture_script": "tools/capture_golden.py",
        "probe_source": "tools/cpp_probe/rng_probe.cc",
        "probe_source_path": CPP_PROBE / "rng_probe.cc",
        "probe_kind": "cpp_probe",
        "prefix_dir": _PREFIX_DIR,
        "inputs": {
            "seeds": [0, 1, 7, 42, 12345, 4294967295],
            "draws_per_seed": 2000,
            "series": ["raw", "randint", "randint_7", "randbool", "randdouble"],
            "verification": (
                "rng_probe.cc/tools/capture_golden.py were rewritten to link the "
                "real cogutil MT19937RandGen (previously a hand-copy against bare "
                "std::mt19937). A resource-bounded rebuild-and-run during this "
                "task's own work (see the fixture-integrity report) proved the "
                "rewritten probe's stdout is byte-for-byte identical to the "
                "already-committed tests/fixtures/rng.json -- so this manifest "
                "records the CURRENT probe/script hashes as verified-consistent, "
                "not a stale pre-rewrite hash that would spuriously fail "
                "tools/check_fixture_manifests.py forever."
            ),
        },
    },
)


def main(argv: list[str]) -> int:
    del argv
    written = []
    for stem, commit, date, kwargs in _ENTRIES:
        fixture_path = FIXTURE_DIR / f"{stem}.json"
        if not fixture_path.is_file():
            raise FileNotFoundError(f"expected fixture not found: {fixture_path}")
        capture_script_path = REPO_ROOT / kwargs["capture_script"]
        manifest = build_manifest(
            repo_root=REPO_ROOT,
            capture_script_path=capture_script_path,
            repo_commit=commit,
            generated_at_utc=date,
            **kwargs,
        )
        sidecar = write_sidecar(fixture_path, manifest)
        written.append(sidecar)

    stem, kwargs = _RNG_ENTRY
    fixture_path = FIXTURE_DIR / f"{stem}.json"
    capture_script_path = REPO_ROOT / kwargs["capture_script"]
    manifest = build_manifest(
        repo_root=REPO_ROOT,
        capture_script_path=capture_script_path,
        repo_commit="AUTO",
        **kwargs,
    )
    sidecar = write_sidecar(fixture_path, manifest)
    written.append(sidecar)

    for path in written:
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    print(f"backfilled {len(written)} sidecar manifest(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
