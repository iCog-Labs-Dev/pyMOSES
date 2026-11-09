"""Server round 9: the plans (plan9.json, plan9-smoke.json) and the configuration of 1ba, 1bb
and 1bc, as SERVER-ROUND9-PREREGISTRATION.md registers them: the new base BASE, an evaluation
budget of 20,000 with no time limit, three independent phases, no choice step and no chain."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.gomea.stats import holm, wilcoxon_signed_rank

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs" / "gomea"
SERVER = REPO / "tools" / "server"
sys.path.insert(0, str(REPO / "tools" / "gomea"))
import phase_report  # noqa: E402

# The new base of the pre-registration, flag for flag.
BASE = ["-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "dynamic",
        "--hc-fraction-of-nn", "0.05", "--gomea-skip-inactive", "0",
        "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
        "--knob-probe-baseline", "reduced", "--linear-scaling", "1",
        "--const-opt", "lm", "--const-opt-guard", "val",
        "--gomea-accept-complexity-ratio", "0.125", "--gomea-deme-store", "accepted"]
# BASE-N16: --gomea-pop-size 16 in place of the dynamic flags
N16 = ["-a", "gomea", "--gomea-fos", "lt-b", "--gomea-pop-size", "16",
       "--gomea-skip-inactive", "0",
       "--gomea-genotype-memo", "1", "--gomea-contin", "copy",
       "--knob-probe-baseline", "reduced", "--linear-scaling", "1",
       "--const-opt", "lm", "--const-opt-guard", "val",
       "--gomea-accept-complexity-ratio", "0.125", "--gomea-deme-store", "accepted"]
CAP = ["--gomea-max-evals-per-deme", "250"]
ARMS = {
    "1ba": {"BASE": BASE, "BASE-N16": N16},
    "1bb": {"BASE": BASE, "BASE-S": BASE},
    "1bc": {"BASE": BASE, "BASE-C250": BASE + CAP, "N16": N16, "N16-C250": N16 + CAP},
}
SEED = {"binary": "../gomea-results/gpgomea-seed-build/gpgomea-seed", "share": 0.2,
        "height": 3, "k": 5, "functions": "+_-_*"}
D1BA = ["Air", "Bos", "Dow", "EnC", "Tow", "WiR", "WiW", "Yac"]
D1BB = ["Con", "EnH", "Air", "Yac"]
D1BC = {"Con": range(11, 21), "Tow": range(11, 21), "Yac": range(11, 21),
        "WiR": range(11, 21), "Bos": range(1, 11), "EnC": range(1, 11)}
DATASETS = {"1ba": D1BA, "1bb": D1BB, "1bc": list(D1BC)}
PHASES = ["1ba", "1bb", "1bc"]
PREFIXES = ["phase", "server-smoke9-"]


def load(path):
    return json.loads(path.read_text())


def stage1(q):
    """The stage-1 manifests of phase_run.sh: sr.json, then sr-*.json without sr-ext-*."""
    return [q / "sr.json"] + sorted(f for f in q.glob("sr-*.json")
                                    if not f.name.startswith("sr-ext-"))


def ext(q):
    return sorted(q.glob("*-ext-*.json"))


def runs(ms):
    """(dataset, split) -> arms, over a list of manifests."""
    out = {}
    for f in ms:
        m = load(f)
        for d in m["datasets"]:
            for s in m["splits"]:
                k = (d, s)
                assert k not in out, (f, k)
                out[k] = list(m["arms"])
    return out


def all_manifests(prefix):
    for p in PHASES:
        q = CONFIGS / f"{prefix}{p}"
        for f in stage1(q) + ext(q):
            yield p, f, load(f)


@pytest.mark.parametrize("prefix", PREFIXES)
def test_arm_flags_match_the_preregistration(prefix):
    for p, f, m in all_manifests(prefix):
        assert {a: s["argv"] for a, s in m["arms"].items()} == ARMS[p], f


def test_n16_is_base_with_16_in_place_of_the_dynamic_flags():
    i = BASE.index("dynamic")
    assert N16 == BASE[:i] + ["16"] + BASE[i + 3:]
    assert "--gomea-max-evals-per-deme" not in BASE


def test_budget_is_evaluations_with_no_time_limit():
    for p, f, m in all_manifests("phase"):
        assert m["wall_cap_s"] == 36000, f
        assert m["datasets_dir"] == "../gomea-results/phase1f/datasets"
        for a, s in m["arms"].items():
            assert s["budget"] == {"evals": 20000}, (f, a)
            assert s["binary"] == ".venv/bin/moses"
            assert not any(t.startswith(("--max-time", "-m")) for t in s["argv"]), (f, a)


def test_the_seeded_arm_has_the_seed_block_of_1ab():
    assert load(CONFIGS / "phase1ab" / "sr.json")["arms"]["GOMEA-S"]["seed"] == SEED
    for prefix in PREFIXES:
        for p, f, m in all_manifests(prefix):
            for a, s in m["arms"].items():
                if a == "BASE-S":
                    assert s["seed"] == SEED, f  # no "with_empty": the runner keeps it
                    assert set(s) == {"binary", "budget", "argv", "seed"}
                else:
                    assert set(s) == {"binary", "budget", "argv"}, (f, a)


def test_datasets_and_splits_of_1ba_and_1bb():
    for p, ds in (("1ba", D1BA), ("1bb", D1BB)):
        q = CONFIGS / f"phase{p}"
        assert set(runs(stage1(q))) == {(d, s) for d in ds for s in range(1, 11)}
        assert [f.name for f in ext(q)] == sorted(f"sr-ext-{d}.json" for d in ds)
        for d in ds:
            m = load(q / f"sr-ext-{d}.json")
            assert m["datasets"] == [d] and m["splits"] == list(range(11, 21))
            assert m["name"] == f"phase{p}-sr-ext-{d}"


def test_dow_of_1ba_has_a_manifest_of_its_own():
    q = CONFIGS / "phase1ba"
    assert "Dow" not in load(q / "sr.json")["datasets"]
    for f in ("sr-Dow.json", "sr-ext-Dow.json"):
        m = load(q / f)
        assert m["datasets"] == ["Dow"] and m["mem_limit_mb"] == 3072
    for f in stage1(q) + ext(q):
        assert load(f)["mem_limit_mb"] == 3072


def test_1bc_uses_only_the_listed_splits_and_no_extension():
    q = CONFIGS / "phase1bc"
    assert ext(q) == []
    assert set(runs(stage1(q))) == {(d, s) for d, ss in D1BC.items() for s in ss}
    assert load(q / "spec.json")["extend"] is False


# The 1aw extensions that round 7 ran (server-run7: phase1aw-sr-ext-Air and -EnH only).
AW_EXTENDED = ["Air", "EnH"]


def test_1bc_never_uses_the_data_of_1aw():
    q = CONFIGS / "phase1aw"
    m = load(q / "template.json")
    aw = {(d, s) for d in m["datasets"] for s in m["splits"]}
    for d in AW_EXTENDED:
        e = load(q / f"template-sr-ext-{d}.json")
        aw |= {(d, s) for s in e["splits"]}
    assert ("Con", 1) in aw and ("EnH", 20) in aw
    assert not aw & set(runs(stage1(CONFIGS / "phase1bc")))


@pytest.mark.parametrize("p, s1, s2", [("1ba", 160, 160), ("1bb", 80, 80), ("1bc", 240, 0)])
def test_run_counts(p, s1, s2):
    q = CONFIGS / f"phase{p}"
    assert sum(len(a) for a in runs(stage1(q)).values()) == s1
    assert sum(len(a) for a in runs(ext(q)).values()) == s2


def _contrasts(ds, name, a, b, metric):
    return [[d, name, a, b, metric] for d in ds]


SPECS = {
    "1ba": (_contrasts(D1BA, "base-vs-n16", "BASE", "BASE-N16", "test_nmse"),
            _contrasts(D1BA, "base-vs-n16-size", "BASE", "BASE-N16", "program_size")
            + _contrasts(D1BA, "base-vs-n16-gap", "BASE", "BASE-N16", "train_test_gap")
            + _contrasts(D1BA, "base-vs-n16-cpu", "BASE", "BASE-N16", "cpu_s")),
    "1bb": (_contrasts(D1BB, "seeded-vs-base", "BASE-S", "BASE", "test_nmse"),
            _contrasts(D1BB, "seeded-vs-base-size", "BASE-S", "BASE", "program_size")
            + _contrasts(D1BB, "seeded-vs-base-cpu", "BASE-S", "BASE", "cpu_s")),
    "1bc": (_contrasts(list(D1BC), "c250-vs-base", "BASE-C250", "BASE", "test_nmse"),
            _contrasts(list(D1BC), "n16c250-vs-n16", "N16-C250", "N16", "test_nmse")
            + _contrasts(list(D1BC), "c250-vs-base-size", "BASE-C250", "BASE", "program_size")
            + _contrasts(list(D1BC), "c250-vs-base-lm", "BASE-C250", "BASE", "lm_calls")),
}


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("p, n1, n2", [("1ba", 8, 24), ("1bb", 4, 8), ("1bc", 6, 18)])
def test_spec(prefix, p, n1, n2):
    s = load(CONFIGS / f"{prefix}{p}" / "spec.json")
    assert s["phase"] == (p if prefix == "phase" else f"smoke9-{p}")
    assert s["metrics"] == dict.fromkeys(DATASETS[p], "test_nmse")
    assert s["primary"] == SPECS[p][0] and len(s["primary"]) == n1
    assert s["secondary"] == SPECS[p][1] and len(s["secondary"]) == n2
    assert "pooled_primary" not in s and "secondary_verdicts" not in s
    # 1bc: N16-C250 vs N16 is its own Holm family; the other 12 secondary ones are a second
    assert s.get("secondary_families") == ([["n16c250-vs-n16"]] if p == "1bc" else None)
    assert s.get("extend", True) is (p != "1bc")
    for c in s["primary"] + s["secondary"]:
        assert c[4] in phase_report.METRICS


def test_holm_families_at_10_pairs():
    """With 10 pairs the smallest two-sided p is 2/1024. Holm over the 8, 4 and 6 primary
    contrasts can give a verdict at stage 1. In 1bc, Holm over the own family of the 6
    N16-C250 vs N16 contrasts gives at least 6 * 2/1024 = 0.0117, so a verdict is possible;
    so does the second family of 12 (size and LM calls, 0.0234). One family of 18 (the first
    draft) could not (0.035), and neither can the 24 secondary contrasts of 1ba."""
    p = wilcoxon_signed_rank([float(i + 1) for i in range(10)])[1]
    assert p == pytest.approx(2 / 1024)
    for n in (8, 4, 6, 6, 12):
        assert holm([p] * n)[0] < phase_report.ALPHA
    assert holm([p] * 6)[0] == pytest.approx(6 * 2 / 1024)
    for n in (24, 18):
        assert holm([p] * n)[0] > phase_report.ALPHA


@pytest.mark.parametrize("prefix", PREFIXES)
def test_1bc_families_cover_the_secondary_contrasts(prefix):
    s = load(CONFIGS / f"{prefix}1bc" / "spec.json")
    names = [c[1] for c in s["secondary"]]
    own = s["secondary_families"][0]
    assert sum(n in own for n in names) == 6
    assert sum(n not in own for n in names) == 12


@pytest.mark.parametrize("plan, name, state_dir, prefix, sid", [
    ("plan9.json", "server-round9-2026-11", "server-run9", "phase", "phase"),
    ("plan9-smoke.json", "server-round9-smoke", "server-run9-smoke", "server-smoke9-",
     "smoke9-"),
])
def test_plan_is_s0_then_the_three_phases(plan, name, state_dir, prefix, sid):
    p = load(SERVER / plan)
    assert p["name"] == name and p["state_dir"] == state_dir
    assert [s["type"] for s in p["steps"]] == ["check", "phase", "phase", "phase"]
    assert [s["id"] for s in p["steps"]] == ["S0"] + [f"{sid}{x}" for x in PHASES]
    for st, x in zip(p["steps"][1:], PHASES):
        assert st["phase"] == f"{prefix}{x}"
        assert "config_dir" not in st and not st.get("pilot") and not st.get("set_commit_in")
        want = [f"datasets:{','.join(DATASETS[x])}"]
        assert st["requires"] == (["gpgomea"] + want if x == "1bb" else want)
        assert (CONFIGS / f"{prefix}{x}" / "spec.json").exists()
    assert "then" not in p


def test_s0_of_the_round_is_complete():
    s0 = load(SERVER / "plan9.json")["steps"][0]
    assert s0["pytest"] == {} and s0["repro"] == {}


def test_s0_of_the_smoke_runs_the_new_tests():
    args = load(SERVER / "plan9-smoke.json")["steps"][0]["pytest"]["args"]
    for t in ("tests/server", "tests/gomea/test_phase_report.py",
              "tests/gomea/test_phase_report_round9.py"):
        assert t in args


def test_no_plan_chains_to_round_9():
    for f in SERVER.glob("plan*.json"):
        assert load(f).get("then") not in ("round9", "smoke9"), f


@pytest.mark.parametrize("name, plan", [("round9", "plan9.json"), ("smoke9", "plan9-smoke.json")])
def test_plan_names_resolve(name, plan):
    r = subprocess.run(
        ["bash", "-c", f'source tools/server/_common.sh --plan {name}; echo "$PLAN"'],
        cwd=REPO, capture_output=True, text=True, check=True)
    assert r.stdout.strip() == str(SERVER / plan)
    sys.path.insert(0, str(SERVER))
    import chain
    assert chain.PLAN_FILES[name] == plan
    assert chain.resolve(name) == SERVER / plan
    assert [f.name for f in chain.forward(SERVER / plan)] == [plan]


@pytest.mark.parametrize("script", ["_common.sh", "start.sh", "stop.sh", "status.sh",
                                    "install_autostart.sh"])
def test_help_text_names_the_plans(script):
    assert "round9|smoke9" in (SERVER / script).read_text()


def test_setup_builds_every_dataset_of_the_plan():
    text = (SERVER / "setup.sh").read_text()
    built = text.split("DATASETS=(", 1)[1].split(")", 1)[0].split()
    need = set(D1BA) | set(D1BB) | set(D1BC)
    assert need - {"Tow", "Dow"} <= set(built)
    assert '"Tow tower ' in text and '"Dow dowchem ' in text


@pytest.mark.parametrize("p", PHASES)
def test_smoke_has_the_same_arms_and_is_small(p):
    full, q = CONFIGS / f"phase{p}", CONFIGS / f"server-smoke9-{p}"
    assert ext(q) == []
    assert [f.name for f in stage1(q)] == [f.name for f in stage1(full)]
    for f in stage1(q):
        m, fm = load(f), load(full / f.name)
        assert m["name"] == fm["name"].replace(f"phase{p}", f"server-smoke9-{p}")
        assert m["datasets"] == fm["datasets"] and len(m["splits"]) == 2
        assert set(m["splits"]) <= set(fm["splits"])
        assert m["wall_cap_s"] <= 300 and m["mem_limit_mb"] == fm["mem_limit_mb"]
        assert list(m["arms"]) == list(fm["arms"])
        for a, s in m["arms"].items():
            assert s["argv"] == fm["arms"][a]["argv"]
            assert s.get("seed") == fm["arms"][a].get("seed")
            # 1bb: GP-GOMEA's first generation alone costs about 600 evaluations
            assert s["budget"] == {"evals": 1000 if p == "1bb" else 300}


def test_smoke_has_a_dow_run_and_a_seeded_arm():
    assert load(CONFIGS / "server-smoke9-1ba" / "sr-Dow.json")["datasets"] == ["Dow"]
    assert "seed" in load(CONFIGS / "server-smoke9-1bb" / "sr.json")["arms"]["BASE-S"]
