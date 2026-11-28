"""Server round 17 (phase 1bl): the design of ``SERVER-ROUND17-PREREGISTRATION.md`` ("Arms",
"Contrasts", "The Dow rule", "Memory"), for the tests of the configuration
(``tests/server/test_plan17.py``) and of the report (``test_round17_report.py``).

**HC-TLS.** ``HC_TLS`` is True: the configs have the HC-TLS arms. If the HC-TLS check of clone A
fails, the pre-registration leaves HC-TLS out (P3, P4 and the HC-TLS secondaries drop). Then:

1. run ``.venv/bin/python tools/gomea/drop_arms.py configs/gomea/phase1bl
   configs/gomea/phase1bl-a configs/gomea/server-smoke17-1bl configs/gomea/server-smoke17-1bl-a
   --arms HC-TLS HC-TLS-100K``;
2. set ``HC_TLS = False`` below;
3. run the tests (``test_plan17.py``, ``test_round17_report.py``) and commit.

The tests then check the design without HC-TLS (``spec(..., hc_tls=False)``), which is the
output of ``drop_arms.py`` (``test_round17_report.py`` checks that too).
"""

from __future__ import annotations

HC_TLS = True  # False only after drop_arms.py removed HC-TLS (see the module docstring)

DS6 = ["Air", "Con", "EnC", "EnH", "Tow", "Yac"]  # every family with verdicts (60 pairs)
DS7 = ["Air", "Con", "Dow", "EnC", "EnH", "Tow", "Yac"]
SPLITS = list(range(81, 91))
HALF_SPLITS = SPLITS[:5]
SMOKE_SPLITS = [1, 2]  # old splits: the smoke never uses splits 81-90
HC_LM_ARGV = ["-a", "hc", "--knob-probe-baseline", "reduced", "--linear-scaling", "1",
              "--const-opt", "lm", "--const-opt-guard", "val"]  # the round 13 and 14 flags
HC_TLS_ARGV = ["-a", "hc", "--knob-probe-baseline", "reduced", "--linear-scaling", "terms",
               "--const-opt", "none"]
ALL_ARMS = {
    "HC-LM": (HC_LM_ARGV, 20000),
    "HC-TLS": (HC_TLS_ARGV, 20000),
    "HC-LM-100K": (HC_LM_ARGV, 100000),
    "HC-TLS-100K": (HC_TLS_ARGV, 100000),
}
PARTNER = {"HC-LM": "BASE-TLS", "HC-TLS": "BASE-TLS", "HC-LM-100K": "BASE-TLS-100K",
           "HC-TLS-100K": "BASE-TLS-100K"}
MANIFESTS = ["sr.json", "sr-100K.json", "sr-Dow.json"]  # the order of phase_run.sh
# the local pilot (SERVER-ROUND17-PREREGISTRATION.md, amendment A3, decided by the user on
# 2026-11-28): each measured peak at most 50 % of its limit; the data sets without a pilot peak
# get room for HC-TLS; sr-100K.json 6,553 MB, the largest limit for W 9 on the server
MB = {"sr.json": 4096, "sr-Dow.json": 8192, "sr-100K.json": 6553}
# the wall cap, a guard only (A3, the user's decision of 2026-11-28): 72,000 s for the 100,000
# arms (the pilot Tow HC-TLS-100K run used 89 % of 36,000 s), 36,000 s for the 20,000 arms
WALL_CAP_S = {"sr.json": 36000, "sr-Dow.json": 36000, "sr-100K.json": 72000}
DATASETS_OF = {"sr.json": DS6, "sr-Dow.json": ["Dow"],
               "sr-100K.json": ["Tow"] + [d for d in DS6 if d != "Tow"]}
REAL_PARTNER_DIRS = [f"../gomea-results/phase1bk/phase1bk-{x}"
                     for x in ("sr", "sr-100K", "sr-Dow")]
SMOKE_PARTNER_DIRS = [f"../gomea-results/server-smoke17-1bk/server-smoke17-1bk-{x}"
                      for x in ("sr", "sr-100K", "sr-Dow")]


def has_tls(hc_tls: bool | None = None) -> bool:
    return HC_TLS if hc_tls is None else hc_tls


def arms(hc_tls: bool | None = None) -> dict:
    return {a: v for a, v in ALL_ARMS.items() if has_tls(hc_tls) or "TLS" not in a}


def arms_of(name: str, hc_tls: bool | None = None) -> list[str]:
    big = name == "sr-100K.json"
    return [a for a in arms(hc_tls) if a.endswith("-100K") == big]


def _c(base, metric, suffix=""):
    return [base[0] + suffix, *base[1:], metric]


P1 = ["basetls-vs-hclm-gain-100k-vs-20k", "BASE-TLS-100K", "HC-LM-100K", "BASE-TLS", "HC-LM"]
P2 = ["basetls100k-vs-hclm100k", "BASE-TLS-100K", "HC-LM-100K"]
P3 = ["basetls-vs-hctls", "BASE-TLS", "HC-TLS"]
P4 = ["basetls100k-vs-hctls100k", "BASE-TLS-100K", "HC-TLS-100K"]
B20 = ["basetls-vs-hclm", "BASE-TLS", "HC-LM"]


def _dow(base):
    """The Dow contrast of `base` (a per-dataset name of its own: linkage_report.check_spec)."""
    return ["dow-" + base[0], *base[1:], "test_nmse"]


def families(hc_tls: bool | None = None) -> list[dict]:
    """The "contrast_families" of the pre-registration ("Contrasts", "The Dow rule")."""
    tls = has_tls(hc_tls)
    prim = [P1, P2] + ([P3, P4] if tls else [])
    primary = [_c(p, "test_nmse") for p in prim]
    secondary = [_c(B20, "test_nmse"), ["hclm100k-vs-hclm", "HC-LM-100K", "HC-LM", "test_nmse"]]
    if tls:
        secondary += [["hctls-vs-hclm", "HC-TLS", "HC-LM", "test_nmse"],
                      ["hctls100k-vs-hclm100k", "HC-TLS-100K", "HC-LM-100K", "test_nmse"],
                      ["basetls-vs-hctls-gain-100k-vs-20k", "BASE-TLS-100K", "HC-TLS-100K",
                       "BASE-TLS", "HC-TLS", "test_nmse"]]
    secondary += [_c(p, "train_nmse", "-train") for p in prim]
    size_cpu = [_c(p, m, f"-{x}") for p in ([P2] + ([P3, P4] if tls else []) + [B20])
                for m, x in (("program_size", "size"), ("cpu_s", "cpu"))]
    out = [
        {"name": "P", "role": "primary", "contrasts": primary},
        {"name": "S", "role": "secondary", "contrasts": secondary},
        {"name": "SC", "role": "secondary", "contrasts": size_cpu},
        {"name": "D", "role": "descriptive", "per_dataset": True, "contrasts": primary},
        {"name": "DD", "role": "descriptive", "per_dataset": True, "problems": ["Dow"],
         "contrasts": [_dow(B20)] + ([_dow(P3)] if tls else [])},
    ]
    if tls:
        out.append({"name": "DP", "role": "descriptive", "contrasts": [
            ["hctls100k-vs-hctls", "HC-TLS-100K", "HC-TLS", "test_nmse"]]})
    return out


def spec(phase: str, partner_dirs: list[str], hc_tls: bool | None = None) -> dict:
    """The spec.json of 1bl (`phase` "1bl") or of its smoke (`phase` "smoke17-1bl")."""
    a = arms(hc_tls)
    return {
        "phase": phase,
        "metrics": {d: "test_nmse" for d in DS7},
        "primary": [],
        "extend": False,
        "pooled_problems": DS6,
        "partners": {"dirs": partner_dirs, "pairs": {x: PARTNER[x] for x in a}},
        "contrast_families": families(hc_tls),
        "anytime_ratios": {x: PARTNER[x] for x in a},
        "deme_table": True,
        "size_terms": True,
        "run_table": True,
    }
