import json
import sys
from collections import defaultdict
from statistics import mean, median

A = json.load(open(sys.argv[1]))
recs = A["records"]


def m(xs):
    xs = [x for x in xs if x is not None]
    return f"{mean(xs):.3f}" if xs else "-"


print("== Q1/Q4 per (problem, set, N, gen) ==")
g = defaultdict(list)
for r in recs:
    prob = r["run"].split("_")[0]
    gen = r["gen"] if r["gen"] < 4 else 4
    g[(prob, r["mode"], r["N"], gen)].append(r)
hdr = ("key n | d varying hamming uniq min1 | LTB>p95 LTB>p99 LTBtopRank LTBabsmax | MI>p95 MI>p99 "
       "MItopRank | sel: LTB>p95 MI>p95 MI>p99 varying | CB: MI>p95 MI>p99 MItopRank LTB>p95 selCB: MI>p95 MI>p99 LTB>p95 | stab")
print(hdr)
for k in sorted(g):
    L = g[k]
    sa = [r.get("sim_all") for r in L]
    ss = [r.get("sim_sel") for r in L]
    sc = [r.get("sim_all_cb") for r in L]
    scs = [r.get("sim_sel_cb") for r in L]
    print(k, len(L), "|",
          m([r["div"]["d_linked"] for r in L]), m([r["div"]["n_varying"] for r in L]),
          m([r["div"]["mean_hamming"] for r in L]), m([r["div"]["unique_genotypes"] for r in L]),
          m([r["div"]["share_minority_eq1"] for r in L]), "|",
          m([s and s["ltb"]["share_gt_p95"] for s in sa]), m([s and s["ltb"]["share_gt_p99"] for s in sa]),
          m([s and s["ltb"]["top_vs_nullmax_rank"] for s in sa]), m([s and s["ltb"]["abs_max_all"] for s in sa]), "|",
          m([s and s["mi"]["share_gt_p95"] for s in sa]), m([s and s["mi"]["share_gt_p99"] for s in sa]),
          m([s and s["mi"]["top_vs_nullmax_rank"] for s in sa]), "|",
          m([s and s["ltb"]["share_gt_p95"] for s in ss]), m([s and s["mi"]["share_gt_p95"] for s in ss]),
          m([s and s["mi"]["share_gt_p99"] for s in ss]),
          m([r.get("div_sel", {}).get("n_varying") for r in L]), "| CB:",
          m([s and s["mi"]["share_gt_p95"] for s in sc]), m([s and s["mi"]["share_gt_p99"] for s in sc]),
          m([s and s["mi"]["top_vs_nullmax_rank"] for s in sc]),
          m([s and s["ltb"]["share_gt_p95"] for s in sc]), "selCB:",
          m([s and s["mi"]["share_gt_p95"] for s in scs]), m([s and s["mi"]["share_gt_p99"] for s in scs]),
          m([s and s["ltb"]["share_gt_p95"] for s in scs]), "|",
          m([r.get("stab_retained") for r in L]))

print("\n== Q2 GOM attribution (group -> bucket: trials, noop%, memo%, eval, improve, improve share) ==")
for grp, d in sorted(A["att"].items()):
    tot_imp = sum(v.get("improve", 0) for v in d.values())
    tot_tr = sum(v.get("trials", 0) for v in d.values())
    tot_noop = sum(v.get("noop", 0) for v in d.values())
    tot_memo = sum(v.get("memo", 0) for v in d.values())
    tot_eval = sum(v.get("eval", 0) for v in d.values())
    print(f"{grp}: trials={tot_tr} noop={tot_noop/tot_tr:.3f} memo={tot_memo/tot_tr:.3f} "
          f"scored={tot_eval/tot_tr:.3f} improve={tot_imp/tot_tr:.4f} (n={tot_imp})")
    for b in ("1", "2-3", "4-15", "16+"):
        v = d.get(b)
        if not v:
            continue
        ev = v["eval"] + v["memo"]
        print(f"   {b:>5}: trials={v['trials']:6d} noop={v['noop']/v['trials']:.3f} "
              f"memo={v['memo']/v['trials']:.3f} scored={v['eval']/v['trials']:.3f} "
              f"improve/trial={v['improve']/v['trials']:.4f} improve/eval={v['improve']/max(1,ev):.3f} "
              f"share_of_impr={v['improve']/max(1,tot_imp):.3f} meanchg={v['changed_pos_sum']/max(1,v['trials']-v['noop']):.2f}")
print("\n== Q2 improvements by number of positions actually changed ==")
for grp, d in sorted(A["imp"].items()):
    n = d.get("n", 0)
    if n:
        print(f"{grp}: n={n} changed1={d['nchg1']/n:.3f} changed2-3={d['nchg2_3']/n:.3f} "
              f"changed4+={d['nchg4p']/n:.3f} near_root_subset={d['near_root']/n:.3f}")
print("\n== Q3 epistasis ==")
for grp, d in sorted(A["epi"].items()):
    for k, v in sorted(d.items()):
        n = v["n"]
        eff = v["pen_any_effect"]
        print(f"{grp} {k:8s} probes={n} any_effect={eff/n:.3f} pen_epi={v['pen_epi']/n:.3f} "
              f"raw_epi={v['raw_epi']/n:.3f} epi|effect={v['pen_epi_given_effect']/max(1,eff):.3f} "
              f"joint_only_gain(pen)={v['pen_joint_only_gain']/n:.3f} (raw)={v['raw_joint_only_gain']/n:.3f} "
              f"donor_avail={v['donor_avail']/n:.3f}")
