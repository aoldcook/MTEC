#!/usr/bin/env python
"""Merged Config C long-video calculation with explicit record accounting
(no double counting) + paired C-vs-A on shared completed keys."""
import json
import glob

R = "/root/autodl-tmp/MTEC/outputs/ablation_20260701/runs"


def load(label, sub):
    recs = {}
    for f in glob.glob("%s/%s/%s/s*/modelscope_mtec_anchor_api_full_results.jsonl" % (R, label, sub)):
        for l in open(f):
            l = l.strip()
            if not l:
                continue
            try:
                r = json.loads(l)
            except Exception:
                continue
            k = r.get("record_key")
            if r.get("status") == "completed":
                recs[k] = r
            elif k not in recs:
                recs[k] = r
    return {k: v for k, v in recs.items() if v.get("status") == "completed"}


def acc(d):
    c = sum(1 for r in d.values() if r.get("correct") is True)
    return c, len(d), (100.0 * c / len(d) if d else 0.0)


c_mlvu = load("C_no_timeline", "mlvu")
c_ext = load("C_no_timeline", "mlvu_ext")
c_vl = load("C_no_timeline", "vmme_long")
a_mlvu = load("A_full", "mlvu")
a_ext = load("A_full", "mlvu_ext")
a_vl = load("A_full", "vmme_long")

print("=== Config C completed long records (merged, deduped, no double-count) ===")
for nm, d in [("MLVU original", c_mlvu), ("MLVU ext (new)", c_ext), ("Video-MME-long (new)", c_vl)]:
    cc, n, a = acc(d)
    print("  %-22s: %d/%d = %.1f%%" % (nm, cc, n, a))
c_all = {**c_mlvu, **c_ext, **c_vl}
cc, n, a = acc(c_all)
print("  %-22s: %d/%d = %.1f%%" % ("TOTAL C long", cc, n, a))

print("\n=== Config A completed long records (baseline; A ext paused) ===")
for nm, d in [("MLVU original", a_mlvu), ("MLVU ext (new)", a_ext), ("Video-MME-long (new)", a_vl)]:
    cc, n, a = acc(d)
    print("  %-22s: %d/%d = %.1f%%" % (nm, cc, n, a))
a_all = {**a_mlvu, **a_ext, **a_vl}
cc, n, a = acc(a_all)
print("  %-22s: %d/%d = %.1f%%" % ("TOTAL A long", cc, n, a))

print("\n=== PAIRED C-vs-A (shared completed keys only = valid delta) ===")
tot_c = tot_a = tot_n = 0
for nm, cd, ad in [("MLVU", {**c_mlvu, **c_ext}, {**a_mlvu, **a_ext}), ("Video-MME-long", c_vl, a_vl)]:
    sh = sorted(set(cd) & set(ad))
    cc = sum(1 for k in sh if cd[k].get("correct") is True)
    ac = sum(1 for k in sh if ad[k].get("correct") is True)
    if sh:
        print("  %-16s n=%3d: C %.1f%% vs A %.1f%% -> %+.1f pts" % (
            nm, len(sh), 100.0 * cc / len(sh), 100.0 * ac / len(sh), 100.0 * (cc - ac) / len(sh)))
    tot_c += cc
    tot_a += ac
    tot_n += len(sh)
print("  %-16s n=%3d: C %.1f%% vs A %.1f%% -> %+.1f pts" % (
    "COMBINED", tot_n, 100.0 * tot_c / tot_n, 100.0 * tot_a / tot_n, 100.0 * (tot_c - tot_a) / tot_n))
