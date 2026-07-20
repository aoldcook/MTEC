#!/usr/bin/env python
"""Merge-and-compare the expanded long-video ablation: Config C (global-timeline
off) vs Config A on the COMBINED long sample = original 30 MLVU + new MLVU_ext +
new Video-MME-long. Paired on shared completed keys per source. No dependence on
the fixed pruned-subset files -- uses whatever keys are present in both runs.

Usage: analyze_long_expanded.py <CONFIG_LABEL> <BASELINE_LABEL>   (e.g. C_no_timeline A_full)
"""
import os
import sys
import json
import glob
import statistics

ROOT = "/root/autodl-tmp/MTEC"
RUNSDIR = os.path.join(ROOT, "outputs/ablation_20260701/runs")

# source -> list of run subdirs (globbed for s*/results.jsonl)
SOURCES = {
    "MLVU (orig+ext)": ["mlvu", "mlvu_ext"],
    "Video-MME-long":  ["vmme_long"],
}


def load(label, subdirs):
    recs = {}
    for sd in subdirs:
        for f in glob.glob(os.path.join(RUNSDIR, label, sd, "s*",
                                        "modelscope_mtec_anchor_api_full_results.jsonl")):
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                k = r.get("record_key")
                if r.get("status") == "completed":
                    recs[k] = r
                elif k not in recs:
                    recs[k] = r
    return recs


def med_tsr(recs):
    v = [r.get("token_saving_ratio_raw") for r in recs
         if r.get("status") == "completed" and isinstance(r.get("token_saving_ratio_raw"), (int, float))]
    return statistics.median(v) * 100 if v else None


def main():
    cfg_label, bl_label = sys.argv[1], sys.argv[2]
    print("=" * 74)
    print("EXPANDED LONG-VIDEO: %s vs %s (paired on shared completed keys)" % (cfg_label, bl_label))
    print("=" * 74)

    tot_c = tot_bc = tot_n = 0
    for src, subdirs in SOURCES.items():
        cfg = load(cfg_label, subdirs)
        bl = load(bl_label, subdirs)
        cfg_c = {k: r for k, r in cfg.items() if r.get("status") == "completed"}
        bl_c = {k: r for k, r in bl.items() if r.get("status") == "completed"}
        shared = sorted(set(cfg_c) & set(bl_c))
        if not shared:
            print("\n[%s] no shared completed keys yet (cfg_completed=%d baseline_completed=%d)" % (
                src, len(cfg_c), len(bl_c)))
            continue
        cc = sum(1 for k in shared if cfg_c[k].get("correct") is True)
        bc = sum(1 for k in shared if bl_c[k].get("correct") is True)
        dacc = (cc - bc) / len(shared) * 100
        ct = med_tsr(list(cfg_c.values())); bt = med_tsr(list(bl_c.values()))
        print("\n[%s] paired n=%d : %s %.1f%% (%d) vs %s %.1f%% (%d) -> dAcc %+.1f pts" % (
            src, len(shared), cfg_label, cc / len(shared) * 100, cc,
            bl_label, bc / len(shared) * 100, bc, dacc))
        print("      completion: %s=%d  %s=%d ; token-save med %s vs %s" % (
            cfg_label, len(cfg_c), bl_label, len(bl_c),
            ("%.1f%%" % ct) if ct else "n/a", ("%.1f%%" % bt) if bt else "n/a"))
        tot_c += cc; tot_bc += bc; tot_n += len(shared)

    if tot_n:
        print("\n" + "-" * 74)
        print("COMBINED LONG (n=%d): %s %.1f%% vs %s %.1f%% -> dAcc %+.1f pts" % (
            tot_n, cfg_label, tot_c / tot_n * 100, bl_label, tot_bc / tot_n * 100,
            (tot_c - tot_bc) / tot_n * 100))
        # simple ±95% CI half-width for the paired accuracy difference (normal approx)
        import math
        p = tot_c / tot_n
        se = math.sqrt(p * (1 - p) / tot_n) * 100
        print("      (per-config 95%% CI half-width ~ %.1f pts at n=%d)" % (1.96 * se, tot_n))


if __name__ == "__main__":
    main()
