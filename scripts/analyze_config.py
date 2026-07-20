#!/usr/bin/env python
"""Analyze one ablation config vs a Config A baseline, per dataset.

For each dataset the config touched: config accuracy, completion rate, median
token-save, and the delta vs Config A computed on the SHARED completed keys
(paired comparison).

Usage: analyze_config.py <CONFIG_LABEL> [BASELINE]
  BASELINE = "archive" (default) uses PAPER_ARCHIVE compressed results.
  BASELINE = "A_full" (or any config label) uses that fresh run under runs/ as
             the baseline -- same-run, same-code, removes run-to-run drift.
"""
import os
import sys
import json
import glob
import statistics

ROOT = "/root/autodl-tmp/MTEC"
RUNSDIR = os.path.join(ROOT, "outputs/ablation_20260701/runs")
KEYSDIR = os.path.join(ROOT, "outputs/ablation_20260701/keys")
ARCH = os.path.join(ROOT, "outputs/PAPER_ARCHIVE_20260625")

ARCHIVE_FILE = {
    "vmme": os.path.join(ARCH, "videomme_300_compressed/results.jsonl"),
    "nextqa": os.path.join(ARCH, "nextqa_300_compressed/results.jsonl"),
    "mlvu": os.path.join(ARCH, "mlvu_long_180_compressed/results.jsonl"),
    "vmme_long": os.path.join(ARCH, "videomme_300_compressed/results.jsonl"),
}
LONGEXT_KEYSDIR = os.path.join(ROOT, "outputs/ablation_20260701/keys_long_ext")
SUBSET_KEYS = {
    "vmme": lambda: set(open(os.path.join(KEYSDIR, "sf_vmme_short.txt")).read().split()
                        + open(os.path.join(KEYSDIR, "sf_vmme_medium.txt")).read().split()),
    "nextqa": lambda: set(open(os.path.join(KEYSDIR, "sf_nextqa.txt")).read().split()),
    "mlvu": lambda: set(open(os.path.join(KEYSDIR, "bailian_mlvu_long.txt")).read().split()),
    "vmme_long": lambda: set(open(os.path.join(LONGEXT_KEYSDIR, "vmme_long.txt")).read().split()),
}


def load_records(files):
    """key -> record; prefer completed, else last-seen."""
    recs = {}
    for f in files:
        if not os.path.exists(f):
            continue
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


def median_tsr(recs):
    vals = [r.get("token_saving_ratio_raw") for r in recs
            if r.get("status") == "completed" and isinstance(r.get("token_saving_ratio_raw"), (int, float))]
    return statistics.median(vals) * 100 if vals else None


def baseline_files(ds, baseline):
    if baseline == "archive":
        return [ARCHIVE_FILE[ds]]
    return glob.glob(os.path.join(RUNSDIR, baseline, ds, "s*",
                                  "modelscope_mtec_anchor_api_full_results.jsonl"))


def main():
    label = sys.argv[1]
    baseline = sys.argv[2] if len(sys.argv) > 2 else "archive"
    cfg_dir = os.path.join(RUNSDIR, label)
    bl_name = "archived Config A" if baseline == "archive" else ("Config A (fresh: %s)" % baseline)
    print("=" * 72)
    print("ABLATION CONFIG %s  vs  %s  [Bailian qwen3.7-plus]" % (label, bl_name))
    print("=" * 72)

    grand = {"cfg_c": 0, "cfg_n": 0, "paired_cfg_c": 0, "paired_a_c": 0, "paired_n": 0}

    for ds in ["vmme", "nextqa", "mlvu", "vmme_long"]:
        files = glob.glob(os.path.join(cfg_dir, ds, "s*", "modelscope_mtec_anchor_api_full_results.jsonl"))
        if not files:
            continue
        subset = SUBSET_KEYS[ds]()
        cfg = {k: v for k, v in load_records(files).items() if k in subset}
        arch = {k: v for k, v in load_records(baseline_files(ds, baseline)).items() if k in subset}

        cfg_completed = {k: r for k, r in cfg.items() if r.get("status") == "completed"}
        cfg_correct = sum(1 for r in cfg_completed.values() if r.get("correct") is True)
        cfg_acc = (cfg_correct / len(cfg_completed) * 100) if cfg_completed else 0.0
        comp_rate = (len(cfg_completed) / len(cfg) * 100) if cfg else 0.0
        cfg_tsr = median_tsr(list(cfg.values()))

        # paired: keys completed in BOTH config and archive
        arch_completed = {k: r for k, r in arch.items() if r.get("status") == "completed"}
        shared = sorted(set(cfg_completed) & set(arch_completed))
        pc = sum(1 for k in shared if cfg_completed[k].get("correct") is True)
        pa = sum(1 for k in shared if arch_completed[k].get("correct") is True)
        arch_tsr = median_tsr([arch_completed[k] for k in shared]) if shared else None

        print("\n[%s] config: completed %d/%d (%.0f%% completion), acc %.1f%% (%d/%d), median_token_save %s" % (
            ds, len(cfg_completed), len(cfg), comp_rate, cfg_acc, cfg_correct, len(cfg_completed),
            ("%.1f%%" % cfg_tsr) if cfg_tsr is not None else "n/a"))
        if shared:
            d_acc = (pc - pa) / len(shared) * 100
            print("      paired vs A on %d shared keys: config %.1f%% (%d) vs A %.1f%% (%d) -> dAcc %+.1f pts" % (
                len(shared), pc / len(shared) * 100, pc, pa / len(shared) * 100, pa, d_acc))
            if cfg_tsr is not None and arch_tsr is not None:
                print("      token-save median: config %.1f%% vs A %.1f%% -> d %+.1f pts" % (
                    cfg_tsr, arch_tsr, cfg_tsr - arch_tsr))
            grand["paired_cfg_c"] += pc
            grand["paired_a_c"] += pa
            grand["paired_n"] += len(shared)
        grand["cfg_c"] += cfg_correct
        grand["cfg_n"] += len(cfg_completed)

    if grand["paired_n"]:
        print("\n" + "-" * 72)
        d = (grand["paired_cfg_c"] - grand["paired_a_c"]) / grand["paired_n"] * 100
        print("OVERALL paired (n=%d shared): config %.1f%% vs A %.1f%% -> dAcc %+.1f pts" % (
            grand["paired_n"], grand["paired_cfg_c"] / grand["paired_n"] * 100,
            grand["paired_a_c"] / grand["paired_n"] * 100, d))


if __name__ == "__main__":
    main()
