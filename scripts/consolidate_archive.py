#!/usr/bin/env python3
"""Consolidate ALL experiment runs into a timestamped paper archive:
 - copy each run's per-record results.jsonl into archive/<dataset>/
 - compute per-dataset summary (accuracy, token savings, failure breakdown, config)
 - write MASTER_SUMMARY.json + MASTER_SUMMARY.md
"""
import os, json, glob, math, shutil, statistics as st
from collections import Counter, defaultdict
import pandas as pd

ROOT = "/root/autodl-tmp/MTEC"
STAMP = "20260625"
ARCH = os.path.join(ROOT, "outputs", "PAPER_ARCHIVE_%s" % STAMP)
VMETA = os.path.join(ROOT, "data/datasets/video-mme/videomme/test-00000-of-00001.parquet")

# duration map for Video-MME records
_vm = pd.read_parquet(VMETA)
K2DUR = {f"video:{r.videoID}:{r.question_id}": r.duration for r in _vm.itertuples()}

# (label, archive_subdir, [result globs], platform, model, mode, notes, videomme_breakdown)
SETS = [
    ("Video-MME 300 (compressed)", "videomme_300_compressed",
     ["outputs/eval100_baseline_20260622/*results.jsonl", "outputs/new50_oss_20260623/*results.jsonl",
      "outputs/new50_redo_20260623/*results.jsonl", "outputs/payload3_oss_20260623/*results.jsonl",
      "outputs/err6_oss_20260623/*results.jsonl", "outputs/ab_baseline_20260622/*results.jsonl",
      "outputs/vmme300_shards/s*/*results.jsonl"],
     "bailian", "qwen3.7-plus", "MTEC compressed (anchors+evidence, OSS)", "balanced 100/100/100", True),
    ("NExT-QA 300 (compressed)", "nextqa_300_compressed",
     ["outputs/eval_nextqa_300_20260624/*results.jsonl"], "bailian", "qwen3.7-plus",
     "MTEC compressed", "video embedded in parquet", False),
    ("TempCompass 300 (compressed)", "tempcompass_300_compressed",
     ["outputs/eval_tempcompass_300_20260624/*results.jsonl"], "bailian", "qwen3.7-plus",
     "MTEC compressed", "short clips; dedup of double-process artifact", False),
    ("MLVU-long 180 (compressed)", "mlvu_long_180_compressed",
     ["outputs/eval_mlvu_long_20260625/s*/*results.jsonl"], "bailian", "qwen3.7-plus",
     "MTEC compressed (streamed)", "10-60min videos", False),
    ("Video-MME short/med 105 (raw direct)", "videomme_105_raw_bailian",
     ["outputs/direct_raw_qwen37_105_20260624/*results.jsonl"], "bailian", "qwen3.7-plus",
     "raw video via OSS (no compression)", "early raw probe", False),
    ("Video-MME long 100 (raw, Bailian pool)", "videomme_long_raw_bailian_pool",
     ["outputs/direct_raw_long_qwen36plus_20260625/s*/results.jsonl", "outputs/direct_raw_long_pool_20260625/s*/results.jsonl"],
     "bailian", "qwen3.6-plus + pool", "raw video via OSS", "mixed-model pool; quota-limited", False),
    ("NExT-QA 300 (raw, SiliconFlow)", "nextqa_300_raw_siliconflow",
     ["outputs/raw_sf_nextqa/s*/results.jsonl"], "siliconflow", "Qwen3-VL-32B-Instruct",
     "raw video base64", "small videos", False),
    ("Video-MME medium 102 (raw, SiliconFlow)", "videomme_medium_raw_siliconflow",
     ["outputs/raw_sf_videomme_medium/s*/results.jsonl"], "siliconflow", "Qwen3-VL-32B-Instruct",
     "raw video base64", "many exceed ~50-80MB base64 ceiling", False),
    ("MLVU-long 180 (raw, SiliconFlow)", "mlvu_long_raw_siliconflow",
     ["outputs/raw_sf_mlvu_long/s*/results.jsonl"], "siliconflow", "Qwen3-VL-32B-Instruct",
     "raw video base64", "most exceed base64 ceiling -> video_too_large", False),
    ("MLVU-long 141-rerun (raw, Bailian OSS)", "mlvu_long_141rerun_raw_bailian",
     ["outputs/raw_bailian_mlvu_long_20260625/s*/results.jsonl"], "bailian", "qwen3.6-plus",
     "raw video via OSS; the 141 SF-failed videos", "PARTIAL/in-progress; platform comparison", False),
]


def load(globs):
    d = {}
    for g in globs:
        for f in glob.glob(os.path.join(ROOT, g)):
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                k = r.get("record_key")
                if k is None:
                    continue
                if k not in d or r.get("status") == "completed":
                    d[k] = r
    return d


def summ(d, vbreak):
    comp = [r for r in d.values() if r.get("status") == "completed"]
    cor = [r for r in comp if r.get("correct")]
    fails = [r for r in d.values() if r.get("status") != "completed"]
    n = len(d)
    out = {"n_records": n, "completed": len(comp), "failed": len(fails), "correct": len(cor),
           "acc_over_completed_pct": round(100 * len(cor) / len(comp), 1) if comp else None,
           "acc_incl_failures_pct": round(100 * len(cor) / n, 1) if n else None,
           "failure_classes": dict(Counter(r.get("fail_class") or "(none)" for r in fails))}
    tr = [r["token_saving_ratio_raw"] for r in comp if isinstance(r.get("token_saving_ratio_raw"), (int, float))]
    if tr:
        pos = [x for x in tr if x > 0]
        out["token_saving_raw_mean_pct"] = round(st.mean(tr) * 100, 1)
        out["token_saving_raw_median_pct"] = round(st.median(tr) * 100, 1)
        out["token_saving_positive_count"] = "%d/%d" % (len(pos), len(tr))
    if vbreak:
        bd = {}
        for dur in ["short", "medium", "long"]:
            sub = {k: r for k, r in d.items() if K2DUR.get(k) == dur and r.get("status") == "completed"}
            if sub:
                c = sum(1 for r in sub.values() if r.get("correct"))
                strr = [r["token_saving_ratio_raw"] for r in sub.values() if isinstance(r.get("token_saving_ratio_raw"), (int, float))]
                bd[dur] = {"n": len(sub), "acc_pct": round(100 * c / len(sub), 1),
                           "token_saving_median_pct": round(st.median(strr) * 100, 1) if strr else None}
        out["duration_breakdown"] = bd
    return out


def main():
    if os.path.exists(ARCH):
        shutil.rmtree(ARCH)
    os.makedirs(ARCH)
    master = {"archive": os.path.basename(ARCH), "generated": STAMP, "datasets": []}
    for label, sub, globs, platform, model, mode, notes, vbreak in SETS:
        d = load(globs)
        if not d:
            print("  (skip, no data) %s" % label)
            continue
        ddir = os.path.join(ARCH, sub)
        os.makedirs(ddir)
        # write deduped per-record results
        with open(os.path.join(ddir, "results.jsonl"), "w") as f:
            for r in d.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        s = summ(d, vbreak)
        cfg = {"label": label, "platform": platform, "model": model, "mode": mode, "notes": notes,
               "source_globs": globs}
        json.dump({"config": cfg, "summary": s}, open(os.path.join(ddir, "summary.json"), "w"), indent=2)
        master["datasets"].append({"label": label, "dir": sub, "platform": platform, "model": model,
                                   "mode": mode, "notes": notes, **s})
        print("  archived %-44s n=%d completed=%d acc_compl=%s" % (label, s["n_records"], s["completed"], s["acc_over_completed_pct"]))

    json.dump(master, open(os.path.join(ARCH, "MASTER_SUMMARY.json"), "w"), indent=2)

    # markdown
    L = ["# MTEC Experiment Archive — %s\n" % STAMP,
         "Consolidated per-record results + metrics for all runs. Each `<dataset>/` holds the deduped",
         "`results.jsonl` (per-record logs) and a `summary.json` (config + metrics).\n",
         "## Master metrics table\n",
         "| Dataset | Platform | Model | n | Completed | Acc (compl.) | Acc (incl. fail) | Token save (med raw) | Top failures |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m in master["datasets"]:
        ts = m.get("token_saving_raw_median_pct")
        ts = ("%.1f%%" % ts) if ts is not None else "—"
        fails = m.get("failure_classes") or {}
        topf = ", ".join("%s:%d" % (k, v) for k, v in sorted(fails.items(), key=lambda x: -x[1]) if k != "(none)")[:60] or "—"
        L.append("| %s | %s | %s | %d | %d | %s | %s | %s | %s |" % (
            m["label"], m["platform"], m["model"], m["n_records"], m["completed"],
            ("%.1f%%" % m["acc_over_completed_pct"]) if m["acc_over_completed_pct"] is not None else "—",
            ("%.1f%%" % m["acc_incl_failures_pct"]) if m["acc_incl_failures_pct"] is not None else "—",
            ts, topf))
    # Video-MME duration breakdown
    for m in master["datasets"]:
        if m.get("duration_breakdown"):
            L.append("\n### %s — duration breakdown\n" % m["label"])
            L.append("| Split | n | Accuracy | Token save (median raw) |\n|---|---|---|---|")
            for dur in ["short", "medium", "long"]:
                b = m["duration_breakdown"].get(dur)
                if b:
                    L.append("| %s | %d | %.1f%% | %s |" % (dur, b["n"], b["acc_pct"],
                             ("%.1f%%" % b["token_saving_median_pct"]) if b["token_saving_median_pct"] is not None else "—"))
    L.append("\n## Notes / caveats per run\n")
    for m in master["datasets"]:
        L.append("- **%s** (`%s/`): %s. Failures: %s" % (m["label"], m["dir"], m["notes"], m.get("failure_classes")))
    open(os.path.join(ARCH, "MASTER_SUMMARY.md"), "w").write("\n".join(L) + "\n")
    print("\nWROTE", ARCH)


if __name__ == "__main__":
    main()
