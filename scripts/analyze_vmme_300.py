#!/usr/bin/env python
"""Merge ALL Video-MME results (existing dirs + new 300-run shards), dedup by
record_key, and report overall + per-duration accuracy and token savings."""
import json, glob, math, statistics as st, os
from collections import defaultdict
import pandas as pd

ROOT = "/root/autodl-tmp/MTEC"
META = os.path.join(ROOT, "data/datasets/video-mme/videomme/test-00000-of-00001.parquet")
EXISTING_DIRS = ["eval100_baseline_20260622", "new50_oss_20260623", "new50_redo_20260623",
                 "payload3_oss_20260623", "err6_oss_20260623", "ab_baseline_20260622"]


def load_all():
    files = []
    for d in EXISTING_DIRS:
        files += glob.glob(os.path.join(ROOT, "outputs", d, "*results.jsonl"))
    files += glob.glob(os.path.join(ROOT, "outputs/vmme300_shards/s*/*results.jsonl"))
    recs = []
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if str(r.get("record_key", "")).startswith("video:"):
                    recs.append(r)
            except Exception:
                pass
    byk = defaultdict(list)
    for r in recs:
        byk[r["record_key"]].append(r)
    out = {}
    for k, v in byk.items():
        c = [x for x in v if x.get("status") == "completed"]
        if c:
            out[k] = c[0]
    return out


def report(comp, k2dur):
    n = len(comp)
    cor = [r for r in comp.values() if r.get("correct")]
    p = len(cor) / n
    ci = 1.96 * math.sqrt(p * (1 - p) / n)
    print("TOTAL Video-MME completed=%d  ACCURACY=%d/%d=%.1f%% (95%% CI +/-%.1f)" %
          (n, len(cor), n, p * 100, ci * 100))
    # per duration
    for dur in ["short", "medium", "long"]:
        sub = {k: r for k, r in comp.items() if k2dur.get(k) == dur}
        if not sub:
            continue
        c = sum(1 for r in sub.values() if r.get("correct"))
        tr = [r["token_saving_ratio_raw"] for r in sub.values() if isinstance(r.get("token_saving_ratio_raw"), (int, float))]
        pos = [x for x in tr if x > 0]
        print("  %-7s n=%3d acc=%d/%d=%.1f%%  token_save raw mean=%.1f%% median=%.1f%% pos=%d/%d" %
              (dur, len(sub), c, len(sub), 100*c/len(sub),
               st.mean(tr)*100 if tr else 0, st.median(tr)*100 if tr else 0, len(pos), len(tr)))
    tr = [r["token_saving_ratio_raw"] for r in comp.values() if isinstance(r.get("token_saving_ratio_raw"), (int, float))]
    pos = [x for x in tr if x > 0]
    print("OVERALL token_saving raw: mean=%.1f%% median=%.1f%% positive-only mean=%.1f%% (pos=%d/%d)" %
          (st.mean(tr)*100, st.median(tr)*100, (st.mean(pos)*100 if pos else 0), len(pos), len(tr)))


def main():
    meta = pd.read_parquet(META)
    k2dur = {f"video:{r.videoID}:{r.question_id}": r.duration for r in meta.itertuples()}
    comp = load_all()
    report(comp, k2dur)


if __name__ == "__main__":
    main()
