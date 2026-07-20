#!/usr/bin/env python
"""Symmetric majority-vote over the 100 Video-MME long records.
Loads run1 (from the merged 300 set), run2, and (if present) run3 tie-breaks,
then computes maj@3 correctness per record. Reports flips both directions and
the unbiased updated long-split accuracy + token savings.
"""
import json, glob, os, statistics as st
import pandas as pd

ROOT = "/root/autodl-tmp/MTEC"
META = os.path.join(ROOT, "data/datasets/video-mme/videomme/test-00000-of-00001.parquet")
EXISTING = ["eval100_baseline_20260622", "new50_oss_20260623", "new50_redo_20260623",
            "payload3_oss_20260623", "err6_oss_20260623", "ab_baseline_20260622"]


def load(globs):
    recs = {}
    for g in globs:
        for f in glob.glob(g):
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("status") == "completed" and str(r.get("record_key", "")).startswith("video:"):
                        recs[r["record_key"]] = r
                except Exception:
                    pass
    return recs


def main():
    meta = pd.read_parquet(META)
    k2dur = {f"video:{r.videoID}:{r.question_id}": r.duration for r in meta.itertuples()}

    run1_all = load([os.path.join(ROOT, "outputs", d, "*results.jsonl") for d in EXISTING] +
                    [os.path.join(ROOT, "outputs/vmme300_shards/s*/*results.jsonl")])
    longkeys = [k for k in run1_all if k2dur.get(k) == "long"]
    run2 = load([os.path.join(ROOT, "outputs/vmme_long_run2/s*/*results.jsonl")])
    run3 = load([os.path.join(ROOT, "outputs/vmme_long_run3/s*/*results.jsonl")])

    print("long=%d run2=%d run3=%d" % (len(longkeys), len(run2), len(run3)))

    maj = {}
    flips_up, flips_down, still_wrong, unstable = 0, 0, 0, 0
    for k in longkeys:
        votes = [bool(run1_all[k].get("correct"))]
        if k in run2:
            votes.append(bool(run2[k].get("correct")))
        if k in run3:
            votes.append(bool(run3[k].get("correct")))
        # majority of available votes (1, 2, or 3); tie (2 votes split) -> keep run1
        if len(votes) >= 2 and votes.count(True) != votes.count(False):
            m = votes.count(True) > votes.count(False)
        else:
            m = votes[0]
        maj[k] = m
        r1 = bool(run1_all[k].get("correct"))
        if len(set(votes)) > 1:
            unstable += 1
        if (not r1) and m:
            flips_up += 1
        elif r1 and (not m):
            flips_down += 1
        elif not r1 and not m:
            still_wrong += 1

    n = len(longkeys)
    r1c = sum(bool(run1_all[k].get("correct")) for k in longkeys)
    majc = sum(maj.values())
    print("run1 long accuracy:  %d/%d = %.1f%%" % (r1c, n, 100*r1c/n))
    print("maj@k long accuracy: %d/%d = %.1f%%" % (majc, n, 100*majc/n))
    print("nondeterministic (vote disagreed): %d/%d" % (unstable, n))
    print("flips wrong->right: %d   right->wrong: %d   net recovered: %+d" %
          (flips_up, flips_down, flips_up - flips_down))
    # token savings unchanged (same media); report from run1 long
    tr = [run1_all[k].get("token_saving_ratio_raw") for k in longkeys
          if isinstance(run1_all[k].get("token_saving_ratio_raw"), (int, float))]
    print("long token_saving raw mean=%.1f%% median=%.1f%%" % (st.mean(tr)*100, st.median(tr)*100))
    json.dump({k: maj[k] for k in longkeys}, open(os.path.join(ROOT, "outputs/vmme_long_maj.json"), "w"))


if __name__ == "__main__":
    main()
