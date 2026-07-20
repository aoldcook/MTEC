#!/usr/bin/env python
"""Summarize accuracy + token savings for the new-dataset eval runs.
Usage: python analyze_new_eval.py <name>:<dir> [<name>:<dir> ...]
"""
import sys, json, glob, statistics as st


def load(d):
    fs = glob.glob(f"{d}/*results.jsonl")
    recs = []
    for f in fs:
        for line in open(f):
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def summarize(name, d):
    recs = load(d)
    completed = [r for r in recs if r.get("status") == "completed"]
    failed = [r for r in recs if r.get("status") not in ("completed", None)]
    correct = [r for r in completed if r.get("correct") is True]
    print(f"\n===== {name}  ({d}) =====")
    print(f"records={len(recs)} completed={len(completed)} failed={len(failed)} correct={len(correct)}")
    if completed:
        acc = len(correct) / len(completed)
        # Wilson 95% CI half-width approx
        import math
        n = len(completed); p = acc
        ci = 1.96 * math.sqrt(p * (1 - p) / n)
        print(f"ACCURACY = {len(correct)}/{len(completed)} = {acc*100:.1f}%  (95% CI +/-{ci*100:.1f})")
    # token savings (use token_saving_ratio fields)
    tsr = [r.get("token_saving_ratio") for r in completed if isinstance(r.get("token_saving_ratio"), (int, float))]
    tsr_raw = [r.get("token_saving_ratio_raw") for r in completed if isinstance(r.get("token_saving_ratio_raw"), (int, float))]
    if tsr:
        pos = [x for x in tsr if x > 0]
        print(f"TOKEN_SAVING (clamped>=0): mean={st.mean(tsr)*100:.1f}%  median={st.median(tsr)*100:.1f}%  positive-only mean={ (st.mean(pos)*100 if pos else 0):.1f}%  (#pos={len(pos)}/{len(tsr)})")
    if tsr_raw:
        print(f"TOKEN_SAVING (raw, can be negative): mean={st.mean(tsr_raw)*100:.1f}%  median={st.median(tsr_raw)*100:.1f}%")
    # per-category accuracy
    cats = {}
    for r in completed:
        c = r.get("task_type") or r.get("sub_category") or "?"
        cats.setdefault(c, [0, 0])
        cats[c][1] += 1
        if r.get("correct") is True:
            cats[c][0] += 1
    if len(cats) > 1:
        print("per-category accuracy:")
        for c, (cor, tot) in sorted(cats.items()):
            print(f"   {c:20s} {cor}/{tot} = {cor/tot*100:.1f}%")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        name, d = arg.split(":", 1)
        summarize(name, d)
