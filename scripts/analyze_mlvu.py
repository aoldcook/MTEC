#!/usr/bin/env python
"""Merge MLVU shard outputs, dedup, and report long-video accuracy + token savings."""
import json, glob, math, statistics as st
from collections import defaultdict

ROOT = "/root/autodl-tmp/MTEC/outputs/eval_mlvu_long_20260625"


def load():
    recs = []
    for f in glob.glob(ROOT + "/s*/*results.jsonl") + glob.glob(ROOT + "/*results.jsonl"):
        for line in open(f):
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
    byk = defaultdict(list)
    for r in recs:
        byk[r.get("record_key")].append(r)
    out = []
    for k, v in byk.items():
        c = [x for x in v if x.get("status") == "completed"]
        out.append(c[0] if c else v[-1])
    return out


def main():
    u = load()
    comp = [r for r in u if r.get("status") == "completed"]
    fail = [r for r in u if r.get("status") != "completed"]
    cor = [r for r in comp if r.get("correct") is True]
    n = len(comp)
    print("MLVU LONG: unique=%d completed=%d failed=%d" % (len(u), n, len(fail)))
    if n:
        p = len(cor) / n
        ci = 1.96 * math.sqrt(p * (1 - p) / n)
        print("ACCURACY = %d/%d = %.1f%%  (95%% CI +/-%.1f)" % (len(cor), n, p * 100, ci * 100))
    tr = [r["token_saving_ratio_raw"] for r in comp if isinstance(r.get("token_saving_ratio_raw"), (int, float))]
    if tr:
        pos = [x for x in tr if x > 0]
        print("TOKEN_SAVING raw: mean=%.1f%% median=%.1f%% min=%.1f%% max=%.1f%%  positive=%d/%d" %
              (st.mean(tr) * 100, st.median(tr) * 100, min(tr) * 100, max(tr) * 100, len(pos), len(tr)))
    ot = [r["original_tokens"] for r in comp if isinstance(r.get("original_tokens"), (int, float))]
    ct = [r["compressed_tokens"] for r in comp if isinstance(r.get("compressed_tokens"), (int, float))]
    if ot and ct:
        print("mean original_tokens=%.0f  mean compressed_tokens=%.0f  (%.1fx reduction)" %
              (st.mean(ot), st.mean(ct), st.mean(ot) / st.mean(ct)))
    cats = defaultdict(lambda: [0, 0])
    for r in comp:
        c = r.get("task_type") or r.get("sub_category") or "?"
        cats[c][1] += 1
        cats[c][0] += 1 if r.get("correct") else 0
    print("per-task accuracy:")
    for c, (a, t) in sorted(cats.items()):
        print("   %-18s %d/%d = %.1f%%" % (c, a, t, a / t * 100))
    if fail:
        from collections import Counter
        print("failure reasons:", Counter(str(r.get("Error"))[:50] for r in fail).most_common(5))


if __name__ == "__main__":
    main()
