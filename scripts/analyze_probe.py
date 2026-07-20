#!/usr/bin/env python
"""Analyze the SF decision-gate probe: blended + per-dataset accuracy, completion
rate, and median token savings. Prints a verdict against the decision line."""
import os
import json
import glob
import statistics

PROBE = "/root/autodl-tmp/MTEC/outputs/ablation_20260701/probe"


def load(pattern):
    recs = {}
    for f in glob.glob(pattern):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            k = r.get("record_key")
            # keep the last completed record per key (retries append)
            if r.get("status") == "completed":
                recs[k] = r
            elif k not in recs:
                recs[k] = r
    return list(recs.values())


def summarize(name, recs):
    total = len(recs)
    completed = [r for r in recs if r.get("status") == "completed"]
    correct = [r for r in completed if r.get("correct") is True]
    tsr = [r.get("token_saving_ratio_raw") for r in completed
           if isinstance(r.get("token_saving_ratio_raw"), (int, float))]
    acc = (len(correct) / len(completed) * 100) if completed else 0.0
    comp_rate = (len(completed) / total * 100) if total else 0.0
    med_tsr = (statistics.median(tsr) * 100) if tsr else None
    print("[%s] records=%d completed=%d (%.0f%% completion) correct=%d accuracy=%.1f%% median_token_save=%s" % (
        name, total, len(completed), comp_rate, len(correct), acc,
        ("%.1f%%" % med_tsr) if med_tsr is not None else "n/a"))
    return len(completed), len(correct)


vmme = load(os.path.join(PROBE, "vmme_short/s*/modelscope_mtec_anchor_api_full_results.jsonl"))
nextqa = load(os.path.join(PROBE, "nextqa/s*/modelscope_mtec_anchor_api_full_results.jsonl"))

print("=== SF Qwen3-VL-32B decision-gate probe ===")
vc, vcorr = summarize("Video-MME short", vmme)
nc, ncorr = summarize("NExT-QA", nextqa)

tot_completed = vc + nc
tot_correct = vcorr + ncorr
blended = (tot_correct / tot_completed * 100) if tot_completed else 0.0
print("-" * 60)
print("BLENDED accuracy (completed only): %d/%d = %.1f%%" % (tot_correct, tot_completed, blended))
print("Reference (Bailian qwen3.7-plus compressed): VM-short 88%, NExT-QA 82.7% (~85%% blended)")
print("-" * 60)
if blended >= 78:
    print("VERDICT: >=78%% -> CONTINUE on SiliconFlow.")
elif blended >= 70:
    print("VERDICT: 70-78%% -> MARGINAL. Usable but note the cross-model gap.")
else:
    print("VERDICT: <70%% -> TOO LOW. Recommend switching downstream to Bailian qwen3.7-plus for short+medium.")
