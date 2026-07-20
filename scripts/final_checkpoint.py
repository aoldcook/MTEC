#!/usr/bin/env python
"""Full ablation-matrix checkpoint: per-config completion + resume plan."""
import json
import glob
import os
import time

ROOT = "/root/autodl-tmp/MTEC/outputs/ablation_20260701"


def uniq_completed(label):
    tot = 0
    for f in glob.glob("%s/runs/%s/*/s*/modelscope_mtec_anchor_api_full_results.jsonl" % (ROOT, label)):
        pass
    per = {}
    for f in glob.glob("%s/runs/%s/*/s*/modelscope_mtec_anchor_api_full_results.jsonl" % (ROOT, label)):
        ds = f.split("/runs/%s/" % label)[1].split("/")[0]
        per.setdefault(ds, set())
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("status") == "completed":
                per[ds].add(r["record_key"])
    return {ds: len(s) for ds, s in per.items()}, sum(len(s) for s in per.values())


CONFIGS = ["A_full", "B_no_evidence", "C_no_timeline", "D_no_visual_context",
           "E_no_query_retrieval", "F_no_global_anchor", "H_no_detail_crops",
           "I_no_task_routing", "J_tiny", "J_medium",
           "A_sf", "D_sf", "E_sf", "H_sf"]

lines = ["ABLATION FULL CHECKPOINT %s" % time.strftime("%Y-%m-%d %H:%M:%S"), ""]
for c in CONFIGS:
    if not os.path.isdir("%s/runs/%s" % (ROOT, c)):
        continue
    per, tot = uniq_completed(c)
    lines.append("%-22s total=%d  %s" % (c, tot, per))
lines.append("")
lines.append("STATUS: All strong-model configs DONE (B,C,D,E,F,H,I,J_tiny) except J_medium (final, PARTIAL).")
lines.append("SF weak-model DONE: A_sf, D_sf, E_sf, H_sf.")
lines.append("")
lines.append("RESUME TOMORROW: boot server, then:")
lines.append("  1. restart balance monitor: cd /root/autodl-tmp/MTEC && setsid bash scripts/balance_monitor.sh &")
lines.append("  2. finish J_medium (resume-safe, --resume in generator): bash scripts/launch_config.sh J_medium 3")
lines.append("     (hybrid: vmme+nextqa via closeAI base64; mlvu+vmme_long via Bailian OSS.")
lines.append("      If an SF/closeAI shard hangs: pkill -9 -f 'J_medium/queue.txt'; pkill -9 -f 'runs/J_medium/'; relaunch.)")
lines.append("  3. When J_medium QUEUE_ALL_DONE: python scripts/analyze_config.py J_medium A_full")
lines.append("  4. Then the FULL matrix is complete -> write consolidated final report (RESULT_*.txt are per-config).")
lines.append("")
lines.append("KEY RESULTS (Delta vs baseline, paired):")
lines.append("  STRONG model (qwen3.7-plus) vs A_full: B+1.9 C+5.2 D+5.2 E+3.4 F-10.3(REAL) H+3.8(+tokens) I-0.8 J_tiny+1.1(big token-save on short)")
lines.append("  WEAK model (Qwen3-VL-32B) vs A_sf: D-3.0 E-3.1 H-3.8  (all HELP weak model = flip vs strong)")
lines.append("  Headline: only global-anchor(F) is load-bearing on strong model; visual-context/query-retrieval/detail-crops are")
lines.append("            pure overhead on strong model but help the weak model. J_tiny: near-auto accuracy, much better compression on short.")
lines.append("")
lines.append("PROVIDER NOTE: qwen3.7-plus now via closeAI (api.openai-proxy.org, key sk-8LUX...) for vmme+nextqa; Bailian OSS for large.")
lines.append("SF old key sk-qqid ($41.93 paid) works; new key sk-otyq had $0 paid -> 403.")

txt = "\n".join(lines)
open("%s/FINAL_CHECKPOINT.txt" % ROOT, "w").write(txt + "\n")
print(txt)
