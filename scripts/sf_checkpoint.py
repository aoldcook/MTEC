#!/usr/bin/env python
"""Snapshot SF weak-model D/E test state + write a resume checkpoint."""
import json
import glob
import os
import time

ROOT = "/root/autodl-tmp/MTEC/outputs/ablation_20260701"


def counts(label):
    out = {}
    for ds in ["vmme", "nextqa"]:
        comp = set()
        seen = set()
        for f in glob.glob("%s/runs/%s/%s/s*/modelscope_mtec_anchor_api_full_results.jsonl" % (ROOT, label, ds)):
            for l in open(f):
                try:
                    r = json.loads(l)
                except Exception:
                    continue
                seen.add(r["record_key"])
                if r.get("status") == "completed":
                    comp.add(r["record_key"])
        out[ds] = (len(comp), len(seen))
    return out


lines = []
lines.append("SF WEAK-MODEL D/E TEST — CHECKPOINT %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
lines.append("Model: SiliconFlow Qwen/Qwen3-VL-32B-Instruct; base64 (oss off); conc 3; answer-enable-thinking omit.")
lines.append("Test set: 100 Video-MME short + 100 NExT-QA per config. Baseline=A_sf.")
lines.append("")
for label in ["A_sf", "D_sf", "E_sf"]:
    c = counts(label)
    total_c = c["vmme"][0] + c["nextqa"][0]
    lines.append("%-6s completed=%d  vmme=%d/%dseen  nextqa=%d/%dseen"
                 % (label, total_c, c["vmme"][0], c["vmme"][1], c["nextqa"][0], c["nextqa"][1]))
lines.append("")
lines.append("STATUS: A_sf DONE (baseline). D_sf DONE (analyzed: -3.0 vs A_sf; nextqa -4.6).")
lines.append("        E_sf PARTIAL (query-retrieval off) -- resume tomorrow.")
lines.append("")
lines.append("RESUME TOMORROW (after server boot):")
lines.append("  1. restart watchdogs: setsid bash scripts/balance_monitor.sh & ; setsid bash scripts/sf_balance_watchdog.sh &")
lines.append("  2. finish E_sf (resume-safe, skips completed): bash scripts/launch_config.sh E_sf 3")
lines.append("     (NOTE: SF hangs intermittently -- if a shard freezes after INPUT_CHECK with stale log mtime,")
lines.append("      pkill -9 -f 'E_sf/queue.txt'; pkill -9 -f 'runs/E_sf/'; then relaunch. Resume-safe.)")
lines.append("  3. When E_sf QUEUE_ALL_DONE: /root/miniconda3/envs/venv/bin/python scripts/analyze_config.py E_sf A_sf")
lines.append("     and scripts/analyze_config.py D_sf A_sf ; compare D/E deltas weak(SF) vs strong(qwen3.7-plus).")
lines.append("  (analyze_config.py header says 'Bailian qwen3.7-plus' -- COSMETIC hardcoded-label bug; these are SF runs.)")
lines.append("")
lines.append("Also pending (Bailian, another day): H (detail-crops), I (task-routing) on vmme80+nextqa50; J_tiny/J_medium on all 4 datasets.")

txt = "\n".join(lines)
open(os.path.join(ROOT, "SF_DE_CHECKPOINT.txt"), "w").write(txt + "\n")
print(txt)
