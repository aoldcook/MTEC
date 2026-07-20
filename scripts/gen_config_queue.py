#!/usr/bin/env python
"""Emit the shard-job queue for ONE ablation config (all Bailian qwen3.7-plus).

Config A (full reference) is NOT run here -- it comes from PAPER_ARCHIVE. This
generator only builds the component-off variants B..K, each over the same fixed
record-key subsets as the archive, so deltas are valid.

Usage: gen_config_queue.py <CONFIG_LABEL>
Writes outputs/ablation_20260701/runs/<label>/queue.txt
"""
import os
import sys
import json
from _env import require

ROOT = "/root/autodl-tmp/MTEC"
PY = "/root/miniconda3/envs/venv/bin/python"
KEYSDIR = os.path.join(ROOT, "outputs/ablation_20260701/keys_pruned")
LONGEXT_KEYSDIR = os.path.join(ROOT, "outputs/ablation_20260701/keys_long_ext")
RUNSDIR = os.path.join(ROOT, "outputs/ablation_20260701/runs")
BAILIAN_API_KEY = require("BAILIAN_API_KEY")
BURL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# closeAI proxy for qwen3.7-plus (base64 media only -- no DashScope OSS). Used for
# small-video datasets (vmme, nextqa). Large-video datasets stay on Bailian OSS.
CLOSEAI_KEY = require("CLOSEAI_API_KEY")
CLOSEAI_URL = "https://api.openai-proxy.org/v1"

VMME_META = os.path.join(ROOT, "data/datasets/video-mme/videomme/test-00000-of-00001.parquet")
VMME_ZIPS = os.path.join(ROOT, "data/modelscope/video-mme-zips")
NEXTQA_META = os.path.join(ROOT, "data/datasets/nextqa/nextqa_meta.parquet")
NEXTQA_ZIPS = os.path.join(ROOT, "data/datasets/nextqa/videos")
SUBS_DIR = os.path.join(ROOT, "outputs/videomme_asr_subtitles_base_en_no_vad")
MLVU_KEYS_FILE = os.path.join(KEYSDIR, "bailian_mlvu_long.txt")

VMME_SHARDS, NEXTQA_SHARDS, MLVU_SHARDS = 3, 2, 3
MLVU_BATCH = 4

# config -> (extra runner args, env overrides, datasets touched)
# Long-video testing set expanded (2026-07-02): mlvu 30 + vmme_long 40.
# G and K dropped from the plan.
CONFIGS = {
    "A_full":              ([],                                         {}, ["vmme", "nextqa", "mlvu", "vmme_long"]),
    "B_no_evidence":       (["--evidence-pass", "false"],               {}, ["vmme", "nextqa", "mlvu"]),
    "C_no_timeline":       (["--global-timeline-pass", "false"],        {}, ["mlvu"]),
    "D_no_visual_context": (["--video-visual-context-pass", "false"],   {}, ["mlvu", "vmme_long"]),
    "E_no_query_retrieval":(["--video-query-retrieval", "false"],       {}, ["mlvu", "vmme_long"]),
    "F_no_global_anchor":  (["--video-global-anchor", "false"],         {}, ["mlvu", "vmme_long"]),
    "H_no_detail_crops":   ([], {"MTEC_DETAIL_MAX_CROPS": "0"},          ["vmme", "nextqa"]),
    "I_no_task_routing":   ([], {"MTEC_DISABLE_TASK_FAMILY_ROUTING": "1"}, ["vmme", "nextqa"]),
    "J_tiny":              (["--video-anchor-policy", "tiny"],          {}, ["vmme", "nextqa", "mlvu", "vmme_long"]),
    "J_medium":            (["--video-anchor-policy", "medium"],        {}, ["vmme", "nextqa", "mlvu", "vmme_long"]),
}


def load(name):
    return open(os.path.join(KEYSDIR, name + ".txt")).read().split()


def shard(keys, n):
    out = [[] for _ in range(n)]
    for i, k in enumerate(keys):
        out[i % n].append(k)
    return [s for s in out if s]


def bailian_local_cmd(meta, zips, keys, outdir, with_subs, extra):
    cmd = [PY, os.path.join(ROOT, "scripts/run_modelscope_mtec_anchor_api_full.py"),
           "--modalities", "video",
           "--model", "qwen3.7-plus", "--base-url", BURL, "--api-key-env", "BAILIAN_API_KEY",
           "--answer-model", "qwen3.7-plus", "--answer-base-url", BURL, "--answer-api-key-env", "BAILIAN_API_KEY",
           "--evidence-pass", "true", "--prompt-style", "compact", "--evidence-prompt-style", "minimal",
           "--video-anchor-policy", "auto", "--global-timeline-pass", "true",
           "--oss-media-upload", "auto", "--cleanup-record-artifacts", "--resume",
           "--videomme-metadata", meta, "--video-zips-dir", zips]
    if with_subs:
        cmd += ["--precomputed-subtitles-dir", SUBS_DIR]
    cmd += list(extra) + ["--video-record-keys"] + keys + ["--output-dir", outdir]
    return cmd


def closeai_local_cmd(meta, zips, keys, outdir, with_subs, extra):
    """qwen3.7-plus via closeAI proxy, base64 media (oss off). For small-video sets."""
    cmd = [PY, os.path.join(ROOT, "scripts/run_modelscope_mtec_anchor_api_full.py"),
           "--modalities", "video",
           "--model", "qwen3.7-plus", "--base-url", CLOSEAI_URL, "--api-key-env", "CLOSEAI_API_KEY",
           "--answer-model", "qwen3.7-plus", "--answer-base-url", CLOSEAI_URL, "--answer-api-key-env", "CLOSEAI_API_KEY",
           "--evidence-pass", "true", "--prompt-style", "compact", "--evidence-prompt-style", "minimal",
           "--video-anchor-policy", "auto", "--global-timeline-pass", "true",
           "--oss-media-upload", "off", "--cleanup-record-artifacts", "--resume",
           "--videomme-metadata", meta, "--video-zips-dir", zips]
    if with_subs:
        cmd += ["--precomputed-subtitles-dir", SUBS_DIR]
    cmd += list(extra) + ["--video-record-keys"] + keys + ["--output-dir", outdir]
    return cmd


def mlvu_cmd(out_base, shard_i, extra):
    return [PY, os.path.join(ROOT, "scripts/run_ablation_mlvu.py"),
            "--keys-file", MLVU_KEYS_FILE, "--output-dir", out_base,
            "--batch-size", str(MLVU_BATCH), "--shards", str(MLVU_SHARDS), "--shard", str(shard_i),
            "--extra-args"] + list(extra)


def main():
    label = sys.argv[1]
    extra, env, datasets = CONFIGS[label]
    cfg_dir = os.path.join(RUNSDIR, label)
    jobs = []

    if "vmme" in datasets:
        keys = load("sf_vmme_short") + load("sf_vmme_medium")  # 150
        base = os.path.join(cfg_dir, "vmme")
        for i, sk in enumerate(shard(keys, VMME_SHARDS)):
            outdir = os.path.join(base, "s%d" % i)
            os.makedirs(outdir, exist_ok=True)
            jobs.append({"name": "%s/vmme/s%d" % (label, i),
                         "cmd": closeai_local_cmd(VMME_META, VMME_ZIPS, sk, outdir, True, extra),
                         "cwd": ROOT, "log": os.path.join(base, "s%d.log" % i),
                         "env": {"CLOSEAI_API_KEY": CLOSEAI_KEY, **env}})

    if "nextqa" in datasets:
        keys = load("sf_nextqa")  # 100
        base = os.path.join(cfg_dir, "nextqa")
        for i, sk in enumerate(shard(keys, NEXTQA_SHARDS)):
            outdir = os.path.join(base, "s%d" % i)
            os.makedirs(outdir, exist_ok=True)
            jobs.append({"name": "%s/nextqa/s%d" % (label, i),
                         "cmd": closeai_local_cmd(NEXTQA_META, NEXTQA_ZIPS, sk, outdir, False, extra),
                         "cwd": ROOT, "log": os.path.join(base, "s%d.log" % i),
                         "env": {"CLOSEAI_API_KEY": CLOSEAI_KEY, **env}})

    if "mlvu" in datasets:
        base = os.path.join(cfg_dir, "mlvu")
        os.makedirs(base, exist_ok=True)
        for i in range(MLVU_SHARDS):
            jobs.append({"name": "%s/mlvu/s%d" % (label, i),
                         "cmd": mlvu_cmd(base, i, extra),
                         "cwd": ROOT, "log": os.path.join(base, "s%d.log" % i),
                         "env": {"BAILIAN_API_KEY": BAILIAN_API_KEY, **env}})

    if "vmme_long" in datasets:
        keys = open(os.path.join(LONGEXT_KEYSDIR, "vmme_long.txt")).read().split()  # 40
        base = os.path.join(cfg_dir, "vmme_long")
        for i, sk in enumerate(shard(keys, VMME_SHARDS)):
            outdir = os.path.join(base, "s%d" % i)
            os.makedirs(outdir, exist_ok=True)
            jobs.append({"name": "%s/vmme_long/s%d" % (label, i),
                         "cmd": bailian_local_cmd(VMME_META, VMME_ZIPS, sk, outdir, True, extra),
                         "cwd": ROOT, "log": os.path.join(base, "s%d.log" % i),
                         "env": {"BAILIAN_API_KEY": BAILIAN_API_KEY, **env}})

    os.makedirs(cfg_dir, exist_ok=True)
    qf = os.path.join(cfg_dir, "queue.txt")
    with open(qf, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print("config=%s datasets=%s jobs=%d -> %s" % (label, datasets, len(jobs), qf))


if __name__ == "__main__":
    main()
