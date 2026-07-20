#!/usr/bin/env python
"""Build the SF decision-gate probe queue: 25 Video-MME short + 25 NExT-QA records
on SiliconFlow Qwen3-VL-32B, sharded for bounded-concurrency execution."""
import os
import json
from _env import require

ROOT = "/root/autodl-tmp/MTEC"
PY = "/root/miniconda3/envs/venv/bin/python"
KEYSDIR = os.path.join(ROOT, "outputs/ablation_20260701/keys")
PROBEDIR = os.path.join(ROOT, "outputs/ablation_20260701/probe")
QUEUE_FILE = os.path.join(PROBEDIR, "queue.txt")

SF_API_KEY = require("SF_API_KEY")
VMME_META = os.path.join(ROOT, "data/datasets/video-mme/videomme/test-00000-of-00001.parquet")
VMME_ZIPS = os.path.join(ROOT, "data/modelscope/video-mme-zips")
NEXTQA_META = os.path.join(ROOT, "data/datasets/nextqa/nextqa_meta.parquet")
NEXTQA_ZIPS = os.path.join(ROOT, "data/datasets/nextqa/videos")
SUBS_DIR = os.path.join(ROOT, "outputs/videomme_asr_subtitles_base_en_no_vad")

VMME_SHORT = open(os.path.join(KEYSDIR, "sf_vmme_short.txt")).read().split()[:25]
NEXTQA = open(os.path.join(KEYSDIR, "sf_nextqa.txt")).read().split()[:25]


def shard(keys, n):
    out = [[] for _ in range(n)]
    for i, k in enumerate(keys):
        out[i % n].append(k)
    return [s for s in out if s]


def base_cmd(meta, zips, keys, outdir, with_subs):
    cmd = [PY, os.path.join(ROOT, "scripts/run_modelscope_mtec_anchor_api_full.py"),
           "--modalities", "video",
           "--model", "Qwen/Qwen3-VL-32B-Instruct", "--base-url", "https://api.siliconflow.cn/v1",
           "--api-key-env", "SF_API_KEY",
           "--answer-model", "Qwen/Qwen3-VL-32B-Instruct", "--answer-base-url", "https://api.siliconflow.cn/v1",
           "--answer-api-key-env", "SF_API_KEY", "--answer-enable-thinking", "omit",
           "--evidence-pass", "true", "--prompt-style", "compact", "--evidence-prompt-style", "minimal",
           "--video-anchor-policy", "auto", "--global-timeline-pass", "true",
           "--oss-media-upload", "off", "--cleanup-record-artifacts",
           "--videomme-metadata", meta, "--video-zips-dir", zips]
    if with_subs:
        cmd += ["--precomputed-subtitles-dir", SUBS_DIR]
    cmd += ["--video-record-keys"] + keys + ["--output-dir", outdir]
    return cmd


jobs = []
vmme_base = os.path.join(PROBEDIR, "vmme_short")
nextqa_base = os.path.join(PROBEDIR, "nextqa")
for i, sk in enumerate(shard(VMME_SHORT, 3)):
    outdir = os.path.join(vmme_base, "s%d" % i)
    os.makedirs(outdir, exist_ok=True)
    jobs.append({"name": "probe/vmme_short/s%d" % i,
                 "cmd": base_cmd(VMME_META, VMME_ZIPS, sk, outdir, True), "cwd": ROOT,
                 "log": os.path.join(vmme_base, "s%d.log" % i),
                 "env": {"SF_API_KEY": SF_API_KEY}})
for i, sk in enumerate(shard(NEXTQA, 2)):
    outdir = os.path.join(nextqa_base, "s%d" % i)
    os.makedirs(outdir, exist_ok=True)
    jobs.append({"name": "probe/nextqa/s%d" % i,
                 "cmd": base_cmd(NEXTQA_META, NEXTQA_ZIPS, sk, outdir, False), "cwd": ROOT,
                 "log": os.path.join(nextqa_base, "s%d.log" % i),
                 "env": {"SF_API_KEY": SF_API_KEY}})

os.makedirs(PROBEDIR, exist_ok=True)
with open(QUEUE_FILE, "w") as f:
    for j in jobs:
        f.write(json.dumps(j) + "\n")
print("wrote %d probe jobs (vmme_short=25, nextqa=25) to %s" % (len(jobs), QUEUE_FILE))
