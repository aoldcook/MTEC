#!/usr/bin/env python
"""Emit the long-video EXPANSION queue for one base config (A_full or
C_no_timeline). Runs vmme_long (local, OSS-auto) + mlvu_ext (streaming) on the
new keys, into dedicated subdirs so they merge with the originals at analysis
time. Writes runs/<BASE_LABEL>/queue_ext.txt.

Usage: gen_expansion_queue.py <BASE_LABEL>
"""
import os
import sys
import json
from _env import require

ROOT = "/root/autodl-tmp/MTEC"
PY = "/root/miniconda3/envs/venv/bin/python"
EXTKEYS = os.path.join(ROOT, "outputs/ablation_20260701/keys_long_ext")
RUNSDIR = os.path.join(ROOT, "outputs/ablation_20260701/runs")
BAILIAN_API_KEY = require("BAILIAN_API_KEY")
BURL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VMME_META = os.path.join(ROOT, "data/datasets/video-mme/videomme/test-00000-of-00001.parquet")
VMME_ZIPS = os.path.join(ROOT, "data/modelscope/video-mme-zips")
SUBS_DIR = os.path.join(ROOT, "outputs/videomme_asr_subtitles_base_en_no_vad")
MLVU_EXT_KEYS = os.path.join(EXTKEYS, "mlvu_ext.txt")

EXTRA = {
    "A_full": [],
    "C_no_timeline": ["--global-timeline-pass", "false"],
}
VMME_LONG_SHARDS, MLVU_EXT_SHARDS, MLVU_BATCH = 3, 3, 4


def shard(keys, n):
    out = [[] for _ in range(n)]
    for i, k in enumerate(keys):
        out[i % n].append(k)
    return [s for s in out if s]


def vmme_cmd(keys, outdir, extra):
    return [PY, os.path.join(ROOT, "scripts/run_modelscope_mtec_anchor_api_full.py"),
            "--modalities", "video",
            "--model", "qwen3.7-plus", "--base-url", BURL, "--api-key-env", "BAILIAN_API_KEY",
            "--answer-model", "qwen3.7-plus", "--answer-base-url", BURL, "--answer-api-key-env", "BAILIAN_API_KEY",
            "--evidence-pass", "true", "--prompt-style", "compact", "--evidence-prompt-style", "minimal",
            "--video-anchor-policy", "auto", "--global-timeline-pass", "true",
            "--precomputed-subtitles-dir", SUBS_DIR,
            "--oss-media-upload", "auto", "--cleanup-record-artifacts",
            "--videomme-metadata", VMME_META, "--video-zips-dir", VMME_ZIPS,
            ] + list(extra) + ["--video-record-keys"] + keys + ["--output-dir", outdir]


def mlvu_cmd(out_base, shard_i, extra):
    return [PY, os.path.join(ROOT, "scripts/run_ablation_mlvu.py"),
            "--keys-file", MLVU_EXT_KEYS, "--output-dir", out_base,
            "--batch-size", str(MLVU_BATCH), "--shards", str(MLVU_EXT_SHARDS), "--shard", str(shard_i),
            "--extra-args"] + list(extra)


def main():
    label = sys.argv[1]
    extra = EXTRA[label]
    cfg_dir = os.path.join(RUNSDIR, label)
    jobs = []

    vlong = open(os.path.join(EXTKEYS, "vmme_long.txt")).read().split()
    base = os.path.join(cfg_dir, "vmme_long")
    for i, sk in enumerate(shard(vlong, VMME_LONG_SHARDS)):
        outdir = os.path.join(base, "s%d" % i)
        os.makedirs(outdir, exist_ok=True)
        jobs.append({"name": "%s/vmme_long/s%d" % (label, i),
                     "cmd": vmme_cmd(sk, outdir, extra), "cwd": ROOT,
                     "log": os.path.join(base, "s%d.log" % i),
                     "env": {"BAILIAN_API_KEY": BAILIAN_API_KEY}})

    mbase = os.path.join(cfg_dir, "mlvu_ext")
    os.makedirs(mbase, exist_ok=True)
    for i in range(MLVU_EXT_SHARDS):
        jobs.append({"name": "%s/mlvu_ext/s%d" % (label, i),
                     "cmd": mlvu_cmd(mbase, i, extra), "cwd": ROOT,
                     "log": os.path.join(mbase, "s%d.log" % i),
                     "env": {"BAILIAN_API_KEY": BAILIAN_API_KEY}})

    qf = os.path.join(cfg_dir, "queue_ext.txt")
    with open(qf, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print("expansion %s: jobs=%d (vmme_long=%d shards + mlvu_ext=%d shards) -> %s" % (
        label, len(jobs), VMME_LONG_SHARDS, MLVU_EXT_SHARDS, qf))


if __name__ == "__main__":
    main()
