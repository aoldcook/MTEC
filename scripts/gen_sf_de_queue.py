#!/usr/bin/env python
"""Emit an SF-Qwen3-VL-32B queue for one D/E config on the SF-safe short test set
(Video-MME short 50 + NExT-QA 50). Tests whether the visual-context pass (D) and
query-retrieval (E) help MORE on a weaker model than on qwen3.7-plus.

Configs: A_sf (baseline, both ON), D_sf (visual-context off), E_sf (query-retrieval off).
SF specifics: base64 (oss off), --answer-enable-thinking omit, low concurrency.

Usage: gen_sf_de_queue.py <A_sf|D_sf|E_sf>
"""
import os
import sys
import json
from _env import require

ROOT = "/root/autodl-tmp/MTEC"
PY = "/root/miniconda3/envs/venv/bin/python"
KEYSDIR = os.path.join(ROOT, "outputs/ablation_20260701/keys")  # full fixed set: 100 short + 100 nextqa
RUNSDIR = os.path.join(ROOT, "outputs/ablation_20260701/runs")
SF_API_KEY = require("SF_API_KEY")
SURL = os.environ.get("SF_URL_OVERRIDE", "https://api.siliconflow.cn/v1")
SF_MODEL = os.environ.get("SF_MODEL_OVERRIDE", "Qwen/Qwen3-VL-32B-Instruct")

VMME_META = os.path.join(ROOT, "data/datasets/video-mme/videomme/test-00000-of-00001.parquet")
VMME_ZIPS = os.path.join(ROOT, "data/modelscope/video-mme-zips")
NEXTQA_META = os.path.join(ROOT, "data/datasets/nextqa/nextqa_meta.parquet")
NEXTQA_ZIPS = os.path.join(ROOT, "data/datasets/nextqa/videos")
SUBS_DIR = os.path.join(ROOT, "outputs/videomme_asr_subtitles_base_en_no_vad")

EXTRA = {
    "A_sf": ([], {}),
    "D_sf": (["--video-visual-context-pass", "false"], {}),
    "E_sf": (["--video-query-retrieval", "false"], {}),
    "H_sf": ([], {"MTEC_DETAIL_MAX_CROPS": "0"}),
    "I_sf": ([], {"MTEC_DISABLE_TASK_FAMILY_ROUTING": "1"}),
    "A_lz": ([], {}),
    "I_lz": ([], {"MTEC_DISABLE_TASK_FAMILY_ROUTING": "1"}),
}
VMME_SHARDS, NEXTQA_SHARDS = 2, 2
N_PER = 100  # 100 short + 100 nextqa per config (expanded for statistical power)


def shard(keys, n):
    out = [[] for _ in range(n)]
    for i, k in enumerate(keys):
        out[i % n].append(k)
    return [s for s in out if s]


def sf_cmd(meta, zips, keys, outdir, with_subs, extra):
    cmd = [PY, os.path.join(ROOT, "scripts/run_modelscope_mtec_anchor_api_full.py"),
           "--modalities", "video",
           "--model", SF_MODEL, "--base-url", SURL, "--api-key-env", "SF_API_KEY",
           "--answer-model", SF_MODEL, "--answer-base-url", SURL, "--answer-api-key-env", "SF_API_KEY",
           "--answer-enable-thinking", "omit",
           "--evidence-pass", "true", "--prompt-style", "compact", "--evidence-prompt-style", "minimal",
           "--video-anchor-policy", "auto", "--global-timeline-pass", "true",
           "--oss-media-upload", "off", "--cleanup-record-artifacts", "--resume",
           "--videomme-metadata", meta, "--video-zips-dir", zips]
    if with_subs:
        cmd += ["--precomputed-subtitles-dir", SUBS_DIR]
    cmd += list(extra) + ["--video-record-keys"] + keys + ["--output-dir", outdir]
    return cmd


def main():
    label = sys.argv[1]
    extra, cfg_env = EXTRA[label]
    job_env = {"SF_API_KEY": SF_API_KEY, **cfg_env}
    cfg_dir = os.path.join(RUNSDIR, label)
    jobs = []

    # Video-MME short only (SF-safe size)
    short = open(os.path.join(KEYSDIR, "sf_vmme_short.txt")).read().split()[:N_PER]
    base = os.path.join(cfg_dir, "vmme")
    for i, sk in enumerate(shard(short, VMME_SHARDS)):
        outdir = os.path.join(base, "s%d" % i)
        os.makedirs(outdir, exist_ok=True)
        jobs.append({"name": "%s/vmme/s%d" % (label, i),
                     "cmd": sf_cmd(VMME_META, VMME_ZIPS, sk, outdir, True, extra),
                     "cwd": ROOT, "log": os.path.join(base, "s%d.log" % i),
                     "env": dict(job_env)})

    nextqa = open(os.path.join(KEYSDIR, "sf_nextqa.txt")).read().split()[:N_PER]
    base = os.path.join(cfg_dir, "nextqa")
    for i, sk in enumerate(shard(nextqa, NEXTQA_SHARDS)):
        outdir = os.path.join(base, "s%d" % i)
        os.makedirs(outdir, exist_ok=True)
        jobs.append({"name": "%s/nextqa/s%d" % (label, i),
                     "cmd": sf_cmd(NEXTQA_META, NEXTQA_ZIPS, sk, outdir, False, extra),
                     "cwd": ROOT, "log": os.path.join(base, "s%d.log" % i),
                     "env": dict(job_env)})

    qf = os.path.join(cfg_dir, "queue.txt")
    os.makedirs(cfg_dir, exist_ok=True)
    with open(qf, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print("SF config %s: jobs=%d (vmme_short=%d + nextqa=%d) -> %s" % (
        label, len(jobs), len(short), len(nextqa), qf))


if __name__ == "__main__":
    main()
