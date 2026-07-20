#!/usr/bin/env python
"""Generic MLVU-long batch runner for the ablation study (Bailian regime).

Streams videos in small batches (download -> zip -> compressed-pipeline eval ->
delete) to keep disk usage bounded, same pattern as scripts/stream_eval_mlvu.py,
but: (1) uses curl (not urllib) for downloads per operational lessons, (2) accepts
an arbitrary keys-file (the fixed ablation subset, not the full 180), (3) accepts
extra CLI flags to pass through to the runner so each ablation config is just a
flag flip, (4) is shardable for controlled parallelism.
"""
import os
import sys
import json
import time
import zipfile
import argparse
import subprocess
import urllib.parse

ROOT = "/root/autodl-tmp/MTEC"
DSDIR = os.path.join(ROOT, "data/datasets/mlvu")
META = os.path.join(DSDIR, "mlvu_meta.parquet")
STREAM_MAP = os.path.join(DSDIR, "mlvu_stream_map.json")
DL_BASE = "https://modelscope.cn/api/v1/datasets/AI-ModelScope/MLVU/repo?Revision=master&FilePath="


def done_keys(results_jsonl):
    if not os.path.exists(results_jsonl):
        return set()
    ks = set()
    for line in open(results_jsonl):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if r.get("status") == "completed":
                ks.add(r.get("record_key"))
        except Exception:
            pass
    return ks


def curl_download(url, dest, tries=1):
    rc = subprocess.call(["curl", "-sL", "--retry", "6", "--retry-delay", "4",
                           "--retry-all-errors", "-o", dest, url])
    return rc == 0 and os.path.exists(dest) and os.path.getsize(dest) > 10000


def run_batch(keys, out_dir, vid_dir, extra_args):
    cmd = [
        sys.executable, os.path.join(ROOT, "scripts/run_modelscope_mtec_anchor_api_full.py"),
        "--modalities", "video", "--model", "qwen3.7-plus",
        "--base-url", "https://dashscope.aliyuncs.com/compatible-mode/v1", "--api-key-env", "BAILIAN_API_KEY",
        "--answer-model", "qwen3.7-plus", "--answer-base-url", "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "--answer-api-key-env", "BAILIAN_API_KEY",
        "--evidence-pass", "true", "--prompt-style", "compact", "--evidence-prompt-style", "minimal",
        "--video-anchor-policy", "auto", "--global-timeline-pass", "true",
        "--oss-media-upload", "auto", "--cleanup-record-artifacts",
        "--videomme-metadata", META, "--video-zips-dir", vid_dir,
        "--output-dir", out_dir,
    ] + list(extra_args) + ["--video-record-keys", *keys]
    return subprocess.call(cmd, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys-file", required=True, help="whitespace-separated record_keys")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    out_dir = args.output_dir if args.shards == 1 else os.path.join(args.output_dir, "s%d" % args.shard)
    vid_dir = os.path.join(out_dir, "videos_tmp")
    results_jsonl = os.path.join(out_dir, "modelscope_mtec_anchor_api_full_results.jsonl")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)

    all_keys = open(args.keys_file).read().split()
    meta_by_key = {}
    for k in all_keys:
        # record_key format video:{videoID}:{question_id}
        parts = k.split(":")
        vid = parts[1]
        meta_by_key[k] = vid

    shard_keys = [k for i, k in enumerate(all_keys) if i % args.shards == args.shard]
    done = done_keys(results_jsonl)
    pending = [k for k in shard_keys if k not in done]
    print("shard %d/%d: assigned=%d done=%d pending=%d extra_args=%s" %
          (args.shard, args.shards, len(shard_keys), len(done), len(pending), args.extra_args), flush=True)

    smap = json.load(open(STREAM_MAP))

    for bi in range(0, len(pending), args.batch_size):
        batch = pending[bi: bi + args.batch_size]
        for f in os.listdir(vid_dir):
            os.remove(os.path.join(vid_dir, f))
        zpath = os.path.join(vid_dir, "videos_chunked_00.zip")
        t0 = time.time()
        ok = []
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
            for k in batch:
                vid = meta_by_key[k]
                info = smap.get(vid)
                if not info:
                    print("  no stream info for", vid, flush=True)
                    continue
                url = DL_BASE + urllib.parse.quote(info["repo_path"], safe="")
                tmp = os.path.join(vid_dir, "_tmp.mp4")
                if curl_download(url, tmp):
                    z.write(tmp, arcname="%s.mp4" % vid)
                    os.remove(tmp)
                    ok.append(k)
                else:
                    print("  FAILED download", vid, flush=True)
        dlsz = os.path.getsize(zpath) / 1e6
        print("[s%d batch %d-%d] dl %d/%d (%.0fMB) %.0fs -> eval" %
              (args.shard, bi, bi + len(batch), len(ok), len(batch), dlsz, time.time() - t0), flush=True)
        rc = run_batch(batch, out_dir, vid_dir, args.extra_args)
        for f in os.listdir(vid_dir):
            try:
                os.remove(os.path.join(vid_dir, f))
            except Exception:
                pass
        print("[s%d batch done] rc=%d cumulative_completed=%d elapsed=%.0fs" %
              (args.shard, rc, len(done_keys(results_jsonl)), time.time() - t0), flush=True)
    print("SHARD_%d_DONE" % args.shard, flush=True)


if __name__ == "__main__":
    main()
