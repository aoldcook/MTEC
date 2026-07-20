#!/usr/bin/env python
"""Streaming long-video eval for MLVU: for each batch, download the videos from
ModelScope, zip them into the runner's videos_chunked_00.zip, run the eval on just
that batch (resume-safe), then delete the videos. Keeps disk usage bounded.
Supports sharding (--shards N --shard I) so several copies can run in parallel over
disjoint slices, each with its own videos dir + output dir.
"""
import os, json, time, zipfile, argparse, subprocess, urllib.request, urllib.parse
import pandas as pd

ROOT = "/root/autodl-tmp/MTEC"
DSDIR = os.path.join(ROOT, "data/datasets/mlvu")
META = os.path.join(DSDIR, "mlvu_meta.parquet")
STREAM_MAP = os.path.join(DSDIR, "mlvu_stream_map.json")
OUT_BASE = os.path.join(ROOT, "outputs/eval_mlvu_long_20260625")
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


def download(url, dest, tries=3):
    for t in range(tries):
        try:
            urllib.request.urlretrieve(url, dest)
            if os.path.getsize(dest) > 1000:
                return True
        except Exception as e:
            print("   dl retry", t, e)
            time.sleep(3)
    return False


def run_batch(keys, out_dir, vid_dir):
    cmd = [
        "python", os.path.join(ROOT, "scripts/run_modelscope_mtec_anchor_api_full.py"),
        "--modalities", "video", "--model", "qwen3.7-plus",
        "--base-url", "https://dashscope.aliyuncs.com/compatible-mode/v1", "--api-key-env", "BAILIAN_API_KEY",
        "--answer-model", "qwen3.7-plus", "--answer-base-url", "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "--answer-api-key-env", "BAILIAN_API_KEY",
        "--evidence-pass", "true", "--prompt-style", "compact", "--evidence-prompt-style", "minimal",
        "--video-anchor-policy", "auto", "--global-timeline-pass", "true", "--video-transcript-backend", "none",
        "--oss-media-upload", "auto", "--cleanup-record-artifacts",
        "--videomme-metadata", META, "--video-zips-dir", vid_dir,
        "--output-dir", out_dir, "--video-record-keys", *keys,
    ]
    return subprocess.call(cmd, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0=all")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    args = ap.parse_args()

    out_dir = OUT_BASE if args.shards == 1 else os.path.join(OUT_BASE, "s%d" % args.shard)
    vid_dir = os.path.join(DSDIR, "videos" if args.shards == 1 else "videos_s%d" % args.shard)
    results_jsonl = os.path.join(out_dir, "modelscope_mtec_anchor_api_full_results.jsonl")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)

    meta = pd.read_parquet(META)
    smap = json.load(open(STREAM_MAP))
    items = [(f"video:{r['videoID']}:{r['question_id']}", str(r["videoID"])) for _, r in meta.iterrows()]
    items = [it for idx, it in enumerate(items) if idx % args.shards == args.shard]
    done = done_keys(results_jsonl)
    pending = [(k, v) for (k, v) in items if k not in done]
    if args.limit:
        pending = pending[: args.limit]
    print("shard %d/%d: assigned=%d done=%d pending=%d" % (args.shard, args.shards, len(items), len(done), len(pending)))

    for bi in range(0, len(pending), args.batch_size):
        batch = pending[bi: bi + args.batch_size]
        keys = [k for k, _ in batch]
        zpath = os.path.join(vid_dir, "videos_chunked_00.zip")
        for f in os.listdir(vid_dir):
            os.remove(os.path.join(vid_dir, f))
        t0 = time.time()
        ok = []
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
            for k, vid in batch:
                info = smap.get(vid)
                if not info:
                    print("  no stream info for", vid)
                    continue
                url = DL_BASE + urllib.parse.quote(info["repo_path"], safe="")
                tmp = os.path.join(vid_dir, "_tmp.mp4")
                if download(url, tmp):
                    z.write(tmp, arcname="%s.mp4" % vid)
                    os.remove(tmp)
                    ok.append(k)
                else:
                    print("  FAILED download", vid)
        dlsz = os.path.getsize(zpath) / 1e6
        print("[s%d batch %d-%d] dl %d/%d (%.0fMB) %.0fs -> eval" %
              (args.shard, bi, bi + len(batch), len(ok), len(batch), dlsz, time.time() - t0), flush=True)
        rc = run_batch(keys, out_dir, vid_dir)
        for f in os.listdir(vid_dir):
            try:
                os.remove(os.path.join(vid_dir, f))
            except Exception:
                pass
        print("[s%d batch done] rc=%d cumulative_completed=%d elapsed=%.0fs" %
              (args.shard, rc, len(done_keys(results_jsonl)), time.time() - t0), flush=True)
    print("SHARD_%d_DONE" % args.shard)


if __name__ == "__main__":
    main()
