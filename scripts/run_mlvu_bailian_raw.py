#!/usr/bin/env python3
"""Platform-comparison rerun: the 141 MLVU-long videos that FAILED the SiliconFlow
raw baseline (base64 size ceiling) are retried on BAILIAN/DashScope qwen3.6-plus,
which transports large media via OSS (no base64 size limit). Tests whether the
SiliconFlow failures were platform-specific (size) vs model-specific.
"""
import os, re, sys, json, time, argparse, urllib.request, urllib.parse
from pathlib import Path
import pandas as pd

ROOT = Path("/root/autodl-tmp/MTEC")
sys.path.insert(0, str(ROOT))
from scripts.run_modelscope_mtec_anchor_api_full import (  # noqa
    OSS_MEDIA, SiliconFlowClient, media_content, extract_letter,
)

KEY = os.environ["BAILIAN_API_KEY"]
BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3.6-plus"
OUT_BASE = ROOT / "outputs/raw_bailian_mlvu_long_20260625"
SIZE_CAP = int(os.environ.get("DL_CAP_MB", "2000")) * 1_000_000  # skip > this (disk safety)
DL = "https://modelscope.cn/api/v1/datasets/AI-ModelScope/MLVU/repo?Revision=master&FilePath="


def first_letter(t):
    m = re.search(r"\b([A-G])\b", str(t).upper())
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys-file", required=True)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    args = ap.parse_args()

    OSS_MEDIA.update({"enabled": True, "api_key": KEY, "model": MODEL,
                      "threshold_bytes": 1, "always_kinds": {"video"}})
    out_dir = OUT_BASE / ("s%d" % args.shard)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / "results.jsonl"
    tmp = out_dir / "tmp"; tmp.mkdir(exist_ok=True)

    done = set()
    for rf in OUT_BASE.glob("s*/results.jsonl"):
        for l in rf.open():
            l = l.strip()
            if l and json.loads(l).get("status") == "completed":
                done.add(json.loads(l)["record_key"])

    meta = pd.read_parquet(ROOT / "data/datasets/mlvu/mlvu_meta.parquet")
    smap = json.load(open(ROOT / "data/datasets/mlvu/mlvu_stream_map.json"))
    row = {}
    for _, r in meta.iterrows():
        row[f"video:{r['videoID']}:{r['question_id']}"] = r

    keys = [l.strip() for l in open(args.keys_file) if l.strip()]
    keys = [k for i, k in enumerate(keys) if i % args.shards == args.shard]
    client = SiliconFlowClient(api_key=KEY, model=MODEL, base_url=BASE, timeout=1200,
                               max_retries=2, retry_sleep=4.0, temperature=0.0, enable_thinking=None)
    print("shard %d/%d keys=%d model=%s" % (args.shard, args.shards, len(keys), MODEL), flush=True)

    for k in keys:
        if k in done or k not in row:
            continue
        r = row[k]
        vid = str(r["videoID"])
        info = smap.get(vid)
        gt = (extract_letter(r["answer"]) or str(r["answer"])).strip().upper()
        opts = "\n".join(str(x) for x in list(r["options"]))
        prompt = ("%s\n%s\n\nWatch the original video carefully and answer the multiple-choice question. "
                  "Output exactly one line: FINAL_ANSWER: <letter>." % (str(r["question"]), opts))
        rec = {"record_key": k, "dataset": "mlvu_long", "platform": "bailian", "model": MODEL,
               "ground_truth": gt, "video_source_bytes": (info or {}).get("size"), "status": "started"}
        t0 = time.time()
        vpath = None
        try:
            sz = (info or {}).get("size", 0)
            if not info:
                rec.update({"status": "failed", "Error": "no stream info"})
            elif sz > SIZE_CAP:
                rec.update({"status": "failed", "fail_class": "video_too_large_for_disk",
                            "Error": "video %.0fMB exceeds local cap %dMB" % (sz / 1e6, SIZE_CAP // 1_000_000)})
            else:
                url = DL + urllib.parse.quote(info["repo_path"], safe="")
                vpath = tmp / (vid + ".mp4")
                import subprocess
                rc = subprocess.call(["curl", "-sL", "--retry", "6", "--retry-delay", "4",
                                      "--retry-all-errors", "-o", str(vpath), url])
                if rc != 0 or not vpath.exists() or vpath.stat().st_size < 10000:
                    raise RuntimeError("download failed (curl rc=%s)" % rc)
                content = [media_content("video", vpath), {"type": "text", "text": prompt}]
                try:
                    resp, meta_info = client._generate_once(content, max_tokens=64)
                    ans = first_letter(resp) or resp.strip()
                    rec.update({"status": "completed", "Answer": ans, "raw_response": str(resp)[:200],
                                "correct": bool(ans and str(ans).upper() == gt), "oss_media_upload": True})
                except Exception as exc:
                    msg = str(exc)
                    fc = "content_inspection" if "DataInspectionFailed" in msg else (
                        "quota" if ("Access denied" in msg or "quota" in msg) else "api_error")
                    rec.update({"status": "failed", "fail_class": fc, "Error": "%s: %s" % (type(exc).__name__, msg[:150])})
        except Exception as e:
            rec.update({"status": "failed", "fail_class": "download_or_other", "Error": "%s: %s" % (type(e).__name__, str(e)[:120])})
        finally:
            if vpath is not None:
                try:
                    vpath.unlink()
                except FileNotFoundError:
                    pass
        rec["elapsed_seconds"] = round(time.time() - t0, 1)
        with results.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("%s: %s pred=%s gt=%s correct=%s %s" % (k, rec["status"], rec.get("Answer"), gt, rec.get("correct"), str(rec.get("Error", ""))[:70]), flush=True)
    print("SHARD_%d_DONE" % args.shard, flush=True)


if __name__ == "__main__":
    main()
