#!/usr/bin/env python3
"""Uncompressed raw-video baseline on SiliconFlow (Qwen/Qwen3-VL-32B-Instruct).
Sends the ORIGINAL video as a base64 data-URI (no compression). Three dataset
adapters: nextqa, videomme_medium, mlvu_long. Sharded; size-capped to avoid OOM.

SiliconFlow base64 video ceiling is ~50-80MB; larger videos reliably HTTP 500
(downstream black-box limit) and are recorded as failed.
"""
import os, re, sys, json, time, base64, zipfile, argparse, urllib.request, urllib.error
from pathlib import Path
import pandas as pd
from _env import require

ROOT = Path("/root/autodl-tmp/MTEC")
KEY = require("SF_API_KEY")
BASE = "https://api.siliconflow.cn/v1/chat/completions"
MODEL = "Qwen/Qwen3-VL-32B-Instruct"
ATTEMPT_CAP = int(os.environ.get("SF_SIZE_CAP_MB", "80")) * 1_000_000  # skip > this (avoid OOM; will 500 anyway)
LETTERS = ["A", "B", "C", "D", "E", "F", "G"]


def first_letter(txt):
    m = re.search(r"\b([A-G])\b", str(txt).upper())
    return m.group(1) if m else (str(txt).strip().upper()[:1] if str(txt).strip() else None)


def sf_call(video_bytes, prompt, tries=3):
    vurl = "data:video/mp4;base64," + base64.b64encode(video_bytes).decode()
    body = {"model": MODEL, "temperature": 0.0, "max_tokens": 24,
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": vurl}},
                {"type": "text", "text": prompt}]}]}
    data = json.dumps(body).encode()
    last = None
    for t in range(tries):
        req = urllib.request.Request(BASE, data=data, headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
        try:
            r = urllib.request.urlopen(req, timeout=300)
            return json.loads(r.read())["choices"][0]["message"]["content"], None
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("message", "")
            except Exception:
                msg = "http %d" % e.code
            last = "HTTP %d: %s" % (e.code, msg[:120])
            if e.code in (429, 500, 502, 503):
                time.sleep(6); continue  # 500 'Unknown error' is often transient on small videos
            return None, last
        except Exception as e:
            last = "%s: %s" % (type(e).__name__, str(e)[:100]); time.sleep(3)
    return None, last


# ---------- dataset adapters: yield (record_key, question_prompt, gt_letter, video_loader) ----------
def adapter_nextqa():
    meta = pd.read_parquet(ROOT / "data/datasets/nextqa/nextqa_meta.parquet")
    zf = zipfile.ZipFile(ROOT / "data/datasets/nextqa/videos/videos_chunked_00.zip")
    names = {Path(n).stem: n for n in zf.namelist() if n.endswith(".mp4")}
    for _, r in meta.iterrows():
        vid = str(r["videoID"]); qid = str(r["question_id"])
        if vid not in names:
            continue
        yield (f"video:{vid}:{qid}", _prompt(r["question"], r["options"]), str(r["answer"]).strip().upper(),
               (lambda n=names[vid]: zf.read(n)), None)


def adapter_videomme_medium():
    meta = pd.read_parquet(ROOT / "data/datasets/video-mme/videomme/test-00000-of-00001.parquet")
    meta = meta[meta["duration"] == "medium"]
    zips = list((ROOT / "data/modelscope/video-mme-zips").glob("*.zip"))
    member = {}
    for z in zips:
        with zipfile.ZipFile(z) as a:
            for i in a.infolist():
                if i.filename.endswith(".mp4"):
                    member[Path(i.filename).stem] = (z, i.filename, i.file_size)
    for _, r in meta.iterrows():
        vid = str(r["videoID"]); qid = str(r["question_id"])
        if vid not in member:
            continue
        z, fn, sz = member[vid]
        yield (f"video:{vid}:{qid}", _prompt(r["question"], list(r["options"])), str(r["answer"]).strip().upper(),
               (lambda zz=z, ff=fn: zipfile.ZipFile(zz).read(ff)), sz)


def adapter_mlvu_long():
    meta = pd.read_parquet(ROOT / "data/datasets/mlvu/mlvu_meta.parquet")
    smap = json.load(open(ROOT / "data/datasets/mlvu/mlvu_stream_map.json"))
    DL = "https://modelscope.cn/api/v1/datasets/AI-ModelScope/MLVU/repo?Revision=master&FilePath="
    import urllib.parse
    for _, r in meta.iterrows():
        vid = str(r["videoID"]); qid = str(r["question_id"])
        info = smap.get(vid)
        if not info:
            continue
        sz = info.get("size", 0)
        url = DL + urllib.parse.quote(info["repo_path"], safe="")
        def loader(u=url):
            return urllib.request.urlopen(u, timeout=600).read()
        yield (f"video:{vid}:{qid}", _prompt(r["question"], list(r["options"])), str(r["answer"]).strip().upper(), loader, sz)


def _prompt(q, options):
    opts = "\n".join(str(x) for x in options)
    return ("%s\n%s\n\nWatch the original video carefully and answer the multiple-choice question. "
            "Output exactly one line: FINAL_ANSWER: <letter>." % (str(q), opts))


ADAPTERS = {"nextqa": adapter_nextqa, "videomme_medium": adapter_videomme_medium, "mlvu_long": adapter_mlvu_long}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(ADAPTERS))
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out_dir = ROOT / ("outputs/raw_sf_%s/s%d" % (args.dataset, args.shard))
    out_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / "results.jsonl"
    done = set()
    if results.exists():
        for l in results.open():
            l = l.strip()
            if l and json.loads(l).get("status") == "completed":
                done.add(json.loads(l)["record_key"])

    items = list(ADAPTERS[args.dataset]())
    items = [it for idx, it in enumerate(items) if idx % args.shards == args.shard]
    if args.limit:
        items = items[: args.limit]
    print("dataset=%s shard=%d/%d items=%d model=%s" % (args.dataset, args.shard, args.shards, len(items), MODEL), flush=True)

    for rk, prompt, gt, loader, sz in items:
        if rk in done:
            continue
        rec = {"record_key": rk, "dataset": args.dataset, "model": MODEL, "ground_truth": gt,
               "video_source_bytes": sz, "status": "started"}
        t0 = time.time()
        try:
            if sz and sz > ATTEMPT_CAP:
                rec.update({"status": "failed", "fail_class": "video_too_large",
                            "Error": "video %.0fMB exceeds SiliconFlow base64 cap %dMB" % (sz / 1e6, ATTEMPT_CAP // 1_000_000)})
            else:
                vb = loader()
                if len(vb) > ATTEMPT_CAP:
                    rec.update({"status": "failed", "fail_class": "video_too_large",
                                "Error": "video %.0fMB exceeds cap" % (len(vb) / 1e6), "video_source_bytes": len(vb)})
                else:
                    rec["video_source_bytes"] = len(vb)
                    resp, err = sf_call(vb, prompt)
                    if resp is not None:
                        ans = first_letter(resp)
                        rec.update({"status": "completed", "Answer": ans, "raw_response": resp[:200],
                                    "correct": bool(ans and ans == gt)})
                    else:
                        fc = "video_too_large" if len(vb) > 50_000_000 else "api_error"
                        rec.update({"status": "failed", "fail_class": fc, "Error": err})
        except Exception as e:
            rec.update({"status": "failed", "fail_class": "exception", "Error": "%s: %s" % (type(e).__name__, str(e)[:120])})
        rec["elapsed_seconds"] = round(time.time() - t0, 1)
        with results.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("%s: %s pred=%s gt=%s correct=%s %s" % (rk, rec["status"], rec.get("Answer"), gt, rec.get("correct"), str(rec.get("Error", ""))[:60]), flush=True)
    print("SHARD_%d_DONE" % args.shard, flush=True)


if __name__ == "__main__":
    main()
