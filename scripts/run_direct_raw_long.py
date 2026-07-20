#!/usr/bin/env python3
"""Uncompressed baseline: send each ORIGINAL long video (via DashScope temp OSS) +
question directly to qwen3.7-plus, no MTEC anchors/compression. Sharded for parallelism.
"""
import json, os, re, sys, time, zipfile, argparse
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.run_modelscope_mtec_anchor_api_full import (  # noqa: E402
    OSS_MEDIA, SiliconFlowClient, downloaded_videos, extract_letter, media_content,
)

META_PATH = REPO_ROOT / "data/datasets/video-mme/videomme/test-00000-of-00001.parquet"
ZIPS_DIR = REPO_ROOT / "data/modelscope/video-mme-zips"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = os.environ.get("RAW_MODEL", "qwen3.7-plus")
OUT_BASE = REPO_ROOT / os.environ.get(
    "RAW_OUT_BASE", "outputs/direct_raw_long_%s_20260625" % re.sub(r"[.\-]", "", MODEL))


def read_jsonl(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open() if l.strip()]


def key_parts(key: str) -> Tuple[str, str]:
    m = re.fullmatch(r"video:(.*):([^:]+)", key)
    if not m:
        raise ValueError("bad key: %s" % key)
    return m.group(1), m.group(2)


def option_text(o):
    return "\n".join(str(x) for x in o) if isinstance(o, (list, tuple)) else str(o)


def make_question(row):
    return "%s\n%s\nRespond with only the option letter." % (str(row.get("question") or ""), option_text(row.get("options")))


def extract_member(zip_path, member, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(member.filename).name
    if out_path.exists() and out_path.stat().st_size == member.file_size:
        return out_path
    import shutil
    with zipfile.ZipFile(zip_path) as a, a.open(member) as s, out_path.open("wb") as t:
        shutil.copyfileobj(s, t)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys-file", required=True)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    args = ap.parse_args()

    api_key = os.environ["BAILIAN_API_KEY"]
    OSS_MEDIA["enabled"] = True
    OSS_MEDIA["api_key"] = api_key
    OSS_MEDIA["model"] = os.environ.get("OSS_UPLOAD_MODEL", MODEL)
    OSS_MEDIA["threshold_bytes"] = 1
    OSS_MEDIA["always_kinds"] = {"video"}

    out_dir = OUT_BASE / ("s%d" % args.shard)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / "results.jsonl"
    tmp_media = out_dir / "tmp_media"

    all_keys = [l.strip() for l in open(args.keys_file) if l.strip()]
    meta = pd.read_parquet(META_PATH)
    row_by_key = {}
    for _, row in meta.iterrows():
        d = row.to_dict()
        row_by_key["video:%s:%s" % (d.get("videoID"), d.get("question_id") or d.get("index") or "")] = d

    # group by video, then assign whole videos to shards (so a video is uploaded once)
    by_vid = OrderedDict()
    for k in all_keys:
        vid, _ = key_parts(k)
        by_vid.setdefault(vid, []).append(k)
    vids = [v for j, v in enumerate(by_vid) if j % args.shards == args.shard]

    # global resume: skip any record already completed in ANY shard dir
    done = set()
    for rf in OUT_BASE.glob("s*/results.jsonl"):
        for r in read_jsonl(rf):
            if r.get("status") == "completed":
                done.add(r.get("record_key"))
    video_lookup = downloaded_videos(ZIPS_DIR)
    # model pool with priority-order fallback (env RAW_MODEL_POOL, comma-separated)
    MODELS = [m.strip() for m in os.environ.get("RAW_MODEL_POOL", MODEL).split(",") if m.strip()]
    clients = {m: SiliconFlowClient(api_key=api_key, model=m, base_url=BASE_URL,
                                    timeout=1200, max_retries=2, retry_sleep=4.0,
                                    temperature=0.0, enable_thinking=None) for m in MODELS}
    dead = set()  # models that hit quota/capability failure -> skip for later records
    print("model pool:", MODELS, flush=True)

    def append(rec):
        with results.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("shard %d/%d videos=%d" % (args.shard, args.shards, len(vids)), flush=True)
    for video_id in vids:
        group = [k for k in by_vid[video_id] if k not in done]
        if not group:
            continue
        if video_id not in video_lookup:
            for k in group:
                append({"record_key": k, "status": "failed", "Error": "video not found: %s" % video_id})
            continue
        zip_path, member = video_lookup[video_id]
        vpath = None
        try:
            vpath = extract_member(zip_path, member, tmp_media)
            size = vpath.stat().st_size
            for k in group:
                t0 = time.perf_counter()
                row = row_by_key[k]
                gt = extract_letter(row.get("answer")) or str(row.get("answer") or "")
                prompt = make_question(row) + "\n\nWatch the original video carefully and answer the multiple-choice question. Output exactly one line: FINAL_ANSWER: <letter>."
                rec = {"record_key": k, "dataset": "lmms-lab/Video-MME", "modality": "video",
                       "mode": "direct_original_video_no_compression", "model": MODEL, "status": "started",
                       "videoID": video_id, "question_id": str(row.get("question_id") or ""),
                       "video_source_bytes": size, "ground_truth": gt}
                content = [media_content("video", vpath), {"type": "text", "text": prompt}]
                last_err = None
                tried = []
                last_class = None
                for m in MODELS:
                    if m in dead:
                        continue
                    tried.append(m)
                    try:
                        response, meta_info = clients[m]._generate_once(content, max_tokens=128)
                        ans = extract_letter(response) or response.strip()
                        rec.update({"status": "completed", "Answer": ans, "raw_response": response,
                                    "correct": bool(gt and str(ans).strip().upper() == str(gt).strip().upper()),
                                    "api_meta": meta_info, "oss_media_upload": True,
                                    "model": m, "models_tried": list(tried)})
                        break
                    except Exception as exc:
                        msg = str(exc)
                        last_err = "%s: %s" % (type(exc).__name__, exc)
                        if "DataInspectionFailed" in msg or "inappropriate" in msg:
                            # content filters are MODEL-SPECIFIC -> try the next model, keep this one alive
                            last_class = "content_inspection"
                            continue
                        # quota or capability (text model on video) -> retire model, try next
                        dead.add(m)
                        last_class = "quota_or_capability"
                        print("   model %s retired (%s)" % (m, msg[:70]), flush=True)
                if rec.get("status") == "started":
                    rec.update({"status": "failed", "Error": last_err or "no model produced an answer",
                                "models_tried": list(tried),
                                "fail_class": last_class or "all_models_exhausted"})
                rec["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
                append(rec)
                print("%s: %s pred=%s gt=%s correct=%s err=%s" % (k, rec["status"], rec.get("Answer"), gt, rec.get("correct"), str(rec.get("Error", ""))[:120]), flush=True)
        finally:
            if vpath is not None:
                try:
                    vpath.unlink()
                except FileNotFoundError:
                    pass
    print("SHARD_%d_DONE" % args.shard, flush=True)


if __name__ == "__main__":
    main()
