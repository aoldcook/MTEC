#!/usr/bin/env python
"""Build fixed stratified record-key subsets for the cost-controlled ablation study.

Regimes:
  SF (short+medium+nextqa): Video-MME short n=100, Video-MME medium n=50, NExT-QA n=100
  Bailian (long): MLVU-long moderate (8-20min, <=300MB) n=50

Same record_keys are reused across every config in the matrix.
"""
import os
import json
import glob
import random
import zipfile
import pathlib

import pandas as pd

ROOT = "/root/autodl-tmp/MTEC"
OUTDIR = os.path.join(ROOT, "outputs/ablation_20260701/keys")
SEED = 20260701

VMME_META = os.path.join(ROOT, "data/datasets/video-mme/videomme/test-00000-of-00001.parquet")
VMME_ZIPS = os.path.join(ROOT, "data/modelscope/video-mme-zips")
NEXTQA_META = os.path.join(ROOT, "data/datasets/nextqa/nextqa_meta.parquet")
MLVU_META = os.path.join(ROOT, "data/datasets/mlvu/mlvu_meta.parquet")
MLVU_STREAM_MAP = os.path.join(ROOT, "data/datasets/mlvu/mlvu_stream_map.json")


def vmme_available_videos():
    have = set()
    for z in glob.glob(os.path.join(VMME_ZIPS, "*.zip")):
        with zipfile.ZipFile(z) as a:
            for i in a.infolist():
                if i.filename.endswith(".mp4"):
                    have.add(pathlib.Path(i.filename).stem)
    return have


def stratified_sample_by_video(df, n, rng, video_col="videoID"):
    """Round-robin sample across distinct videos to spread questions evenly,
    rather than clustering on a few videos."""
    by_video = {}
    for _, row in df.iterrows():
        by_video.setdefault(row[video_col], []).append(row)
    videos = list(by_video.keys())
    rng.shuffle(videos)
    for v in videos:
        rng.shuffle(by_video[v])
    picked = []
    i = 0
    while len(picked) < n:
        progressed = False
        for v in videos:
            if i < len(by_video[v]):
                picked.append(by_video[v][i])
                progressed = True
                if len(picked) >= n:
                    break
        if not progressed:
            break
        i += 1
    return picked


def build_vmme():
    rng = random.Random(SEED)
    meta = pd.read_parquet(VMME_META)
    have = vmme_available_videos()
    meta = meta[meta["videoID"].astype(str).isin(have)].copy()

    short_df = meta[meta.duration == "short"]
    medium_df = meta[meta.duration == "medium"]

    short_picked = stratified_sample_by_video(short_df, 100, rng)
    medium_picked = stratified_sample_by_video(medium_df, 50, rng)

    short_keys = [f"video:{r['videoID']}:{r['question_id']}" for r in short_picked]
    medium_keys = [f"video:{r['videoID']}:{r['question_id']}" for r in medium_picked]

    print(f"[vmme] short: {len(short_keys)} keys over {len({r['videoID'] for r in short_picked})} videos")
    print(f"[vmme] medium: {len(medium_keys)} keys over {len({r['videoID'] for r in medium_picked})} videos")
    return short_keys, medium_keys


def build_nextqa():
    rng = random.Random(SEED)
    meta = pd.read_parquet(NEXTQA_META)
    picked = stratified_sample_by_video(meta, 100, rng)
    keys = [f"video:{r['videoID']}:{r['question_id']}" for r in picked]
    print(f"[nextqa] {len(keys)} keys over {len({r['videoID'] for r in picked})} videos")
    return keys


def build_mlvu():
    rng = random.Random(SEED)
    stream_map = json.load(open(MLVU_STREAM_MAP))
    meta = pd.read_parquet(MLVU_META)

    moderate_video_ids = {
        vid for vid, info in stream_map.items()
        if 480 <= info["duration"] <= 1200 and info["size"] <= 300 * 1024 * 1024
    }
    cand = meta[meta["videoID"].astype(str).isin(moderate_video_ids)].copy()
    print(f"[mlvu] moderate candidates: {len(cand)} (videos: {cand.videoID.nunique()})")

    # stratify across sub_category (task_type) so the 50-video sample spans task families
    by_cat = {}
    for _, row in cand.iterrows():
        by_cat.setdefault(row["sub_category"], []).append(row)
    cats = sorted(by_cat.keys())
    for c in cats:
        rng.shuffle(by_cat[c])
    picked = []
    i = 0
    while len(picked) < 50:
        progressed = False
        for c in cats:
            if i < len(by_cat[c]):
                picked.append(by_cat[c][i])
                progressed = True
                if len(picked) >= 50:
                    break
        if not progressed:
            break
        i += 1

    keys = [f"video:{r['videoID']}:{r['question_id']}" for r in picked]
    meta_out = [
        {
            "record_key": f"video:{r['videoID']}:{r['question_id']}",
            "videoID": r["videoID"],
            "sub_category": r["sub_category"],
            "duration_sec": stream_map[r["videoID"]]["duration"],
            "size_bytes": stream_map[r["videoID"]]["size"],
        }
        for r in picked
    ]
    cat_counts = pd.Series([r["sub_category"] for r in picked]).value_counts().to_dict()
    print(f"[mlvu] picked {len(keys)} keys, category spread: {cat_counts}")
    return keys, meta_out


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    short_keys, medium_keys = build_vmme()
    nextqa_keys = build_nextqa()
    mlvu_keys, mlvu_meta_out = build_mlvu()

    def write_keys(name, keys):
        path = os.path.join(OUTDIR, f"{name}.txt")
        open(path, "w").write(" ".join(keys))
        print(f"  wrote {path} ({len(keys)} keys)")

    write_keys("sf_vmme_short", short_keys)
    write_keys("sf_vmme_medium", medium_keys)
    write_keys("sf_nextqa", nextqa_keys)
    write_keys("bailian_mlvu_long", mlvu_keys)
    write_keys("sf_all", short_keys + medium_keys + nextqa_keys)

    json.dump(
        {
            "seed": SEED,
            "sf_vmme_short": short_keys,
            "sf_vmme_medium": medium_keys,
            "sf_nextqa": nextqa_keys,
            "bailian_mlvu_long": mlvu_keys,
            "bailian_mlvu_long_meta": mlvu_meta_out,
        },
        open(os.path.join(OUTDIR, "manifest.json"), "w"),
        indent=2,
    )
    print("\nTotals: sf_short=%d sf_medium=%d nextqa=%d mlvu_long=%d" % (
        len(short_keys), len(medium_keys), len(nextqa_keys), len(mlvu_keys)))


if __name__ == "__main__":
    main()
