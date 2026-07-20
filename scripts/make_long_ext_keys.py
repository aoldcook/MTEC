#!/usr/bin/env python
"""Build long-video expansion key sets (disjoint from the original 30-key MLVU
pruned run) to raise statistical power for the long-video ablations.

  keys_long_ext/mlvu_ext.txt  : +N_MLVU new moderate MLVU keys (8-20min, <=300MB)
  keys_long_ext/vmme_long.txt : +N_VMME new Video-MME long keys (locally available)
"""
import os
import json
import random
import glob
import zipfile
import pathlib

import pandas as pd

ROOT = "/root/autodl-tmp/MTEC"
OUT = os.path.join(ROOT, "outputs/ablation_20260701/keys_long_ext")
SEED = 20260701
N_MLVU = 30
N_VMME = 40

VMME_META = os.path.join(ROOT, "data/datasets/video-mme/videomme/test-00000-of-00001.parquet")
VMME_ZIPS = os.path.join(ROOT, "data/modelscope/video-mme-zips")
MLVU_META = os.path.join(ROOT, "data/datasets/mlvu/mlvu_meta.parquet")
MLVU_STREAM = os.path.join(ROOT, "data/datasets/mlvu/mlvu_stream_map.json")
USED_MLVU = os.path.join(ROOT, "outputs/ablation_20260701/keys_pruned/bailian_mlvu_long.txt")


def rr_pick(by_group, n, rng):
    groups = sorted(by_group.keys())
    for g in groups:
        rng.shuffle(by_group[g])
    picked, i = [], 0
    while len(picked) < n:
        prog = False
        for g in groups:
            if i < len(by_group[g]):
                picked.append(by_group[g][i]); prog = True
                if len(picked) >= n:
                    break
        if not prog:
            break
        i += 1
    return picked


def build_mlvu():
    rng = random.Random(SEED + 1)
    smap = json.load(open(MLVU_STREAM))
    mm = pd.read_parquet(MLVU_META)
    used = set(open(USED_MLVU).read().split())
    moderate = {v for v, inf in smap.items()
                if 480 <= inf["duration"] <= 1200 and inf["size"] <= 300 * 1024 * 1024}
    cand = mm[mm.videoID.astype(str).isin(moderate)].copy()
    cand["rk"] = cand.apply(lambda r: "video:%s:%s" % (r.videoID, r.question_id), axis=1)
    cand = cand[~cand.rk.isin(used)]
    by_cat = {}
    for _, r in cand.iterrows():
        by_cat.setdefault(r.sub_category, []).append(r.rk)
    picked = rr_pick(by_cat, N_MLVU, rng)
    return picked


def build_vmme_long():
    rng = random.Random(SEED + 2)
    meta = pd.read_parquet(VMME_META)
    have = set()
    for z in glob.glob(os.path.join(VMME_ZIPS, "*.zip")):
        with zipfile.ZipFile(z) as a:
            for i in a.infolist():
                if i.filename.endswith(".mp4"):
                    have.add(pathlib.Path(i.filename).stem)
    ml = meta[(meta.duration == "long") & (meta.videoID.astype(str).isin(have))].copy()
    ml["rk"] = ml.apply(lambda r: "video:%s:%s" % (r.videoID, r.question_id), axis=1)
    by_video = {}
    for _, r in ml.iterrows():
        by_video.setdefault(r.videoID, []).append(r.rk)
    picked = rr_pick(by_video, N_VMME, rng)
    return picked


os.makedirs(OUT, exist_ok=True)
mlvu = build_mlvu()
vmme = build_vmme_long()
open(os.path.join(OUT, "mlvu_ext.txt"), "w").write(" ".join(mlvu))
open(os.path.join(OUT, "vmme_long.txt"), "w").write(" ".join(vmme))
print("mlvu_ext: %d keys (disjoint from used 30)" % len(mlvu))
print("vmme_long: %d keys over %d videos" % (len(vmme), len({k.split(':')[1] for k in vmme})))
